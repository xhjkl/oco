use super::AttentionParams;
use crate::gpu::{
    Gpu, Pipeline, dispatch,
    tests::{buffer, read},
};
use crate::tensor::WeightFormat;
use crate::weights::{compute::Compute, pipelines::Builder};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use half::{bf16, f16};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    attention: AttentionParams,
    splits: u32,
}

struct Fixture {
    p: Params,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    sinks: Vec<f32>,
}

/// Compute split attention independently in FP64 after the declared query/probability half conversions.
fn reference(f: &Fixture) -> Vec<u16> {
    let p = f.p.attention;
    let (heads, kv_heads, dim, splits) = (
        p.heads as usize,
        p.kv_heads as usize,
        p.dim as usize,
        f.p.splits as usize,
    );
    let mut output = Vec::new();
    for token in 0..p.m as usize {
        let end = p.pos as usize + token + 1;
        let begin = if p.window == 0 {
            0
        } else {
            end.saturating_sub(p.window as usize)
        };
        let (span, extra) = ((end - begin) / splits, (end - begin) % splits);
        for head in 0..heads {
            let kv = head / (heads / kv_heads);
            let query: Vec<_> = f.q[(token * heads + head) * dim..(token * heads + head + 1) * dim]
                .iter()
                .map(|&bits| f16::from_f32(bf16::from_bits(bits).to_f32() * p.scale).to_f64())
                .collect();
            let scores: Vec<_> = (begin..end)
                .map(|key| {
                    query
                        .iter()
                        .enumerate()
                        .map(|(d, &q)| {
                            q * f16::from_bits(f.k[(key * kv_heads + kv) * dim + d]).to_f64()
                        })
                        .sum::<f64>()
                })
                .collect();
            let mut partials = Vec::new();
            for part in 0..splits {
                let start = begin + part * span + part.min(extra);
                let stop = start + span + usize::from(part < extra);
                let sink = p.sinks != 0 && part == 0;
                let mut peak = if sink {
                    bf16::from_f32(f.sinks[head]).to_f64()
                } else {
                    -f64::from(f32::MAX)
                };
                let mut mass = if sink { 1.0 } else { 0.0 };
                let mut values = vec![0.0f64; dim];
                for block in (start..stop).step_by(32) {
                    let block_end = (block + 32).min(stop);
                    let next = scores[block - begin..block_end - begin]
                        .iter()
                        .copied()
                        .fold(peak, f64::max);
                    let factor = (peak - next).exp();
                    mass *= factor;
                    for value in &mut values {
                        *value *= factor;
                    }
                    for key in block..block_end {
                        let probability = (scores[key - begin] - next).exp();
                        mass += probability;
                        let probability = f16::from_f64(probability).to_f64();
                        for (d, value) in values.iter_mut().enumerate() {
                            *value += probability
                                * f16::from_bits(f.v[(key * kv_heads + kv) * dim + d]).to_f64();
                        }
                    }
                    peak = next;
                }
                partials.push((peak, mass, values));
            }
            let peak = partials
                .iter()
                .map(|p| p.0)
                .fold(f64::NEG_INFINITY, f64::max);
            let factors: Vec<_> = partials.iter().map(|p| (p.0 - peak).exp()).collect();
            let mass: f64 = partials
                .iter()
                .zip(&factors)
                .map(|(p, &factor)| p.1 * factor)
                .sum();
            for d in 0..dim {
                let value: f64 = partials
                    .iter()
                    .zip(&factors)
                    .map(|(p, &factor)| p.2[d] * factor)
                    .sum();
                output.push(bf16::from_f64(value / mass).to_bits());
            }
        }
    }
    output
}

fn fixture(
    heads: usize,
    kv: usize,
    dim: usize,
    position: usize,
    rows: usize,
    window: usize,
    sinks: bool,
) -> Fixture {
    let q = (0..rows * heads * dim)
        .map(|i| bf16::from_f32(((i * 13 + i / dim * 7) % 41) as f32 / 64.0 - 0.3125).to_bits())
        .collect();
    let count = (position + rows) * kv * dim;
    let k = (0..count)
        .map(|i| f16::from_f32(((i * 11 + i / dim * 3) % 47) as f32 / 64.0 - 0.359375).to_bits())
        .collect();
    let v = (0..count)
        .map(|i| f16::from_f32(((i * 17 + i / dim * 5) % 59) as f32 / 32.0 - 0.875).to_bits())
        .collect();
    let sink_values = (0..heads)
        .map(|h| h as f32 * 0.133 - 0.7)
        .collect::<Vec<_>>();
    Fixture {
        p: Params {
            attention: AttentionParams {
                heads: heads as u32,
                kv_heads: kv as u32,
                dim: dim as u32,
                pos: position as u32,
                window: window as u32,
                sinks: u32::from(sinks),
                scale: 1.0 / (dim as f32).sqrt(),
                m: rows as u32,
                keys: 0,
            },
            splits: 32,
        },
        q,
        k,
        v,
        sinks: sink_values,
    }
}

fn check(
    gpu: &Gpu,
    compute: &Compute,
    partial: &Pipeline,
    merge: &Pipeline,
    f: &Fixture,
    invalid: bool,
) -> Result<()> {
    const GUARD: usize = 32;
    const MARKER: u16 = 0x7fc1;
    let guarded = |values: &[u16]| {
        let mut words = vec![MARKER; GUARD + values.len() + GUARD];
        words[GUARD..GUARD + values.len()].copy_from_slice(values);
        words
    };
    let mut query = guarded(&f.q);
    if invalid {
        query[GUARD] = bf16::NAN.to_bits();
    }
    let key = guarded(&f.k);
    let value = guarded(&f.v);
    let q = buffer(gpu, &query);
    let k = buffer(gpu, &key);
    let v = buffer(gpu, &value);
    let sinks = buffer(gpu, &f.sinks);
    let p = f.p.attention;
    let count = (p.m * p.heads * p.dim) as usize;
    let partial_count = (p.m * p.heads * f.p.splits * (p.dim + 2)) as usize;
    let mut partials = vec![8192.0f32; GUARD + partial_count + GUARD];
    partials[GUARD..GUARD + partial_count].fill(f32::NAN);
    let scratch = buffer(gpu, &partials);
    let mut output = vec![MARKER; GUARD + count + GUARD];
    let y = buffer(gpu, &output);
    let head_tiles = (p.heads as usize / p.kv_heads as usize).div_ceil(8);
    let submission = compute.submit(|encoder| {
        dispatch(
            encoder,
            partial,
            &[
                (&q, GUARD * 2),
                (&k, GUARD * 2),
                (&v, GUARD * 2),
                (&sinks, 0),
                (&scratch, GUARD * 4),
            ],
            bytemuck::bytes_of(&f.p),
            [
                p.kv_heads as usize * head_tiles,
                p.m as usize,
                f.p.splits as usize,
            ],
            128,
        );
        dispatch(
            encoder,
            merge,
            &[(&scratch, GUARD * 4), (&y, GUARD * 2)],
            bytemuck::bytes_of(&f.p),
            [p.heads as usize, p.m as usize, 1],
            128,
        );
    });
    futures_lite::future::block_on(submission)?;
    read(&y, &mut output);
    read(&scratch, &mut partials);
    assert!(
        output[..GUARD]
            .iter()
            .chain(&output[GUARD + count..])
            .all(|&x| x == MARKER)
    );
    assert!(
        partials[..GUARD]
            .iter()
            .chain(&partials[GUARD + partial_count..])
            .all(|&x| x == 8192.0)
    );
    for (buffer, expected) in [(&q, &query), (&k, &key), (&v, &value)] {
        let mut actual = vec![0u16; expected.len()];
        read(buffer, &mut actual);
        assert_eq!(
            &actual, expected,
            "attention must not mutate query/cache inputs"
        );
    }
    if invalid {
        assert!(
            output[GUARD..GUARD + count]
                .iter()
                .any(|&bits| !bf16::from_bits(bits).is_finite()),
            "nonfinite query must not become a finite attention result"
        );
        return Ok(());
    }
    assert!(
        partials[GUARD..GUARD + partial_count]
            .iter()
            .all(|x| x.is_finite()),
        "every partial word must overwrite its poison, including empty splits"
    );
    let expected = reference(f);
    for (index, (&actual, &expected)) in output[GUARD..GUARD + count]
        .iter()
        .zip(&expected)
        .enumerate()
    {
        let actual = bf16::from_bits(actual).to_f32();
        let expected = bf16::from_bits(expected).to_f32();
        let next = bf16::from_bits(bf16::from_f32(expected.abs()).to_bits() + 1).to_f32();
        let bound = (next - expected.abs()).max(1e-5);
        assert!(
            actual.is_finite() && (actual - expected).abs() <= bound,
            "split attention D={} H={} KV={} M={} pos={} window={} sinks={} index={index}: {actual} != {expected}, bound={bound}",
            p.dim,
            p.heads,
            p.kv_heads,
            p.m,
            p.pos,
            p.window,
            p.sinks
        );
    }
    Ok(())
}

/// Cover native GQA, uneven fixed-32 partitions, sinks and exact-sized long cache boundaries.
#[test]
fn gpu_split_attention_matches_half_operand_oracle_and_guards() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let partial = builder.get("attention_gqa_split", WeightFormat::F32, 0)?;
    let merge = builder.get("attention_gqa_merge", WeightFormat::F32, 0)?;
    let cases = [
        (24, 4, 256, 4098, 1, 0, false),
        (64, 8, 64, 4096, 1, 0, true),
    ];
    for (heads, kv, dim, pos, rows, window, sinks) in cases {
        let f = fixture(heads, kv, dim, pos, rows, window, sinks);
        check(&gpu, &compute, &partial, &merge, &f, false)?;
    }
    let f = fixture(8, 1, 64, 4096, 1, 0, false);
    check(&gpu, &compute, &partial, &merge, &f, true)?;
    Ok(())
}
