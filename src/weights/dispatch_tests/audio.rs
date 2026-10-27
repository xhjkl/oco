//! Audio-encoder kernels against host references: layer normalization, GELU, positional rows,
//! convolution unfolding, and bidirectional attention.
use super::GUARD;
use crate::weights::compute::AttentionParams;
use crate::{
    gpu::{
        Gpu, dispatch,
        tests::{buffer, read},
    },
    tensor::WeightFormat,
    weights::{
        compute::{Compute, Im2col, LayerNorm},
        pipelines::Builder,
    },
};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use half::bf16;

const MARKER: u16 = 0x7fc1;

fn value(i: usize, salt: usize) -> f32 {
    ((i * 37 + salt) % 251) as f32 / 64.0 - 2.0
}

fn guarded(values: &[u16]) -> Vec<u16> {
    let mut storage = vec![MARKER; GUARD + values.len() + GUARD];
    storage[GUARD..GUARD + values.len()].copy_from_slice(values);
    storage
}

fn close(got: f32, want: f32, tolerance: f32, what: &str) {
    assert!(
        (got - want).abs() <= tolerance * (1.0 + want.abs()),
        "{what}: got {got}, want {want}"
    );
}

#[test]
fn gpu_layer_norm_variants_match_reference() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let plain = builder.get("layer_norm", WeightFormat::F32, 0)?;
    let fused = builder.get("add_layer_norm", WeightFormat::F32, 0)?;
    let (n, rows) = (1024usize, 5usize);
    let x: Vec<f32> = (0..n * rows).map(|i| value(i, 3) * 3.0).collect();
    let addend: Vec<f32> = (0..n * rows).map(|i| value(i, 91) * 0.5).collect();
    let w: Vec<f32> = (0..n).map(|i| 0.5 + value(i, 7) * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| value(i, 11) * 0.05).collect();
    let bits = |v: &[f32]| -> Vec<u16> { v.iter().map(|&f| bf16::from_f32(f).to_bits()).collect() };
    let p = LayerNorm {
        n: n as u32,
        epsilon: 1e-5,
    };
    let reference = |row: &[f32]| -> Vec<f32> {
        let mean = row.iter().sum::<f32>() / n as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + p.epsilon).sqrt();
        row.iter()
            .enumerate()
            .map(|(i, v)| (v - mean) * inv * w[i] + b[i])
            .collect()
    };
    let wb = buffer(&gpu, &w);
    let bb = buffer(&gpu, &b);
    // Plain normalization.
    let xb = buffer(&gpu, &guarded(&bits(&x)));
    let mut out = vec![MARKER; GUARD + n * rows + GUARD];
    let yb = buffer(&gpu, &out);
    let submission = compute.submit(|e| {
        dispatch(
            e,
            &plain,
            &[(&xb, GUARD * 2), (&wb, 0), (&bb, 0), (&yb, GUARD * 2)],
            bytemuck::bytes_of(&p),
            [rows, 1, 1],
            256,
        );
    });
    futures_lite::future::block_on(submission)?;
    read(&yb, &mut out);
    assert!(
        out[..GUARD]
            .iter()
            .chain(&out[GUARD + n * rows..])
            .all(|&v| v == MARKER)
    );
    for r in 0..rows {
        let input: Vec<f32> = x[r * n..(r + 1) * n]
            .iter()
            .map(|&v| bf16::from_f32(v).to_f32())
            .collect();
        let want = reference(&input);
        for i in 0..n {
            let got = bf16::from_bits(out[GUARD + r * n + i]).to_f32();
            close(got, want[i], 2e-2, &format!("layer_norm row {r} col {i}"));
        }
    }
    // Fused residual add: the stream must hold the sum, the output its normalization.
    let xb = buffer(&gpu, &guarded(&bits(&x)));
    let ab = buffer(&gpu, &guarded(&bits(&addend)));
    let yb = buffer(&gpu, &out);
    let submission = compute.submit(|e| {
        dispatch(
            e,
            &fused,
            &[
                (&xb, GUARD * 2),
                (&ab, GUARD * 2),
                (&wb, 0),
                (&bb, 0),
                (&yb, GUARD * 2),
            ],
            bytemuck::bytes_of(&p),
            [rows, 1, 1],
            256,
        );
    });
    futures_lite::future::block_on(submission)?;
    let mut stream = vec![0u16; GUARD + n * rows + GUARD];
    read(&xb, &mut stream);
    read(&yb, &mut out);
    for r in 0..rows {
        let summed: Vec<f32> = (0..n)
            .map(|i| {
                let a = bf16::from_f32(x[r * n + i]).to_f32();
                let b = bf16::from_f32(addend[r * n + i]).to_f32();
                bf16::from_f32(a + b).to_f32()
            })
            .collect();
        let want = reference(&summed);
        for i in 0..n {
            let got_sum = bf16::from_bits(stream[GUARD + r * n + i]).to_f32();
            close(
                got_sum,
                summed[i],
                1e-6,
                &format!("residual row {r} col {i}"),
            );
            let got = bf16::from_bits(out[GUARD + r * n + i]).to_f32();
            close(
                got,
                want[i],
                2e-2,
                &format!("add_layer_norm row {r} col {i}"),
            );
        }
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Gelu {
    n: u32,
    erf_form: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AddRows {
    n: u32,
    rows: u32,
    offset: u32,
    period: u32,
}

#[test]
fn gpu_gelu_and_positional_rows_match_reference() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let gelu = builder.get("gelu", WeightFormat::F32, 0)?;
    let add_rows = builder.get("add_rows", WeightFormat::F32, 0)?;
    let n = 1000usize;
    let x: Vec<u16> = (0..n)
        .map(|i| bf16::from_f32(value(i, 5) * 2.5).to_bits())
        .collect();
    for erf_form in [0u32, 1] {
        let xb = buffer(&gpu, &guarded(&x));
        let p = Gelu {
            n: n as u32,
            erf_form,
        };
        let submission = compute.submit(|e| {
            dispatch(
                e,
                &gelu,
                &[(&xb, GUARD * 2)],
                bytemuck::bytes_of(&p),
                [n.div_ceil(256), 1, 1],
                256,
            );
        });
        futures_lite::future::block_on(submission)?;
        let mut out = vec![0u16; GUARD + n + GUARD];
        read(&xb, &mut out);
        assert!(
            out[..GUARD]
                .iter()
                .chain(&out[GUARD + n..])
                .all(|&v| v == MARKER)
        );
        for i in 0..n {
            let v = bf16::from_bits(x[i]).to_f64();
            let want = if erf_form == 1 {
                0.5 * v * (1.0 + erf(v / 2f64.sqrt()))
            } else {
                0.5 * v * (1.0 + (0.7978845608028654 * (v + 0.044715 * v * v * v)).tanh())
            };
            let got = bf16::from_bits(out[GUARD + i]).to_f64();
            assert!(
                (got - want).abs() <= 1e-2 * (1.0 + want.abs()),
                "gelu(erf={erf_form}) at {v}: got {got}, want {want}"
            );
        }
    }
    let (width, rows, period, offset) = (96usize, 7usize, 3usize, 2usize);
    let x: Vec<u16> = (0..width * rows)
        .map(|i| bf16::from_f32(value(i, 17)).to_bits())
        .collect();
    let table: Vec<f32> = (0..width * period).map(|i| value(i, 23) * 0.25).collect();
    let xb = buffer(&gpu, &guarded(&x));
    let tb = buffer(&gpu, &table);
    let p = AddRows {
        n: width as u32,
        rows: rows as u32,
        offset: offset as u32,
        period: period as u32,
    };
    let submission = compute.submit(|e| {
        dispatch(
            e,
            &add_rows,
            &[(&xb, GUARD * 2), (&tb, 0)],
            bytemuck::bytes_of(&p),
            [(width * rows).div_ceil(256), 1, 1],
            256,
        );
    });
    futures_lite::future::block_on(submission)?;
    let mut out = vec![0u16; GUARD + width * rows + GUARD];
    read(&xb, &mut out);
    for r in 0..rows {
        for i in 0..width {
            let want = bf16::from_bits(x[r * width + i]).to_f32()
                + table[((offset + r) % period) * width + i];
            let got = bf16::from_bits(out[GUARD + r * width + i]).to_f32();
            close(got, want, 1e-2, &format!("add_rows row {r} col {i}"));
        }
    }
    Ok(())
}

fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun 7.1.26, adequate for comparing BF16 outputs.
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly =
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592;
    (1.0 - poly * t * (-x * x).exp()).copysign(x)
}

#[test]
fn gpu_im2col_matches_reference_with_padding_and_batch() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let pipeline = builder.get("im2col", WeightFormat::F32, 0)?;
    // Two batched 7x5 images of 6 channels, 3x3 taps, stride 2, unit padding, K padded to 64.
    let (b, c, h, w) = (2usize, 6usize, 7usize, 5usize);
    let mut p = Im2col::new(c, h, w, (3, 3), 2, (1, 1));
    p.rows *= b as u32;
    let (oh, ow) = p.output();
    let k = p.k as usize;
    assert_eq!(k, 64);
    let x: Vec<u16> = (0..b * h * w * c)
        .map(|i| bf16::from_f32(value(i, 29)).to_bits())
        .collect();
    let xb = buffer(&gpu, &guarded(&x));
    let rows = b * oh * ow;
    let mut out = vec![MARKER; GUARD + rows * k + GUARD];
    let yb = buffer(&gpu, &out);
    let submission = compute.submit(|e| {
        dispatch(
            e,
            &pipeline,
            &[(&xb, GUARD * 2), (&yb, GUARD * 2)],
            bytemuck::bytes_of(&p),
            [(rows * k).div_ceil(256), 1, 1],
            256,
        );
    });
    futures_lite::future::block_on(submission)?;
    read(&yb, &mut out);
    assert!(
        out[..GUARD]
            .iter()
            .chain(&out[GUARD + rows * k..])
            .all(|&v| v == MARKER)
    );
    for batch in 0..b {
        for oy in 0..oh {
            for ox in 0..ow {
                let row = (batch * oh + oy) * ow + ox;
                for j in 0..k {
                    let want = if j < 9 * c {
                        let (tap, ch) = (j / c, j % c);
                        let iy = (oy * 2 + tap / 3) as isize - 1;
                        let ix = (ox * 2 + tap % 3) as isize - 1;
                        if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                            x[((batch * h + iy as usize) * w + ix as usize) * c + ch]
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    assert_eq!(out[GUARD + row * k + j], want, "row {row} col {j}");
                }
            }
        }
    }
    // One-dimensional taps along the first axis only, no padding on the second.
    let (t, c) = (9usize, 4usize);
    let p = Im2col::new(c, t, 1, (3, 1), 2, (1, 0));
    let k = p.k as usize;
    let x: Vec<u16> = (0..t * c)
        .map(|i| bf16::from_f32(value(i, 31)).to_bits())
        .collect();
    let xb = buffer(&gpu, &x);
    let rows = p.rows as usize;
    let mut out = vec![MARKER; rows * k];
    let yb = buffer(&gpu, &out);
    let submission = compute.submit(|e| {
        dispatch(
            e,
            &pipeline,
            &[(&xb, 0), (&yb, 0)],
            bytemuck::bytes_of(&p),
            [(rows * k).div_ceil(256), 1, 1],
            256,
        );
    });
    futures_lite::future::block_on(submission)?;
    read(&yb, &mut out);
    for r in 0..rows {
        for j in 0..k {
            let (tap, ch) = (j / c, j % c);
            let i = (r * 2 + tap) as isize - 1;
            let want = if j < 3 * c && i >= 0 && (i as usize) < t {
                x[i as usize * c + ch]
            } else {
                0
            };
            assert_eq!(out[r * k + j], want, "1-D row {r} column {j}");
        }
    }
    Ok(())
}

/// With `keys` set, every query row sees the same full key range regardless of its position.
#[test]
fn gpu_bidirectional_attention_matches_reference() -> Result<()> {
    use half::f16;
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let (heads, dim, rows) = (4usize, 64usize, 29usize);
    let q: Vec<u16> = (0..rows * heads * dim)
        .map(|i| bf16::from_f32(value(i, 41) * 0.3).to_bits())
        .collect();
    let k: Vec<u16> = (0..rows * heads * dim)
        .map(|i| f16::from_f32(value(i, 43) * 0.3).to_bits())
        .collect();
    let v: Vec<u16> = (0..rows * heads * dim)
        .map(|i| f16::from_f32(value(i, 47) * 0.5).to_bits())
        .collect();
    let p = AttentionParams {
        heads: heads as u32,
        kv_heads: heads as u32,
        dim: dim as u32,
        pos: 0,
        window: 0,
        sinks: 0,
        scale: 1.0 / (dim as f32).sqrt(),
        m: rows as u32,
        keys: rows as u32,
    };
    for (symbol, groups, threads) in [
        ("attention_block_64", [heads, rows.div_ceil(32), 1], 128),
        ("attention_tiled", [heads, rows.div_ceil(8), 1], 128),
        ("attention", [heads, rows, 1], 1024),
        ("attention_vector_64", [heads, rows, 1], 256),
    ] {
        let pipeline = builder.get(symbol, WeightFormat::F32, 0)?;
        let qb = buffer(&gpu, &guarded(&q));
        let kb = buffer(&gpu, &guarded(&k));
        let vb = buffer(&gpu, &guarded(&v));
        let sinks = buffer(&gpu, &[0f32; 4]);
        let mut out = vec![MARKER; GUARD + rows * heads * dim + GUARD];
        let yb = buffer(&gpu, &out);
        let submission = compute.submit(|e| {
            dispatch(
                e,
                &pipeline,
                &[
                    (&qb, GUARD * 2),
                    (&kb, GUARD * 2),
                    (&vb, GUARD * 2),
                    (&sinks, 0),
                    (&yb, GUARD * 2),
                ],
                bytemuck::bytes_of(&p),
                groups,
                threads,
            );
        });
        futures_lite::future::block_on(submission)?;
        read(&yb, &mut out);
        for (t, head) in [(0, 0), (3, 1), (8, 2), (28, 3), (16, 0)] {
            let query: Vec<f64> = (0..dim)
                .map(|d| {
                    f64::from(bf16::from_bits(q[(t * heads + head) * dim + d]).to_f32() * p.scale)
                })
                .collect();
            let scores: Vec<f64> = (0..rows)
                .map(|key| {
                    (0..dim)
                        .map(|d| {
                            query[d] * f16::from_bits(k[(key * heads + head) * dim + d]).to_f64()
                        })
                        .sum()
                })
                .collect();
            let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - peak).exp()).collect();
            let mass: f64 = weights.iter().sum();
            for d in [0, 17, 63] {
                let want: f64 = (0..rows)
                    .map(|key| {
                        weights[key] * f16::from_bits(v[(key * heads + head) * dim + d]).to_f64()
                    })
                    .sum::<f64>()
                    / mass;
                let got = bf16::from_bits(out[GUARD + (t * heads + head) * dim + d]).to_f64();
                assert!(
                    (got - want).abs() <= 2e-2 * (1.0 + want.abs()),
                    "{symbol} row {t} head {head} dim {d}: got {got}, want {want}"
                );
            }
        }
    }
    Ok(())
}
