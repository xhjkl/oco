use super::AttentionParams;
use crate::gpu::{
    Gpu, dispatch,
    tests::{buffer, read},
};
use crate::tensor::WeightFormat;
use crate::weights::{compute::Compute, pipelines::Builder};
use anyhow::Result;
use half::{bf16, f16};

/// Check live scalar and tiled attention against mathematical softmax at causal/window tails.
#[test]
fn attention_matches_independent_reference_and_cache_tails() -> Result<()> {
    const GUARD: usize = 32;
    const MARKER: u16 = 0x7fc1;
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    for (dim, heads, kv_heads, position, rows, window, has_sinks, symbol, tile, threads) in [
        (
            256usize,
            4usize,
            2usize,
            31usize,
            512usize,
            0usize,
            false,
            "attention_wide16",
            16usize,
            128,
        ),
        (64, 8, 2, 129, 512, 128, true, "attention_wide16", 16, 128),
        (64, 8, 2, 129, 512, 128, true, "attention_block_64", 32, 128),
        (128, 8, 2, 31, 77, 0, false, "attention_block_128", 32, 128),
        (64, 8, 2, 127, 9, 128, true, "attention_block_64", 32, 128),
        (256, 4, 2, 31, 9, 0, false, "attention_tiled", 8, 128),
        (64, 8, 2, 127, 9, 128, true, "attention_tiled", 8, 128),
        (64, 8, 2, 129, 1, 0, true, "attention", 1, 1024),
        (64, 8, 2, 129, 1, 0, true, "attention_vector_64", 1, 256),
        (64, 8, 2, 1000, 3, 128, false, "attention_vector_64", 1, 256),
        (128, 8, 4, 517, 1, 0, false, "attention_vector_128", 1, 256),
        (128, 32, 2, 517, 1, 0, false, "attention_vector_128", 1, 256),
        (128, 32, 2, 0, 1, 0, false, "attention_vector_128", 1, 256),
        (128, 32, 2, 16, 7, 0, false, "attention_vector_128", 1, 256),
        (128, 8, 4, 5, 2, 0, true, "attention_vector_128", 1, 256),
        (256, 4, 2, 700, 1, 0, false, "attention_vector_256", 1, 256),
    ] {
        let pipeline = builder.get(symbol, WeightFormat::F32, 0)?;
        let count = rows * heads * dim;
        let cache_rows = position + rows;
        let mut queries = vec![MARKER; GUARD + count + GUARD];
        let mut keys = vec![MARKER; GUARD + cache_rows * kv_heads * dim + GUARD];
        let mut values = keys.clone();
        for (i, value) in queries[GUARD..GUARD + count].iter_mut().enumerate() {
            *value = bf16::from_f32(((i * 13 + i / dim * 7) % 41) as f32 / 16.0 - 1.25).to_bits();
        }
        for i in 0..cache_rows * kv_heads * dim {
            keys[GUARD + i] =
                f16::from_f32(((i * 11 + i / dim * 3) % 47) as f32 / 16.0 - 23.0 / 16.0).to_bits();
            values[GUARD + i] =
                f16::from_f32(((i * 17 + i / dim * 5) % 59) as f32 / 32.0 - 29.0 / 32.0).to_bits();
        }
        let biases: Vec<_> = (0..heads).map(|h| -0.7 + h as f32 * 0.133).collect();
        let q = buffer(&gpu, &queries);
        let k = buffer(&gpu, &keys);
        let v = buffer(&gpu, &values);
        let sinks = buffer(&gpu, &biases);
        let mut output = vec![MARKER; GUARD + count + GUARD];
        let y = buffer(&gpu, &output);
        let p = AttentionParams {
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            dim: dim as u32,
            pos: position as u32,
            window: window as u32,
            sinks: u32::from(has_sinks),
            scale: 1.0 / (dim as f32).sqrt(),
            m: rows as u32,
            keys: 0,
        };
        let submission = compute.submit(|encoder| {
            dispatch(
                encoder,
                &pipeline,
                &[
                    (&q, GUARD * 2),
                    (&k, GUARD * 2),
                    (&v, GUARD * 2),
                    (&sinks, 0),
                    (&y, GUARD * 2),
                ],
                bytemuck::bytes_of(&p),
                [heads, rows.div_ceil(tile), 1],
                threads,
            );
        });
        futures_lite::future::block_on(submission)?;
        read(&y, &mut output);
        assert!(output[..GUARD].iter().all(|&v| v == MARKER));
        assert!(output[GUARD + count..].iter().all(|&v| v == MARKER));
        assert!(
            output[GUARD..GUARD + count]
                .iter()
                .all(|&v| bf16::from_bits(v).is_finite()),
            "non-finite or unwritten output at position {position}"
        );

        // Sampling both sides of query-tile boundaries and the final causal/cache fragment.
        for token in [0, 7, 8, 15, 16, rows - 1]
            .into_iter()
            .filter(|&t| t < rows)
        {
            let end = position + token + 1;
            let begin = if window == 0 {
                0
            } else {
                end.saturating_sub(window)
            };
            let ratio = heads / kv_heads;
            for head in [0, ratio - 1, ratio, heads - 1] {
                let kv = head / (heads / kv_heads);
                let query: Vec<_> = (0..dim)
                    .map(|d| {
                        let bits = queries[GUARD + (token * heads + head) * dim + d];
                        let q = bf16::from_bits(bits).to_f32() * p.scale;
                        if tile == 1 {
                            f64::from(q)
                        } else {
                            f16::from_f32(q).to_f64()
                        }
                    })
                    .collect();
                let scores: Vec<_> = (begin..end)
                    .map(|key| {
                        query
                            .iter()
                            .enumerate()
                            .map(|(d, &q)| {
                                q * f16::from_bits(keys[GUARD + (key * kv_heads + kv) * dim + d])
                                    .to_f64()
                            })
                            .sum::<f64>()
                    })
                    .collect();
                let sink = if has_sinks {
                    bf16::from_f32(biases[head]).to_f64()
                } else {
                    f64::NEG_INFINITY
                };
                let peak = scores.iter().copied().fold(sink, f64::max);
                let probabilities: Vec<_> = scores.iter().map(|&s| (s - peak).exp()).collect();
                let mass = probabilities.iter().sum::<f64>() + (sink - peak).exp();

                // These power-of-two operands make every QK product and partial sum exact in
                // FP32. Use gamma_n for the remaining FP32 softmax/value stages, plus the
                // half-probability relative and subnormal bounds. Weight magnitudes, rather
                // than a cancelled result's ULP, carry those errors through the value sum.
                let active = end - begin;
                let operations = (2 * active + 12 * active.div_ceil(32) + 16) as f64;
                let gamma = operations * 2.0f64.powi(-24);
                let gamma = gamma / (1.0 - gamma);
                for d in 0..dim {
                    let mut expected = 0.0;
                    let mut magnitude = 0.0;
                    let mut unweighted_magnitude = 0.0;
                    for (offset, &probability) in probabilities.iter().enumerate() {
                        let key = begin + offset;
                        let value = f16::from_bits(values[GUARD + (key * kv_heads + kv) * dim + d])
                            .to_f64();
                        expected += probability * value;
                        magnitude += probability * value.abs();
                        unweighted_magnitude += value.abs();
                    }
                    expected /= mass;
                    magnitude /= mass;
                    let relative_error =
                        (1.0 + 2.0f64.powi(-11)) * (1.0 + gamma) / (1.0 - gamma) - 1.0;
                    let internal_error = relative_error * magnitude
                        + 2.0f64.powi(-25) * unweighted_magnitude / (mass * (1.0 - gamma));
                    // Rounding the final result to BF16 adds at most u_bf16 times its magnitude.
                    let bound = internal_error * (1.0 + 2.0f64.powi(-8))
                        + expected.abs() * 2.0f64.powi(-8)
                        + 2.0f64.powi(-134);
                    let actual =
                        bf16::from_bits(output[GUARD + (token * heads + head) * dim + d]).to_f64();
                    assert!(
                        (actual - expected).abs() <= bound,
                        "D={dim} position={position} window={window} token={token} head={head} dim={d}: \
                         actual={actual} reference={expected} bound={bound}"
                    );
                }
            }
        }
    }
    Ok(())
}
