use super::{GUARD, SENTINEL};
use crate::gpu::{
    Gpu, dispatch,
    tests::{buffer, read},
};
use crate::weights::{compute::Compute, pipelines::Builder};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use half::bf16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Ssm {
    channels: u32,
    m: u32,
    stride: u32,
    offset: u32,
    heads: u32,
    head_dim: u32,
    groups: u32,
    state: u32,
    dt_offset: u32,
    epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RouterParams {
    n: u32,
    experts: u32,
    active: u32,
    normalize: u32,
    scale: f32,
}

/// Check the convolution, selective scan, and gated group norm against a scalar reference,
/// and that chunked updates reproduce one whole-sequence pass.
#[test]
fn mamba_block_matches_reference_across_chunking() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let pipelines = builder.nemotron()?;
    let (heads, head_dim, groups, state, m) = (6usize, 8usize, 2usize, 128usize, 7usize);
    let inner = heads * head_dim;
    let channels = inner + 2 * groups * state;
    let stride = inner + channels + heads;
    let bf = |value: f32| bf16::from_f32(value).to_bits();
    let projected: Vec<u16> = (0..m * stride)
        .map(|i| bf(((i * 7 + i / stride * 13) % 53) as f32 / 24.0 - 1.1))
        .collect();
    let conv_w: Vec<f32> = (0..4 * channels)
        .map(|i| ((i * 5) % 17) as f32 / 16.0 - 0.5)
        .collect();
    let conv_b: Vec<f32> = (0..channels)
        .map(|i| ((i * 3) % 11) as f32 / 32.0 - 0.15)
        .collect();
    let a: Vec<f32> = (0..heads).map(|h| -0.3 - h as f32 * 0.7).collect();
    let d: Vec<f32> = (0..heads).map(|h| 1.5 - h as f32 * 0.4).collect();
    let dt_bias: Vec<f32> = (0..heads).map(|h| h as f32 * 0.5 - 1.0).collect();
    let norm_w: Vec<f32> = (0..inner).map(|i| 0.75 + (i % 7) as f32 / 16.0).collect();
    let conv_state: Vec<f32> = (0..channels * 3)
        .map(|i| ((i * 11) % 23) as f32 / 40.0 - 0.3)
        .collect();
    let ssm_state: Vec<f32> = (0..inner * state)
        .map(|i| ((i * 13) % 31) as f32 / 256.0 - 0.06)
        .collect();
    let epsilon = 1e-5f32;
    let dt_in: Vec<f32> = (0..m * heads)
        .map(|i| ((i * 29) % 37) as f32 / 12.0 - 1.4)
        .collect();

    // Scalar reference in f64.
    let f = |bits: u16| bf16::from_bits(bits).to_f64();
    let mut ref_conv_state: Vec<f64> = conv_state.iter().map(|&v| f64::from(v)).collect();
    let mut ref_ssm: Vec<f64> = ssm_state.iter().map(|&v| f64::from(v)).collect();
    let mut convolved = vec![0.0f64; m * channels];
    for c in 0..channels {
        let (mut p0, mut p1, mut p2) = (
            ref_conv_state[c * 3],
            ref_conv_state[c * 3 + 1],
            ref_conv_state[c * 3 + 2],
        );
        for t in 0..m {
            let v = f(projected[t * stride + inner + c]);
            let s = f64::from(conv_w[c * 4]) * p0
                + f64::from(conv_w[c * 4 + 1]) * p1
                + f64::from(conv_w[c * 4 + 2]) * p2
                + f64::from(conv_w[c * 4 + 3]) * v
                + f64::from(conv_b[c]);
            let activated = s / (1.0 + (-s).exp());
            convolved[t * channels + c] = activated;
            p0 = p1;
            p1 = p2;
            p2 = v;
        }
        ref_conv_state[c * 3] = p0;
        ref_conv_state[c * 3 + 1] = p1;
        ref_conv_state[c * 3 + 2] = p2;
    }
    let mut mixed = vec![0u16; m * inner];
    let mut scanned = vec![0.0f64; m * inner];
    for t in 0..m {
        for h in 0..heads {
            let g = h / (heads / groups);
            let dt = f64::from(dt_in[t * heads + h]) + f64::from(dt_bias[h]);
            let dt = if dt <= 20.0 {
                (1.0 + dt.exp()).ln()
            } else {
                dt
            };
            let decay = (dt * f64::from(a[h])).exp();
            for ch in 0..head_dim {
                let x = convolved[t * channels + h * head_dim + ch];
                let row = (h * head_dim + ch) * state;
                let mut acc = 0.0;
                for n in 0..state {
                    let b = convolved[t * channels + inner + g * state + n];
                    let c = convolved[t * channels + inner + groups * state + g * state + n];
                    ref_ssm[row + n] = ref_ssm[row + n] * decay + b * x * dt;
                    acc += ref_ssm[row + n] * c;
                }
                scanned[t * inner + h * head_dim + ch] = acc + f64::from(d[h]) * x;
            }
        }
        let size = inner / groups;
        for g in 0..groups {
            let mut values = vec![0.0f64; size];
            let mut sq = 0.0;
            for i in 0..size {
                let z = f(projected[t * stride + g * size + i]);
                values[i] = scanned[t * inner + g * size + i] * z / (1.0 + (-z).exp());
                sq += values[i] * values[i];
            }
            let inv = 1.0 / (sq / size as f64 + f64::from(epsilon)).sqrt();
            for i in 0..size {
                mixed[t * inner + g * size + i] =
                    bf16::from_f64(values[i] * inv * f64::from(norm_w[g * size + i])).to_bits();
            }
        }
    }

    let projected_buffer = buffer(&gpu, &projected);
    let conv_w = buffer(&gpu, &conv_w);
    let conv_b = buffer(&gpu, &conv_b);
    let a = buffer(&gpu, &a);
    let d = buffer(&gpu, &d);
    let dt_bias = buffer(&gpu, &dt_bias);
    let dt_in = buffer(&gpu, &dt_in);
    let norm_w = buffer(&gpu, &norm_w);
    let mut expected_state = None;
    for chunk in [m, 1, 3] {
        let mut conv_words = vec![SENTINEL; GUARD + channels * 3 + GUARD];
        for (slot, value) in conv_words[GUARD..].iter_mut().zip(&conv_state) {
            *slot = value.to_bits();
        }
        let mut ssm_words = vec![SENTINEL; GUARD + inner * state + GUARD];
        for (slot, value) in ssm_words[GUARD..].iter_mut().zip(&ssm_state) {
            *slot = value.to_bits();
        }
        let conv_buffer = buffer(&gpu, &conv_words);
        let ssm_buffer = buffer(&gpu, &ssm_words);
        let convolved_buffer = buffer(&gpu, &vec![SENTINEL; GUARD + m * channels + GUARD]);
        let scanned_buffer = buffer(&gpu, &vec![SENTINEL; GUARD + m * inner + GUARD]);
        let mut out = vec![0x7fc1u16; GUARD + m * inner + GUARD];
        let out_buffer = buffer(&gpu, &out);
        let submission = compute.submit(|encoder| {
            for start in (0..m).step_by(chunk) {
                let rows = chunk.min(m - start);
                let p = Ssm {
                    channels: channels as u32,
                    m: rows as u32,
                    stride: stride as u32,
                    offset: inner as u32,
                    heads: heads as u32,
                    head_dim: head_dim as u32,
                    groups: groups as u32,
                    state: state as u32,
                    dt_offset: (inner + channels) as u32,
                    epsilon,
                };
                dispatch(
                    encoder,
                    &pipelines.conv,
                    &[
                        (&projected_buffer, start * stride * 2),
                        (&conv_w, 0),
                        (&conv_b, 0),
                        (&conv_buffer, GUARD * 4),
                        (&convolved_buffer, (GUARD + start * channels) * 4),
                    ],
                    bytemuck::bytes_of(&p),
                    [channels.div_ceil(256), 1, 1],
                    256,
                );
                dispatch(
                    encoder,
                    &pipelines.scan,
                    &[
                        (&convolved_buffer, (GUARD + start * channels) * 4),
                        (&dt_in, start * heads * 4),
                        (&a, 0),
                        (&d, 0),
                        (&dt_bias, 0),
                        (&ssm_buffer, GUARD * 4),
                        (&scanned_buffer, (GUARD + start * inner) * 4),
                    ],
                    bytemuck::bytes_of(&p),
                    [head_dim.div_ceil(4), heads, 1],
                    128,
                );
                dispatch(
                    encoder,
                    &pipelines.gate_norm,
                    &[
                        (&scanned_buffer, (GUARD + start * inner) * 4),
                        (&projected_buffer, start * stride * 2),
                        (&norm_w, 0),
                        (&out_buffer, (GUARD + start * inner) * 2),
                    ],
                    bytemuck::bytes_of(&p),
                    [groups, rows, 1],
                    256,
                );
            }
        });
        futures_lite::future::block_on(submission)?;
        read(&conv_buffer, &mut conv_words);
        read(&ssm_buffer, &mut ssm_words);
        read(&out_buffer, &mut out);
        for words in [&conv_words, &ssm_words] {
            assert!(words[..GUARD].iter().all(|&w| w == SENTINEL));
            assert!(words[words.len() - GUARD..].iter().all(|&w| w == SENTINEL));
        }
        assert!(out[..GUARD].iter().all(|&w| w == 0x7fc1));
        assert!(out[out.len() - GUARD..].iter().all(|&w| w == 0x7fc1));
        for (i, (&actual, &expected)) in conv_words[GUARD..GUARD + channels * 3]
            .iter()
            .zip(&ref_conv_state)
            .enumerate()
        {
            let actual = f64::from(f32::from_bits(actual));
            assert!(
                (actual - expected).abs() <= 1e-5,
                "conv state {i} chunk {chunk}: {actual} vs {expected}"
            );
        }
        for (i, (&actual, &expected)) in ssm_words[GUARD..GUARD + inner * state]
            .iter()
            .zip(&ref_ssm)
            .enumerate()
        {
            let actual = f64::from(f32::from_bits(actual));
            assert!(
                actual.is_finite() && (actual - expected).abs() <= 1e-4 + 1e-4 * expected.abs(),
                "ssm state {i} chunk {chunk}: {actual} vs {expected}"
            );
        }
        for (i, (&actual, &expected)) in
            out[GUARD..GUARD + m * inner].iter().zip(&mixed).enumerate()
        {
            let actual = bf16::from_bits(actual).to_f32();
            let expected = bf16::from_bits(expected).to_f32();
            let next = bf16::from_bits(bf16::from_f32(expected.abs()).to_bits() + 1).to_f32();
            let tolerance = 2.0 * (next - expected.abs()).max(1e-4);
            assert!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
                "output {i} chunk {chunk}: {actual} vs {expected}"
            );
        }
        if let Some(previous) = &expected_state {
            assert_eq!(&ssm_words, previous, "state must not depend on chunking");
        }
        expected_state = Some(ssm_words);
    }
    Ok(())
}

/// Sigmoid routing selects by biased score, weights by unbiased probability, normalizes and scales.
#[test]
fn sigmoid_router_selects_biased_and_weights_unbiased() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let pipelines = builder.nemotron()?;
    let (n, experts, active, rows) = (256usize, 100usize, 6usize, 3usize);
    let x: Vec<u16> = (0..rows * n)
        .map(|i| bf16::from_f32(((i * 7) % 19) as f32 / 8.0 - 1.0).to_bits())
        .collect();
    let w: Vec<f32> = (0..experts * n)
        .map(|i| ((i * 13 + i / n) % 29) as f32 / 64.0 - 0.2)
        .collect();
    let bias: Vec<f32> = (0..experts)
        .map(|e| if e % 10 == 3 { 0.9 } else { 0.0 })
        .collect();
    let scale = 2.5f32;
    let mut expected_ids = Vec::new();
    let mut expected_probs = Vec::new();
    for t in 0..rows {
        let logits: Vec<f64> = (0..experts)
            .map(|e| {
                (0..n)
                    .map(|k| bf16::from_bits(x[t * n + k]).to_f64() * f64::from(w[e * n + k]))
                    .sum()
            })
            .collect();
        let probs: Vec<f64> = logits.iter().map(|&l| 1.0 / (1.0 + (-l).exp())).collect();
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| {
            (probs[b] + f64::from(bias[b]))
                .partial_cmp(&(probs[a] + f64::from(bias[a])))
                .unwrap()
                .then(a.cmp(&b))
        });
        let chosen = &order[..active];
        let total: f64 = chosen
            .iter()
            .map(|&e| probs[e])
            .sum::<f64>()
            .max(6.103515625e-5);
        for &e in chosen {
            expected_ids.push(e as u32);
            expected_probs.push(probs[e] / total * f64::from(scale));
        }
    }
    let x = buffer(&gpu, &x);
    let w = buffer(&gpu, &w);
    let bias = buffer(&gpu, &bias);
    let mut ids = vec![SENTINEL; GUARD + rows * active + GUARD];
    let mut probs = vec![SENTINEL; GUARD + rows * active + GUARD];
    let ids_buffer = buffer(&gpu, &ids);
    let probs_buffer = buffer(&gpu, &probs);
    let p = RouterParams {
        n: n as u32,
        experts: experts as u32,
        active: active as u32,
        normalize: 1,
        scale,
    };
    let submission = compute.submit(|encoder| {
        dispatch(
            encoder,
            &pipelines.router,
            &[
                (&x, 0),
                (&w, 0),
                (&bias, 0),
                (&ids_buffer, GUARD * 4),
                (&probs_buffer, GUARD * 4),
            ],
            bytemuck::bytes_of(&p),
            [rows, 1, 1],
            256,
        );
    });
    futures_lite::future::block_on(submission)?;
    read(&ids_buffer, &mut ids);
    read(&probs_buffer, &mut probs);
    for words in [&ids, &probs] {
        assert!(words[..GUARD].iter().all(|&w| w == SENTINEL));
        assert!(words[words.len() - GUARD..].iter().all(|&w| w == SENTINEL));
    }
    assert_eq!(&ids[GUARD..GUARD + rows * active], &expected_ids[..]);
    for (i, (&actual, &expected)) in probs[GUARD..GUARD + rows * active]
        .iter()
        .zip(&expected_probs)
        .enumerate()
    {
        let actual = f64::from(f32::from_bits(actual));
        assert!(
            (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
            "prob {i}: {actual} vs {expected}"
        );
    }
    // Biased experts (e % 10 == 3) win selection despite lower probability.
    assert!(
        ids[GUARD..GUARD + active]
            .iter()
            .filter(|&&e| e % 10 == 3)
            .count()
            >= 3
    );
    Ok(())
}
