use super::{GUARD, SENTINEL};
use crate::gpu::tests::buffer;
use crate::gpu::{Gpu, dispatch, tests::read};
use crate::weights::{
    compute::Compute,
    pipelines::{Builder, RecurrentPipelines},
};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use half::bf16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Gdn {
    kh: u32,
    vh: u32,
    vd: u32,
    m: u32,
    conv_dim: u32,
    epsilon: f32,
}

struct Fixture {
    params: Gdn,
    qkv: Vec<u16>,
    z: Vec<u16>,
    alpha: Vec<u16>,
    beta: Vec<u16>,
    a: Vec<f32>,
    dt: Vec<f32>,
    norm: Vec<f32>,
    state: Vec<f32>,
}

/// Check the state equation independently and preserve state across decode/prefill partitions.
#[test]
fn recurrent_state_and_partitions_match_reference() -> Result<()> {
    let gpu = Gpu::new()?;
    let mut builder = Builder::new(&gpu)?;
    let pipelines = builder.recurrent()?;
    for (kh, vh, vd, m) in [(3, 6, 17, 9), (16, 48, 128, 21)] {
        let fixture = fixture(kh, vh, vd, m);
        let (expected_state, expected_output) = reference(&fixture);
        let mut prior_state = None;
        for (chunk, persistent) in [(1, false), (m, true), (8, true)] {
            let (state, output) = run_chunks(&gpu, &pipelines, &fixture, chunk, persistent)?;
            for (&actual, &expected) in state.iter().zip(&expected_state) {
                let actual = f64::from(f32::from_bits(actual));
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= 1e-5 + 2e-5 * expected.abs()
                );
            }
            for (&actual, &expected) in output.iter().zip(&expected_output) {
                let actual = bf16::from_bits(actual).to_f32();
                let expected = bf16::from_bits(expected).to_f32();
                let next = bf16::from_bits(bf16::from_f32(expected.abs()).to_bits() + 1).to_f32();
                assert!(
                    actual.is_finite()
                        && (actual - expected).abs() <= (next - expected.abs()).max(1e-5)
                );
            }
            if let Some(prior) = &prior_state {
                assert_eq!(
                    &state, prior,
                    "state partition kh={kh},vh={vh},vd={vd},chunk={chunk}"
                );
            }
            prior_state = Some(state);
        }
    }
    Ok(())
}

fn fixture(kh: usize, vh: usize, vd: usize, m: usize) -> Fixture {
    let conv_dim = 2 * kh * 128 + vh * vd;
    let bf = |value| bf16::from_f32(value).to_bits();
    let qkv = (0..m * conv_dim)
        .map(|i| bf(((i / conv_dim * 3 + i % conv_dim * 7) % 37) as f32 / 16.0 - 1.125))
        .collect();
    let z = (0..m * vh * vd)
        .map(|i| bf(((i * 11 + i / vd * 3) % 41) as f32 / 4.0 - 5.0))
        .collect();
    let alpha = (0..m * vh)
        .map(|i| bf((i * 5 % 25) as f32 / 4.0 - 3.0))
        .collect();
    let beta = (0..m * vh)
        .map(|i| bf((i * 7 % 17) as f32 / 4.0 - 2.0))
        .collect();
    let a = (0..vh).map(|i| -0.5 - (i % 5) as f32 / 32.0).collect();
    let dt = (0..vh).map(|i| (i % 7) as f32 / 8.0 - 0.375).collect();
    let norm = (0..vd).map(|i| 0.875 + (i % 9) as f32 / 32.0).collect();
    let state = (0..vh * vd * 128)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) / 1024.0)
        .collect();
    Fixture {
        params: Gdn {
            kh: kh as u32,
            vh: vh as u32,
            vd: vd as u32,
            m: m as u32,
            conv_dim: conv_dim as u32,
            epsilon: 1e-6,
        },
        qkv,
        z,
        alpha,
        beta,
        a,
        dt,
        norm,
        state,
    }
}

fn reference(f: &Fixture) -> (Vec<f64>, Vec<u16>) {
    let p = f.params;
    let (kh, vh, vd, m) = (p.kh as usize, p.vh as usize, p.vd as usize, p.m as usize);
    let conv_dim = p.conv_dim as usize;
    let mut state: Vec<_> = f.state.iter().map(|&value| f64::from(value)).collect();
    let mut output = vec![0; m * vh * vd];
    let bf = |value| bf16::from_bits(value).to_f64();
    for t in 0..m {
        for h in 0..vh {
            let hk = h % kh;
            let mut q = [0.0f64; 128];
            let mut k = [0.0f64; 128];
            for d in 0..128 {
                q[d] = bf(f.qkv[t * conv_dim + hk * 128 + d]);
                k[d] = bf(f.qkv[t * conv_dim + (kh + hk) * 128 + d]);
            }
            let qi = 1.0 / (q.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt() / 128.0f64.sqrt();
            let ki = 1.0 / (k.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
            for d in 0..128 {
                q[d] *= qi;
                k[d] *= ki;
            }
            let kq = q.iter().zip(&k).map(|(q, k)| q * k).sum::<f64>();
            let beta = 1.0 / (1.0 + (-bf(f.beta[t * vh + h])).exp());
            let alpha = bf(f.alpha[t * vh + h]) + f64::from(f.dt[h]);
            let softplus = alpha.max(0.0) + (-alpha.abs()).exp().ln_1p();
            let decay = (f64::from(f.a[h]) * softplus).exp();
            let mut mixed = vec![0.0; vd];
            for (d, mixed) in mixed.iter_mut().enumerate() {
                let row = (h * vd + d) * 128;
                let sq = state[row..row + 128]
                    .iter()
                    .zip(&q)
                    .map(|(s, q)| s * q)
                    .sum::<f64>();
                let sk = state[row..row + 128]
                    .iter()
                    .zip(&k)
                    .map(|(s, k)| s * k)
                    .sum::<f64>();
                let value = bf(f.qkv[t * conv_dim + 2 * kh * 128 + h * vd + d]);
                let change = beta * (value - decay * sk);
                for column in 0..128 {
                    state[row + column] = decay * state[row + column] + k[column] * change;
                }
                *mixed = decay * sq + change * kq;
            }
            let inv = 1.0
                / (mixed.iter().map(|x| x * x).sum::<f64>() / vd as f64 + f64::from(p.epsilon))
                    .sqrt();
            for (d, mixed) in mixed.iter().enumerate() {
                let index = (t * vh + h) * vd + d;
                let z = bf(f.z[index]);
                output[index] =
                    bf16::from_f64(mixed * inv * f64::from(f.norm[d]) * z / (1.0 + (-z).exp()))
                        .to_bits();
            }
        }
    }
    (state, output)
}

fn run_chunks(
    gpu: &Gpu,
    pipelines: &RecurrentPipelines,
    fixture: &Fixture,
    chunk_size: usize,
    persistent: bool,
) -> Result<(Vec<u32>, Vec<u16>)> {
    let compute = Compute::new(gpu, None);
    let p = fixture.params;
    let (m, vh, vd, conv_dim) = (
        p.m as usize,
        p.vh as usize,
        p.vd as usize,
        p.conv_dim as usize,
    );
    let qkv = buffer(gpu, &fixture.qkv);
    let z = buffer(gpu, &fixture.z);
    let alpha = buffer(gpu, &fixture.alpha);
    let beta = buffer(gpu, &fixture.beta);
    let a = buffer(gpu, &fixture.a);
    let dt = buffer(gpu, &fixture.dt);
    let norm = buffer(gpu, &fixture.norm);
    let mut state = vec![SENTINEL; GUARD + fixture.state.len() + GUARD];
    for (slot, value) in state[GUARD..].iter_mut().zip(&fixture.state) {
        *slot = value.to_bits();
    }
    let state_buffer = buffer(gpu, &state);
    let mut output = vec![0x7fc1u16; GUARD + m * vh * vd + GUARD];
    let output_buffer = buffer(gpu, &output);
    let normalized = 2 * p.kh as usize * 128;
    let factor_width = 2 * vh + p.kh as usize;
    let scratch_counts = [m * normalized, m * factor_width, m * vh * vd];
    let mut scratch = Vec::new();
    for count in scratch_counts {
        let storage = buffer(gpu, &vec![SENTINEL; count + 2 * GUARD]);
        scratch.push(storage);
    }
    let submission = compute.submit(|encoder| {
        for start in (0..m).step_by(chunk_size) {
            let params = Gdn {
                m: chunk_size.min(m - start) as u32,
                ..p
            };
            if persistent {
                dispatch(
                    encoder,
                    &pipelines.delta_prepare,
                    &[
                        (&qkv, start * conv_dim * 2),
                        (&alpha, start * vh * 2),
                        (&beta, start * vh * 2),
                        (&a, 0),
                        (&dt, 0),
                        (&scratch[0], GUARD * 4),
                        (&scratch[1], GUARD * 4),
                    ],
                    bytemuck::bytes_of(&params),
                    [p.kh as usize, params.m as usize, 1],
                    32,
                );
                dispatch(
                    encoder,
                    &pipelines.delta_persistent,
                    &[
                        (&scratch[0], GUARD * 4),
                        (&scratch[1], GUARD * 4),
                        (&qkv, start * conv_dim * 2),
                        (&state_buffer, GUARD * 4),
                        (&scratch[2], GUARD * 4),
                    ],
                    bytemuck::bytes_of(&params),
                    [vd.div_ceil(4), vh, 1],
                    128,
                );
                dispatch(
                    encoder,
                    &pipelines.delta_output,
                    &[
                        (&scratch[2], GUARD * 4),
                        (&z, start * vh * vd * 2),
                        (&norm, 0),
                        (&output_buffer, (GUARD + start * vh * vd) * 2),
                    ],
                    bytemuck::bytes_of(&params),
                    [vh, params.m as usize, 1],
                    128,
                );
            } else {
                dispatch(
                    encoder,
                    &pipelines.delta,
                    &[
                        (&qkv, start * conv_dim * 2),
                        (&z, start * vh * vd * 2),
                        (&alpha, start * vh * 2),
                        (&beta, start * vh * 2),
                        (&a, 0),
                        (&dt, 0),
                        (&norm, 0),
                        (&state_buffer, GUARD * 4),
                        (&output_buffer, (GUARD + start * vh * vd) * 2),
                    ],
                    bytemuck::bytes_of(&params),
                    [vh, 1, 1],
                    512,
                );
            }
        }
    });
    futures_lite::future::block_on(submission)?;
    for (buffer, count) in scratch.iter().zip(scratch_counts) {
        let mut words = vec![0u32; count + 2 * GUARD];
        read(buffer, &mut words);
        assert!(words[..GUARD].iter().all(|&value| value == SENTINEL));
        assert!(
            words[GUARD + count..]
                .iter()
                .all(|&value| value == SENTINEL)
        );
    }
    read(&state_buffer, &mut state);
    read(&output_buffer, &mut output);
    assert!(state[..GUARD].iter().all(|&word| word == SENTINEL));
    assert!(
        state[GUARD + fixture.state.len()..]
            .iter()
            .all(|&word| word == SENTINEL)
    );
    assert!(output[..GUARD].iter().all(|&word| word == 0x7fc1));
    assert!(
        output[GUARD + m * vh * vd..]
            .iter()
            .all(|&word| word == 0x7fc1)
    );
    Ok((
        state[GUARD..GUARD + fixture.state.len()].to_vec(),
        output[GUARD..GUARD + m * vh * vd].to_vec(),
    ))
}
