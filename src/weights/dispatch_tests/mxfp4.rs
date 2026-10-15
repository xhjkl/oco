use super::{GUARD, SENTINEL, noise};
use crate::{
    gpu::{
        Gpu, dispatch, dispatch_indirect,
        tests::{buffer, read, write},
    },
    weights::{
        compute::Compute,
        pipelines::{Builder, ExpertPipelines},
    },
};
use anyhow::Result;
use half::{bf16, f16};

// Unaligned GGUF blocks exercise byte-addressed E8M0 scales and paired E2M1 nibbles.
const WEIGHT_OFFSET: usize = 7;

#[test]
fn mxfp4_decode_matches_independent_coefficients() -> Result<()> {
    let gpu = Gpu::new()?;
    let mut builder = Builder::new(&gpu)?;
    let pipelines = builder.experts()?;
    check_decode_reference(&gpu, &pipelines, 2880, 33)
}

#[test]
fn routed_experts_preserve_half_products_and_route_boundaries() -> Result<()> {
    let gpu = Gpu::new()?;
    let mut builder = Builder::new(&gpu)?;
    let pipelines = builder.experts()?;
    // Sparse experts and one overfull expert exercise both production indirect tile plans.
    for batch in [17, 257] {
        check_experts(&gpu, &pipelines, batch, 96, 96, 65)?;
    }
    Ok(())
}

fn logits_selecting(ids: &[u32]) -> Vec<f32> {
    let mut logits = vec![-10.0f32; 32];
    for (rank, &id) in ids.iter().enumerate() {
        logits[id as usize] = 4.0 - rank as f32;
    }
    logits
}

fn check_decode_reference(
    gpu: &Gpu,
    pipelines: &ExpertPipelines,
    k: usize,
    n: usize,
) -> Result<()> {
    let compute = Compute::new(gpu, None);
    let halves = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let mut bytes = vec![0xa5; WEIGHT_OFFSET];
    let mut coefficients = Vec::with_capacity(4 * n * k);
    for expert in 0..4 {
        for row in 0..n {
            for block in 0..k / 32 {
                let power = (noise(expert * n + row + block % (k / 64)) % 9) as i32 - 5;
                let scale = 2.0f64.powi(power);
                let mut codes = [0u8; 32];
                for (column, code) in codes.iter_mut().enumerate() {
                    let column = block * 32 + column;
                    let source = if row % 3 == 0 {
                        column % (k / 2)
                    } else {
                        column
                    };
                    *code = (noise((expert * n + row) * k + source + 1327) % 16) as u8;
                    if row % 3 == 0 && column >= k / 2 {
                        *code ^= 8;
                    }
                    coefficients.push(f64::from(halves[*code as usize]) * 0.5 * scale);
                }
                bytes.push((127 + power) as u8);
                for column in 0..16 {
                    bytes.push(codes[column] | (codes[column + 16] << 4));
                }
            }
        }
    }
    let weight = buffer(gpu, &bytes);
    let input: Vec<_> = (0..4 * k)
        .map(|i| {
            let index = i / k * k + i % (k / 2);
            ((noise(index + 8719) % 10007) as i32 - 5003) as f32 / 4093.0
        })
        .collect();
    let bf_input: Vec<_> = input[..k]
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect();
    let bf_values: Vec<_> = bf_input
        .iter()
        .map(|&value| bf16::from_bits(value).to_f32())
        .collect();
    let bf_input = buffer(gpu, &bf_input);
    let fp_input = buffer(gpu, &input);
    let bias: Vec<_> = (0..4 * n)
        .map(|i| ((noise(i + 151) % 997) as i32 - 498) as f32 / 509.0)
        .collect();
    let biases = buffer(gpu, &bias);
    let ids = [3u32, 0, 2, 1];
    let routing = buffer(gpu, &ids);
    let logits = buffer(gpu, &logits_selecting(&ids));
    let probs = buffer(gpu, &[0.0f32; 4]);
    let mut up = vec![SENTINEL; 4 * n + 2 * GUARD];
    let mut gate = up.clone();
    let up_buffer = buffer(gpu, &up);
    let gate_buffer = buffer(gpu, &gate);
    let params = [k as u32, n as u32, 4, 0, 4, 4, 1];
    let submission = compute.submit(|encoder| {
        dispatch(
            encoder,
            &pipelines.expand,
            &[
                (&weight, WEIGHT_OFFSET),
                (&weight, WEIGHT_OFFSET),
                (&bf_input, 0),
                (&up_buffer, GUARD * 4),
                (&gate_buffer, GUARD * 4),
                (&biases, 0),
                (&biases, 0),
                (&routing, 0),
                (&logits, 0),
                (&probs, 0),
            ],
            bytemuck::cast_slice(&params),
            [n.div_ceil(16), 4, 1],
            256,
        );
    });
    futures_lite::future::block_on(submission)?;
    read(&up_buffer, &mut up);
    read(&gate_buffer, &mut gate);
    for storage in [&up, &gate] {
        assert!(storage[..GUARD].iter().all(|&word| word == SENTINEL));
        assert!(
            storage[storage.len() - GUARD..]
                .iter()
                .all(|&word| word == SENTINEL)
        );
    }
    for (route, &expert) in ids.iter().enumerate() {
        for row in 0..n {
            let offset = (expert as usize * n + row) * k;
            for (output, shift) in [(&up, 1.0), (&gate, 0.0)] {
                let bias = bf16::from_f32(bias[expert as usize * n + row] + shift).to_f64();
                let reference = reference_dot(&coefficients[offset..offset + k], &bf_values, bias);
                let actual = f64::from(f32::from_bits(output[GUARD + route * n + row]));
                check_roundoff(actual, reference, k, false, route, row);
            }
        }
    }
    {
        let mut output = vec![0x7fc1u16; 4 * n + 2 * GUARD];
        let buffer = buffer(gpu, &output);
        let params = [k as u32, n as u32, 4, 1, 4, 4, 1];
        let submission = compute.submit(|encoder| {
            dispatch(
                encoder,
                &pipelines.down_vector,
                &[
                    (&weight, WEIGHT_OFFSET),
                    (&fp_input, 0),
                    (&buffer, GUARD * 2),
                    (&biases, 0),
                    (&routing, 0),
                ],
                bytemuck::cast_slice(&params),
                [n.div_ceil(32), 4, 1],
                256,
            );
        });
        futures_lite::future::block_on(submission)?;
        read(&buffer, &mut output);
        assert!(output[..GUARD].iter().all(|&word| word == 0x7fc1));
        assert!(
            output[output.len() - GUARD..]
                .iter()
                .all(|&word| word == 0x7fc1)
        );
        for (route, &expert) in ids.iter().enumerate() {
            let input = &input[route * k..(route + 1) * k];
            for row in 0..n {
                let offset = (expert as usize * n + row) * k;
                let bias = bf16::from_f32(bias[expert as usize * n + row]).to_f64();
                let reference = reference_dot(&coefficients[offset..offset + k], input, bias);
                let actual = bf16::from_bits(output[GUARD + route * n + row]).to_f64();
                check_roundoff(actual, reference, k, true, route, row);
            }
        }
    }
    let bad_header = WEIGHT_OFFSET + (ids[0] as usize * n) * k / 32 * 17;
    write(&weight, bad_header, &[255u8]);
    let submission = compute.submit(|encoder| {
        dispatch(
            encoder,
            &pipelines.expand,
            &[
                (&weight, WEIGHT_OFFSET),
                (&weight, WEIGHT_OFFSET),
                (&bf_input, 0),
                (&up_buffer, GUARD * 4),
                (&gate_buffer, GUARD * 4),
                (&biases, 0),
                (&biases, 0),
                (&routing, 0),
                (&logits, 0),
                (&probs, 0),
            ],
            bytemuck::cast_slice(&params),
            [n.div_ceil(16), 4, 1],
            256,
        );
    });
    futures_lite::future::block_on(submission)?;
    read(&up_buffer, &mut up);
    assert!(
        !f32::from_bits(up[GUARD]).is_finite(),
        "reserved E8M0 scale must propagate NaN"
    );
    Ok(())
}

fn reference_dot(coefficients: &[f64], input: &[f32], bias: f64) -> (f64, f64) {
    let mut sum = bias;
    let mut magnitude = bias.abs();
    for (&coefficient, &input) in coefficients.iter().zip(input) {
        let product = coefficient * f64::from(input);
        sum += product;
        magnitude += product.abs();
    }
    (sum, magnitude)
}

fn expert_reference(coefficients: &[f32], input: &[f32], bias: f32) -> (f64, f64) {
    let mut sum = f64::from(bias);
    let mut magnitude = sum.abs();
    for (&coefficient, &input) in coefficients.iter().zip(input) {
        let coefficient = f16::from_f32(coefficient).to_f64();
        let input = f16::from_f32(input).to_f64();
        let product = coefficient * input;
        sum += product;
        magnitude += product.abs();
    }
    (sum, magnitude)
}

fn check_roundoff(
    actual: f64,
    reference: (f64, f64),
    k: usize,
    stored_bf16: bool,
    route: usize,
    row: usize,
) {
    let (expected, magnitude) = reference;
    // Using the existing dense-dot gamma bound; BF16 storage contributes one final rounding.
    let operations = (2 * k + 8) as f64;
    let roundoff = 2.0f64.powi(-24);
    let gamma = operations * roundoff / (1.0 - operations * roundoff);
    let storage_roundoff = if stored_bf16 { 2.0f64.powi(-8) } else { 0.0 };
    let bound = (1.0 + storage_roundoff) * gamma * magnitude + storage_roundoff * expected.abs();
    assert!(
        actual.is_finite() && (actual - expected).abs() <= bound,
        "MXFP4 K={k} route={route} row={row} BF16={stored_bf16}: actual={actual} reference={expected} error={} bound={bound}",
        (actual - expected).abs()
    );
}

fn weights(k: usize, n: usize, salt: usize) -> (Vec<u8>, Vec<f32>) {
    let halves = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let mut bytes = vec![0xa5; WEIGHT_OFFSET];
    let mut values = Vec::with_capacity(32 * n * k);
    for expert in 0..32 {
        for row in 0..n {
            for block in 0..k / 32 {
                let power = ((expert + row + block + salt) % 5) as i32 - 3;
                let scale = 2.0f32.powi(power);
                let mut codes = [0; 32];
                for (column, code) in codes.iter_mut().enumerate() {
                    *code =
                        ((column * 13 + column / 16 * 5 + expert * 3 + row * 5 + block * 7 + salt)
                            % 16) as u8;
                    values.push(halves[*code as usize] as f32 * 0.5 * scale);
                }
                bytes.push((127 + power) as u8);
                for column in 0..16 {
                    bytes.push(codes[column] | (codes[column + 16] << 4));
                }
            }
        }
    }
    (bytes, values)
}

fn check_experts(
    gpu: &Gpu,
    pipelines: &ExpertPipelines,
    batch: usize,
    k: usize,
    n: usize,
    down_n: usize,
) -> Result<()> {
    let compute = Compute::new(gpu, None);
    let routes = batch * 4;
    let (up_bytes, up_weights) = weights(k, n, 0);
    let (gate_bytes, gate_weights) = weights(k, n, 3);
    let (down_bytes, down_weights) = weights(n, down_n, 5);
    let up_weight = buffer(gpu, &up_bytes);
    let gate_weight = buffer(gpu, &gate_bytes);
    let down_weight = buffer(gpu, &down_bytes);
    let input: Vec<_> = (0..batch * k)
        .map(|i| {
            let value = ((i / k * 3 + i % k * 5) % 17) as f32 / 16.0 - 0.5;
            bf16::from_f32(value).to_bits()
        })
        .collect();
    let x = buffer(gpu, &input);
    let up_bias: Vec<_> = (0..32 * n)
        .map(|i| ((i * 7 % 97) as f32 - 48.0) / 100.0)
        .collect();
    let gate_bias: Vec<_> = (0..32 * n)
        .map(|i| ((i * 3 % 97) as f32 - 48.0) / 100.0)
        .collect();
    let down_bias: Vec<_> = (0..32 * down_n)
        .map(|i| ((i * 5 % 97) as f32 - 48.0) / 100.0)
        .collect();
    let ub = buffer(gpu, &up_bias);
    let gb = buffer(gpu, &gate_bias);
    let db = buffer(gpu, &down_bias);
    // Expert 31 and expert 30 exceed one tile in the largest batch; other experts stay sparse.
    let ids: Vec<_> = (0..batch)
        .flat_map(|token| [31, (token % 15) as u32, 15 + (token % 8) as u32, 30])
        .collect();
    let routing = buffer(gpu, &ids);
    let mut plan =
        vec![SENTINEL; (6 + 2 * (routes.div_ceil(16) + 32) + 32 * (routes + 1)) + 2 * GUARD];
    let route_plan = buffer(gpu, &plan);
    let mut up = vec![SENTINEL; routes * n + 2 * GUARD];
    let mut gate = up.clone();
    let uy = buffer(gpu, &up);
    let gy = buffer(gpu, &gate);
    let mut down = vec![0x7fc1u16; routes * down_n + 2 * GUARD];
    let dy = buffer(gpu, &down);
    let up_params = [k as u32, n as u32, routes as u32, 0, 4, 32, 1];
    let down_params = [n as u32, down_n as u32, routes as u32, 1, 4, 32, 1];
    let plan_params = [
        routes as u32,
        (2 * n.div_ceil(64)) as u32,
        down_n.div_ceil(64) as u32,
    ];
    let (plan_pipeline, up_pipeline, down_pipeline) = if batch >= 256 {
        (
            &pipelines.route_plan_large,
            &pipelines.up_large,
            &pipelines.down_large,
        )
    } else {
        (&pipelines.route_plan, &pipelines.up, &pipelines.down)
    };
    let submission = compute.submit(|encoder| {
        dispatch(
            encoder,
            plan_pipeline,
            &[(&routing, 0), (&route_plan, GUARD * 4)],
            bytemuck::cast_slice(&plan_params),
            [1, 1, 1],
            32,
        );
        dispatch_indirect(
            encoder,
            up_pipeline,
            &[
                (&up_weight, WEIGHT_OFFSET),
                (&gate_weight, WEIGHT_OFFSET),
                (&x, 0),
                (&uy, GUARD * 4),
                (&gy, GUARD * 4),
                (&ub, 0),
                (&gb, 0),
                (&route_plan, GUARD * 4),
            ],
            bytemuck::cast_slice(&up_params),
            (&route_plan, GUARD * 4),
            128,
        );
        dispatch_indirect(
            encoder,
            down_pipeline,
            &[
                (&down_weight, WEIGHT_OFFSET),
                (&uy, GUARD * 4),
                (&dy, GUARD * 2),
                (&db, 0),
                (&route_plan, GUARD * 4),
            ],
            bytemuck::cast_slice(&down_params),
            (&route_plan, GUARD * 4 + 12),
            128,
        );
    });
    futures_lite::future::block_on(submission)?;
    read(&uy, &mut up);
    read(&gy, &mut gate);
    read(&dy, &mut down);
    read(&route_plan, &mut plan);
    for storage in [&up, &gate, &plan] {
        assert!(storage[..GUARD].iter().all(|&word| word == SENTINEL));
        assert!(
            storage[storage.len() - GUARD..]
                .iter()
                .all(|&word| word == SENTINEL)
        );
    }
    assert!(down[..GUARD].iter().all(|&word| word == 0x7fc1));
    assert!(
        down[down.len() - GUARD..]
            .iter()
            .all(|&word| word == 0x7fc1)
    );
    let mut expanded = vec![0.0; routes * n];
    for (route, &expert) in ids.iter().enumerate() {
        let expert = expert as usize;
        let values: Vec<_> = input[route / 4 * k..(route / 4 + 1) * k]
            .iter()
            .map(|&value| bf16::from_bits(value).to_f32())
            .collect();
        for (weights, bias, output, shift) in [
            (&up_weights, &up_bias, &up, 1.0),
            (&gate_weights, &gate_bias, &gate, 0.0),
        ] {
            for row in 0..n {
                let offset = (expert * n + row) * k;
                let bias = bf16::from_f32(bias[expert * n + row] + shift).to_f32();
                let reference = expert_reference(&weights[offset..offset + k], &values, bias);
                let actual = f32::from_bits(output[GUARD + route * n + row]);
                check_roundoff(f64::from(actual), reference, k, false, route, row);
                if shift == 1.0 {
                    // The contraction consumes this already-validated FP32 projection output.
                    expanded[route * n + row] = actual;
                }
            }
        }
        for row in 0..down_n {
            let offset = (expert * down_n + row) * n;
            let bias = bf16::from_f32(down_bias[expert * down_n + row]).to_f32();
            let reference = expert_reference(
                &down_weights[offset..offset + n],
                &expanded[route * n..(route + 1) * n],
                bias,
            );
            let actual = bf16::from_bits(down[GUARD + route * down_n + row]).to_f64();
            check_roundoff(actual, reference, n, true, route, row);
        }
    }
    Ok(())
}
