use super::{GUARD, noise};
use crate::{
    gpu::{
        Gpu, dispatch,
        tests::{buffer, read},
    },
    tensor::WeightFormat,
    weights::{compute::Compute, pipelines::Builder},
};
use anyhow::Result;
use half::{bf16, f16};

#[path = "q4_prefill.rs"]
mod q4_prefill;

/// Decode independently packed GGUF rows, including SIMD tails, before FP32 accumulation.
#[test]
fn dense_products_match_declared_operand_precision() -> Result<()> {
    use WeightFormat::*;
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    for format in [F32, F16, Bf16, Q4_0, Q4_1, Q8_0, Q4K, Q5K, Q6K] {
        let (k, n) = (512usize, 33usize);
        let (bytes, coefficients) = dense_weights(format, n, k);
        let weights = buffer(&gpu, &bytes);
        let bias = buffer(&gpu, &[0.0f32]);
        let ids = buffer(&gpu, &[0u32]);
        let (symbol, columns) = match format {
            Q4_0 | Q4_1 => ("matvec_q4", n.div_ceil(32)),
            Q5K => ("matvec_q5k", n.div_ceil(16)),
            Q6K => ("matvec_q6k", n.div_ceil(16)),
            F32 | F16 | Bf16 => ("matvec_split", n),
            _ => ("matvec", n.div_ceil(32)),
        };
        // Pairing each packed decode format with the two live tiled families.
        let mut cases = vec![(symbol, 0, 2usize, columns, 256)];
        if matches!(format, Q4_0 | Q4_1) {
            cases.push(("matmul_q4", 3, 21, n.div_ceil(64), 128));
        }
        if matches!(format, Q5K | Q8_0) {
            cases.push(("matmul", 2, 9, n.div_ceil(64), 128));
        }
        for (symbol, variant, m, columns, threads) in cases {
            let pipeline = builder.get(symbol, format, variant)?;
            let input: Vec<_> = (0..m * k)
                .map(|i| bf16::from_f32((noise(i + 9173) % 251) as f32 / 127.0 - 1.0).to_bits())
                .collect();
            let expected: Vec<_> = input
                .chunks_exact(k)
                .flat_map(|x| {
                    coefficients
                        .chunks_exact(k)
                        .map(move |w| product_reference(w, x, threads == 128))
                })
                .collect();
            let x = buffer(&gpu, &input);
            let mut output = vec![0x7fc1u16; GUARD + m * n + GUARD];
            let y = buffer(&gpu, &output);
            let params = [k as u32, n as u32, m as u32, 0, 4, 1, 0];
            let groups = if threads == 128 {
                [m.div_ceil(8 * variant as usize), columns, 1]
            } else {
                [columns, m, 1]
            };
            let submission = compute.submit(|encoder| {
                dispatch(
                    encoder,
                    &pipeline,
                    &[
                        (&weights, 0),
                        (&x, 0),
                        (&y, GUARD * 2),
                        (&bias, 0),
                        (&ids, 0),
                    ],
                    bytemuck::cast_slice(&params),
                    groups,
                    threads,
                );
            });
            futures_lite::future::block_on(submission)?;
            read(&y, &mut output);
            assert!(
                output[..GUARD]
                    .iter()
                    .chain(&output[GUARD + m * n..])
                    .all(|&v| v == 0x7fc1)
            );
            for (i, (&bits, &(expected, magnitude))) in output[GUARD..GUARD + m * n]
                .iter()
                .zip(&expected)
                .enumerate()
            {
                let actual = bf16::from_bits(bits).to_f64();
                let bound = roundoff_bound(expected, magnitude, k);
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= bound,
                    "{symbol} {format:?} output {i}: {actual} != {expected}, bound={bound}"
                );
            }
        }
    }
    Ok(())
}

fn half_operand(value: f64) -> f64 {
    let value = f16::from_f32(value as f32).to_f64();
    assert!(
        value.is_finite(),
        "fixture exceeds finite FP16 operand domain"
    );
    value
}

/// Accumulate in FP64 after applying the kernel's explicit operand conversions.
fn product_reference(coefficients: &[(f64, f64)], input: &[u16], half: bool) -> (f64, f64) {
    let mut sum = 0.0;
    let mut magnitude = 0.0;
    for (&(coefficient, terms), &input) in coefficients.iter().zip(input) {
        let input = bf16::from_bits(input).to_f64();
        let (coefficient, input, terms) = if half {
            let coefficient = half_operand(coefficient);
            (coefficient, half_operand(input), coefficient.abs())
        } else {
            (coefficient, input, terms)
        };
        sum += coefficient * input;
        magnitude += terms * input.abs();
    }
    (sum, magnitude)
}

fn roundoff_bound(expected: f64, magnitude: f64, k: usize) -> f64 {
    let unit = 2.0f64.powi(-24);
    let operations = (2 * k + 8) as f64;
    let gamma = operations * unit / (1.0 - operations * unit);
    let storage = 2.0f64.powi(-8);
    (1.0 + storage) * gamma * magnitude + storage * expected.abs()
}

/// Pack GGUF codes while retaining independently defined coefficients and affine magnitudes.
fn dense_weights(format: WeightFormat, n: usize, k: usize) -> (Vec<u8>, Vec<(f64, f64)>) {
    let mut bytes = Vec::new();
    let mut coefficients = Vec::with_capacity(n * k);
    for row in 0..n {
        if matches!(
            format,
            WeightFormat::F32 | WeightFormat::F16 | WeightFormat::Bf16
        ) {
            for column in 0..k {
                let value = (noise(row * k + column + 37) % 2001) as i32 - 1000;
                let value = value as f32 * 0.000_131;
                let value = match format {
                    WeightFormat::F32 => {
                        bytes.extend(value.to_le_bytes());
                        f64::from(value)
                    }
                    WeightFormat::F16 => {
                        let value = f16::from_f32(value);
                        bytes.extend(value.to_bits().to_le_bytes());
                        value.to_f64()
                    }
                    WeightFormat::Bf16 => {
                        let value = bf16::from_f32(value);
                        bytes.extend(value.to_bits().to_le_bytes());
                        value.to_f64()
                    }
                    _ => unreachable!(),
                };
                coefficients.push((value, value.abs()));
            }
            continue;
        }
        let block = if matches!(
            format,
            WeightFormat::Q4_0 | WeightFormat::Q4_1 | WeightFormat::Q8_0
        ) {
            32
        } else {
            256
        };
        for start in (0..k).step_by(block) {
            let seed = row * k + start;
            let scale = f16::from_f32((5 + noise(seed + 41) % 43) as f32 * 0.000_31);
            let d = scale.to_f64();
            let mut codes: Vec<_> = (0..block)
                .map(|index| noise(seed + index + 113) as u8)
                .collect();
            match format {
                WeightFormat::Q4_0 | WeightFormat::Q4_1 | WeightFormat::Q8_0 => {
                    bytes.extend(scale.to_bits().to_le_bytes());
                    if format == WeightFormat::Q8_0 {
                        for code in codes {
                            let value = d * f64::from(code as i8);
                            bytes.push(code);
                            coefficients.push((value, value.abs()));
                        }
                        continue;
                    }
                    let offset = if format == WeightFormat::Q4_0 {
                        -8.0 * d
                    } else {
                        let offset = f16::from_f64(-7.5 * d);
                        bytes.extend(offset.to_bits().to_le_bytes());
                        offset.to_f64()
                    };
                    for code in &mut codes {
                        *code &= 15;
                        let product = d * f64::from(*code);
                        coefficients.push((product + offset, product.abs() + offset.abs()));
                    }
                    for index in 0..16 {
                        bytes.push(codes[index] | (codes[index + 16] << 4));
                    }
                }
                WeightFormat::Q4K | WeightFormat::Q5K => {
                    let minimum = f16::from_f64(d * 0.9375);
                    let minimum = minimum.to_f64();
                    bytes.extend(scale.to_bits().to_le_bytes());
                    bytes.extend(f16::from_f64(minimum).to_bits().to_le_bytes());
                    let scales: Vec<_> = (0..8)
                        .map(|index| (1 + noise(seed + index + 223) % 63) as u8)
                        .collect();
                    let minima: Vec<_> = (0..8)
                        .map(|index| (noise(seed + index + 317) % 64) as u8)
                        .collect();
                    let mut packed = [0u8; 12];
                    for index in 0..8 {
                        if index < 4 {
                            packed[index] = scales[index];
                            packed[index + 4] = minima[index];
                        } else {
                            packed[index + 4] = (scales[index] & 15) | ((minima[index] & 15) << 4);
                            packed[index - 4] |= (scales[index] >> 4) << 6;
                            packed[index] |= (minima[index] >> 4) << 6;
                        }
                    }
                    bytes.extend(packed);
                    let mut high = [0u8; 32];
                    for (index, code) in codes.iter_mut().enumerate() {
                        *code &= if format == WeightFormat::Q5K { 31 } else { 15 };
                        let product = d * f64::from(scales[index / 32]) * f64::from(*code);
                        let offset = minimum * f64::from(minima[index / 32]);
                        coefficients.push((product - offset, product.abs() + offset.abs()));
                        high[index % 32] |= (*code >> 4) << (index / 32);
                    }
                    if format == WeightFormat::Q5K {
                        bytes.extend(high);
                    }
                    for codes in codes.as_chunks::<64>().0 {
                        for index in 0..32 {
                            bytes.push((codes[index] & 15) | ((codes[index + 32] & 15) << 4));
                        }
                    }
                }
                WeightFormat::Q6K => {
                    let scales: Vec<_> = (0..16)
                        .map(|index| noise(seed + index + 433) as i8)
                        .collect();
                    let mut low = [0u8; 128];
                    let mut high = [0u8; 64];
                    for (index, code) in codes.iter_mut().enumerate() {
                        *code &= 63;
                        let value =
                            d * f64::from(scales[index / 16]) * f64::from(i32::from(*code) - 32);
                        coefficients.push((value, value.abs()));
                        let half = index / 128;
                        let quarter = index % 128 / 32;
                        let lane = index % 32;
                        low[half * 64 + quarter % 2 * 32 + lane] |=
                            (*code & 15) << (quarter / 2 * 4);
                        high[half * 32 + lane] |= (*code >> 4) << (quarter * 2);
                    }
                    bytes.extend(low);
                    bytes.extend(high);
                    bytes.extend(scales.iter().map(|&scale| scale as u8));
                    bytes.extend(scale.to_bits().to_le_bytes());
                }
                _ => unreachable!("floating-point and expert formats do not use these blocks"),
            }
        }
    }
    (bytes, coefficients)
}
