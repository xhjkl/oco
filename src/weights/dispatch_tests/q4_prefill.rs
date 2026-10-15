use super::{dense_weights, half_operand, noise};
use crate::{
    gpu::{
        Gpu, Pipeline, dispatch,
        tests::{buffer, read},
    },
    tensor::WeightFormat,
    weights::{compute::Compute, pipelines::Builder},
};
use anyhow::Result;
use half::{bf16, f16};

/// Check scaled half operands, bias, row-gain offsets and tile tails against FP64 products.
#[test]
fn q4_scaled_prefill_preserves_products_and_boundaries() -> Result<()> {
    let gpu = Gpu::new()?;
    let mut builder = Builder::new(&gpu)?;
    let prepare = builder.get("q4_input_scales", WeightFormat::F32, 0)?;
    for format in [WeightFormat::Q4_0, WeightFormat::Q4_1] {
        // Pairing each live tile with one boundary, rather than multiplying shape grids.
        for (variant, m, n, k) in [
            (0, 65, 65, 160),
            (1, 1, 31, 32),
            (2, 9, 65, 160),
            (3, 21, 31, 512),
            (4, 33, 65, 160),
        ] {
            let pipeline = builder.get("matmul_q4_half", format, variant)?;
            let (weights, coefficients) = dense_weights(format, n, k);
            let rows: Vec<_> = (0..3 * k)
                .map(|i| bf16::from_f32((noise(i + 7411) % 251) as f32 / 127.0 - 1.0).to_bits())
                .collect();
            let bias: Vec<_> = (0..n).map(|i| (i as f32 - 16.0) * 0.031337).collect();
            check_scaled_input(
                &gpu,
                &prepare,
                &pipeline,
                format,
                variant as usize,
                &weights,
                &coefficients,
                &rows,
                Some(&bias),
                m,
                n,
                k,
            )?;
        }
        let pipeline = builder.get("matmul_q4_half", format, 1)?;
        let (weights, coefficients, rows) = scaled_input_rounding_fixture(format);
        check_scaled_input(
            &gpu,
            &prepare,
            &pipeline,
            format,
            1,
            &weights,
            &coefficients,
            &rows,
            None,
            9,
            1,
            32,
        )?;
        let (mut weights, mut coefficients, _) = scaled_input_rounding_fixture(format);
        let header = if format == WeightFormat::Q4_0 { 2 } else { 4 };
        let zero = if format == WeightFormat::Q4_0 { 8 } else { 0 };
        weights[header] = (weights[header] & 0xf0) | zero;
        coefficients[0] = (0.0, 0.0);
        let mut rows = vec![0u16; 6 * 32];
        for (row, value) in [
            2.0f32.powi(64),
            -2.0f32.powi(64),
            2.0f32.powi(-80),
            -2.0f32.powi(-80),
        ]
        .into_iter()
        .enumerate()
        {
            rows[row * 32 + 1] = bf16::from_f32(value).to_bits();
        }
        // Zero rows and an unweighted large operand expose gain and tiny-value behavior.
        rows[5 * 32] = bf16::from_f32(2.0f32.powi(64)).to_bits();
        rows[5 * 32 + 1] = bf16::from_f32(2.0f32.powi(-80)).to_bits();
        check_scaled_input(
            &gpu,
            &prepare,
            &pipeline,
            format,
            1,
            &weights,
            &coefficients,
            &rows,
            None,
            129,
            1,
            32,
        )?;
    }
    Ok(())
}

fn input_row_scale(row: &[u16]) -> [f32; 2] {
    let peak = row
        .iter()
        .map(|&bits| bf16::from_bits(bits).to_f32().abs())
        .fold(0.0f32, f32::max);
    assert!(peak.is_finite());
    if peak == 0.0 {
        return [1.0, 1.0];
    }
    let mut fraction = f64::from(peak);
    let mut exponent = 0i32;
    while fraction >= 2.0 {
        fraction *= 0.5;
        exponent += 1;
    }
    while fraction < 1.0 {
        fraction *= 2.0;
        exponent -= 1;
    }
    let shift = (14 - exponent).clamp(-120, 120);
    [2.0f32.powi(shift), 2.0f32.powi(-shift)]
}

/// Distinguish coefficient rounding from cancellation in the declared half operands.
fn scaled_input_rounding_fixture(format: WeightFormat) -> (Vec<u8>, Vec<(f64, f64)>, Vec<u16>) {
    let scale = 1.0 + 1.0 / 1024.0;
    let mut codes = [0u8; 32];
    codes[0] = 7;
    codes[1] = 1;
    let coefficients = codes
        .iter()
        .map(|&code| {
            let v = f64::from(code) * scale;
            (v, v.abs())
        })
        .collect();
    let mut weights = f16::from_f64(scale).to_bits().to_le_bytes().to_vec();
    if format == WeightFormat::Q4_1 {
        weights.extend(0u16.to_le_bytes());
    }
    let zero = if format == WeightFormat::Q4_0 { 8 } else { 0 };
    for index in 0..16 {
        weights.push((codes[index] + zero) | ((codes[index + 16] + zero) << 4));
    }
    let mut row = vec![0u16; 32];
    row[0] = bf16::from_f32(1.0).to_bits();
    row[1] = bf16::from_f32(-7.0).to_bits();
    (weights, coefficients, row)
}

/// Apply row scaling and both half conversions before the independent FP64 product.
fn scaled_input_reference(
    coefficients: &[(f64, f64)],
    rows: &[u16],
    k: usize,
    bias: Option<&[f32]>,
) -> Vec<(f64, f64)> {
    let coefficients: Vec<_> = coefficients
        .iter()
        .map(|&(value, _)| half_operand(value))
        .collect();
    let mut reference = Vec::new();
    for row in rows.chunks_exact(k) {
        let [gain, inverse_gain] = input_row_scale(row);
        for (column, coefficients) in coefficients.chunks_exact(k).enumerate() {
            let mut sum = 0.0;
            let mut magnitude = 0.0;
            for (&input, &coefficient) in row.iter().zip(coefficients) {
                let input = bf16::from_bits(input).to_f32() * gain;
                let input = f16::from_f32(input).to_f64() * f64::from(inverse_gain);
                let product = input * coefficient;
                sum += product;
                magnitude += product.abs();
            }
            let bias = bias.map_or(0.0, |bias| bf16::from_f32(bias[column]).to_f64());
            reference.push((sum + bias, magnitude + bias.abs()));
        }
    }
    reference
}

#[allow(clippy::too_many_arguments)]
fn check_scaled_input(
    gpu: &Gpu,
    prepare: &Pipeline,
    pipeline: &Pipeline,
    format: WeightFormat,
    row_tiles: usize,
    weights: &[u8],
    coefficients: &[(f64, f64)],
    rows: &[u16],
    bias: Option<&[f32]>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    assert_eq!(coefficients.len(), n * k);
    assert!(!rows.is_empty() && rows.len().is_multiple_of(k));
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n);
    }
    let reference = scaled_input_reference(coefficients, rows, k, bias);
    let compute = Compute::new(gpu, None);
    const WEIGHT_OFFSET: usize = 32;
    let mut bytes = vec![0xa5; WEIGHT_OFFSET];
    bytes.extend_from_slice(weights);
    bytes.extend([0xa5; WEIGHT_OFFSET]);
    let weight = buffer(gpu, &bytes);
    const GUARD: usize = 32;
    const MARKER: u16 = 0x7fc1;
    let mut input = vec![MARKER; GUARD];
    input.extend(rows.iter().copied().cycle().take(m * k));
    input.extend([MARKER; GUARD]);
    let input = buffer(gpu, &input);
    let mut bias_values = vec![f32::NAN; 8];
    bias_values.extend_from_slice(bias.unwrap_or(&[f32::NAN]));
    bias_values.extend([f32::NAN; 8]);
    let bias_buffer = buffer(gpu, &bias_values);
    let seed_scales: Vec<_> = rows.chunks_exact(k).map(input_row_scale).collect();
    let expected_scales: Vec<_> = seed_scales
        .iter()
        .copied()
        .cycle()
        .take(m)
        .flatten()
        .collect();
    const SCALE_GUARD: usize = 8;
    const SCALE_MARKER: u32 = 0x7fc0_01a5;
    let mut scales = vec![SCALE_MARKER; SCALE_GUARD + 2 * m + SCALE_GUARD];
    let scale_buffer = buffer(gpu, &scales);
    let mut output = vec![MARKER; GUARD + m * n + GUARD];
    let output_buffer = buffer(gpu, &output);
    let params = [
        k as u32,
        n as u32,
        m as u32,
        0,
        4,
        1,
        u32::from(bias.is_some()),
    ];
    let tile_m = if row_tiles == 0 { 64 } else { 8 * row_tiles };
    let submission = compute.submit(|encoder| {
        dispatch(
            encoder,
            prepare,
            &[(&input, GUARD * 2), (&scale_buffer, SCALE_GUARD * 4)],
            bytemuck::cast_slice(&params),
            [m, 1, 1],
            256,
        );
        dispatch(
            encoder,
            pipeline,
            &[
                (&weight, WEIGHT_OFFSET),
                (&input, GUARD * 2),
                (&output_buffer, GUARD * 2),
                (&bias_buffer, 8 * 4),
                (&scale_buffer, SCALE_GUARD * 4),
            ],
            bytemuck::cast_slice(&params),
            [m.div_ceil(tile_m), n.div_ceil(64), 1],
            128,
        );
    });
    futures_lite::future::block_on(submission)?;
    read(&scale_buffer, &mut scales);
    assert!(
        scales[..SCALE_GUARD]
            .iter()
            .all(|&bits| bits == SCALE_MARKER)
    );
    assert!(
        scales[SCALE_GUARD + 2 * m..]
            .iter()
            .all(|&bits| bits == SCALE_MARKER)
    );
    for (index, (&actual, &expected)) in scales[SCALE_GUARD..SCALE_GUARD + 2 * m]
        .iter()
        .zip(&expected_scales)
        .enumerate()
    {
        assert_eq!(
            actual,
            expected.to_bits(),
            "{format:?} K={k} M={m} scale word {index}"
        );
    }
    read(&output_buffer, &mut output);
    assert!(output[..GUARD].iter().all(|&value| value == MARKER));
    assert!(output[GUARD + m * n..].iter().all(|&value| value == MARKER));
    // Bounding FP32 accumulation and BF16 storage; both FP16 operand roundings are explicit in the oracle.
    let operations = (2 * k + 8) as f64;
    let roundoff = 2.0f64.powi(-24);
    let gamma = operations * roundoff / (1.0 - operations * roundoff);
    let storage_roundoff = 2.0f64.powi(-8);
    for (index, &actual) in output[GUARD..GUARD + m * n].iter().enumerate() {
        let (expected, magnitude) = reference[index % reference.len()];
        let actual = bf16::from_bits(actual).to_f64();
        let bound =
            (1.0 + storage_roundoff) * gamma * magnitude + storage_roundoff * expected.abs();
        assert!(
            actual.is_finite() && (actual - expected).abs() <= bound,
            "scaled-input {format:?} K={k} N={n} M={m} row_tiles={row_tiles} index={index}: actual={actual} expected={expected} bound={bound}"
        );
    }
    if m == 129 && n == 1 && k == 32 {
        // Reusing already prepared gains for a selected row and a tail. The six-row
        // seed has different gain exponents, so an unshifted scales binding is observable.
        for (first, count) in [(68, 1), (64, 33), (64, 63), (128, 1)] {
            let mut output = vec![MARKER; GUARD + count * n + GUARD];
            let output_buffer = buffer(gpu, &output);
            let mut params = params;
            params[2] = count as u32;
            let submission = compute.submit(|encoder| {
                dispatch(
                    encoder,
                    pipeline,
                    &[
                        (&weight, WEIGHT_OFFSET),
                        (&input, GUARD * 2 + first * k * 2),
                        (&output_buffer, GUARD * 2),
                        (&bias_buffer, 8 * 4),
                        (&scale_buffer, SCALE_GUARD * 4 + first * 8),
                    ],
                    bytemuck::cast_slice(&params),
                    [count.div_ceil(tile_m), n.div_ceil(64), 1],
                    128,
                );
            });
            futures_lite::future::block_on(submission)?;
            read(&output_buffer, &mut output);
            assert!(output[..GUARD].iter().all(|&value| value == MARKER));
            assert!(
                output[GUARD + count * n..]
                    .iter()
                    .all(|&value| value == MARKER)
            );
            for (index, &actual) in output[GUARD..GUARD + count * n].iter().enumerate() {
                let (expected, magnitude) = reference[(first * n + index) % reference.len()];
                let actual = bf16::from_bits(actual).to_f64();
                let bound = (1.0 + storage_roundoff) * gamma * magnitude
                    + storage_roundoff * expected.abs();
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= bound,
                    "scaled-input offset {format:?} row_tiles={row_tiles} first={first} count={count} index={index}: actual={actual} expected={expected} bound={bound}"
                );
            }
        }
    }
    Ok(())
}
