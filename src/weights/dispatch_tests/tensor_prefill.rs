//! Tensor-operation dense products (Apple10 GPUs) against FP64 references.
use super::{dense_weights, noise};
use crate::Result;
use crate::{
    gpu::{
        Gpu, dispatch,
        tests::{buffer, read},
    },
    tensor::WeightFormat,
    weights::{compute::Compute, pipelines::Builder},
};
use half::{bf16, f16};
use objc2_metal::{MTLDevice, MTLGPUFamily};

/// Products of half-rounded coefficients with exact BF16 activations, accumulated in FP64.
fn reference(coefficients: &[(f64, f64)], input: &[u16], exact: bool) -> (f64, f64) {
    let mut sum = 0.0;
    let mut magnitude = 0.0;
    for (&(coefficient, _), &input) in coefficients.iter().zip(input) {
        let coefficient = if exact { coefficient } else { f16::from_f64(coefficient).to_f64() };
        let input = bf16::from_bits(input).to_f64();
        sum += coefficient * input;
        magnitude += (coefficient * input).abs();
    }
    (sum, magnitude)
}

fn run(
    gpu: &Gpu,
    compute: &Compute,
    builder: &mut Builder,
    format: WeightFormat,
    (m, n, k): (usize, usize, usize),
    rows: &[u16],
    bias: Option<&[f32]>,
) -> Result<Vec<(f64, f64, f64)>> {
    let symbol = match format {
        WeightFormat::F32 => "matmul_tensor_f32_128",
        _ => "matmul_tensor_128",
    };
    let pipeline = builder.get(symbol, format, 0)?;
    let (weights, coefficients) = dense_weights(format, n, k);
    let w = buffer(gpu, &weights);
    let x = buffer(gpu, rows);
    let y = buffer(gpu, &vec![0x7fc1u16; m * n]);
    let b = buffer(gpu, bias.unwrap_or(&[0.0]));
    let p = [k as u32, n as u32, m as u32, 0, 4, 1, bias.is_some() as u32];
    futures_lite::future::block_on(compute.submit(|e| {
        dispatch(
            e,
            &pipeline,
            &[(&*w, 0), (&*x, 0), (&*y, 0), (&*b, 0), (&*b, 0)],
            bytemuck::cast_slice(&p),
            [m.div_ceil(128), n.div_ceil(64), 1],
            128,
        )
    }))?;
    let mut output = vec![0u16; m * n];
    read(&y, &mut output);
    Ok((0..m * n)
        .map(|i| {
            let (row, column) = (i / n, i % n);
            let (mut expected, magnitude) = reference(
                &coefficients[column * k..(column + 1) * k],
                &rows[row * k..(row + 1) * k],
                format == WeightFormat::F32,
            );
            if let Some(bias) = bias {
                expected += bf16::from_f32(bias[column]).to_f64();
            }
            (bf16::from_bits(output[i]).to_f64(), expected, magnitude)
        })
        .collect())
}

/// Shapes cover token and column tails, several K steps and every dense weight format the
/// tensor path serves, with and without bias.
#[test]
fn tensor_products_match_reference() -> Result<()> {
    let gpu = Gpu::new()?;
    if !gpu.device.supportsFamily(MTLGPUFamily::Apple10) {
        eprintln!("no Apple10 GPU; skipping");
        return Ok(());
    }
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    for format in [
        WeightFormat::F32,
        WeightFormat::Q4_0,
        WeightFormat::Q4_1,
        WeightFormat::Q8_0,
        WeightFormat::Q5K,
        WeightFormat::F16,
        WeightFormat::Bf16,
    ] {
        let k_step = if matches!(format, WeightFormat::Q5K) { 256 } else { 64 };
        for (m, n, k) in [(1, 64, k_step * 5), (129, 65, k_step * 3), (300, 31, k_step * 16)] {
            let rows: Vec<_> = (0..m * k)
                .map(|i| bf16::from_f32((noise(i + 7411) % 251) as f32 / 31.0 - 4.0).to_bits())
                .collect();
            let bias: Vec<_> = (0..n).map(|i| (i as f32 - 16.0) * 0.031337).collect();
            for bias in [None, Some(&bias[..])] {
                for (i, (actual, expected, magnitude)) in
                    run(&gpu, &compute, &mut builder, format, (m, n, k), &rows, bias)?
                        .into_iter()
                        .enumerate()
                {
                    // BF16 output rounding plus FP32 accumulation over 2k operations.
                    let bound = 2f64.powi(-8) * expected.abs()
                        + (2 * k + 8) as f64 * 2f64.powi(-23) * magnitude;
                    assert!(
                        actual.is_finite() && (actual - expected).abs() <= bound,
                        "{format:?} m={m} n={n} k={k} output {i}: {actual} != {expected} (bound {bound})"
                    );
                }
            }
        }
    }
    Ok(())
}

/// BF16 activations far outside the half range must reach the products unscaled.
#[test]
fn tensor_products_keep_bf16_range() -> Result<()> {
    let gpu = Gpu::new()?;
    if !gpu.device.supportsFamily(MTLGPUFamily::Apple10) {
        eprintln!("no Apple10 GPU; skipping");
        return Ok(());
    }
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let (m, n, k) = (4, 64, 64);
    let mut rows = vec![0u16; m * k];
    for (row, value) in [2f32.powi(20), -2f32.powi(40), 2f32.powi(-30), 1.0]
        .into_iter()
        .enumerate()
    {
        for column in 0..k {
            rows[row * k + column] = bf16::from_f32(value).to_bits();
        }
    }
    for (i, (actual, expected, magnitude)) in run(
        &gpu,
        &compute,
        &mut builder,
        WeightFormat::Q4_0,
        (m, n, k),
        &rows,
        None,
    )?
    .into_iter()
    .enumerate()
    {
        let bound = 2f64.powi(-8) * expected.abs() + 2f64.powi(-16) * magnitude;
        assert!(
            actual.is_finite() && (actual - expected).abs() <= bound,
            "row {}: {actual} != {expected} (bound {bound})",
            i / n
        );
    }
    Ok(())
}


/// Every token width and split count against FP64 products, finishing splits through
/// matmul_split_sum; raw FP32 partials also bound the relaxed-precision accumulation.
#[test]
fn tensor_tiles_and_splits_match_reference() -> Result<()> {
    let gpu = Gpu::new()?;
    if !gpu.device.supportsFamily(MTLGPUFamily::Apple10) {
        eprintln!("no Apple10 GPU; skipping");
        return Ok(());
    }
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let sum = builder.get("matmul_split_sum", WeightFormat::F32, 0)?;
    let (n, k) = (96usize, 5120usize);
    let format = WeightFormat::Q4_0;
    let (weights, coefficients) = dense_weights(format, n, k);
    let w = buffer(&gpu, &weights);
    for (symbol, nt, nr) in [
        ("matmul_tensor_32", 32usize, 64usize),
        ("matmul_tensor_64", 64, 64),
        ("matmul_tensor_128", 128, 64),
    ] {
        let pipeline = builder.get(symbol, format, 0)?;
        for m in [1usize, nt - 1, nt + 3] {
            let rows: Vec<_> = (0..m * k)
                .map(|i| bf16::from_f32((noise(i + 99) % 251) as f32 / 31.0 - 4.0).to_bits())
                .collect();
            let x = buffer(&gpu, &rows);
            let bias: Vec<_> = (0..n).map(|i| (i as f32 - 40.0) * 0.0173).collect();
            let b = buffer(&gpu, &bias);
            let expected: Vec<(f64, f64)> = (0..m * n)
                .map(|i| {
                    let (row, column) = (i / n, i % n);
                    reference(
                        &coefficients[column * k..(column + 1) * k],
                        &rows[row * k..(row + 1) * k],
                        false,
                    )
                })
                .collect();
            for splits in [1usize, 2, 4, 16] {
                let y = buffer(&gpu, &vec![0x7fc1u16; m * n]);
                let partials = buffer(&gpu, &vec![f32::NAN; splits * m * n]);
                let p = [k as u32, n as u32, m as u32, 0, 4, 1, 1];
                futures_lite::future::block_on(compute.submit(|e| {
                    let target = if splits > 1 { &partials } else { &y };
                    dispatch(
                        e,
                        &pipeline,
                        &[(&*w, 0), (&*x, 0), (&**target, 0), (&*b, 0), (&*b, 0)],
                        bytemuck::cast_slice(&p),
                        [m.div_ceil(nt), n.div_ceil(nr), splits],
                        128,
                    );
                    if splits > 1 {
                        let finish = [m as u32, n as u32, splits as u32, 1, 0];
                        dispatch(
                            e,
                            &sum,
                            &[(&*partials, 0), (&*y, 0), (&*b, 0), (&*b, 0)],
                            bytemuck::cast_slice(&finish),
                            [(m * n).div_ceil(256), 1, 1],
                            256,
                        );
                    }
                }))?;
                let mut output = vec![0u16; m * n];
                read(&y, &mut output);
                if splits > 1 {
                    // Raw FP32 sums: relaxed accumulation within a small multiple of FP32 rounding.
                    let mut raw = vec![0f32; splits * m * n];
                    read(&partials, &mut raw);
                    for i in 0..m * n {
                        let total: f64 = (0..splits).map(|z| f64::from(raw[z * m * n + i])).sum();
                        let (exact, magnitude) = expected[i];
                        assert!(
                            (total - exact).abs() <= 2f64.powi(-16) * magnitude,
                            "{symbol} m={m} splits={splits} raw sum {i}: {total} != {exact} (magnitude {magnitude})"
                        );
                    }
                }
                for i in 0..m * n {
                    let actual = bf16::from_bits(output[i]).to_f64();
                    let exact = expected[i].0 + bf16::from_f32(bias[i % n]).to_f64();
                    let bound = 2f64.powi(-8) * exact.abs() + 2f64.powi(-16) * expected[i].1;
                    assert!(
                        actual.is_finite() && (actual - exact).abs() <= bound,
                        "{symbol} m={m} splits={splits} output {i}: {actual} != {exact} (bound {bound})"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Qwen3.8's alpha/beta shape through FP32-staged tiles with the sixteen-way split production
/// selects, finished by matmul_split_sum: tiny coefficients survive exactly.
#[test]
fn f32_tiles_split_sixteen_ways_exactly() -> Result<()> {
    let gpu = Gpu::new()?;
    if !gpu.device.supportsFamily(MTLGPUFamily::Apple10) {
        eprintln!("no Apple10 GPU; skipping");
        return Ok(());
    }
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let tile = builder.get("matmul_tensor_f32_128", WeightFormat::F32, 0)?;
    let sum = builder.get("matmul_split_sum", WeightFormat::F32, 0)?;
    let (m, n, k, splits) = (512usize, 48usize, 5120usize, 16usize);
    let coefficient = 2f32.powi(-30);
    let mut weights = vec![0f32; n * k];
    for column in 0..n {
        weights[column * k + (column * 97) % k] = coefficient;
        weights[column * k + (column * 97 + k / 2) % k] = -0.5 * coefficient;
    }
    let rows: Vec<u16> = (0..m * k)
        .map(|i| bf16::from_f32((noise(i + 3) % 61) as f32 / 32.0 - 0.875).to_bits())
        .collect();
    let (w, x) = (buffer(&gpu, &weights), buffer(&gpu, &rows));
    let partials = buffer(&gpu, &vec![f32::NAN; splits * m * n]);
    let y = buffer(&gpu, &vec![0x7fc1u16; m * n]);
    let none = buffer(&gpu, &[0f32]);
    let p = [k as u32, n as u32, m as u32, 0, 4, 1, 0];
    futures_lite::future::block_on(compute.submit(|e| {
        dispatch(
            e,
            &tile,
            &[(&*w, 0), (&*x, 0), (&*partials, 0), (&*none, 0), (&*none, 0)],
            bytemuck::cast_slice(&p),
            [m / 128, 1, splits],
            128,
        );
        let finish = [m as u32, n as u32, splits as u32, 0, 0];
        dispatch(
            e,
            &sum,
            &[(&*partials, 0), (&*y, 0), (&*none, 0), (&*none, 0)],
            bytemuck::cast_slice(&finish),
            [(m * n).div_ceil(256), 1, 1],
            256,
        );
    }))?;
    let mut output = vec![0u16; m * n];
    read(&y, &mut output);
    for row in 0..m {
        for column in 0..n {
            let first = bf16::from_bits(rows[row * k + (column * 97) % k]).to_f64();
            let second = bf16::from_bits(rows[row * k + (column * 97 + k / 2) % k]).to_f64();
            let expected = bf16::from_f64(f64::from(coefficient) * (first - 0.5 * second)).to_bits();
            assert_eq!(output[row * n + column], expected, "row {row} column {column}");
        }
    }
    Ok(())
}
