//! Kernel timings at checkpoint shapes, for choosing dispatch thresholds. Ignored by default:
//! `cargo test --release --lib dense_bench -- --ignored --nocapture --test-threads=1`.
use crate::Result;
use crate::gpu::{Buffer, Gpu, Pipeline, dispatch, tests::buffer};
use crate::tensor::WeightFormat;
use crate::weights::compute::AttentionParams;
use crate::weights::{compute::Compute, pipelines::Builder};
use std::time::Instant;

/// Best per-dispatch time of `iterations` ordered launches in one submission.
fn time(
    compute: &Compute,
    iterations: usize,
    launch: impl Fn(&crate::gpu::Commands),
) -> Result<f64> {
    futures_lite::future::block_on(compute.submit(&launch))?;
    let mut best = f64::MAX;
    for _ in 0..3 {
        let started = Instant::now();
        futures_lite::future::block_on(compute.submit(|e| {
            for _ in 0..iterations {
                launch(e);
            }
        }))?;
        best = best.min(started.elapsed().as_secs_f64() / iterations as f64);
    }
    Ok(best)
}

/// Q4_0 weights with a small positive scale in every block header.
fn q4_weights(gpu: &Gpu, n: usize, k: usize) -> Buffer {
    let bytes: Vec<u8> = (0..n * k / 32 * 18)
        .map(|i| match i % 18 {
            0 => 0x00,
            1 => 0x2c,
            _ => ((i * 2654435761usize) >> 7) as u8,
        })
        .collect();
    buffer(gpu, &bytes)
}

/// Qwen3.8 projection products for 1 to 128 rows on both paths: the per-row vector kernel and
/// the tiled products with their gain scan, which `DensePlan` chooses between.
#[test]
#[ignore]
fn q4_batch_paths() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let format = WeightFormat::Q4_0;
    let vector = builder.get("matvec_q4", format, 0)?;
    let scales = builder.get("q4_input_scales", WeightFormat::F32, 0)?;
    let sum = builder.get("matmul_split_sum", WeightFormat::F32, 0)?;
    let tiles: Vec<Pipeline> = (0..5)
        .map(|variant| builder.get("matmul_q4_half", format, variant))
        .collect::<Result<_>>()?;
    for (k, n) in [(5120usize, 17408usize), (17408, 5120), (5120, 10240)] {
        let w = q4_weights(&gpu, n, k);
        let x = buffer(&gpu, &vec![0x3f80u16; 512 * k]);
        let y = buffer(&gpu, &vec![0u16; 512 * n]);
        let gains = buffer(&gpu, &vec![1.0f32; 2 * 512]);
        let partials = buffer(&gpu, &vec![0f32; 16 * 32 * n]);
        let bytes = (n * k / 32 * 18) as f64;
        eprintln!("{k} -> {n} ({:.1} MB)", bytes / 1e6);
        for m in [1usize, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512] {
            let p = [k as u32, n as u32, m as u32, 0, 4, 1, 0];
            let params = bytemuck::cast_slice(&p);
            let buffers = [(&*w, 0), (&*x, 0), (&*y, 0), (&*y, 0), (&*gains, 0)];
            let threads = Builder::vector_threads(n);
            let groups = n.div_ceil(threads / 8);
            let iterations = 20;
            let per_row = time(&compute, iterations, |e| {
                dispatch(e, &vector, &buffers, params, [groups, m, 1], threads)
            })?;
            let tiled = time(&compute, iterations, |e| {
                dispatch(
                    e,
                    &scales,
                    &[(&*x, 0), (&*gains, 0)],
                    params,
                    [m, 1, 1],
                    256,
                );
                let full = m / 64 * 64;
                if full > 0 {
                    let p = [k as u32, n as u32, full as u32, 0, 4, 1, 0];
                    let groups = [full / 64, n.div_ceil(64), 1];
                    dispatch(
                        e,
                        &tiles[0],
                        &buffers,
                        bytemuck::cast_slice(&p),
                        groups,
                        128,
                    );
                }
                let tail = m - full;
                if tail > 0 {
                    let variant = tail.min(32).div_ceil(8);
                    let p = [k as u32, n as u32, tail as u32, 0, 4, 1, 0];
                    let groups = [tail.div_ceil(variant * 8), n.div_ceil(64), 1];
                    dispatch(
                        e,
                        &tiles[variant],
                        &buffers,
                        bytemuck::cast_slice(&p),
                        groups,
                        128,
                    );
                }
            })?;
            let mut sweep = String::new();
            for splits in [2usize, 4, 8, 16] {
                if m > 32 || !(k / 32).is_multiple_of(splits) {
                    continue;
                }
                let seconds = time(&compute, iterations, |e| {
                    dispatch(
                        e,
                        &scales,
                        &[(&*x, 0), (&*gains, 0)],
                        params,
                        [m, 1, 1],
                        256,
                    );
                    let variant = m.div_ceil(8);
                    let product = [(&*w, 0), (&*x, 0), (&*partials, 0), (&*y, 0), (&*gains, 0)];
                    let groups = [1, n.div_ceil(64), splits];
                    dispatch(e, &tiles[variant], &product, params, groups, 128);
                    let finish = [m as u32, n as u32, splits as u32, 0, 1];
                    let buffers = [(&*partials, 0), (&*y, 0), (&*y, 0), (&*gains, 0)];
                    let groups = [(m * n).div_ceil(256), 1, 1];
                    dispatch(
                        e,
                        &sum,
                        &buffers,
                        bytemuck::cast_slice(&finish),
                        groups,
                        256,
                    );
                })?;
                sweep.push_str(&format!("  split{splits} {:7.1} us", seconds * 1e6));
            }
            eprintln!(
                "  m {m:>3}: per-row {:8.1} us  tiled {:8.1} us  ({:5.1} us per row){sweep}",
                per_row * 1e6,
                tiled * 1e6,
                tiled * 1e6 / m as f64,
            );
        }
    }
    Ok(())
}

/// Single-query attention at Qwen3.8's (24 query heads over 4 KV heads of 256) and GPT-OSS's
/// (64 over 8 of 64) geometry, from short to long agent contexts.
#[test]
#[ignore]
fn decode_attention_latency() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    for (heads, kv_heads, dim) in [(24usize, 4usize, 256usize), (64, 8, 64)] {
        let symbol = if dim == 256 {
            "attention_vector_256"
        } else {
            "attention_vector_64"
        };
        let vector = builder.get(symbol, WeightFormat::F32, 0)?;
        let q = buffer(
            &gpu,
            &vec![half::bf16::from_f32(0.01).to_bits(); heads * dim],
        );
        let out = buffer(&gpu, &vec![0u16; heads * dim]);
        for keys in [1024usize, 2048, 4096, 8192, 16384, 32768] {
            let kv = buffer(
                &gpu,
                &vec![half::f16::from_f32(0.02).to_bits(); keys * kv_heads * dim],
            );
            let p = AttentionParams {
                heads: heads as u32,
                kv_heads: kv_heads as u32,
                dim: dim as u32,
                pos: keys as u32 - 1,
                window: 0,
                sinks: 0,
                scale: 1.0 / (dim as f32).sqrt(),
                m: 1,
                keys: 0,
            };
            let buffers = [(&*q, 0), (&*kv, 0), (&*kv, 0), (&*out, 0), (&*out, 0)];
            let seconds = time(&compute, 100, |e| {
                dispatch(
                    e,
                    &vector,
                    &buffers,
                    bytemuck::bytes_of(&p),
                    [heads, 1, 1],
                    256,
                )
            })?;
            let bytes = (2 * keys * kv_heads * dim * 2) as f64;
            eprintln!(
                "{heads}/{kv_heads}x{dim} {keys:>5} keys: {:7.1} us ({:5.1} GB/s of KV)",
                seconds * 1e6,
                bytes / seconds / 1e9
            );
        }
    }
    Ok(())
}

/// Causal prompt attention at both checkpoints' head geometries (and GPT-OSS's 128-token
/// window) over agent-length histories: one vector threadgroup per head and row against the
/// sixteen-query tiles `AttentionPlan` keeps for windowed full chunks.
#[test]
#[ignore]
fn prefill_attention() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let wide = builder.get("attention_wide16", WeightFormat::F32, 0)?;
    for (heads, kv_heads, dim, window) in [
        (24usize, 4usize, 256usize, 0usize),
        (64, 8, 64, 0),
        (64, 8, 64, 128),
    ] {
        let symbol = if dim == 256 {
            "attention_vector_256"
        } else {
            "attention_vector_64"
        };
        let vector = builder.get(symbol, WeightFormat::F32, 0)?;
        eprintln!("{heads}/{kv_heads}x{dim}, window {window}");
        for history in [0usize, 512, 2048, 4096, 8192] {
            let kv = buffer(
                &gpu,
                &vec![half::f16::from_f32(0.02).to_bits(); (history + 512) * kv_heads * dim],
            );
            let q = buffer(
                &gpu,
                &vec![half::bf16::from_f32(0.01).to_bits(); 512 * heads * dim],
            );
            let out = buffer(&gpu, &vec![0u16; 512 * heads * dim]);
            for m in [8usize, 32, 128, 512] {
                let p = AttentionParams {
                    heads: heads as u32,
                    kv_heads: kv_heads as u32,
                    dim: dim as u32,
                    pos: history as u32,
                    window: window as u32,
                    sinks: 0,
                    scale: 1.0 / (dim as f32).sqrt(),
                    m: m as u32,
                    keys: 0,
                };
                let buffers = [(&*q, 0), (&*kv, 0), (&*kv, 0), (&*out, 0), (&*out, 0)];
                let params = bytemuck::bytes_of(&p);
                let rows = time(&compute, 10, |e| {
                    dispatch(e, &vector, &buffers, params, [heads, m, 1], 256)
                })?;
                let sixteen = if m == 512 {
                    time(&compute, 10, |e| {
                        dispatch(e, &wide, &buffers, params, [heads, m / 16, 1], 128)
                    })?
                } else {
                    f64::NAN
                };
                eprintln!(
                    "  history {history:>5}, {m:>3} rows: vector {:8.1} us  wide16 {:8.1} us",
                    rows * 1e6,
                    sixteen * 1e6
                );
            }
        }
    }
    Ok(())
}
