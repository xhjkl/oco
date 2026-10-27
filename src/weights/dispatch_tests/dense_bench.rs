//! Bandwidth of the dense single-row kernels at Qwen3-ASR's Q8_0 shapes. Ignored by default:
//! `cargo test --release --lib dense_matvec_bandwidth -- --ignored --nocapture`.
use crate::gpu::{Gpu, dispatch, tests::buffer};
use crate::tensor::WeightFormat;
use crate::weights::compute::AttentionParams;
use crate::weights::{compute::Compute, pipelines::Builder};
use anyhow::Result;
use std::time::Instant;

#[test]
#[ignore]
fn dense_matvec_bandwidth() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let iterations = 200;
    {
        let add = builder.get("add", WeightFormat::F32, 0)?;
        let a = buffer(&gpu, &[0u16; 256]);
        let n = 1u32;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let started = Instant::now();
            futures_lite::future::block_on(compute.submit(|e| {
                for _ in 0..iterations {
                    dispatch(
                        e,
                        &add,
                        &[(&a, 0), (&a, 0)],
                        bytemuck::bytes_of(&n),
                        [1, 1, 1],
                        256,
                    );
                }
            }))?;
            best = best.min(started.elapsed().as_secs_f64() / iterations as f64);
        }
        eprintln!("dispatch floor: {:.1} us", best * 1e6);
    }
    for (k, n) in [
        (2048usize, 2048usize),
        (2048, 1024),
        (2048, 6144),
        (6144, 2048),
        (2048, 151936),
    ] {
        let bytes = n * k / 32 * 34;
        let w: Vec<u8> = (0..bytes)
            .map(|i| {
                if i % 34 < 2 {
                    if i % 34 == 0 { 0x00 } else { 0x2c }
                } else {
                    ((i * 2654435761usize) >> 7) as u8
                }
            })
            .collect();
        let w = buffer(&gpu, &w);
        let x: Vec<u16> = (0..k)
            .map(|i| half::bf16::from_f32(((i * 7) % 97) as f32 / 48.0 - 1.0).to_bits())
            .collect();
        let x = buffer(&gpu, &x);
        let y = buffer(&gpu, &vec![0u16; n * 2]);
        let vector = builder.vector_with(WeightFormat::Q8_0, n, n > 100_000)?;
        let p = [k as u32, n as u32, 1, 0, 4, 1, 0];
        let run = |threads: usize, groups: usize| -> Result<f64> {
            futures_lite::future::block_on(compute.submit(|e| {
                dispatch(
                    e,
                    &vector.pipeline,
                    &[(&w, 0), (&x, 0), (&y, 0), (&y, 0), (&y, 0)],
                    bytemuck::cast_slice(&p),
                    [groups, 1, 1],
                    threads,
                );
            }))?;
            let mut best = f64::MAX;
            for _ in 0..3 {
                let started = Instant::now();
                futures_lite::future::block_on(compute.submit(|e| {
                    for _ in 0..iterations {
                        dispatch(
                            e,
                            &vector.pipeline,
                            &[(&w, 0), (&x, 0), (&y, 0), (&y, 0), (&y, 0)],
                            bytemuck::cast_slice(&p),
                            [groups, 1, 1],
                            threads,
                        );
                    }
                }))?;
                best = best.min(started.elapsed().as_secs_f64() / iterations as f64);
            }
            Ok(best)
        };
        let mut line = format!("{k:>5} -> {n:>6} ({:6.1} MB):", bytes as f64 / 1e6);
        for threads in [64usize, 256] {
            let groups = n.div_ceil(threads / 8);
            let seconds = run(threads, groups)?;
            line.push_str(&format!(
                "  {threads:>3} thr {:7.1} us {:6.1} GB/s",
                seconds * 1e6,
                bytes as f64 / seconds / 1e9
            ));
        }
        let split = builder.get("matvec_q8_split", WeightFormat::Q8_0, 0)?;
        let run_split = || -> Result<f64> {
            futures_lite::future::block_on(compute.submit(|e| {
                dispatch(
                    e,
                    &split,
                    &[(&w, 0), (&x, 0), (&y, 0), (&y, 0), (&y, 0)],
                    bytemuck::cast_slice(&p),
                    [n.div_ceil(2), 1, 1],
                    128,
                );
            }))?;
            let mut best = f64::MAX;
            for _ in 0..3 {
                let started = Instant::now();
                futures_lite::future::block_on(compute.submit(|e| {
                    for _ in 0..iterations {
                        dispatch(
                            e,
                            &split,
                            &[(&w, 0), (&x, 0), (&y, 0), (&y, 0), (&y, 0)],
                            bytemuck::cast_slice(&p),
                            [n.div_ceil(2), 1, 1],
                            128,
                        );
                    }
                }))?;
                best = best.min(started.elapsed().as_secs_f64() / iterations as f64);
            }
            Ok(best)
        };
        let seconds = run_split()?;
        line.push_str(&format!(
            "  split {:7.1} us {:6.1} GB/s",
            seconds * 1e6,
            bytes as f64 / seconds / 1e9
        ));
        eprintln!("{line}");
    }
    Ok(())
}

/// Single-query attention kernels at Qwen3-ASR's head geometry over growing histories.
/// `cargo test --release --lib decode_attention_latency -- --ignored --nocapture`.
#[test]
#[ignore]
fn decode_attention_latency() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let iterations = 200;
    for (heads, kv_heads, dim, keys) in [
        (16usize, 8usize, 128usize, 64usize),
        (16, 8, 128, 256),
        (16, 8, 128, 1024),
        (16, 8, 128, 4096),
        (32, 4, 256, 256),
        (32, 4, 256, 1024),
        (32, 4, 256, 4096),
    ] {
        let q = buffer(
            &gpu,
            &vec![half::bf16::from_f32(0.01).to_bits(); heads * dim],
        );
        let kv = buffer(
            &gpu,
            &vec![half::f16::from_f32(0.02).to_bits(); keys * kv_heads * dim],
        );
        let out = buffer(&gpu, &vec![0u16; heads * dim]);
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
        let mut line = format!("D{dim} {keys:>5} keys:");
        let vector: &str = if dim == 128 {
            "attention_vector_128"
        } else {
            "attention_vector_256"
        };
        for (symbol, threads) in [("attention", 1024usize), (vector, 256)] {
            let pipeline = builder.get(symbol, WeightFormat::F32, 0)?;
            let mut best = f64::MAX;
            for _ in 0..3 {
                let started = Instant::now();
                futures_lite::future::block_on(compute.submit(|e| {
                    for _ in 0..iterations {
                        dispatch(
                            e,
                            &pipeline,
                            &[(&q, 0), (&kv, 0), (&kv, 0), (&out, 0), (&out, 0)],
                            bytemuck::bytes_of(&p),
                            [heads, 1, 1],
                            threads,
                        );
                    }
                }))?;
                best = best.min(started.elapsed().as_secs_f64() / iterations as f64);
            }
            line.push_str(&format!("  {symbol:>14} {:6.1} us", best * 1e6));
        }
        eprintln!("{line}");
    }
    Ok(())
}
