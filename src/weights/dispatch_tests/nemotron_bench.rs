//! Throughput of the gathered expert kernels at Nemotron's prefill shape. Ignored by default:
//! `cargo test --release --lib expert_kernel_throughput -- --ignored --nocapture`.
use crate::gpu::{Gpu, dispatch, dispatch_indirect, tests::buffer};
use crate::weights::{compute::Compute, pipelines::Builder};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RoutePlanN {
    routes: u32,
    up_tiles: u32,
    down_tiles: u32,
    experts: u32,
    tile_rows: u32,
}

#[test]
#[ignore]
fn expert_kernel_throughput() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let pipelines = builder.nemotron()?;
    let (experts, active, hidden, width) = (128usize, 6usize, 2688usize, 1856usize);
    let tokens: usize = std::env::var("OCO_BENCH_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let routes = tokens * active;
    // Q4_0 banks with pseudo-random bytes; scales kept small so sums stay finite.
    let block_bytes = |n: usize, k: usize| experts * n * k / 32 * 18;
    let fill = |bytes: usize| -> Vec<u8> {
        (0..bytes)
            .map(|i| {
                if i % 18 < 2 {
                    if i % 18 == 0 { 0x00 } else { 0x2c }
                } else {
                    ((i * 2654435761usize) >> 7) as u8
                }
            })
            .collect()
    };
    let up = buffer(&gpu, &fill(block_bytes(width, hidden)));
    let down = buffer(&gpu, &fill(block_bytes(hidden, width)));
    let x: Vec<u16> = (0..tokens * hidden)
        .map(|i| half::bf16::from_f32(((i * 7) % 97) as f32 / 48.0 - 1.0).to_bits())
        .collect();
    let x = buffer(&gpu, &x);
    let mid = buffer(&gpu, &vec![0u16; routes * width]);
    let out = buffer(&gpu, &vec![0u16; routes * hidden]);
    // Routes drawn uniformly over experts, like a well-balanced router.
    let ids: Vec<u32> = (0..routes)
        .map(|i| ((i as u64 * 2654435761 + 12345) % experts as u64) as u32)
        .collect();
    let ids = buffer(&gpu, &ids);
    let plan = buffer(
        &gpu,
        &vec![0u32; super::super::compute::plan_words(routes, experts)],
    );
    let up_matrix = [
        hidden as u32,
        width as u32,
        routes as u32,
        0,
        active as u32,
        experts as u32,
        0,
    ];
    let down_matrix = [
        width as u32,
        hidden as u32,
        routes as u32,
        1,
        active as u32,
        experts as u32,
        0,
    ];
    let iterations = 10;
    for (size, tile_rows) in [(0usize, 8u32), (1, 32)] {
        let p = RoutePlanN {
            routes: routes as u32,
            up_tiles: width.div_ceil(64) as u32,
            down_tiles: hidden.div_ceil(64) as u32,
            experts: experts as u32,
            tile_rows,
        };
        futures_lite::future::block_on(compute.submit(|e| {
            dispatch(
                e,
                &pipelines.route_plan,
                &[(&ids, 0), (&plan, 0)],
                bytemuck::bytes_of(&p),
                [1, 1, 1],
                128,
            );
        }))?;
        let mut line = format!("tiles of {tile_rows:>2} routes   ");
        for (pipeline, w, input, output, matrix, offset) in [
            (&pipelines.expert_up[size], &up, &x, &mid, up_matrix, 0usize),
            (
                &pipelines.expert_down[size],
                &down,
                &mid,
                &out,
                down_matrix,
                12,
            ),
        ] {
            // One warm-up submission, then the timed batch.
            futures_lite::future::block_on(compute.submit(|e| {
                dispatch_indirect(
                    e,
                    pipeline,
                    &[(w, 0), (input, 0), (output, 0), (&plan, 0)],
                    bytemuck::cast_slice(&matrix),
                    (&plan, offset),
                    128,
                );
            }))?;
            let started = Instant::now();
            futures_lite::future::block_on(compute.submit(|e| {
                for _ in 0..iterations {
                    dispatch_indirect(
                        e,
                        pipeline,
                        &[(w, 0), (input, 0), (output, 0), (&plan, 0)],
                        bytemuck::cast_slice(&matrix),
                        (&plan, offset),
                        128,
                    );
                }
            }))?;
            let per_call = started.elapsed().as_secs_f64() / iterations as f64;
            let flops = 2.0 * routes as f64 * hidden as f64 * width as f64;
            line.push_str(&format!(
                "  {:8.1} us {:5.2} TFLOPS",
                per_call * 1e6,
                flops / per_call / 1e12
            ));
        }
        eprintln!("{line}");
    }
    // Single-token gathered matvec, six routes, at several threadgroup sizes.
    let ids6 = buffer(&gpu, &[3u32, 17, 42, 77, 99, 120]);
    for threads in [64usize, 128, 256] {
        let p = [
            hidden as u32,
            width as u32,
            active as u32,
            0,
            active as u32,
            experts as u32,
            0,
        ];
        let started = Instant::now();
        futures_lite::future::block_on(compute.submit(|e| {
            for _ in 0..100 {
                dispatch(
                    e,
                    &pipelines.expert_vector,
                    &[(&up, 0), (&x, 0), (&mid, 0), (&plan, 0), (&ids6, 0)],
                    bytemuck::cast_slice(&p),
                    [width.div_ceil(threads / 8), active, 1],
                    threads,
                );
            }
        }))?;
        let up_us = started.elapsed().as_secs_f64() * 1e4;
        let p = [
            width as u32,
            hidden as u32,
            active as u32,
            1,
            active as u32,
            experts as u32,
            0,
        ];
        let started = Instant::now();
        futures_lite::future::block_on(compute.submit(|e| {
            for _ in 0..100 {
                dispatch(
                    e,
                    &pipelines.expert_vector_relu,
                    &[(&down, 0), (&mid, 0), (&out, 0), (&plan, 0), (&ids6, 0)],
                    bytemuck::cast_slice(&p),
                    [hidden.div_ceil(threads / 8), active, 1],
                    threads,
                );
            }
        }))?;
        eprintln!(
            "decode matvec x6 routes, {threads:>3} threads: up {up_us:6.1} us  down {:6.1} us",
            started.elapsed().as_secs_f64() * 1e4
        );
    }
    // Dense tiled products at the Mamba in-projection and Qwen feed-forward shapes.
    for (kind, k, n, name) in [
        (
            crate::tensor::WeightFormat::Q8_0,
            2688usize,
            10304usize,
            "matmul",
        ),
        (
            crate::tensor::WeightFormat::Q4_0,
            5120,
            17408,
            "matmul_q4_half",
        ),
    ] {
        let row_bytes = if kind == crate::tensor::WeightFormat::Q8_0 {
            34
        } else {
            18
        };
        let w = buffer(
            &gpu,
            &(0..n * k / 32 * row_bytes)
                .map(|i| {
                    if i % row_bytes < 2 {
                        if i % row_bytes == 0 { 0 } else { 0x2c }
                    } else {
                        ((i * 2654435761usize) >> 9) as u8
                    }
                })
                .collect::<Vec<u8>>(),
        );
        let xin = buffer(&gpu, &vec![half::bf16::from_f32(0.5).to_bits(); tokens * k]);
        let y = buffer(&gpu, &vec![0u16; tokens * n]);
        let scales = buffer(&gpu, &vec![[1.0f32, 1.0f32]; tokens]);
        let pipeline = builder.get(name, kind, 0)?;
        let p = [k as u32, n as u32, tokens as u32, 0, 4, 1, 0];
        let groups = [tokens / 64, n.div_ceil(64), 1];
        futures_lite::future::block_on(compute.submit(|e| {
            dispatch(
                e,
                &pipeline,
                &[(&w, 0), (&xin, 0), (&y, 0), (&plan, 0), (&scales, 0)],
                bytemuck::cast_slice(&p),
                groups,
                128,
            );
        }))?;
        let started = Instant::now();
        futures_lite::future::block_on(compute.submit(|e| {
            for _ in 0..iterations {
                dispatch(
                    e,
                    &pipeline,
                    &[(&w, 0), (&xin, 0), (&y, 0), (&plan, 0), (&scales, 0)],
                    bytemuck::cast_slice(&p),
                    groups,
                    128,
                );
            }
        }))?;
        let per_call = started.elapsed().as_secs_f64() / iterations as f64;
        eprintln!(
            "dense {name} {k}x{n} x{tokens}: {:8.1} us  {:5.2} TFLOPS",
            per_call * 1e6,
            2.0 * (tokens * k * n) as f64 / per_call / 1e12
        );
    }
    // Dense Q4_0 single-row products at the Qwen feed-forward shapes.
    let q4 = |n: usize, k: usize| -> Vec<u8> {
        (0..n * k / 32 * 18)
            .map(|i| {
                if i % 18 < 2 {
                    if i % 18 == 0 { 0 } else { 0x2c }
                } else {
                    ((i * 2654435761usize) >> 9) as u8
                }
            })
            .collect()
    };
    let xq: Vec<u16> = (0..17408)
        .map(|i| half::bf16::from_f32(((i * 7) % 97) as f32 / 48.0 - 1.0).to_bits())
        .collect();
    let xq = buffer(&gpu, &xq);
    for (k, n) in [(5120usize, 17408usize), (17408, 5120), (5120, 12288)] {
        let w = buffer(&gpu, &q4(n, k));
        let y = buffer(&gpu, &vec![0u16; n]);
        for threads in [64usize, 128, 256] {
            let pipeline = builder.get("matvec_q4", crate::tensor::WeightFormat::Q4_0, 0)?;
            let p = [k as u32, n as u32, 1, 0, 4, 1, 0];
            futures_lite::future::block_on(compute.submit(|e| {
                dispatch(
                    e,
                    &pipeline,
                    &[(&w, 0), (&xq, 0), (&y, 0), (&plan, 0), (&ids6, 0)],
                    bytemuck::cast_slice(&p),
                    [n.div_ceil(threads / 8), 1, 1],
                    threads,
                );
            }))?;
            let started = Instant::now();
            futures_lite::future::block_on(compute.submit(|e| {
                for _ in 0..100 {
                    dispatch(
                        e,
                        &pipeline,
                        &[(&w, 0), (&xq, 0), (&y, 0), (&plan, 0), (&ids6, 0)],
                        bytemuck::cast_slice(&p),
                        [n.div_ceil(threads / 8), 1, 1],
                        threads,
                    );
                }
            }))?;
            let us = started.elapsed().as_secs_f64() * 1e4;
            eprintln!(
                "q4 matvec {k}x{n}, {threads:>3} threads: {us:6.1} us  {:5.0} GB/s",
                (n * k) as f64 * 18.0 / 32.0 / us / 1e3
            );
        }
    }
    // Dense Q8_0 single-row products at the Mamba projection shapes.
    let q8 = |n: usize, k: usize| -> Vec<u8> {
        (0..n * k / 32 * 34)
            .map(|i| {
                if i % 34 < 2 {
                    if i % 34 == 0 { 0 } else { 0x2c }
                } else {
                    ((i * 2654435761usize) >> 9) as u8
                }
            })
            .collect()
    };
    for (k, n) in [(2688usize, 10304usize), (4096, 2688), (2688, 3712)] {
        let w = buffer(&gpu, &q8(n, k));
        let y = buffer(&gpu, &vec![0u16; n]);
        for threads in [64usize, 128, 256] {
            let pipeline = builder.get("matvec", crate::tensor::WeightFormat::Q8_0, 0)?;
            let p = [k as u32, n as u32, 1, 0, 4, 1, 0];
            let started = Instant::now();
            futures_lite::future::block_on(compute.submit(|e| {
                for _ in 0..100 {
                    dispatch(
                        e,
                        &pipeline,
                        &[(&w, 0), (&x, 0), (&y, 0), (&plan, 0), (&ids6, 0)],
                        bytemuck::cast_slice(&p),
                        [n.div_ceil(threads / 8), 1, 1],
                        threads,
                    );
                }
            }))?;
            let us = started.elapsed().as_secs_f64() * 1e4;
            eprintln!(
                "q8 matvec {k}x{n}, {threads:>3} threads: {us:6.1} us  {:5.0} GB/s",
                (n * k) as f64 * 34.0 / 32.0 / us / 1e3
            );
        }
    }
    Ok(())
}
