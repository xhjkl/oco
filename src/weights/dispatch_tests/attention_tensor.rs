//! Tensor-operation prompt attention (Apple10 GPUs) against the vector kernel.
use crate::Result;
use crate::weights::compute::AttentionParams;
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

fn noise(i: usize, salt: usize) -> f32 {
    let value = super::noise(i * 7 + salt) % 2001;
    value as f32 / 1000.0 - 1.0
}

/// Geometry pairs Qwen3.8 (24 over 4 heads of 256) and Nemotron (32 over 2 of 128) with causal
/// history, tile tails, windows and sinks.
#[test]
fn tensor_attention_matches_vector_kernel() -> Result<()> {
    let gpu = Gpu::new()?;
    if !gpu.device.supportsFamily(MTLGPUFamily::Apple10) {
        eprintln!("no Apple10 GPU; skipping");
        return Ok(());
    }
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    for (heads, kv_heads, dim, pos, m, window, sinks) in [
        (24usize, 4usize, 256usize, 0usize, 100usize, 0usize, false),
        (24, 4, 256, 300, 77, 0, false),
        (32, 2, 128, 129, 64, 0, false),
        (32, 2, 128, 50, 45, 37, true),
        (64, 8, 64, 200, 70, 128, true),
        // Scores that keep rising along the key axis rescale non-zero accumulators repeatedly.
        (24, 4, 256, 900, 40, 0, false),
    ] {
        let (tensor, vector) = match dim {
            256 => ("attention_tensor_256", "attention_vector_256"),
            128 => ("attention_tensor_128", "attention_vector_128"),
            _ => ("attention_tensor_64", "attention_vector_64"),
        };
        let tensor = builder.get(tensor, WeightFormat::F32, 0)?;
        let vector = builder.get(vector, WeightFormat::F32, 0)?;
        // Stale cache rows past the last visible key hold NaN, as after a rewind.
        let keys = pos + m;
        let stale = 80;
        let q: Vec<u16> = (0..m * heads * dim).map(|i| bf16::from_f32(noise(i, 1) * 3.0).to_bits()).collect();
        let cache = |salt: usize, scale: f32| -> Vec<u16> {
            (0..(keys + stale) * kv_heads * dim)
                .map(|i| {
                    if i < keys * kv_heads * dim {
                        f16::from_f32(noise(i, salt) * scale).to_bits()
                    } else {
                        0x7e00
                    }
                })
                .collect()
        };
        let (mut k, v) = (cache(2, 3.0), cache(3, 1.0));
        if pos == 900 {
            for key in 0..keys {
                let gain = 1.0 + 4.0 * key as f32 / keys as f32;
                for i in 0..kv_heads * dim {
                    let index = key * kv_heads * dim + i;
                    k[index] = f16::from_f32(f16::from_bits(k[index]).to_f32().abs() * gain).to_bits();
                }
            }
        }
        // A late key aligned with every query raises the running peak by far more than 8.
        let late = keys - 1 - keys / 5;
        for kv in 0..kv_heads {
            for d in 0..dim {
                k[(late * kv_heads + kv) * dim + d] = f16::from_f32(40.0 / (dim as f32).sqrt()).to_bits();
            }
        }
        // Sinks range from negligible to dominant.
        let sink: Vec<f32> = (0..heads).map(|i| noise(i, 4) * 2.0 + if i % 3 == 0 { 12.0 } else { 0.0 }).collect();
        let (q, k, v, sink) = (buffer(&gpu, &q), buffer(&gpu, &k), buffer(&gpu, &v), buffer(&gpu, &sink));
        let p = AttentionParams {
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            dim: dim as u32,
            pos: pos as u32,
            window: window as u32,
            sinks: sinks as u32,
            scale: 1.0 / (dim as f32).sqrt(),
            m: m as u32,
            keys: 0,
        };
        let mut outputs = Vec::new();
        for (pipeline, groups, threads) in [
            (&vector, [heads, m, 1], 256),
            (&tensor, [heads, m.div_ceil(32), 1], 256),
        ] {
            let out = buffer(&gpu, &vec![0x7fc1u16; m * heads * dim]);
            futures_lite::future::block_on(compute.submit(|e| {
                dispatch(
                    e,
                    pipeline,
                    &[(&*q, 0), (&*k, 0), (&*v, 0), (&*sink, 0), (&*out, 0)],
                    bytemuck::bytes_of(&p),
                    groups,
                    threads,
                )
            }))?;
            let mut values = vec![0u16; m * heads * dim];
            read(&out, &mut values);
            outputs.push(values);
        }
        for (variant, output) in outputs[1..].iter().enumerate() {
            let mut worst = 0f32;
            for (i, (&a, &b)) in output.iter().zip(&outputs[0]).enumerate() {
                let (a, b) = (bf16::from_bits(a).to_f32(), bf16::from_bits(b).to_f32());
                assert!(
                    a.is_finite() && b.is_finite(),
                    "dim {dim} pos {pos} variant {variant} output {i}: {a} against {b}"
                );
                worst = worst.max((a - b).abs());
            }
            eprintln!("dim {dim} pos {pos} m {m} window {window} sinks {sinks} variant {variant}: worst |tensor - vector| {worst:.2e}");
            assert!(worst < 2e-2, "dim {dim} pos {pos} variant {variant}: worst difference {worst}");
        }
    }
    Ok(())
}

/// Split-key single queries (Qwen3.8's six and GPT-OSS's eight query heads per KV head, with a
/// window and sinks) against the vector kernel.
#[test]
fn split_attention_matches_vector_kernel() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    for (heads, kv_heads, dim, pos, window, sinks, chunks) in [
        (24usize, 4usize, 256usize, 2999usize, 0usize, false, 32usize),
        (24, 4, 256, 70, 0, false, 32),
        (64, 8, 64, 4100, 0, true, 32),
        (64, 8, 64, 900, 128, true, 8),
    ] {
        let group = heads / kv_heads;
        let partial = builder.get(
            if dim == 256 { "attention_split_256_6" } else { "attention_split_64_8" },
            WeightFormat::F32,
            0,
        )?;
        let merge = builder.get(
            if dim == 256 { "attention_merge_256" } else { "attention_merge_64" },
            WeightFormat::F32,
            0,
        )?;
        let vector = builder.get(
            if dim == 256 { "attention_vector_256" } else { "attention_vector_64" },
            WeightFormat::F32,
            0,
        )?;
        assert_eq!(group, if dim == 256 { 6 } else { 8 });
        let keys = pos + 1;
        let q: Vec<u16> = (0..heads * dim).map(|i| bf16::from_f32(noise(i, 1) * 3.0).to_bits()).collect();
        let k: Vec<u16> = (0..keys * kv_heads * dim).map(|i| f16::from_f32(noise(i, 2) * 3.0).to_bits()).collect();
        let v: Vec<u16> = (0..keys * kv_heads * dim).map(|i| f16::from_f32(noise(i, 3)).to_bits()).collect();
        let sink: Vec<f32> = (0..heads).map(|i| noise(i, 4) * 2.0).collect();
        let (q, k, v, sink) = (buffer(&gpu, &q), buffer(&gpu, &k), buffer(&gpu, &v), buffer(&gpu, &sink));
        let partials = buffer(&gpu, &vec![0f32; chunks * heads * (dim + 2)]);
        let p = AttentionParams {
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            dim: dim as u32,
            pos: pos as u32,
            window: window as u32,
            sinks: sinks as u32,
            scale: 1.0 / (dim as f32).sqrt(),
            m: 1,
            keys: 0,
        };
        let split_out = buffer(&gpu, &vec![0x7fc1u16; heads * dim]);
        let vector_out = buffer(&gpu, &vec![0x7fc1u16; heads * dim]);
        futures_lite::future::block_on(compute.submit(|e| {
            dispatch(
                e,
                &partial,
                &[(&*q, 0), (&*k, 0), (&*v, 0), (&*sink, 0), (&*partials, 0)],
                bytemuck::bytes_of(&p),
                [kv_heads, chunks, 1],
                256,
            );
            dispatch(
                e,
                &merge,
                &[(&*partials, 0), (&*sink, 0), (&*split_out, 0)],
                bytemuck::bytes_of(&AttentionParams { m: chunks as u32, ..p }),
                [heads, 1, 1],
                256,
            );
            dispatch(
                e,
                &vector,
                &[(&*q, 0), (&*k, 0), (&*v, 0), (&*sink, 0), (&*vector_out, 0)],
                bytemuck::bytes_of(&p),
                [heads, 1, 1],
                256,
            );
        }))?;
        let (mut a, mut b) = (vec![0u16; heads * dim], vec![0u16; heads * dim]);
        read(&split_out, &mut a);
        read(&vector_out, &mut b);
        let worst = a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| {
                let (a, b) = (bf16::from_bits(a).to_f32(), bf16::from_bits(b).to_f32());
                assert!(a.is_finite() && b.is_finite(), "{a} against {b}");
                (a - b).abs()
            })
            .fold(0f32, f32::max);
        eprintln!("split {heads}/{kv_heads}x{dim} pos {pos} window {window}: worst {worst:.2e}");
        assert!(worst < 1e-2, "worst difference {worst}");
    }
    Ok(())
}
