//! Inkling attention: the tensor-operation prompt tiles (Apple10) and ring caches against the
//! vector kernel over a capacity-long cache.
use crate::Result;
use crate::weights::compute::InklingAttention;
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

const HEADS: usize = 24;
const KV_HEADS: usize = 8;
const D: usize = 128;
const REL: usize = 16;

/// Inkling's geometry (24 query heads over 8 KV heads of 128) with tile tails, unaligned chunk
/// starts, windows past and inside the history, relative extents shorter than the history, ring
/// caches that wrap inside a key tile's span, and per-query sinks from negligible to dominant.
#[test]
fn inkling_attention_variants_match_the_linear_vector_kernel() -> Result<()> {
    let gpu = Gpu::new()?;
    let tensor_ok = gpu.device.supportsFamily(MTLGPUFamily::Apple10);
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let vector = builder.get("inkling_attention", WeightFormat::F32, 0)?;
    let tensor = match tensor_ok {
        true => Some(builder.get("inkling_attention_tensor", WeightFormat::F32, 0)?),
        false => {
            eprintln!("no Apple10 GPU; checking ring caches on the vector kernel only");
            None
        }
    };
    // (pos, m, window, extent, ring)
    for (pos, m, window, extent, ring) in [
        (0usize, 77usize, 0usize, 200usize, 0usize),
        (300, 64, 0, 8192, 0),
        (129, 45, 0, 100, 0),
        (900, 100, 256, 256, 0),
        (1000, 100, 256, 256, 384),
        (1919, 33, 1024, 1024, 1536),
        (5000, 512, 1024, 1024, 1536),
        (2047, 17, 0, 1500, 0),
    ] {
        let keys = pos + m;
        let stale = 64;
        let q: Vec<u16> = (0..m * HEADS * D)
            .map(|i| bf16::from_f32(noise(i, 1) * 1.5).to_bits())
            .collect();
        let r: Vec<f32> = (0..m * HEADS * REL).map(|i| noise(i, 5) * 0.8).collect();
        let table: Vec<f32> = (0..extent * REL).map(|i| noise(i, 6) * 1.5).collect();
        // Sinks per query and head, a third of them dominant.
        let sinks: Vec<u16> = (0..m * HEADS)
            .map(|i| {
                bf16::from_f32(noise(i, 4) * 2.0 + if i % 3 == 0 { 9.0 } else { 0.0 }).to_bits()
            })
            .collect();
        let gate: Vec<u16> = (0..m * HEADS * D)
            .map(|i| bf16::from_f32(noise(i, 7) * 3.0).to_bits())
            .collect();
        let row = |s: usize, salt: usize, scale: f32| -> Vec<u16> {
            (0..KV_HEADS * D)
                .map(|i| f16::from_f32(noise(s * KV_HEADS * D + i, salt) * scale).to_bits())
                .collect()
        };
        // The capacity-long cache, NaN past the last key as after a rewind.
        let linear = |salt: usize, scale: f32| -> Vec<u16> {
            let mut cache: Vec<u16> = (0..keys).flat_map(|s| row(s, salt, scale)).collect();
            cache.resize((keys + stale) * KV_HEADS * D, 0x7e00);
            cache
        };
        // A ring holding the newest `ring` positions at their rows.
        let ringed = |salt: usize, scale: f32| -> Vec<u16> {
            let mut cache = vec![0x7e00u16; ring * KV_HEADS * D];
            for s in keys.saturating_sub(ring)..keys {
                let at = (s % ring) * KV_HEADS * D;
                cache[at..at + KV_HEADS * D].copy_from_slice(&row(s, salt, scale));
            }
            cache
        };
        // Keys at five times unit scale make scores peaky: a few keys carry most of each row's
        // mass, so a dropped, duplicated or misplaced key moves outputs at full scale.
        let (k, v) = (linear(2, 5.0), linear(3, 1.0));
        let p = InklingAttention {
            heads: HEADS as u32,
            kv_heads: KV_HEADS as u32,
            pos: pos as u32,
            window: window as u32,
            m: m as u32,
            extent: extent as u32,
            scale: 1.0 / (D as f32).sqrt(),
            ring: 0,
        };
        let (q, r, table, sinks, gate) = (
            buffer(&gpu, &q),
            buffer(&gpu, &r),
            buffer(&gpu, &table),
            buffer(&gpu, &sinks),
            buffer(&gpu, &gate),
        );
        let run =
            |pipeline, k: &[u16], v: &[u16], ring: usize, groups: [usize; 3]| -> Result<Vec<u16>> {
                let (k, v) = (buffer(&gpu, k), buffer(&gpu, v));
                let out = buffer(&gpu, &vec![0x7fc1u16; m * HEADS * D]);
                let p = InklingAttention {
                    ring: ring as u32,
                    ..p
                };
                futures_lite::future::block_on(compute.submit(|e| {
                    dispatch(
                        e,
                        pipeline,
                        &[
                            (&*q, 0),
                            (&*k, 0),
                            (&*v, 0),
                            (&*r, 0),
                            (&*table, 0),
                            (&*sinks, 0),
                            (&*gate, 0),
                            (&*out, 0),
                        ],
                        bytemuck::bytes_of(&p),
                        groups,
                        256,
                    )
                }))?;
                let mut values = vec![0u16; m * HEADS * D];
                read(&out, &mut values);
                Ok(values)
            };
        let reference = run(&vector, &k, &v, 0, [HEADS, m, 1])?;
        let mut variants = Vec::new();
        if ring > 0 {
            let (rk, rv) = (ringed(2, 5.0), ringed(3, 1.0));
            variants.push(("vector ring", run(&vector, &rk, &rv, ring, [HEADS, m, 1])?));
            if let Some(tensor) = &tensor {
                variants.push((
                    "tensor ring",
                    run(tensor, &rk, &rv, ring, [HEADS, m.div_ceil(32), 1])?,
                ));
            }
        }
        if let Some(tensor) = &tensor {
            variants.push((
                "tensor",
                run(tensor, &k, &v, 0, [HEADS, m.div_ceil(32), 1])?,
            ));
        }
        for (name, output) in &variants {
            let mut worst = 0f32;
            for (i, (&a, &b)) in output.iter().zip(&reference).enumerate() {
                let (a, b) = (bf16::from_bits(a).to_f32(), bf16::from_bits(b).to_f32());
                assert!(
                    a.is_finite() && b.is_finite(),
                    "pos {pos} m {m} {name} output {i}: {a} against {b}"
                );
                worst = worst.max((a - b).abs());
            }
            eprintln!(
                "pos {pos} m {m} window {window} extent {extent} ring {ring} {name}: worst {worst:.2e}"
            );
            let bound = if name.starts_with("vector") {
                0.0
            } else {
                8e-3
            };
            assert!(
                worst <= bound,
                "pos {pos} m {m} {name}: worst difference {worst}"
            );
        }
    }
    Ok(())
}

/// Split-key single queries, with one or three query heads per threadgroup, against the vector
/// kernel: long full-attention histories past the relative extent, a ring, and a window.
#[test]
fn inkling_split_attention_matches_the_vector_kernel() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let vector = builder.get("inkling_attention", WeightFormat::F32, 0)?;
    let merge = builder.get("inkling_merge", WeightFormat::F32, 0)?;
    // (pos, window, extent, ring, chunks)
    for (pos, window, extent, ring, chunks) in [
        (20000usize, 0usize, 8192usize, 0usize, 32usize),
        (4099, 0, 8192, 0, 32),
        (70, 0, 100, 0, 32),
        (3000, 1024, 1024, 1536, 8),
    ] {
        let keys = pos + 1;
        let rows = if ring > 0 { ring } else { keys };
        let q: Vec<u16> = (0..HEADS * D)
            .map(|i| bf16::from_f32(noise(i, 1) * 1.5).to_bits())
            .collect();
        let r: Vec<f32> = (0..HEADS * REL).map(|i| noise(i, 5) * 0.8).collect();
        let table: Vec<f32> = (0..extent * REL).map(|i| noise(i, 6) * 1.5).collect();
        let sinks: Vec<u16> = (0..HEADS)
            .map(|i| {
                bf16::from_f32(noise(i, 4) * 2.0 + if i % 3 == 0 { 9.0 } else { 0.0 }).to_bits()
            })
            .collect();
        let gate: Vec<u16> = (0..HEADS * D)
            .map(|i| bf16::from_f32(noise(i, 7) * 3.0).to_bits())
            .collect();
        let mut k = vec![0u16; rows * KV_HEADS * D];
        let mut v = vec![0u16; rows * KV_HEADS * D];
        for s in keys.saturating_sub(rows)..keys {
            let at = (if ring > 0 { s % ring } else { s }) * KV_HEADS * D;
            for i in 0..KV_HEADS * D {
                k[at + i] = f16::from_f32(noise(s * KV_HEADS * D + i, 2) * 5.0).to_bits();
                v[at + i] = f16::from_f32(noise(s * KV_HEADS * D + i, 3)).to_bits();
            }
        }
        let p = InklingAttention {
            heads: HEADS as u32,
            kv_heads: KV_HEADS as u32,
            pos: pos as u32,
            window: window as u32,
            m: 1,
            extent: extent as u32,
            scale: 1.0 / (D as f32).sqrt(),
            ring: ring as u32,
        };
        let [q, r, table, sinks, gate, k, v] = [
            buffer(&gpu, &q),
            buffer(&gpu, &r),
            buffer(&gpu, &table),
            buffer(&gpu, &sinks),
            buffer(&gpu, &gate),
            buffer(&gpu, &k),
            buffer(&gpu, &v),
        ];
        let partials = buffer(&gpu, &vec![0f32; chunks * HEADS * (D + 2)]);
        let mut outputs = Vec::new();
        for variant in ["vector", "inkling_split_1", "inkling_split_3"] {
            let out = buffer(&gpu, &vec![0x7fc1u16; HEADS * D]);
            let split = (variant != "vector")
                .then(|| builder.get(variant, WeightFormat::F32, 0))
                .transpose()?;
            futures_lite::future::block_on(compute.submit(|e| match &split {
                None => dispatch(
                    e,
                    &vector,
                    &[
                        (&*q, 0),
                        (&*k, 0),
                        (&*v, 0),
                        (&*r, 0),
                        (&*table, 0),
                        (&*sinks, 0),
                        (&*gate, 0),
                        (&*out, 0),
                    ],
                    bytemuck::bytes_of(&p),
                    [HEADS, 1, 1],
                    256,
                ),
                Some(split) => {
                    let groups = if variant.ends_with('3') {
                        KV_HEADS
                    } else {
                        HEADS
                    };
                    dispatch(
                        e,
                        split,
                        &[
                            (&*q, 0),
                            (&*k, 0),
                            (&*v, 0),
                            (&*r, 0),
                            (&*table, 0),
                            (&*partials, 0),
                        ],
                        bytemuck::bytes_of(&p),
                        [groups, chunks, 1],
                        256,
                    );
                    dispatch(
                        e,
                        &merge,
                        &[(&*partials, 0), (&*sinks, 0), (&*gate, 0), (&*out, 0)],
                        bytemuck::bytes_of(&InklingAttention {
                            m: chunks as u32,
                            ..p
                        }),
                        [HEADS, 1, 1],
                        128,
                    );
                }
            }))?;
            let mut values = vec![0u16; HEADS * D];
            read(&out, &mut values);
            outputs.push((variant, values));
        }
        for (variant, output) in &outputs[1..] {
            let mut worst = 0f32;
            for (i, (&a, &b)) in output.iter().zip(&outputs[0].1).enumerate() {
                let (a, b) = (bf16::from_bits(a).to_f32(), bf16::from_bits(b).to_f32());
                assert!(
                    a.is_finite() && b.is_finite(),
                    "pos {pos} {variant} output {i}: {a} against {b}"
                );
                worst = worst.max((a - b).abs());
            }
            eprintln!(
                "pos {pos} window {window} extent {extent} ring {ring} chunks {chunks} {variant}: worst {worst:.2e}"
            );
            assert!(
                worst < 1e-3,
                "pos {pos} {variant}: worst difference {worst}"
            );
        }
    }
    Ok(())
}

/// One strong key and a 131,071-key tail 18 below it: the tail holds 0.2% of the attention mass
/// and the whole output, since the strong key's value is zero. Unscaled half probabilities would
/// flush every tail weight to zero (after Astra's fixture).
#[test]
fn inkling_tensor_attention_keeps_weak_tails() -> Result<()> {
    let gpu = Gpu::new()?;
    if !gpu.device.supportsFamily(MTLGPUFamily::Apple10) {
        eprintln!("no Apple10 GPU; skipping");
        return Ok(());
    }
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let vector = builder.get("inkling_attention", WeightFormat::F32, 0)?;
    let tensor = builder.get("inkling_attention_tensor", WeightFormat::F32, 0)?;
    let (m, keys) = (33usize, 131_072usize);
    let pos = keys - m;
    let mut q = vec![0u16; m * HEADS * D];
    for t in 0..m * HEADS {
        q[t * D] = bf16::from_f32(16.0).to_bits();
    }
    let mut k = vec![0u16; keys * KV_HEADS * D];
    let mut v = vec![f16::from_f32(1.0).to_bits(); keys * KV_HEADS * D];
    for kv in 0..KV_HEADS {
        k[kv * D] = f16::from_f32(18.0 * (D as f32).sqrt() / 16.0).to_bits();
        v[kv * D..(kv + 1) * D].fill(0);
    }
    let r = vec![0f32; m * HEADS * REL];
    let table = vec![0f32; 8192 * REL];
    let sinks = vec![bf16::from_f32(-30.0).to_bits(); m * HEADS];
    let gate = vec![bf16::from_f32(30.0).to_bits(); m * HEADS * D];
    let [q, k, v, r, table, sinks, gate] = [
        buffer(&gpu, &q),
        buffer(&gpu, &k),
        buffer(&gpu, &v),
        buffer(&gpu, &r),
        buffer(&gpu, &table),
        buffer(&gpu, &sinks),
        buffer(&gpu, &gate),
    ];
    let p = InklingAttention {
        heads: HEADS as u32,
        kv_heads: KV_HEADS as u32,
        pos: pos as u32,
        window: 0,
        m: m as u32,
        extent: 8192,
        scale: 1.0 / (D as f32).sqrt(),
        ring: 0,
    };
    let mut outputs = Vec::new();
    for (pipeline, groups) in [
        (&vector, [HEADS, m, 1]),
        (&tensor, [HEADS, m.div_ceil(32), 1]),
    ] {
        let out = buffer(&gpu, &vec![0x7fc1u16; m * HEADS * D]);
        futures_lite::future::block_on(compute.submit(|e| {
            dispatch(
                e,
                pipeline,
                &[
                    (&*q, 0),
                    (&*k, 0),
                    (&*v, 0),
                    (&*r, 0),
                    (&*table, 0),
                    (&*sinks, 0),
                    (&*gate, 0),
                    (&*out, 0),
                ],
                bytemuck::bytes_of(&p),
                groups,
                256,
            )
        }))?;
        let mut values = vec![0u16; m * HEADS * D];
        read(&out, &mut values);
        outputs.push(values);
    }
    // Every query sees the strong key and `pos + t` tail keys.
    let _ = &outputs;
    for t in [0, m - 1] {
        let tail = (pos + t) as f64;
        let weak = (-18.0f64).exp();
        let expected = tail * weak / (1.0 + tail * weak);
        for (name, output) in ["vector", "tensor"].iter().zip(&outputs) {
            let got = bf16::from_bits(output[t * HEADS * D + 1]).to_f32() as f64;
            eprintln!("row {t} {name}: {got:.6e} (expected {expected:.6e})");
            assert!(
                (got - expected).abs() <= 0.01 * expected,
                "row {t} {name}: {got} against {expected}"
            );
        }
    }
    Ok(())
}

/// A dominant key whose value holds an 11.3 component, the most an RMS-normalized 128-wide head
/// allows: its scaled probability (up to 32768 e^0.5) times that value is far past half range,
/// so products must widen before they accumulate (Fable's review).
#[test]
fn inkling_tensor_attention_keeps_large_values_finite() -> Result<()> {
    let gpu = Gpu::new()?;
    if !gpu.device.supportsFamily(MTLGPUFamily::Apple10) {
        eprintln!("no Apple10 GPU; skipping");
        return Ok(());
    }
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let vector = builder.get("inkling_attention", WeightFormat::F32, 0)?;
    let tensor = builder.get("inkling_attention_tensor", WeightFormat::F32, 0)?;
    let (m, keys) = (64usize, 4096usize);
    let pos = keys - m;
    let mut q = vec![0u16; m * HEADS * D];
    for t in 0..m * HEADS {
        q[t * D] = bf16::from_f32(16.0).to_bits();
    }
    let mut k = vec![0u16; keys * KV_HEADS * D];
    let mut v = vec![0u16; keys * KV_HEADS * D];
    for s in 0..keys {
        for kv in 0..KV_HEADS {
            for d in 0..D {
                v[(s * KV_HEADS + kv) * D + d] = f16::from_f32(noise(s * D + d, 3) * 0.5).to_bits();
            }
        }
    }
    // Key 100 dominates every query by 12; its value is one-hot at 11.3.
    for kv in 0..KV_HEADS {
        k[(100 * KV_HEADS + kv) * D] = f16::from_f32(12.0 * (D as f32).sqrt() / 16.0).to_bits();
        let at = (100 * KV_HEADS + kv) * D;
        v[at..at + D].fill(0);
        v[at + 5] = f16::from_f32(11.3).to_bits();
    }
    let r = vec![0f32; m * HEADS * REL];
    let table = vec![0f32; 1024 * REL];
    let sinks = vec![bf16::from_f32(-30.0).to_bits(); m * HEADS];
    let gate = vec![bf16::from_f32(30.0).to_bits(); m * HEADS * D];
    let [q, k, v, r, table, sinks, gate] = [
        buffer(&gpu, &q),
        buffer(&gpu, &k),
        buffer(&gpu, &v),
        buffer(&gpu, &r),
        buffer(&gpu, &table),
        buffer(&gpu, &sinks),
        buffer(&gpu, &gate),
    ];
    let p = InklingAttention {
        heads: HEADS as u32,
        kv_heads: KV_HEADS as u32,
        pos: pos as u32,
        window: 0,
        m: m as u32,
        extent: 1024,
        scale: 1.0 / (D as f32).sqrt(),
        ring: 0,
    };
    let mut outputs = Vec::new();
    for (pipeline, groups) in [
        (&vector, [HEADS, m, 1]),
        (&tensor, [HEADS, m.div_ceil(32), 1]),
    ] {
        let out = buffer(&gpu, &vec![0x7fc1u16; m * HEADS * D]);
        futures_lite::future::block_on(compute.submit(|e| {
            dispatch(
                e,
                pipeline,
                &[
                    (&*q, 0),
                    (&*k, 0),
                    (&*v, 0),
                    (&*r, 0),
                    (&*table, 0),
                    (&*sinks, 0),
                    (&*gate, 0),
                    (&*out, 0),
                ],
                bytemuck::bytes_of(&p),
                groups,
                256,
            )
        }))?;
        let mut values = vec![0u16; m * HEADS * D];
        read(&out, &mut values);
        outputs.push(values);
    }
    let mut worst = 0f32;
    for (i, (&a, &b)) in outputs[1].iter().zip(&outputs[0]).enumerate() {
        let (a, b) = (bf16::from_bits(a).to_f32(), bf16::from_bits(b).to_f32());
        assert!(
            a.is_finite() && b.is_finite(),
            "output {i}: tensor {a} against vector {b}"
        );
        worst = worst.max((a - b).abs() / b.abs().max(1.0));
    }
    let big = bf16::from_bits(outputs[1][5]).to_f32();
    eprintln!("dominant channel {big:.4} (about 11.3); worst relative difference {worst:.2e}");
    assert!(big > 10.0, "the dominant value was lost: {big}");
    assert!(worst < 8e-3, "worst relative difference {worst}");
    Ok(())
}

/// Over 1M keys, one early key 17 above all others and values of one everywhere: the output is
/// one. Accumulating the weak tail inside the tensor operation onto the strong key's large sum
/// returned 0.957 (Astra's review).
#[test]
#[ignore = "allocates 4 GiB of keys and values; run on Apple10 with --ignored"]
fn inkling_tensor_attention_sums_a_million_weak_keys() -> Result<()> {
    let gpu = Gpu::new()?;
    if !gpu.device.supportsFamily(MTLGPUFamily::Apple10) {
        eprintln!("no Apple10 GPU; skipping");
        return Ok(());
    }
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let tensor = builder.get("inkling_attention_tensor", WeightFormat::F32, 0)?;
    let (m, keys) = (32usize, 1usize << 20);
    let pos = keys - m;
    let mut q = vec![0u16; m * HEADS * D];
    for t in 0..m * HEADS {
        q[t * D] = bf16::from_f32(16.0).to_bits();
    }
    let mut k = vec![0u16; keys * KV_HEADS * D];
    for kv in 0..KV_HEADS {
        k[kv * D] = f16::from_f32(17.0 * (D as f32).sqrt() / 16.0).to_bits();
    }
    let v = vec![f16::from_f32(1.0).to_bits(); keys * KV_HEADS * D];
    let r = vec![0f32; m * HEADS * REL];
    let table = vec![0f32; 1024 * REL];
    let sinks = vec![bf16::from_f32(-30.0).to_bits(); m * HEADS];
    let gate = vec![bf16::from_f32(30.0).to_bits(); m * HEADS * D];
    let [q, k, v, r, table, sinks, gate] = [
        buffer(&gpu, &q),
        buffer(&gpu, &k),
        buffer(&gpu, &v),
        buffer(&gpu, &r),
        buffer(&gpu, &table),
        buffer(&gpu, &sinks),
        buffer(&gpu, &gate),
    ];
    let p = InklingAttention {
        heads: HEADS as u32,
        kv_heads: KV_HEADS as u32,
        pos: pos as u32,
        window: 0,
        m: m as u32,
        extent: 1024,
        scale: 1.0 / (D as f32).sqrt(),
        ring: 0,
    };
    let out = buffer(&gpu, &vec![0x7fc1u16; m * HEADS * D]);
    futures_lite::future::block_on(compute.submit(|e| {
        dispatch(
            e,
            &tensor,
            &[
                (&*q, 0),
                (&*k, 0),
                (&*v, 0),
                (&*r, 0),
                (&*table, 0),
                (&*sinks, 0),
                (&*gate, 0),
                (&*out, 0),
            ],
            bytemuck::bytes_of(&p),
            [HEADS, 1, 1],
            256,
        )
    }))?;
    let mut values = vec![0u16; m * HEADS * D];
    read(&out, &mut values);
    let values: Vec<f32> = values
        .iter()
        .map(|&x| bf16::from_bits(x).to_f32())
        .collect();
    assert!(values.iter().all(|x| x.is_finite()), "non-finite output");
    let worst = values.iter().map(|x| (x - 1.0).abs()).fold(0f32, f32::max);
    eprintln!("worst |output - 1| over 1M keys: {worst:.3e}");
    assert!(worst < 1e-2, "lost the weak tail: worst {worst}");
    Ok(())
}

/// Bias tables that random fixtures cannot pin: an impulse at the last row of the extent, seen
/// by exactly one key per query at the extent boundary and one past it (Fable's review), and
/// entries in half's subnormal range amplified by large relative vectors, in case the hardware
/// flushes them when staging.
#[test]
fn inkling_tensor_attention_matches_special_bias_tables() -> Result<()> {
    let gpu = Gpu::new()?;
    if !gpu.device.supportsFamily(MTLGPUFamily::Apple10) {
        eprintln!("no Apple10 GPU; skipping");
        return Ok(());
    }
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let vector = builder.get("inkling_attention", WeightFormat::F32, 0)?;
    let tensor = builder.get("inkling_attention_tensor", WeightFormat::F32, 0)?;
    // (name, pos, m, extent, r, table entry at distance d)
    type Entry = Box<dyn Fn(usize) -> f32>;
    let impulse = |extent: usize| move |d: usize| if d == extent - 1 { 4.0 } else { 0.0 };
    let cases: [(&str, usize, usize, usize, f32, Entry); 3] = [
        (
            "impulse at 1537",
            2047,
            17,
            1537,
            0.5,
            Box::new(impulse(1537)),
        ),
        (
            "impulse at 1536",
            2047,
            17,
            1536,
            0.5,
            Box::new(impulse(1536)),
        ),
        (
            "subnormal",
            300,
            64,
            1024,
            1000.0,
            Box::new(|d: usize| if d.is_multiple_of(2) { 3e-5 } else { 0.0 }),
        ),
    ];
    for (name, pos, m, extent, rv, entry) in cases {
        let keys = pos + m;
        let q: Vec<u16> = (0..m * HEADS * D)
            .map(|i| bf16::from_f32(noise(i, 1) * 1.5).to_bits())
            .collect();
        let r = vec![rv / (REL as f32).sqrt(); m * HEADS * REL];
        let table: Vec<f32> = (0..extent * REL).map(|i| entry(i / REL)).collect();
        let sinks = vec![bf16::from_f32(-30.0).to_bits(); m * HEADS];
        let gate = vec![bf16::from_f32(30.0).to_bits(); m * HEADS * D];
        let k: Vec<u16> = (0..keys * KV_HEADS * D)
            .map(|i| f16::from_f32(noise(i, 2) * 0.5).to_bits())
            .collect();
        let v: Vec<u16> = (0..keys * KV_HEADS * D)
            .map(|i| f16::from_f32(noise(i, 3)).to_bits())
            .collect();
        let [q, k, v, r, table, sinks, gate] = [
            buffer(&gpu, &q),
            buffer(&gpu, &k),
            buffer(&gpu, &v),
            buffer(&gpu, &r),
            buffer(&gpu, &table),
            buffer(&gpu, &sinks),
            buffer(&gpu, &gate),
        ];
        let p = InklingAttention {
            heads: HEADS as u32,
            kv_heads: KV_HEADS as u32,
            pos: pos as u32,
            window: 0,
            m: m as u32,
            extent: extent as u32,
            scale: 1.0 / (D as f32).sqrt(),
            ring: 0,
        };
        let mut outputs = Vec::new();
        for (pipeline, groups) in [
            (&vector, [HEADS, m, 1]),
            (&tensor, [HEADS, m.div_ceil(32), 1]),
        ] {
            let out = buffer(&gpu, &vec![0x7fc1u16; m * HEADS * D]);
            futures_lite::future::block_on(compute.submit(|e| {
                dispatch(
                    e,
                    pipeline,
                    &[
                        (&*q, 0),
                        (&*k, 0),
                        (&*v, 0),
                        (&*r, 0),
                        (&*table, 0),
                        (&*sinks, 0),
                        (&*gate, 0),
                        (&*out, 0),
                    ],
                    bytemuck::bytes_of(&p),
                    groups,
                    256,
                )
            }))?;
            let mut values = vec![0u16; m * HEADS * D];
            read(&out, &mut values);
            outputs.push(values);
        }
        let mut worst = 0f32;
        for (i, (&a, &b)) in outputs[1].iter().zip(&outputs[0]).enumerate() {
            let (a, b) = (bf16::from_bits(a).to_f32(), bf16::from_bits(b).to_f32());
            assert!(
                a.is_finite() && b.is_finite(),
                "{name} output {i}: {a} against {b}"
            );
            worst = worst.max((a - b).abs());
        }
        eprintln!("{name}: worst |tensor - vector| {worst:.2e}");
        assert!(worst < 8e-3, "{name}: worst difference {worst}");
    }
    Ok(())
}
