use super::GUARD;
use crate::{
    gpu::{
        BufferRef, Gpu, dispatch, dispatch_concurrent,
        tests::{buffer, read},
    },
    tensor::WeightFormat,
    weights::{
        compute::{Compute, Rope},
        pipelines::Builder,
    },
};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use half::{bf16, f16};

#[path = "attention_split.rs"]
mod split;
#[path = "attention_wide.rs"]
mod wide;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttentionParams {
    heads: u32,
    kv_heads: u32,
    dim: u32,
    pos: u32,
    window: u32,
    sinks: u32,
    scale: f32,
    m: u32,
}

/// Preserve KV writes when only the last query is consumed or no query is consumed.
#[test]
fn gpu_selected_rope_preserves_queries_cache_and_guards() -> Result<()> {
    let gpu = Gpu::new()?;
    let compute = Compute::new(&gpu, None);
    let mut builder = Builder::new(&gpu)?;
    let pipeline = builder.get("rope", WeightFormat::F32, 0)?;
    const MARKER: u16 = 0x7fc1;
    let heads = 4;
    let kv_heads = 2;
    let guarded = |values: &[u16]| {
        let mut storage = vec![MARKER; GUARD + values.len() + GUARD];
        storage[GUARD..GUARD + values.len()].copy_from_slice(values);
        storage
    };
    let value = |i: usize, salt: usize| {
        bf16::from_f32(((i * 37 + salt) % 251) as f32 / 64.0 - 2.0).to_bits()
    };
    for (dim, stride, position, batch) in [(64, 1, 19, 7), (256, 2, 257, 8)] {
        let query_width = heads * dim;
        let key_width = kv_heads * dim;
        let queries: Vec<_> = (0..batch * query_width * stride)
            .map(|i| value(i, 13))
            .collect();
        let keys: Vec<_> = (0..batch * key_width).map(|i| value(i, 73)).collect();
        let values: Vec<_> = (0..batch * key_width).map(|i| value(i, 151)).collect();
        let original_queries = guarded(&queries);
        let original_keys = guarded(&keys);
        let original_values = guarded(&values);
        let q = buffer(&gpu, &original_queries);
        let k = buffer(&gpu, &original_keys);
        let v = buffer(&gpu, &original_values);
        let query_storage = vec![MARKER; GUARD + batch * query_width + GUARD];
        let cache_storage = vec![MARKER; GUARD + (position + batch + 3) * key_width + GUARD];
        let mut full_queries = query_storage.clone();
        let mut full_keys = cache_storage.clone();
        let mut full_values = cache_storage.clone();
        let qo = buffer(&gpu, &full_queries);
        let kc = buffer(&gpu, &full_keys);
        let vc = buffer(&gpu, &full_values);
        let rope = Rope {
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            dim: dim as u32,
            rope_dim: (if stride == 1 { dim } else { dim / 2 }) as u32,
            m: batch as u32,
            pos: position as u32,
            q_stride: stride as u32,
            base: 10000.0,
            factor: 2.0,
            low: 2.0,
            high: 8.0,
            scale: 0.875,
        };
        let submission = compute.submit(|encoder| {
            dispatch(
                encoder,
                &pipeline,
                &[
                    (&q, GUARD * 2),
                    (&k, GUARD * 2),
                    (&v, GUARD * 2),
                    (&qo, GUARD * 2),
                    (&kc, GUARD * 2),
                    (&vc, GUARD * 2),
                ],
                bytemuck::bytes_of(&rope),
                [(batch * (query_width + 2 * key_width)).div_ceil(256), 1, 1],
                256,
            );
        });
        futures_lite::future::block_on(submission)?;
        read(&qo, &mut full_queries);
        read(&kc, &mut full_keys);
        read(&vc, &mut full_values);
        let cache_begin = GUARD + position * key_width;
        let cache_end = cache_begin + batch * key_width;
        for (storage, begin, end, cache) in [
            (&full_queries, GUARD, GUARD + batch * query_width, false),
            (&full_keys, cache_begin, cache_end, true),
            (&full_values, cache_begin, cache_end, true),
        ] {
            assert!(storage[..begin].iter().all(|&word| word == MARKER));
            assert!(storage[end..].iter().all(|&word| word == MARKER));
            assert!(storage[begin..end].iter().all(|&word| if cache {
                f16::from_bits(word).is_finite()
            } else {
                bf16::from_bits(word).is_finite()
            }));
        }
        let expected_values: Vec<_> = values
            .iter()
            .map(|&word| f16::from_f32(bf16::from_bits(word).to_f32()).to_bits())
            .collect();
        assert_eq!(
            &full_values[cache_begin..cache_end],
            &expected_values,
            "full value cache"
        );
        for selection in [batch - 1..batch, 0..0] {
            let context =
                format!("dim={dim}, stride={stride}, pos={position}, selection={selection:?}");
            let source_begin = selection.start * query_width * stride;
            let source_end = selection.end * query_width * stride;
            let mut selected_input = vec![MARKER; original_queries.len()];
            selected_input[GUARD..GUARD + source_end - source_begin]
                .copy_from_slice(&queries[source_begin..source_end]);
            let selected_q = buffer(&gpu, &selected_input);
            // Poisoning fresh outputs for every selection exposes omitted writes, including no-ops.
            let mut selected_queries = query_storage.clone();
            let mut selected_keys = cache_storage.clone();
            let mut selected_values = cache_storage.clone();
            let qo = buffer(&gpu, &selected_queries);
            let kc = buffer(&gpu, &selected_keys);
            let vc = buffer(&gpu, &selected_values);
            let buffers: [(&BufferRef, usize); 6] = [
                (&selected_q, GUARD * 2),
                (&k, GUARD * 2),
                (&v, GUARD * 2),
                (&qo, GUARD * 2),
                (&kc, GUARD * 2),
                (&vc, GUARD * 2),
            ];
            let count = selection.len() * query_width;
            let submission = compute.submit(|encoder| {
                let keys = Rope { heads: 0, ..rope };
                dispatch(
                    encoder,
                    &pipeline,
                    &buffers,
                    bytemuck::bytes_of(&keys),
                    [(2 * batch * key_width).div_ceil(256), 1, 1],
                    256,
                );
                if selection.is_empty() {
                    return;
                }
                let queries = Rope {
                    kv_heads: 0,
                    m: selection.len() as u32,
                    pos: (position + selection.start) as u32,
                    ..rope
                };
                // Writing disjoint query/cache outputs through the same compiled arithmetic.
                dispatch_concurrent(
                    encoder,
                    &pipeline,
                    &buffers,
                    bytemuck::bytes_of(&queries),
                    [count.div_ceil(256), 1, 1],
                    256,
                );
            });
            futures_lite::future::block_on(submission)?;
            read(&qo, &mut selected_queries);
            read(&kc, &mut selected_keys);
            read(&vc, &mut selected_values);
            let begin = GUARD + selection.start * query_width;
            assert_eq!(
                &selected_queries[GUARD..GUARD + count],
                &full_queries[begin..begin + count],
                "query output: {context}"
            );
            assert!(
                selected_queries[..GUARD].iter().all(|&word| word == MARKER),
                "{context}"
            );
            assert!(
                selected_queries[GUARD + count..]
                    .iter()
                    .all(|&word| word == MARKER),
                "{context}"
            );
            assert_eq!(&selected_keys, &full_keys, "key cache: {context}");
            assert_eq!(&selected_values, &full_values, "value cache: {context}");
            let mut after = vec![0; selected_input.len()];
            read(&selected_q, &mut after);
            assert_eq!(&after, &selected_input, "query input mutation: {context}");
        }
        for (input, original) in [
            (&q, &original_queries),
            (&k, &original_keys),
            (&v, &original_values),
        ] {
            let mut after = vec![0; original.len()];
            read(input, &mut after);
            assert_eq!(
                &after, original,
                "input mutation: dim={dim}, stride={stride}"
            );
        }
    }
    Ok(())
}
