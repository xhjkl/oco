use crate::{
    gpu::{Gpu, ReadOnly},
    tensor::{Tensor, WeightFormat},
    weights::{
        Architecture, Session,
        common::CommonWeights,
        compute::{
            AttentionPlan, Norm, Rope,
            tests::{Gate, drain},
        },
        pipelines::Builder,
        qwen::attention::QwenAttention,
    },
};
use memmap2::MmapMut;
use std::{pin::pin, sync::Arc};

use anyhow::Result;
use futures_lite::{
    StreamExt,
    future::{block_on, poll_once},
};

#[test]
fn producer_backpressure_and_receiver_closure_bound_consumed_input() -> Result<()> {
    block_on(async {
        let weights = weights(1.0)?;
        let mut unused = Session::new(&weights, 4)?;
        let inference = unused.infer(vec![1])?;
        let (producer, receiver) = inference.into_producer();
        drop(receiver);
        producer.await;
        assert_eq!(
            unused.position()?,
            0,
            "a closed receiver must prevent the first pull"
        );

        for consumed in [0, 1] {
            let mut session = Session::new(&weights, 8)?;
            let queue = session.compute.queue.clone();
            {
                let inference = session.infer(vec![1])?;
                let (producer, mut receiver) = inference.into_producer();
                let mut producer = pin!(producer);
                for round in 0..=consumed {
                    for _ in 0..4 {
                        let pending = poll_once(producer.as_mut()).await;
                        assert!(pending.is_none());
                        drain(&queue)?;
                    }
                    if round < consumed {
                        let item = poll_once(receiver.next()).await;
                        let item = item
                            .expect("one token must be queued")
                            .expect("producer remains live");
                        item?;
                    }
                }
                drop(receiver);
                producer.await;
            }
            assert_eq!(
                session.position()?,
                consumed + 1,
                "only one token may be queued or in progress"
            );
            let mut resumed = session.infer(vec![1])?;
            let next = resumed.next().await?;
            assert!(next.is_some(), "idle receiver closure leaves healthy state");
        }

        let mut session = Session::new(&weights, 532)?;
        let gate = Gate::new(&session.compute.queue);
        {
            let inference = session.infer(vec![1; 529])?;
            let (producer, receiver) = inference.into_producer();
            let mut producer = pin!(producer);
            let pending = poll_once(producer.as_mut()).await;
            assert!(pending.is_none());
            drop(receiver);
            drop(gate);
            producer.await;
        }
        assert_eq!(
            session.position()?,
            529,
            "receiver closure finishes every chunk of the active pull"
        );
        let mut resumed = session.infer(vec![1])?;
        let next = resumed.next().await?;
        assert!(next.is_some());
        Ok(())
    })
}

#[test]
fn invalid_gpu_output_is_forwarded_as_an_error_and_poisoned_state() -> Result<()> {
    block_on(async {
        let weights = weights(f32::NAN)?;
        let mut session = Session::new(&weights, 4)?;
        let inference = session.infer(vec![1])?;
        let (producer, mut receiver) = inference.into_producer();
        producer.await;
        let item = receiver.next().await;
        let error = item
            .expect("failed GPU output must not appear as EOF")
            .unwrap_err();
        assert!(error.to_string().contains("non-finite logits"));
        let closed = receiver.next().await;
        assert!(
            closed.is_none(),
            "a failed producer must close after its error"
        );
        assert!(session.position().is_err());
        let reset = session.reset();
        assert!(reset.is_err());
        Ok(())
    })
}

#[test]
fn cancelled_prefill_and_decode_remain_unusable_after_gpu_completion() -> Result<()> {
    block_on(async {
        let weights = weights(1.0)?;
        for decode in [false, true] {
            let mut session = Session::new(&weights, 4)?;
            let queue = session.compute.queue.clone();
            let mut inference = session.infer(vec![1])?;
            if decode {
                let token = inference.next().await?;
                assert!(token.is_some());
            }
            // Dropping before polling is harmless; dropping after submission is not.
            drop(inference.next());
            let gate = Gate::new(&queue);
            let pending = poll_once(inference.next()).await;
            assert!(pending.is_none(), "the GPU must still be behind the gate");
            let retry = inference.next().await;
            assert!(retry.is_err(), "an abandoned pull cannot be resumed");
            drop(inference);
            assert!(session.position().is_err());
            let reset = session.reset();
            assert!(reset.is_err());
            let rejected = session.infer(vec![1]);
            assert!(rejected.is_err());
            drop(rejected);
            drop(gate);
            drain(&queue)?;
            assert!(
                session.position().is_err(),
                "completion cannot repair abandoned state"
            );
        }
        Ok(())
    })
}

/// Build an embedding/output graph for submission lifecycle tests.
/// Empty transformer layers keep these tests independent of model arithmetic.
fn weights(output_coefficient: f32) -> Result<Architecture> {
    let gpu = Gpu::new()?;
    let mut builder = Builder::new(&gpu)?;
    let embedding = floats(&gpu, &[1.0; 128])?;
    let embedding = Tensor::bound(embedding, WeightFormat::F32, &[64, 2])?;
    let embedding = builder.embedding(embedding)?;
    let output = floats(&gpu, &[output_coefficient; 128])?;
    let output = Tensor::bound(output, WeightFormat::F32, &[64, 2])?;
    let output = builder.output(output, None)?;
    let output_norm = floats(&gpu, &[1.0; 64])?;
    let pipelines = builder.elementwise()?;
    let attention = builder.attention(64, 544)?;
    let attention = AttentionPlan::new(
        Rope {
            heads: 1,
            kv_heads: 1,
            dim: 64,
            rope_dim: 64,
            m: 0,
            pos: 0,
            q_stride: 2,
            base: 10000.0,
            factor: 1.0,
            low: 0.0,
            high: 1.0,
            scale: 1.0,
        },
        attention,
    );
    let gated = builder.gated_attention()?;
    let model = QwenAttention {
        base: CommonWeights {
            gpu,
            residency: None,
            context: 544,
            embedding,
            output,
            output_norm,
            pipelines,
            norm: Norm {
                n: 64,
                epsilon: 1e-5,
                full_layer: 0,
            },
        },
        layers: Vec::new(),
        attention,
        gated,
        ffn_width: 64,
    };
    Ok(Architecture::QwenAttention(Arc::new(model)))
}

fn floats(gpu: &Gpu, values: &[f32]) -> Result<ReadOnly> {
    let bytes = bytemuck::cast_slice(values);
    let mut mapping = MmapMut::map_anon(bytes.len().next_multiple_of(16384))?;
    mapping[..bytes.len()].copy_from_slice(bytes);
    let mapping = mapping.make_read_only()?;
    let mapping = Arc::new(mapping);
    gpu.load(&mapping, 0, bytes.len())
}
