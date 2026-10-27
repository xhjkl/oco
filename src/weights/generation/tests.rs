use crate::{
    gpu::{Gpu, ReadOnly},
    tensor::{Tensor, WeightFormat},
    weights::{
        Architecture, Session,
        common::{CommonWeights, Kernels},
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
fn audio_embedding_preserves_position_and_reset_forgets_unconsumed_features() -> Result<()> {
    block_on(async {
        let weights = audio_weights()?;
        let samples = vec![0.1f32; 201];
        for consumed in [0, 1] {
            let mut session = Session::new(&weights, 32)?;
            if consumed > 0 {
                let mut inference = session.infer(vec![1])?;
                let token = inference.next().await?;
                assert!(token.is_some());
            }
            let invalid = session.embed_audio(&[]).await;
            assert!(invalid.is_err());
            assert_eq!(session.position()?, consumed);
            let ids = session.embed_audio(&samples).await?;
            assert_eq!(ids.len(), 13);
            assert_eq!(session.position()?, consumed);
            let more = session.embed_audio(&samples).await?;
            assert_eq!(more[0], ids[0] + 13);
            let overflow = session.embed_audio(&samples).await;
            assert!(overflow.is_err());
            assert_eq!(session.position()?, consumed);
            session.reset()?;
            let stale = session.infer(vec![ids[0]]);
            assert!(stale.is_err());
            drop(stale);
            let fresh = session.embed_audio(&samples).await?;
            assert_eq!(fresh, ids);
        }
        Ok(())
    })
}

#[test]
fn cancelled_audio_embedding_remains_unusable_after_gpu_completion() -> Result<()> {
    block_on(async {
        let weights = audio_weights()?;
        let samples = vec![0.1f32; 201];
        for consumed in [0, 1] {
            let mut session = Session::new(&weights, 32)?;
            if consumed > 0 {
                let mut inference = session.infer(vec![1])?;
                let token = inference.next().await?;
                assert!(token.is_some());
            }
            drop(session.embed_audio(&samples));
            assert_eq!(session.position()?, consumed);
            let queue = session.compute.queue.clone();
            let gate = Gate::new(&queue);
            let pending = poll_once(session.embed_audio(&samples)).await;
            assert!(pending.is_none());
            assert!(session.position().is_err());
            let reset = session.reset();
            assert!(reset.is_err());
            let retry = session.embed_audio(&samples).await;
            assert!(retry.is_err());
            drop(gate);
            drain(&queue)?;
            assert!(session.position().is_err());
            let retry = session.embed_audio(&samples).await;
            assert!(retry.is_err());
        }
        Ok(())
    })
}

/// Attach a small encoder to the existing submission lifecycle graph.
fn audio_weights() -> Result<Architecture> {
    let mut weights = weights(1.0)?;
    let Architecture::QwenAttention(model) = &mut weights else {
        unreachable!()
    };
    let model = Arc::get_mut(model).unwrap();
    let encoder = crate::weights::qwen::audio::tests::encoder(&model.base.gpu, 64)?;
    model.audio = Some(encoder);
    Ok(weights)
}

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
            kernels: Kernels {
                pipelines,
                norm: Norm {
                    n: 64,
                    epsilon: 1e-5,
                    full_layer: 0,
                },
            },
        },
        layers: Vec::new(),
        attention,
        gated,
        ffn_width: 64,
        audio: None,
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
