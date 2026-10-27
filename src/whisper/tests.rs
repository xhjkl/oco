use super::*;
use crate::{
    tensor::{Tensor, WeightFormat},
    weights::{
        compute::{
            Im2col, Rope,
            tests::{Gate, command_queue, drain},
        },
        pipelines::Builder,
    },
};
use futures_lite::future::{block_on, poll_once};
use objc2::Message;
use std::pin::pin;

#[test]
fn completed_transcription_and_invalid_input_leave_transcriber_reusable() -> Result<()> {
    block_on(async {
        let mut transcriber = transcriber()?;
        let samples = vec![0.1f32; 201];
        let invalid = transcriber.transcribe(&samples, Some("fr")).await;
        assert!(invalid.is_err());
        let empty = transcriber.transcribe(&[], None).await;
        assert!(empty.is_err());
        let first = transcriber.transcribe(&samples, None).await?;
        let second = transcriber.transcribe(&samples, Some("en")).await?;
        assert!(first.tokens > 0);
        assert_eq!(first.text, second.text);
        assert_eq!(first.tokens, second.tokens);
        assert_eq!(second.language, "en");
        Ok(())
    })
}

#[test]
fn cancelled_encoder_and_decoder_leave_transcriber_unusable() -> Result<()> {
    block_on(async {
        for decode in [false, true] {
            let mut transcriber = transcriber()?;
            let samples = vec![0.1f32; 201];
            drop(transcriber.transcribe(&samples, None));
            assert!(transcriber.usable);
            let queue = command_queue(&transcriber.compute);
            let queue = queue.retain();
            let gate = {
                let mut transcription = pin!(transcriber.transcribe(&samples, None));
                let gate = Gate::new(&queue);
                let pending = poll_once(transcription.as_mut()).await;
                assert!(pending.is_none());
                if decode {
                    drop(gate);
                    drain(&queue)?;
                    let gate = Gate::new(&queue);
                    let pending = poll_once(transcription.as_mut()).await;
                    assert!(pending.is_none());
                    gate
                } else {
                    gate
                }
            };
            assert!(!transcriber.usable);
            let retry = transcriber.transcribe(&samples, None).await;
            assert!(retry.is_err());
            drop(gate);
            drain(&queue)?;
            let retry = transcriber.transcribe(&samples, None).await;
            assert!(retry.is_err());
        }
        Ok(())
    })
}

/// Build a small encoder and one decoder block, keeping lifecycle tests independent of checkpoints.
fn transcriber() -> Result<Transcriber> {
    let gpu = Gpu::new()?;
    let mut builder = Builder::new(&gpu)?;
    let projection = |builder: &mut Builder, input, output| -> Result<Projection> {
        let storage = gpu.upload(&vec![0u8; input * output * 2]);
        let tensor = Tensor::bound(storage, WeightFormat::F16, &[input, output])?;
        let matrix = builder.dense(tensor)?;
        Ok(Projection { matrix, bias: None })
    };
    let norm = || {
        let weight = gpu.upload(bytemuck::cast_slice(&[1.0f32; 64]));
        let bias = gpu.upload(&[0u8; 64 * 4]);
        LayerNormWeights { weight, bias }
    };
    let attention = |builder: &mut Builder| -> Result<AttentionProjections<Projection>> {
        let q = projection(builder, 64, 64)?;
        let k = projection(builder, 64, 64)?;
        let v = projection(builder, 64, 64)?;
        let out = projection(builder, 64, 64)?;
        Ok(AttentionProjections { q, k, v, out })
    };
    let first = projection(&mut builder, 64, 64)?;
    let second = projection(&mut builder, 64, 64)?;
    let convolutions = [
        Conv {
            weights: first,
            geometry: Im2col::new(1, 64, 1, (1, 1), 1, (0, 0)),
        },
        Conv {
            weights: second,
            geometry: Im2col::new(64, 64, 1, (1, 1), 2, (0, 0)),
        },
    ];
    let positions = gpu.upload(&vec![0u8; 32 * 64 * 4]);
    let post_norm = norm();
    let pipelines = builder.encoder_attention(64, 32)?;
    let encoder = Encoder {
        convolutions,
        positions,
        layers: Vec::new(),
        post_norm,
        attention: AttentionPlan::new(Rope::unrotated(1, 1, 64), pipelines),
    };
    let vocabulary = vocabulary::Vocabulary::new(Vec::new(), 51_864)?;
    let table = gpu.upload(&vec![0u8; 64 * vocabulary.size * 2]);
    let embedding = Tensor::bound(table.clone(), WeightFormat::F16, &[64, vocabulary.size])?;
    let embedding = builder.embedding(embedding)?;
    let output = Tensor::bound(table, WeightFormat::F16, &[64, vocabulary.size])?;
    let output = builder.output(output, None)?;
    let norm_weights = norm();
    let self_attention = attention(&mut builder)?;
    let cross_norm = norm();
    let cross = attention(&mut builder)?;
    let ffn_norm = norm();
    let up = projection(&mut builder, 64, 256)?;
    let down = projection(&mut builder, 256, 64)?;
    let ffn = GeluFfn {
        up,
        down,
        form: GeluForm::Tanh,
    };
    let layer = DecoderLayer {
        norm: norm_weights,
        attention: self_attention,
        cross_norm,
        cross,
        ffn_norm,
        ffn,
    };
    let positions = gpu.upload(&vec![0u8; 16 * 64 * 4]);
    let norm = norm();
    let self_attention = builder.attention(64, 16)?;
    let cross = builder.attention(64, 32)?;
    let decoder = Decoder {
        embedding,
        output,
        positions,
        layers: vec![layer],
        norm,
        attention: AttentionPlan::new(Rope::unrotated(1, 1, 64), self_attention),
        cross: AttentionPlan::new(Rope::unrotated(1, 1, 64), cross),
    };
    let kernels = builder.kernels(64, 1e-5, false)?;
    let audio = builder.audio()?;
    let buffer = gpu.new_buffer(4);
    let model = Model {
        gpu,
        kernels,
        audio,
        filters: MelFilters::slaney(1),
        vocabulary,
        encoder,
        decoder,
        layer_norm: LayerNorm {
            n: 64,
            epsilon: 1e-5,
        },
        state: 64,
        mels: 1,
        audio_context: 32,
        text_context: 16,
        _weights: BufferView::new(&buffer, 0, 4),
    };
    Transcriber::new(Arc::new(model))
}
