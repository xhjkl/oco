//! Whisper speech recognition from whisper.cpp's ggml checkpoints.
//!
//! A convolutional front end and bidirectional encoder turn each 30-second log-mel window
//! into audio context; a text decoder then attends to that context and to its own output,
//! one greedy token per submission, while the host applies Whisper's logit rules. The encoder
//! and decoder reuse the dense, attention, and audio kernels of the language models.
mod load;
mod vocabulary;

use crate::{
    audio::{self, MelFilters, WHISPER_FRAMES},
    gpu::{BufferView, Gpu, arena::Plan},
    weights::{
        common::{AttentionProjections, Kernels},
        compute::{
            AttentionPlan, Bindings, Compute, Conv, EncoderLayer, GeluFfn, GeluForm, LayerNorm,
            LayerNormWeights,
        },
        operands::{Embedding, Output, Projection},
        pipelines::AudioPipelines,
        state::{AttentionScratch, CommonScratch, Kv, bind_kv_states, kv_states, product},
    },
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::{path::Path, sync::Arc};
pub use vocabulary::LANGUAGES;

/// Loaded Whisper weights and kernels, shared by any number of transcribers.
#[derive(Clone)]
pub struct Whisper {
    model: Arc<Model>,
}

pub(crate) struct Model {
    gpu: Gpu,
    kernels: Kernels,
    audio: AudioPipelines,
    filters: MelFilters,
    vocabulary: vocabulary::Vocabulary,
    encoder: Encoder,
    decoder: Decoder,
    layer_norm: LayerNorm,
    state: usize,
    mels: usize,
    audio_context: usize,
    text_context: usize,
    _weights: BufferView,
}

struct Encoder {
    convolutions: [Conv; 2],
    positions: crate::gpu::ReadOnly,
    layers: Vec<EncoderLayer>,
    post_norm: LayerNormWeights,
    attention: AttentionPlan,
}

struct Decoder {
    embedding: Embedding,
    output: Output,
    positions: crate::gpu::ReadOnly,
    layers: Vec<DecoderLayer>,
    norm: LayerNormWeights,
    attention: AttentionPlan,
    cross: AttentionPlan,
}

struct DecoderLayer {
    norm: LayerNormWeights,
    attention: AttentionProjections<Projection>,
    cross_norm: LayerNormWeights,
    cross: AttentionProjections<Projection>,
    ffn_norm: LayerNormWeights,
    ffn: GeluFfn,
}

impl Whisper {
    /// Load a whisper.cpp `ggml-*.bin` checkpoint and prepare its kernels.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            model: Arc::new(load::load(path.as_ref())?),
        })
    }

    /// Allocate the audio context, decoder history, and scratch for one transcription at a time.
    pub fn new_transcriber(&self) -> Result<Transcriber> {
        Transcriber::new(self.model.clone())
    }

    /// Language codes this checkpoint can be steered to.
    pub fn languages(&self) -> &'static [&'static str] {
        &LANGUAGES[..self.model.vocabulary.languages]
    }
}

/// Mutable state for transcribing one recording at a time: encoder scratch, the encoded
/// audio as cross-attention keys and values, and the decoder's own history.
pub struct Transcriber {
    model: Arc<Model>,
    compute: Compute,
    mel: BufferView,
    unfolded: BufferView,
    stage: BufferView,
    encoder: CommonScratch,
    encoder_attention: AttentionScratch,
    encoder_kv: Vec<Kv>,
    cross: Vec<Kv>,
    decoder: CommonScratch,
    decoder_attention: AttentionScratch,
    decoder_kv: Vec<Kv>,
    logits: Vec<f32>,
}

/// Whole transcription of one recording.
pub struct Transcript {
    pub text: String,
    /// Language code chosen or detected for the first window.
    pub language: &'static str,
    /// Text tokens produced across all windows.
    pub tokens: usize,
    pub timings: Timings,
}

/// Wall-clock split of one transcription.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    pub spectrogram: std::time::Duration,
    pub encoding: std::time::Duration,
    pub decoding: std::time::Duration,
}

impl Transcriber {
    /// Encoder rows padded to whole 64-row product tiles; the surplus rows are never read.
    fn encoder_rows(m: &Model) -> usize {
        m.audio_context.next_multiple_of(64)
    }

    fn new(model: Arc<Model>) -> Result<Self> {
        let m = &model;
        let rows = Self::encoder_rows(m);
        let frames = 2 * m.audio_context;
        let mut p = Plan::with_capacity(4 * m.decoder.layers.len() + 40);
        let unfolded = m
            .encoder
            .convolutions
            .iter()
            .map(|c| product(c.geometry.rows as usize, c.geometry.k as usize))
            .try_fold(0, |widest, n| n.map(|n| widest.max(n)))?;
        let mel = p.slot(product(frames, m.mels)?, 2)?;
        let unfolded = p.slot(unfolded, 2)?;
        let stage = p.slot(product(frames, m.state)?, 2)?;
        let encoder = CommonScratch::plan(&mut p, rows, m.state, 4 * m.state, 1, 2, 1)?;
        let encoder_attention = AttentionScratch::plan(&mut p, rows, rows, &m.encoder.attention)?;
        let encoder_kv = kv_states(&mut p, 1, rows, &m.encoder.attention)?;
        let cross = kv_states(&mut p, m.decoder.layers.len(), rows, &m.decoder.cross)?;
        let text = m.text_context;
        let vocabulary = m.vocabulary.size;
        let decoder = CommonScratch::plan(&mut p, text, m.state, 4 * m.state, 1, 2, vocabulary)?;
        let decoder_attention = AttentionScratch::plan(&mut p, text, text, &m.decoder.attention)?;
        let decoder_kv = kv_states(&mut p, m.decoder.layers.len(), text, &m.decoder.attention)?;
        let a = p.allocate(&m.gpu)?;
        Ok(Self {
            compute: Compute::new(&m.gpu, None),
            mel: a.get(mel),
            unfolded: a.get(unfolded),
            stage: a.get(stage),
            encoder: encoder.bind(&a),
            encoder_attention: encoder_attention.bind(&a),
            encoder_kv: bind_kv_states(encoder_kv, &a),
            cross: bind_kv_states(cross, &a),
            decoder: decoder.bind(&a),
            decoder_attention: decoder_attention.bind(&a),
            decoder_kv: bind_kv_states(decoder_kv, &a),
            logits: vec![0.0; vocabulary],
            model,
        })
    }

    /// Transcribe 16 kHz mono samples, greedily and without timestamps, one 30-second window
    /// at a time. `language` is an ISO code such as `"en"`; `None` detects it from the first
    /// window. Each window is decoded independently, as whisper.cpp does by default.
    pub async fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<Transcript> {
        let m = self.model.clone();
        let v = &m.vocabulary;
        // English-only checkpoints have no language or task tokens and take no language.
        let multilingual = v.languages > 0;
        let forced = match language {
            None => None,
            Some("en") if !multilingual => None,
            Some(code) if !multilingual => {
                anyhow::bail!("this model transcribes only English, not {code}")
            }
            Some(code) => Some(
                v.language(code)
                    .with_context(|| format!("language {code} is unknown to this model"))?,
            ),
        };
        let mut timings = Timings::default();
        let clock = std::time::Instant::now();
        let mel = audio::whisper_log_mel(samples, &m.filters)?;
        timings.spectrogram = clock.elapsed();
        // whisper.cpp's frame count for the recording, and its rule that a trailing window
        // shorter than a second is not worth decoding.
        let content = 1 + samples.len().saturating_sub(audio::FFT_SIZE / 2) / audio::HOP;
        let mut text = Vec::new();
        let mut tokens = 0;
        let mut detected = None;
        for start in (0..content).step_by(WHISPER_FRAMES) {
            if start > 0 && start + 100 >= content {
                break;
            }
            let clock = std::time::Instant::now();
            self.encode(&m, &mel, start).await?;
            timings.encoding += clock.elapsed();
            let clock = std::time::Instant::now();
            let mut prompt = vec![v.start_of_transcript];
            if multilingual {
                let language = match (forced, detected) {
                    (Some(token), _) => token,
                    (None, Some(token)) => token,
                    (None, None) => {
                        let token = self.detect_language(&m).await?;
                        detected = Some(token);
                        token
                    }
                };
                prompt.extend([language, v.transcribe]);
            }
            prompt.push(v.no_timestamps);
            // Like whisper.cpp, cap each window at half the text context to bound repetition loops.
            let limit = m.text_context / 2 - prompt.len();
            let mut position = 0;
            let mut pending = prompt;
            let mut generated = 0usize;
            while generated < limit {
                let logits = self.decode(&m, &pending, position).await?;
                position += pending.len();
                suppress(v, logits, generated == 0);
                let next = argmax(logits);
                if next == v.end_of_text {
                    break;
                }
                v.append(next, &mut text);
                generated += 1;
                pending = vec![next];
            }
            tokens += generated;
            timings.decoding += clock.elapsed();
        }
        let language = detected
            .or(forced)
            .and_then(|token| v.language_code(token))
            .unwrap_or("en");
        Ok(Transcript {
            text: String::from_utf8_lossy(&text).trim().to_owned(),
            language,
            tokens,
            timings,
        })
    }

    /// Run the encoder on the window starting at mel frame `start` and refresh the cross keys.
    async fn encode(&self, m: &Model, mel: &audio::Spectrogram, start: usize) -> Result<()> {
        let frames = 2 * m.audio_context;
        let window: Vec<u16> = mel
            .window(start, frames)
            .iter()
            .map(|&value| bf16::from_f32(value).to_bits())
            .collect();
        self.mel.write(0, &window);
        // The convolutions fill only the real rows; keep the padding rows defined.
        let (keys, rows, width) = (m.audio_context, Self::encoder_rows(m), m.state * 2);
        self.encoder
            .hidden
            .slice(keys * width, (rows - keys) * width)
            .zero();
        let submission = self.compute.submit(|commands| {
            let compute = Bindings {
                kernels: &m.kernels,
                scratch: &self.encoder,
                encoder: commands,
            };
            let a = &m.audio;
            let e = &m.encoder;
            let [first, second] = &e.convolutions;
            let hidden = &self.encoder.hidden;
            compute.convolve(
                a,
                first,
                self.mel.binding(),
                &self.unfolded,
                &self.stage,
                GeluForm::Tanh,
            );
            compute.convolve(
                a,
                second,
                self.stage.binding(),
                &self.unfolded,
                hidden,
                GeluForm::Tanh,
            );
            // Dense products and norms run over the padded rows; attention reads only real keys.
            let keys = m.audio_context;
            let rows = Self::encoder_rows(m);
            compute.add_rows(a, hidden.binding(), &e.positions, m.state, rows, 0, keys);
            let memory = (&self.encoder_attention, &self.encoder_kv[0]);
            compute.encoder_layers(
                a,
                &e.layers,
                &e.post_norm,
                m.layer_norm,
                &e.attention,
                memory,
                rows,
                keys,
            );
            // Every decoder layer's cross-attention keys and values come from the same encoding.
            for (layer, cross) in m.decoder.layers.iter().zip(&self.cross) {
                let memory = (&self.encoder_attention, cross);
                compute.store_cross_kv(
                    &layer.cross.k,
                    &layer.cross.v,
                    &m.decoder.cross,
                    memory,
                    keys,
                );
            }
        });
        submission.await
    }

    /// Pick the most likely language token after the start-of-transcript marker.
    async fn detect_language(&mut self, m: &Model) -> Result<u32> {
        let v = &m.vocabulary;
        let logits = self.decode(m, &[v.start_of_transcript], 0).await?;
        let first = v.start_of_transcript as usize + 1;
        let best = logits[first..first + v.languages]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| first + index)
            .context("no language tokens")?;
        Ok(best as u32)
    }

    /// Feed `tokens` at `position` through the decoder and return the last row's logits.
    async fn decode(&mut self, m: &Model, tokens: &[u32], position: usize) -> Result<&mut [f32]> {
        let rows = tokens.len();
        ensure!(
            rows > 0 && position + rows <= m.text_context,
            "decoder context exhausted"
        );
        let s = &self.decoder;
        s.input.write(0, tokens);
        let submission = self.compute.submit(|commands| {
            let compute = Bindings {
                kernels: &m.kernels,
                scratch: s,
                encoder: commands,
            };
            let a = &m.audio;
            let d = &m.decoder;
            let ln = m.layer_norm;
            let (hidden, projected, norm) = (s.hidden.binding(), s.projected.binding(), &s.norm);
            compute.embed(&d.embedding, &s.input, rows, s.next.binding());
            compute.add_rows(
                a,
                hidden,
                &d.positions,
                m.state,
                rows,
                position,
                m.text_context,
            );
            compute.layer_norm(a, &d.layers[0].norm, ln, hidden, norm, rows);
            for (index, layer) in d.layers.iter().enumerate() {
                let memory = (&self.decoder_attention, &self.decoder_kv[index]);
                let positions = position..position + rows;
                compute.plain_attention(&layer.attention, &d.attention, memory, positions, 0..rows);
                compute.add_layer_norm(a, &layer.cross_norm, ln, hidden, projected, norm, rows);
                let memory = (&self.decoder_attention, &self.cross[index]);
                let keys = m.audio_context;
                compute.cross_attention(
                    &layer.cross.q,
                    &layer.cross.out,
                    &d.cross,
                    memory,
                    keys,
                    rows,
                );
                compute.add_layer_norm(a, &layer.ffn_norm, ln, hidden, projected, norm, rows);
                compute.gelu_feed_forward(a, &layer.ffn, rows);
                match d.layers.get(index + 1) {
                    Some(next) => {
                        compute.add_layer_norm(a, &next.norm, ln, hidden, projected, norm, rows)
                    }
                    None => {
                        // Only the last row predicts; normalize just that one for the output head.
                        compute.add(&s.projected, hidden, rows);
                        let last = s.hidden.at((rows - 1) * m.state * 2);
                        compute.layer_norm(a, &d.norm, ln, last, norm, 1);
                    }
                }
            }
            compute.output(&d.output);
        });
        submission.await?;
        s.logits.read(&mut self.logits);
        Ok(&mut self.logits)
    }
}

/// Whisper's greedy decoding rules without timestamps: never emit control tokens, and never
/// start a transcript with silence or a blank.
fn suppress(v: &vocabulary::Vocabulary, logits: &mut [f32], initial: bool) {
    let mut forbid = |token: u32| {
        if let Some(logit) = logits.get_mut(token as usize) {
            *logit = f32::NEG_INFINITY;
        }
    };
    if initial {
        forbid(v.end_of_text);
        if let Some(space) = v.space {
            forbid(space);
        }
    }
    for token in [
        v.start_of_transcript,
        v.no_speech,
        v.speaker_turn,
        v.translate,
        v.transcribe,
        v.previous,
        v.no_timestamps,
    ] {
        forbid(token);
    }
    for index in 0..v.languages as u32 {
        forbid(v.start_of_transcript + 1 + index);
    }
    for logit in &mut logits[v.timestamps as usize..] {
        *logit = f32::NEG_INFINITY;
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .filter(|(_, l)| l.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(index, _)| index as u32)
}
