//! Qwen3-ASR audio encoder, loaded from the companion `mmproj` GGUF: three strided
//! convolutions over 100-frame mel chunks, a ViT-style transformer attending within one
//! window of up to eight chunks, and an MLP projector onto the language model's width.
//! Its output rows are spliced into the token stream by IDs past the vocabulary.
use crate::{
    audio::{self, MelFilters},
    gguf::Gguf,
    gpu::{
        BufferView, Gpu, ReadOnly, ResidentWeights,
        arena::{Arena, Plan, Slot},
    },
    tensor::WeightFormat,
    weights::{
        common::{AttentionProjections, Kernels},
        compute::{
            AttentionPlan, Bindings, Conv, EncoderLayer, GeluFfn, GeluForm, Im2col, LayerNorm,
            LayerNormWeights, Rope, half_bits, repacked_matrix,
        },
        load::{Import, Parameters, Selection, import},
        operands::Projection,
        pipelines::AudioPipelines,
        state::{AttentionScratch, CommonScratch, Kv, bind_kv_states, kv_states, product},
    },
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::sync::Arc;

#[cfg(test)]
pub(crate) mod tests;

/// Mel frames per convolution chunk and per attention window.
pub const CHUNK_FRAMES: usize = 100;
pub const WINDOW_FRAMES: usize = 800;
pub const WINDOW_CHUNKS: usize = WINDOW_FRAMES / CHUNK_FRAMES;
/// Convolution channels, taps per axis, and the tokens each chunk collapses to after three
/// stride-2 stages.
const CHANNELS: usize = 480;
const TAPS: usize = 3;
pub const CHUNK_TOKENS: usize = 13;
const WINDOW_TOKENS: usize = WINDOW_CHUNKS * CHUNK_TOKENS;

pub struct AudioEncoder {
    pub kernels: Kernels,
    pub pipelines: AudioPipelines,
    pub attention: AttentionPlan,
    pub layer_norm: LayerNorm,
    pub convolutions: [Conv; 3],
    /// Projection of each chunk token's `(mel position, channel)` features onto the encoder width.
    pub conv_out: Projection,
    pub positions: ReadOnly,
    pub layers: Vec<EncoderLayer>,
    pub post_norm: LayerNormWeights,
    pub projector: GeluFfn,
    pub mels: usize,
    pub width: usize,
    pub ffn_width: usize,
    /// Width of the language model's embeddings, which the projector produces.
    pub output: usize,
    _residency: Arc<ResidentWeights>,
}

/// Session scratch for one window: mel input, convolution stages, and transformer rows.
pub struct AudioScratch<S = BufferView> {
    pub mel: S,
    pub unfolded: S,
    pub stage: [S; 2],
    /// Chunk tokens of the whole window after the last convolution, `(token, mel, channel)` order.
    pub tokens: S,
    pub common: CommonScratch<S>,
    pub attention: AttentionScratch<S>,
    pub kv: Vec<Kv<S>>,
}

impl AudioScratch<Slot> {
    pub fn plan(p: &mut Plan, e: &AudioEncoder) -> Result<Self> {
        let unfolded = e
            .convolutions
            .iter()
            .map(|c| product(c.geometry.rows as usize, c.geometry.k as usize))
            .try_fold(0, |widest, n| n.map(|n| widest.max(n)))?;
        let mut stage = |c: &Conv| p.slot(product(c.geometry.rows as usize, CHANNELS)?, 2);
        let stage = [stage(&e.convolutions[0])?, stage(&e.convolutions[1])?];
        Ok(Self {
            mel: p.slot(product(WINDOW_FRAMES, e.mels)?, 2)?,
            unfolded: p.slot(unfolded, 2)?,
            stage,
            tokens: p.slot(product(WINDOW_TOKENS, e.conv_out.matrix.input)?, 2)?,
            common: CommonScratch::plan(p, WINDOW_TOKENS, e.width, e.ffn_width, 1, 2, 1)?,
            attention: AttentionScratch::plan(p, WINDOW_TOKENS, WINDOW_TOKENS, &e.attention)?,
            kv: kv_states(p, 1, WINDOW_TOKENS, &e.attention)?,
        })
    }

    pub fn bind(self, a: &Arena) -> AudioScratch {
        AudioScratch {
            mel: a.get(self.mel),
            unfolded: a.get(self.unfolded),
            stage: self.stage.map(|slot| a.get(slot)),
            tokens: a.get(self.tokens),
            common: self.common.bind(a),
            attention: self.attention.bind(a),
            kv: bind_kv_states(self.kv, a),
        }
    }
}

/// One attention window's mel frames in frame-major BF16, zero-padded to whole chunks.
pub struct Window {
    pub chunks: usize,
    pub mel: Vec<u16>,
}

impl Window {
    pub fn tokens(&self) -> usize {
        self.chunks * CHUNK_TOKENS
    }
}

impl AudioEncoder {
    /// Split 16 kHz mono audio into the windows the encoder consumes one at a time.
    pub fn windows(&self, samples: &[f32]) -> Result<Vec<Window>> {
        let filters = MelFilters::slaney(self.mels);
        let mel = audio::centered_log_mel(samples, &filters)?;
        let mut windows = Vec::with_capacity(mel.frames.div_ceil(WINDOW_FRAMES));
        for start in (0..mel.frames).step_by(WINDOW_FRAMES) {
            let frames = WINDOW_FRAMES.min(mel.frames - start);
            let chunks = frames.div_ceil(CHUNK_FRAMES);
            let values = mel.window(start, chunks * CHUNK_FRAMES);
            windows.push(Window {
                chunks,
                mel: values
                    .iter()
                    .map(|&v| bf16::from_f32(v).to_bits())
                    .collect(),
            });
        }
        Ok(windows)
    }

    /// Encode the window already written to `s.mel` and project its tokens into `features`.
    pub fn encode(
        &self,
        compute: &Bindings<'_>,
        s: &AudioScratch,
        chunks: usize,
        features: &BufferView,
    ) {
        debug_assert!((1..=WINDOW_CHUNKS).contains(&chunks));
        let a = &self.pipelines;
        let token_bytes = self.conv_out.matrix.input * 2;
        let [first, second, third] = &self.convolutions;
        for chunk in 0..chunks {
            let mel = s.mel.at(chunk * CHUNK_FRAMES * self.mels * 2);
            compute.convolve(a, first, mel, &s.unfolded, &s.stage[0], GeluForm::Erf);
            let staged = s.stage[0].binding();
            compute.convolve(a, second, staged, &s.unfolded, &s.stage[1], GeluForm::Erf);
            let tokens = s.tokens.slice(
                chunk * CHUNK_TOKENS * token_bytes,
                CHUNK_TOKENS * token_bytes,
            );
            let staged = s.stage[1].binding();
            compute.convolve(a, third, staged, &s.unfolded, &tokens, GeluForm::Erf);
        }
        let rows = chunks * CHUNK_TOKENS;
        let hidden = &s.common.hidden;
        self.conv_out
            .apply(compute, s.tokens.binding(), hidden, rows);
        // Positions restart in every chunk.
        let n = self.width;
        compute.add_rows(
            a,
            hidden.binding(),
            &self.positions,
            n,
            rows,
            0,
            CHUNK_TOKENS,
        );
        let memory = (&s.attention, &s.kv[0]);
        let (layers, post) = (&self.layers, &self.post_norm);
        compute.encoder_layers(
            a,
            layers,
            post,
            self.layer_norm,
            &self.attention,
            memory,
            rows,
            rows,
        );
        compute.gelu_project(a, &self.projector, rows, features);
    }
}

/// Metadata the audio encoder's GGUF must supply.
pub const METADATA: &[&str] = &[
    "general.architecture",
    "general.alignment",
    "clip.has_audio_encoder",
    "clip.audio.projector_type",
    "clip.audio.embedding_length",
    "clip.audio.feed_forward_length",
    "clip.audio.block_count",
    "clip.audio.attention.head_count",
    "clip.audio.num_mel_bins",
    "clip.audio.attention.layer_norm_epsilon",
    "clip.audio.projection_dim",
];

type Selected<'a> = crate::gguf::Tensor<'a>;

fn layer_norm<'a>(s: &mut Selection<'a>, name: &str, width: usize) -> Result<[Selected<'a>; 2]> {
    Ok([
        s.float(&format!("{name}.weight"), &[width])?,
        s.float(&format!("{name}.bias"), &[width])?,
    ])
}

fn bind_layer_norm(i: &mut Import, [weight, bias]: [Selected<'_>; 2]) -> Result<LayerNormWeights> {
    Ok(LayerNormWeights {
        weight: i.binding(weight)?,
        bias: i.binding(bias)?,
    })
}

/// Coefficients of an F16 tensor read straight from the mapped file.
fn mapped_f16(file: &Gguf, t: &Selected<'_>) -> Result<Vec<u16>> {
    ensure!(
        t.kind == WeightFormat::F16,
        "convolution weights must be F16"
    );
    Ok(half_bits(&file.map[t.offset..t.offset + t.bytes]))
}

/// Build one 3×3 stride-2 convolution stage over `(channels, frames, mels)` input.
fn convolution(
    file: &Gguf,
    directory: &std::collections::HashMap<&str, crate::gguf::Entry>,
    i: &mut Import,
    gpu: &Gpu,
    index: usize,
    bias: Selected<'_>,
    (channels, height, width): (usize, usize, usize),
) -> Result<Conv> {
    let name = format!("a.conv2d.{index}.weight");
    let entry = directory
        .get(name.as_str())
        .with_context(|| format!("missing tensor {name}"))?;
    let t = entry.select()?;
    ensure!(
        t.shape == [TAPS, TAPS, channels, CHANNELS],
        "unexpected convolution shape for {name}"
    );
    let source = mapped_f16(file, &t)?;
    let bias = i.binding(bias)?;
    let geometry = Im2col::new(channels, height, width, (TAPS, TAPS), 2, (1, 1));
    // The checkpoint stores (out, in, mel tap, time tap); the unfold emits (time tap, mel tap, in).
    Conv::repacked(
        gpu,
        &mut i.builder,
        geometry,
        CHANNELS,
        bias,
        |out, tap, c| {
            let (time_tap, mel_tap) = (tap / TAPS, tap % TAPS);
            source[((out * channels + c) * TAPS + mel_tap) * TAPS + time_tap]
        },
    )
}

/// Load the encoder whose projector produces `output`-wide rows for the language model.
pub fn load(file: &Gguf, gpu: &Gpu, output: usize) -> Result<AudioEncoder> {
    ensure!(
        file.text("general.architecture")? == "clip" && file.boolean("clip.has_audio_encoder")?,
        "not an audio encoder GGUF"
    );
    let projector = file.text("clip.audio.projector_type")?;
    ensure!(
        projector == "qwen3a",
        "unsupported audio projector {projector}"
    );
    let p = Parameters::new(file, "clip.audio");
    let width = p.integer("embedding_length")?;
    let ffn_width = p.integer("feed_forward_length")?;
    let count = p.integer("block_count")?;
    let heads = p.integer("attention.head_count")?;
    let mels = p.integer("num_mel_bins")?;
    let epsilon = p.number("attention.layer_norm_epsilon")? as f32;
    let projection = p.integer("projection_dim")?;
    ensure!(
        projection == output,
        "audio projector width does not match the language model"
    );
    ensure!(
        width.is_multiple_of(64)
            && ffn_width.is_multiple_of(64)
            && heads > 0
            && width.is_multiple_of(heads),
        "unsupported audio encoder geometry"
    );
    let dim = width / heads;
    ensure!(
        matches!(dim, 64 | 128 | 256),
        "unsupported audio head width {dim}"
    );
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "invalid layer norm epsilon"
    );
    ensure!((1..=128).contains(&count), "unsupported audio layer count");
    let directory = file.directory()?;
    let mut s = Selection::new(&directory);
    let positions = s.tensor("a.position_embd.weight")?;
    ensure!(
        positions.kind == WeightFormat::F32
            && positions.shape.len() == 2
            && positions.shape[0] == width
            && positions.shape[1] >= CHUNK_TOKENS,
        "invalid audio position table"
    );
    let conv_out = s.tensor("a.conv_out.weight")?;
    let post_norm = layer_norm(&mut s, "a.post_ln", width)?;
    let mut layers = Vec::with_capacity(count);
    for index in 0..count {
        let prefix = format!("a.blk.{index}");
        let norm = layer_norm(&mut s, &format!("{prefix}.ln1"), width)?;
        let q = s.biased(&format!("{prefix}.attn_q"), width, width)?;
        let k = s.biased(&format!("{prefix}.attn_k"), width, width)?;
        let v = s.biased(&format!("{prefix}.attn_v"), width, width)?;
        let out = s.biased(&format!("{prefix}.attn_out"), width, width)?;
        let ffn_norm = layer_norm(&mut s, &format!("{prefix}.ln2"), width)?;
        let up = s.biased(&format!("{prefix}.ffn_up"), width, ffn_width)?;
        let down = s.biased(&format!("{prefix}.ffn_down"), ffn_width, width)?;
        layers.push((norm, [q, k, v, out], ffn_norm, up, down));
    }
    let mm1 = s.biased("mm.a.mlp.1", width, width)?;
    let mm2 = s.biased("mm.a.mlp.2", width, output)?;
    let mut biases = Vec::with_capacity(3);
    for index in 1..=3 {
        biases.push(s.float(&format!("a.conv2d.{index}.bias"), &[1, 1, CHANNELS])?);
    }
    let mut i = import(file, s.spans, gpu)?;
    // Convolutions read their coefficients from the mapping and repack them into new storage.
    let mut biases = biases.into_iter();
    let mut stage = |i: &mut Import, index, shape| {
        let bias = biases.next().context("missing convolution bias")?;
        convolution(file, &directory, i, gpu, index, bias, shape)
    };
    let first = stage(&mut i, 1, (1, CHUNK_FRAMES, mels))?;
    let (height, width_out) = first.geometry.output();
    let second = stage(&mut i, 2, (CHANNELS, height, width_out))?;
    let (height, width_out) = second.geometry.output();
    let third = stage(&mut i, 3, (CHANNELS, height, width_out))?;
    let (tokens, mel_positions) = third.geometry.output();
    let channels = CHANNELS;
    ensure!(
        tokens == CHUNK_TOKENS,
        "unexpected chunk token count {tokens}"
    );
    let features = mel_positions * channels;
    ensure!(
        conv_out.shape == [features, width],
        "unexpected conv_out shape {:?}",
        conv_out.shape
    );
    // conv_out consumes (channel, mel) features; the stages emit (mel, channel), so permute columns.
    let source = mapped_f16(file, &conv_out)?;
    i.binding(conv_out)?;
    let matrix = repacked_matrix(gpu, &mut i.builder, features, width, |out, k| {
        let (mel, c) = (k / channels, k % channels);
        source[out * features + c * mel_positions + mel]
    })?;
    let conv_out = Projection { matrix, bias: None };
    let positions = i.binding(positions)?;
    let post_norm = bind_layer_norm(&mut i, post_norm)?;
    let mut bound = Vec::with_capacity(count);
    for (norm, [q, k, v, out], ffn_norm, up, down) in layers {
        bound.push(EncoderLayer {
            norm: bind_layer_norm(&mut i, norm)?,
            attention: AttentionProjections {
                q: i.biased(q)?,
                k: i.biased(k)?,
                v: i.biased(v)?,
                out: i.biased(out)?,
            },
            ffn_norm: bind_layer_norm(&mut i, ffn_norm)?,
            ffn: GeluFfn {
                up: i.biased(up)?,
                down: i.biased(down)?,
                form: GeluForm::Erf,
            },
        });
    }
    let projector = GeluFfn {
        up: i.biased(mm1)?,
        down: i.biased(mm2)?,
        form: GeluForm::Erf,
    };
    let rope = Rope::unrotated(heads, heads, dim);
    let attention = AttentionPlan::new(rope, i.builder.encoder_attention(dim, WINDOW_TOKENS)?);
    let kernels = i.builder.kernels(width, epsilon, false)?;
    let pipelines = i.builder.audio()?;
    let residency = i.finish()?;
    Ok(AudioEncoder {
        kernels,
        pipelines,
        attention,
        layer_norm: LayerNorm {
            n: width as u32,
            epsilon,
        },
        convolutions: [first, second, third],
        conv_out,
        positions,
        layers: bound,
        post_norm,
        projector,
        mels,
        width,
        ffn_width,
        output,
        _residency: residency,
    })
}
