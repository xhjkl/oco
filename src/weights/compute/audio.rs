//! Building blocks shared by the audio encoders and the Whisper decoder: mean-centered
//! normalization, GELU, positional tables, and unfolding strided convolutions into dense
//! products. The caller owns the prepared kernels and the operands.
use super::{AttentionPlan, Bindings};
use crate::gpu::{BufferRef, BufferView, Gpu, ReadOnly, dispatch};
use crate::tensor::{Tensor, WeightFormat};
use crate::weights::{
    common::AttentionProjections,
    operands::{DenseMatrix, Projection},
    pipelines::{AudioPipelines, Builder},
    state::{AttentionScratch, Kv},
};
use anyhow::Result;
use bytemuck::{Pod, Zeroable, bytes_of};

/// Width and epsilon of one mean-centered normalization.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LayerNorm {
    pub n: u32,
    pub epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Gelu {
    n: u32,
    erf_form: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AddRows {
    n: u32,
    rows: u32,
    offset: u32,
    period: u32,
}

/// One strided convolution over channel-minor `(batch, h, w, c)` input, unfolded into
/// `rows` product rows of `k` BF16 values each (taps `(th, tw, c)`, zero-padded to `k`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Im2col {
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub oh: u32,
    pub ow: u32,
    pub kh: u32,
    pub kw: u32,
    pub stride: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub k: u32,
    pub rows: u32,
}

impl Im2col {
    /// Geometry of a `kh`×`kw` convolution with the given stride and padding over a
    /// channel-minor `(h, w, c)` input, its products padded to whole 64-wide tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c: usize,
        h: usize,
        w: usize,
        (kh, kw): (usize, usize),
        stride: usize,
        (pad_h, pad_w): (usize, usize),
    ) -> Self {
        let extent = |input: usize, taps: usize, pad: usize| (input + 2 * pad - taps) / stride + 1;
        let (oh, ow) = (extent(h, kh, pad_h), extent(w, kw, pad_w));
        Self {
            c: c as u32,
            h: h as u32,
            w: w as u32,
            oh: oh as u32,
            ow: ow as u32,
            kh: kh as u32,
            kw: kw as u32,
            stride: stride as u32,
            pad_h: pad_h as u32,
            pad_w: pad_w as u32,
            k: (kh * kw * c).next_multiple_of(64) as u32,
            rows: (oh * ow) as u32,
        }
    }

    pub fn output(&self) -> (usize, usize) {
        (self.oh as usize, self.ow as usize)
    }
}

/// Reinterpret little-endian bytes as half-precision bit patterns.
pub fn half_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| u16::from_le_bytes(b))
        .collect()
}

/// Upload an F16 matrix of `output` rows and `input` columns gathered through `source(row, column)`:
/// the way to give the dense kernels a permuted or zero-padded view of checkpoint coefficients.
pub fn repacked_matrix(
    gpu: &Gpu,
    builder: &mut Builder,
    input: usize,
    output: usize,
    source: impl Fn(usize, usize) -> u16,
) -> Result<DenseMatrix> {
    let mut values = vec![0u16; output * input];
    for (row, values) in values.chunks_exact_mut(input).enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = source(row, column);
        }
    }
    let storage = gpu.upload(bytemuck::cast_slice(&values));
    builder.dense(Tensor::bound(storage, WeightFormat::F16, &[input, output])?)
}

/// Learned gain and offset of one mean-centered normalization.
pub struct LayerNormWeights {
    pub weight: ReadOnly,
    pub bias: ReadOnly,
}

/// Activation between the two projections of a bias-carrying GELU feed-forward network.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GeluForm {
    /// Exact form (`gelu_erf`), used by the Qwen3-ASR encoder.
    Erf,
    /// Tanh approximation, matching ggml's `gelu` used for Whisper.
    Tanh,
}

/// Up and down projections of a GELU feed-forward network.
pub struct GeluFfn {
    pub up: Projection,
    pub down: Projection,
    pub form: GeluForm,
}

/// One pre-norm encoder block: full self-attention with biases, then a GELU network.
pub struct EncoderLayer {
    pub norm: LayerNormWeights,
    pub attention: AttentionProjections<Projection>,
    pub ffn_norm: LayerNormWeights,
    pub ffn: GeluFfn,
}

impl Bindings<'_> {
    pub fn layer_norm(
        &self,
        a: &AudioPipelines,
        w: &LayerNormWeights,
        p: LayerNorm,
        x: (&BufferRef, usize),
        y: &BufferView,
        rows: usize,
    ) {
        dispatch(
            self.encoder,
            &a.layer_norm,
            &[x, w.weight.binding(), w.bias.binding(), y.binding()],
            bytes_of(&p),
            [rows, 1, 1],
            256,
        );
    }

    /// Add `addend` rows into the `hidden` stream and normalize the sums into `y` in one pass.
    #[allow(clippy::too_many_arguments)]
    pub fn add_layer_norm(
        &self,
        a: &AudioPipelines,
        w: &LayerNormWeights,
        p: LayerNorm,
        hidden: (&BufferRef, usize),
        addend: (&BufferRef, usize),
        y: &BufferView,
        rows: usize,
    ) {
        dispatch(
            self.encoder,
            &a.add_layer_norm,
            &[
                hidden,
                addend,
                w.weight.binding(),
                w.bias.binding(),
                y.binding(),
            ],
            bytes_of(&p),
            [rows, 1, 1],
            256,
        );
    }

    pub fn gelu(&self, a: &AudioPipelines, x: (&BufferRef, usize), n: usize, form: GeluForm) {
        let p = Gelu {
            n: n as u32,
            erf_form: (form == GeluForm::Erf) as u32,
        };
        dispatch(
            self.encoder,
            &a.gelu,
            &[x],
            bytes_of(&p),
            [n.div_ceil(256), 1, 1],
            256,
        );
    }

    /// Add FP32 `table` rows `(offset + row) % period` to `rows` feature rows of width `n`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rows(
        &self,
        a: &AudioPipelines,
        x: (&BufferRef, usize),
        table: &ReadOnly,
        n: usize,
        rows: usize,
        offset: usize,
        period: usize,
    ) {
        let p = AddRows {
            n: n as u32,
            rows: rows as u32,
            offset: offset as u32,
            period: period as u32,
        };
        dispatch(
            self.encoder,
            &a.add_rows,
            &[x, table.binding()],
            bytes_of(&p),
            [(n * rows).div_ceil(256), 1, 1],
            256,
        );
    }

    pub fn im2col(
        &self,
        a: &AudioPipelines,
        x: (&BufferRef, usize),
        y: (&BufferRef, usize),
        p: Im2col,
    ) {
        let n = p.rows as usize * p.k as usize;
        dispatch(
            self.encoder,
            &a.im2col,
            &[x, y],
            bytes_of(&p),
            [n.div_ceil(256), 1, 1],
            256,
        );
    }

    /// Transform the normalized rows through a GELU network, leaving the result in `projected`.
    pub fn gelu_feed_forward(&self, a: &AudioPipelines, w: &GeluFfn, rows: usize) {
        self.gelu_project(a, w, rows, &self.scratch.projected);
    }
}

/// One strided convolution lowered to a dense product over unfolded rows.
/// Weights are repacked at load time into `(th, tw, c)` tap order, zero-padded to the
/// product's input width.
pub struct Conv {
    pub weights: Projection,
    pub geometry: Im2col,
}

impl Conv {
    /// Repack F16 taps for `geometry`, where `source(out, tap, channel)` addresses the
    /// checkpoint's coefficient and `tap` runs row-major over the kernel window.
    pub fn repacked(
        gpu: &Gpu,
        builder: &mut Builder,
        geometry: Im2col,
        outputs: usize,
        bias: ReadOnly,
        source: impl Fn(usize, usize, usize) -> u16,
    ) -> Result<Self> {
        let channels = geometry.c as usize;
        let taps = (geometry.kh * geometry.kw) as usize;
        let matrix = repacked_matrix(gpu, builder, geometry.k as usize, outputs, |out, j| {
            if j < taps * channels {
                source(out, j / channels, j % channels)
            } else {
                0
            }
        })?;
        Ok(Self {
            weights: Projection {
                matrix,
                bias: Some(bias),
            },
            geometry,
        })
    }
}

impl Bindings<'_> {
    /// Unfold `x` into `unfolded`, apply the convolution's weights and bias, and activate.
    #[allow(clippy::too_many_arguments)]
    pub fn convolve(
        &self,
        a: &AudioPipelines,
        conv: &Conv,
        x: (&BufferRef, usize),
        unfolded: &BufferView,
        y: &BufferView,
        form: GeluForm,
    ) {
        let g = conv.geometry;
        self.im2col(a, x, unfolded.binding(), g);
        conv.weights
            .apply(self, unfolded.binding(), y, g.rows as usize);
        let n = g.rows as usize * conv.weights.matrix.output;
        self.gelu(a, y.binding(), n, form);
    }
}

impl Bindings<'_> {
    /// Run `rows` of `hidden` through the encoder blocks, normalizing each block's input in the
    /// previous block's residual pass. `norm` ends holding the `post`-normalized output.
    /// Only the first `keys` rows are real; any padding rows attend to the real keys and are
    /// otherwise carried along so dense products see whole tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn encoder_layers(
        &self,
        a: &AudioPipelines,
        layers: &[EncoderLayer],
        post: &LayerNormWeights,
        ln: LayerNorm,
        plan: &AttentionPlan,
        memory: (&AttentionScratch, &Kv),
        rows: usize,
        keys: usize,
    ) {
        let s = self.scratch;
        let (hidden, projected, norm) = (s.hidden.binding(), s.projected.binding(), &s.norm);
        let first = layers.first().map_or(post, |l| &l.norm);
        self.layer_norm(a, first, ln, hidden, norm, rows);
        for (index, layer) in layers.iter().enumerate() {
            self.encoder_attention(&layer.attention, plan, memory, rows, keys);
            self.add_layer_norm(a, &layer.ffn_norm, ln, hidden, projected, norm, rows);
            self.gelu_feed_forward(a, &layer.ffn, rows);
            let next = layers.get(index + 1).map_or(post, |l| &l.norm);
            self.add_layer_norm(a, next, ln, hidden, projected, norm, rows);
        }
    }

    /// Transform the normalized rows through a GELU network into `y`.
    pub fn gelu_project(&self, a: &AudioPipelines, w: &GeluFfn, rows: usize, y: &BufferView) {
        let s = self.scratch;
        self.prepare_norm_scales(&[&w.up.matrix], rows, 0..rows);
        w.up.apply_norm(self, &s.up, rows);
        self.gelu(a, s.up.at(0), rows * w.up.matrix.output, w.form);
        w.down.apply(self, s.up.at(0), y, rows);
    }
}
