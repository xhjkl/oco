//! Read whisper.cpp's ggml checkpoint format and bind it for Metal.
//!
//! The format is a header of eleven integers, the mel filter bank, the text vocabulary, and
//! then unaligned tensor records to the end of the file. Tensors are copied into one shared
//! allocation at 256-byte offsets, which both respects Metal alignment and lets the
//! convolution kernels be repacked for the dense product kernels on the way in.
use super::{Decoder, DecoderLayer, Encoder, Model, vocabulary::Vocabulary};
use crate::{
    audio::{BINS, MelFilters},
    gpu::{BufferView, Gpu, ReadOnly},
    tensor::{Tensor, WeightFormat},
    weights::{
        common::AttentionProjections,
        compute::{
            AttentionPlan, Conv, EncoderLayer, GeluFfn, GeluForm, Im2col, LayerNorm,
            LayerNormWeights, Rope, half_bits,
        },
        operands::Projection,
        pipelines::Builder,
    },
};
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use std::{collections::HashMap, fs::File, path::Path};

const MAGIC: u32 = 0x6767_6d6c;
const TAPS: usize = 3;

/// Header fields in file order.
struct Header {
    vocabulary: usize,
    audio_context: usize,
    audio_state: usize,
    audio_heads: usize,
    audio_layers: usize,
    text_context: usize,
    text_state: usize,
    text_heads: usize,
    text_layers: usize,
    mels: usize,
}

/// One tensor record: shape in ggml order (innermost first) and its bytes in the file.
struct Record {
    shape: Vec<usize>,
    kind: WeightFormat,
    offset: usize,
    bytes: usize,
}

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .cursor
            .checked_add(n)
            .context("checkpoint offset overflow")?;
        let slice = self
            .bytes
            .get(self.cursor..end)
            .context("truncated checkpoint")?;
        self.cursor = end;
        Ok(slice)
    }

    fn i32(&mut self) -> Result<i32> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn count(&mut self) -> Result<usize> {
        let n = self.i32()?;
        ensure!(n >= 0, "negative count in checkpoint");
        Ok(n as usize)
    }
}

/// Everything the file declares besides tensor bytes.
struct Checkpoint {
    header: Header,
    filters: MelFilters,
    tokens: Vec<Vec<u8>>,
    records: HashMap<String, Record>,
}

fn parse(bytes: &[u8]) -> Result<Checkpoint> {
    let mut r = Reader { bytes, cursor: 0 };
    let magic = r.i32()? as u32;
    ensure!(magic == MAGIC, "not a whisper.cpp ggml checkpoint");
    let mut fields = [0usize; 11];
    for field in &mut fields {
        *field = r.count()?;
    }
    let [
        vocabulary,
        audio_context,
        audio_state,
        audio_heads,
        audio_layers,
        text_context,
        text_state,
        text_heads,
        text_layers,
        mels,
        _ftype,
    ] = fields;
    let header = Header {
        vocabulary,
        audio_context,
        audio_state,
        audio_heads,
        audio_layers,
        text_context,
        text_state,
        text_heads,
        text_layers,
        mels,
    };
    let filter_mels = r.count()?;
    let filter_bins = r.count()?;
    ensure!(
        filter_mels == mels && filter_bins == BINS,
        "mel filters do not match the model"
    );
    let filters = r.take(filter_mels * filter_bins * 4)?;
    let filters = filters
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect();
    let filters = MelFilters::new(mels, filters)?;
    let count = r.count()?;
    ensure!(count <= vocabulary, "vocabulary longer than declared");
    let mut tokens = Vec::with_capacity(count);
    for _ in 0..count {
        let length = r.count()?;
        tokens.push(r.take(length)?.to_vec());
    }
    let mut records = HashMap::new();
    while r.cursor < bytes.len() {
        let dims = r.count()?;
        ensure!((1..=3).contains(&dims), "unsupported tensor rank {dims}");
        let name_length = r.count()?;
        let kind = match r.i32()? {
            0 => WeightFormat::F32,
            1 => WeightFormat::F16,
            8 => WeightFormat::Q8_0,
            2 => WeightFormat::Q4_0,
            other => bail!("unsupported whisper tensor type {other}"),
        };
        let mut shape = Vec::with_capacity(dims);
        for _ in 0..dims {
            shape.push(r.count()?);
        }
        let name = std::str::from_utf8(r.take(name_length)?)?.to_owned();
        let elements = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .context("tensor size overflow")?;
        let (block, size) = kind.block_layout();
        ensure!(elements.is_multiple_of(block), "tensor not block aligned");
        let bytes = elements / block * size;
        let offset = r.cursor;
        r.take(bytes)?;
        let previous = records.insert(
            name,
            Record {
                shape,
                kind,
                offset,
                bytes,
            },
        );
        ensure!(previous.is_none(), "duplicate tensor name");
    }
    Ok(Checkpoint {
        header,
        filters,
        tokens,
        records,
    })
}

/// Copies every tensor into one shared allocation and hands out typed views.
struct Store<'a> {
    file: &'a [u8],
    records: HashMap<String, Record>,
    placements: HashMap<String, usize>,
    storage: BufferView,
    buffer: crate::gpu::Buffer,
}

impl<'a> Store<'a> {
    fn new(gpu: &Gpu, file: &'a [u8], records: HashMap<String, Record>) -> Result<Self> {
        let mut placements = HashMap::with_capacity(records.len());
        let mut total = 0usize;
        let mut names: Vec<&String> = records.keys().collect();
        names.sort();
        for name in names {
            placements.insert(name.clone(), total);
            total = total
                .checked_add(records[name].bytes.next_multiple_of(256))
                .context("checkpoint size overflow")?;
        }
        let buffer = gpu.new_buffer(total);
        let storage = BufferView::new(&buffer, 0, total);
        for (name, record) in &records {
            let source = &file[record.offset..record.offset + record.bytes];
            storage.write(placements[name], source);
        }
        Ok(Self {
            file,
            records,
            placements,
            storage,
            buffer,
        })
    }

    fn record(&self, name: &str) -> Result<&Record> {
        self.records
            .get(name)
            .with_context(|| format!("missing tensor {name}"))
    }

    fn view(&self, name: &str, kind: WeightFormat, shape: &[usize]) -> Result<ReadOnly> {
        let record = self.record(name)?;
        ensure!(
            record.kind == kind && record.shape == shape,
            "tensor {name} is {:?} {:?}, expected {kind:?} {shape:?}",
            record.kind,
            record.shape
        );
        Ok(ReadOnly::within(
            &self.buffer,
            self.placements[name],
            record.bytes,
        ))
    }

    fn float(&self, name: &str, shape: &[usize]) -> Result<ReadOnly> {
        self.view(name, WeightFormat::F32, shape)
    }

    /// A weight matrix of any supported encoding with `input` columns and `output` rows.
    fn matrix(
        &self,
        builder: &mut Builder,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Projection> {
        let record = self.record(name)?;
        ensure!(
            record.shape == [input, output],
            "tensor {name} has shape {:?}, expected [{input}, {output}]",
            record.shape
        );
        let storage = ReadOnly::within(&self.buffer, self.placements[name], record.bytes);
        let tensor = Tensor::bound(storage, record.kind, &[input, output])?;
        Ok(Projection {
            matrix: builder.dense(tensor)?,
            bias: None,
        })
    }

    fn biased(
        &self,
        builder: &mut Builder,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Projection> {
        let mut projection = self.matrix(builder, &format!("{name}.weight"), input, output)?;
        projection.bias = Some(self.float(&format!("{name}.bias"), &[output])?);
        Ok(projection)
    }

    fn layer_norm(&self, name: &str, width: usize) -> Result<LayerNormWeights> {
        Ok(LayerNormWeights {
            weight: self.float(&format!("{name}.weight"), &[width])?,
            bias: self.float(&format!("{name}.bias"), &[width])?,
        })
    }

    fn attention(
        &self,
        builder: &mut Builder,
        prefix: &str,
        width: usize,
    ) -> Result<AttentionProjections<Projection>> {
        Ok(AttentionProjections {
            q: self.biased(builder, &format!("{prefix}.query"), width, width)?,
            k: self.matrix(builder, &format!("{prefix}.key.weight"), width, width)?,
            v: self.biased(builder, &format!("{prefix}.value"), width, width)?,
            out: self.biased(builder, &format!("{prefix}.out"), width, width)?,
        })
    }

    fn mlp(&self, builder: &mut Builder, prefix: &str, width: usize) -> Result<GeluFfn> {
        Ok(GeluFfn {
            up: self.biased(builder, &format!("{prefix}.mlp.0"), width, 4 * width)?,
            down: self.biased(builder, &format!("{prefix}.mlp.2"), 4 * width, width)?,
            width: 4 * width,
            form: GeluForm::Tanh,
        })
    }

    /// One-dimensional convolution over `length` frames of `channels`.
    fn convolution(
        &self,
        gpu: &Gpu,
        builder: &mut Builder,
        name: &str,
        (channels, outputs, length, stride): (usize, usize, usize, usize),
    ) -> Result<Conv> {
        let record = self.record(&format!("{name}.weight"))?;
        ensure!(
            record.kind == WeightFormat::F16 && record.shape == [TAPS, channels, outputs],
            "unexpected convolution tensor {name}"
        );
        let source = half_bits(&self.file[record.offset..record.offset + record.bytes]);
        let bias = self.float(&format!("{name}.bias"), &[1, outputs])?;
        let geometry = Im2col::new(channels, length, 1, (TAPS, 1), stride, (1, 0));
        // The checkpoint stores (out, in, tap).
        Conv::repacked(gpu, builder, geometry, outputs, bias, |out, tap, c| {
            source[(out * channels + c) * TAPS + tap]
        })
    }
}

fn plan(
    builder: &mut Builder,
    heads: usize,
    dim: usize,
    context: usize,
    encoder: bool,
) -> Result<AttentionPlan> {
    let pipelines = if encoder {
        builder.encoder_attention(dim, context)?
    } else {
        builder.attention(dim, context)?
    };
    Ok(AttentionPlan::new(
        Rope::unrotated(heads, heads, dim),
        pipelines,
    ))
}

pub(super) fn load(path: &Path) -> Result<Model> {
    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };
    let Checkpoint {
        header: h,
        filters,
        tokens,
        records,
    } = parse(&map)?;
    ensure!(
        h.audio_state == h.text_state && h.audio_heads == h.text_heads,
        "Whisper encoder and decoder widths differ"
    );
    let state = h.audio_state;
    ensure!(
        state.is_multiple_of(64) && h.audio_heads > 0 && state.is_multiple_of(h.audio_heads),
        "unsupported Whisper width {state}"
    );
    let dim = state / h.audio_heads;
    ensure!(
        matches!(dim, 64 | 128 | 256),
        "unsupported Whisper head width {dim}"
    );
    ensure!(
        h.mels > 0 && h.audio_context > 0 && h.text_context > 0,
        "invalid Whisper geometry"
    );
    let vocabulary = Vocabulary::new(tokens, h.vocabulary)?;
    let gpu = Gpu::new()?;
    let store = Store::new(&gpu, &map, records)?;
    let mut builder = Builder::new(&gpu)?;
    let frames = 2 * h.audio_context;
    let conv1 = store.convolution(
        &gpu,
        &mut builder,
        "encoder.conv1",
        (h.mels, state, frames, 1),
    )?;
    let conv2 = store.convolution(
        &gpu,
        &mut builder,
        "encoder.conv2",
        (state, state, frames, 2),
    )?;
    ensure!(
        conv2.geometry.rows as usize == h.audio_context,
        "convolutions do not produce the audio context"
    );
    let mut encoder_layers = Vec::with_capacity(h.audio_layers);
    for index in 0..h.audio_layers {
        let prefix = format!("encoder.blocks.{index}");
        encoder_layers.push(EncoderLayer {
            norm: store.layer_norm(&format!("{prefix}.attn_ln"), state)?,
            attention: store.attention(&mut builder, &format!("{prefix}.attn"), state)?,
            ffn_norm: store.layer_norm(&format!("{prefix}.mlp_ln"), state)?,
            ffn: store.mlp(&mut builder, &prefix, state)?,
        });
    }
    let encoder = Encoder {
        convolutions: [conv1, conv2],
        positions: store.float("encoder.positional_embedding", &[state, h.audio_context])?,
        layers: encoder_layers,
        post_norm: store.layer_norm("encoder.ln_post", state)?,
        attention: plan(&mut builder, h.audio_heads, dim, h.audio_context, true)?,
    };
    let mut decoder_layers = Vec::with_capacity(h.text_layers);
    for index in 0..h.text_layers {
        let prefix = format!("decoder.blocks.{index}");
        decoder_layers.push(DecoderLayer {
            norm: store.layer_norm(&format!("{prefix}.attn_ln"), state)?,
            attention: store.attention(&mut builder, &format!("{prefix}.attn"), state)?,
            cross_norm: store.layer_norm(&format!("{prefix}.cross_attn_ln"), state)?,
            cross: store.attention(&mut builder, &format!("{prefix}.cross_attn"), state)?,
            ffn_norm: store.layer_norm(&format!("{prefix}.mlp_ln"), state)?,
            ffn: store.mlp(&mut builder, &prefix, state)?,
        });
    }
    let table = store.record("decoder.token_embedding.weight")?;
    ensure!(
        table.shape == [state, h.vocabulary],
        "token embedding does not match the vocabulary"
    );
    let table_view = || {
        ReadOnly::within(
            &store.buffer,
            store.placements["decoder.token_embedding.weight"],
            table.bytes,
        )
    };
    let embedding = builder.embedding(Tensor::bound(table_view(), table.kind, &table.shape)?)?;
    let output = builder.output(Tensor::bound(table_view(), table.kind, &table.shape)?, None)?;
    let decoder = Decoder {
        embedding,
        output,
        positions: store.float("decoder.positional_embedding", &[state, h.text_context])?,
        layers: decoder_layers,
        norm: store.layer_norm("decoder.ln", state)?,
        attention: plan(&mut builder, h.text_heads, dim, h.text_context, false)?,
        cross: plan(&mut builder, h.text_heads, dim, h.audio_context, false)?,
    };
    let kernels = builder.kernels(state, 1e-5, false)?;
    let audio = builder.audio()?;
    let Store { storage, .. } = store;
    Ok(Model {
        gpu,
        kernels,
        audio,
        filters,
        vocabulary,
        encoder,
        decoder,
        layer_norm: LayerNorm {
            n: state as u32,
            epsilon: 1e-5,
        },
        state,
        mels: h.mels,
        audio_context: h.audio_context,
        text_context: h.text_context,
        _weights: storage,
    })
}
