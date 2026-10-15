//! Prepare immutable execution resources from a GGUF checkpoint.
//!
//! Selection validates the weights needed by the supported model before registering
//! their pages with Metal. Import then turns those selections into operation-specific
//! operands and prepared kernels. File indexes and compilation caches end here;
//! sessions retain only the resources needed to run the loaded model.

pub mod selected;

use super::{
    Weights, gpt_oss,
    operands::*,
    pipelines::{Builder, ElementwisePipelines},
    qwen,
};
use crate::{
    Tokenizer,
    gguf::{self, Gguf},
    gpu::{Gpu, MappedRegions, ReadOnly},
    tensor::{Tensor, WeightFormat},
};
use anyhow::{Context, Result, bail, ensure};
use std::{collections::HashMap, sync::Arc};

/// Checkpoint dimensions and numerical parameters used to validate weights and prepare execution.
/// This import record is discarded once each runtime operation has the parameters it needs.
pub struct Geometry {
    pub hidden: usize,
    pub heads: usize,
    pub kv: usize,
    pub dim: usize,
    pub ffn_width: usize,
    pub context: usize,
    pub epsilon: f32,
    pub base: f32,
    pub count: usize,
}

pub struct Parameters<'a> {
    file: &'a Gguf,
    prefix: &'a str,
}

impl Parameters<'_> {
    pub fn number(&self, key: &str) -> Result<f64> {
        self.file.number(&format!("{}.{key}", self.prefix))
    }

    pub fn integer(&self, key: &str) -> Result<usize> {
        let n = self.number(key)?;
        ensure!(
            n > 0.0 && n <= u32::MAX as f64 && n.fract() == 0.0,
            "invalid positive integer {}.{key}",
            self.prefix
        );
        Ok(n as usize)
    }

    pub fn optional(&self, key: &str, default: f64) -> Result<f64> {
        let key = format!("{}.{key}", self.prefix);
        if !self.file.meta.contains_key(&key) {
            return Ok(default);
        }
        self.file.number(&key)
    }

    pub fn scaling(&self, supported: &str) -> Result<()> {
        let key = format!("{}.rope.scaling.type", self.prefix);
        if self.file.meta.contains_key(&key) {
            let value = self.file.text(&key)?;
            ensure!(value == supported, "unsupported RoPE scaling {value}");
        }
        Ok(())
    }

    fn geometry(&self) -> Result<Geometry> {
        let hidden = self.integer("embedding_length")?;
        let heads = self.integer("attention.head_count")?;
        let kv = self.integer("attention.head_count_kv")?;
        let dim = self.integer("attention.key_length")?;
        let ffn_width = self.integer("feed_forward_length")?;
        let context = self.integer("context_length")?;
        let epsilon = self.number("attention.layer_norm_rms_epsilon")? as f32;
        let base = self.number("rope.freq_base")? as f32;
        let count = self.integer("block_count")?;
        ensure!(
            hidden <= 16384 && ffn_width <= 65536 && heads <= 128 && kv <= 128,
            "model geometry exceeds the supported kernel range"
        );
        ensure!(
            hidden.is_multiple_of(128) || hidden == 2880,
            "unsupported hidden dimension {hidden}"
        );
        ensure!(
            ffn_width.is_multiple_of(64),
            "feed-forward dimension must be divisible by 64"
        );
        ensure!(
            matches!(dim, 64 | 128 | 256) && heads.is_multiple_of(kv),
            "unsupported attention geometry"
        );
        ensure!(
            epsilon.is_finite() && epsilon > 0.0 && base.is_finite() && base > 1.0,
            "invalid normalization or RoPE parameters"
        );
        ensure!(count <= 128, "unsupported layer count");
        Ok(Geometry {
            hidden,
            heads,
            kv,
            dim,
            ffn_width,
            context,
            epsilon,
            base,
            count,
        })
    }
}

/// Required weight selection before GPU registration, with claims tracking each operand's file span.
/// Import consumes those claims to catch accidental duplicate or omitted bindings.
pub struct Selection<'a> {
    directory: &'a HashMap<&'a str, gguf::Entry>,
    pub spans: HashMap<usize, usize>,
}

impl<'a> Selection<'a> {
    fn new(directory: &'a HashMap<&'a str, gguf::Entry>) -> Self {
        Self {
            directory,
            spans: HashMap::new(),
        }
    }

    pub fn tensor(&mut self, name: &str) -> Result<gguf::Tensor<'a>> {
        let tensor = self.optional_tensor(name)?;
        tensor.with_context(|| format!("missing tensor {name}"))
    }

    fn optional_tensor(&mut self, name: &str) -> Result<Option<gguf::Tensor<'a>>> {
        let Some(entry) = self.directory.get(name) else {
            return Ok(None);
        };
        let tensor = entry
            .select()
            .with_context(|| format!("invalid tensor {name}"))?;
        ensure!(
            tensor.shape.len() <= 3,
            "unsupported runtime tensor geometry for {name}"
        );
        let previous = self.spans.insert(tensor.offset, tensor.bytes);
        ensure!(previous.is_none(), "duplicate selected operand {name}");
        Ok(Some(tensor))
    }

    pub fn float(&mut self, name: &str, shape: &[usize]) -> Result<gguf::Tensor<'a>> {
        let t = self.tensor(name)?;
        ensure!(
            t.kind == WeightFormat::F32 && t.shape == shape,
            "invalid float tensor {name}"
        );
        Ok(t)
    }

    pub fn matrix(&mut self, name: &str, input: usize, output: usize) -> Result<gguf::Tensor<'a>> {
        let t = self.tensor(&format!("{name}.weight"))?;
        ensure!(
            input.is_multiple_of(64) && t.shape == [input, output],
            "unexpected matrix shape for {name}"
        );
        ensure!(
            t.kind != WeightFormat::Mxfp4,
            "MXFP4 is only supported for expert coefficients"
        );
        Ok(t)
    }

    pub fn biased(
        &mut self,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<selected::Projection<'a, gguf::Tensor<'a>>> {
        let matrix = self.matrix(name, input, output)?;
        let bias = self.float(&format!("{name}.bias"), &[output])?;
        Ok(selected::Projection { matrix, bias })
    }

    pub fn projection_with_optional_bias(
        &mut self,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<selected::Projection<'a, Option<gguf::Tensor<'a>>>> {
        let matrix = self.matrix(name, input, output)?;
        let bias = self.optional_tensor(&format!("{name}.bias"))?;
        if let Some(bias) = &bias {
            ensure!(
                bias.kind == WeightFormat::F32 && bias.shape == [output],
                "invalid float tensor {name}.bias"
            );
        }
        Ok(selected::Projection { matrix, bias })
    }
}

/// Conversion of validated selections into GPU operands and prepared kernels.
/// Names and layer topology were resolved during selection; each binding consumes its span claim.
pub struct Import {
    spans: HashMap<usize, usize>,
    regions: MappedRegions,
    pub builder: Builder,
}

impl Import {
    fn tensor(&mut self, t: gguf::Tensor<'_>) -> Result<Tensor> {
        let kind = t.kind;
        let shape = t.shape;
        let storage = self.binding(t)?;
        Tensor::bound(storage, kind, shape)
    }

    pub fn binding(&mut self, t: gguf::Tensor<'_>) -> Result<ReadOnly> {
        let bytes = self.spans.remove(&t.offset);
        ensure!(
            bytes == Some(t.bytes),
            "unselected or already bound operand at byte {}",
            t.offset
        );
        self.regions.get(t.offset, t.bytes)
    }

    pub fn biased(
        &mut self,
        selected::Projection { matrix, bias }: selected::Projection<'_, gguf::Tensor<'_>>,
    ) -> Result<Projection> {
        let matrix = self.tensor(matrix)?;
        let matrix = self.builder.dense(matrix)?;
        let bias = self.binding(bias)?;
        Ok(Projection {
            matrix,
            bias: Some(bias),
        })
    }

    pub fn projection_with_optional_bias(
        &mut self,
        selected::Projection { matrix, bias }: selected::Projection<'_, Option<gguf::Tensor<'_>>>,
    ) -> Result<Projection> {
        let matrix = self.tensor(matrix)?;
        let matrix = self.builder.dense(matrix)?;
        let bias = match bias {
            None => None,
            Some(bias) => {
                let bias = self.binding(bias)?;
                Some(bias)
            }
        };
        Ok(Projection { matrix, bias })
    }

    pub fn finish(self) -> Result<std::sync::Arc<crate::gpu::ResidentWeights>> {
        ensure!(
            self.spans.is_empty(),
            "{} unbound selected operands",
            self.spans.len()
        );
        Ok(self.regions.residency)
    }
}

pub(crate) fn load(file: &Gguf) -> Result<Weights> {
    let prefix = file.text("general.architecture")?;
    let p = Parameters { file, prefix };
    let inner = match prefix {
        "gpt-oss" => {
            let g = p.geometry()?;
            gpt_oss::load(file, &p, g)?
        }
        "qwen35" => {
            let g = p.geometry()?;
            qwen::load(file, &p, g)?
        }
        _ => bail!("unsupported architecture {prefix}"),
    };
    let tokenizer = Tokenizer::load(file, inner.base().embedding.vocabulary)?;
    Ok(Weights {
        inner,
        tokenizer: Arc::new(tokenizer),
    })
}

pub fn common_selection<'a>(
    directory: &'a HashMap<&'a str, gguf::Entry>,
    g: &Geometry,
) -> Result<(Selection<'a>, selected::Common<'a>)> {
    let mut s = Selection::new(directory);
    let embedding = s.tensor("token_embd.weight")?;
    ensure!(
        g.hidden.is_multiple_of(64)
            && embedding.shape.len() == 2
            && embedding.shape[0] == g.hidden
            && embedding.kind != WeightFormat::Mxfp4,
        "invalid embedding table"
    );
    let vocabulary = embedding.shape[1];
    ensure!(
        !directory.contains_key("token_embd.bias"),
        "token_embd.bias is unsupported by embedding lookup"
    );
    let output = s.projection_with_optional_bias("output", g.hidden, vocabulary)?;
    let norm = s.float("output_norm.weight", &[g.hidden])?;
    Ok((
        s,
        selected::Common {
            embedding,
            output,
            norm,
        },
    ))
}

pub fn import(file: &Gguf, spans: HashMap<usize, usize>, gpu: &Gpu) -> Result<Import> {
    let regions = MappedRegions::new(
        gpu,
        &file.map,
        spans.iter().map(|(&offset, &bytes)| (offset, bytes)),
    )?;
    let builder = Builder::new(gpu)?;
    Ok(Import {
        spans,
        regions,
        builder,
    })
}

pub fn common_weights(
    i: &mut Import,
    selected::Common {
        embedding,
        output: selected::Projection { matrix, bias },
        norm,
    }: selected::Common<'_>,
) -> Result<(Embedding, Output, ReadOnly, ElementwisePipelines)> {
    let embedding = i.tensor(embedding)?;
    let embedding = i.builder.embedding(embedding)?;
    let matrix = i.tensor(matrix)?;
    let bias = match bias {
        None => None,
        Some(bias) => {
            let bias = i.binding(bias)?;
            Some(bias)
        }
    };
    let output = i.builder.output(matrix, bias)?;
    let norm = i.binding(norm)?;
    let pipelines = i.builder.elementwise()?;
    Ok((embedding, output, norm, pipelines))
}

/// Metadata admitted for the supported model architectures.
pub(crate) const METADATA: &[&str] = &[
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "general.architecture",
    "general.alignment",
    "gpt-oss.embedding_length",
    "gpt-oss.attention.head_count",
    "gpt-oss.attention.head_count_kv",
    "gpt-oss.attention.key_length",
    "gpt-oss.attention.layer_norm_rms_epsilon",
    "gpt-oss.attention.sliding_window",
    "gpt-oss.feed_forward_length",
    "gpt-oss.context_length",
    "gpt-oss.block_count",
    "gpt-oss.expert_count",
    "gpt-oss.expert_used_count",
    "gpt-oss.rope.freq_base",
    "gpt-oss.rope.dimension_count",
    "gpt-oss.rope.scaling.type",
    "gpt-oss.rope.scaling.factor",
    "gpt-oss.rope.scaling.original_context_length",
    "gpt-oss.rope.scaling.yarn_beta_fast",
    "gpt-oss.rope.scaling.yarn_beta_slow",
    "qwen35.embedding_length",
    "qwen35.attention.head_count",
    "qwen35.attention.head_count_kv",
    "qwen35.attention.key_length",
    "qwen35.attention.layer_norm_rms_epsilon",
    "qwen35.feed_forward_length",
    "qwen35.context_length",
    "qwen35.block_count",
    "qwen35.full_attention_interval",
    "qwen35.nextn_predict_layers",
    "qwen35.rope.freq_base",
    "qwen35.rope.dimension_count",
    "qwen35.rope.scaling.type",
    "qwen35.rope.scaling.factor",
    "qwen35.ssm.group_count",
    "qwen35.ssm.time_step_rank",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.state_size",
    "qwen35.ssm.conv_kernel",
];
