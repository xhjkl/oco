use super::{common::Kernels, compute::Norm};
use crate::gpu::Library;
use crate::{
    gpu::{Gpu, Pipeline},
    tensor::WeightFormat,
};
use anyhow::Result;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTL4Compiler, MTL4CompilerDescriptor, MTLDevice};
use std::collections::HashMap;

pub struct AttentionPipelines {
    pub rope: Pipeline,
    pub dispatch: AttentionDispatch,
    pub tiled: Option<Pipeline>,
    pub wide: Option<Pipeline>,
    /// 32-query register-softmax tiles for long batches of 64- or 128-wide heads.
    pub block: Option<Pipeline>,
    pub split: Option<SplitAttentionPipelines>,
}

pub struct SplitAttentionPipelines {
    pub partial: Pipeline,
    pub merge: Pipeline,
}

pub enum AttentionDispatch {
    /// Several keys in flight per SIMD group, for 64-, 128-, and 256-wide heads.
    Vector(Pipeline),
    /// One SIMD group per key step, for other head widths.
    Plain(Pipeline),
}

pub struct GatedAttentionPipelines {
    /// Normalizes every live query head and every key head in one launch.
    pub qk_norm: Pipeline,
    pub gate: Pipeline,
}

pub struct RecurrentPipelines {
    pub conv: Pipeline,
    pub delta: Pipeline,
    pub delta_prepare: Pipeline,
    pub delta_persistent: Pipeline,
    pub delta_output: Pipeline,
}

/// Kernels for Nemotron's Mamba-2 blocks and sigmoid-routed relu² experts.
pub struct NemotronPipelines {
    pub conv: Pipeline,
    pub scan: Pipeline,
    pub gate_norm: Pipeline,
    pub router: Pipeline,
    pub relu_sqr: Pipeline,
    pub moe_add: Pipeline,
    /// Gathered Q4_0 expert rows for single-route products.
    pub expert_vector: Pipeline,
    /// The same contraction with relu² applied to its input on load.
    pub expert_vector_relu: Pipeline,
    /// Indirect tile plan over the routed experts and the gathered Q4_0 tiles it drives:
    /// eight-route SIMD-group tiles for sparse routing, 32-route tensor tiles otherwise.
    pub route_plan: Pipeline,
    pub expert_up: [Pipeline; 2],
    pub expert_down: [Pipeline; 2],
}

/// Kernels shared by the audio encoders and the Whisper decoder: mean-centered normalization,
/// GELU, positional tables, and convolution unfolding.
pub struct AudioPipelines {
    pub layer_norm: Pipeline,
    pub add_layer_norm: Pipeline,
    pub gelu: Pipeline,
    pub add_rows: Pipeline,
    pub im2col: Pipeline,
}

pub struct ElementwisePipelines {
    pub norm: Pipeline,
    pub add_norm: Pipeline,
    pub q4_input_scales: Pipeline,
    pub add: Pipeline,
    pub gate: Pipeline,
    pub argmax: Pipeline,
}

pub struct ExpertPipelines {
    pub router_logits: Pipeline,
    /// Indirect plan of 32-route tiles per expert and the tensor-operation tiles it drives.
    pub route_plan: Pipeline,
    pub up_tensor: Pipeline,
    pub down_tensor: Pipeline,
    pub expand: Pipeline,
    pub down_vector: Pipeline,
    pub router: Pipeline,
    pub sum: Pipeline,
}

/// Import-time builder that shares compiled kernels across operations needing the same specialization.
/// Prepared operands retain the Metal handles, so token generation needs neither
/// compilation nor this lookup cache. Each independent model load creates its own builder.
pub struct Builder {
    library: Library,
    compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
    cache: HashMap<(&'static str, WeightFormat, u32), Pipeline>,
}

impl Builder {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let library = gpu.new_library(include_bytes!(concat!(env!("OUT_DIR"), "/oco.metallib")))?;
        let compiler = gpu
            .device
            .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())?;
        Ok(Self {
            compiler,
            library,
            cache: HashMap::new(),
        })
    }

    /// `variant` is function constant 1: matrix kernels read it as the row-tile count,
    /// where 0 selects wide tiles and 1..4 select 8..32-row tiles. Other kernels ignore it.
    pub(crate) fn get(
        &mut self,
        symbol: &'static str,
        format: WeightFormat,
        variant: u32,
    ) -> Result<Pipeline> {
        self.specialized(symbol, format, variant, false, false)
    }

    /// Vector kernels specialized to square the positive part of their input on load.
    pub(crate) fn get_relu_sqr(
        &mut self,
        symbol: &'static str,
        format: WeightFormat,
    ) -> Result<Pipeline> {
        self.specialized(symbol, format, 0, true, false)
    }

    fn specialized(
        &mut self,
        symbol: &'static str,
        format: WeightFormat,
        variant: u32,
        relu_sqr: bool,
        f32_output: bool,
    ) -> Result<Pipeline> {
        // Folding the specialization flags above the row-tile range keeps one cache key.
        let key =
            variant | if relu_sqr { 1 << 16 } else { 0 } | if f32_output { 1 << 17 } else { 0 };
        if let Some(pipeline) = self.cache.get(&(symbol, format, key)) {
            return Ok(pipeline.clone());
        }
        let pipeline = Gpu::new_pipeline(
            &self.compiler,
            &self.library,
            symbol,
            &[
                (0, format as u32),
                (1, variant),
                (2, relu_sqr as u32),
                (3, f32_output as u32),
            ],
        )?;
        self.cache.insert((symbol, format, key), pipeline.clone());
        Ok(pipeline)
    }

    pub fn embedding(
        &mut self,
        tensor: crate::tensor::Tensor,
    ) -> Result<super::operands::Embedding> {
        let [width, vocabulary, _] = tensor.shape;
        let pipeline = self.get("embed", tensor.kind, 0)?;
        Ok(super::operands::Embedding {
            values: tensor.storage,
            pipeline,
            width,
            vocabulary,
        })
    }

    pub fn output(
        &mut self,
        tensor: crate::tensor::Tensor,
        bias: Option<crate::gpu::ReadOnly>,
    ) -> Result<super::operands::Output> {
        let [width, vocabulary, _] = tensor.shape;
        // Scoring in FP32 keeps near-tied candidates ordered as the model intended.
        let vector = self.vector_with(tensor.kind, vocabulary, true)?;
        Ok(super::operands::Output {
            values: tensor.storage,
            bias,
            vector,
            width,
            vocabulary,
        })
    }

    fn vector(&mut self, kind: WeightFormat, output: usize) -> Result<super::operands::Vector> {
        self.vector_with(kind, output, false)
    }

    /// Single-row kernel for one weight encoding, optionally storing FP32 results.
    pub(crate) fn vector_with(
        &mut self,
        kind: WeightFormat,
        output: usize,
        f32_output: bool,
    ) -> Result<super::operands::Vector> {
        self.vector_kernel(kind, output, false, f32_output)
    }

    /// Single-row kernel, optionally squaring the positive part of its input on load and
    /// optionally storing FP32 results.
    pub(crate) fn vector_kernel(
        &mut self,
        kind: WeightFormat,
        output: usize,
        relu_sqr: bool,
        f32_output: bool,
    ) -> Result<super::operands::Vector> {
        use WeightFormat::*;
        // Four-row SIMD groups: narrow outputs get small threadgroups so enough of them exist
        // to cover every GPU core; wide outputs amortize launch work over 32 rows.
        let threads = Self::vector_threads(output);
        let (symbol, column_groups) = match kind {
            Q4_0 | Q4_1 => ("matvec_q4", output.div_ceil(threads / 8)),
            // Two rows per threadgroup with K split across SIMD groups keeps more loads in
            // flight for Q8_0 than four rows per SIMD group.
            Q8_0 => ("matvec_q8_split", output.div_ceil(2)),
            Q5K => ("matvec_q5k", output.div_ceil(16)),
            Q6K => ("matvec_q6k", output.div_ceil(16)),
            // The one-row-per-threadgroup kernel does not apply the relu² load transform.
            F32 | F16 | Bf16 if output < 32768 && !relu_sqr => ("matvec_split", output),
            _ => ("matvec", output.div_ceil(threads / 8)),
        };
        let threads = match symbol {
            "matvec" | "matvec_q4" => threads,
            "matvec_q8_split" => 128,
            _ => 256,
        };
        let pipeline = self.specialized(symbol, kind, 0, relu_sqr, f32_output)?;
        Ok(super::operands::Vector {
            pipeline,
            column_groups,
            threads,
        })
    }

    /// Threads per threadgroup for the four-row vector kernels, by output width.
    pub(crate) fn vector_threads(output: usize) -> usize {
        if output <= 8192 { 64 } else { 256 }
    }

    pub fn dense(&mut self, tensor: crate::tensor::Tensor) -> Result<super::operands::DenseMatrix> {
        use super::operands::{DenseMatrix, DensePlan, Tile};
        use WeightFormat::*;
        let [input, output, _] = tensor.shape;
        let kind = tensor.kind;
        let vector_rows = match (kind, input, output) {
            (Q4_0 | Q4_1, _, _) => 4,
            (Q5K, 6144, 5120) => 4,
            // Keeping tiny recurrent alpha/beta coefficients in FP32 across Qwen sizes.
            (F32, _, 1..=128) => super::state::MAX_PREFILL_ROWS,
            (Q8_0, 2880, 512) => 128,
            (Q8_0, 2880, 4096) => 4,
            (Q8_0, 4096, 2880) => 4,
            _ => 1,
        };
        let vector = self.vector(kind, output)?;
        let q4 = matches!(kind, Q4_0 | Q4_1);
        // Sharing row gains across scaled Q4 tiles; small products avoid the extra scan.
        let input_scales = q4 && output > 1024;
        let symbol = match (q4, input_scales) {
            (true, true) => "matmul_q4_half",
            (true, false) => "matmul_q4",
            _ => "matmul",
        };
        let column_groups = output.div_ceil(64);
        let plan = if vector_rows >= super::state::MAX_PREFILL_ROWS {
            DensePlan::Vector(vector)
        } else {
            let tile = |builder: &mut Self, variant: u32, rows: usize| -> Result<Tile> {
                Ok(Tile {
                    pipeline: builder.get(symbol, kind, variant)?,
                    column_groups,
                    rows,
                })
            };
            let wide = tile(self, 0, 64)?;
            let narrow = [
                tile(self, 1, 8)?,
                tile(self, 2, 16)?,
                tile(self, 3, 24)?,
                tile(self, 4, 32)?,
            ];
            DensePlan::Tiled {
                vector,
                vector_rows,
                input_scales,
                narrow,
                wide,
            }
        };
        Ok(DenseMatrix {
            values: tensor.storage,
            input,
            output,
            plan,
        })
    }

    /// Kernels for causal attention over up to `context` positions.
    pub fn attention(&mut self, dim: usize, context: usize) -> Result<AttentionPipelines> {
        self.attention_with(dim, context, false)
    }

    /// Kernels for an encoder's bidirectional attention, which adds the 32-query block tiles.
    pub fn encoder_attention(&mut self, dim: usize, context: usize) -> Result<AttentionPipelines> {
        self.attention_with(dim, context, true)
    }

    fn attention_with(
        &mut self,
        dim: usize,
        context: usize,
        bidirectional: bool,
    ) -> Result<AttentionPipelines> {
        use WeightFormat::F32;
        let rope = self.get("rope", F32, 0)?;
        let dispatch = match dim {
            64 => AttentionDispatch::Vector(self.get("attention_vector_64", F32, 0)?),
            128 => AttentionDispatch::Vector(self.get("attention_vector_128", F32, 0)?),
            256 => AttentionDispatch::Vector(self.get("attention_vector_256", F32, 0)?),
            _ => AttentionDispatch::Plain(self.get("attention", F32, 0)?),
        };
        let tiled = if context >= 8 && dim <= 256 && dim.is_multiple_of(32) {
            Some(self.get("attention_tiled", F32, 0)?)
        } else {
            None
        };
        let wide = if context >= 512 && matches!(dim, 64 | 256) {
            Some(self.get("attention_wide16", F32, 0)?)
        } else {
            None
        };
        let block = match dim {
            64 if bidirectional && context >= 8 => Some(self.get("attention_block_64", F32, 0)?),
            128 if bidirectional && context >= 8 => {
                Some(self.get("attention_block_128", F32, 0)?)
            }
            _ => None,
        };
        let split = if context >= super::compute::AttentionPlan::SPLIT_MIN_KEYS
            && matches!(dim, 64 | 96 | 128 | 256)
        {
            Some(SplitAttentionPipelines {
                partial: self.get("attention_gqa_split", F32, 0)?,
                merge: self.get("attention_gqa_merge", F32, 0)?,
            })
        } else {
            None
        };
        Ok(AttentionPipelines {
            rope,
            dispatch,
            tiled,
            wide,
            block,
            split,
        })
    }

    pub fn gated_attention(&mut self) -> Result<GatedAttentionPipelines> {
        use WeightFormat::F32;
        let qk_norm = self.get("qk_norm_pair", F32, 0)?;
        let gate = self.get("attention_gate", F32, 0)?;
        Ok(GatedAttentionPipelines { qk_norm, gate })
    }

    pub fn recurrent(&mut self) -> Result<RecurrentPipelines> {
        use WeightFormat::F32;
        let conv = self.get("conv", F32, 0)?;
        let delta = self.get("delta", F32, 0)?;
        let delta_prepare = self.get("delta_prepare", F32, 0)?;
        let delta_persistent = self.get("delta_persistent", F32, 0)?;
        let delta_output = self.get("delta_output", F32, 0)?;
        Ok(RecurrentPipelines {
            conv,
            delta,
            delta_prepare,
            delta_persistent,
            delta_output,
        })
    }

    pub fn experts(&mut self) -> Result<ExpertPipelines> {
        use WeightFormat::{F32, Mxfp4};
        let route_plan = self.get("route_plan", F32, 0)?;
        let up_tensor = self.get("expert_up_tensor", Mxfp4, 0)?;
        let down_tensor = self.get("expert_down_tensor", Mxfp4, 0)?;
        let expand = self.get("expert_expand", F32, 0)?;
        let down_vector = self.get("matvec_mxfp4", Mxfp4, 0)?;
        let router = self.get("router", F32, 0)?;
        let router_logits = self.get("router_logits", F32, 0)?;
        let sum = self.get("moe_sum", F32, 0)?;
        Ok(ExpertPipelines {
            router_logits,
            route_plan,
            up_tensor,
            down_tensor,
            expand,
            down_vector,
            router,
            sum,
        })
    }

    pub fn nemotron(&mut self) -> Result<NemotronPipelines> {
        use WeightFormat::{F32, Q4_0};
        Ok(NemotronPipelines {
            conv: self.get("ssm_conv", F32, 0)?,
            scan: self.get("mamba_scan", F32, 0)?,
            gate_norm: self.get("mamba_gate_norm", F32, 0)?,
            router: self.get("router_sigmoid", F32, 0)?,
            relu_sqr: self.get("relu_sqr", F32, 0)?,
            moe_add: self.get("moe_add", F32, 0)?,
            expert_vector: self.get("matvec_q4", Q4_0, 0)?,
            expert_vector_relu: self.get_relu_sqr("matvec_q4", Q4_0)?,
            route_plan: self.get("route_plan_n", F32, 0)?,
            expert_up: [
                self.get("expert_q4_up_8", Q4_0, 0)?,
                self.get("expert_q4_tensor_up", Q4_0, 0)?,
            ],
            expert_down: [
                self.get_relu_sqr("expert_q4_down_8", Q4_0)?,
                self.get_relu_sqr("expert_q4_tensor_down", Q4_0)?,
            ],
        })
    }

    pub fn audio(&mut self) -> Result<AudioPipelines> {
        use WeightFormat::F32;
        Ok(AudioPipelines {
            layer_norm: self.get("layer_norm", F32, 0)?,
            add_layer_norm: self.get("add_layer_norm", F32, 0)?,
            gelu: self.get("gelu", F32, 0)?,
            add_rows: self.get("add_rows", F32, 0)?,
            im2col: self.get("im2col", F32, 0)?,
        })
    }

    /// Elementary kernels with the normalization policy of one `width`-wide residual stream.
    pub fn kernels(&mut self, width: usize, epsilon: f32, full_layer: bool) -> Result<Kernels> {
        Ok(Kernels {
            pipelines: self.elementwise()?,
            norm: Norm {
                n: width as u32,
                epsilon,
                full_layer: full_layer as u32,
            },
        })
    }

    pub fn elementwise(&mut self) -> Result<ElementwisePipelines> {
        use WeightFormat::F32;
        let norm = self.get("norm", F32, 0)?;
        let add_norm = self.get("add_norm", F32, 0)?;
        let q4_input_scales = self.get("q4_input_scales", F32, 0)?;
        let add = self.get("add", F32, 0)?;
        let gate = self.get("gate", F32, 0)?;
        let argmax = self.get("argmax_f32", F32, 0)?;
        Ok(ElementwisePipelines {
            norm,
            add_norm,
            q4_input_scales,
            add,
            gate,
            argmax,
        })
    }
}

#[cfg(test)]
pub(super) mod tests;
