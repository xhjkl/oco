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
    pub split: Option<SplitAttentionPipelines>,
}

pub struct SplitAttentionPipelines {
    pub partial: Pipeline,
    pub merge: Pipeline,
}

pub enum AttentionDispatch {
    Plain(Pipeline),
    Compact256(Pipeline),
    Dim64 {
        full: Pipeline,
        compact_256: Pipeline,
    },
    Dim256 {
        full: Pipeline,
        compact_256: Pipeline,
    },
}

pub struct GatedAttentionPipelines {
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

pub struct ElementwisePipelines {
    pub norm: Pipeline,
    pub q4_input_scales: Pipeline,
    pub add: Pipeline,
    pub gate: Pipeline,
    pub argmax: Pipeline,
}

pub struct ExpertPipelines {
    pub router_logits: Pipeline,
    pub route_plan: Pipeline,
    pub route_plan_large: Pipeline,
    pub up: Pipeline,
    pub up_large: Pipeline,
    pub expand: Pipeline,
    pub down: Pipeline,
    pub down_large: Pipeline,
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
        if let Some(pipeline) = self.cache.get(&(symbol, format, variant)) {
            return Ok(pipeline.clone());
        }
        let pipeline = Gpu::new_pipeline(
            &self.compiler,
            &self.library,
            symbol,
            &[(0, format as u32), (1, variant)],
        )?;
        self.cache
            .insert((symbol, format, variant), pipeline.clone());
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
        let vector = self.vector(tensor.kind, vocabulary)?;
        Ok(super::operands::Output {
            values: tensor.storage,
            bias,
            vector,
            width,
            vocabulary,
        })
    }

    fn vector(&mut self, kind: WeightFormat, output: usize) -> Result<super::operands::Vector> {
        use WeightFormat::*;
        let (symbol, column_groups) = match kind {
            Q4_0 | Q4_1 => ("matvec_q4", output.div_ceil(32)),
            Q5K => ("matvec_q5k", output.div_ceil(16)),
            Q6K => ("matvec_q6k", output.div_ceil(16)),
            F32 | F16 | Bf16 if output < 32768 => ("matvec_split", output),
            _ => ("matvec", output.div_ceil(32)),
        };
        let pipeline = self.get(symbol, kind, 0)?;
        Ok(super::operands::Vector {
            pipeline,
            column_groups,
        })
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

    pub fn attention(&mut self, dim: usize, context: usize) -> Result<AttentionPipelines> {
        use WeightFormat::F32;
        let rope = self.get("rope", F32, 0)?;
        let dispatch = match dim {
            64 if context == 1 => {
                let compact = self.get("attention_256", F32, 0)?;
                AttentionDispatch::Compact256(compact)
            }
            256 if context <= 1024 => {
                let compact = self.get("attention_256", F32, 0)?;
                AttentionDispatch::Compact256(compact)
            }
            64 => {
                let full = self.get("attention", F32, 0)?;
                let compact_256 = self.get("attention_256", F32, 0)?;
                AttentionDispatch::Dim64 { full, compact_256 }
            }
            256 => {
                let full = self.get("attention", F32, 0)?;
                let compact_256 = self.get("attention_256", F32, 0)?;
                AttentionDispatch::Dim256 { full, compact_256 }
            }
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
            split,
        })
    }

    pub fn gated_attention(&mut self) -> Result<GatedAttentionPipelines> {
        use WeightFormat::F32;
        let qk_norm = self.get("qk_norm", F32, 0)?;
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
        let route_plan_large = self.get("route_plan_large", F32, 0)?;
        let up = self.get("expert_up", Mxfp4, 0)?;
        let up_large = self.get("expert_up_large", Mxfp4, 0)?;
        let expand = self.get("expert_expand", F32, 0)?;
        let down = self.get("expert_down", Mxfp4, 0)?;
        let down_large = self.get("expert_down_large", Mxfp4, 0)?;
        let down_vector = self.get("matvec_mxfp4", Mxfp4, 0)?;
        let router = self.get("router", F32, 0)?;
        let router_logits = self.get("router_logits", F32, 0)?;
        let sum = self.get("moe_sum", F32, 0)?;
        Ok(ExpertPipelines {
            router_logits,
            route_plan,
            route_plan_large,
            up,
            up_large,
            expand,
            down,
            down_large,
            down_vector,
            router,
            sum,
        })
    }

    pub fn elementwise(&mut self) -> Result<ElementwisePipelines> {
        use WeightFormat::F32;
        let norm = self.get("norm", F32, 0)?;
        let q4_input_scales = self.get("q4_input_scales", F32, 0)?;
        let add = self.get("add", F32, 0)?;
        let gate = self.get("gate", F32, 0)?;
        let argmax = self.get("argmax", F32, 0)?;
        Ok(ElementwisePipelines {
            norm,
            q4_input_scales,
            add,
            gate,
            argmax,
        })
    }
}

#[cfg(test)]
pub(super) mod tests;
