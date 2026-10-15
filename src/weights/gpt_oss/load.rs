//! Validate and bind GPT-OSS weights before constructing its concrete model.
mod selected;
use super::{ACTIVE_EXPERTS, EXPERT_COUNT, Experts, GptOss, SinkAttention};
use crate::{
    gguf::Gguf,
    gpu::Gpu,
    tensor::WeightFormat,
    weights::{
        Architecture,
        common::{AttentionProjections, CommonWeights, Layer},
        compute::{self, AttentionPlan, Norm, Rope},
        load::{Geometry, Import, Parameters, Selection, common_selection, common_weights, import},
        operands::{ExpertWeight, Router},
    },
};
use anyhow::{Result, ensure};
use std::sync::Arc;

impl<'a> Selection<'a> {
    fn expert(&mut self, name: &str, input: usize, output: usize) -> Result<selected::Expert<'a>> {
        let values = self.tensor(&format!("{name}.weight"))?;
        ensure!(
            input.is_multiple_of(64)
                && values.shape == [input, output, EXPERT_COUNT]
                && values.kind == WeightFormat::Mxfp4,
            "invalid expert coefficients {name}"
        );
        let bias = self.float(&format!("{name}.bias"), &[output, EXPERT_COUNT])?;
        Ok(selected::Expert { values, bias })
    }

    fn router(&mut self, name: &str, hidden: usize) -> Result<selected::Router<'a>> {
        let values = self.matrix(name, hidden, EXPERT_COUNT)?;
        ensure!(
            values.kind == WeightFormat::F32,
            "router weights must be float32"
        );
        let bias = self.float(&format!("{name}.bias"), &[EXPERT_COUNT])?;
        Ok(selected::Router { values, bias })
    }
}

impl Import {
    fn expert(
        &mut self,
        selected::Expert { values, bias }: selected::Expert<'_>,
    ) -> Result<ExpertWeight> {
        let values = self.binding(values)?;
        let bias = self.binding(bias)?;
        Ok(ExpertWeight { values, bias })
    }
}

pub fn load(file: &Gguf, p: &Parameters<'_>, g: Geometry) -> Result<Architecture> {
    p.scaling("yarn")?;
    let rope_dim = p.optional("rope.dimension_count", g.dim as f64)?;
    ensure!(rope_dim == g.dim as f64, "GPT-OSS requires full-head RoPE");
    let factor = p.number("rope.scaling.factor")?;
    ensure!(
        factor >= 1.0 && factor <= f32::MAX as f64,
        "invalid RoPE scale"
    );
    let factor = factor as f32;
    let original = p.number("rope.scaling.original_context_length")?;
    ensure!(
        original > 0.0 && original <= u32::MAX as f64 && original.fract() == 0.0,
        "invalid original RoPE context"
    );
    let fast = p.optional("rope.scaling.yarn_beta_fast", 32.0)?;
    let slow = p.optional("rope.scaling.yarn_beta_slow", 1.0)?;
    ensure!(fast > 0.0 && slow > 0.0, "invalid YaRN beta");
    let correction = |beta: f64| {
        g.dim as f64 * (original / (beta * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * (g.base as f64).ln())
    };
    let low = correction(fast).max(0.0) as f32;
    let high = correction(slow).min((g.dim - 1) as f64) as f32;
    ensure!(
        low.is_finite() && high.is_finite() && high > low,
        "invalid YaRN correction range"
    );
    let window = p.number("attention.sliding_window")?;
    ensure!(
        window >= 0.0 && window <= u32::MAX as f64 && window.fract() == 0.0,
        "invalid sliding window"
    );
    let window = window as usize;
    let expert_count = p.integer("expert_count")?;
    let expert_used = p.integer("expert_used_count")?;
    ensure!(
        expert_count == EXPERT_COUNT && expert_used == ACTIVE_EXPERTS,
        "GPT-OSS currently requires 32 experts, 4 active"
    );
    let rope = Rope {
        heads: g.heads as u32,
        kv_heads: g.kv as u32,
        dim: g.dim as u32,
        rope_dim: g.dim as u32,
        m: 0,
        pos: 0,
        q_stride: 1,
        base: g.base,
        factor,
        low,
        high,
        scale: 1.0 + 0.1 * factor.ln(),
    };
    let directory = file.directory()?;
    let (mut s, common) = common_selection(&directory, &g)?;
    let mut selected: Vec<selected::GptLayer<'_>> = Vec::with_capacity(g.count);
    for index in 0..g.count {
        let prefix = format!("blk.{index}");
        let norm = s.float(&format!("{prefix}.attn_norm.weight"), &[g.hidden])?;
        let post_norm = s.float(&format!("{prefix}.post_attention_norm.weight"), &[g.hidden])?;
        let q = s.biased(&format!("{prefix}.attn_q"), g.hidden, g.heads * g.dim)?;
        let k = s.biased(&format!("{prefix}.attn_k"), g.hidden, g.kv * g.dim)?;
        let v = s.biased(&format!("{prefix}.attn_v"), g.hidden, g.kv * g.dim)?;
        let out = s.biased(&format!("{prefix}.attn_output"), g.heads * g.dim, g.hidden)?;
        let sinks = s.float(&format!("{prefix}.attn_sinks.weight"), &[g.heads])?;
        let mixer = selected::SinkAttention {
            projections: AttentionProjections { q, k, v, out },
            sinks,
            window: if index % 2 == 0 { window } else { 0 },
        };
        let gate = s.expert(&format!("{prefix}.ffn_gate_exps"), g.hidden, g.ffn_width)?;
        let up = s.expert(&format!("{prefix}.ffn_up_exps"), g.hidden, g.ffn_width)?;
        let down = s.expert(&format!("{prefix}.ffn_down_exps"), g.ffn_width, g.hidden)?;
        let router = s.router(&format!("{prefix}.ffn_gate_inp"), g.hidden)?;
        let ff = selected::Experts {
            gate,
            up,
            down,
            router,
        };
        selected.push(selected::Layer {
            norm,
            post_norm,
            mixer,
            ff,
        });
    }
    let gpu = Gpu::new()?;
    let mut i = import(file, s.spans, &gpu)?;
    let (embedding, output, output_norm, pipelines) = common_weights(&mut i, common)?;
    let attention = i.builder.attention(g.dim, g.context)?;
    let attention = AttentionPlan::new(rope, attention);
    let experts = i.builder.experts()?;
    let experts = compute::ExpertPlan::new(g.hidden, g.ffn_width, experts);
    let mut layers = Vec::with_capacity(g.count);
    for selected::Layer {
        norm,
        post_norm,
        mixer:
            selected::SinkAttention {
                projections: AttentionProjections { q, k, v, out },
                sinks,
                window,
            },
        ff:
            selected::Experts {
                gate,
                up,
                down,
                router: selected::Router { values, bias },
            },
    } in selected
    {
        let norm = i.binding(norm)?;
        let post_norm = i.binding(post_norm)?;
        let q = i.biased(q)?;
        let k = i.biased(k)?;
        let v = i.biased(v)?;
        let out = i.biased(out)?;
        let sinks = i.binding(sinks)?;
        let mixer = SinkAttention {
            projections: AttentionProjections { q, k, v, out },
            sinks,
            window,
        };
        let gate = i.expert(gate)?;
        let up = i.expert(up)?;
        let down = i.expert(down)?;
        let values = i.binding(values)?;
        let bias = i.binding(bias)?;
        let router = Router { values, bias };
        layers.push(Layer {
            norm,
            post_norm,
            mixer,
            ff: Experts {
                gate,
                up,
                down,
                router,
            },
        });
    }
    let norm = Norm {
        n: g.hidden as u32,
        epsilon: g.epsilon,
        full_layer: 1,
    };
    let residency = i.finish()?;
    let base = CommonWeights {
        gpu,
        residency: Some(residency),
        context: g.context,
        embedding,
        output,
        output_norm,
        norm,
        pipelines,
    };
    let weights = GptOss {
        base,
        layers,
        attention,
        experts,
    };
    Ok(Architecture::GptOss(Arc::new(weights)))
}
