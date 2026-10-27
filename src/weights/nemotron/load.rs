//! Validate Nemotron-H MoE weights and per-layer topology before binding its concrete model.
mod selected;
use super::{Mamba, Mixer, Nemotron, NemotronLayer, ReluExperts};
use crate::{
    gguf::Gguf,
    gpu::Gpu,
    tensor::WeightFormat,
    weights::{
        Architecture,
        common::{AttentionProjections, CommonWeights},
        compute::{AttentionPlan, MambaPlan, Norm, ReluExpertPlan, Rope},
        load::{Geometry, Import, Parameters, Selection, common_selection, common_weights, import},
        operands::{DenseMatrix, DensePlan, Projection, Router},
    },
};
use anyhow::{Context, Result, ensure};
use std::sync::Arc;

const PREFIX: &str = "nemotron_h_moe";

/// Block-independent dimensions shared by every mixer of one checkpoint.
struct Topology {
    ssm_heads: usize,
    head_dim: usize,
    groups: usize,
    inner: usize,
    conv_width: usize,
    projected_width: usize,
    experts: usize,
    expert_width: usize,
    shared_width: usize,
}

impl<'a> Selection<'a> {
    fn mamba(&mut self, prefix: &str, g: &Geometry, t: &Topology) -> Result<selected::Mamba<'a>> {
        let in_proj = self.projection_with_optional_bias(
            &format!("{prefix}.ssm_in"),
            g.hidden,
            t.projected_width,
        )?;
        let out =
            self.projection_with_optional_bias(&format!("{prefix}.ssm_out"), t.inner, g.hidden)?;
        let conv = self.float(&format!("{prefix}.ssm_conv1d.weight"), &[4, t.conv_width])?;
        let conv_bias = self.float(&format!("{prefix}.ssm_conv1d.bias"), &[t.conv_width])?;
        let a = self.float(&format!("{prefix}.ssm_a"), &[1, t.ssm_heads])?;
        let d = self.float(&format!("{prefix}.ssm_d"), &[1, t.ssm_heads])?;
        let dt_bias = self.float(&format!("{prefix}.ssm_dt.bias"), &[t.ssm_heads])?;
        let norm = self.float(
            &format!("{prefix}.ssm_norm.weight"),
            &[t.inner / t.groups, t.groups],
        )?;
        Ok(selected::Mamba {
            in_proj,
            out,
            conv,
            conv_bias,
            a,
            d,
            dt_bias,
            norm,
        })
    }

    fn plain_attention(
        &mut self,
        prefix: &str,
        g: &Geometry,
    ) -> Result<AttentionProjections<selected::Dense<'a>>> {
        let q = self.projection_with_optional_bias(
            &format!("{prefix}.attn_q"),
            g.hidden,
            g.heads * g.dim,
        )?;
        let k =
            self.projection_with_optional_bias(&format!("{prefix}.attn_k"), g.hidden, g.kv * g.dim)?;
        let v =
            self.projection_with_optional_bias(&format!("{prefix}.attn_v"), g.hidden, g.kv * g.dim)?;
        let out = self.projection_with_optional_bias(
            &format!("{prefix}.attn_output"),
            g.heads * g.dim,
            g.hidden,
        )?;
        Ok(AttentionProjections { q, k, v, out })
    }

    fn relu_experts(
        &mut self,
        prefix: &str,
        g: &Geometry,
        t: &Topology,
    ) -> Result<selected::Experts<'a>> {
        let router = self.matrix(&format!("{prefix}.ffn_gate_inp"), g.hidden, t.experts)?;
        ensure!(
            router.kind == WeightFormat::F32,
            "router weights must be float32"
        );
        let router_bias = self.float(&format!("{prefix}.exp_probs_b.bias"), &[t.experts])?;
        let up = self.tensor(&format!("{prefix}.ffn_up_exps.weight"))?;
        ensure!(
            up.shape == [g.hidden, t.expert_width, t.experts] && up.kind == WeightFormat::Q4_0,
            "unsupported expert up coefficients in {prefix}"
        );
        let down = self.tensor(&format!("{prefix}.ffn_down_exps.weight"))?;
        ensure!(
            down.shape == [t.expert_width, g.hidden, t.experts] && down.kind == WeightFormat::Q4_0,
            "unsupported expert down coefficients in {prefix}"
        );
        let shared_up = self.projection_with_optional_bias(
            &format!("{prefix}.ffn_up_shexp"),
            g.hidden,
            t.shared_width,
        )?;
        let shared_down = self.projection_with_optional_bias(
            &format!("{prefix}.ffn_down_shexp"),
            t.shared_width,
            g.hidden,
        )?;
        Ok(selected::Experts {
            router,
            router_bias,
            up,
            down,
            shared_up,
            shared_down,
        })
    }
}

impl Import {
    fn mamba(
        &mut self,
        dt_rows_before: usize,
        selected::Mamba {
            in_proj,
            out,
            conv,
            conv_bias,
            a,
            d,
            dt_bias,
            norm,
        }: selected::Mamba<'_>,
    ) -> Result<Mamba> {
        let kind = in_proj.matrix.kind;
        let in_proj = self.projection_with_optional_bias(in_proj)?;
        // The last `heads` output rows hold the time steps; alias them as an FP32 product.
        let hidden = in_proj.matrix.input;
        let (block, size) = kind.block_layout();
        let row_bytes = hidden / block * size;
        let heads = in_proj.matrix.output - dt_rows_before;
        let values = in_proj.matrix.values.slice(dt_rows_before * row_bytes, heads * row_bytes);
        let dt = DenseMatrix {
            values,
            input: hidden,
            output: heads,
            plan: DensePlan::Vector(self.builder.vector_with(kind, heads, true)?),
        };
        let out = self.projection_with_optional_bias(out)?;
        let conv = self.binding(conv)?;
        let conv_bias = self.binding(conv_bias)?;
        let a = self.binding(a)?;
        let d = self.binding(d)?;
        let dt_bias = self.binding(dt_bias)?;
        let norm = self.binding(norm)?;
        Ok(Mamba {
            in_proj,
            dt,
            out,
            conv,
            conv_bias,
            a,
            d,
            dt_bias,
            norm,
        })
    }

    fn plain_attention(
        &mut self,
        AttentionProjections { q, k, v, out }: AttentionProjections<selected::Dense<'_>>,
    ) -> Result<AttentionProjections<Projection>> {
        let q = self.projection_with_optional_bias(q)?;
        let k = self.projection_with_optional_bias(k)?;
        let v = self.projection_with_optional_bias(v)?;
        let out = self.projection_with_optional_bias(out)?;
        Ok(AttentionProjections { q, k, v, out })
    }

    fn relu_experts(
        &mut self,
        selected::Experts {
            router,
            router_bias,
            up,
            down,
            shared_up,
            shared_down,
        }: selected::Experts<'_>,
    ) -> Result<ReluExperts> {
        let values = self.binding(router)?;
        let bias = self.binding(router_bias)?;
        let up = self.binding(up)?;
        let down = self.binding(down)?;
        let shared_up = self.projection_with_optional_bias(shared_up)?;
        let shared_down = self.projection_with_optional_bias(shared_down)?;
        Ok(ReluExperts {
            router: Router { values, bias },
            up,
            down,
            shared_up,
            shared_down,
        })
    }
}

pub fn load(file: &Gguf, p: &Parameters<'_>) -> Result<Architecture> {
    let hidden = p.integer("embedding_length")?;
    let heads = p.integer("attention.head_count")?;
    let dim = p.integer("attention.key_length")?;
    let value_dim = p.optional("attention.value_length", dim as f64)?;
    ensure!(value_dim == dim as f64, "key and value widths must match");
    let context = p.integer("context_length")?;
    let epsilon = p.number("attention.layer_norm_rms_epsilon")? as f32;
    let count = p.integer("block_count")?;
    let base = p.optional("rope.freq_base", 10000.0)? as f32;
    let kv_layers = file.unsigned_integers(&format!("{PREFIX}.attention.head_count_kv"))?;
    let ff_layers = file.unsigned_integers(&format!("{PREFIX}.feed_forward_length"))?;
    ensure!(
        kv_layers.len() == count && ff_layers.len() == count,
        "per-layer topology arrays must cover every block"
    );
    let kv = kv_layers
        .iter()
        .copied()
        .find(|&kv| kv > 0)
        .context("Nemotron requires at least one attention block")?;
    let kv = usize::try_from(kv)?;
    ensure!(
        kv_layers.iter().all(|&layer| layer == 0 || layer as usize == kv),
        "attention blocks must share one key/value head count"
    );
    let expert_width = p.integer("expert_feed_forward_length")?;
    ensure!(
        ff_layers
            .iter()
            .all(|&layer| layer == 0 || layer as usize == expert_width),
        "expert blocks must share one expert width"
    );
    ensure!(
        hidden <= 16384
            && hidden.is_multiple_of(128)
            && expert_width.is_multiple_of(64)
            && matches!(dim, 64 | 128 | 256)
            && heads <= 128
            && heads.is_multiple_of(kv)
            && count <= 128
            && epsilon.is_finite()
            && epsilon > 0.0,
        "unsupported Nemotron geometry"
    );
    let g = Geometry {
        hidden,
        heads,
        kv,
        dim,
        ffn_width: expert_width,
        context,
        epsilon,
        base,
        count,
    };
    let experts = p.integer("expert_count")?;
    let active = p.integer("expert_used_count")?;
    ensure!(
        experts <= 128 && active <= 8 && active <= experts,
        "unsupported expert routing size"
    );
    let shared_width = p.integer("expert_shared_feed_forward_length")?;
    ensure!(
        shared_width.is_multiple_of(64) && shared_width <= active * expert_width,
        "unsupported shared expert width"
    );
    let shared_count = p.optional("expert_shared_count", 1.0)?;
    ensure!(shared_count == 1.0, "Nemotron requires one shared expert");
    let normalize_key = format!("{PREFIX}.expert_weights_norm");
    let normalize = if file.meta.contains_key(&normalize_key) {
        file.boolean(&normalize_key)?
    } else {
        false
    };
    let scale = p.optional("expert_weights_scale", 1.0)? as f32;
    ensure!(scale.is_finite() && scale > 0.0, "invalid expert weight scale");
    let groups = p.integer("ssm.group_count")?;
    let ssm_heads = p.integer("ssm.time_step_rank")?;
    let inner = p.integer("ssm.inner_size")?;
    let state = p.integer("ssm.state_size")?;
    let taps = p.integer("ssm.conv_kernel")?;
    ensure!(
        state == MambaPlan::STATE
            && taps == 4
            && inner.is_multiple_of(ssm_heads)
            && inner.is_multiple_of(groups)
            && ssm_heads.is_multiple_of(groups)
            && inner.is_multiple_of(64)
            && inner / groups <= 512
            && (inner / ssm_heads).is_multiple_of(4),
        "unsupported Mamba-2 geometry"
    );
    let t = Topology {
        ssm_heads,
        head_dim: inner / ssm_heads,
        groups,
        inner,
        conv_width: inner + 2 * groups * state,
        projected_width: 2 * inner + 2 * groups * state + ssm_heads,
        experts,
        expert_width,
        shared_width,
    };
    let rope = Rope {
        heads: heads as u32,
        kv_heads: kv as u32,
        dim: dim as u32,
        // Nemotron attention carries no positional rotation.
        rope_dim: 0,
        m: 0,
        pos: 0,
        q_stride: 1,
        base,
        factor: 1.0,
        low: 0.0,
        high: 1.0,
        scale: 1.0,
    };
    let directory = file.directory()?;
    let (mut s, common) = common_selection(&directory, &g)?;
    let mut selected = Vec::with_capacity(count);
    let (mut attention_layers, mut mamba_layers) = (0, 0);
    for index in 0..count {
        let prefix = format!("blk.{index}");
        let norm = s.float(&format!("{prefix}.attn_norm.weight"), &[hidden])?;
        let mixer = if kv_layers[index] > 0 {
            let weights = s.plain_attention(&prefix, &g)?;
            attention_layers += 1;
            selected::Mixer::Attention {
                weights,
                slot: attention_layers - 1,
            }
        } else if ff_layers[index] > 0 {
            selected::Mixer::Experts(s.relu_experts(&prefix, &g, &t)?)
        } else {
            let weights = s.mamba(&prefix, &g, &t)?;
            mamba_layers += 1;
            selected::Mixer::Mamba {
                weights,
                slot: mamba_layers - 1,
            }
        };
        selected.push(selected::Layer { norm, mixer });
    }
    let gpu = Gpu::new()?;
    let mut i = import(file, s.spans, &gpu)?;
    let (embedding, output, output_norm, pipelines) = common_weights(&mut i, common)?;
    let attention = i.builder.attention(dim, context)?;
    let attention = AttentionPlan::new(rope, attention);
    let nemotron = Arc::new(i.builder.nemotron()?);
    let mamba = MambaPlan::new(ssm_heads, t.head_dim, groups, epsilon, nemotron.clone());
    let shared_kind = selected
        .iter()
        .find_map(|layer| match &layer.mixer {
            selected::Mixer::Experts(experts) => Some(experts.shared_up.matrix.kind),
            _ => None,
        })
        .context("Nemotron requires at least one expert block")?;
    ensure!(
        selected.iter().all(|layer| match &layer.mixer {
            selected::Mixer::Experts(experts) =>
                experts.shared_up.matrix.kind == shared_kind
                    && experts.shared_down.matrix.kind == shared_kind,
            _ => true,
        }),
        "shared experts must share one weight encoding"
    );
    let shared_vectors = i.builder.vector_pair(shared_kind)?;
    let experts = ReluExpertPlan::new(
        hidden,
        expert_width,
        shared_width,
        experts,
        active,
        normalize,
        scale,
        nemotron,
        shared_vectors,
    );
    let mut layers = Vec::with_capacity(count);
    for selected::Layer { norm, mixer } in selected {
        let norm = i.binding(norm)?;
        let mixer = match mixer {
            selected::Mixer::Mamba { weights, slot } => Mixer::Mamba {
                weights: i.mamba(t.projected_width - t.ssm_heads, weights)?,
                slot,
            },
            selected::Mixer::Attention { weights, slot } => Mixer::Attention {
                weights: i.plain_attention(weights)?,
                slot,
            },
            selected::Mixer::Experts(weights) => Mixer::Experts(i.relu_experts(weights)?),
        };
        layers.push(NemotronLayer { norm, mixer });
    }
    let norm = Norm {
        n: hidden as u32,
        epsilon,
        full_layer: 1,
    };
    let residency = i.finish()?;
    let base = CommonWeights {
        gpu,
        residency: Some(residency),
        context,
        embedding,
        output,
        output_norm,
        norm,
        pipelines,
    };
    Ok(Architecture::Nemotron(Arc::new(Nemotron {
        base,
        layers,
        attention,
        mamba,
        experts,
        attention_layers,
        mamba_layers,
    })))
}
