//! Validate Qwen weights and topology before binding its concrete model.
mod selected;
use super::delta::{CONV_TAPS, KEY_WIDTH};
use super::{
    DenseFfn, QwenLayer,
    attention::{GatedAttention, QwenAttention},
    delta::{Delta, QwenDelta},
    hybrid::{AttentionSlot, DeltaSlot, HybridMixer, QwenHybrid},
};
use crate::{
    gguf::Gguf,
    gpu::Gpu,
    weights::{
        Architecture,
        common::{AttentionProjections, CommonWeights, Layer},
        compute::{AttentionPlan, DeltaPlan, Norm, Rope},
        load::{Geometry, Import, Parameters, Selection, common_selection, common_weights, import},
    },
};
use anyhow::{Context, Result, ensure};
use std::sync::Arc;

impl<'a> Selection<'a> {
    fn gated_attention(
        &mut self,
        prefix: &str,
        g: &Geometry,
    ) -> Result<selected::GatedAttention<'a>> {
        let q = self.projection_with_optional_bias(
            &format!("{prefix}.attn_q"),
            g.hidden,
            g.heads * g.dim * 2,
        )?;
        let k = self.projection_with_optional_bias(
            &format!("{prefix}.attn_k"),
            g.hidden,
            g.kv * g.dim,
        )?;
        let v = self.projection_with_optional_bias(
            &format!("{prefix}.attn_v"),
            g.hidden,
            g.kv * g.dim,
        )?;
        let out = self.projection_with_optional_bias(
            &format!("{prefix}.attn_output"),
            g.heads * g.dim,
            g.hidden,
        )?;
        let qnorm = self.float(&format!("{prefix}.attn_q_norm.weight"), &[g.dim])?;
        let knorm = self.float(&format!("{prefix}.attn_k_norm.weight"), &[g.dim])?;
        Ok(selected::GatedAttention {
            projections: AttentionProjections { q, k, v, out },
            qnorm,
            knorm,
        })
    }

    fn delta(
        &mut self,
        prefix: &str,
        hidden: usize,
        kh: usize,
        vh: usize,
        vd: usize,
    ) -> Result<selected::Delta<'a>> {
        let conv_dim = 2 * kh * KEY_WIDTH + vh * vd;
        let qkv =
            self.projection_with_optional_bias(&format!("{prefix}.attn_qkv"), hidden, conv_dim)?;
        let z =
            self.projection_with_optional_bias(&format!("{prefix}.attn_gate"), hidden, vh * vd)?;
        let alpha =
            self.projection_with_optional_bias(&format!("{prefix}.ssm_alpha"), hidden, vh)?;
        let beta = self.projection_with_optional_bias(&format!("{prefix}.ssm_beta"), hidden, vh)?;
        let out =
            self.projection_with_optional_bias(&format!("{prefix}.ssm_out"), vh * vd, hidden)?;
        let conv = self.float(
            &format!("{prefix}.ssm_conv1d.weight"),
            &[CONV_TAPS, conv_dim],
        )?;
        let a = self.float(&format!("{prefix}.ssm_a"), &[vh])?;
        let dt = self.float(&format!("{prefix}.ssm_dt.bias"), &[vh])?;
        let norm = self.float(&format!("{prefix}.ssm_norm.weight"), &[vd])?;
        Ok(selected::Delta {
            qkv,
            z,
            alpha,
            beta,
            out,
            conv,
            a,
            dt,
            norm,
        })
    }

    fn qwen_layers<M>(
        &mut self,
        g: &Geometry,
        mut mixer: impl FnMut(&mut Self, usize, &str) -> Result<M>,
    ) -> Result<Vec<selected::QwenLayer<'a, M>>> {
        let mut layers = Vec::with_capacity(g.count);
        for index in 0..g.count {
            let prefix = format!("blk.{index}");
            let norm = self.float(&format!("{prefix}.attn_norm.weight"), &[g.hidden])?;
            let post_norm =
                self.float(&format!("{prefix}.post_attention_norm.weight"), &[g.hidden])?;
            let gate = self.projection_with_optional_bias(
                &format!("{prefix}.ffn_gate"),
                g.hidden,
                g.ffn_width,
            )?;
            let up = self.projection_with_optional_bias(
                &format!("{prefix}.ffn_up"),
                g.hidden,
                g.ffn_width,
            )?;
            let down = self.projection_with_optional_bias(
                &format!("{prefix}.ffn_down"),
                g.ffn_width,
                g.hidden,
            )?;
            let ff = selected::DenseFfn { gate, up, down };
            let mixer = mixer(self, index, &prefix)?;
            layers.push(selected::Layer {
                norm,
                post_norm,
                mixer,
                ff,
            });
        }
        Ok(layers)
    }
}

impl Import {
    fn feed_forward(
        &mut self,
        selected::DenseFfn { gate, up, down }: selected::DenseFfn<'_>,
    ) -> Result<DenseFfn> {
        let gate = self.projection_with_optional_bias(gate)?;
        let up = self.projection_with_optional_bias(up)?;
        let down = self.projection_with_optional_bias(down)?;
        Ok(DenseFfn { gate, up, down })
    }

    fn gated_attention(
        &mut self,
        selected::GatedAttention {
            projections: AttentionProjections { q, k, v, out },
            qnorm,
            knorm,
        }: selected::GatedAttention<'_>,
    ) -> Result<GatedAttention> {
        let q = self.projection_with_optional_bias(q)?;
        let k = self.projection_with_optional_bias(k)?;
        let v = self.projection_with_optional_bias(v)?;
        let out = self.projection_with_optional_bias(out)?;
        let qnorm = self.binding(qnorm)?;
        let knorm = self.binding(knorm)?;
        Ok(GatedAttention {
            projections: AttentionProjections { q, k, v, out },
            qnorm,
            knorm,
        })
    }

    fn delta(
        &mut self,
        selected::Delta {
            qkv,
            z,
            alpha,
            beta,
            out,
            conv,
            a,
            dt,
            norm,
        }: selected::Delta<'_>,
    ) -> Result<Delta> {
        let qkv = self.projection_with_optional_bias(qkv)?;
        let z = self.projection_with_optional_bias(z)?;
        let alpha = self.projection_with_optional_bias(alpha)?;
        let beta = self.projection_with_optional_bias(beta)?;
        let out = self.projection_with_optional_bias(out)?;
        let conv = self.binding(conv)?;
        let a = self.binding(a)?;
        let dt = self.binding(dt)?;
        let norm = self.binding(norm)?;
        Ok(Delta {
            qkv,
            z,
            alpha,
            beta,
            out,
            conv,
            a,
            dt,
            norm,
        })
    }

    fn qwen_layers<M, B>(
        &mut self,
        selected: Vec<selected::QwenLayer<'_, M>>,
        mut mixer: impl FnMut(&mut Self, M) -> Result<B>,
    ) -> Result<Vec<QwenLayer<B>>> {
        let mut layers = Vec::with_capacity(selected.len());
        for selected::Layer {
            norm,
            post_norm,
            mixer: weights,
            ff,
        } in selected
        {
            let norm = self.binding(norm)?;
            let post_norm = self.binding(post_norm)?;
            let mixer = mixer(self, weights)?;
            let ff = self.feed_forward(ff)?;
            layers.push(Layer {
                norm,
                post_norm,
                mixer,
                ff,
            });
        }
        Ok(layers)
    }
}

pub fn load(file: &Gguf, p: &Parameters<'_>, mut g: Geometry) -> Result<Architecture> {
    p.scaling("none")?;
    let factor = p.optional("rope.scaling.factor", 1.0)?;
    ensure!(factor == 1.0, "unsupported Qwen RoPE scaling factor");
    let rope_dim = p.integer("rope.dimension_count")?;
    ensure!(
        rope_dim <= g.dim && rope_dim.is_multiple_of(2),
        "invalid RoPE dimensions"
    );
    let kh = p.integer("ssm.group_count")?;
    let vh = p.integer("ssm.time_step_rank")?;
    let inner = p.integer("ssm.inner_size")?;
    ensure!(inner.is_multiple_of(vh), "invalid gated delta inner size");
    let vd = inner / vh;
    let state = p.integer("ssm.state_size")?;
    let conv = p.integer("ssm.conv_kernel")?;
    ensure!(
        state == KEY_WIDTH
            && conv == CONV_TAPS
            && vd <= KEY_WIDTH
            && vh <= 128
            && kh <= 128
            && vh.is_multiple_of(kh),
        "unsupported gated delta geometry"
    );
    let interval = p.integer("full_attention_interval")?;
    let mtp = file
        .optional_unsigned("qwen35.nextn_predict_layers")?
        .unwrap_or(0);
    let mtp = usize::try_from(mtp).context("MTP layer count overflow")?;
    ensure!(mtp < g.count, "invalid layer counts");
    g.count -= mtp;
    let attention_count = g.count / interval;
    let delta_count = g.count - attention_count;
    let rope = Rope {
        heads: g.heads as u32,
        kv_heads: g.kv as u32,
        dim: g.dim as u32,
        rope_dim: rope_dim as u32,
        m: 0,
        pos: 0,
        q_stride: 2,
        base: g.base,
        factor: 1.0,
        low: 0.0,
        high: 1.0,
        scale: 1.0,
    };
    let directory = file.directory()?;
    let (mut s, common) = common_selection(&directory, &g)?;
    let selected = if delta_count == 0 {
        let layers = s.qwen_layers(&g, |s, _, prefix| s.gated_attention(prefix, &g))?;
        selected::Qwen::Attention(layers)
    } else if attention_count == 0 {
        let layers = s.qwen_layers(&g, |s, _, prefix| s.delta(prefix, g.hidden, kh, vh, vd))?;
        selected::Qwen::Delta(layers)
    } else {
        let mut attention_layers = 0;
        let mut delta_layers = 0;
        let layers = s.qwen_layers(&g, |s, index, prefix| {
            let mixer = if (index + 1) % interval == 0 {
                let weights = s.gated_attention(prefix, &g)?;
                let slot = AttentionSlot(attention_layers);
                attention_layers += 1;
                selected::Mixer::Attention { weights, slot }
            } else {
                let weights = s.delta(prefix, g.hidden, kh, vh, vd)?;
                let slot = DeltaSlot(delta_layers);
                delta_layers += 1;
                selected::Mixer::Delta { weights, slot }
            };
            Ok(mixer)
        })?;
        selected::Qwen::Hybrid {
            layers,
            attention_layers,
            delta_layers,
        }
    };
    let gpu = Gpu::new()?;
    let mut i = import(file, s.spans, &gpu)?;
    let (embedding, output, output_norm, pipelines) = common_weights(&mut i, common)?;
    let norm = Norm {
        n: g.hidden as u32,
        epsilon: g.epsilon,
        full_layer: 0,
    };
    let base = |gpu, residency| CommonWeights {
        gpu,
        residency: Some(residency),
        context: g.context,
        embedding,
        output,
        output_norm,
        norm,
        pipelines,
    };
    let inner = match selected {
        selected::Qwen::Attention(layers) => {
            let attention = i.builder.attention(g.dim, g.context)?;
            let attention = AttentionPlan::new(rope, attention);
            let gated = i.builder.gated_attention()?;
            let layers = i.qwen_layers(layers, |i, weights| i.gated_attention(weights))?;
            let residency = i.finish()?;
            let base = base(gpu, residency);
            let weights = QwenAttention {
                base,
                layers,
                attention,
                gated,
                ffn_width: g.ffn_width,
            };
            Architecture::QwenAttention(Arc::new(weights))
        }
        selected::Qwen::Delta(layers) => {
            let recurrent = i.builder.recurrent()?;
            let delta = DeltaPlan::new(kh, vh, vd, g.epsilon, recurrent);
            let layers = i.qwen_layers(layers, |i, weights| i.delta(weights))?;
            let residency = i.finish()?;
            let base = base(gpu, residency);
            let weights = QwenDelta {
                base,
                layers,
                delta,
                ffn_width: g.ffn_width,
            };
            Architecture::QwenDelta(Arc::new(weights))
        }
        selected::Qwen::Hybrid {
            layers,
            attention_layers,
            delta_layers,
        } => {
            let recurrent = i.builder.recurrent()?;
            let delta = DeltaPlan::new(kh, vh, vd, g.epsilon, recurrent);
            let attention = i.builder.attention(g.dim, g.context)?;
            let attention = AttentionPlan::new(rope, attention);
            let gated = i.builder.gated_attention()?;
            let layers = i.qwen_layers(layers, |i, mixer| {
                let mixer = match mixer {
                    selected::Mixer::Attention { weights, slot } => {
                        let weights = i.gated_attention(weights)?;
                        HybridMixer::Attention { weights, slot }
                    }
                    selected::Mixer::Delta { weights, slot } => {
                        let weights = i.delta(weights)?;
                        HybridMixer::Delta { weights, slot }
                    }
                };
                Ok(mixer)
            })?;
            let residency = i.finish()?;
            let base = base(gpu, residency);
            let weights = QwenHybrid {
                base,
                layers,
                attention,
                gated,
                delta,
                ffn_width: g.ffn_width,
                attention_layers,
                delta_layers,
            };
            Architecture::QwenHybrid(Arc::new(weights))
        }
    };
    Ok(inner)
}
