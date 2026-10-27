//! Hybrid Qwen layers with typed slots into separate attention and recurrent history.
use super::{QwenLayer, attention::GatedAttention, delta::Delta};
use crate::gpu::arena::Plan;
use crate::weights::{
    common::CommonWeights,
    compute::{self, Bindings},
    pipelines::GatedAttentionPipelines,
    state::*,
};
use anyhow::Result;
use std::ops::Range;

pub struct QwenHybrid {
    pub base: CommonWeights,
    pub layers: Vec<QwenLayer<HybridMixer>>,
    pub attention: compute::AttentionPlan,
    pub gated: GatedAttentionPipelines,
    pub delta: compute::DeltaPlan,
    pub ffn_width: usize,
    pub attention_layers: usize,
    pub delta_layers: usize,
}

/// Index of an attention layer's KV cache among the hybrid session's attention layers.
#[derive(Clone, Copy)]
pub struct AttentionSlot(pub usize);

/// Index of a recurrent layer's memory among the hybrid session's recurrent layers.
#[derive(Clone, Copy)]
pub struct DeltaSlot(pub usize);

/// Per-layer choice of history mixer, paired with a slot into its matching state array.
#[allow(clippy::large_enum_variant)]
pub enum HybridMixer {
    Attention {
        weights: GatedAttention,
        slot: AttentionSlot,
    },
    Delta {
        weights: Delta,
        slot: DeltaSlot,
    },
}

pub struct Memory {
    pub common: CommonScratch,
    pub attention: AttentionScratch,
    pub delta: DeltaScratch,
    pub kv_states: Vec<Kv>,
    pub recurrent: Vec<Recurrence>,
}

impl Memory {
    pub fn new(q: &QwenHybrid, capacity: usize, batch: usize) -> Result<Self> {
        let c = &q.base;
        let mut p = Plan::with_capacity(2 * q.layers.len() + 15);
        let common = CommonScratch::plan(
            &mut p,
            batch,
            c.embedding.width,
            q.ffn_width,
            1,
            2,
            c.embedding.vocabulary,
        )?;
        // Sharing the sequential mixers' workspaces while retaining their distinct semantic bindings.
        let query = p.slot(
            product(batch, q.attention.query_width.max(q.delta.conv_width()))?,
            2,
        )?;
        let value = p.slot(
            product(batch, q.attention.key_width.max(q.delta.output_width()))?,
            2,
        )?;
        let mixed = p.slot(
            product(batch, q.attention.output_width.max(q.delta.output_width()))?,
            2,
        )?;
        let rotated = p.slot(
            product(batch, q.attention.output_width.max(q.delta.conv_width()))?,
            2,
        )?;
        let partials = AttentionScratch::plan_partials(&mut p, capacity, &q.attention)?;
        let attention = AttentionScratch {
            q: query,
            k: p.slot(product(batch, q.attention.key_width)?, 2)?,
            v: value,
            mixed,
            rotated,
            partials,
        };
        let delta = DeltaScratch {
            qkv: query,
            z: value,
            mixed,
            convolved: rotated,
            alpha: p.slot(product(batch, q.delta.heads())?, 2)?,
            beta: p.slot(product(batch, q.delta.heads())?, 2)?,
            qk: p.slot(product(batch, q.delta.normalized_width())?, 4)?,
            factors: p.slot(product(batch, q.delta.factor_width())?, 4)?,
            values: p.slot(product(batch, q.delta.output_width())?, 4)?,
        };
        let kv_states = kv_states(&mut p, q.attention_layers, capacity, &q.attention)?;
        let states = recurrence(&mut p, q.delta_layers, &q.delta)?;
        let a = p.allocate(&c.gpu)?;
        let common = common.bind(&a);
        let attention = attention.bind(&a);
        let delta = delta.bind(&a);
        let kv_states = bind_kv_states(kv_states, &a);
        let recurrent = bind_recurrence(states, &a);
        Ok(Self {
            common,
            attention,
            delta,
            kv_states,
            recurrent,
        })
    }
}

impl QwenHybrid {
    pub fn encode_layers(
        &self,
        compute: &Bindings<'_>,
        memory: &Memory,
        positions: Range<usize>,
        output_row: Option<usize>,
    ) {
        let s = &memory.common;
        for (index, layer) in self.layers.iter().enumerate() {
            let output = if index + 1 == self.layers.len() {
                output_row.map_or(0..0, |row| row..row + 1)
            } else {
                0..positions.len()
            };
            compute.enter_layer(&layer.norm, positions.len(), index > 0);
            match &layer.mixer {
                HybridMixer::Attention { weights, slot } => compute.gated_attention(
                    weights,
                    &self.attention,
                    &self.gated,
                    (&memory.attention, &memory.kv_states[slot.0]),
                    positions.clone(),
                    output.clone(),
                ),
                HybridMixer::Delta { weights, slot } => compute.delta(
                    weights,
                    &self.delta,
                    (&memory.delta, &memory.recurrent[slot.0]),
                    positions.len(),
                    output.clone(),
                ),
            };
            // Updating all history before pruning unused final-layer outputs.
            if output.is_empty() {
                break;
            }
            let rows = output.len();
            let hidden = s.hidden.at(output.start * self.base.embedding.width * 2);
            compute.add_norm(&layer.post_norm, &s.projected, hidden, &s.norm, rows);
            compute.feed_forward(&layer.ff, self.ffn_width, rows);
            if index + 1 == self.layers.len() {
                compute.add(&s.projected, hidden, rows);
            }
        }
    }
}
