//! Homogeneous Qwen gated-attention layers and their KV history.
use super::QwenLayer;
use crate::gpu::{ReadOnly, arena::Plan};
use crate::weights::{
    common::{AttentionProjections, CommonWeights},
    compute::{self, Bindings},
    operands::Projection,
    pipelines::GatedAttentionPipelines,
    state::*,
};
use anyhow::Result;
use std::ops::Range;

pub struct QwenAttention {
    pub base: CommonWeights,
    pub layers: Vec<QwenLayer<GatedAttention>>,
    pub attention: compute::AttentionPlan,
    pub gated: GatedAttentionPipelines,
    pub ffn_width: usize,
}

pub struct GatedAttention {
    pub projections: AttentionProjections<Projection>,
    pub qnorm: ReadOnly,
    pub knorm: ReadOnly,
}

pub struct Memory {
    pub common: CommonScratch,
    pub attention: AttentionScratch,
    pub states: Vec<Kv>,
}

impl Memory {
    pub fn new(q: &QwenAttention, capacity: usize, batch: usize) -> Result<Self> {
        let c = &q.base;
        let mut p = Plan::with_capacity(2 * q.layers.len() + 13);
        let common = CommonScratch::plan(
            &mut p,
            batch,
            c.embedding.width,
            q.ffn_width,
            1,
            2,
            c.embedding.vocabulary,
        )?;
        let attention = AttentionScratch::plan(&mut p, batch, capacity, &q.attention)?;
        let states = kv_states(&mut p, q.layers.len(), capacity, &q.attention)?;
        let a = p.allocate(&c.gpu)?;
        let common = common.bind(&a);
        let attention = attention.bind(&a);
        let states = bind_kv_states(states, &a);
        Ok(Self {
            common,
            attention,
            states,
        })
    }
}

impl QwenAttention {
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
            compute.norm(&layer.norm, s.hidden.binding(), &s.norm, positions.len());
            compute.gated_attention(
                &layer.mixer,
                &self.attention,
                &self.gated,
                (&memory.attention, &memory.states[index]),
                positions.clone(),
                output.clone(),
            );
            // Updating all history before pruning unused final-layer outputs.
            if output.is_empty() {
                break;
            }
            let rows = output.len();
            let hidden = s.hidden.at(output.start * self.base.embedding.width * 2);
            compute.add(&s.projected, hidden, rows);
            compute.norm(&layer.post_norm, hidden, &s.norm, rows);
            compute.feed_forward(&layer.ff, self.ffn_width, hidden, rows);
        }
    }
}
