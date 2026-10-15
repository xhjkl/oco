//! Homogeneous Qwen gated-delta layers and their recurrent history.
use super::QwenLayer;
use crate::gpu::{ReadOnly, arena::Plan};
use crate::weights::{
    common::CommonWeights,
    compute::{self, Bindings},
    operands::Projection,
    state::*,
};
use anyhow::Result;
use std::ops::Range;

pub const KEY_WIDTH: usize = 128;
pub const CONV_TAPS: usize = 4;

pub struct QwenDelta {
    pub base: CommonWeights,
    pub layers: Vec<QwenLayer<Delta>>,
    pub delta: compute::DeltaPlan,
    pub ffn_width: usize,
}

/// Learned updates for Qwen's recurrent memory, which summarizes history without a KV row per token.
pub struct Delta {
    pub qkv: Projection,
    pub z: Projection,
    pub alpha: Projection,
    pub beta: Projection,
    pub out: Projection,
    pub conv: ReadOnly,
    pub a: ReadOnly,
    pub dt: ReadOnly,
    pub norm: ReadOnly,
}

pub struct Memory {
    pub common: CommonScratch,
    pub delta: DeltaScratch,
    pub states: Vec<Recurrence>,
}

impl Memory {
    pub fn new(q: &QwenDelta, batch: usize) -> Result<Self> {
        let c = &q.base;
        let mut p = Plan::with_capacity(2 * q.layers.len() + 14);
        let common = CommonScratch::plan(
            &mut p,
            batch,
            c.embedding.width,
            q.ffn_width,
            1,
            2,
            c.embedding.vocabulary,
        )?;
        let delta = DeltaScratch::plan(&mut p, batch, &q.delta)?;
        let states = recurrence(&mut p, q.layers.len(), &q.delta)?;
        let a = p.allocate(&c.gpu)?;
        let common = common.bind(&a);
        let delta = delta.bind(&a);
        let states = bind_recurrence(states, &a);
        Ok(Self {
            common,
            delta,
            states,
        })
    }
}

impl QwenDelta {
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
            compute.delta(
                &layer.mixer,
                &self.delta,
                (&memory.delta, &memory.states[index]),
                positions.len(),
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
