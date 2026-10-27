//! Homogeneous Qwen gated-attention layers and their KV history.
use super::{QwenLayer, audio};
use crate::gpu::{BufferView, ReadOnly, arena::Plan};
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
    /// Companion speech encoder whose rows sessions splice into the token stream (Qwen3-ASR).
    pub audio: Option<audio::AudioEncoder>,
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
    /// Encoder workspace and the feature rows it fills, one row per possible context position.
    pub audio: Option<(audio::AudioScratch, BufferView)>,
}

impl Memory {
    pub fn new(q: &QwenAttention, capacity: usize, batch: usize) -> Result<Self> {
        let c = &q.base;
        let mut p = Plan::with_capacity(2 * q.layers.len() + 40);
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
        let audio = match &q.audio {
            None => None,
            Some(encoder) => {
                let scratch = audio::AudioScratch::plan(&mut p, encoder)?;
                let features = p.slot(product(capacity, encoder.output)?, 2)?;
                Some((scratch, features))
            }
        };
        let a = p.allocate(&c.gpu)?;
        let common = common.bind(&a);
        let attention = attention.bind(&a);
        let states = bind_kv_states(states, &a);
        let audio = audio.map(|(scratch, features)| (scratch.bind(&a), a.get(features)));
        Ok(Self {
            common,
            attention,
            states,
            audio,
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
            compute.enter_layer(&layer.norm, positions.len(), index > 0);
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
            compute.add_norm(&layer.post_norm, &s.projected, hidden, &s.norm, rows);
            compute.feed_forward(&layer.ff, self.ffn_width, rows);
            if index + 1 == self.layers.len() {
                compute.add(&s.projected, hidden, rows);
            }
        }
    }
}
