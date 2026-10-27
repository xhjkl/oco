//! Nemotron-H MoE: Mamba-2 recurrence, position-free attention, and relu² routed experts,
//! one history mixer or feed-forward per block with a single pre-norm and residual.
mod load;
use super::{
    common::{AttentionProjections, CommonWeights},
    compute::{self, Bindings},
    operands::{Projection, Router},
    state::*,
};
use crate::gpu::{ReadOnly, arena::Plan};
use anyhow::Result;
pub use load::load;
use std::ops::Range;

pub struct Nemotron {
    pub base: CommonWeights,
    pub layers: Vec<NemotronLayer>,
    pub attention: compute::AttentionPlan,
    pub mamba: compute::MambaPlan,
    pub experts: compute::ReluExpertPlan,
    pub attention_layers: usize,
    pub mamba_layers: usize,
}

pub struct NemotronLayer {
    pub norm: ReadOnly,
    pub mixer: Mixer,
}

/// Per-block choice of operation, with slots into the matching session history arrays.
#[allow(clippy::large_enum_variant)]
pub enum Mixer {
    Mamba { weights: Mamba, slot: usize },
    Attention {
        weights: AttentionProjections<Projection>,
        slot: usize,
    },
    Experts(ReluExperts),
}

/// Learned Mamba-2 parameters: fused input projection, depthwise convolution, per-head decay
/// (stored negated), skip and time-step biases, grouped output norm, and output projection.
pub struct Mamba {
    pub in_proj: Projection,
    /// The time-step rows of `in_proj`, recomputed into FP32 for the recurrence.
    pub dt: super::operands::DenseMatrix,
    pub out: Projection,
    pub conv: ReadOnly,
    pub conv_bias: ReadOnly,
    pub a: ReadOnly,
    pub d: ReadOnly,
    pub dt_bias: ReadOnly,
    pub norm: ReadOnly,
}

/// Sigmoid router with selection bias, gathered Q4_0 expert banks, and one dense shared expert.
pub struct ReluExperts {
    pub router: Router,
    pub up: ReadOnly,
    pub down: ReadOnly,
    pub shared_up: Projection,
    pub shared_down: Projection,
}

pub struct Memory {
    pub common: CommonScratch,
    pub attention: AttentionScratch,
    pub mamba: MambaScratch,
    pub routing: Routing,
    pub kv_states: Vec<Kv>,
    pub recurrent: Vec<Recurrence>,
}

impl Memory {
    pub fn new(n: &Nemotron, capacity: usize, batch: usize) -> Result<Self> {
        let c = &n.base;
        let mut p = Plan::with_capacity(2 * n.layers.len() + 20);
        let active = n.experts.active();
        // One extra route slot holds the shared expert's output beside the routed ones.
        let common = CommonScratch::plan(
            &mut p,
            batch,
            c.embedding.width,
            n.experts.expert_width(),
            active + 1,
            2,
            c.embedding.vocabulary,
        )?;
        let attention = AttentionScratch::plan(&mut p, batch, capacity, &n.attention)?;
        let mamba = MambaScratch::plan(&mut p, batch, &n.mamba)?;
        let routes = batch * active;
        let routing = Routing {
            ids: p.slot(routes, 4)?,
            prob: p.slot(routes, 4)?,
            plan: p.slot(compute::plan_words(routes, n.experts.experts()), 4)?,
        };
        let kv_states = kv_states(&mut p, n.attention_layers, capacity, &n.attention)?;
        let recurrent = mamba_states(&mut p, n.mamba_layers, &n.mamba)?;
        let a = p.allocate(&c.gpu)?;
        Ok(Self {
            common: common.bind(&a),
            attention: attention.bind(&a),
            mamba: mamba.bind(&a),
            routing: Routing {
                ids: a.get(routing.ids),
                prob: a.get(routing.prob),
                plan: a.get(routing.plan),
            },
            kv_states: bind_kv_states(kv_states, &a),
            recurrent: bind_recurrence(recurrent, &a),
        })
    }
}

impl Nemotron {
    pub fn encode_layers(
        &self,
        compute: &Bindings<'_>,
        memory: &Memory,
        positions: Range<usize>,
        output_row: Option<usize>,
    ) {
        let s = &memory.common;
        let width = self.base.embedding.width;
        // A mixer's residual add is deferred into the next block's normalization.
        let mut pending: Option<Range<usize>> = None;
        for (index, layer) in self.layers.iter().enumerate() {
            let output = if index + 1 == self.layers.len() {
                output_row.map_or(0..0, |row| row..row + 1)
            } else {
                0..positions.len()
            };
            // Experts see each row independently, so only continuing rows are normalized.
            let live = match &layer.mixer {
                Mixer::Experts(_) => output.clone(),
                _ => 0..positions.len(),
            };
            if live.is_empty() {
                break;
            }
            let hidden = s.hidden.at(live.start * width * 2);
            match pending.take() {
                Some(added) if added == live => {
                    compute.add_norm(&layer.norm, &s.projected, hidden, &s.norm, live.len());
                }
                Some(added) => {
                    compute.add(&s.projected, s.hidden.at(added.start * width * 2), added.len());
                    compute.norm(&layer.norm, hidden, &s.norm, live.len());
                }
                None => compute.norm(&layer.norm, hidden, &s.norm, live.len()),
            }
            match &layer.mixer {
                Mixer::Experts(weights) => {
                    compute.relu_experts(weights, &self.experts, &memory.routing, hidden, live.len());
                }
                Mixer::Mamba { weights, slot } => {
                    compute.mamba(
                        weights,
                        &self.mamba,
                        (&memory.mamba, &memory.recurrent[*slot]),
                        positions.len(),
                        output.clone(),
                    );
                    pending = Some(output.clone());
                }
                Mixer::Attention { weights, slot } => {
                    compute.plain_attention(
                        weights,
                        &self.attention,
                        (&memory.attention, &memory.kv_states[*slot]),
                        positions.clone(),
                        output.clone(),
                    );
                    pending = Some(output.clone());
                }
            }
        }
        if let Some(added) = pending.filter(|added| !added.is_empty()) {
            compute.add(&s.projected, s.hidden.at(added.start * width * 2), added.len());
        }
    }
}
