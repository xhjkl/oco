//! GPT-OSS sink attention and routed experts.
mod load;
use super::{
    common::{AttentionProjections, CommonWeights, Layer},
    compute::{self, Bindings},
    operands::{ExpertWeight, Projection, Router},
    state::*,
};
use crate::gpu::{ReadOnly, arena::Plan};
use anyhow::Result;
pub use load::load;
use std::ops::Range;

pub const EXPERT_COUNT: usize = 32;
pub const ACTIVE_EXPERTS: usize = 4;

pub struct GptOss {
    pub base: CommonWeights,
    pub layers: Vec<GptLayer>,
    pub attention: compute::AttentionPlan,
    pub experts: compute::ExpertPlan,
}

pub type GptLayer = Layer<SinkAttention, Experts>;

pub struct SinkAttention {
    pub projections: AttentionProjections<Projection>,
    pub sinks: ReadOnly,
    pub window: usize,
}

/// Alternative feed-forward networks; the router selects and weights a few for each token.
pub struct Experts {
    pub gate: ExpertWeight,
    pub up: ExpertWeight,
    pub down: ExpertWeight,
    pub router: Router,
}

pub struct Memory {
    pub common: CommonScratch,
    pub attention: AttentionScratch,
    pub routing: Routing,
    pub states: Vec<Kv>,
}

impl Memory {
    pub fn new(g: &GptOss, capacity: usize, batch: usize) -> Result<Self> {
        let c = &g.base;
        let mut p = Plan::with_capacity(2 * g.layers.len() + 16);
        let common = CommonScratch::plan(
            &mut p,
            batch,
            c.embedding.width,
            g.experts.width(),
            ACTIVE_EXPERTS,
            4,
            c.embedding.vocabulary,
        )?;
        let attention = AttentionScratch::plan(&mut p, batch, capacity, &g.attention)?;
        let states = kv_states(&mut p, g.layers.len(), capacity, &g.attention)?;
        let routes = batch * ACTIVE_EXPERTS;
        let routing = Routing {
            ids: p.slot(routes, 4)?,
            prob: p.slot(routes, 4)?,
            plan: p.slot(
                6 + 2 * (routes.div_ceil(16) + EXPERT_COUNT) + EXPERT_COUNT * (routes + 1),
                4,
            )?,
        };
        let a = p.allocate(&c.gpu)?;
        let common = common.bind(&a);
        let attention = attention.bind(&a);
        let routing = Routing {
            ids: a.get(routing.ids),
            prob: a.get(routing.prob),
            plan: a.get(routing.plan),
        };
        let states = bind_kv_states(states, &a);
        Ok(Self {
            common,
            attention,
            routing,
            states,
        })
    }
}

impl GptOss {
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
            compute.sink_attention(
                &layer.mixer,
                &self.attention,
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
            compute.experts(&layer.ff, &self.experts, &memory.routing, hidden, rows);
        }
    }
}
