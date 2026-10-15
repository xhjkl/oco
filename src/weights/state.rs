//! Session-owned history and temporary GPU workspaces.
//!
//! KV caches and recurrent memory carry information from one forward pass to the
//! next. Scratch holds intermediate token representations and is reused across
//! layers and calls. A row here means one token in the current chunk, not a separate
//! conversation. History capacity and scratch row capacity therefore differ.
//!
//! The same binding structures first hold planned arena slots, then concrete storage.
//! Only buffers whose uses cannot overlap may share slots; history remains distinct
//! for every layer even when that layer's scratch is reused by the next one.

use super::{
    compute::{AttentionPlan, DeltaPlan},
    qwen::delta::{CONV_TAPS, KEY_WIDTH},
};
use crate::gpu::{
    BufferView,
    arena::{Arena, Plan, Slot},
};
use anyhow::{Context, Result};

/// Maximum prompt tokens processed per submission, bounding scratch independently of history length.
pub const MAX_PREFILL_ROWS: usize = 512;

/// Intermediate representations reused as tokens pass through successive model layers.
pub struct CommonScratch<S = BufferView> {
    pub input: S,
    /// Each token's evolving feature vector, refined by history mixing and feed-forward steps.
    pub hidden: S,
    pub norm: S,
    /// Row gains retained until the next normalized projection group.
    pub norm_scales: S,
    /// Row gains reused by serial projections of non-normalized intermediates.
    pub projection_scales: S,
    /// Shared with the feed-forward output after the mixer residual has consumed it.
    pub projected: S,
    pub gate: S,
    pub up: S,
    /// Vocabulary scores for GPU greedy selection after the final input row.
    pub logits: S,
    /// Greedy next-token ID, reused directly as decode input without a CPU-to-GPU copy.
    pub next: S,
}

/// Temporary queries, keys, and values used to read from and extend one layer's attention history.
pub struct AttentionScratch<S = BufferView> {
    pub q: S,
    pub k: S,
    pub v: S,
    pub mixed: S,
    pub rotated: S,
    /// FP32 split-attention peak, mass and weighted values, reused across layers for one live query.
    pub partials: Option<S>,
}

/// Per-chunk intermediates for updating recurrent memory and producing token features.
pub struct DeltaScratch<S = BufferView> {
    pub qkv: S,
    pub z: S,
    pub mixed: S,
    pub convolved: S,
    pub alpha: S,
    pub beta: S,
    /// Normalized FP32 queries and keys, shared by value heads that use the same key head.
    pub qk: S,
    /// Per-token decay, beta, and query-key dot products.
    pub factors: S,
    /// Recurrent output before gated RMS normalization.
    pub values: S,
}

/// Expert selections and combination weights for dispatching each token to its chosen networks.
pub struct Routing<S = BufferView> {
    pub ids: S,
    pub plan: S,
    pub prob: S,
}

/// One attention layer's remembered keys and values, indexed by consumed token position.
/// Keys let later queries find relevant history; values supply the features to combine.
/// Attention reads only through the current position, so reset need not erase unused rows.
pub struct Kv<S = BufferView> {
    pub k: S,
    pub v: S,
}

/// One recurrent layer's summary of prior input, updated in token order.
/// Unlike a KV cache, its size does not grow with the session's context capacity.
pub struct Recurrence<S = BufferView> {
    /// Recent input needed by the short convolution at the start of the next update.
    pub conv: S,
    /// Accumulated memory carried through the gated delta updates.
    pub ssm: S,
}

impl Recurrence {
    pub fn reset(&self) {
        self.conv.zero();
        self.ssm.zero();
    }
}

pub fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).context("workspace element count overflow")
}

impl CommonScratch<Slot> {
    pub fn plan(
        plan: &mut Plan,
        rows: usize,
        hidden: usize,
        ffn_width: usize,
        routes: usize,
        activation_bytes: usize,
        vocabulary: usize,
    ) -> Result<Self> {
        Ok(Self {
            input: plan.slot(rows, 4)?,
            hidden: plan.slot(product(rows, hidden)?, 2)?,
            norm: plan.slot(product(rows, hidden)?, 2)?,
            norm_scales: plan.slot(rows, 8)?,
            projection_scales: plan.slot(rows, 8)?,
            projected: plan.slot(product(product(rows, hidden)?, routes)?, 2)?,
            gate: plan.slot(
                product(product(rows, ffn_width)?, routes)?,
                activation_bytes,
            )?,
            up: plan.slot(
                product(product(rows, ffn_width)?, routes)?,
                activation_bytes,
            )?,
            logits: plan.slot(vocabulary, 2)?,
            next: plan.slot(1, 4)?,
        })
    }

    pub fn bind(self, a: &Arena) -> CommonScratch {
        CommonScratch {
            input: a.get(self.input),
            hidden: a.get(self.hidden),
            norm: a.get(self.norm),
            norm_scales: a.get(self.norm_scales),
            projection_scales: a.get(self.projection_scales),
            projected: a.get(self.projected),
            gate: a.get(self.gate),
            up: a.get(self.up),
            logits: a.get(self.logits),
            next: a.get(self.next),
        }
    }
}

impl AttentionScratch<Slot> {
    pub fn plan(p: &mut Plan, batch: usize, capacity: usize, a: &AttentionPlan) -> Result<Self> {
        let partials = Self::plan_partials(p, capacity, a)?;
        Ok(Self {
            q: p.slot(product(batch, a.query_width)?, 2)?,
            k: p.slot(product(batch, a.key_width)?, 2)?,
            v: p.slot(product(batch, a.key_width)?, 2)?,
            mixed: p.slot(product(batch, a.output_width)?, 2)?,
            rotated: p.slot(product(batch, a.output_width)?, 2)?,
            partials,
        })
    }

    /// Reserve fixed-fan-out partials only for sessions that can reach the long-context selector.
    pub fn plan_partials(p: &mut Plan, capacity: usize, a: &AttentionPlan) -> Result<Option<Slot>> {
        if capacity < AttentionPlan::SPLIT_MIN_KEYS || a.pipelines.split.is_none() {
            return Ok(None);
        }
        let rows = product(a.rope.heads as usize, AttentionPlan::SPLITS)?;
        let elements = product(rows, a.rope.dim as usize + 2)?;
        let partials = p.slot(elements, 4)?;
        Ok(Some(partials))
    }

    pub fn bind(self, a: &Arena) -> AttentionScratch {
        let partials = match self.partials {
            None => None,
            Some(slot) => {
                let partials = a.get(slot);
                Some(partials)
            }
        };
        AttentionScratch {
            q: a.get(self.q),
            k: a.get(self.k),
            v: a.get(self.v),
            mixed: a.get(self.mixed),
            rotated: a.get(self.rotated),
            partials,
        }
    }
}

impl DeltaScratch<Slot> {
    pub fn plan(p: &mut Plan, batch: usize, d: &DeltaPlan) -> Result<Self> {
        Ok(Self {
            qkv: p.slot(product(batch, d.conv_width())?, 2)?,
            z: p.slot(product(batch, d.output_width())?, 2)?,
            mixed: p.slot(product(batch, d.output_width())?, 2)?,
            convolved: p.slot(product(batch, d.conv_width())?, 2)?,
            alpha: p.slot(product(batch, d.heads())?, 2)?,
            beta: p.slot(product(batch, d.heads())?, 2)?,
            qk: p.slot(product(batch, d.normalized_width())?, 4)?,
            factors: p.slot(product(batch, d.factor_width())?, 4)?,
            values: p.slot(product(batch, d.output_width())?, 4)?,
        })
    }

    pub fn bind(self, a: &Arena) -> DeltaScratch {
        DeltaScratch {
            qkv: a.get(self.qkv),
            z: a.get(self.z),
            mixed: a.get(self.mixed),
            convolved: a.get(self.convolved),
            alpha: a.get(self.alpha),
            beta: a.get(self.beta),
            qk: a.get(self.qk),
            factors: a.get(self.factors),
            values: a.get(self.values),
        }
    }
}

pub fn kv_states(
    p: &mut Plan,
    count: usize,
    capacity: usize,
    a: &AttentionPlan,
) -> Result<Vec<Kv<Slot>>> {
    let elements = product(capacity, a.key_width)?;
    let mut slots = Vec::with_capacity(count);
    for _ in 0..count {
        slots.push(Kv {
            k: p.slot(elements, 2)?,
            v: p.slot(elements, 2)?,
        });
    }
    Ok(slots)
}

pub fn recurrence(p: &mut Plan, count: usize, d: &DeltaPlan) -> Result<Vec<Recurrence<Slot>>> {
    let mut slots = Vec::with_capacity(count);
    for _ in 0..count {
        slots.push(Recurrence {
            conv: p.slot(product(d.conv_width(), CONV_TAPS - 1)?, 4)?,
            ssm: p.slot(product(d.output_width(), KEY_WIDTH)?, 4)?,
        });
    }
    Ok(slots)
}

pub fn bind_kv_states(slots: Vec<Kv<Slot>>, a: &Arena) -> Vec<Kv> {
    slots
        .into_iter()
        .map(|s| Kv {
            k: a.get(s.k),
            v: a.get(s.v),
        })
        .collect()
}

pub fn bind_recurrence(slots: Vec<Recurrence<Slot>>, a: &Arena) -> Vec<Recurrence> {
    slots
        .into_iter()
        .map(|s| {
            let state = Recurrence {
                conv: a.get(s.conv),
                ssm: a.get(s.ssm),
            };
            state.reset();
            state
        })
        .collect()
}
