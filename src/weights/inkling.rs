//! Inkling 2 (`inkling2_loop`): 24 stored layers whose middle twelve run three times per token,
//! with depth-mixed residual blocks, short convolutions around attention and the routed experts,
//! and position-free attention biased by learned relative distances.
mod load;
use super::{
    common::CommonWeights,
    compute::{Bindings, ControlChain, ConvCall, InklingPlan, QkvConv},
    operands::{DenseMatrix, Projection, Router},
    state::*,
};
use crate::Result;
use crate::gpu::{
    BufferView, ReadOnly,
    arena::{Plan, Slot},
};
pub use load::load;
use std::ops::Range;

pub struct Inkling {
    pub base: CommonWeights,
    pub embedding_norm: ReadOnly,
    pub output_query: ReadOnly,
    pub layers: Vec<InklingLayer>,
    /// Loop controllers of the core layers, indexed from `core.start`.
    pub loops: Vec<LoopLayer>,
    pub core: Range<usize>,
    pub passes: usize,
    /// Layers per residual block; the stream restarts at each block's first layer.
    pub block: usize,
    pub plan: InklingPlan,
}

/// One stored layer's weights; core layers serve every pass, each with its own history.
pub struct InklingLayer {
    /// Full-attention layer: separate Q/K/V inputs, dynamic K/V taps, log-scaled positions.
    pub full: bool,
    /// Input norms for the query, key and value paths (the same SWA norm three times).
    pub norms: [ReadOnly; 3],
    /// Depth-mix queries for the attention inputs: none on layer 0, three on full layers.
    pub attn_query: Option<ReadOnly>,
    pub q: Projection,
    pub k: Projection,
    pub v: Projection,
    pub gate: Projection,
    pub out: Projection,
    pub rel: Projection,
    pub sink: Projection,
    pub k_taps: Option<Projection>,
    pub v_taps: Option<Projection>,
    pub q_clip: ReadOnly,
    pub rel_clip: ReadOnly,
    pub rel_table: ReadOnly,
    pub k_conv: ReadOnly,
    pub v_conv: ReadOnly,
    pub out_conv: ReadOnly,
    pub ffn_conv: ReadOnly,
    pub ffn_query: ReadOnly,
    pub ffn_norm: ReadOnly,
    pub router: Router,
    pub gate_up: ReadOnly,
    pub down: ReadOnly,
    /// Index into the plan's expert kernels: 0 for IQ4_NL banks, 1 for Q8_0.
    pub expert_bank: usize,
}

/// Cross-pass message queries and the four controllers (Q, K, V, FFN inputs) of one core layer.
pub struct LoopLayer {
    pub messages: ReadOnly,
    pub controls: [Control; 4],
}

/// Rank-64 controller mixing a cross-pass message into a looped layer input.
pub struct Control {
    pub down: DenseMatrix,
    pub gain: DenseMatrix,
    pub forget: DenseMatrix,
}

/// Per-chunk intermediates beyond the shared scratch.
pub struct InklingScratch<S = BufferView> {
    /// Stream buffers: normalized embeddings, prelude, two core halves per pass parity, coda.
    pub embedded: S,
    pub prelude: S,
    pub first: [S; 2],
    pub second: [S; 2],
    pub coda: S,
    /// Depth-mixed (and controlled) Q/K/V inputs, three planes, and their normalized forms.
    pub inputs: S,
    pub normed: [S; 2],
    pub q: S,
    pub gate: S,
    pub k: S,
    pub v: S,
    pub rel: S,
    pub r: S,
    pub sinks: S,
    pub k_taps: S,
    pub v_taps: S,
    pub mixed: S,
    pub attn_out: S,
    pub messages: S,
    /// The single depth-mixed input of a looped SWA layer, before its three controllers.
    pub shared: S,
    /// Private scratch of the Q, K and V controllers (the FFN controller reuses the first).
    pub controls: [ControlScratch<S>; 3],
    pub ffn: S,
    /// Combined expert outputs of multi-row chunks.
    pub moe: S,
    /// FP32 router logits.
    pub logits: S,
}

/// One loop controller's intermediates.
pub struct ControlScratch<S = BufferView> {
    pub cat: S,
    pub low: S,
    pub gain: S,
    pub forget: S,
}

pub struct Memory {
    pub common: CommonScratch,
    pub scratch: InklingScratch,
    pub routing: Routing,
    /// The capacity-long histories, in visit order: every visit's, or with ring caches only
    /// the full-attention visits'.
    pub kv_states: Vec<Kv>,
    /// Convolution state of every visit (keys, values, attention and FFN outputs), as
    /// recurrent memory so checkpoints and rewinds treat it like other recurrences.
    pub recurrent: Vec<Recurrence>,
    /// Each visit's cache, in visit order, and its ring size in rows (zero for a capacity-long
    /// history). Position `p` of a ring is its row `p % ring`.
    pub caches: Vec<(Kv, usize)>,
}

impl Inkling {
    /// Layer visits per token: prelude, every pass of the core, then the coda.
    pub fn visits(&self) -> usize {
        self.layers.len() + (self.passes - 1) * self.core.len()
    }

    /// The stored layer a visit slot runs: prelude, first core pass and coda in layer order,
    /// then the later core passes.
    fn slot_layer(&self, slot: usize) -> usize {
        match slot.checked_sub(self.layers.len()) {
            None => slot,
            Some(later) => self.core.start + later % self.core.len(),
        }
    }

    /// Rows of each sliding-window visit's ring cache in sessions of `capacity` positions, or
    /// zero when a capacity-long history is no larger. A prompt chunk writes all its keys before
    /// attending, so the ring holds the window behind the chunk's first row plus the chunk.
    pub fn ring_rows(&self, capacity: usize) -> usize {
        let rows = self.plan.window + MAX_PREFILL_ROWS;
        if self.plan.window > 0 && capacity > rows {
            rows
        } else {
            0
        }
    }

    /// The largest capacity whose key and value caches fit in `free` bytes: every visit grows
    /// with capacity up to the ring size, then only the full-attention visits do.
    pub fn capacity_for(&self, free: usize) -> usize {
        let row = 2 * self.plan.key_width() * 2;
        let visits = self.visits();
        let full = (0..visits)
            .filter(|&slot| self.layers[self.slot_layer(slot)].full)
            .count();
        let ring = self.ring_rows(usize::MAX);
        let linear = free / (visits * row);
        if ring == 0 {
            return linear;
        }
        // Past the ring size the sliding-window visits hold a fixed ring each.
        let ringed = match (free / row).checked_sub((visits - full) * ring) {
            None => 0,
            // Nothing grows past the ring size.
            Some(_) if full == 0 => usize::MAX,
            Some(rows) => rows / full,
        };
        if ringed > ring {
            ringed
        } else {
            linear.min(ring)
        }
    }
}

impl Memory {
    pub fn new(n: &Inkling, capacity: usize, batch: usize) -> Result<Self> {
        let c = &n.base;
        let x = &n.plan;
        let hidden = x.hidden;
        let mut p = Plan::with_capacity(4 * n.visits() + 48);
        let common = CommonScratch::plan(
            &mut p,
            batch,
            hidden,
            2 * x.expert_width,
            x.active,
            2,
            c.embedding.vocabulary,
        )?;
        let rows = |p: &mut Plan, width: usize, bytes: usize| p.slot(product(batch, width)?, bytes);
        let control = |p: &mut Plan| -> Result<ControlScratch<Slot>> {
            Ok(ControlScratch {
                cat: rows(p, 2 * hidden, 2)?,
                low: rows(p, 64, 2)?,
                gain: rows(p, hidden, 2)?,
                forget: rows(p, hidden, 2)?,
            })
        };
        let scratch = InklingScratch {
            embedded: rows(&mut p, hidden, 2)?,
            prelude: rows(&mut p, hidden, 2)?,
            first: [rows(&mut p, hidden, 2)?, rows(&mut p, hidden, 2)?],
            second: [rows(&mut p, hidden, 2)?, rows(&mut p, hidden, 2)?],
            coda: rows(&mut p, hidden, 2)?,
            inputs: rows(&mut p, 3 * hidden, 2)?,
            normed: [rows(&mut p, hidden, 2)?, rows(&mut p, hidden, 2)?],
            q: rows(&mut p, x.query_width(), 2)?,
            gate: rows(&mut p, x.query_width(), 2)?,
            k: rows(&mut p, x.key_width(), 2)?,
            v: rows(&mut p, x.key_width(), 2)?,
            rel: rows(&mut p, x.heads * x.rel, 2)?,
            r: rows(&mut p, x.heads * x.rel, 4)?,
            sinks: rows(&mut p, x.heads, 2)?,
            k_taps: rows(&mut p, x.kv_heads * x.taps, 2)?,
            v_taps: rows(&mut p, x.kv_heads * x.taps, 2)?,
            mixed: rows(&mut p, x.query_width(), 2)?,
            attn_out: rows(&mut p, hidden, 2)?,
            messages: rows(&mut p, 4 * hidden, 2)?,
            shared: rows(&mut p, hidden, 2)?,
            controls: [control(&mut p)?, control(&mut p)?, control(&mut p)?],
            ffn: rows(&mut p, hidden, 2)?,
            moe: rows(&mut p, hidden, 2)?,
            logits: rows(&mut p, x.expert_count, 4)?,
        };
        let routes = batch * x.active;
        let routing = Routing {
            ids: p.slot(routes, 4)?,
            prob: p.slot(routes, 4)?,
            plan: p.slot(super::compute::plan_words(routes, x.expert_count), 4)?,
        };
        let visits = n.visits();
        let elements = product(capacity, x.key_width())?;
        let ring = n.ring_rows(capacity);
        let ringed = |slot: usize| ring > 0 && !n.layers[n.slot_layer(slot)].full;
        let mut kv = Vec::with_capacity(visits);
        let mut conv = Vec::with_capacity(visits);
        let ring_elements = product(ring, x.key_width())?;
        let mut rings = Vec::new();
        for slot in 0..visits {
            let rows = if ringed(slot) {
                ring_elements
            } else {
                elements
            };
            let cache = Kv {
                k: p.slot(rows, 2)?,
                v: p.slot(rows, 2)?,
            };
            if ringed(slot) {
                rings.push(cache);
            } else {
                kv.push(cache);
            }
            conv.push(Recurrence {
                conv: p.slot(x.conv_channels() * 3, 4)?,
                ssm: p.slot(1, 4)?,
            });
        }
        let a = p.allocate(&c.gpu)?;
        let get = |slot| a.get(slot);
        let s = scratch;
        let kv_states = bind_kv_states(kv, &a);
        let recurrent = bind_recurrence(conv, &a);
        let (mut linear, mut rings) = (kv_states.iter(), bind_kv_states(rings, &a).into_iter());
        let caches = (0..visits)
            .map(|slot| match ringed(slot) {
                true => (
                    rings.next().expect("one ring per sliding-window visit"),
                    ring,
                ),
                false => {
                    let kv = linear.next().expect("one history per other visit");
                    let (k, v) = (kv.k.clone(), kv.v.clone());
                    (Kv { k, v }, 0)
                }
            })
            .collect();
        Ok(Self {
            common: common.bind(&a),
            scratch: InklingScratch {
                embedded: get(s.embedded),
                prelude: get(s.prelude),
                first: s.first.map(get),
                second: s.second.map(get),
                coda: get(s.coda),
                inputs: get(s.inputs),
                normed: s.normed.map(get),
                q: get(s.q),
                gate: get(s.gate),
                k: get(s.k),
                v: get(s.v),
                rel: get(s.rel),
                r: get(s.r),
                sinks: get(s.sinks),
                k_taps: get(s.k_taps),
                v_taps: get(s.v_taps),
                mixed: get(s.mixed),
                attn_out: get(s.attn_out),
                messages: get(s.messages),
                shared: get(s.shared),
                controls: s.controls.map(|c| ControlScratch {
                    cat: get(c.cat),
                    low: get(c.low),
                    gain: get(c.gain),
                    forget: get(c.forget),
                }),
                ffn: get(s.ffn),
                moe: get(s.moe),
                logits: get(s.logits),
            },
            routing: Routing {
                ids: get(routing.ids),
                prob: get(routing.prob),
                plan: get(routing.plan),
            },
            kv_states,
            recurrent,
            caches,
        })
    }
}

/// One layer visit's place in the schedule: which stored layer, which pass, which history,
/// and the residual streams around it.
struct Visit<'a> {
    layer: usize,
    slot: usize,
    /// Messages from the previous pass (passes after the first, core layers only).
    messages: Option<&'a BufferView>,
    /// Stream entering the layer.
    stream: &'a BufferView,
    /// Stream the layer leaves: a fresh buffer at a block's first layer, else `stream`.
    target: &'a BufferView,
    /// Block states visible to this layer's mixes before it (the entering stream joins them
    /// at a block's first layer, after attention).
    blocks: &'a [&'a BufferView],
}

impl Inkling {
    pub fn encode_layers(
        &self,
        c: &Bindings<'_>,
        memory: &Memory,
        positions: Range<usize>,
        output_row: Option<usize>,
    ) {
        let s = &memory.scratch;
        let rows = positions.len();
        let hidden = self.plan.hidden;
        // Token embeddings are RMS-normalized once; they form the first residual block.
        c.norm(
            &self.embedding_norm,
            c.scratch.hidden.binding(),
            &s.embedded,
            rows,
        );
        let mut slot = 0;
        let run = |c: &Bindings<'_>, v: Visit<'_>| {
            self.encode_visit(c, memory, &positions, v);
        };
        let start = self.core.start;
        let middle = start + self.block;
        // Prelude: the embedding block, then the prelude stream.
        for layer in 0..start {
            let (stream, target): (&BufferView, &BufferView) = if layer == 0 {
                (&s.embedded, &s.prelude)
            } else {
                (&s.prelude, &s.prelude)
            };
            let blocks: &[&BufferView] = if layer == 0 { &[] } else { &[&s.embedded] };
            run(
                c,
                Visit {
                    layer,
                    slot,
                    messages: None,
                    stream,
                    target,
                    blocks,
                },
            );
            slot += 1;
        }
        // Core passes restart from the prelude; each keeps its own histories.
        let mut core_slot = slot;
        let coda_slot = slot + self.core.len();
        for pass in 0..self.passes {
            let (first, second) = (&s.first[pass % 2], &s.second[pass % 2]);
            for layer in self.core.clone() {
                let visit_slot = if pass == 0 {
                    core_slot + (layer - start)
                } else {
                    coda_slot
                        + (self.layers.len() - self.core.end)
                        + (pass - 1) * self.core.len()
                        + (layer - start)
                };
                let messages = if pass > 0 {
                    let previous_first = &s.first[(pass - 1) % 2];
                    let previous_second = &s.second[(pass - 1) % 2];
                    let lp = &self.loops[layer - start];
                    let plane = rows * hidden * 2;
                    let outs = [0, 1, 2, 3].map(|j| s.messages.at(j * plane));
                    c.inkling_mix(
                        &self.plan,
                        &[previous_first],
                        previous_second,
                        &lp.messages,
                        &outs,
                        &[],
                        rows,
                        false,
                    );
                    Some(&s.messages)
                } else {
                    None
                };
                let (stream, target, blocks): (&BufferView, &BufferView, Vec<&BufferView>) =
                    if layer == start {
                        (&s.prelude, first, vec![&s.embedded])
                    } else if layer < middle {
                        (first, first, vec![&s.embedded, &s.prelude])
                    } else if layer == middle {
                        (first, second, vec![&s.embedded, &s.prelude])
                    } else {
                        (second, second, vec![&s.embedded, &s.prelude, first])
                    };
                run(
                    c,
                    Visit {
                        layer,
                        slot: visit_slot,
                        messages,
                        stream,
                        target,
                        blocks: &blocks,
                    },
                );
            }
        }
        core_slot += self.core.len();
        debug_assert_eq!(core_slot, coda_slot);
        // Coda: the last pass's blocks, then its own stream.
        let last = (self.passes - 1) % 2;
        let (first, second) = (&s.first[last], &s.second[last]);
        for layer in self.core.end..self.layers.len() {
            let visit_slot = coda_slot + (layer - self.core.end);
            let (stream, target, blocks): (&BufferView, &BufferView, Vec<&BufferView>) =
                if layer == self.core.end {
                    (second, &s.coda, vec![&s.embedded, &s.prelude, first])
                } else {
                    (
                        &s.coda,
                        &s.coda,
                        vec![&s.embedded, &s.prelude, first, second],
                    )
                };
            run(
                c,
                Visit {
                    layer,
                    slot: visit_slot,
                    messages: None,
                    stream,
                    target,
                    blocks: &blocks,
                },
            );
        }
        // The output mix of the predicted row replaces its hidden row for the final norm.
        if let Some(row) = output_row {
            let at = |b: &BufferView| b.slice(row * hidden * 2, hidden * 2);
            let blocks = [at(&s.embedded), at(&s.prelude), at(first), at(second)];
            let coda = at(&s.coda);
            let out = c.scratch.hidden.at(row * hidden * 2);
            c.inkling_mix(
                &self.plan,
                &[&blocks[0], &blocks[1], &blocks[2], &blocks[3]],
                &coda,
                &self.output_query,
                &[out],
                &[],
                1,
                false,
            );
        }
    }

    fn encode_visit(
        &self,
        c: &Bindings<'_>,
        memory: &Memory,
        positions: &Range<usize>,
        v: Visit<'_>,
    ) {
        let x = &self.plan;
        let s = &memory.scratch;
        let layer = &self.layers[v.layer];
        let rows = positions.len();
        let hidden = x.hidden;
        let plane = rows * hidden * 2;
        let input = |j: usize| s.inputs.slice(j * plane, plane);
        let inputs = [input(0), input(1), input(2)];
        let (kv, ring) = &memory.caches[v.slot];
        // A longer chunk would overwrite keys its own first rows still attend to.
        assert!(
            *ring == 0 || rows + x.window <= *ring,
            "a chunk must fit its ring"
        );
        let kv = (kv, *ring);
        let state = &memory.recurrent[v.slot].conv;
        let state_at = |channel: usize| state.at(channel * 3 * 4);
        let loops = v
            .messages
            .map(|m| (m, &self.loops[v.layer - self.core.start]));
        // Normalized attention inputs, written straight into the projection inputs: one shared
        // row on plain SWA layers, three on full or looped ones. The depth mix applies the input
        // norms itself unless loop controllers come between, which then apply them instead.
        let n0 = &c.scratch.norm;
        let shared = !layer.full && loops.is_none();
        let (n1, n2) = if shared {
            (n0, n0)
        } else {
            (&s.normed[0], &s.normed[1])
        };
        let normed = [n0.binding(), n1.binding(), n2.binding()];
        let norms = [&layer.norms[0], &layer.norms[1], &layer.norms[2]];
        match (&layer.attn_query, loops) {
            (None, None) => c.inkling_norm(norms[0], v.stream.binding(), n0, rows, false),
            (Some(query), None) => {
                let count = if layer.full { 3 } else { 1 };
                c.inkling_mix(
                    x,
                    v.blocks,
                    v.stream,
                    query,
                    &normed[..count],
                    &norms[..count],
                    rows,
                    false,
                );
            }
            (Some(query), Some((m, lp))) => {
                // Looped: mix without norms, then three controllers side by side.
                let sources: [&BufferView; 3] = if layer.full {
                    let outs = [0, 1, 2].map(|j| inputs[j].binding());
                    c.inkling_mix(x, v.blocks, v.stream, query, &outs, &[], rows, true);
                    [&inputs[0], &inputs[1], &inputs[2]]
                } else {
                    c.inkling_mix(
                        x,
                        v.blocks,
                        v.stream,
                        query,
                        &[s.shared.binding()],
                        &[],
                        rows,
                        true,
                    );
                    [&s.shared, &s.shared, &s.shared]
                };
                let chains: Vec<ControlChain<'_>> = (0..3)
                    .map(|j| ControlChain {
                        weights: &lp.controls[j],
                        cur: sources[j].binding(),
                        message: m.at(j * plane),
                        out: normed[j],
                        scratch: &s.controls[j],
                        norm: Some(norms[j]),
                    })
                    .collect();
                c.inkling_controls(x, &chains, rows);
            }
            (None, Some(_)) => unreachable!("layer 0 is never looped"),
        }
        let mut products = vec![
            (n0.binding(), &layer.q, &s.q),
            (n0.binding(), &layer.gate, &s.gate),
            (n0.binding(), &layer.rel, &s.rel),
            (n0.binding(), &layer.sink, &s.sinks),
            (n1.binding(), &layer.k, &s.k),
            (n2.binding(), &layer.v, &s.v),
        ];
        if let (Some(k_taps), Some(v_taps)) = (&layer.k_taps, &layer.v_taps) {
            products.push((n1.binding(), k_taps, &s.k_taps));
            products.push((n2.binding(), v_taps, &s.v_taps));
        }
        c.inkling_project(&products, rows);
        let key_width = x.key_width();
        let fused = rows == 1;
        if !fused {
            c.inkling_conv(
                x,
                ConvCall {
                    x: s.k.binding(),
                    weights: &layer.k_conv,
                    dynamic: layer.k_taps.as_ref().map(|_| s.k_taps.binding()),
                    state: state_at(0),
                    out: s.k.binding(),
                    channels: key_width,
                    accumulate: false,
                    routes: None,
                },
                rows,
                false,
            );
            c.inkling_conv(
                x,
                ConvCall {
                    x: s.v.binding(),
                    weights: &layer.v_conv,
                    dynamic: layer.v_taps.as_ref().map(|_| s.v_taps.binding()),
                    state: state_at(key_width),
                    out: s.v.binding(),
                    channels: key_width,
                    accumulate: false,
                    routes: None,
                },
                rows,
                true,
            );
        }
        let conv = fused.then(|| QkvConv {
            state: state_at(0),
            taps: layer
                .k_taps
                .as_ref()
                .map(|_| (s.k_taps.binding(), s.v_taps.binding())),
        });
        c.inkling_qkv(
            x,
            layer,
            (&s.q, &s.k, &s.v, &s.rel, &s.r),
            kv,
            conv,
            positions.start,
            rows,
        );
        c.inkling_attention(
            x,
            layer,
            (&s.q, &s.r, &s.sinks, &s.gate, &s.mixed),
            kv,
            positions.start,
            rows,
        );
        layer.out.apply(c, s.mixed.binding(), &s.attn_out, rows);
        // A block's first layer starts a new stream from the attention output alone.
        let restart = !std::ptr::eq(v.stream, v.target);
        c.inkling_conv(
            x,
            ConvCall {
                x: s.attn_out.binding(),
                weights: &layer.out_conv,
                dynamic: None,
                state: state_at(2 * key_width),
                out: v.target.binding(),
                channels: hidden,
                accumulate: !restart,
                routes: None,
            },
            rows,
            false,
        );
        // Feed-forward input: the blocks (with the entering stream at a block start) and the
        // stream after attention, then the FFN controller in later passes.
        let mut blocks: Vec<&BufferView> = v.blocks.to_vec();
        if restart {
            blocks.push(v.stream);
        }
        if let Some((m, lp)) = loops {
            c.inkling_mix(
                x,
                &blocks,
                v.target,
                &layer.ffn_query,
                &[s.ffn.binding()],
                &[],
                rows,
                false,
            );
            let chain = ControlChain {
                weights: &lp.controls[3],
                cur: s.ffn.binding(),
                message: m.at(3 * plane),
                out: n0.binding(),
                scratch: &s.controls[0],
                norm: Some(&layer.ffn_norm),
            };
            c.inkling_controls(x, &[chain], rows);
        } else {
            c.inkling_mix(
                x,
                &blocks,
                v.target,
                &layer.ffn_query,
                &[n0.binding()],
                &[&layer.ffn_norm],
                rows,
                false,
            );
        }
        c.inkling_experts(x, layer, &memory.routing, &s.logits, rows);
        // Single rows weigh the routes inside the convolution; chunks combine them in parallel.
        if !fused {
            c.inkling_combine(x, &memory.routing, &s.moe, rows);
        }
        c.inkling_conv(
            x,
            ConvCall {
                x: if fused {
                    c.scratch.projected.binding()
                } else {
                    s.moe.binding()
                },
                weights: &layer.ffn_conv,
                dynamic: None,
                state: state_at(2 * key_width + hidden),
                out: v.target.binding(),
                channels: hidden,
                accumulate: true,
                routes: fused.then(|| (memory.routing.prob.binding(), x.active)),
            },
            rows,
            false,
        );
    }
}
