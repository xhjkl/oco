//! Inkling 2 operations: depth mixing of residual streams, loop controllers, causal short
//! convolutions, clipped relative-bias attention, and SITU-activated routed experts.
use super::{Bindings, matrix::Matrix};
use crate::gpu::{
    BufferRef, BufferView, ReadOnly, dispatch, dispatch_concurrent, dispatch_indirect,
};
use crate::weights::{
    inkling::{Control, ControlScratch, InklingLayer},
    operands::{DenseMatrix, Projection},
    pipelines::{InklingExpertPipelines, InklingPipelines},
    state::{Kv, Routing},
};
use bytemuck::{Pod, Zeroable, bytes_of};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Mix {
    n: u32,
    states: u32,
    queries: u32,
    norms: u32,
    epsilon: f32,
    norm_epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ControlParams {
    n: u32,
    norm: u32,
    epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LowRank {
    n: u32,
    rank: u32,
    m: u32,
}

/// The controllers' shared bottleneck width.
fn c_rank(chains: &[ControlChain<'_>]) -> usize {
    chains[0].weights.gain.input
}

/// Threads per row for the row-reduction kernels: wide groups hide latency for a few rows.
fn row_threads(rows: usize) -> usize {
    if rows <= 16 { 1024 } else { 256 }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Rows {
    n: u32,
    m: u32,
    epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Conv {
    channels: u32,
    m: u32,
    stride: u32,
    head_width: u32,
    taps: u32,
    scale: f32,
    accumulate: u32,
    routes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Qkv {
    pub heads: u32,
    pub kv_heads: u32,
    pub dim: u32,
    pub rel: u32,
    pub m: u32,
    pub pos: u32,
    pub clip: f32,
    pub rel_clip: f32,
    pub epsilon: f32,
    pub alpha: f32,
    pub floor: u32,
    pub scaled: u32,
    pub conv: u32,
    pub taps: u32,
    pub conv_scale: f32,
    pub ring: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct InklingAttention {
    pub heads: u32,
    pub kv_heads: u32,
    pub pos: u32,
    pub window: u32,
    pub m: u32,
    pub extent: u32,
    pub scale: f32,
    pub ring: u32,
}

/// Fewest prompt rows that take the tensor attention tiles: on M5 Pro, 16-row appends at 8k
/// depth already ran 9% faster than the vector kernel (48 rows: 19%).
const TENSOR_ATTENTION_ROWS: usize = 16;
/// Fewest visible keys at which a single query splits them across threadgroups, without and
/// with tensor hardware, and into how many chunks.
const SPLIT_KEYS: usize = 16384;
const SPLIT_KEYS_TENSOR: usize = 32768;
const SPLIT_CHUNKS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Situ {
    width: u32,
    n: u32,
    beta: f32,
    linear_beta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Combine {
    hidden: u32,
    batch: u32,
    routes: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RouterParams {
    n: u32,
    experts: u32,
    active: u32,
    normalize: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RoutePlan {
    routes: u32,
    up_tiles: u32,
    down_tiles: u32,
    experts: u32,
    tile_rows: u32,
}

/// Geometry and numerical constants of one Inkling checkpoint, fixed at load.
pub struct InklingPlan {
    pub pipelines: InklingPipelines,
    /// Expert kernels for each bank encoding: IQ4_NL, then Q8_0.
    pub experts: [InklingExpertPipelines; 2],
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub rel: usize,
    pub extent: usize,
    pub window: usize,
    pub epsilon: f32,
    pub clip: f32,
    pub rel_clip: f32,
    pub alpha: f32,
    pub floor: usize,
    pub expert_count: usize,
    pub active: usize,
    pub expert_width: usize,
    pub route_scale: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub taps: usize,
}

impl InklingPlan {
    pub fn key_width(&self) -> usize {
        self.kv_heads * self.dim
    }

    pub fn query_width(&self) -> usize {
        self.heads * self.dim
    }

    /// Convolution state channels of one layer visit: keys, values, attention and FFN outputs.
    pub fn conv_channels(&self) -> usize {
        2 * self.key_width() + 2 * self.hidden
    }
}

/// A buffer and byte offset, as dispatches bind them.
type Binding<'a> = (&'a BufferRef, usize);

/// The single-row K/V convolution fused into `inkling_qkv`: the visit's state and, on
/// full-attention layers, the projected dynamic K and V taps.
pub struct QkvConv<'a> {
    pub state: (&'a BufferRef, usize),
    pub taps: Option<(Binding<'a>, Binding<'a>)>,
}

/// One loop controller's operands and its private scratch.
pub struct ControlChain<'a> {
    pub weights: &'a Control,
    pub cur: (&'a BufferRef, usize),
    pub message: (&'a BufferRef, usize),
    pub out: (&'a BufferRef, usize),
    pub scratch: &'a ControlScratch,
    /// Learned RMS gain applied to the result, for controllers feeding a projection directly.
    pub norm: Option<&'a ReadOnly>,
}

/// One short convolution's operands: input rows, static taps, optional dynamic taps, and the
/// output, which the result overwrites or accumulates into.
pub struct ConvCall<'a> {
    pub x: (&'a BufferRef, usize),
    pub weights: &'a ReadOnly,
    pub dynamic: Option<(&'a BufferRef, usize)>,
    pub state: (&'a BufferRef, usize),
    pub out: (&'a BufferRef, usize),
    pub channels: usize,
    pub accumulate: bool,
    /// Routed input: the per-row probabilities and the routes per row of `x`.
    pub routes: Option<((&'a BufferRef, usize), usize)>,
}

impl Bindings<'_> {
    /// Mix up to four block states and the current stream with one learned query per output,
    /// optionally RMS-normalizing each result with the matching learned gain.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling_mix(
        &self,
        plan: &InklingPlan,
        states: &[&BufferView],
        cur: &BufferView,
        query: &ReadOnly,
        outs: &[(&BufferRef, usize)],
        gains: &[&ReadOnly],
        rows: usize,
        concurrent: bool,
    ) {
        let queries = outs.len();
        debug_assert!(states.len() <= 4 && (1..=4).contains(&queries));
        debug_assert!(gains.is_empty() || gains.len() == queries);
        let pick = |i: usize| states.get(i).map_or(cur.binding(), |s| s.binding());
        let out = |i: usize| outs.get(i).copied().unwrap_or(outs[0]);
        let gain = |i: usize| gains.get(i).map_or(query.binding(), |g| g.binding());
        let p = Mix {
            n: plan.hidden as u32,
            states: states.len() as u32,
            queries: queries as u32,
            norms: gains.len() as u32,
            epsilon: 1e-6,
            norm_epsilon: plan.epsilon,
        };
        let launch = if concurrent {
            dispatch_concurrent
        } else {
            dispatch
        };
        launch(
            self.encoder,
            &plan.pipelines.mix,
            &[
                pick(0),
                pick(1),
                pick(2),
                pick(3),
                cur.binding(),
                query.binding(),
                out(0),
                out(1),
                out(2),
                out(3),
                gain(0),
                gain(1),
                gain(2),
            ],
            bytes_of(&p),
            [rows, 1, 1],
            row_threads(rows),
        );
    }

    /// Loop controllers: `out = cur - tanh(forget) cur + tanh(gain) message`, with gain and
    /// forget projected from the RMS-normalized pair through a rank-64 bottleneck. Independent
    /// controllers (distinct outputs and scratch, no output read by another) share four ordered
    /// stages.
    pub fn inkling_controls(&self, plan: &InklingPlan, chains: &[ControlChain<'_>], rows: usize) {
        let p = Rows {
            n: plan.hidden as u32,
            m: rows as u32,
            epsilon: 1e-6,
        };
        let launch = |index: usize| {
            if index == 0 {
                dispatch
            } else {
                dispatch_concurrent
            }
        };
        for (index, c) in chains.iter().enumerate() {
            launch(index)(
                self.encoder,
                &plan.pipelines.control_input,
                &[c.cur, c.message, c.scratch.cat.binding()],
                bytes_of(&p),
                [rows, 1, 1],
                256,
            );
        }
        for (index, c) in chains.iter().enumerate() {
            self.inkling_dense(
                &c.weights.down,
                c.scratch.cat.binding(),
                &c.scratch.low,
                rows,
                index > 0,
            );
        }
        let lowrank = LowRank {
            n: plan.hidden as u32,
            rank: c_rank(chains) as u32,
            m: rows as u32,
        };
        // Long batches keep the dense tensor tiles; short ones take one thread per output.
        if rows > 16 {
            for (index, c) in chains.iter().enumerate() {
                self.inkling_dense(
                    &c.weights.gain,
                    c.scratch.low.binding(),
                    &c.scratch.gain,
                    rows,
                    index > 0,
                );
                self.inkling_dense(
                    &c.weights.forget,
                    c.scratch.low.binding(),
                    &c.scratch.forget,
                    rows,
                    true,
                );
            }
        }
        for (index, c) in chains.iter().enumerate().filter(|_| rows <= 16) {
            launch(index)(
                self.encoder,
                &plan.pipelines.lowrank,
                &[
                    c.scratch.low.binding(),
                    c.weights.gain.values.binding(),
                    c.weights.forget.values.binding(),
                    c.scratch.gain.binding(),
                    c.scratch.forget.binding(),
                ],
                bytes_of(&lowrank),
                [(2 * plan.hidden).div_ceil(256), rows, 1],
                256,
            );
        }
        for (index, c) in chains.iter().enumerate() {
            let params = ControlParams {
                n: plan.hidden as u32,
                norm: c.norm.is_some() as u32,
                epsilon: plan.epsilon,
            };
            let norm = c.norm.map_or(c.cur, ReadOnly::binding);
            launch(index)(
                self.encoder,
                &plan.pipelines.control,
                &[
                    c.cur,
                    c.message,
                    c.scratch.gain.binding(),
                    c.scratch.forget.binding(),
                    c.out,
                    norm,
                ],
                bytes_of(&params),
                [rows, 1, 1],
                row_threads(rows),
            );
        }
    }

    /// RMS-normalize rows with learned gains, optionally overlapping the previous dispatch.
    pub fn inkling_norm(
        &self,
        w: &ReadOnly,
        x: (&BufferRef, usize),
        y: &BufferView,
        rows: usize,
        concurrent: bool,
    ) {
        let launch = if concurrent {
            dispatch_concurrent
        } else {
            dispatch
        };
        launch(
            self.encoder,
            &self.kernels.pipelines.norm,
            &[x, w.binding(), y.binding()],
            bytes_of(&self.kernels.norm),
            [rows, 1, 1],
            256,
        );
    }

    /// Product of an unbiased matrix that needs no row gains (FP32 or Q8_0 coefficients).
    fn inkling_dense(
        &self,
        w: &DenseMatrix,
        x: (&BufferRef, usize),
        y: &BufferView,
        rows: usize,
        concurrent: bool,
    ) {
        let s = self.scratch;
        w.apply(
            self.encoder,
            None,
            x,
            y,
            rows,
            s,
            s.projection_scales.binding(),
            concurrent,
        );
    }

    /// Causal four-tap convolution with a residual over `rows` rows, carrying its state.
    pub fn inkling_conv(
        &self,
        plan: &InklingPlan,
        call: ConvCall<'_>,
        rows: usize,
        concurrent: bool,
    ) {
        let taps = if call.dynamic.is_some() {
            plan.taps * plan.kv_heads
        } else {
            0
        };
        let factor = if call.dynamic.is_some() { 2.0 } else { 1.0 };
        let p = Conv {
            channels: call.channels as u32,
            m: rows as u32,
            stride: call.channels as u32,
            head_width: plan.dim as u32,
            taps: taps as u32,
            scale: 1.0 / (factor * plan.taps as f32 + 1.0).sqrt(),
            accumulate: call.accumulate as u32,
            routes: call.routes.map_or(0, |(_, routes)| routes as u32),
        };
        let dynamic = call.dynamic.unwrap_or(call.x);
        let probs = call.routes.map_or(call.x, |(probs, _)| probs);
        let launch = if concurrent {
            dispatch_concurrent
        } else {
            dispatch
        };
        launch(
            self.encoder,
            &plan.pipelines.conv,
            &[
                call.x,
                call.weights.binding(),
                dynamic,
                call.state,
                call.out,
                probs,
            ],
            bytes_of(&p),
            [call.channels.div_ceil(256), 1, 1],
            256,
        );
    }

    /// Clip queries, keys and relative vectors, normalize values, store keys and values at
    /// `pos..pos + rows`, and log-scale full-attention queries.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling_qkv(
        &self,
        plan: &InklingPlan,
        layer: &InklingLayer,
        (q, k, v, rel, r): (
            &BufferView,
            &BufferView,
            &BufferView,
            &BufferView,
            &BufferView,
        ),
        (kv, ring): (&Kv, usize),
        conv: Option<QkvConv<'_>>,
        pos: usize,
        rows: usize,
    ) {
        debug_assert!(conv.is_none() || rows == 1);
        let p = Qkv {
            heads: plan.heads as u32,
            kv_heads: plan.kv_heads as u32,
            dim: plan.dim as u32,
            rel: plan.rel as u32,
            m: rows as u32,
            pos: pos as u32,
            clip: plan.clip,
            rel_clip: plan.rel_clip,
            epsilon: plan.epsilon,
            alpha: plan.alpha,
            floor: plan.floor as u32,
            scaled: layer.full as u32,
            conv: conv.is_some() as u32,
            taps: if conv.as_ref().is_some_and(|c| c.taps.is_some()) {
                plan.taps as u32
            } else {
                0
            },
            conv_scale: 1.0
                / ((if layer.full { 2.0 } else { 1.0 }) * plan.taps as f32 + 1.0).sqrt(),
            ring: ring as u32,
        };
        let items = rows * (2 * plan.heads + 2 * plan.kv_heads);
        let dummy = r.binding();
        let (state, taps) = conv.map_or((dummy, (dummy, dummy)), |c| {
            (c.state, c.taps.unwrap_or((dummy, dummy)))
        });
        dispatch(
            self.encoder,
            &plan.pipelines.qkv,
            &[
                q.binding(),
                k.binding(),
                v.binding(),
                rel.binding(),
                layer.q_clip.binding(),
                layer.rel_clip.binding(),
                kv.k.binding(),
                kv.v.binding(),
                r.binding(),
                state,
                layer.k_conv.binding(),
                layer.v_conv.binding(),
                taps.0,
                taps.1,
            ],
            bytes_of(&p),
            [items.div_ceil(8), 1, 1],
            256,
        );
    }

    /// Attend `rows` queries at `pos..` over the stored history with the relative-position bias
    /// and per-query sinks, and gate the result by sigmoid(gate).
    #[allow(clippy::too_many_arguments)]
    pub fn inkling_attention(
        &self,
        plan: &InklingPlan,
        layer: &InklingLayer,
        (q, r, sinks, gate, out): (
            &BufferView,
            &BufferView,
            &BufferView,
            &BufferView,
            &BufferView,
        ),
        (kv, ring): (&Kv, usize),
        pos: usize,
        rows: usize,
    ) {
        let p = InklingAttention {
            heads: plan.heads as u32,
            kv_heads: plan.kv_heads as u32,
            pos: pos as u32,
            window: if layer.full { 0 } else { plan.window as u32 },
            m: rows as u32,
            // Each layer's bias table covers its own extent: the window on SWA layers.
            extent: if layer.full { plan.extent } else { plan.window } as u32,
            scale: 1.0 / (plan.dim as f32).sqrt(),
            ring: ring as u32,
        };
        // Single queries over long histories split the keys into chunks across threadgroups.
        let visible = pos + 1;
        let visible = if layer.full {
            visible
        } else {
            visible.min(plan.window)
        };
        // Measured on M5 Pro, split decode pays from 32k keys (+2.7% there, +9% at 128k) and the
        // vector kernel wins below; on M1 Max it already gains at 16k.
        let threshold = match plan.pipelines.attention_tensor {
            Some(_) => SPLIT_KEYS_TENSOR,
            None => SPLIT_KEYS,
        };
        let chunks = SPLIT_CHUNKS;
        if rows == 1
            && visible >= threshold
            && chunks * plan.heads * (plan.dim + 2) * 4 <= self.scratch.partials.length()
        {
            // Sharing keys across a KV head's three query heads measured 10% faster than one head
            // per threadgroup at 32k on M5 Pro; any other ratio takes one head per group.
            let grouped = plan.heads == 3 * plan.kv_heads;
            let (pipeline, groups) = match grouped {
                true => (&plan.pipelines.split[1], plan.kv_heads),
                false => (&plan.pipelines.split[0], plan.heads),
            };
            let partials = self.scratch.partials.binding();
            let buffers = [
                q.binding(),
                kv.k.binding(),
                kv.v.binding(),
                r.binding(),
                layer.rel_table.binding(),
                partials,
            ];
            dispatch(
                self.encoder,
                pipeline,
                &buffers,
                bytes_of(&p),
                [groups, chunks, 1],
                256,
            );
            // The merge reads the chunk count from `m`, which a single query leaves free.
            let merge = InklingAttention {
                m: chunks as u32,
                ..p
            };
            let buffers = [partials, sinks.binding(), gate.binding(), out.binding()];
            dispatch(
                self.encoder,
                &plan.pipelines.merge,
                &buffers,
                bytes_of(&merge),
                [plan.heads, 1, 1],
                128,
            );
            return;
        }
        let buffers = [
            q.binding(),
            kv.k.binding(),
            kv.v.binding(),
            r.binding(),
            layer.rel_table.binding(),
            sinks.binding(),
            gate.binding(),
            out.binding(),
        ];
        // Prompts take 32-query tensor tiles on Apple10, whose 128-key steps never straddle a
        // ring's wrap; single queries and short batches take the vector kernel.
        let tensor = plan
            .pipelines
            .attention_tensor
            .as_ref()
            .filter(|_| rows >= TENSOR_ATTENTION_ROWS && layer.table_half && ring % 128 == 0);
        match tensor {
            Some(pipeline) => dispatch(
                self.encoder,
                pipeline,
                &buffers,
                bytes_of(&p),
                [plan.heads, rows.div_ceil(32), 1],
                256,
            ),
            None => dispatch(
                self.encoder,
                &plan.pipelines.attention,
                &buffers,
                bytes_of(&p),
                [plan.heads, rows, 1],
                256,
            ),
        }
    }

    /// Route each normalized row (in `scratch.norm`) to its experts and leave their SITU
    /// feed-forward outputs in `scratch.projected`, `active` routes per row, for the post-conv
    /// to weigh by `routing.prob`.
    pub fn inkling_experts(
        &self,
        plan: &InklingPlan,
        layer: &InklingLayer,
        routing: &Routing,
        logits: &BufferView,
        rows: usize,
    ) {
        let e = self.encoder;
        let s = self.scratch;
        let x = &plan.experts[layer.expert_bank];
        let (hidden, width, active, experts) = (
            plan.hidden,
            plan.expert_width,
            plan.active,
            plan.expert_count,
        );
        let router = RouterParams {
            n: hidden as u32,
            experts: experts as u32,
            active: active as u32,
            normalize: 1,
            scale: plan.route_scale,
        };
        dispatch(
            e,
            &plan.pipelines.router_logits,
            &[
                s.norm.at(0),
                layer.router.values.binding(),
                logits.binding(),
            ],
            bytes_of(&router),
            [experts.div_ceil(8), rows, 1],
            256,
        );
        dispatch(
            e,
            &plan.pipelines.route,
            &[
                logits.binding(),
                layer.router.bias.binding(),
                routing.ids.at(0),
                routing.prob.at(0),
            ],
            bytes_of(&router),
            [rows, 1, 1],
            32,
        );
        let routes = rows * active;
        let up = Matrix {
            k: hidden as u32,
            n: 2 * width as u32,
            m: routes as u32,
            route_major: 0,
            routes: active as u32,
            experts: experts as u32,
            has_bias: 0,
        };
        let down = Matrix {
            k: width as u32,
            n: hidden as u32,
            route_major: 1,
            ..up
        };
        let dummy = s.next.binding();
        if rows == 1 {
            let groups = |n: usize| [n.div_ceil(x.vector_rows), active, 1];
            dispatch(
                e,
                &x.vector,
                &[
                    layer.gate_up.binding(),
                    s.norm.at(0),
                    s.gate.at(0),
                    dummy,
                    routing.ids.at(0),
                ],
                bytes_of(&up),
                groups(2 * width),
                x.vector_threads,
            );
            self.inkling_situ(plan, routes);
            dispatch(
                e,
                &x.vector,
                &[
                    layer.down.binding(),
                    s.up.at(0),
                    s.projected.at(0),
                    dummy,
                    routing.ids.at(0),
                ],
                bytes_of(&down),
                groups(hidden),
                x.vector_threads,
            );
        } else {
            // Tensor tiles reuse decoded coefficients across 32 routes; eight-route tiles waste
            // less padding when experts see few routes each.
            let size = usize::from(routes >= 8 * experts);
            let tile_rows = [8u32, 32][size];
            let p = RoutePlan {
                routes: routes as u32,
                up_tiles: (2 * width).div_ceil(64) as u32,
                down_tiles: hidden.div_ceil(64) as u32,
                experts: experts as u32,
                tile_rows,
            };
            dispatch(
                e,
                &plan.pipelines.route_plan,
                &[routing.ids.at(0), routing.plan.at(0)],
                bytes_of(&p),
                [1, 1, 1],
                128,
            );
            dispatch_indirect(
                e,
                &x.up[size],
                &[
                    layer.gate_up.binding(),
                    s.norm.at(0),
                    s.gate.at(0),
                    routing.plan.binding(),
                ],
                bytes_of(&up),
                routing.plan.at(0),
                128,
            );
            self.inkling_situ(plan, routes);
            dispatch_indirect(
                e,
                &x.down[size],
                &[
                    layer.down.binding(),
                    s.up.at(0),
                    s.projected.at(0),
                    routing.plan.binding(),
                ],
                bytes_of(&down),
                routing.plan.at(12),
                128,
            );
        }
    }

    /// Weighted sum of each row's routed expert outputs into `out`.
    pub fn inkling_combine(
        &self,
        plan: &InklingPlan,
        routing: &Routing,
        out: &BufferView,
        rows: usize,
    ) {
        let p = Combine {
            hidden: plan.hidden as u32,
            batch: rows as u32,
            routes: plan.active as u32,
        };
        dispatch(
            self.encoder,
            &plan.pipelines.combine,
            &[
                self.scratch.projected.at(0),
                routing.prob.at(0),
                out.binding(),
            ],
            bytes_of(&p),
            [(rows * plan.hidden).div_ceil(256), 1, 1],
            256,
        );
    }

    fn inkling_situ(&self, plan: &InklingPlan, routes: usize) {
        let s = self.scratch;
        let n = routes * plan.expert_width;
        let p = Situ {
            width: plan.expert_width as u32,
            n: n as u32,
            beta: plan.situ_beta,
            linear_beta: plan.situ_linear_beta,
        };
        dispatch(
            self.encoder,
            &plan.pipelines.situ,
            &[s.gate.at(0), s.up.at(0)],
            bytes_of(&p),
            [n.div_ceil(256), 1, 1],
            256,
        );
    }

    /// Independent projections of already-normalized inputs, in one barrier group: later
    /// products overlap the first, as each reads finished inputs and writes its own output.
    pub fn inkling_project(
        &self,
        products: &[((&BufferRef, usize), &Projection, &BufferView)],
        rows: usize,
    ) {
        for (index, &(x, projection, y)) in products.iter().enumerate() {
            let s = self.scratch;
            projection.matrix.apply(
                self.encoder,
                projection.bias.as_ref(),
                x,
                y,
                rows,
                s,
                s.projection_scales.binding(),
                index > 0,
            );
        }
    }
}
