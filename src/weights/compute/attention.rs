use super::Bindings;
use crate::{
    gpu::{BufferRef, ReadOnly, dispatch, dispatch_concurrent},
    weights::{
        common::AttentionProjections,
        gpt_oss::SinkAttention,
        operands::Projection,
        pipelines::{AttentionDispatch, AttentionPipelines, GatedAttentionPipelines},
        qwen::attention::GatedAttention,
        state::{AttentionScratch, Kv},
    },
};
use bytemuck::{Pod, Zeroable, bytes_of};
use std::ops::Range;

/// Positional rotation parameters that let attention distinguish where tokens occurred.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Rope {
    pub heads: u32,
    pub kv_heads: u32,
    pub dim: u32,
    pub rope_dim: u32,
    pub m: u32,
    pub pos: u32,
    pub q_stride: u32,
    pub base: f32,
    pub factor: f32,
    pub low: f32,
    pub high: f32,
    pub scale: f32,
}

impl Rope {
    /// Geometry for attention without positional rotation (encoders, cross-attention, NoPE).
    pub fn unrotated(heads: usize, kv_heads: usize, dim: usize) -> Self {
        Self {
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            dim: dim as u32,
            rope_dim: 0,
            m: 0,
            pos: 0,
            q_stride: 1,
            base: 10000.0,
            factor: 1.0,
            low: 0.0,
            high: 1.0,
            scale: 1.0,
        }
    }
}

/// Per-dispatch attention geometry shared with the kernels (and reused by their tests).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct AttentionParams {
    pub heads: u32,
    pub kv_heads: u32,
    pub dim: u32,
    pub pos: u32,
    pub window: u32,
    pub sinks: u32,
    pub scale: f32,
    pub m: u32,
    /// Nonzero attends every query to exactly this many stored keys; zero is causal.
    pub keys: u32,
}

impl AttentionParams {
    fn new(plan: &AttentionPlan, pos: usize, m: usize, keys: usize) -> Self {
        Self {
            heads: plan.rope.heads,
            kv_heads: plan.rope.kv_heads,
            dim: plan.rope.dim,
            pos: pos as u32,
            window: 0,
            sinks: 0,
            scale: plan.scale,
            m: m as u32,
            keys: keys as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SplitAttentionParams {
    attention: AttentionParams,
    splits: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QkNorm {
    dim: u32,
    /// Query head stride in head widths: 2 interleaves a gate after each query head.
    q_stride: u32,
    q_heads: u32,
    k_heads: u32,
    epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttentionGateParams {
    n: u32,
    dim: u32,
}

/// Prepared geometry and kernels for matching token queries against retained keys and values.
pub struct AttentionPlan {
    pub rope: Rope,
    pub pipelines: AttentionPipelines,
    scale: f32,
    pub query_width: usize,
    pub key_width: usize,
    pub output_width: usize,
}

impl AttentionPlan {
    pub const SPLIT_MIN_KEYS: usize = 4096;

    /// Fixed fan-out measured on long-context Qwen and GPT-OSS attention.
    pub const SPLITS: usize = 32;

    /// Select measured full-chunk geometries; other shapes keep their existing dispatch choices.
    fn wide_prefill(dim: usize, rows: usize, window: usize) -> bool {
        rows == 512 && ((dim == 256 && window == 0) || (dim == 64 && matches!(window, 0 | 128)))
    }

    pub fn new(rope: Rope, pipelines: AttentionPipelines) -> Self {
        let output_width = (rope.heads * rope.dim) as usize;
        let query_width = output_width * rope.q_stride as usize;
        let key_width = (rope.kv_heads * rope.dim) as usize;
        Self {
            query_width,
            key_width,
            output_width,
            scale: 1.0 / (rope.dim as f32).sqrt(),
            rope,
            pipelines,
        }
    }
}

impl AttentionDispatch {
    /// Kernel and threadgroup size for batches too short for the tiled kernels.
    pub fn select(&self) -> (&crate::gpu::Pipeline, usize) {
        match self {
            Self::Vector(p) => (p, 256),
            Self::Plain(p) => (p, 1024),
        }
    }
}

impl Bindings<'_> {
    /// Update history at the supplied sequence positions and mix the requested chunk-relative rows.
    pub fn sink_attention(
        &self,
        a: &SinkAttention,
        plan: &AttentionPlan,
        (s, state): (&AttentionScratch, &Kv),
        positions: Range<usize>,
        output: Range<usize>,
    ) {
        let w = &a.projections;
        let m = positions.len();
        self.project_qkv(w, s, m, output.clone());
        self.attention(
            plan,
            (s, state),
            Some(&a.sinks),
            a.window,
            positions,
            output.clone(),
        );
        if output.is_empty() {
            return;
        }
        w.out
            .apply(self, s.mixed.at(0), &self.scratch.projected, output.len());
    }

    /// Position-free attention: store every input's keys and values, mix the requested rows.
    pub fn plain_attention(
        &self,
        w: &AttentionProjections<Projection>,
        plan: &AttentionPlan,
        (s, state): (&AttentionScratch, &Kv),
        positions: Range<usize>,
        output: Range<usize>,
    ) {
        let m = positions.len();
        self.project_qkv(w, s, m, output.clone());
        self.attention(plan, (s, state), None, 0, positions, output.clone());
        if output.is_empty() {
            return;
        }
        w.out
            .apply(self, s.mixed.at(0), &self.scratch.projected, output.len());
    }

    /// Update history at the supplied sequence positions and gate the requested chunk-relative rows.
    pub fn gated_attention(
        &self,
        a: &GatedAttention,
        plan: &AttentionPlan,
        gated: &GatedAttentionPipelines,
        (s, state): (&AttentionScratch, &Kv),
        positions: Range<usize>,
        output: Range<usize>,
    ) {
        let w = &a.projections;
        let m = positions.len();
        let e = self.encoder;
        self.project_qkv(w, s, m, output.clone());
        // Gated layouts interleave each head's gate after its query; plain Qwen3 queries are packed.
        let q_heads = output.len() * plan.rope.heads as usize;
        let k_heads = m * plan.rope.kv_heads as usize;
        let p = QkNorm {
            dim: plan.rope.dim,
            q_stride: plan.rope.q_stride,
            q_heads: q_heads as u32,
            k_heads: k_heads as u32,
            epsilon: self.kernels.norm.epsilon,
        };
        dispatch(
            e,
            &gated.qk_norm,
            &[s.q.at(0), a.qnorm.binding(), s.k.at(0), a.knorm.binding()],
            bytes_of(&p),
            [q_heads + k_heads, 1, 1],
            256,
        );
        self.attention(plan, (s, state), None, 0, positions, output.clone());
        if output.is_empty() {
            return;
        }
        if plan.rope.q_stride == 2 {
            let n = output.len() * plan.output_width;
            let p = AttentionGateParams {
                n: n as u32,
                dim: plan.rope.dim,
            };
            dispatch(
                e,
                &gated.gate,
                &[s.mixed.at(0), s.q.at(0)],
                bytes_of(&p),
                [n.div_ceil(256), 1, 1],
                256,
            );
        }
        w.out
            .apply(self, s.mixed.at(0), &self.scratch.projected, output.len());
    }

    /// Share normalized gains and project the live queries and full K/V suffix.
    fn project_qkv(
        &self,
        w: &AttentionProjections<Projection>,
        s: &AttentionScratch,
        m: usize,
        output: Range<usize>,
    ) {
        // Ordering the first live projection; K/V remain necessary when no query is consumed.
        let projections = [&w.q.matrix, &w.k.matrix, &w.v.matrix];
        let live = if output.is_empty() {
            &projections[1..]
        } else {
            &projections[..]
        };
        // K/V consume all rows; a query alone needs gains only for its selected output.
        let scale_rows = if projections[1..]
            .iter()
            .any(|matrix| matrix.uses_input_scales(m))
        {
            0..m
        } else {
            output.clone()
        };
        self.prepare_norm_scales(live, m, scale_rows);
        if output.is_empty() {
            w.k.apply_norm(self, &s.k, m);
        } else {
            if output.len() == m {
                w.q.apply_norm(self, &s.q, m);
            } else {
                debug_assert_eq!(output.len(), 1);
                w.q.apply_row(self, &s.q, m, output.start);
            }
            w.k.apply_norm_concurrent(self, &s.k, m);
        }
        w.v.apply_norm_concurrent(self, &s.v, m);
    }

    /// Store the chunk's keys and values, rotate them and the live queries, and mix the
    /// requested chunk-relative rows against the causal history.
    fn attention(
        &self,
        plan: &AttentionPlan,
        (s, state): (&AttentionScratch, &Kv),
        sinks: Option<&ReadOnly>,
        window: usize,
        positions: Range<usize>,
        output: Range<usize>,
    ) {
        self.rotate(plan, s, state, positions.clone(), output.clone());
        // Storing every input's keys and values before pruning outputs; later tokens
        // can attend to these rows even when no prediction is requested from this chunk.
        if output.is_empty() {
            return;
        }
        let pos = positions.start + output.start;
        let keys = pos + output.len();
        let keys = if window == 0 { keys } else { keys.min(window) };
        let p = AttentionParams {
            window: window as u32,
            sinks: sinks.is_some() as u32,
            ..AttentionParams::new(plan, pos, output.len(), 0)
        };
        self.attend(plan, s, state, s.rotated.at(0), sinks, p, keys);
    }

    /// Mix every one of `rows` encoder rows against the first `keys` of them, with no causal
    /// mask. Positions restart at zero: the layer's keys and values are transient, not history.
    pub fn encoder_attention(
        &self,
        w: &AttentionProjections<Projection>,
        plan: &AttentionPlan,
        (s, state): (&AttentionScratch, &Kv),
        rows: usize,
        keys: usize,
    ) {
        debug_assert!(keys <= rows);
        self.project_qkv(w, s, rows, 0..rows);
        self.rotate(plan, s, state, 0..rows, 0..rows);
        let p = AttentionParams::new(plan, 0, rows, keys);
        self.attend(plan, s, state, s.rotated.at(0), None, p, keys);
        w.out
            .apply(self, s.mixed.at(0), &self.scratch.projected, rows);
    }

    /// Mix normalized decoder rows against `keys` precomputed encoder keys and values.
    /// Queries are used as projected; the plan must not rotate them.
    pub fn cross_attention(
        &self,
        q: &Projection,
        out: &Projection,
        plan: &AttentionPlan,
        (s, cross): (&AttentionScratch, &Kv),
        keys: usize,
        rows: usize,
    ) {
        debug_assert_eq!(plan.rope.rope_dim, 0);
        debug_assert_eq!(plan.rope.q_stride, 1);
        self.prepare_norm_scales(&[&q.matrix], rows, 0..rows);
        q.apply_norm(self, &s.q, rows);
        let p = AttentionParams::new(plan, 0, rows, keys);
        self.attend(plan, s, cross, s.q.at(0), None, p, keys);
        out.apply(self, s.mixed.at(0), &self.scratch.projected, rows);
    }

    /// Convert projected keys and values of `rows` encoder outputs into the half-precision
    /// cache layout attention reads, starting at key position zero.
    pub fn store_cross_kv(
        &self,
        k: &Projection,
        v: &Projection,
        plan: &AttentionPlan,
        (s, cross): (&AttentionScratch, &Kv),
        rows: usize,
    ) {
        debug_assert_eq!(plan.rope.rope_dim, 0);
        self.prepare_norm_scales(&[&k.matrix, &v.matrix], rows, 0..rows);
        k.apply_norm(self, &s.k, rows);
        v.apply_norm_concurrent(self, &s.v, rows);
        self.rotate(plan, s, cross, 0..rows, 0..0);
    }

    /// Rotate and store keys and values for all `positions`, and rotate the `output` queries.
    fn rotate(
        &self,
        plan: &AttentionPlan,
        s: &AttentionScratch,
        state: &Kv,
        positions: Range<usize>,
        output: Range<usize>,
    ) {
        let e = self.encoder;
        let m = positions.len();
        let p = Rope {
            m: m as u32,
            pos: positions.start as u32,
            ..plan.rope
        };
        let buffers = [
            s.q.at(0),
            s.k.at(0),
            s.v.at(0),
            s.rotated.at(0),
            state.k.binding(),
            state.v.binding(),
        ];
        if output.len() == m {
            let width = plan.output_width + 2 * plan.key_width;
            dispatch(
                e,
                &plan.pipelines.rope,
                &buffers,
                bytes_of(&p),
                [(m * width).div_ceil(256), 1, 1],
                256,
            );
            return;
        }
        // Reusing the original arithmetic for both ranges; splitting only the index geometry.
        let kv = Rope { heads: 0, ..p };
        dispatch(
            e,
            &plan.pipelines.rope,
            &buffers,
            bytes_of(&kv),
            [(2 * m * plan.key_width).div_ceil(256), 1, 1],
            256,
        );
        if !output.is_empty() {
            let q = Rope {
                kv_heads: 0,
                m: output.len() as u32,
                pos: (positions.start + output.start) as u32,
                ..p
            };
            dispatch_concurrent(
                e,
                &plan.pipelines.rope,
                &buffers,
                bytes_of(&q),
                [(output.len() * plan.output_width).div_ceil(256), 1, 1],
                256,
            );
        }
    }

    /// Score `p.m` query rows against the stored keys and write the mixed rows.
    /// `keys` counts the keys the longest row reads, which selects the kernel.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        plan: &AttentionPlan,
        s: &AttentionScratch,
        state: &Kv,
        q: (&BufferRef, usize),
        sinks: Option<&ReadOnly>,
        p: AttentionParams,
        keys: usize,
    ) {
        let e = self.encoder;
        let m = p.m as usize;
        let window = p.window as usize;
        let sinks = sinks
            .map(ReadOnly::binding)
            .unwrap_or(self.scratch.next.at(0));
        // Splitting keys pays off only for grouped-query decoding over long causal history;
        // cross-attention's single-head groups would leave most of each tile idle.
        if m == 1
            && p.keys == 0
            && keys >= AttentionPlan::SPLIT_MIN_KEYS
            && let (Some(pipelines), Some(partials)) = (&plan.pipelines.split, &s.partials)
        {
            let splits = AttentionPlan::SPLITS;
            let params = SplitAttentionParams {
                attention: p,
                splits: splits as u32,
            };
            let ratio = plan.rope.heads as usize / plan.rope.kv_heads as usize;
            let head_tiles = ratio.div_ceil(8);
            dispatch(
                e,
                &pipelines.partial,
                &[
                    q,
                    state.k.binding(),
                    state.v.binding(),
                    sinks,
                    partials.at(0),
                ],
                bytes_of(&params),
                [plan.rope.kv_heads as usize * head_tiles, 1, splits],
                128,
            );
            // Ordering the merge after every independent key partition has published its FP32 result.
            dispatch(
                e,
                &pipelines.merge,
                &[partials.at(0), s.mixed.at(0)],
                bytes_of(&params),
                [plan.rope.heads as usize, 1, 1],
                128,
            );
            return;
        }
        let heads = plan.rope.heads as usize;
        let pipelines = &plan.pipelines;
        let (pipeline, threads, groups) =
            match (&pipelines.block, &pipelines.wide, &pipelines.tiled) {
                // Bidirectional batches share every key across all rows; the block tiles amortize them.
                (Some(pipeline), _, _) if p.keys != 0 && m >= 8 => {
                    (pipeline, 128, [heads, m.div_ceil(32), 1])
                }
                (_, Some(pipeline), _)
                    if AttentionPlan::wide_prefill(p.dim as usize, m, window) =>
                {
                    (pipeline, 128, [heads, m.div_ceil(16), 1])
                }
                (_, _, Some(pipeline)) if m >= 8 => (pipeline, 128, [heads, m.div_ceil(8), 1]),
                _ => {
                    let (pipeline, threads) = plan.pipelines.dispatch.select();
                    (pipeline, threads, [plan.rope.heads as usize, m, 1])
                }
            };
        dispatch(
            e,
            pipeline,
            &[
                q,
                state.k.binding(),
                state.v.binding(),
                sinks,
                s.mixed.at(0),
            ],
            bytes_of(&p),
            groups,
            threads,
        );
    }
}
