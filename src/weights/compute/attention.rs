use super::Bindings;
use crate::{
    gpu::{ReadOnly, dispatch, dispatch_concurrent},
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttentionParams {
    heads: u32,
    kv_heads: u32,
    dim: u32,
    pos: u32,
    window: u32,
    sinks: u32,
    scale: f32,
    m: u32,
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
    stride: u32,
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
    /// Select the fallback kernel after wide and tiled prefill have been handled.
    /// D64/D256 reach this path only with fewer than eight output rows.
    pub fn select(&self, rows: usize, keys: usize) -> (&crate::gpu::Pipeline, usize) {
        match self {
            Self::Plain(full) => (full, 1024),
            Self::Compact256(p) => (p, 256),
            Self::Dim64 { compact_256, .. } if rows == 1 && keys <= 128 => (compact_256, 256),
            Self::Dim64 { full, .. } => (full, 1024),
            Self::Dim256 { compact_256, .. }
                if (rows == 1 && keys <= 2048) || ((2..=7).contains(&rows) && keys <= 1024) =>
            {
                (compact_256, 256)
            }
            Self::Dim256 { full, .. } => (full, 1024),
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
        let mut p = QkNorm {
            dim: plan.rope.dim,
            stride: 2,
            epsilon: self.base.norm.epsilon,
        };
        if !output.is_empty() {
            dispatch(
                e,
                &gated.qk_norm,
                &[s.q.at(0), a.qnorm.binding()],
                bytes_of(&p),
                [output.len() * plan.rope.heads as usize, 1, 1],
                256,
            );
        }
        p.stride = 1;
        dispatch(
            e,
            &gated.qk_norm,
            &[s.k.at(0), a.knorm.binding()],
            bytes_of(&p),
            [m * plan.rope.kv_heads as usize, 1, 1],
            256,
        );
        self.attention(plan, (s, state), None, 0, positions, output.clone());
        if output.is_empty() {
            return;
        }
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

    fn attention(
        &self,
        plan: &AttentionPlan,
        (s, state): (&AttentionScratch, &Kv),
        sinks: Option<&ReadOnly>,
        window: usize,
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
        } else {
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
        // Storing every input's keys and values before pruning outputs; later tokens
        // can attend to these rows even when no prediction is requested from this chunk.
        if output.is_empty() {
            return;
        }
        let m = output.len();
        let first = output.start;
        let p = AttentionParams {
            heads: plan.rope.heads,
            kv_heads: plan.rope.kv_heads,
            dim: plan.rope.dim,
            pos: (positions.start + first) as u32,
            window: window as u32,
            sinks: sinks.is_some() as u32,
            scale: plan.scale,
            m: m as u32,
        };
        let sinks = sinks
            .map(ReadOnly::binding)
            .unwrap_or(self.scratch.next.at(0));
        let keys = positions.start + first + m;
        let keys = if window == 0 { keys } else { keys.min(window) };
        if m == 1
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
                    s.rotated.at(0),
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
        let (pipeline, threads, groups) = match (&plan.pipelines.wide, &plan.pipelines.tiled) {
            (Some(pipeline), _) if AttentionPlan::wide_prefill(p.dim as usize, m, window) => {
                (pipeline, 128, [plan.rope.heads as usize, m.div_ceil(16), 1])
            }
            (_, Some(pipeline)) if m >= 8 => {
                (pipeline, 128, [plan.rope.heads as usize, m.div_ceil(8), 1])
            }
            _ => {
                let (pipeline, threads) = plan.pipelines.dispatch.select(m, keys);
                (pipeline, threads, [plan.rope.heads as usize, m, 1])
            }
        };
        dispatch(
            e,
            pipeline,
            &[
                s.rotated.at(0),
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
