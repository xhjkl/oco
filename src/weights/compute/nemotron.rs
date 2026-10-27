//! Nemotron-H operations: the Mamba-2 history mixer and sigmoid-routed relu² experts.
use super::{Bindings, matrix::Matrix};
use crate::gpu::{BufferRef, ReadOnly, dispatch, dispatch_concurrent, dispatch_indirect};
use crate::weights::{
    nemotron::{Mamba, ReluExperts},
    pipelines::NemotronPipelines,
    state::{MambaScratch, Recurrence, Routing},
};
use crate::weights::{
    operands::{DenseMatrix, Vector},
    pipelines::Builder,
};
use bytemuck::{Pod, Zeroable, bytes_of};
use std::ops::Range;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Ssm {
    pub channels: u32,
    pub m: u32,
    pub stride: u32,
    pub offset: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub groups: u32,
    pub state: u32,
    pub dt_offset: u32,
    pub epsilon: f32,
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
struct RoutePlanN {
    routes: u32,
    up_tiles: u32,
    down_tiles: u32,
    experts: u32,
    tile_rows: u32,
}

/// Words in the indirect tile plan for `routes` gathered rows: two grids, (expert, first row)
/// pairs for every possible tile, and per-expert route lists with their counts.
pub fn plan_words(routes: usize, experts: usize) -> usize {
    6 + 2 * (routes.div_ceil(16) + experts) + experts * (routes + 1)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeAdd {
    hidden: u32,
    batch: u32,
    routes: u32,
    shared: u32,
}

/// Mamba-2 geometry lowered once per model; only the row count varies per submission.
pub struct MambaPlan {
    pipelines: std::sync::Arc<NemotronPipelines>,
    params: Ssm,
}

impl MambaPlan {
    pub const STATE: usize = 128;

    pub fn new(
        heads: usize,
        head_dim: usize,
        groups: usize,
        epsilon: f32,
        pipelines: std::sync::Arc<NemotronPipelines>,
    ) -> Self {
        let inner = heads * head_dim;
        let channels = inner + 2 * groups * Self::STATE;
        Self {
            pipelines,
            params: Ssm {
                channels: channels as u32,
                m: 0,
                stride: (inner + channels + heads) as u32,
                offset: inner as u32,
                heads: heads as u32,
                head_dim: head_dim as u32,
                groups: groups as u32,
                state: Self::STATE as u32,
                dt_offset: (inner + channels) as u32,
                epsilon,
            },
        }
    }

    /// Width of the input projection: gate, convolution inputs, and per-head time steps.
    pub fn projected_width(&self) -> usize {
        self.params.stride as usize
    }

    pub fn conv_width(&self) -> usize {
        self.params.channels as usize
    }

    pub fn inner_width(&self) -> usize {
        (self.params.heads * self.params.head_dim) as usize
    }

    pub fn groups(&self) -> usize {
        self.params.groups as usize
    }

    pub fn heads(&self) -> usize {
        self.params.heads as usize
    }

    pub fn head_dim(&self) -> usize {
        self.params.head_dim as usize
    }
}

/// Routed relu² expert geometry shared by every Nemotron MoE block.
pub struct ReluExpertPlan {
    pipelines: std::sync::Arc<NemotronPipelines>,
    up: Matrix,
    down: Matrix,
    router: RouterParams,
    pub shared_width: usize,
    /// Single-row kernels for the shared expert's up (plain) and down (relu²-input) products.
    shared_vector: Vector,
    shared_vector_relu: Vector,
}

impl ReluExpertPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        expert_width: usize,
        shared_width: usize,
        experts: usize,
        active: usize,
        normalize: bool,
        scale: f32,
        pipelines: std::sync::Arc<NemotronPipelines>,
        (shared_vector, shared_vector_relu): (Vector, Vector),
    ) -> Self {
        let up = Matrix {
            k: hidden as u32,
            n: expert_width as u32,
            m: 0,
            route_major: 0,
            routes: active as u32,
            experts: experts as u32,
            has_bias: 0,
        };
        let down = Matrix {
            k: expert_width as u32,
            n: hidden as u32,
            route_major: 1,
            ..up
        };
        Self {
            pipelines,
            up,
            down,
            router: RouterParams {
                n: hidden as u32,
                experts: experts as u32,
                active: active as u32,
                normalize: normalize as u32,
                scale,
            },
            shared_width,
            shared_vector,
            shared_vector_relu,
        }
    }

    pub fn active(&self) -> usize {
        self.router.active as usize
    }

    pub fn experts(&self) -> usize {
        self.router.experts as usize
    }

    pub fn expert_width(&self) -> usize {
        self.up.n as usize
    }
}

impl Bindings<'_> {
    /// Update the recurrent history with every input row, then emit only the requested rows.
    pub fn mamba(
        &self,
        w: &Mamba,
        plan: &MambaPlan,
        (s, state): (&MambaScratch, &Recurrence),
        rows: usize,
        output: Range<usize>,
    ) {
        let e = self.encoder;
        let pipelines = &plan.pipelines;
        self.prepare_norm_scales(&[&w.in_proj.matrix], rows, 0..rows);
        w.in_proj.apply_norm(self, &s.projected, rows);
        // The time steps alongside, in FP32, from the same normalized input.
        w.dt.apply(
            e,
            None,
            self.scratch.norm.binding(),
            &s.dt,
            rows,
            &self.scratch.next,
            self.scratch.norm_scales.binding(),
            true,
        );
        let p = Ssm {
            m: rows as u32,
            ..plan.params
        };
        dispatch(
            e,
            &pipelines.conv,
            &[
                s.projected.at(0),
                w.conv.binding(),
                w.conv_bias.binding(),
                state.conv.binding(),
                s.convolved.at(0),
            ],
            bytes_of(&p),
            [plan.conv_width().div_ceil(256), 1, 1],
            256,
        );
        dispatch(
            e,
            &pipelines.scan,
            &[
                s.convolved.at(0),
                s.dt.at(0),
                w.a.binding(),
                w.d.binding(),
                w.dt_bias.binding(),
                state.ssm.binding(),
                s.scanned.at(0),
            ],
            bytes_of(&p),
            [plan.head_dim().div_ceil(4), plan.heads(), 1],
            128,
        );
        if output.is_empty() {
            return;
        }
        // Gating and normalizing only the rows that continue; history already holds the rest.
        let inner = plan.inner_width();
        let p = Ssm {
            m: output.len() as u32,
            ..p
        };
        dispatch(
            e,
            &pipelines.gate_norm,
            &[
                s.scanned.at(output.start * inner * 4),
                s.projected.at(output.start * plan.projected_width() * 2),
                w.norm.binding(),
                s.mixed.at(0),
            ],
            bytes_of(&p),
            [plan.groups(), output.len(), 1],
            256,
        );
        w.out
            .apply(self, s.mixed.at(0), &self.scratch.projected, output.len());
    }

    /// Add the shared expert and the weighted routed experts to the given residual rows.
    pub fn relu_experts(
        &self,
        w: &ReluExperts,
        plan: &ReluExpertPlan,
        routing: &Routing,
        residual: (&BufferRef, usize),
        rows: usize,
    ) {
        if rows == 1 {
            return self.relu_experts_single(w, plan, routing, residual);
        }
        let e = self.encoder;
        let s = self.scratch;
        let pipelines = &plan.pipelines;
        let hidden = plan.up.k as usize;
        let active = plan.active();
        // Shared expert: dense up, relu², dense down, straight onto the residual.
        self.prepare_norm_scales(&[&w.shared_up.matrix], rows, 0..rows);
        w.shared_up.apply_norm(self, &s.up, rows);
        let n = (rows * plan.shared_width) as u32;
        dispatch(
            e,
            &pipelines.relu_sqr,
            &[s.up.at(0)],
            bytes_of(&n),
            [(n as usize).div_ceil(256), 1, 1],
            256,
        );
        w.shared_down.apply(self, s.up.at(0), &s.projected, rows);
        self.add(&s.projected, residual, rows);
        // Routed experts: select, plan tiles per expert, gather, contract, and combine.
        // Relu² is applied as the down tiles load their inputs.
        dispatch(
            e,
            &pipelines.router,
            &[
                s.norm.at(0),
                w.router.values.binding(),
                w.router.bias.binding(),
                routing.ids.at(0),
                routing.prob.at(0),
            ],
            bytes_of(&plan.router),
            [rows, 1, 1],
            256,
        );
        let routes = rows * active;
        // Tensor tiles reuse decoded coefficients across 32 routes; eight-route tiles waste
        // less padding when experts see few routes each.
        let size = usize::from(routes >= 8 * plan.experts());
        let tile_rows = [8u32, 32][size];
        let (up, down) = (&pipelines.expert_up[size], &pipelines.expert_down[size]);
        let p = RoutePlanN {
            routes: routes as u32,
            up_tiles: plan.expert_width().div_ceil(64) as u32,
            down_tiles: hidden.div_ceil(64) as u32,
            experts: plan.experts() as u32,
            tile_rows,
        };
        dispatch(
            e,
            &pipelines.route_plan,
            &[routing.ids.at(0), routing.plan.at(0)],
            bytes_of(&p),
            [1, 1, 1],
            128,
        );
        let p = Matrix {
            m: routes as u32,
            ..plan.up
        };
        dispatch_indirect(
            e,
            up,
            &[
                w.up.binding(),
                s.norm.at(0),
                s.gate.at(0),
                routing.plan.binding(),
            ],
            bytes_of(&p),
            routing.plan.at(0),
            128,
        );
        let p = Matrix {
            m: routes as u32,
            ..plan.down
        };
        dispatch_indirect(
            e,
            down,
            &[
                w.down.binding(),
                s.gate.at(0),
                s.projected.at(0),
                routing.plan.binding(),
            ],
            bytes_of(&p),
            routing.plan.at(12),
            128,
        );
        let p = MoeAdd {
            hidden: hidden as u32,
            batch: rows as u32,
            routes: active as u32,
            shared: 0,
        };
        dispatch(
            e,
            &pipelines.moe_add,
            &[
                s.projected.at(0),
                routing.prob.at(0),
                residual,
                s.next.binding(),
            ],
            bytes_of(&p),
            [(rows * hidden).div_ceil(256), 1, 1],
            256,
        );
    }

    /// One row: four ordered stages, with the shared expert overlapping the routed products
    /// and relu² folded into the down projections' input loads.
    fn relu_experts_single(
        &self,
        w: &ReluExperts,
        plan: &ReluExpertPlan,
        routing: &Routing,
        residual: (&BufferRef, usize),
    ) {
        let e = self.encoder;
        let s = self.scratch;
        let pipelines = &plan.pipelines;
        let hidden = plan.up.k as usize;
        let active = plan.active();
        let shared_out = s.projected.at(active * hidden * 2);
        let dense = |matrix: &DenseMatrix, bias: Option<&ReadOnly>| Matrix {
            k: matrix.input as u32,
            n: matrix.output as u32,
            m: 1,
            route_major: 0,
            routes: 4,
            experts: 1,
            has_bias: bias.is_some() as u32,
        };
        let dummy = s.next.binding();
        let up_bias = w
            .shared_up
            .bias
            .as_ref()
            .map(ReadOnly::binding)
            .unwrap_or(dummy);
        let down_bias = w
            .shared_down
            .bias
            .as_ref()
            .map(ReadOnly::binding)
            .unwrap_or(dummy);
        // Stage 1: router and shared up both read the normalized row.
        dispatch(
            e,
            &pipelines.router,
            &[
                s.norm.at(0),
                w.router.values.binding(),
                w.router.bias.binding(),
                routing.ids.at(0),
                routing.prob.at(0),
            ],
            bytes_of(&plan.router),
            [1, 1, 1],
            256,
        );
        let up = &w.shared_up.matrix;
        dispatch_concurrent(
            e,
            &plan.shared_vector.pipeline,
            &[
                up.values.binding(),
                s.norm.at(0),
                s.up.at(0),
                up_bias,
                dummy,
            ],
            bytes_of(&dense(up, w.shared_up.bias.as_ref())),
            [plan.shared_vector.column_groups, 1, 1],
            plan.shared_vector.threads,
        );
        // Stage 2: routed up needs the selection; shared down needs only the shared up.
        let p = Matrix {
            m: active as u32,
            ..plan.up
        };
        let threads = Builder::vector_threads(plan.expert_width());
        dispatch(
            e,
            &pipelines.expert_vector,
            &[
                w.up.binding(),
                s.norm.at(0),
                s.gate.at(0),
                s.next.binding(),
                routing.ids.at(0),
            ],
            bytes_of(&p),
            [plan.expert_width().div_ceil(threads / 8), active, 1],
            threads,
        );
        let down = &w.shared_down.matrix;
        dispatch_concurrent(
            e,
            &plan.shared_vector_relu.pipeline,
            &[
                down.values.binding(),
                s.up.at(0),
                shared_out,
                down_bias,
                dummy,
            ],
            bytes_of(&dense(down, w.shared_down.bias.as_ref())),
            [plan.shared_vector_relu.column_groups, 1, 1],
            plan.shared_vector_relu.threads,
        );
        // Stage 3: routed down over the activated expert intermediates.
        let p = Matrix {
            m: active as u32,
            ..plan.down
        };
        let threads = Builder::vector_threads(hidden);
        dispatch(
            e,
            &pipelines.expert_vector_relu,
            &[
                w.down.binding(),
                s.gate.at(0),
                s.projected.at(0),
                s.next.binding(),
                routing.ids.at(0),
            ],
            bytes_of(&p),
            [hidden.div_ceil(threads / 8), active, 1],
            threads,
        );
        // Stage 4: weighted routes plus the shared expert onto the residual.
        let p = MoeAdd {
            hidden: hidden as u32,
            batch: 1,
            routes: active as u32,
            shared: 1,
        };
        dispatch(
            e,
            &pipelines.moe_add,
            &[s.projected.at(0), routing.prob.at(0), residual, shared_out],
            bytes_of(&p),
            [hidden.div_ceil(256), 1, 1],
            256,
        );
    }
}
