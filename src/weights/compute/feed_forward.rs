use super::{Bindings, matrix::Matrix};
use crate::gpu::BufferRef;
use crate::gpu::{dispatch, dispatch_indirect};
use crate::weights::{
    gpt_oss::{ACTIVE_EXPERTS, EXPERT_COUNT, Experts},
    pipelines::ExpertPipelines,
    qwen::DenseFfn,
    state::Routing,
};
use bytemuck::{Pod, Zeroable, bytes_of};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Gate {
    n: u32,
    clipped: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RoutePlan {
    routes: u32,
    up_tiles: u32,
    down_tiles: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Moe {
    hidden: u32,
    batch: u32,
}

/// GPT expert geometry lowered once; only route count and active rows vary per submission.
pub struct ExpertPlan {
    pipelines: ExpertPipelines,
    route_plan: RoutePlan,
    up: Matrix,
    down: Matrix,
    expand_groups: usize,
    down_groups: usize,
}

impl ExpertPlan {
    pub fn new(hidden: usize, ffn_width: usize, pipelines: ExpertPipelines) -> Self {
        let up = Matrix {
            k: hidden as u32,
            n: ffn_width as u32,
            m: 0,
            route_major: 0,
            routes: ACTIVE_EXPERTS as u32,
            experts: EXPERT_COUNT as u32,
            has_bias: 1,
        };
        let down = Matrix {
            k: ffn_width as u32,
            n: hidden as u32,
            route_major: 1,
            ..up
        };
        Self {
            pipelines,
            route_plan: RoutePlan {
                routes: 0,
                up_tiles: (2 * ffn_width.div_ceil(64)) as u32,
                down_tiles: hidden.div_ceil(64) as u32,
            },
            up,
            down,
            expand_groups: ffn_width.div_ceil(16),
            down_groups: hidden.div_ceil(32),
        }
    }

    pub fn width(&self) -> usize {
        self.up.n as usize
    }
}

impl Bindings<'_> {
    fn activate(&self, n: usize, clipped: bool) {
        let p = Gate {
            n: n as u32,
            clipped: clipped as u32,
        };
        // Overwriting each consumed gate element after reading it.
        dispatch(
            self.encoder,
            &self.base.pipelines.gate,
            &[
                self.scratch.gate.at(0),
                self.scratch.up.at(0),
                self.scratch.gate.at(0),
            ],
            bytes_of(&p),
            [n.div_ceil(256), 1, 1],
            256,
        );
    }

    /// Refine each token's features independently, using a learned gate before projecting back.
    pub fn feed_forward(
        &self,
        w: &DenseFfn,
        ffn_width: usize,
        hidden: (&BufferRef, usize),
        rows: usize,
    ) {
        let s = self.scratch;
        self.prepare_norm_scales(&[&w.gate.matrix, &w.up.matrix], rows, 0..rows);
        w.gate.apply_norm(self, &s.gate, rows);
        w.up.apply_norm_concurrent(self, &s.up, rows);
        self.activate(rows * ffn_width, false);
        w.down.apply(self, s.gate.at(0), &s.projected, rows);
        self.add(&s.projected, hidden, rows);
    }

    /// Route each token to four experts and add their weighted contributions to its features.
    pub fn experts(
        &self,
        w: &Experts,
        plan: &ExpertPlan,
        routing: &Routing,
        residual: (&BufferRef, usize),
        rows: usize,
    ) {
        let e = self.encoder;
        let s = self.scratch;
        let pipelines = &plan.pipelines;
        // Tensor tiles cover 32 routes per expert and skip their empty upper half.
        let (route_plan, up, down) = (
            &pipelines.route_plan,
            &pipelines.up_tensor,
            &pipelines.down_tensor,
        );
        let hidden = plan.up.k as usize;
        if rows == 1 {
            // Single tokens compute the 32 router logits across threadgroups; the expansion
            // selects the experts itself and records ids and probabilities for contraction.
            dispatch(
                e,
                &pipelines.router_logits,
                &[
                    s.norm.at(0),
                    w.router.values.binding(),
                    w.router.bias.binding(),
                    routing.plan.at(0),
                ],
                bytes_of(&(hidden as u32)),
                [EXPERT_COUNT, 1, 1],
                32,
            );
            let p = Matrix {
                m: ACTIVE_EXPERTS as u32,
                ..plan.up
            };
            dispatch(
                e,
                &pipelines.expand,
                &[
                    w.up.values.binding(),
                    w.gate.values.binding(),
                    s.norm.at(0),
                    s.up.at(0),
                    s.gate.at(0),
                    w.up.bias.binding(),
                    w.gate.bias.binding(),
                    routing.ids.binding(),
                    routing.plan.at(0),
                    routing.prob.binding(),
                ],
                bytes_of(&p),
                [plan.expand_groups, ACTIVE_EXPERTS, 1],
                256,
            );
        } else {
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
                bytes_of(&(hidden as u32)),
                [rows, 1, 1],
                256,
            );
            let p = RoutePlan {
                routes: (rows * ACTIVE_EXPERTS) as u32,
                ..plan.route_plan
            };
            dispatch(
                e,
                route_plan,
                &[routing.ids.at(0), routing.plan.at(0)],
                bytes_of(&p),
                [1, 1, 1],
                32,
            );
            let p = Matrix {
                m: (rows * ACTIVE_EXPERTS) as u32,
                ..plan.up
            };
            let buffers = [
                w.up.values.binding(),
                w.gate.values.binding(),
                s.norm.at(0),
                s.up.at(0),
                s.gate.at(0),
                w.up.bias.binding(),
                w.gate.bias.binding(),
                routing.plan.binding(),
            ];
            dispatch_indirect(e, up, &buffers, bytes_of(&p), routing.plan.at(0), 128);
        }
        let p = Matrix {
            m: (rows * ACTIVE_EXPERTS) as u32,
            ..plan.down
        };
        if rows > 1 {
            self.activate(rows * ACTIVE_EXPERTS * plan.width(), true);
            dispatch_indirect(
                e,
                down,
                &[
                    w.down.values.binding(),
                    s.gate.at(0),
                    s.projected.at(0),
                    w.down.bias.binding(),
                    routing.plan.binding(),
                ],
                bytes_of(&p),
                routing.plan.at(12),
                128,
            );
        } else {
            // One token: the expansion already activated its outputs in place.
            dispatch(
                e,
                &pipelines.down_vector,
                &[
                    w.down.values.binding(),
                    s.gate.at(0),
                    s.projected.at(0),
                    w.down.bias.binding(),
                    routing.ids.binding(),
                ],
                bytes_of(&p),
                [plan.down_groups, rows * ACTIVE_EXPERTS, 1],
                256,
            );
        }
        let p = Moe {
            hidden: hidden as u32,
            batch: rows as u32,
        };
        dispatch(
            e,
            &pipelines.sum,
            &[s.projected.at(0), routing.prob.at(0), residual],
            bytes_of(&p),
            [(rows * hidden).div_ceil(256), 1, 1],
            256,
        );
    }
}
