use super::Bindings;
use crate::gpu::dispatch;
use crate::weights::{
    pipelines::RecurrentPipelines,
    qwen::delta::{Delta, KEY_WIDTH},
    state::{DeltaScratch, Recurrence},
};
use bytemuck::{Pod, Zeroable, bytes_of};
use std::ops::Range;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Gdn {
    kh: u32,
    vh: u32,
    vd: u32,
    m: u32,
    conv_dim: u32,
    epsilon: f32,
}

/// Prepared geometry and kernels for Qwen's gated delta updates to recurrent memory.
pub struct DeltaPlan {
    pipelines: RecurrentPipelines,
    params: Gdn,
    conv_groups: usize,
    output_row_bytes: usize,
}

impl DeltaPlan {
    pub fn new(
        kh: usize,
        vh: usize,
        vd: usize,
        epsilon: f32,
        pipelines: RecurrentPipelines,
    ) -> Self {
        let conv_dim = 2 * kh * KEY_WIDTH + vh * vd;
        Self {
            pipelines,
            params: Gdn {
                kh: kh as u32,
                vh: vh as u32,
                vd: vd as u32,
                m: 0,
                conv_dim: conv_dim as u32,
                epsilon,
            },
            conv_groups: conv_dim.div_ceil(256),
            output_row_bytes: vh * vd * 2,
        }
    }

    pub fn conv_width(&self) -> usize {
        self.params.conv_dim as usize
    }

    pub fn output_width(&self) -> usize {
        self.output_row_bytes / 2
    }

    pub fn normalized_width(&self) -> usize {
        2 * self.params.kh as usize * KEY_WIDTH
    }

    pub fn factor_width(&self) -> usize {
        2 * self.params.vh as usize + self.params.kh as usize
    }

    pub fn heads(&self) -> usize {
        self.params.vh as usize
    }
}

impl Bindings<'_> {
    /// Update recurrent history with every input row, then project only the requested output rows.
    /// Skipping an output does not skip its contribution to the memory used by future tokens.
    pub fn delta(
        &self,
        d: &Delta,
        plan: &DeltaPlan,
        (s, state): (&DeltaScratch, &Recurrence),
        rows: usize,
        output: Range<usize>,
    ) {
        let e = self.encoder;
        let pipelines = &plan.pipelines;
        // The four projections share normalized inputs/gains and write distinct buffers.
        self.prepare_norm_scales(
            &[&d.qkv.matrix, &d.z.matrix, &d.alpha.matrix, &d.beta.matrix],
            rows,
            0..rows,
        );
        d.qkv.apply_norm(self, &s.qkv, rows);
        // Keeping the fused small-row path intact; chunked state-only passes need no output gate.
        if rows < 4 || !output.is_empty() {
            d.z.apply_norm_concurrent(self, &s.z, rows);
        }
        d.alpha.apply_norm_concurrent(self, &s.alpha, rows);
        d.beta.apply_norm_concurrent(self, &s.beta, rows);
        let p = Gdn {
            m: rows as u32,
            ..plan.params
        };
        dispatch(
            e,
            &pipelines.conv,
            &[
                s.qkv.at(0),
                d.conv.binding(),
                state.conv.binding(),
                s.convolved.at(0),
            ],
            bytes_of(&p),
            [plan.conv_groups, 1, 1],
            256,
        );
        if rows < 4 {
            dispatch(
                e,
                &pipelines.delta,
                &[
                    s.convolved.at(0),
                    s.z.at(0),
                    s.alpha.at(0),
                    s.beta.at(0),
                    d.a.binding(),
                    d.dt.binding(),
                    d.norm.binding(),
                    state.ssm.binding(),
                    s.mixed.at(0),
                ],
                bytes_of(&p),
                [plan.heads(), 1, 1],
                512,
            );
        } else {
            dispatch(
                e,
                &pipelines.delta_prepare,
                &[
                    s.convolved.at(0),
                    s.alpha.at(0),
                    s.beta.at(0),
                    d.a.binding(),
                    d.dt.binding(),
                    s.qk.at(0),
                    s.factors.at(0),
                ],
                bytes_of(&p),
                [p.kh as usize, rows, 1],
                32,
            );
            dispatch(
                e,
                &pipelines.delta_persistent,
                &[
                    s.qk.at(0),
                    s.factors.at(0),
                    s.convolved.at(0),
                    state.ssm.binding(),
                    s.values.at(0),
                ],
                bytes_of(&p),
                [(p.vd as usize).div_ceil(4), plan.heads(), 1],
                128,
            );
            if output.is_empty() {
                return;
            }
            dispatch(
                e,
                &pipelines.delta_output,
                &[s.values.at(0), s.z.at(0), d.norm.binding(), s.mixed.at(0)],
                bytes_of(&p),
                [plan.heads(), rows, 1],
                128,
            );
        }
        if output.is_empty() {
            return;
        }
        d.out.apply(
            self,
            s.mixed.at(output.start * plan.output_row_bytes),
            &self.scratch.projected,
            output.len(),
        );
    }
}
