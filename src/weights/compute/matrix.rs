use super::Bindings;
use crate::gpu::{BufferRef, Commands};
use crate::{
    gpu::{BufferView, ReadOnly, dispatch, dispatch_concurrent},
    weights::operands::*,
};
use bytemuck::{Pod, Zeroable, bytes_of};
use std::ops::Range;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Matrix {
    pub k: u32,
    pub n: u32,
    pub m: u32,
    pub route_major: u32,
    pub routes: u32,
    pub experts: u32,
    pub has_bias: u32,
}

impl DenseMatrix {
    pub(super) fn uses_input_scales(&self, rows: usize) -> bool {
        matches!(&self.plan, DensePlan::Tiled { input_scales: true, vector_rows, .. } if rows > *vector_rows)
    }

    fn prepare_input_scales(
        &self,
        compute: &Bindings<'_>,
        x: (&BufferRef, usize),
        scales: &BufferView,
        rows: usize,
        selected: Range<usize>,
    ) {
        if !self.uses_input_scales(rows) || selected.is_empty() {
            return;
        }
        debug_assert!(selected.end <= rows);
        let (buffer, offset) = x;
        let x = (buffer, offset + selected.start * self.input * 2);
        let scales = scales.at(selected.start * 8);
        let p = Matrix {
            k: self.input as u32,
            n: self.output as u32,
            m: selected.len() as u32,
            route_major: 0,
            routes: 4,
            experts: 1,
            has_bias: 0,
        };
        dispatch(
            compute.encoder,
            &compute.kernels.pipelines.q4_input_scales,
            &[x, scales],
            bytes_of(&p),
            [selected.len(), 1, 1],
            256,
        );
    }

    /// Project one row while preserving its original vector or matrix reduction path.
    /// The normalized projection group has already prepared any required row gains.
    fn apply_row(
        &self,
        compute: &Bindings<'_>,
        bias: Option<&ReadOnly>,
        y: &BufferView,
        rows: usize,
        row: usize,
    ) {
        debug_assert!(row < rows);
        let (pipeline, columns, threads) = match &self.plan {
            DensePlan::Vector(v) => (&v.pipeline, v.column_groups, v.threads),
            DensePlan::Tiled {
                vector: v,
                vector_rows,
                ..
            } if rows <= *vector_rows => (&v.pipeline, v.column_groups, v.threads),
            DensePlan::Tiled { narrow, .. } => {
                // Keeping the matrix reduction; the smallest tile avoids discarded query rows.
                let tile = &narrow[0];
                (&tile.pipeline, tile.column_groups, 128)
            }
        };
        let p = Matrix {
            k: self.input as u32,
            n: self.output as u32,
            m: 1,
            route_major: 0,
            routes: 4,
            experts: 1,
            has_bias: bias.is_some() as u32,
        };
        let dummy = &compute.scratch.next;
        let bias = bias.map(ReadOnly::binding).unwrap_or(dummy.binding());
        let scales = if self.uses_input_scales(rows) {
            compute.scratch.norm_scales.at(row * 8)
        } else {
            dummy.binding()
        };
        let buffers = [
            self.values.binding(),
            compute.scratch.norm.at(row * self.input * 2),
            y.binding(),
            bias,
            scales,
        ];
        let groups = if threads == 128 {
            [1, columns, 1]
        } else {
            [columns, 1, 1]
        };
        dispatch(
            compute.encoder,
            pipeline,
            &buffers,
            bytes_of(&p),
            groups,
            threads,
        );
    }

    /// `concurrent` skips the ordering barrier: the caller guarantees this product depends on
    /// nothing dispatched since the previous barrier, so it overlaps its predecessor.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply(
        &self,
        e: &Commands,
        bias: Option<&ReadOnly>,
        x: (&BufferRef, usize),
        y: &BufferView,
        m: usize,
        dummy: &BufferView,
        scales: (&BufferRef, usize),
        concurrent: bool,
    ) {
        let first = if concurrent {
            dispatch_concurrent
        } else {
            dispatch
        };
        let p = Matrix {
            k: self.input as u32,
            n: self.output as u32,
            m: m as u32,
            route_major: 0,
            routes: 4,
            experts: 1,
            has_bias: bias.is_some() as u32,
        };
        let bias = bias.map(ReadOnly::binding).unwrap_or(dummy.binding());
        let scaled = self.uses_input_scales(m);
        let scales = if scaled { scales } else { dummy.binding() };
        let buffers = [self.values.binding(), x, y.binding(), bias, scales];
        let (narrow, wide) = match &self.plan {
            DensePlan::Vector(v) => {
                first(
                    e,
                    &v.pipeline,
                    &buffers,
                    bytes_of(&p),
                    [v.column_groups, m, 1],
                    v.threads,
                );
                return;
            }
            DensePlan::Tiled {
                vector,
                vector_rows,
                narrow,
                wide,
                ..
            } => {
                if m <= *vector_rows {
                    let groups = [vector.column_groups, m, 1];
                    first(
                        e,
                        &vector.pipeline,
                        &buffers,
                        bytes_of(&p),
                        groups,
                        vector.threads,
                    );
                    return;
                }
                (narrow, wide)
            }
        };
        // Complete wide tiles share decoded weights across 64 rows. One independent tail
        // dispatch covers the remainder, using two guarded 32-row groups when needed.
        let full = if m >= wide.rows {
            m / wide.rows * wide.rows
        } else {
            0
        };
        if full > 0 {
            let p = Matrix {
                m: full as u32,
                ..p
            };
            let groups = [full / wide.rows, wide.column_groups, 1];
            first(e, &wide.pipeline, &buffers, bytes_of(&p), groups, 128);
        }
        let tail = m - full;
        if tail > 0 {
            let tile = &narrow[tail.min(32).div_ceil(8) - 1];
            let p = Matrix {
                m: tail as u32,
                ..p
            };
            let (x_buffer, x_offset) = x;
            let (y_buffer, y_offset) = y.binding();
            let (scale_buffer, scale_offset) = scales;
            let scales = if scaled {
                (scale_buffer, scale_offset + full * 8)
            } else {
                scales
            };
            let buffers = [
                self.values.binding(),
                (x_buffer, x_offset + full * self.input * 2),
                (y_buffer, y_offset + full * self.output * 2),
                bias,
                scales,
            ];
            let groups = [tail.div_ceil(tile.rows), tile.column_groups, 1];
            if full > 0 {
                dispatch_concurrent(e, &tile.pipeline, &buffers, bytes_of(&p), groups, 128);
            } else {
                first(e, &tile.pipeline, &buffers, bytes_of(&p), groups, 128);
            }
        }
    }
}

impl Projection {
    pub(super) fn apply_row(
        &self,
        compute: &Bindings<'_>,
        y: &BufferView,
        rows: usize,
        row: usize,
    ) {
        self.matrix
            .apply_row(compute, self.bias.as_ref(), y, rows, row);
    }

    /// Project an intermediate, preparing transient gains only when its tiled plan needs them.
    pub(crate) fn apply(
        &self,
        compute: &Bindings<'_>,
        x: (&BufferRef, usize),
        y: &BufferView,
        m: usize,
    ) {
        let s = compute.scratch;
        self.matrix
            .prepare_input_scales(compute, x, &s.projection_scales, m, 0..m);
        self.matrix.apply(
            compute.encoder,
            self.bias.as_ref(),
            x,
            y,
            m,
            &s.next,
            s.projection_scales.binding(),
            false,
        );
    }

    /// Consume the normalized input and gains prepared for this projection group.
    pub(super) fn apply_norm(&self, compute: &Bindings<'_>, y: &BufferView, m: usize) {
        let s = compute.scratch;
        self.matrix.apply(
            compute.encoder,
            self.bias.as_ref(),
            s.norm.binding(),
            y,
            m,
            &s.next,
            s.norm_scales.binding(),
            false,
        );
    }

    /// Consume the same normalized input and gains alongside another independent projection.
    pub(super) fn apply_norm_concurrent(&self, compute: &Bindings<'_>, y: &BufferView, m: usize) {
        let s = compute.scratch;
        self.matrix.apply(
            compute.encoder,
            self.bias.as_ref(),
            s.norm.binding(),
            y,
            m,
            &s.next,
            s.norm_scales.binding(),
            true,
        );
    }
}

impl Bindings<'_> {
    /// Prepare gains once for all live projections of the current normalized input.
    /// Every matrix in the group consumes the same row width; gains remain immutable until
    /// the next ordered group starts after all current products have consumed them.
    /// `rows` selects the original tiled path; `selected` limits the rows actually scanned.
    pub(super) fn prepare_norm_scales(
        &self,
        matrices: &[&DenseMatrix],
        rows: usize,
        selected: Range<usize>,
    ) {
        let Some(matrix) = matrices
            .iter()
            .find(|matrix| matrix.uses_input_scales(rows))
        else {
            return;
        };
        matrix.prepare_input_scales(
            self,
            self.scratch.norm.binding(),
            &self.scratch.norm_scales,
            rows,
            selected,
        );
    }

    /// Look up the learned feature vector for each supplied token before the first layer.
    /// IDs at or beyond the vocabulary copy the corresponding precomputed `features` row instead.
    pub fn embed(
        &self,
        w: &Embedding,
        input: &BufferView,
        rows: usize,
        features: (&BufferRef, usize),
    ) {
        let p = Matrix {
            k: w.width as u32,
            n: w.vocabulary as u32,
            m: rows as u32,
            route_major: 0,
            routes: 0,
            experts: 0,
            has_bias: 0,
        };
        dispatch(
            self.encoder,
            &w.pipeline,
            &[
                w.values.binding(),
                input.binding(),
                self.scratch.hidden.at(0),
                features,
            ],
            bytes_of(&p),
            [(rows * w.width).div_ceil(256), 1, 1],
            256,
        );
    }

    /// Score every vocabulary token as a possible successor to the final input row.
    pub fn output(&self, w: &Output) {
        let p = Matrix {
            k: w.width as u32,
            n: w.vocabulary as u32,
            m: 1,
            route_major: 0,
            routes: 4,
            experts: 1,
            has_bias: w.bias.is_some() as u32,
        };
        let bias = w
            .bias
            .as_ref()
            .map(ReadOnly::binding)
            .unwrap_or(self.scratch.next.binding());
        dispatch(
            self.encoder,
            &w.vector.pipeline,
            &[
                w.values.binding(),
                self.scratch.norm.binding(),
                self.scratch.logits.binding(),
                bias,
                self.scratch.next.binding(),
            ],
            bytes_of(&p),
            [w.vector.column_groups, 1, 1],
            w.vector.threads,
        );
    }
}
