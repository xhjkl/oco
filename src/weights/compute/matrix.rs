use super::Bindings;
use crate::gpu::{BufferRef, Commands};
use crate::{
    gpu::{BufferView, ReadOnly, dispatch, dispatch_concurrent},
    weights::{operands::*, state::CommonScratch},
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
        let (pipeline, groups, threads) = match &self.plan {
            DensePlan::Vector(v) => (&v.pipeline, [v.column_groups, 1, 1], v.threads),
            DensePlan::Tiled {
                vector: v,
                vector_rows,
                ..
            } if rows <= *vector_rows => (&v.pipeline, [v.column_groups, 1, 1], v.threads),
            DensePlan::Tiled { narrow, .. } => {
                // Keeping the matrix reduction; the smallest tile avoids discarded query rows.
                let tile = &narrow[0];
                (&tile.pipeline, [1, tile.column_groups, 1], 128)
            }
            DensePlan::Tensor {
                vector: v,
                vector_rows,
                ..
            } if rows <= *vector_rows => (&v.pipeline, [v.column_groups, 1, 1], v.threads),
            DensePlan::Tensor { tiles, .. } => {
                let tile = &tiles[0];
                (&tile.pipeline, [1, tile.column_groups, 1], 128)
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
        scratch: &CommonScratch,
        scales: (&BufferRef, usize),
        concurrent: bool,
    ) {
        let dummy = &scratch.next;
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
        let (narrow, wide, sum) = match &self.plan {
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
                sum,
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
                (narrow, wide, sum)
            }
            DensePlan::Tensor {
                vector,
                vector_rows,
                tiles,
                sum,
            } => {
                if m <= *vector_rows {
                    let groups = [vector.column_groups, m, 1];
                    first(e, &vector.pipeline, &buffers, bytes_of(&p), groups, vector.threads);
                    return;
                }
                self.apply_tensor(e, (tiles, sum.as_ref()), x, y, m, &p, (bias, scales), scratch, first);
                return;
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
            let splits = match sum {
                Some(_) if full == 0 => self.splits(m, scratch),
                _ => 1,
            };
            if let Some(sum) = sum.as_ref().filter(|_| splits > 1) {
                // Ordering both passes: the partials are shared by every product of the batch.
                let buffers = [
                    buffers[0],
                    buffers[1],
                    scratch.partials.binding(),
                    bias,
                    scales,
                ];
                let groups = [groups[0], groups[1], splits];
                dispatch(e, &tile.pipeline, &buffers, bytes_of(&p), groups, 128);
                let finish = SplitSum {
                    m: m as u32,
                    n: self.output as u32,
                    splits: splits as u32,
                    has_bias: p.has_bias,
                    scaled: scaled as u32,
                };
                let buffers = [scratch.partials.binding(), y.binding(), bias, scales];
                let groups = [(m * self.output).div_ceil(256), 1, 1];
                dispatch(e, sum, &buffers, bytes_of(&finish), groups, 256);
            } else if full > 0 {
                dispatch_concurrent(e, &tile.pipeline, &buffers, bytes_of(&p), groups, 128);
            } else {
                first(e, &tile.pipeline, &buffers, bytes_of(&p), groups, 128);
            }
        }
    }

    /// Whole 128-row tiles, then one independent tail dispatch with the smallest tile covering
    /// the remainder; batches of at most 64 rows take a single 32- or 64-row launch. Products
    /// with too few tiles to fill the GPU split K instead and finish their partial sums.
    #[allow(clippy::too_many_arguments)]
    fn apply_tensor(
        &self,
        e: &Commands,
        (tiles, sum): (&[Tile; 3], Option<&crate::gpu::Pipeline>),
        x: (&BufferRef, usize),
        y: &BufferView,
        m: usize,
        p: &Matrix,
        (bias, scales): ((&BufferRef, usize), (&BufferRef, usize)),
        scratch: &CommonScratch,
        first: fn(&Commands, &crate::gpu::Pipeline, &[(&BufferRef, usize)], &[u8], [usize; 3], usize),
    ) {
        let smallest = |rows: usize| &tiles[match rows {
            0..=32 => 0,
            33..=64 => 1,
            _ => 2,
        }];
        let single = smallest(m);
        let launched = m.div_ceil(single.rows) * single.column_groups;
        let splits = sum.map_or(1, |_| self.tensor_splits(launched, m, scratch));
        if let Some(sum) = sum.filter(|_| splits > 1) {
            // Ordering both passes: the partials are shared by every product of the batch.
            let buffers = [self.values.binding(), x, scratch.partials.binding(), bias, scales];
            let groups = [m.div_ceil(single.rows), single.column_groups, splits];
            dispatch(e, &single.pipeline, &buffers, bytes_of(p), groups, 128);
            let finish = SplitSum {
                m: m as u32,
                n: self.output as u32,
                splits: splits as u32,
                has_bias: p.has_bias,
                scaled: 0,
            };
            let buffers = [scratch.partials.binding(), y.binding(), bias, scales];
            let groups = [(m * self.output).div_ceil(256), 1, 1];
            dispatch(e, sum, &buffers, bytes_of(&finish), groups, 256);
            return;
        }
        let full = if m > 64 { m / 128 * 128 } else { 0 };
        let (x_buffer, x_offset) = x;
        let (y_buffer, y_offset) = y.binding();
        if full > 0 {
            let tile = &tiles[2];
            let p = Matrix { m: full as u32, ..*p };
            let buffers = [self.values.binding(), x, y.binding(), bias, scales];
            let groups = [full / 128, tile.column_groups, 1];
            first(e, &tile.pipeline, &buffers, bytes_of(&p), groups, 128);
        }
        let tail = m - full;
        if tail > 0 {
            let tile = smallest(tail);
            let p = Matrix { m: tail as u32, ..*p };
            let buffers = [
                self.values.binding(),
                (x_buffer, x_offset + full * self.input * 2),
                (y_buffer, y_offset + full * self.output * 2),
                bias,
                scales,
            ];
            let groups = [tail.div_ceil(tile.rows), tile.column_groups, 1];
            let launch = if full > 0 { dispatch_concurrent } else { first };
            launch(e, &tile.pipeline, &buffers, bytes_of(&p), groups, 128);
        }
    }

    /// Split K for tensor tiles while fewer than 128 threadgroups would launch: power-of-two
    /// ways up to sixteen, whole 64-column steps per split, partial sums within their scratch.
    fn tensor_splits(&self, launched: usize, rows: usize, scratch: &CommonScratch) -> usize {
        let steps = self.input / 64;
        let mut splits = 1;
        while launched * splits < 128 && splits < 16 {
            let next = splits * 2;
            if !steps.is_multiple_of(next)
                || next * rows * self.output * 4 > scratch.partials.length()
            {
                break;
            }
            splits = next;
        }
        splits
    }

    /// Split K across threadgroups when a short batch launches too few narrow tiles to keep
    /// loads in flight: up to eight ways while the tile count stays near a thousand, every
    /// split covers whole 32-column steps, and the partial sums fit their scratch.
    fn splits(&self, rows: usize, scratch: &CommonScratch) -> usize {
        let tiles = self.output.div_ceil(64) * rows.div_ceil(32);
        let mut splits = 8;
        while splits > 1
            && (tiles * splits > 1100
                || !(self.input / 32).is_multiple_of(splits)
                || splits * rows * self.output * 4 > scratch.partials.length())
        {
            splits /= 2;
        }
        splits
    }
}

/// Geometry for matmul_split_sum, which finishes K-split narrow tiles.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SplitSum {
    m: u32,
    n: u32,
    splits: u32,
    has_bias: u32,
    scaled: u32,
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
            s,
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
            s,
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
            s,
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
