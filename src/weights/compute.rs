//! Session-owned submission and transformer computation.
//! `Compute` owns the queue, reusable command storage, and completion handling.
//! `Bindings` borrows the encoding context, weights, and scratch used by transformer operations. Architecture loops supply the
//! positions and row counts explicitly; the bindings carry no sequence progress.

mod attention;
mod feed_forward;
mod matrix;
mod nemotron;
mod recurrent;
mod submission;
#[cfg(test)]
pub(crate) mod tests;

use super::{common::CommonWeights, state::CommonScratch};
use crate::gpu::{BufferRef, CommandMemory, Commands};
use crate::gpu::{BufferView, Gpu, ReadOnly, dispatch};
pub use attention::{AttentionPlan, Rope};
use bytemuck::{Pod, Zeroable, bytes_of};
pub use feed_forward::ExpertPlan;
pub use nemotron::{MambaPlan, ReluExpertPlan, plan_words};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTL4CommandQueue, MTLDevice};
pub use recurrent::DeltaPlan;
use std::sync::{Arc, Mutex};

/// Command submission owned by one session, independent of every other session's queue.
/// Immutable kernels remain prebound to the model's operands; no runtime lookup or compilation occurs.
pub struct Compute {
    pub(super) queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    idle: Arc<Mutex<Option<CommandMemory>>>,
    weights: Option<Arc<crate::gpu::ResidentWeights>>,
}

impl Compute {
    pub fn new(gpu: &Gpu, weights: Option<Arc<crate::gpu::ResidentWeights>>) -> Self {
        let queue = gpu
            .device
            .newMTL4CommandQueue()
            .expect("Metal command queue creation failed");
        Self {
            queue,
            idle: Arc::new(Mutex::new(None)),
            weights,
        }
    }
}

/// Borrowed resource bindings shared by operations during one submission.
/// Row selection and sequence positions belong to the caller, not these bindings.
pub struct Bindings<'a> {
    pub base: &'a CommonWeights,
    pub scratch: &'a CommonScratch,
    pub encoder: &'a Commands,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Norm {
    pub n: u32,
    pub epsilon: f32,
    /// Wire policy: 1 scales RMS-normalized FP32 values before BF16 rounding (GPT);
    /// 0 rounds the normalized value and scale to BF16 before multiplying (Qwen).
    pub full_layer: u32,
}

impl Bindings<'_> {
    /// Rescale token features with the model's learned normalization before the next transformation.
    pub fn norm(&self, w: &ReadOnly, x: (&BufferRef, usize), y: &BufferView, m: usize) {
        dispatch(
            self.encoder,
            &self.base.pipelines.norm,
            &[x, w.binding(), y.binding()],
            bytes_of(&self.base.norm),
            [m, 1, 1],
            256,
        );
    }

    /// Add a layer's contribution to the residual rows and normalize the result for the next
    /// transformation in one pass.
    pub fn add_norm(
        &self,
        w: &ReadOnly,
        residual: &BufferView,
        hidden: (&BufferRef, usize),
        y: &BufferView,
        rows: usize,
    ) {
        dispatch(
            self.encoder,
            &self.base.pipelines.add_norm,
            &[hidden, residual.binding(), w.binding(), y.binding()],
            bytes_of(&self.base.norm),
            [rows, 1, 1],
            256,
        );
    }

    /// Add a layer's contribution to the running features, preserving the residual path through layers.
    pub fn add(&self, residual: &BufferView, hidden: (&BufferRef, usize), rows: usize) {
        let n = (rows * self.base.embedding.width) as u32;
        dispatch(
            self.encoder,
            &self.base.pipelines.add,
            &[hidden, residual.binding()],
            bytes_of(&n),
            [(n as usize).div_ceil(256), 1, 1],
            256,
        );
    }

    /// Select the largest vocabulary score on the GPU for greedy decoding and direct token feedback.
    pub fn argmax(&self) {
        dispatch(
            self.encoder,
            &self.base.pipelines.argmax,
            &[self.scratch.logits.at(0), self.scratch.next.at(0)],
            bytes_of(&(self.base.output.vocabulary as u32)),
            [1, 1, 1],
            1024,
        );
    }
}
