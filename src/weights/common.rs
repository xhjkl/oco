//! Immutable resources and layer structure shared by the concrete model implementations.
use super::{
    compute,
    operands::{Embedding, Output},
    pipelines::ElementwisePipelines,
};
use crate::gpu::{Gpu, ReadOnly};

/// Architecture-independent part of a model: execution device, token projections,
/// and elementary kernels. Owned inline alongside each concrete model's layers.
pub struct CommonWeights {
    pub gpu: Gpu,
    /// Shared GGUF registrations; synthetic in-memory models use command-local residency.
    pub residency: Option<std::sync::Arc<crate::gpu::ResidentWeights>>,
    pub context: usize,
    pub embedding: Embedding,
    pub output: Output,
    pub output_norm: ReadOnly,
    pub kernels: Kernels,
}

/// Elementary kernels and the normalization policy shared by every operation on one
/// residual stream; encoders own their own set alongside the language model's.
pub struct Kernels {
    pub pipelines: ElementwisePipelines,
    /// Stream width and RMS normalization parameters.
    pub norm: compute::Norm,
}

/// One refinement of each token's representation: mix sequence history, then transform features.
/// The mixer owns the learned weights for consulting history; its state lives in the session.
pub struct Layer<M, F> {
    pub norm: ReadOnly,
    pub post_norm: ReadOnly,
    pub mixer: M,
    pub ff: F,
}

/// Projections for matching the current token's query against stored keys and combining their values.
pub struct AttentionProjections<P> {
    pub q: P,
    pub k: P,
    pub v: P,
    pub out: P,
}
