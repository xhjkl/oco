//! Validated GPT-OSS operands consumed by binding.
pub(super) use crate::weights::load::selected::Layer;
use crate::{
    gguf::Tensor,
    weights::{common::AttentionProjections, load::selected::Projection},
};

pub(super) struct SinkAttention<'a> {
    pub projections: AttentionProjections<Projection<'a, Tensor<'a>>>,
    pub sinks: Tensor<'a>,
    pub window: usize,
}

pub(super) struct Experts<'a> {
    pub gate: Expert<'a>,
    pub up: Expert<'a>,
    pub down: Expert<'a>,
    pub router: Router<'a>,
}

/// MXFP4 expert coefficients and required per-expert F32 biases.
pub(super) struct Expert<'a> {
    pub values: Tensor<'a>,
    pub bias: Tensor<'a>,
}

/// F32 routing coefficients and biases, separate from the expert encoding.
pub(super) struct Router<'a> {
    pub values: Tensor<'a>,
    pub bias: Tensor<'a>,
}

pub(super) type GptLayer<'a> = Layer<'a, SinkAttention<'a>, Experts<'a>>;
