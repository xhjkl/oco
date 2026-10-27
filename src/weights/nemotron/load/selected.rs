//! Validated Nemotron operands consumed by binding.
use crate::{
    gguf::Tensor,
    weights::{common::AttentionProjections, load::selected::Projection},
};

pub(super) type Dense<'a> = Projection<'a, Option<Tensor<'a>>>;

pub(super) struct Mamba<'a> {
    pub in_proj: Dense<'a>,
    pub out: Dense<'a>,
    pub conv: Tensor<'a>,
    pub conv_bias: Tensor<'a>,
    pub a: Tensor<'a>,
    pub d: Tensor<'a>,
    pub dt_bias: Tensor<'a>,
    pub norm: Tensor<'a>,
}

pub(super) struct Experts<'a> {
    pub router: Tensor<'a>,
    pub router_bias: Tensor<'a>,
    pub up: Tensor<'a>,
    pub down: Tensor<'a>,
    pub shared_up: Dense<'a>,
    pub shared_down: Dense<'a>,
}

pub(super) enum Mixer<'a> {
    Mamba { weights: Mamba<'a>, slot: usize },
    Attention {
        weights: AttentionProjections<Dense<'a>>,
        slot: usize,
    },
    Experts(Experts<'a>),
}

pub(super) struct Layer<'a> {
    pub norm: Tensor<'a>,
    pub mixer: Mixer<'a>,
}
