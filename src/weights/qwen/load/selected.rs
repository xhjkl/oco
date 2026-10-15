//! Validated Qwen operands and topology consumed by binding.
pub(super) use crate::weights::load::selected::Layer;
use crate::{
    gguf::Tensor,
    weights::{
        common::AttentionProjections,
        load::selected::Projection,
        qwen::hybrid::{AttentionSlot, DeltaSlot},
    },
};

pub(super) struct GatedAttention<'a> {
    pub projections: AttentionProjections<Projection<'a, Option<Tensor<'a>>>>,
    pub qnorm: Tensor<'a>,
    pub knorm: Tensor<'a>,
}

pub(super) struct DenseFfn<'a> {
    pub gate: Projection<'a, Option<Tensor<'a>>>,
    pub up: Projection<'a, Option<Tensor<'a>>>,
    pub down: Projection<'a, Option<Tensor<'a>>>,
}

pub(super) struct Delta<'a> {
    pub qkv: Projection<'a, Option<Tensor<'a>>>,
    pub z: Projection<'a, Option<Tensor<'a>>>,
    pub alpha: Projection<'a, Option<Tensor<'a>>>,
    pub beta: Projection<'a, Option<Tensor<'a>>>,
    pub out: Projection<'a, Option<Tensor<'a>>>,
    pub conv: Tensor<'a>,
    pub a: Tensor<'a>,
    pub dt: Tensor<'a>,
    pub norm: Tensor<'a>,
}

pub(super) type QwenLayer<'a, M> = Layer<'a, M, DenseFfn<'a>>;

/// Selected topology, including the state slots that survive into the bound model.
pub(super) enum Qwen<'a> {
    Attention(Vec<QwenLayer<'a, GatedAttention<'a>>>),
    Delta(Vec<QwenLayer<'a, Delta<'a>>>),
    Hybrid {
        layers: Vec<QwenLayer<'a, Mixer<'a>>>,
        attention_layers: usize,
        delta_layers: usize,
    },
}

pub(super) enum Mixer<'a> {
    Attention {
        weights: GatedAttention<'a>,
        slot: AttentionSlot,
    },
    Delta {
        weights: Delta<'a>,
        slot: DeltaSlot,
    },
}
