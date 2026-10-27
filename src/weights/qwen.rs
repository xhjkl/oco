//! Qwen dense feed-forward weights and its three supported history-mixer layouts.
pub mod attention;
pub mod audio;
pub mod delta;
pub mod hybrid;
mod load;
use super::{common::Layer, operands::Projection};
pub use load::load;

/// Feed-forward transformation applied independently to each token after history mixing.
pub struct DenseFfn {
    pub gate: Projection,
    pub up: Projection,
    pub down: Projection,
}

pub type QwenLayer<M> = Layer<M, DenseFfn>;

/// Checkpoint conventions that differ between the Qwen generations oco runs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    /// Gated attention interleaved with gated delta layers; `post_attention_norm` feeds the FFN.
    Qwen35,
    /// Dense attention in every layer with packed queries and no gate; `ffn_norm` feeds the FFN.
    /// Covers the `qwen3vl` text stack as well: with one-dimensional positions its multi-axis
    /// rotary sections reduce to plain rotation.
    Qwen3,
}
