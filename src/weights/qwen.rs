//! Qwen dense feed-forward weights and its three supported history-mixer layouts.
pub mod attention;
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
