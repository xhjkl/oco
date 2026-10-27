// Dispatching the same operation across concrete models or their paired session memory;
// each arm keeps its concrete types and remains exhaustively checked.
macro_rules! with_architecture {
    ($value:expr, $enum:ident $bindings:tt => $body:expr) => {
        match $value {
            $enum::GptOss $bindings => $body,
            $enum::QwenAttention $bindings => $body,
            $enum::QwenDelta $bindings => $body,
            $enum::QwenHybrid $bindings => $body,
            $enum::Nemotron $bindings => $body,
        }
    };
}

pub(crate) mod common;
pub(crate) mod compute;
#[cfg(test)]
mod dispatch_tests;
pub(crate) mod generation;
mod gpt_oss;
mod load;
mod nemotron;
pub(crate) mod operands;
pub(crate) mod pipelines;
mod qwen;
pub(crate) mod session;
pub(crate) mod state;

use crate::Tokenizer;
use anyhow::Result;
use common::CommonWeights;
pub use generation::Inference;
pub(crate) use load::{METADATA, load};
pub(crate) use qwen::audio::METADATA as AUDIO_METADATA;
pub use session::Session;
use std::sync::Arc;

/// Shared model weights, tokenizer, and prepared GPU resources.
/// The mapped GGUF must remain unchanged while weights, sessions, or submitted GPU work use it.
#[derive(Clone)]
pub struct Weights {
    inner: Architecture,
    tokenizer: Arc<Tokenizer>,
}

/// Concrete execution layout, paired with the weights and kernels its sessions need.
#[derive(Clone)]
enum Architecture {
    GptOss(Arc<gpt_oss::GptOss>),
    QwenAttention(Arc<qwen::attention::QwenAttention>),
    QwenDelta(Arc<qwen::delta::QwenDelta>),
    QwenHybrid(Arc<qwen::hybrid::QwenHybrid>),
    Nemotron(Arc<nemotron::Nemotron>),
}

impl Architecture {
    fn base(&self) -> &CommonWeights {
        with_architecture!(self, Architecture(weights) => &weights.base)
    }
}

impl Weights {
    /// Return the number of token IDs the model accepts and can predict.
    /// Valid IDs are `0..vocabulary_size()`.
    pub fn vocabulary_size(&self) -> usize {
        self.inner.base().embedding.vocabulary
    }

    /// Return the checkpoint's text codec and special-token metadata.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Return the model's context limit declared in its GGUF metadata.
    pub fn context_limit(&self) -> usize {
        self.inner.base().context
    }

    /// Allocate an empty sequence with room for `capacity` consumed tokens.
    ///
    /// Capacity is fixed and must be positive and within the model's context limit.
    /// Attention cache storage scales with capacity; recurrent state has fixed size.
    /// Prompt scratch holds up to 512 rows, with longer inputs processed in chunks.
    /// Each session owns its mutable buffers and command queue and shares the weights.
    pub fn new_session(&self, capacity: usize) -> Result<Session> {
        Session::new(&self.inner, capacity)
    }
}
