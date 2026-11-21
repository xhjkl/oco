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
use std::{path::Path, sync::Arc};

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

    /// Session bytes each position of capacity costs: attention keys and values, plus audio
    /// feature rows on speech models. Everything else a session holds has a fixed size.
    fn position_bytes(&self) -> usize {
        use state::kv_position_bytes;
        match self {
            Self::GptOss(g) => kv_position_bytes(g.layers.len(), &g.attention),
            Self::QwenAttention(q) => {
                let audio = q.audio.as_ref().map_or(0, |encoder| encoder.output * 2);
                kv_position_bytes(q.layers.len(), &q.attention) + audio
            }
            Self::QwenDelta(_) => 0,
            Self::QwenHybrid(q) => kv_position_bytes(q.attention_layers, &q.attention),
            Self::Nemotron(n) => kv_position_bytes(n.attention_layers, &n.attention),
        }
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

    /// Return the largest capacity, within the context limit, at which the weights and one
    /// session's capacity-bound buffers fit in `share` of the GPU's working set. Deliberately
    /// rough: fixed-size scratch and recurrent memory come out of the rest of the working set,
    /// and other sessions are not counted. Zero means the weights alone exceed the share.
    pub fn capacity_within(&self, share: f64) -> usize {
        let base = self.inner.base();
        let budget = (base.gpu.working_set() as f64 * share) as usize;
        let free = budget.saturating_sub(base.gpu.weight_bytes());
        let capacity = free.checked_div(self.inner.position_bytes());
        capacity.map_or(base.context, |capacity| capacity.min(base.context))
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

    /// Open a session holding a context written by [`Session::save`], checkpointed at its end.
    /// Restoring copies the saved bytes straight into the session's buffers, far cheaper than
    /// prefilling them again. The file is trusted to come from these weights and this build:
    /// naming files so that holds is the caller's job. Fails, so the caller can fall back to
    /// [`Self::new_session`] and prefill, when the file is missing, truncated, or holds more
    /// history than `capacity`.
    pub fn open_session(&self, path: impl AsRef<Path>, capacity: usize) -> Result<Session> {
        Session::open(&self.inner, path.as_ref(), capacity)
    }
}
