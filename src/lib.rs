#![doc = include_str!("../README.md")]
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("Oco currently requires Apple Silicon, macOS, and Metal");
mod gguf;
mod gpu;
mod tensor;
mod tokenizer;
mod weights;

pub use tokenizer::{Decoder, TokenKind, Tokenizer};
pub use weights::{Inference, Session, Weights};

use anyhow::Result;
use gguf::Gguf;
use std::path::Path;

/// Load immutable GGUF weights and prepare shared Metal resources.
///
/// Create independent sequence state with [`Weights::new_session`]. The file is
/// memory-mapped and must remain unchanged while any weights, session, or submitted
/// GPU work uses it. Loading does not run inference.
pub fn load(path: impl AsRef<Path>) -> Result<Weights> {
    let file = Gguf::open(path.as_ref(), weights::METADATA)?;
    weights::load(&file)
}
