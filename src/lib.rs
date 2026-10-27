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

/// Render and reset the per-kernel GPU timing accumulated since the last report.
/// Returns `None` unless the process was started with `OCO_PROFILE=1`.
pub fn profile_report() -> Option<String> {
    gpu::profile::report()
}

/// Checkpoint identity and storage requirements, without allocating GPU resources.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub architecture: String,
    pub name: String,
    pub parameters: u64,
    pub file_bytes: u64,
}

/// Inspect a complete GGUF's identity and tensor storage without loading it onto Metal.
/// Architecture-specific topology and tokenizer validation still happen in [`load`].
pub fn inspect(path: impl AsRef<Path>) -> Result<ModelInfo> {
    let file = Gguf::open(
        path.as_ref(),
        &["general.architecture", "general.name", "general.alignment"],
    )?;
    let architecture = file.text("general.architecture")?.to_owned();
    anyhow::ensure!(
        matches!(
            architecture.as_str(),
            "qwen35" | "gpt-oss" | "nemotron_h_moe"
        ),
        "unsupported architecture {architecture}"
    );
    let directory = file.directory()?;
    let mut parameters = 0u64;
    for entry in directory.values() {
        let tensor = entry.select()?;
        let count = tensor.shape.iter().try_fold(1u64, |count, &dimension| {
            count.checked_mul(dimension as u64)
        });
        let count = count.ok_or_else(|| anyhow::anyhow!("parameter count overflow"))?;
        parameters = parameters
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("parameter count overflow"))?;
    }
    anyhow::ensure!(parameters > 0, "checkpoint has no tensors");
    let name = file
        .text("general.name")
        .unwrap_or(&architecture)
        .to_owned();
    Ok(ModelInfo {
        architecture,
        name,
        parameters,
        file_bytes: file.map.len() as u64,
    })
}
