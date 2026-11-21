//! Failures by whose they are to fix: the checkpoint's, the GPU's, the recording's, or the
//! request's. Every message is meant for a person; the kinds let a caller answer each kind
//! differently, such as falling back from a saved context it handed over.
use objc2::rc::Retained;
use objc2_foundation::NSError;
use std::fmt::Display;

/// Why oco could not do what was asked.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A file could not be read.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The checkpoint is malformed, or describes a model oco does not run.
    #[error("{0}")]
    Checkpoint(String),
    /// No Metal 4 GPU is present, or it could not hold or run the work.
    #[error("{0}")]
    Gpu(String),
    /// The recording is not audio oco reads.
    #[error("{0}")]
    Audio(String),
    /// The request does not fit the model, the session's capacity, or the session's state.
    #[error("{0}")]
    Request(String),
    /// The checkpoint's tokenizer could not encode or decode the text.
    #[error(transparent)]
    Tokenizer(#[from] tokenizers::Error),
}

/// Oco's results, failing with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Metal reports its failures as `NSError`s, which describe what the GPU could not do.
impl From<Retained<NSError>> for Error {
    fn from(error: Retained<NSError>) -> Self {
        Self::Gpu(error.localizedDescription().to_string())
    }
}

/// Name a missing value or a failed step as an error of the given kind, keeping the cause's
/// own message after the context the way chained reports print it.
pub(crate) trait Context<T> {
    fn context(self, kind: fn(String) -> Error, message: &str) -> Result<T>;
    fn with_context(self, kind: fn(String) -> Error, message: impl FnOnce() -> String)
    -> Result<T>;
}

impl<T> Context<T> for Option<T> {
    fn context(self, kind: fn(String) -> Error, message: &str) -> Result<T> {
        self.ok_or_else(|| kind(message.to_string()))
    }

    fn with_context(
        self,
        kind: fn(String) -> Error,
        message: impl FnOnce() -> String,
    ) -> Result<T> {
        self.ok_or_else(|| kind(message()))
    }
}

impl<T, E: Display> Context<T> for std::result::Result<T, E> {
    fn context(self, kind: fn(String) -> Error, message: &str) -> Result<T> {
        self.map_err(|error| kind(format!("{message}: {error}")))
    }

    fn with_context(
        self,
        kind: fn(String) -> Error,
        message: impl FnOnce() -> String,
    ) -> Result<T> {
        self.map_err(|error| kind(format!("{}: {error}", message())))
    }
}

/// Return an error of the named kind, `bail!(Error::Kind("…", …))`, its message formatted
/// like `format!`'s.
macro_rules! bail {
    (Error::$kind:ident($($message:tt)+)) => {
        return Err($crate::Error::$kind(format!($($message)+)))
    };
}

/// Return an error of the named kind unless the condition holds,
/// `ensure!(condition, Error::Kind("…", …))`.
macro_rules! ensure {
    ($condition:expr, Error::$kind:ident($($message:tt)+) $(,)?) => {
        if !$condition {
            $crate::error::bail!(Error::$kind($($message)+));
        }
    };
}

pub(crate) use {bail, ensure};
