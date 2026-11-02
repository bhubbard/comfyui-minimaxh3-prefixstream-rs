//! Error types and `Result` alias for `minimaxh3_prefixstream`.

use thiserror::Error;

/// Unified error type for `minimaxh3_prefixstream`.
#[derive(Error, Debug)]
pub enum PrefixStreamError {
    /// Invalid input parameter or tensor geometry.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Alias for invalid value errors.
    #[error("Invalid value: {0}")]
    ValueError(String),

    /// Index or dimension out of bounds.
    #[error("Index out of range: {0}")]
    IndexError(String),

    /// Security violation (path traversal or symlink escape).
    #[error("Security violation: {0}")]
    SecurityViolation(String),

    /// Requested resource or clip not found.
    #[error("Not found: {0}")]
    NotFound(String),

    /// Internal runtime or invariant error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Runtime error.
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// File not found error.
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Standard I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    /// SafeTensors error string.
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),

    /// Image encoding/decoding error string.
    #[error("Image error: {0}")]
    Image(String),
}

impl From<safetensors::SafeTensorError> for PrefixStreamError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::SafeTensors(e.to_string())
    }
}

impl From<image::ImageError> for PrefixStreamError {
    fn from(e: image::ImageError) -> Self {
        Self::Image(e.to_string())
    }
}

/// Crate-wide `Result` alias.
pub type Result<T> = std::result::Result<T, PrefixStreamError>;
