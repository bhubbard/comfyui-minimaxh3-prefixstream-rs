//! `minimaxh3_prefixstream` — Pure Rust port of `ComfyUI-MiniMaxH3-PrefixStream`.
//!
//! Provides an infinite video & audio continuation engine, 3D-RoPE temporal cursor alignment,
//! asymmetric cached attention, frequency-band texture drift diagnostics, DAG Clip Bin storage,
//! and non-destructive timeline slicing/reassembly for MiniMaxHailuo-03 (H3).

pub mod engine;
pub mod error;
pub mod nodes;
pub mod pipeline;
pub mod tensor;

pub use error::{PrefixStreamError, Result};
pub use tensor::{
    AudioData, AudioVae, AvLatent, ConditioningEntry, IdentityReference, MinimaxKeyframe, Tensor,
    VideoVae,
};
