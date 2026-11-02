//! 3D-RoPE Coordinate Alignment & Temporal Cursor Manager for MiniMax H3.
//!
//! Rust port of `engine/rope_aligner.py`.

use serde::{Deserialize, Serialize};

use crate::engine::cache_manager::FRAME_PER_TOKEN;

/// Return per-step pixel frame spans `[1, 4, 4, 4, 4, ...]` for `n_steps` starting at `start_step`.
///
/// Mirrors `video_t_spans` in `engine/rope_aligner.py`.
pub fn video_t_spans(n_steps: usize, start_step: usize) -> Vec<usize> {
    (0..n_steps)
        .map(|k| FRAME_PER_TOKEN[(start_step + k) % 5])
        .collect()
}

/// Return cumulative pixel-frame temporal coordinates for `n_steps` starting at `start_step`.
///
/// Mirrors `video_t_grid` in `engine/rope_aligner.py`.
pub fn video_t_grid(n_steps: usize, start_step: usize) -> Vec<f32> {
    let mut grid = Vec::with_capacity(n_steps);
    let mut acc = 0usize;
    for k in 0..n_steps {
        grid.push(acc as f32);
        acc += FRAME_PER_TOKEN[(start_step + k) % 5];
    }
    grid
}

/// Total pixel frames spanned by `n_steps` starting at `start_step`.
///
/// Mirrors `total_span_for_steps` in `engine/rope_aligner.py`.
pub fn total_span_for_steps(n_steps: usize, start_step: usize) -> usize {
    video_t_spans(n_steps, start_step).into_iter().sum()
}

/// Metadata recorded per generated clip in [`TemporalCursorTracker`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipCursorInfo {
    /// Clip index (`0`-based).
    pub clip_index: usize,
    /// Global latent start step.
    pub start_step: usize,
    /// Next global latent step (`start_step + net_steps`).
    pub next_step: usize,
    /// Pixel frames spanned by this clip's net steps.
    pub span_frames: usize,
    /// Cumulative pixel frames across the session.
    pub cumulative_frames: usize,
}

/// Summary snapshot of [`TemporalCursorTracker`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CursorSummary {
    /// Current global latent step.
    pub current_global_step: usize,
    /// Total generated pixel frames.
    pub total_generated_frames: usize,
    /// Total clips generated.
    pub total_clips_generated: usize,
}

/// Tracks global video timeline positions across continuous clips for 3D-RoPE.
///
/// Mirrors `TemporalCursorTracker` in `engine/rope_aligner.py`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TemporalCursorTracker {
    /// Current global latent step cursor.
    pub current_global_step: usize,
    /// Total cumulative pixel frames generated.
    pub total_generated_frames: usize,
    /// History of clips registered in this session.
    pub history: Vec<ClipCursorInfo>,
}

impl TemporalCursorTracker {
    /// Create a new [`TemporalCursorTracker`] at step 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the cursor by `latent_steps` (optionally subtracting `overlap_steps` already accounted for).
    pub fn advance_clip(
        &mut self,
        latent_steps: usize,
        overlap_steps: Option<usize>,
    ) -> ClipCursorInfo {
        let ov = overlap_steps.unwrap_or(0).min(latent_steps);
        let net_steps = latent_steps - ov;
        let start_step = self.current_global_step;
        let span_frames = total_span_for_steps(net_steps, start_step);
        let next_step = start_step + net_steps;
        self.current_global_step = next_step;
        self.total_generated_frames += span_frames;

        let info = ClipCursorInfo {
            clip_index: self.history.len(),
            start_step,
            next_step,
            span_frames,
            cumulative_frames: self.total_generated_frames,
        };
        self.history.push(info.clone());
        info
    }

    /// Return a summary of the current cursor state.
    pub fn get_summary(&self) -> CursorSummary {
        CursorSummary {
            current_global_step: self.current_global_step,
            total_generated_frames: self.total_generated_frames,
            total_clips_generated: self.history.len(),
        }
    }

    /// Reset cursor back to origin.
    pub fn reset(&mut self) {
        self.current_global_step = 0;
        self.total_generated_frames = 0;
        self.history.clear();
    }
}
