//! Long Video Director & Streaming Chaining Pipeline for MiniMax H3.
//!
//! Coordinates continuation sessions, keyframe fallback, and prefix trimming.
//!
//! Rust port of `pipeline/long_video_director.py`.

use serde::{Deserialize, Serialize};

use crate::engine::cache_manager::{
    latent_steps_to_pixel_frames, pixel_frames_to_latent_steps, snap_to_run_grid, KVCacheConfig,
    PrefixKVCacheManager, FRAME_PER_TOKEN,
};
use crate::engine::rope_aligner::TemporalCursorTracker;
use crate::error::Result;
use crate::pipeline::seam_protector::trim_prefix_frames;
use crate::tensor::{ConditioningEntry, MinimaxKeyframe, Tensor};

/// Pixel-frame index at which each latent step begins (`len == latent_t`).
///
/// Mirrors `step_offsets` in `pipeline/long_video_director.py`.
pub fn step_offsets(latent_t: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(latent_t);
    let mut acc = 0usize;
    for k in 0..latent_t {
        out.push(acc);
        acc += FRAME_PER_TOKEN[k % 5];
    }
    out
}

/// Inject keyframe anchors into a MiniMax H3 conditioning payload so the model binds context.
///
/// Drops any prior keyframe residing within `[0..head_end)` when `video_tail` or `anchor_video`
/// is injected to prevent conflicting double-anchor conditioning at frame 0.
///
/// Mirrors `inject_minimax_keyframes` in `pipeline/long_video_director.py`.
pub fn inject_minimax_keyframes(
    conditioning: &[ConditioningEntry],
    video_tail: Option<&Tensor>,
    audio_tail: Option<&Tensor>,
    anchor_video: Option<&Tensor>,
) -> Result<Vec<ConditioningEntry>> {
    if conditioning.is_empty() {
        return Ok(Vec::new());
    }

    let mut keyframes = Vec::new();
    let mut head_end = 0usize;

    if let Some(v_tail) = video_tail {
        let tail_t = v_tail.shape[2];
        let tail_offsets = step_offsets(tail_t);
        head_end = latent_steps_to_pixel_frames(tail_t);
        for (k, &offset) in tail_offsets.iter().enumerate() {
            let slice = v_tail.slice_axis(2, k, k + 1)?;
            keyframes.push(MinimaxKeyframe {
                resolved_frame_index: offset as f32,
                latent: Some(slice),
                audio_latent: None,
            });
        }

        if let Some(a_tail) = audio_tail {
            let rt = a_tail.shape[a_tail.ndim() - 1];
            let audio_start_frame = (head_end as f32) - ((rt as f32) / (5.0 / 3.0));
            keyframes.push(MinimaxKeyframe {
                resolved_frame_index: audio_start_frame,
                latent: None,
                audio_latent: Some(a_tail.clone()),
            });
        }
    } else if let Some(anc_v) = anchor_video {
        let anc_t = anc_v.shape[2];
        let anc_offsets = step_offsets(anc_t);
        head_end = latent_steps_to_pixel_frames(anc_t);
        for (k, &offset) in anc_offsets.iter().enumerate() {
            let slice = anc_v.slice_axis(2, k, k + 1)?;
            keyframes.push(MinimaxKeyframe {
                resolved_frame_index: offset as f32,
                latent: Some(slice),
                audio_latent: None,
            });
        }
    }

    if keyframes.is_empty() {
        return Ok(conditioning.to_vec());
    }

    let mut out_cond = Vec::with_capacity(conditioning.len());
    for entry in conditioning {
        let mut kept: Vec<MinimaxKeyframe> = entry
            .minimax_keyframes
            .iter()
            .filter(|kf| kf.resolved_frame_index >= (head_end as f32))
            .cloned()
            .collect();
        kept.extend(keyframes.iter().cloned());
        out_cond.push(ConditioningEntry {
            embedding: entry.embedding.clone(),
            minimax_keyframes: kept,
            extra: entry.extra.clone(),
        });
    }

    Ok(out_cond)
}

/// Manages state across infinite clips in an H3 video generation stream.
///
/// Mirrors `LongVideoSession` in `pipeline/long_video_director.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongVideoSession {
    /// Prefix KV cache configuration.
    pub config: KVCacheConfig,
    /// Prefix KV cache manager.
    pub cache_manager: PrefixKVCacheManager,
    /// Monotonic 3D-RoPE temporal cursor tracker.
    pub cursor_tracker: TemporalCursorTracker,
    /// Current clip index (`0`-based).
    pub current_clip_index: usize,
    /// Rolling latent steps applied to the current clip.
    pub last_rolling_steps: usize,
    /// Rolling pixel frames applied to the current clip.
    pub last_rolling_frames: usize,
    /// Whether the current clip's prefix has already been trimmed.
    pub is_current_clip_trimmed: bool,
    /// Accumulated video latents across committed clips.
    pub accumulated_video_latents: Vec<Tensor>,
    /// Accumulated audio latents across committed clips.
    pub accumulated_audio_latents: Vec<Tensor>,
}

impl Default for LongVideoSession {
    fn default() -> Self {
        Self::new(KVCacheConfig::default())
    }
}

impl LongVideoSession {
    /// Create a new [`LongVideoSession`] with the given [`KVCacheConfig`].
    pub fn new(config: KVCacheConfig) -> Self {
        let cache_manager = PrefixKVCacheManager::new(config.clone());
        Self {
            config,
            cache_manager,
            cursor_tracker: TemporalCursorTracker::new(),
            current_clip_index: 0,
            last_rolling_steps: 0,
            last_rolling_frames: 0,
            is_current_clip_trimmed: false,
            accumulated_video_latents: Vec::new(),
            accumulated_audio_latents: Vec::new(),
        }
    }

    /// Prepare conditioning and session state for the upcoming clip generation.
    ///
    /// 1. Slices grid-aligned rolling context from `previous_video_latent`.
    /// 2. Ensures `(source_start_step + start) % 5 == 0` phase alignment.
    /// 3. Injects keyframe anchors into conditioning while purging colliding prior anchors.
    pub fn prepare_next_clip(
        &mut self,
        conditioning: &[ConditioningEntry],
        previous_video_latent: Option<&Tensor>,
        previous_audio_latent: Option<&Tensor>,
        anchor_video_latent: Option<&Tensor>,
        source_start_step: usize,
    ) -> Result<Vec<ConditioningEntry>> {
        self.cache_manager.reset_for_next_clip();

        let mut tail_video: Option<Tensor> = None;
        let mut tail_audio: Option<Tensor> = None;
        self.last_rolling_steps = 0;
        self.last_rolling_frames = 0;
        self.is_current_clip_trimmed = false;

        if let Some(prev_v) = previous_video_latent {
            let snapped_frames = snap_to_run_grid(self.config.rolling_frames, "nearest");
            let total_steps = prev_v.shape[2];
            let mut rolling_steps = pixel_frames_to_latent_steps(snapped_frames).min(total_steps);

            let mut start = total_steps - rolling_steps;
            let phase = (source_start_step + start) % 5;
            if phase != 0 {
                let shift = (5 - phase) % 5;
                if rolling_steps > shift {
                    rolling_steps -= shift;
                    start = total_steps - rolling_steps;
                }
            }

            self.last_rolling_steps = rolling_steps;
            self.last_rolling_frames = latent_steps_to_pixel_frames(rolling_steps);
            tail_video = Some(prev_v.slice_axis(2, start, start + rolling_steps)?);

            if let Some(prev_a) = previous_audio_latent {
                let audio_steps =
                    ((self.last_rolling_frames as f64) / 24.0 * 40.0).round_ties_even() as usize;
                let last_dim = prev_a.ndim() - 1;
                let total_a = prev_a.shape[last_dim];
                let a_steps = audio_steps.min(total_a);
                tail_audio = Some(prev_a.slice_axis(last_dim, total_a - a_steps, total_a)?);
            }
        }

        let active_anchor = if self.config.use_anchor {
            anchor_video_latent
        } else {
            None
        };

        inject_minimax_keyframes(
            conditioning,
            tail_video.as_ref(),
            tail_audio.as_ref(),
            active_anchor,
        )
    }

    /// Register a completed clip into session history and advance `current_clip_index`.
    pub fn commit_generated_clip(
        &mut self,
        video_latent: &Tensor,
        audio_latent: Option<&Tensor>,
        trim_prefix: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let rolling_steps = self.last_rolling_steps;
        let delivered_video = if trim_prefix {
            trim_prefix_frames(video_latent, rolling_steps)?
        } else {
            video_latent.clone()
        };
        self.accumulated_video_latents.push(delivered_video.clone());

        let delivered_audio = if let Some(aud) = audio_latent {
            let frames = latent_steps_to_pixel_frames(rolling_steps);
            let audio_steps = if trim_prefix {
                ((frames as f64) * 40.0 / 24.0).round_ties_even() as usize
            } else {
                0
            };
            let last_dim = aud.ndim() - 1;
            let total_a = aud.shape[last_dim];
            let trimmed_a = if audio_steps > 0 {
                aud.slice_axis(last_dim, audio_steps.min(total_a), total_a)?
            } else {
                aud.clone()
            };
            self.accumulated_audio_latents.push(trimmed_a.clone());
            Some(trimmed_a)
        } else {
            None
        };

        self.current_clip_index += 1;
        Ok((delivered_video, delivered_audio))
    }

    /// Clear all caches and restart the session from scratch.
    pub fn reset(&mut self) {
        self.cache_manager.clear_all();
        self.cursor_tracker.reset();
        self.current_clip_index = 0;
        self.last_rolling_steps = 0;
        self.last_rolling_frames = 0;
        self.is_current_clip_trimmed = false;
        self.accumulated_video_latents.clear();
        self.accumulated_audio_latents.clear();
    }
}
