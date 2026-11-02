//! Native MiniMax H3 video/audio masked-continuation helpers.
//!
//! Copied video context follows H3's `17k + 5` frame grid. Audio context is rounded
//! to the 40 Hz tick grid, as in the native LatentCut/LatentConcat workflow.
//!
//! Rust port of `pipeline/native_masked_av.py`.

use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::engine::cache_manager::FRAME_PER_TOKEN;
use crate::error::{PrefixStreamError, Result};
use crate::tensor::Tensor;

/// Canonical MiniMax H3 video frame rate.
pub const FPS: usize = 24;
/// Canonical MiniMax H3 audio latent tick rate (Hz).
pub const AUDIO_HZ: usize = 40;
/// Canonical `17k + 5` context lengths supported by Native Masked AV (`5..=192`).
pub const MASKED_AV_CONTEXTS: [usize; 12] = [5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192];

/// Round a floating-point value using Python's round-half-to-even (`round()`) semantics.
fn py_round(val: f64) -> isize {
    val.round_ties_even() as isize
}

/// Return exclusive pixel-frame boundaries for an H3 video latent (`len == video_latent_t + 1`).
///
/// Mirrors `latent_boundaries` in `pipeline/native_masked_av.py`.
pub fn latent_boundaries(video_latent_t: usize, start_step: usize) -> Vec<usize> {
    let mut boundaries = Vec::with_capacity(video_latent_t + 1);
    boundaries.push(0);
    for step in 0..video_latent_t {
        let prev = *boundaries.last().unwrap();
        boundaries.push(prev + FRAME_PER_TOKEN[(start_step + step) % 5]);
    }
    boundaries
}

/// Whether a canonical video run also has an integer audio duration (`(frame_count - 39) % 51 == 0`, e.g. 39, 90, 141, 192).
///
/// Mirrors `is_exact_masked_av_context` in `pipeline/native_masked_av.py`.
pub fn is_exact_masked_av_context(frame_count: usize) -> bool {
    frame_count >= 39 && (frame_count - 39).is_multiple_of(51)
}

/// Snap down to a canonical `17k + 5` video context that leaves new content (`< target_frames`).
///
/// Mirrors `snap_masked_av_context_length` in `pipeline/native_masked_av.py`.
pub fn snap_masked_av_context_length(
    requested: usize,
    available: usize,
    target_frames: usize,
) -> Result<usize> {
    let cap = requested
        .min(available)
        .min(target_frames.saturating_sub(1));
    if cap < 5 {
        return Err(PrefixStreamError::InvalidInput(
            "Native Masked AV needs at least 5 source frames and a target longer than 5 frames"
                .to_string(),
        ));
    }
    Ok(5 + ((cap - 5) / 17) * 17)
}

/// Phase-aligned tail slice plan computed by [`masked_av_tail_slice`] and [`apply_native_masked_av`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskedAvPlan {
    /// Start latent step in source video (`(source_start_step + start_t) % 5 == 0`).
    pub start_t: usize,
    /// Exclusive end latent step in source video (`(source_start_step + end_t) % 5 == 2`).
    pub end_t: usize,
    /// Number of latent steps copied (`end_t - start_t`).
    pub context_steps: usize,
    /// Number of pixel frames represented by `context_steps` (`17k + 5`).
    pub actual_context_frames: usize,
    /// Start pixel frame in source clip.
    pub source_start_frame: usize,
    /// Exclusive end pixel frame in source clip.
    pub source_end_frame: usize,
    /// Total pixel frames in source clip.
    pub previous_frame_count: usize,
    /// Number of trailing frames after `source_end_frame` skipped to preserve 5-step phase alignment.
    pub ignored_tail_frames: usize,
    /// Start audio tick in source audio latent.
    pub audio_start_tick: usize,
    /// Exclusive end audio tick in source audio latent.
    pub audio_end_tick: usize,
    /// Number of audio ticks copied (`audio_end_tick - audio_start_tick`).
    pub audio_steps: usize,
    /// Audio tail carryover mode (`"Match Video Handover"` or `"Full Previous Tail"`).
    pub audio_tail_carryover: String,
    /// Number of half-cosine feathered audio ticks at the end of the protected audio prefix.
    pub audio_feather_ticks: usize,
    /// Denoise mask value applied to the video context (`0.0` = hard lock).
    pub video_context_denoise: f32,
}

/// Select the latest canonical source tail for a target masked prefix.
///
/// Mirrors `masked_av_tail_slice` in `pipeline/native_masked_av.py`.
pub fn masked_av_tail_slice(
    video_latent_t: usize,
    context_frames: usize,
    target_video_t: usize,
    source_start_step: usize,
) -> Result<MaskedAvPlan> {
    let source_boundaries = latent_boundaries(video_latent_t, source_start_step);
    let target_frames = *latent_boundaries(target_video_t, 0).last().unwrap_or(&0);
    let previous_frames = *source_boundaries.last().unwrap_or(&0);
    let frames = snap_masked_av_context_length(context_frames, previous_frames, target_frames)?;
    let context_steps = 2 + 5 * ((frames - 5) / 17);

    if video_latent_t < context_steps {
        return Err(PrefixStreamError::InvalidInput(
            "Source latent has no phase-aligned Native Masked AV tail".to_string(),
        ));
    }

    let mut end_t_opt = None;
    for candidate in (context_steps..=video_latent_t).rev() {
        let start_t = candidate - context_steps;
        if (source_start_step + candidate) % 5 == 2
            && (source_start_step + start_t).is_multiple_of(5)
        {
            end_t_opt = Some(candidate);
            break;
        }
    }

    let end_t = end_t_opt.ok_or_else(|| {
        PrefixStreamError::InvalidInput(
            "Source latent has no phase-aligned Native Masked AV tail".to_string(),
        )
    })?;

    let start_t = end_t - context_steps;
    let source_start_frame = source_boundaries[start_t];
    let source_end_frame = source_boundaries[end_t];
    if source_end_frame - source_start_frame != frames {
        return Err(PrefixStreamError::Internal(
            "Internal Native Masked AV source geometry mismatch".to_string(),
        ));
    }

    Ok(MaskedAvPlan {
        start_t,
        end_t,
        context_steps,
        actual_context_frames: frames,
        source_start_frame,
        source_end_frame,
        previous_frame_count: previous_frames,
        ignored_tail_frames: previous_frames - source_end_frame,
        audio_start_tick: 0,
        audio_end_tick: 0,
        audio_steps: 0,
        audio_tail_carryover: String::new(),
        audio_feather_ticks: 0,
        video_context_denoise: 0.0,
    })
}

/// Output bundle returned by [`apply_native_masked_av`].
#[derive(Debug, Clone)]
pub struct NativeMaskedAvOutput {
    /// Target video latent with copied prefix in `[:, :, :video_steps]`.
    pub video_latent: Tensor,
    /// Target audio latent with copied prefix in `[..., :audio_steps]`.
    pub audio_latent: Tensor,
    /// Video noise mask `[1, 1, T, H, W]` (`video_context_denoise` in prefix, `1.0` elsewhere).
    pub video_mask: Tensor,
    /// Audio noise mask `[1, 1, C_a, T_a]` (`0.0` + optional half-cosine feather in prefix, `1.0` elsewhere).
    pub audio_mask: Tensor,
    /// Detailed continuation slice plan.
    pub plan: MaskedAvPlan,
}

/// Copy a previous AV tail into a target head and build native denoise masks.
///
/// Mirrors `apply_native_masked_av` in `pipeline/native_masked_av.py`.
#[allow(clippy::too_many_arguments)]
pub fn apply_native_masked_av(
    target_video: &Tensor,
    target_audio: &Tensor,
    source_video: &Tensor,
    source_audio: &Tensor,
    context_frames: usize,
    audio_tail_carryover: &str,
    audio_feather_ticks: usize,
    source_start_step: usize,
    source_audio_start_tick: usize,
    video_context_denoise: f32,
) -> Result<NativeMaskedAvOutput> {
    if target_video.ndim() != 5 || source_video.ndim() != 5 {
        return Err(PrefixStreamError::InvalidInput(
            "Native Masked AV expects 5D video latents".to_string(),
        ));
    }
    if target_audio.ndim() != 4 || source_audio.ndim() != 4 {
        return Err(PrefixStreamError::InvalidInput(
            "Native Masked AV expects 4D audio latents".to_string(),
        ));
    }
    if target_video.shape[0] != 1 || source_video.shape[0] != 1 {
        return Err(PrefixStreamError::InvalidInput(
            "Native Masked AV currently supports batch size 1".to_string(),
        ));
    }
    if target_audio.shape[0] != 1 || source_audio.shape[0] != 1 {
        return Err(PrefixStreamError::InvalidInput(
            "Native Masked AV currently supports audio batch size 1".to_string(),
        ));
    }
    if source_video.shape[1] != target_video.shape[1]
        || source_video.shape[3] != target_video.shape[3]
        || source_video.shape[4] != target_video.shape[4]
    {
        return Err(PrefixStreamError::InvalidInput(
            "Native Masked AV requires matching source/target video latent geometry".to_string(),
        ));
    }
    if source_audio.shape[1] != target_audio.shape[1]
        || source_audio.shape[2] != target_audio.shape[2]
    {
        return Err(PrefixStreamError::InvalidInput(
            "Native Masked AV requires matching source/target audio latent geometry".to_string(),
        ));
    }

    let mut plan = masked_av_tail_slice(
        source_video.shape[2],
        context_frames,
        target_video.shape[2],
        source_start_step,
    )?;
    let video_steps = plan.context_steps;
    let source_run = source_video.slice_axis(2, plan.start_t, plan.end_t)?;

    let mode = audio_tail_carryover.trim().to_lowercase();
    if mode != "full previous tail" && mode != "match video handover" {
        return Err(PrefixStreamError::InvalidInput(format!(
            "Unknown audio tail carryover mode: {audio_tail_carryover:?}"
        )));
    }

    let audio_end_frame = if mode == "full previous tail" {
        plan.previous_frame_count
    } else {
        plan.source_end_frame
    };

    let raw_audio_steps =
        py_round(((audio_end_frame - plan.source_start_frame) * AUDIO_HZ) as f64 / (FPS as f64));
    let origin_frame = *latent_boundaries(source_start_step, 0).last().unwrap_or(&0);
    let a1_raw = py_round(((origin_frame + audio_end_frame) * AUDIO_HZ) as f64 / (FPS as f64))
        - (source_audio_start_tick as isize);
    let a1 = a1_raw.clamp(0, source_audio.shape[3] as isize) as usize;
    let a0 = (a1 as isize - raw_audio_steps).max(0) as usize;
    if a1 <= a0 {
        return Err(PrefixStreamError::InvalidInput(
            "Native Masked AV audio context is empty".to_string(),
        ));
    }
    let audio_steps = a1 - a0;
    if audio_steps >= target_audio.shape[3] {
        return Err(PrefixStreamError::InvalidInput(
            "Protected audio would consume the whole target; increase target duration or use Match Video Handover".to_string(),
        ));
    }

    // Splice source_run into target_video[:, :, :video_steps]
    let target_video_tail = target_video.slice_axis(2, video_steps, target_video.shape[2])?;
    let out_video = Tensor::concat(&[&source_run, &target_video_tail], 2)?;

    // Splice source_audio[..., a0:a1] into target_audio[..., :audio_steps]
    let source_audio_slice = source_audio.slice_axis(3, a0, a1)?;
    let target_audio_tail = target_audio.slice_axis(3, audio_steps, target_audio.shape[3])?;
    let out_audio = Tensor::concat(&[&source_audio_slice, &target_audio_tail], 3)?;

    let context_denoise = video_context_denoise.clamp(0.0, 1.0);
    let v_t = out_video.shape[2];
    let v_h = out_video.shape[3];
    let v_w = out_video.shape[4];
    let mut video_mask = Tensor::ones(vec![1, 1, v_t, v_h, v_w]);
    let spatial = v_h * v_w;
    for t in 0..video_steps {
        let base = t * spatial;
        for i in 0..spatial {
            video_mask.data[base + i] = context_denoise;
        }
    }

    let a_c2 = out_audio.shape[2];
    let a_t = out_audio.shape[3];
    let mut audio_mask = Tensor::ones(vec![1, 1, a_c2, a_t]);
    let feather = audio_feather_ticks.min(audio_steps);
    let hard = audio_steps - feather;
    for c2 in 0..a_c2 {
        let row_base = c2 * a_t;
        for t in 0..hard {
            audio_mask.data[row_base + t] = 0.0;
        }
        if feather > 0 {
            for (idx, t) in (hard..audio_steps).enumerate() {
                let i = (idx + 1) as f32;
                let w = 0.5 - 0.5 * (PI * i / (feather as f32)).cos();
                audio_mask.data[row_base + t] = w;
            }
        }
    }

    plan.audio_start_tick = a0;
    plan.audio_end_tick = a1;
    plan.audio_steps = audio_steps;
    plan.audio_tail_carryover = audio_tail_carryover.to_string();
    plan.audio_feather_ticks = feather;
    plan.video_context_denoise = context_denoise;

    Ok(NativeMaskedAvOutput {
        video_latent: out_video,
        audio_latent: out_audio,
        video_mask,
        audio_mask,
        plan,
    })
}
