//! ComfyUI Custom Node implementations and workflow registry for MiniMaxHailuo-03 (H3) PrefixStream.
//!
//! Rust port of `nodes.py` and `pipeline/disk_stream.py`.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::engine::cache_manager::{latent_steps_to_pixel_frames, snap_to_run_grid, KVCacheConfig};
use crate::engine::clip_bin_manager::{
    create_placeholder_card, load_clip_asset, load_multi_safetensors, load_project_index,
    save_clip_asset, save_multi_safetensors, SaveClipRequest,
};
use crate::engine::continuation_state::{
    decode_state, encode_state, prepare_identity_references, ContinuationState,
};
use crate::engine::timeline_session_manager::{
    render_timeline_indicator_image, slice_video_and_audio, standardize_audio_dict, TimelineSession,
};
use crate::error::{PrefixStreamError, Result};
use crate::pipeline::context_noise::{encode_noisy_prefix, encode_noisy_prefix_with_plan};
use crate::pipeline::disk_stream::append_disk_clip;
use crate::pipeline::long_video_director::LongVideoSession;
use crate::pipeline::native_masked_av::{apply_native_masked_av, latent_boundaries, MaskedAvPlan};
use crate::pipeline::seam_protector::{
    standardize_image_tensor, stitch_audio_latents, stitch_audio_waveforms, stitch_video_images,
    stitch_video_latents, trim_audio_latents, trim_images_and_audio, trim_prefix_frames,
};
use crate::tensor::{
    AudioData, AudioVae, AvLatent, ConditioningEntry, IdentityReference, Tensor, VideoVae,
};

/// Canonical node class names registered by `ComfyUI-MiniMaxH3-PrefixStream`.
pub const NODE_CLASS_NAMES: [&str; 15] = [
    "MiniMaxPrefixCacheConfig",
    "MiniMaxPrefixCacheApplier",
    "MiniMaxTrimPrefixLatent",
    "MiniMaxLongVideoStitcher",
    "MiniMaxCacheMonitor",
    "MiniMaxSaveLatent",
    "MiniMaxLoadLatent",
    "MiniMaxClipBinSaver",
    "MiniMaxClipBinPicker",
    "MiniMaxClipBinTreePicker",
    "MiniMaxSafeVAEDecode",
    "MiniMaxSafeVAEDecodeAudio",
    "MiniMaxVideoChunkSlicer",
    "MiniMaxVideoPatchReassembler",
    "MiniMaxDiskVideoStream",
];

/// Return `(class_name, display_name)` mappings mirroring `NODE_DISPLAY_NAME_MAPPINGS` in `nodes.py`.
pub fn node_display_name_mappings() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        (
            "MiniMaxPrefixCacheConfig",
            "MiniMax H3 Prefix Cache Config (Infinite Stream)",
        ),
        (
            "MiniMaxPrefixCacheApplier",
            "MiniMax H3 Apply Prefix KV Cache",
        ),
        (
            "MiniMaxTrimPrefixLatent",
            "MiniMax H3 Trim Prefix Latent (Zero-Decode)",
        ),
        (
            "MiniMaxLongVideoStitcher",
            "MiniMax H3 Long Video Seamless Stitcher",
        ),
        (
            "MiniMaxCacheMonitor",
            "MiniMax H3 Cache & RoPE Telemetry Monitor",
        ),
        ("MiniMaxSaveLatent", "MiniMax H3 Save Latent (SafeTensors)"),
        ("MiniMaxLoadLatent", "MiniMax H3 Load Latent (SafeTensors)"),
        (
            "MiniMaxClipBinSaver",
            "MiniMax H3 Clip Bin Saver (DAG Asset Manager)",
        ),
        (
            "MiniMaxClipBinPicker",
            "MiniMax H3 Clip Bin Picker (Load Take / Branch)",
        ),
        (
            "MiniMaxClipBinTreePicker",
            "MiniMax H3 Clip Bin Tree Picker (Visual DAG)",
        ),
        (
            "MiniMaxSafeVAEDecode",
            "MiniMax H3 Safe VAE Decode (Initial-Aware)",
        ),
        (
            "MiniMaxSafeVAEDecodeAudio",
            "MiniMax H3 Safe VAE Decode Audio (Initial-Aware)",
        ),
        (
            "MiniMaxVideoChunkSlicer",
            "MiniMax H3 Long Video Chunk Slicer (>15s Editor)",
        ),
        (
            "MiniMaxVideoPatchReassembler",
            "MiniMax H3 Video Patch Reassembler (>15s Timeline)",
        ),
        (
            "MiniMaxDiskVideoStream",
            "MiniMax H3 Disk Video Stream (Low-RAM Export)",
        ),
    ])
}

/// 1. `MiniMaxPrefixCacheConfig`: Creates a [`LongVideoSession`] with grid-snapped rolling context.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxPrefixCacheConfigNode;

impl MiniMaxPrefixCacheConfigNode {
    /// Create a configured [`LongVideoSession`].
    #[allow(clippy::too_many_arguments)]
    pub fn create_session(
        &self,
        rolling_frames: usize,
        use_anchor: bool,
        anchor_frames: usize,
        rope_continuity: bool,
        offload_to_cpu: bool,
        clear_cache_now: bool,
        continuation_mode: &str,
        reference_noise_strength: f32,
        reference_noise_end_scale: f32,
        reference_noise_correlation: f32,
    ) -> LongVideoSession {
        let snapped_rolling = snap_to_run_grid(rolling_frames, "nearest");
        let snapped_anchor = snap_to_run_grid(anchor_frames, "nearest");
        let config = KVCacheConfig {
            enabled: true,
            use_anchor,
            anchor_frames: snapped_anchor,
            rolling_frames: snapped_rolling,
            rope_continuity,
            offload_to_cpu,
            continuation_mode: continuation_mode.to_string(),
            reference_noise_strength,
            reference_noise_end_scale,
            reference_noise_correlation,
            ..Default::default()
        };
        let mut session = LongVideoSession::new(config);
        if clear_cache_now {
            session.reset();
        }
        session
    }

    /// Create both a [`KVCacheConfig`] and [`LongVideoSession`] from concise parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn configure(
        &self,
        continuation_mode: &str,
        rolling_frames: usize,
        anchor_frames: usize,
        use_anchor: bool,
        rope_continuity: bool,
        reference_noise_strength: f32,
        reference_noise_correlation: f32,
        reference_noise_end_scale: f32,
    ) -> (KVCacheConfig, LongVideoSession) {
        let sess = self.create_session(
            rolling_frames,
            use_anchor,
            anchor_frames,
            rope_continuity,
            true,
            false,
            continuation_mode,
            reference_noise_strength,
            reference_noise_end_scale,
            reference_noise_correlation,
        );
        (sess.config.clone(), sess)
    }
}

/// Output from [`MiniMaxPrefixCacheApplierNode::apply_cache`].
#[derive(Debug, Clone)]
pub struct ApplierOutput {
    /// Updated conditioning entries.
    pub conditioning: Vec<ConditioningEntry>,
    /// Output AV latent (potentially populated with Native Masked AV prefix & noise masks).
    pub latent: Option<AvLatent>,
    /// Primary anchor image `[1, H, W, C]`.
    pub anchor_image: Option<Tensor>,
    /// Deduplicated identity references.
    pub identity_references: Vec<IdentityReference>,
    /// Optional Native Masked AV slice plan when operating in `"Native Masked AV"` mode.
    pub masked_av_plan: Option<MaskedAvPlan>,
}

/// 2. `MiniMaxPrefixCacheApplier`: Applies Native Masked AV or Safe Native Keyframes continuation.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxPrefixCacheApplierNode;

impl MiniMaxPrefixCacheApplierNode {
    /// Apply continuation prefix and identity references for the upcoming generation clip.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_cache(
        &self,
        session: &mut LongVideoSession,
        conditioning: &[ConditioningEntry],
        previous_latent: Option<&AvLatent>,
        target_latent: Option<&AvLatent>,
        anchor_latent: Option<&AvLatent>,
        reference_frames: Option<&Tensor>,
        vae: Option<&dyn VideoVae>,
        anchor_image: Option<&Tensor>,
        identity_references: Option<&[IdentityReference]>,
        extra_reference_image: Option<&Tensor>,
        anchor_strength: f32,
        extra_reference_strength: f32,
        audio_tail_carryover: &str,
        audio_feather_ticks: usize,
        video_context_denoise: f32,
        seed: u64,
    ) -> Result<ApplierOutput> {
        let mut combined_refs: Vec<IdentityReference> =
            identity_references.map(|r| r.to_vec()).unwrap_or_default();
        if let Some(extra_img) = extra_reference_image {
            combined_refs.push(IdentityReference {
                image: extra_img.first_frame(),
                strength: extra_reference_strength,
                source: "extra_reference".to_string(),
            });
        }

        let (primary_anchor, prepared_refs) =
            prepare_identity_references(anchor_image, Some(&combined_refs), anchor_strength, 1.0);

        let mut prev_v = previous_latent.and_then(|l| l.video.clone());
        let prev_a = previous_latent.and_then(|l| l.audio.clone());
        let source_start_step = previous_latent
            .and_then(|l| l.metadata.get("minimax_source_start_step"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let source_audio_start_tick = previous_latent
            .and_then(|l| l.metadata.get("minimax_source_audio_start_tick"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let is_native_masked_av = session
            .config
            .continuation_mode
            .to_lowercase()
            .contains("masked av");

        let has_ref_noise = session.config.reference_noise_strength > 0.0;
        if has_ref_noise && previous_latent.is_some() {
            if !is_native_masked_av {
                return Err(PrefixStreamError::InvalidInput(
                    "Reference noise requires Native Masked AV mode".into(),
                ));
            }
            if vae.is_none() {
                return Err(PrefixStreamError::InvalidInput(
                    "Reference noise requires an H3 video VAE".into(),
                ));
            }
            if source_start_step != 0 {
                return Err(PrefixStreamError::InvalidInput(
                    "Reference noise requires a full untrimmed source latent (source_start_step == 0)".into(),
                ));
            }
            if let (Some(ref_imgs), Some(v_vae)) = (reference_frames, vae) {
                let (noisy_lat, _) = encode_noisy_prefix(
                    v_vae,
                    ref_imgs,
                    session.config.rolling_frames,
                    session.config.reference_noise_strength,
                    seed,
                    session.config.reference_noise_correlation,
                    session.config.reference_noise_end_scale,
                )?;
                prev_v = Some(noisy_lat);
            }
        }

        if is_native_masked_av {
            if let (Some(ref src_v), Some(ref src_a), Some(tgt)) = (&prev_v, &prev_a, target_latent)
            {
                if let (Some(ref tgt_v), Some(ref tgt_a)) = (&tgt.video, &tgt.audio) {
                    let effective_feather = if has_ref_noise {
                        0
                    } else {
                        audio_feather_ticks
                    };
                    let masked_res = apply_native_masked_av(
                        tgt_v,
                        tgt_a,
                        src_v,
                        src_a,
                        session.config.rolling_frames,
                        audio_tail_carryover,
                        effective_feather,
                        source_start_step,
                        source_audio_start_tick,
                        video_context_denoise,
                    );

                    match masked_res {
                        Ok(mut res) => {
                            if has_ref_noise && reference_frames.is_none() {
                                if let Some(v_vae) = vae {
                                    let noisy_prefix = encode_noisy_prefix_with_plan(
                                        v_vae,
                                        src_v,
                                        &res.plan,
                                        session.config.reference_noise_strength,
                                        seed,
                                    )?;
                                    let c_dim = res.video_latent.shape[1];
                                    let t_dim = res.video_latent.shape[2];
                                    let h_dim = res.video_latent.shape[3];
                                    let w_dim = res.video_latent.shape[4];
                                    let steps = res.plan.context_steps;
                                    for c_i in 0..c_dim {
                                        for t_i in 0..steps {
                                            for h_i in 0..h_dim {
                                                for w_i in 0..w_dim {
                                                    let dst_idx = (((c_i * t_dim) + t_i) * h_dim
                                                        + h_i)
                                                        * w_dim
                                                        + w_i;
                                                    let src_idx = (((c_i * steps) + t_i) * h_dim
                                                        + h_i)
                                                        * w_dim
                                                        + w_i;
                                                    res.video_latent.data[dst_idx] =
                                                        noisy_prefix.data[src_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            session.last_rolling_steps = res.plan.context_steps;
                            session.last_rolling_frames = res.plan.actual_context_frames;
                            session.is_current_clip_trimmed = false;

                            let mut out_lat =
                                AvLatent::from_video_audio(res.video_latent, res.audio_latent);
                            out_lat.video_mask = Some(res.video_mask);
                            out_lat.audio_mask = Some(res.audio_mask);
                            out_lat.metadata.insert(
                                "minimax_masked_av_plan".to_string(),
                                serde_json::to_value(&res.plan).unwrap_or(Value::Null),
                            );
                            out_lat.metadata.insert(
                                "minimax_prefix_frames".to_string(),
                                json!(res.plan.actual_context_frames),
                            );

                            // Drop head keyframes colliding with protected masked prefix
                            let protected_frames = res.plan.actual_context_frames;
                            let out_cond: Vec<ConditioningEntry> = conditioning
                                .iter()
                                .map(|entry| {
                                    let mut cloned = entry.clone();
                                    cloned.minimax_keyframes.retain(|kf| {
                                        kf.resolved_frame_index >= protected_frames as f32
                                    });
                                    cloned
                                })
                                .collect();

                            return Ok(ApplierOutput {
                                conditioning: out_cond,
                                latent: Some(out_lat),
                                anchor_image: primary_anchor,
                                identity_references: prepared_refs,
                                masked_av_plan: Some(res.plan),
                            });
                        }
                        Err(err) => {
                            let msg = err.to_string();
                            let is_geometry_fallback = !has_ref_noise
                                && (msg.contains("at least 5 source frames")
                                    || msg.contains("consume the whole target")
                                    || msg.contains("has no phase-aligned"));
                            if !is_geometry_fallback {
                                return Err(err);
                            }
                            // Fall through to Safe Native fallback below
                        }
                    }
                }
            }
        }

        let anc_v = anchor_latent.and_then(|l| l.video.as_ref());
        let updated_cond = session.prepare_next_clip(
            conditioning,
            prev_v.as_ref(),
            prev_a.as_ref(),
            anc_v,
            source_start_step,
        )?;

        Ok(ApplierOutput {
            conditioning: updated_cond,
            latent: target_latent.cloned(),
            anchor_image: primary_anchor,
            identity_references: prepared_refs,
            masked_av_plan: None,
        })
    }
}

/// Output from [`MiniMaxTrimPrefixLatentNode::trim_frames`].
#[derive(Debug, Clone)]
pub struct TrimOutput {
    /// Trimmed images `[F_rem, H, W, C]`.
    pub images: Option<Tensor>,
    /// Trimmed audio waveform.
    pub audio: Option<AudioData>,
    /// Trimmed AV latent with updated `minimax_source_start_step` and `minimax_source_audio_start_tick`.
    pub latent: Option<AvLatent>,
}

/// 3. `MiniMaxTrimPrefixLatent`: Trims leading prefix latent steps (or decoded frames/audio).
#[derive(Debug, Clone, Default)]
pub struct MiniMaxTrimPrefixLatentNode;

impl MiniMaxTrimPrefixLatentNode {
    /// Trim leading `trim_frames` from pixel `images`, waveform `audio`, and/or `latent`,
    /// mirroring Python `MiniMaxTrimPrefixLatentNode.trim`.
    #[allow(clippy::too_many_arguments)]
    pub fn trim_frames(
        &self,
        trim_frames: usize,
        images: Option<&Tensor>,
        audio: Option<&AudioData>,
        latent: Option<&AvLatent>,
        mut session: Option<&mut LongVideoSession>,
        cache_config: Option<&KVCacheConfig>,
        fps: f32,
        match_tail: bool,
    ) -> Result<TrimOutput> {
        let mut actual_trim_frames = trim_frames;
        if actual_trim_frames == 0 {
            if let Some(ref s) = session {
                actual_trim_frames = s.last_rolling_frames;
            } else if let Some(cfg) = cache_config {
                let has_noise_mask = latent
                    .map(|l| l.video_mask.is_some() || l.audio_mask.is_some())
                    .unwrap_or(false);
                if has_noise_mask {
                    actual_trim_frames = latent
                        .and_then(|l| l.metadata.get("minimax_prefix_frames"))
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                        .unwrap_or(cfg.rolling_frames);
                } else {
                    actual_trim_frames = 0;
                }
            }
        }

        let (out_images, out_audio) = if let Some(imgs) = images {
            let (ti, ta) = trim_images_and_audio(imgs, audio, actual_trim_frames, fps, match_tail)?;
            (Some(ti), ta)
        } else if let Some(aud) = audio {
            let sr = aud.sample_rate as f64;
            let head = ((actual_trim_frames as f64) / (fps as f64) * sr).round_ties_even() as usize;
            let total_s = aud.waveform.shape[2];
            let sliced = aud.waveform.slice_axis(2, head.min(total_s), total_s)?;
            (
                None,
                Some(AudioData {
                    waveform: sliced,
                    sample_rate: aud.sample_rate,
                }),
            )
        } else {
            (None, None)
        };

        if actual_trim_frames > 0 {
            if let Some(ref mut s) = session {
                s.is_current_clip_trimmed = true;
            }
        }

        let out_latent = if let Some(target_latent) = latent {
            let mut cloned = target_latent.clone();
            if let Some(ref v) = target_latent.video {
                let source_step = target_latent
                    .metadata
                    .get("minimax_source_start_step")
                    .and_then(|val| val.as_u64())
                    .unwrap_or(0) as usize;
                let mut trim_steps = 0usize;
                if actual_trim_frames > 0 {
                    let bounds = latent_boundaries(v.shape[2], source_step);
                    trim_steps = bounds
                        .iter()
                        .position(|&end| end >= actual_trim_frames)
                        .unwrap_or(v.shape[2]);
                }
                if trim_steps > 0 {
                    let trimmed_v = v.slice_axis(2, trim_steps.min(v.shape[2]), v.shape[2])?;
                    cloned.video = Some(trimmed_v);
                    let mut audio_trim_steps = 0usize;
                    if let Some(ref a) = target_latent.audio {
                        let end_frame = *latent_boundaries(source_step + trim_steps, 0)
                            .last()
                            .unwrap_or(&0);
                        let audio_origin = target_latent
                            .metadata
                            .get("minimax_source_audio_start_tick")
                            .and_then(|val| val.as_u64())
                            .unwrap_or(0) as usize;
                        let target_tick =
                            ((end_frame as f64) * 40.0 / 24.0).round_ties_even() as usize;
                        let last_dim = a.ndim() - 1;
                        audio_trim_steps = target_tick
                            .saturating_sub(audio_origin)
                            .min(a.shape[last_dim]);
                        cloned.audio =
                            Some(a.slice_axis(last_dim, audio_trim_steps, a.shape[last_dim])?);
                    }
                    cloned.metadata.insert(
                        "minimax_source_start_step".to_string(),
                        json!(source_step + trim_steps.min(v.shape[2])),
                    );
                    let prev_audio_origin = target_latent
                        .metadata
                        .get("minimax_source_audio_start_tick")
                        .and_then(|val| val.as_u64())
                        .unwrap_or(0) as usize;
                    cloned.metadata.insert(
                        "minimax_source_audio_start_tick".to_string(),
                        json!(prev_audio_origin + audio_trim_steps),
                    );
                    cloned.video_mask = None;
                    cloned.audio_mask = None;
                    cloned
                        .metadata
                        .insert("minimax_prefix_frames".to_string(), json!(0));
                }
            }
            Some(cloned)
        } else {
            None
        };

        Ok(TrimOutput {
            images: out_images,
            audio: out_audio,
            latent: out_latent,
        })
    }

    /// Trim prefix latent steps from `samples` and/or leading frames from `images` & `audio`.
    pub fn trim(
        &self,
        session: Option<&mut LongVideoSession>,
        samples: Option<&AvLatent>,
        images: Option<&Tensor>,
        audio: Option<&AudioData>,
        override_trim_steps: isize,
        fps: f32,
    ) -> Result<(Option<AvLatent>, Option<Tensor>, Option<AudioData>)> {
        let trim_steps = if override_trim_steps >= 0 {
            override_trim_steps as usize
        } else if let Some(ref s) = session {
            if s.is_current_clip_trimmed {
                0
            } else {
                s.last_rolling_steps
            }
        } else {
            0
        };

        let trim_frames = latent_steps_to_pixel_frames(trim_steps);

        let out_latent = if let Some(lat) = samples {
            let mut cloned = lat.clone();
            if let Some(ref v) = lat.video {
                if trim_steps < v.shape[2] {
                    cloned.video = Some(trim_prefix_frames(v, trim_steps)?);
                }
            }
            if let Some(ref a) = lat.audio {
                let audio_steps = ((trim_frames as f64) * 40.0 / 24.0).round_ties_even() as usize;
                let last_dim = a.ndim() - 1;
                if audio_steps < a.shape[last_dim] {
                    cloned.audio = Some(trim_audio_latents(a, audio_steps)?);
                }
            }
            cloned.metadata.insert(
                "minimax_source_start_step".to_string(),
                json!(trim_steps % 5),
            );
            Some(cloned)
        } else {
            None
        };

        let (out_images, out_audio) = if let Some(imgs) = images {
            let (ti, ta) = trim_images_and_audio(imgs, audio, trim_frames, fps, true)?;
            (Some(ti), ta)
        } else {
            (None, audio.cloned())
        };

        if let Some(s) = session {
            s.is_current_clip_trimmed = true;
        }

        Ok((out_latent, out_images, out_audio))
    }
}

/// 4. `MiniMaxLongVideoStitcher`: Seamlessly joins decoded `IMAGE`/`AUDIO` streams or `LATENT` streams.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxLongVideoStitcherNode;

impl MiniMaxLongVideoStitcherNode {
    /// Stitch `previous_*` and `current_*` video/audio streams.
    #[allow(clippy::too_many_arguments)]
    pub fn stitch(
        &self,
        session: Option<&LongVideoSession>,
        previous_images: Option<&Tensor>,
        current_images: Option<&Tensor>,
        previous_audio: Option<&AudioData>,
        current_audio: Option<&AudioData>,
        previous_latent: Option<&AvLatent>,
        current_latent: Option<&AvLatent>,
        overlap_frames: isize,
        crossfade_frames: usize,
        audio_crossfade_ms: f32,
        luminance_match: bool,
        fps: f32,
    ) -> Result<(Option<Tensor>, Option<AudioData>, Option<AvLatent>)> {
        let trim_f = if overlap_frames >= 0 {
            overlap_frames as usize
        } else if let Some(s) = session {
            if s.is_current_clip_trimmed {
                0
            } else {
                s.last_rolling_frames
            }
        } else {
            22
        };

        let curr_total_frames = current_images.map(|t| t.shape[0]).unwrap_or(0);

        let stitched_images = if let Some(curr_imgs) = current_images {
            Some(stitch_video_images(
                previous_images,
                curr_imgs,
                trim_f,
                crossfade_frames,
                luminance_match,
                16,
            )?)
        } else {
            previous_images.cloned()
        };

        let stitched_audio = if current_audio.is_some() || previous_audio.is_some() {
            stitch_audio_waveforms(
                previous_audio,
                current_audio,
                curr_total_frames,
                trim_f,
                audio_crossfade_ms,
                fps,
            )?
        } else {
            None
        };

        let stitched_latent = match (previous_latent, current_latent) {
            (Some(prev_l), Some(curr_l)) => {
                let ov_steps = session.map(|s| s.last_rolling_steps).unwrap_or(2);
                let v = match (&prev_l.video, &curr_l.video) {
                    (Some(pv), Some(cv)) => Some(stitch_video_latents(pv, cv, ov_steps, 2)?),
                    (_, Some(cv)) => Some(cv.clone()),
                    (Some(pv), None) => Some(pv.clone()),
                    _ => None,
                };
                let a = match (&prev_l.audio, &curr_l.audio) {
                    (Some(pa), Some(ca)) => {
                        let a_ov = ((trim_f as f64) * 40.0 / 24.0).round_ties_even() as usize;
                        Some(stitch_audio_latents(Some(pa), ca, a_ov)?)
                    }
                    (_, Some(ca)) => Some(ca.clone()),
                    (Some(pa), None) => Some(pa.clone()),
                    _ => None,
                };
                Some(AvLatent {
                    video: v,
                    audio: a,
                    video_mask: None,
                    audio_mask: None,
                    metadata: curr_l.metadata.clone(),
                })
            }
            (None, Some(curr_l)) => Some(curr_l.clone()),
            (Some(prev_l), None) => Some(prev_l.clone()),
            (None, None) => None,
        };

        Ok((stitched_images, stitched_audio, stitched_latent))
    }
}

/// 5. `MiniMaxCacheMonitor`: Formats human-readable telemetry for a [`LongVideoSession`].
#[derive(Debug, Clone, Default)]
pub struct MiniMaxCacheMonitorNode;

impl MiniMaxCacheMonitorNode {
    /// Generate a telemetry report string for `session`.
    pub fn monitor(&self, session: &LongVideoSession) -> String {
        let stats = session.cache_manager.get_telemetry_stats();
        let cursor = session.cursor_tracker.get_summary();
        format!(
            "=== MiniMax H3 PrefixStream Telemetry ===\n\
             Mode: {}\n\
             Clip Index: #{}\n\
             Rolling Context: {} frames ({} latent steps)\n\
             Cached Layers: {} (Anchor: {}, Rolling: {})\n\
             Estimated Cache Size: {:.3} MB\n\
             3D-RoPE Cursor: next_step={}, total_frames={}, clips={}",
            session.config.continuation_mode,
            session.current_clip_index,
            session.last_rolling_frames,
            session.last_rolling_steps,
            stats.cached_layers,
            stats.has_anchor_cache,
            stats.has_rolling_cache,
            stats.estimated_vram_mb,
            cursor.current_global_step,
            cursor.total_generated_frames,
            cursor.total_clips_generated,
        )
    }
}

/// 6. `MiniMaxSaveLatent` & 7. `MiniMaxLoadLatent`: Save and load `.safetensors` AV latents with continuation state.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxSaveLatentNode;

impl MiniMaxSaveLatentNode {
    /// Save an [`AvLatent`] and optional [`ContinuationState`] to a `.safetensors` file.
    pub fn save_latent(
        &self,
        path: &Path,
        latent: &AvLatent,
        state: Option<&ContinuationState>,
    ) -> Result<()> {
        let mut tensors = HashMap::new();
        if let Some(ref v) = latent.video {
            tensors.insert("video_samples".to_string(), v.clone());
        }
        if let Some(ref a) = latent.audio {
            tensors.insert("audio_samples".to_string(), a.clone());
        }
        let mut metadata = HashMap::new();
        if let Some(st) = state {
            let (st_tensors, st_meta) = encode_state(st);
            tensors.extend(st_tensors);
            metadata.extend(st_meta);
        }
        save_multi_safetensors(path, &tensors, Some(metadata))
    }
}

/// 7. `MiniMaxLoadLatent`: Load an [`AvLatent`] and optional [`ContinuationState`] from `.safetensors`.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxLoadLatentNode;

impl MiniMaxLoadLatentNode {
    /// Load an [`AvLatent`] and optional [`ContinuationState`] from a `.safetensors` file.
    pub fn load_latent(&self, path: &Path) -> Result<(AvLatent, Option<ContinuationState>)> {
        let (mut tensors, metadata) = load_multi_safetensors(path)?;
        let state = decode_state(&tensors, Some(&metadata))?;
        let video = tensors
            .remove("video_samples")
            .or_else(|| tensors.remove("samples"));
        let audio = tensors.remove("audio_samples");
        let latent = AvLatent {
            video,
            audio,
            video_mask: None,
            audio_mask: None,
            metadata: HashMap::new(),
        };
        Ok((latent, state))
    }
}

/// 8. `MiniMaxClipBinSaver`: Saves a clip take into the project's DAG Clip Bin.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxClipBinSaverNode;

impl MiniMaxClipBinSaverNode {
    /// Save a clip take to the Clip Bin.
    pub fn save_to_bin(&self, req: SaveClipRequest<'_>) -> Result<(String, String)> {
        let meta = save_clip_asset(req)?;
        let summary = format!(
            "Saved {} (Scene {}, Take {}, {}f)",
            meta.clip_id, meta.scene_num, meta.take_num, meta.frames
        );
        Ok((meta.clip_id, summary))
    }
}

/// Output from [`MiniMaxClipBinPickerNode::pick_clip`].
#[derive(Debug, Clone)]
pub struct PickedClipOutput {
    /// Optional AV latent (`None` in Initial Clip Mode).
    pub latent: Option<AvLatent>,
    /// Tail frames `[F, H, W, 3]` (or placeholder card in Initial Clip Mode).
    pub tail_frame: Tensor,
    /// First frame `[1, H, W, 3]`.
    pub first_frame: Tensor,
    /// Prompt string.
    pub prompt: String,
    /// Selected `clip_id` (or `"[INITIAL_GENERATION]"`).
    pub clip_id: String,
}

/// 9. `MiniMaxClipBinPicker` & 10. `MiniMaxClipBinTreePicker`: Loads a clip take or initializes a root clip.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxClipBinPickerNode;

impl MiniMaxClipBinPickerNode {
    /// Pick a clip from `project_name` or enter Initial Clip Mode when the project is empty / forced.
    pub fn pick_clip(
        &self,
        base_dir: &Path,
        project_name: &str,
        mode: &str,
        filter_rating: i32,
        clip_selection: &str,
    ) -> Result<PickedClipOutput> {
        let index = load_project_index(base_dir, project_name)?;
        let is_initial =
            mode.starts_with("Force Initial") || (mode.starts_with("Auto") && index.is_empty());

        if is_initial {
            let card = create_placeholder_card(256, 256, "Initial Clip Mode");
            let mut data = Vec::with_capacity(256 * 256 * 3);
            for p in card.pixels() {
                data.push(p[0] as f32 / 255.0);
                data.push(p[1] as f32 / 255.0);
                data.push(p[2] as f32 / 255.0);
            }
            let placeholder = Tensor::from_vec(vec![1, 256, 256, 3], data)?;
            return Ok(PickedClipOutput {
                latent: None,
                tail_frame: placeholder.clone(),
                first_frame: placeholder,
                prompt: String::new(),
                clip_id: "[INITIAL_GENERATION]".to_string(),
            });
        }

        if index.is_empty() {
            return Err(PrefixStreamError::NotFound(format!(
                "MiniMaxClipBinPicker: No clips found in project '{project_name}'"
            )));
        }

        let sel = clip_selection.trim();
        let target_id = if sel.is_empty() || sel.eq_ignore_ascii_case("latest") || sel == "0" {
            let mut candidates: Vec<_> = index
                .values()
                .filter(|c| c.rating >= filter_rating)
                .collect();
            if candidates.is_empty() {
                candidates = index.values().collect();
            }
            candidates.sort_by(|a, b| {
                b.scene_num
                    .cmp(&a.scene_num)
                    .then_with(|| b.take_num.cmp(&a.take_num))
            });
            candidates[0].clip_id.clone()
        } else {
            sel.split(' ').next().unwrap_or(sel).to_string()
        };

        let loaded = load_clip_asset(base_dir, project_name, &target_id)?;
        let first_frame = loaded.tail_images.first_frame();
        let latent = if loaded.video_latent.is_some() || loaded.audio_latent.is_some() {
            Some(AvLatent {
                video: loaded.video_latent,
                audio: loaded.audio_latent,
                video_mask: None,
                audio_mask: None,
                metadata: HashMap::new(),
            })
        } else {
            None
        };

        Ok(PickedClipOutput {
            latent,
            tail_frame: loaded.tail_images,
            first_frame,
            prompt: loaded.meta.prompt,
            clip_id: loaded.meta.clip_id,
        })
    }
}

/// 11. `MiniMaxSafeVAEDecode`: Safe video VAE decoder that returns an empty `[0, 768, 1344, 3]` tensor on `None`.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxSafeVAEDecodeNode;

impl MiniMaxSafeVAEDecodeNode {
    /// Decode `samples` using `vae`, or return an empty `[0, 768, 1344, 3]` tensor when `samples` is `None`.
    pub fn decode(&self, vae: &dyn VideoVae, samples: Option<&AvLatent>) -> Result<Tensor> {
        let Some(lat) = samples else {
            return Ok(Tensor::zeros(vec![0, 768, 1344, 3]));
        };
        let Some(ref v) = lat.video else {
            return Ok(Tensor::zeros(vec![0, 768, 1344, 3]));
        };
        let decoded = vae.decode(v)?;
        standardize_image_tensor(&decoded)
    }
}

/// 12. `MiniMaxSafeVAEDecodeAudio`: Safe audio VAE decoder that returns `None` on `None` or missing audio latent.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxSafeVAEDecodeAudioNode;

impl MiniMaxSafeVAEDecodeAudioNode {
    /// Decode audio latent from `samples` using `vae`, or return `None` when absent.
    pub fn decode(
        &self,
        vae: &dyn AudioVae,
        samples: Option<&AvLatent>,
    ) -> Result<Option<AudioData>> {
        let Some(lat) = samples else {
            return Ok(None);
        };
        let Some(ref a) = lat.audio else {
            return Ok(None);
        };
        let decoded = vae.decode(a)?;
        Ok(Some(standardize_audio_dict(Some(&decoded), 32000)))
    }
}

/// Context dictionary emitted by [`MiniMaxVideoChunkSlicerNode`] and consumed by [`MiniMaxVideoPatchReassemblerNode`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceContext {
    /// Session ID.
    pub session_id: String,
    /// Start frame of the sliced chunk.
    pub start_frame: usize,
    /// Exclusive end frame of the sliced chunk.
    pub end_frame: usize,
    /// Number of valid frames before any padding.
    pub valid_frames: usize,
    /// Total frames in the master timeline.
    pub total_frames: usize,
    /// Master video FPS.
    pub fps: f32,
}

/// 13. `MiniMaxVideoChunkSlicer`: Slices a master video/audio stream into manageable chunks and registers the session.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxVideoChunkSlicerNode;

impl MiniMaxVideoChunkSlicerNode {
    /// Slice a video chunk from `images` & `audio` and register the master timeline in `session_id`.
    #[allow(clippy::too_many_arguments)]
    pub fn slice_chunk(
        &self,
        base_dir: &Path,
        session_id: &str,
        images: &Tensor,
        audio: Option<&AudioData>,
        fps: f32,
        start_frame: isize,
        end_frame: isize,
        pad_to_length: bool,
        reset_session: bool,
    ) -> Result<(Tensor, AudioData, SliceContext, Tensor, String)> {
        let mut session = TimelineSession::load_or_create(base_dir, session_id)?;
        if reset_session {
            session.meta.patches.clear();
        }
        session.register_master_source(Some(images), audio, fps, reset_session, None)?;

        let sliced =
            slice_video_and_audio(images, audio, fps, start_frame, end_frame, pad_to_length)?;
        let valid_frames = sliced.actual_end - sliced.actual_start;

        let patched_ranges: Vec<(usize, usize)> = session
            .meta
            .patches
            .values()
            .map(|p| (p.start_frame, p.end_frame))
            .collect();
        let indicator = render_timeline_indicator_image(
            sliced.total_frames,
            Some((sliced.actual_start, sliced.actual_end)),
            Some(&patched_ranges),
            fps,
            session_id,
        )?;

        let ctx = SliceContext {
            session_id: session_id.to_string(),
            start_frame: sliced.actual_start,
            end_frame: sliced.actual_end,
            valid_frames,
            total_frames: sliced.total_frames,
            fps,
        };
        let info = format!(
            "Sliced [{}..{}] ({}f) from total {}f @ {:.1}fps",
            ctx.start_frame, ctx.end_frame, valid_frames, ctx.total_frames, fps
        );

        Ok((
            sliced.chunk_images,
            sliced.chunk_audio,
            ctx,
            indicator,
            info,
        ))
    }
}

/// 14. `MiniMaxVideoPatchReassembler`: Saves a processed chunk patch and optionally reassembles the full timeline.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxVideoPatchReassemblerNode;

impl MiniMaxVideoPatchReassemblerNode {
    /// Save `chunk_images` & `chunk_audio` into the timeline session and reassemble.
    #[allow(clippy::too_many_arguments)]
    pub fn reassemble_patch(
        &self,
        base_dir: &Path,
        slice_context: &SliceContext,
        chunk_images: &Tensor,
        chunk_audio: Option<&AudioData>,
        blend_overlap_frames: usize,
        luminance_match: bool,
        notes: &str,
    ) -> Result<(Tensor, AudioData, Tensor, String)> {
        let mut session = TimelineSession::load_or_create(base_dir, &slice_context.session_id)?;

        // Trim any tail padding added during slicing
        let valid_imgs = if chunk_images.shape[0] > slice_context.valid_frames {
            chunk_images.slice_dim0(0, slice_context.valid_frames)?
        } else {
            chunk_images.clone()
        };

        session.save_chunk_patch(
            &valid_imgs,
            chunk_audio,
            slice_context.start_frame,
            Some(slice_context.end_frame),
            slice_context.fps,
            blend_overlap_frames,
            notes,
        )?;

        let (master_imgs, master_aud, report) = session.reassemble_full_timeline(
            None,
            None,
            Some(blend_overlap_frames),
            luminance_match,
        )?;

        let patched_ranges: Vec<(usize, usize)> = session
            .meta
            .patches
            .values()
            .map(|p| (p.start_frame, p.end_frame))
            .collect();
        let indicator = render_timeline_indicator_image(
            session.meta.total_frames,
            Some((slice_context.start_frame, slice_context.end_frame)),
            Some(&patched_ranges),
            slice_context.fps,
            &slice_context.session_id,
        )?;

        Ok((master_imgs, master_aud, indicator, report))
    }
}

/// 15. `MiniMaxDiskVideoStream`: Appends decoded clips to disk without keeping the full timeline in memory.
#[derive(Debug, Clone, Default)]
pub struct MiniMaxDiskVideoStreamNode;

impl MiniMaxDiskVideoStreamNode {
    /// Append `images` and `audio` to a named disk stream under `project_name`.
    #[allow(clippy::too_many_arguments)]
    pub fn append(
        &self,
        base_dir: &Path,
        project_name: &str,
        stream_name: &str,
        mut trim_frames: usize,
        fps: f32,
        export: bool,
        images: Option<&Tensor>,
        audio: Option<&AudioData>,
        session: Option<&LongVideoSession>,
    ) -> Result<(String, String, usize)> {
        if trim_frames == 0 {
            if let Some(s) = session {
                trim_frames = s.last_rolling_frames;
            }
        }
        append_disk_clip(
            base_dir,
            project_name,
            stream_name,
            images,
            audio,
            trim_frames,
            fps,
            export,
        )
    }
}

/// Summary of a validated ComfyUI workflow JSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowValidationReport {
    /// Total nodes in the workflow.
    pub total_nodes: usize,
    /// MiniMax PrefixStream custom nodes discovered in the workflow.
    pub minimax_nodes: Vec<String>,
    /// Whether the workflow is valid.
    pub valid: bool,
}

/// Validate a ComfyUI workflow JSON payload and extract all MiniMax PrefixStream nodes.
pub fn validate_workflow_json(workflow_json: &str) -> Result<WorkflowValidationReport> {
    let val: Value = serde_json::from_str(workflow_json)?;
    let nodes_arr = val.get("nodes").and_then(|v| v.as_array()).ok_or_else(|| {
        PrefixStreamError::InvalidInput("Workflow JSON missing 'nodes' array".to_string())
    })?;

    let mut minimax_nodes = Vec::new();
    for node in nodes_arr {
        if let Some(node_type) = node.get("type").and_then(|v| v.as_str()) {
            if NODE_CLASS_NAMES.contains(&node_type) || node_type.starts_with("MiniMax") {
                minimax_nodes.push(node_type.to_string());
            }
        }
    }

    Ok(WorkflowValidationReport {
        total_nodes: nodes_arr.len(),
        minimax_nodes,
        valid: true,
    })
}
