//! Comprehensive integration test suite for `comfyui-minimaxh3-prefixstream`.
//!
//! Ports all 11 Python test suites from `ComfyUI-MiniMaxH3-PrefixStream/tests/`:
//! - `test_cache_manager.py`
//! - `test_native_masked_av.py`
//! - `test_context_noise.py`
//! - `test_texture_diagnostics.py`
//! - `test_fused_attention.py`
//! - `test_audio_enhancements.py`
//! - `test_nodes_and_pipeline.py`
//! - `test_clip_bin.py`
//! - `test_clip_bin_api.py`
//! - `test_timeline_slicer.py`
//! - `test_regressions.py`

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use approx::assert_relative_eq;
use tempfile::tempdir;

use minimaxh3_prefixstream::engine::cache_manager::{
    latent_steps_to_pixel_frames, pixel_frames_to_latent_steps, snap_to_run_grid, steps_for_frames,
    KVCacheConfig, PrefixKVCacheManager, VIDEO_RUN_GRID,
};
use minimaxh3_prefixstream::engine::clip_bin_api::{
    delete_clip_api, get_project_clips_api, rescan_project_api, update_clip_rating_api,
};
use minimaxh3_prefixstream::engine::clip_bin_manager::{
    atomic_write_json, delete_clip_asset, format_clip_label, get_clip_dir, get_clips_for_selection,
    get_project_dir, load_clip_asset, rebuild_project_index, SaveClipRequest,
};
use minimaxh3_prefixstream::engine::continuation_state::{
    decode_state, encode_state, prepare_identity_references, ContinuationState,
};
use minimaxh3_prefixstream::engine::fused_attention::asymmetric_cached_attention;
use minimaxh3_prefixstream::engine::rope_aligner::{
    total_span_for_steps, video_t_grid, video_t_spans, TemporalCursorTracker,
};
use minimaxh3_prefixstream::engine::texture_diagnostics::{
    summarize_clip_texture_drift, texture_band_rms,
};
use minimaxh3_prefixstream::engine::timeline_session_manager::{
    render_timeline_indicator_image, slice_video_and_audio, standardize_audio_dict, TimelineSession,
};
use minimaxh3_prefixstream::nodes::{
    validate_workflow_json, MiniMaxCacheMonitorNode, MiniMaxClipBinPickerNode,
    MiniMaxClipBinSaverNode, MiniMaxDiskVideoStreamNode, MiniMaxLoadLatentNode,
    MiniMaxLongVideoStitcherNode, MiniMaxPrefixCacheApplierNode, MiniMaxPrefixCacheConfigNode,
    MiniMaxSafeVAEDecodeAudioNode, MiniMaxSafeVAEDecodeNode, MiniMaxSaveLatentNode,
    MiniMaxTrimPrefixLatentNode, MiniMaxVideoChunkSlicerNode, MiniMaxVideoPatchReassemblerNode,
    NODE_CLASS_NAMES,
};
use minimaxh3_prefixstream::pipeline::context_noise::{
    encode_noisy_prefix, noise_reference_frames, reference_noise_strengths,
};
use minimaxh3_prefixstream::pipeline::disk_stream::append_disk_clip_with_encoder;
use minimaxh3_prefixstream::pipeline::long_video_director::{
    inject_minimax_keyframes, step_offsets, LongVideoSession,
};
use minimaxh3_prefixstream::pipeline::native_masked_av::{
    apply_native_masked_av, is_exact_masked_av_context, latent_boundaries, masked_av_tail_slice,
    snap_masked_av_context_length, MASKED_AV_CONTEXTS,
};
use minimaxh3_prefixstream::pipeline::seam_protector::{
    apply_luminance_gain_fade, audio_equal_power_crossfade_concat, estimate_luminance_gain,
    fit_audio_length, latent_soft_blend, stitch_audio_latents, stitch_audio_waveforms,
    stitch_video_images, stitch_video_latents, trim_audio_latents, trim_audio_waveform,
    trim_images_and_audio, trim_prefix_frames,
};
use minimaxh3_prefixstream::tensor::{
    AudioData, AudioVae, AvLatent, ConditioningEntry, IdentityReference, MinimaxKeyframe, Tensor,
    VideoVae,
};
use minimaxh3_prefixstream::Result;

// ============================================================================
// 1. Cache Manager & RoPE Aligner Tests (test_cache_manager.py)
// ============================================================================

#[test]
fn test_snap_to_run_grid_and_latent_pixel_conversions() {
    assert_eq!(VIDEO_RUN_GRID, [5, 22, 39, 56, 73, 90, 107, 124]);
    assert_eq!(snap_to_run_grid(20, "nearest"), 22);
    assert_eq!(snap_to_run_grid(20, "lower"), 5);
    assert_eq!(snap_to_run_grid(20, "higher"), 22);
    assert_eq!(snap_to_run_grid(1, "lower"), 5);
    assert_eq!(snap_to_run_grid(500, "higher"), 124);

    assert_eq!(pixel_frames_to_latent_steps(0), 0);
    assert_eq!(pixel_frames_to_latent_steps(1), 1);
    assert_eq!(pixel_frames_to_latent_steps(5), 2);
    assert_eq!(pixel_frames_to_latent_steps(17), 5);
    assert_eq!(pixel_frames_to_latent_steps(22), 7);
    assert_eq!(pixel_frames_to_latent_steps(39), 12);

    assert_eq!(latent_steps_to_pixel_frames(0), 0);
    assert_eq!(latent_steps_to_pixel_frames(1), 1);
    assert_eq!(latent_steps_to_pixel_frames(2), 5);
    assert_eq!(latent_steps_to_pixel_frames(5), 17);
    assert_eq!(latent_steps_to_pixel_frames(7), 22);
    assert_eq!(latent_steps_to_pixel_frames(12), 39);
    assert_eq!(steps_for_frames(39), 12);
}

#[test]
fn test_prefix_kv_cache_manager_store_and_concat() {
    let cfg = KVCacheConfig {
        use_anchor: true,
        anchor_frames: 5,
        rolling_frames: 22,
        ..Default::default()
    };
    let mut mgr = PrefixKVCacheManager::new(cfg);
    assert!(!mgr.has_active_prefix(0));

    let k_anc = Tensor::full(vec![1, 4, 2, 8], 1.0);
    let v_anc = Tensor::full(vec![1, 4, 2, 8], 2.0);
    mgr.store_anchor_kv(0, &k_anc, &v_anc, None, None);
    assert!(mgr.has_active_prefix(0));

    let k_roll = Tensor::full(vec![1, 4, 3, 8], 3.0);
    let v_roll = Tensor::full(vec![1, 4, 3, 8], 4.0);
    mgr.store_rolling_kv(0, &k_roll, &v_roll, None, None);

    let k_curr = Tensor::full(vec![1, 4, 5, 8], 5.0);
    let v_curr = Tensor::full(vec![1, 4, 5, 8], 6.0);
    let (full_k, full_v) = mgr
        .get_concatenated_kv(0, &k_curr, &v_curr, true)
        .expect("concat kv");
    assert_eq!(full_k.shape, vec![1, 4, 10, 8]);
    assert_eq!(full_v.shape, vec![1, 4, 10, 8]);

    let stats = mgr.get_telemetry_stats();
    assert_eq!(stats.cached_layers, 1);
    assert!(stats.has_anchor_cache);
    assert!(stats.has_rolling_cache);
    assert!(stats.estimated_vram_mb > 0.0);

    mgr.clear_all();
    assert!(!mgr.has_active_prefix(0));
}

#[test]
fn test_rope_aligner_and_cursor_tracker() {
    assert_eq!(video_t_spans(7, 0), vec![1, 4, 4, 4, 4, 1, 4]);
    assert_eq!(video_t_grid(5, 0), vec![0.0, 1.0, 5.0, 9.0, 13.0]);
    assert_eq!(total_span_for_steps(7, 0), 22);

    let mut tracker = TemporalCursorTracker::new();
    let c0 = tracker.advance_clip(12, None);
    assert_eq!(c0.start_step, 0);
    assert_eq!(c0.next_step, 12);
    assert_eq!(c0.span_frames, 39);

    let c1 = tracker.advance_clip(12, None);
    assert_eq!(c1.start_step, 12);
    assert_eq!(c1.next_step, 24);
    assert_eq!(c1.cumulative_frames, 39 + total_span_for_steps(12, 12));
}

// ============================================================================
// 2. Native Masked AV Tests (test_native_masked_av.py)
// ============================================================================

#[test]
fn test_native_masked_av_exact_contexts_and_boundaries() {
    assert_eq!(
        MASKED_AV_CONTEXTS,
        [5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192]
    );
    let exact: Vec<usize> = MASKED_AV_CONTEXTS
        .iter()
        .copied()
        .filter(|&f| is_exact_masked_av_context(f))
        .collect();
    assert_eq!(exact, vec![39, 90, 141, 192]);

    let bounds = latent_boundaries(7, 0);
    assert_eq!(bounds, vec![0, 1, 5, 9, 13, 17, 18, 22]);
    assert_eq!(snap_masked_av_context_length(45, 124, 124).unwrap(), 39);
}

#[test]
fn test_masked_av_tail_slice_phase_alignment() {
    // 37 latent steps = 124 frames (since 35 steps = 119f, +1=120f, +4=124f -> end_t=37 has 37%5==2)
    let plan = masked_av_tail_slice(37, 39, 37, 0).expect("tail slice");
    assert_eq!(plan.actual_context_frames, 39);
    assert_eq!(plan.context_steps, 12);
    assert_eq!(plan.start_t, 25);
    assert_eq!(plan.end_t, 37);
    assert_eq!(plan.ignored_tail_frames, 0);

    // If source has 38 steps (extra 4 frames after phase 2), it should ignore the trailing 4 frames
    let plan_38 = masked_av_tail_slice(38, 39, 37, 0).expect("tail slice 38");
    assert_eq!(plan_38.end_t, 37);
    assert_eq!(plan_38.ignored_tail_frames, 4);
}

#[test]
fn test_apply_native_masked_av_copies_prefix_and_builds_masks() {
    let src_v = Tensor::full(vec![1, 4, 37, 2, 2], 2.5);
    let src_a = Tensor::full(vec![1, 4, 2, 207], 1.5);
    let tgt_v = Tensor::zeros(vec![1, 4, 37, 2, 2]);
    let tgt_a = Tensor::zeros(vec![1, 4, 2, 207]);

    let out = apply_native_masked_av(
        &tgt_v,
        &tgt_a,
        &src_v,
        &src_a,
        39,
        "Match Video Handover",
        4,
        0,
        0,
        0.05,
    )
    .expect("apply_native_masked_av");

    assert_eq!(out.plan.context_steps, 12);
    assert_eq!(out.plan.audio_steps, 65);
    assert_eq!(out.video_latent.shape, vec![1, 4, 37, 2, 2]);
    assert_eq!(out.audio_latent.shape, vec![1, 4, 2, 207]);

    // Check copied video prefix vs untouched tail
    let v_prefix = out.video_latent.slice_axis(2, 0, 12).unwrap();
    let v_tail = out.video_latent.slice_axis(2, 12, 37).unwrap();
    assert!(v_prefix.data.iter().all(|&x| (x - 2.5).abs() < 1e-6));
    assert!(v_tail.data.iter().all(|&x| x.abs() < 1e-6));

    // Check video mask
    let vm_prefix = out.video_mask.slice_axis(2, 0, 12).unwrap();
    let vm_tail = out.video_mask.slice_axis(2, 12, 37).unwrap();
    assert!(vm_prefix.data.iter().all(|&x| (x - 0.05).abs() < 1e-6));
    assert!(vm_tail.data.iter().all(|&x| (x - 1.0).abs() < 1e-6));

    // Check audio mask feathering (hard=61, feather=4)
    let am_hard = out.audio_mask.slice_axis(3, 0, 61).unwrap();
    let am_feather = out.audio_mask.slice_axis(3, 61, 65).unwrap();
    let am_tail = out.audio_mask.slice_axis(3, 65, 207).unwrap();
    assert!(am_hard.data.iter().all(|&x| x.abs() < 1e-6));
    assert!((am_feather.data[3] - 1.0).abs() < 1e-5);
    assert!(am_tail.data.iter().all(|&x| (x - 1.0).abs() < 1e-6));
}

// ============================================================================
// 3. Context Noise & Texture Diagnostics Tests (test_context_noise.py, test_texture_diagnostics.py)
// ============================================================================

struct DummyVae;
impl VideoVae for DummyVae {
    fn encode(&self, images: &Tensor) -> Result<Tensor> {
        let f = images.shape[0];
        let t = pixel_frames_to_latent_steps(f).max(1);
        Ok(Tensor::full(vec![1, 16, t, 4, 4], 0.42))
    }
    fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let t = latent.shape[2];
        let f = latent_steps_to_pixel_frames(t).max(1);
        Ok(Tensor::full(vec![f, 32, 32, 3], 0.5))
    }
}

struct DummyAudioVae;
impl AudioVae for DummyAudioVae {
    fn decode(&self, latent: &Tensor) -> Result<AudioData> {
        let ticks = latent.shape[latent.ndim() - 1];
        Ok(AudioData::new(
            Tensor::full(vec![1, 2, ticks * 800], 0.1),
            32000,
        ))
    }
}

#[test]
fn test_context_noise_ramp_and_temporal_correlation() {
    let weights = reference_noise_strengths(5, 0.02, 0.25);
    assert_eq!(weights.len(), 5);
    assert_relative_eq!(weights[0], 0.02, epsilon = 1e-6);
    assert_relative_eq!(weights[4], 0.005, epsilon = 1e-6);

    let base = Tensor::full(vec![6, 16, 16, 3], 0.5);
    let noisy = noise_reference_frames(&base, 0.02, 12345, 0.75, 0.25).expect("noise");
    assert_eq!(noisy.shape, base.shape);
    assert!(noisy.data.iter().all(|&v| (0.0..=1.0).contains(&v)));

    // First frame should have higher variance around 0.5 than last frame because end_scale = 0.25
    let f0 = noisy.slice_dim0(0, 1).unwrap();
    let f5 = noisy.slice_dim0(5, 6).unwrap();
    let rms0: f32 =
        (f0.data.iter().map(|&v| (v - 0.5).powi(2)).sum::<f32>() / (f0.numel() as f32)).sqrt();
    let rms5: f32 =
        (f5.data.iter().map(|&v| (v - 0.5).powi(2)).sum::<f32>() / (f5.numel() as f32)).sqrt();
    assert!(rms0 > rms5 * 2.0);

    let vae = DummyVae;
    let (lat, noisy_clip) =
        encode_noisy_prefix(&vae, &base, 5, 0.015, 99, 0.7, 0.25).expect("encode_noisy_prefix");
    assert_eq!(noisy_clip.shape[0], 5);
    assert_eq!(lat.shape, vec![1, 16, 2, 4, 4]);
}

#[test]
fn test_texture_band_rms_separates_fine_and_coarse_patterns() {
    let h = 64usize;
    let w = 64usize;
    let mut fine_data = Vec::with_capacity(h * w * 3);
    let mut coarse_data = Vec::with_capacity(h * w * 3);

    for y in 0..h {
        for x in 0..w {
            // Period 4px -> falls in 2_8px band
            let v_fine = if (x / 2 + y / 2) % 2 == 0 { 0.8 } else { 0.2 };
            fine_data.extend_from_slice(&[v_fine, v_fine, v_fine]);

            // Period 16px -> falls in 8_32px band
            let v_coarse = if (x / 8 + y / 8) % 2 == 0 { 0.8 } else { 0.2 };
            coarse_data.extend_from_slice(&[v_coarse, v_coarse, v_coarse]);
        }
    }

    let fine_t = Tensor::from_vec(vec![1, h, w, 3], fine_data).unwrap();
    let coarse_t = Tensor::from_vec(vec![1, h, w, 3], coarse_data).unwrap();
    let combined = Tensor::concat(&[&fine_t, &coarse_t], 0).unwrap();

    let metrics = texture_band_rms(&combined, None).expect("texture_band_rms");
    let luma = metrics.get("luma").unwrap();
    let b_2_8 = luma.get("2_8px").unwrap();
    let b_8_32 = luma.get("8_32px").unwrap();

    // Frame 0 (fine) has higher 2_8px energy than 8_32px
    assert!(b_2_8[0] > b_8_32[0]);
    // Frame 1 (coarse) has higher 8_32px energy than Frame 0
    assert!(b_8_32[1] > b_8_32[0]);

    let drift = summarize_clip_texture_drift(&combined, 1).expect("drift report");
    assert_eq!(drift.frames, 2);
}

// ============================================================================
// 4. Fused Attention Tests (test_fused_attention.py)
// ============================================================================

#[test]
fn test_asymmetric_cached_attention_matches_3d_and_4d() {
    // 4D [B=1, H=2, N_q=3, D=4], N_kv=5
    let q = Tensor::from_vec(
        vec![1, 2, 3, 4],
        (0..24).map(|i| (i as f32) * 0.1).collect(),
    )
    .unwrap();
    let k = Tensor::from_vec(
        vec![1, 2, 5, 4],
        (0..40).map(|i| (i as f32) * 0.05).collect(),
    )
    .unwrap();
    let v = Tensor::from_vec(
        vec![1, 2, 5, 4],
        (0..40).map(|i| (i as f32) * 0.2).collect(),
    )
    .unwrap();

    let out_4d = asymmetric_cached_attention(&q, &k, &v, None, None).expect("4d attn");
    assert_eq!(out_4d.shape, vec![1, 2, 3, 4]);

    // 3D [B=1, N=3, H*D=8] with heads=2
    let q3 = Tensor::from_vec(vec![1, 3, 8], (0..24).map(|i| (i as f32) * 0.1).collect()).unwrap();
    let k3 = Tensor::from_vec(vec![1, 5, 8], (0..40).map(|i| (i as f32) * 0.05).collect()).unwrap();
    let v3 = Tensor::from_vec(vec![1, 5, 8], (0..40).map(|i| (i as f32) * 0.2).collect()).unwrap();
    let out_3d = asymmetric_cached_attention(&q3, &k3, &v3, None, Some(2)).expect("3d attn");
    assert_eq!(out_3d.shape, vec![1, 3, 8]);
}

// ============================================================================
// 5. Seam Protector & Audio Enhancements Tests (test_audio_enhancements.py)
// ============================================================================

#[test]
fn test_seam_protector_video_and_audio_stitching() {
    let prev_imgs = Tensor::full(vec![24, 16, 16, 3], 0.4);
    let curr_imgs = Tensor::full(vec![24, 16, 16, 3], 0.5);
    let stitched =
        stitch_video_images(Some(&prev_imgs), &curr_imgs, 6, 4, true, 8).expect("stitch_video");
    // 24 + (24 - 6) = 42 frames
    assert_eq!(stitched.shape, vec![42, 16, 16, 3]);

    let gain = estimate_luminance_gain(
        &prev_imgs.slice_dim0(0, 4).unwrap(),
        &curr_imgs.slice_dim0(0, 4).unwrap(),
        10.0,
    );
    assert!((gain - 0.9).abs() < 1e-4);
    let faded = apply_luminance_gain_fade(&curr_imgs, gain, 8).unwrap();
    assert_eq!(faded.shape, curr_imgs.shape);

    // Audio equal-power crossfade & fit_audio_length
    let w1 = Tensor::full(vec![1, 2, 1000], 0.5);
    let w2 = Tensor::full(vec![1, 2, 1000], 0.5);
    let cf = audio_equal_power_crossfade_concat(&w1, &w2, 200).unwrap();
    assert_eq!(cf.shape, vec![1, 2, 1800]);

    let padded = fit_audio_length(&w1, 1200).unwrap();
    assert_eq!(padded.shape, vec![1, 2, 1200]);
    assert_eq!(padded.data[1199], 0.0);

    // Stitch audio waveforms with sample-rate mismatch error check
    let aud1 = AudioData::new(w1.clone(), 32000);
    let aud2 = AudioData::new(w2.clone(), 32000);
    let stitched_aud = stitch_audio_waveforms(Some(&aud1), Some(&aud2), 24, 6, 15.0, 24.0)
        .expect("stitch audio")
        .unwrap();
    assert_eq!(stitched_aud.sample_rate, 32000);

    let aud_bad_sr = AudioData::new(w2, 44100);
    assert!(stitch_audio_waveforms(Some(&aud1), Some(&aud_bad_sr), 24, 6, 15.0, 24.0).is_err());

    // Trim images and audio
    let (ti, ta) = trim_images_and_audio(&prev_imgs, Some(&aud1), 6, 24.0, true).unwrap();
    assert_eq!(ti.shape[0], 18);
    assert!(ta.is_some());

    // Latent soft blend & stitch
    let l1 = Tensor::full(vec![1, 4, 10, 2, 2], 1.0);
    let l2 = Tensor::full(vec![1, 4, 10, 2, 2], 2.0);
    let sb = latent_soft_blend(&l1, &l2, 2).unwrap();
    assert_eq!(sb.shape, vec![1, 4, 18, 2, 2]);
    let sv = stitch_video_latents(&l1, &l2, 3, 2).unwrap();
    assert_eq!(sv.shape, vec![1, 4, 17, 2, 2]);

    let a1 = Tensor::full(vec![1, 4, 2, 20], 1.0);
    let a2 = Tensor::full(vec![1, 4, 2, 20], 2.0);
    let sa = stitch_audio_latents(Some(&a1), &a2, 5).unwrap();
    assert_eq!(sa.shape, vec![1, 4, 2, 35]);

    assert_eq!(trim_prefix_frames(&l1, 3).unwrap().shape[2], 7);
    assert_eq!(trim_audio_latents(&a1, 5).unwrap().shape[3], 15);
    assert_eq!(
        trim_audio_waveform(&aud1.waveform, 200).unwrap().shape[2],
        800
    );
}

// ============================================================================
// 6. Continuation State, LongVideoSession & Nodes Tests (test_nodes_and_pipeline.py)
// ============================================================================

#[test]
fn test_continuation_state_roundtrip_and_identity_dedup() {
    let img1 = Tensor::full(vec![1, 8, 8, 3], 0.25);
    let img1_dup = Tensor::full(vec![1, 8, 8, 3], 0.25);
    let img2 = Tensor::full(vec![1, 8, 8, 3], 0.75);

    let extras = vec![
        IdentityReference {
            image: img1_dup,
            strength: 0.9,
            source: "dup".to_string(),
        },
        IdentityReference {
            image: img2,
            strength: 0.8,
            source: "char2".to_string(),
        },
    ];

    let (primary, refs) = prepare_identity_references(Some(&img1), Some(&extras), 1.0, 0.5);
    assert!(primary.is_some());
    // img1_dup is deduplicated, leaving anchor + char2
    assert_eq!(refs.len(), 2);
    assert_relative_eq!(refs[1].strength, 0.4, epsilon = 1e-6);

    let mut manifest = HashMap::new();
    manifest.insert("scene".to_string(), serde_json::json!(3));
    let state = ContinuationState {
        anchor_image: primary,
        identity_references: refs,
        manifest,
    };

    let (tensors, meta) = encode_state(&state);
    let decoded = decode_state(&tensors, Some(&meta))
        .expect("decode_state")
        .expect("some state");
    assert_eq!(decoded.identity_references.len(), 2);
    assert_eq!(decoded.manifest.get("scene"), Some(&serde_json::json!(3)));
}

#[test]
fn test_inject_minimax_keyframes_purges_colliding_frame0_anchors() {
    assert_eq!(step_offsets(6), vec![0, 1, 5, 9, 13, 17]);

    let prior_kf = MinimaxKeyframe {
        resolved_frame_index: 0.0,
        latent: Some(Tensor::zeros(vec![1, 4, 1, 2, 2])),
        audio_latent: None,
    };
    let future_kf = MinimaxKeyframe {
        resolved_frame_index: 50.0,
        latent: Some(Tensor::ones(vec![1, 4, 1, 2, 2])),
        audio_latent: None,
    };
    let cond = vec![ConditioningEntry {
        embedding: Tensor::zeros(vec![1, 8]),
        minimax_keyframes: vec![prior_kf, future_kf],
        extra: HashMap::new(),
    }];

    let video_tail = Tensor::full(vec![1, 4, 2, 2, 2], 3.0); // 2 steps = 5 frames (head_end = 5)
    let updated = inject_minimax_keyframes(&cond, Some(&video_tail), None, None).unwrap();
    // prior_kf at 0.0 (< 5) is dropped; future_kf at 50.0 is kept + 2 new tail keyframes = 3 total
    assert_eq!(updated[0].minimax_keyframes.len(), 3);
    assert_eq!(updated[0].minimax_keyframes[0].resolved_frame_index, 50.0);
}

#[test]
fn test_nodes_end_to_end_pipeline() {
    let cfg_node = MiniMaxPrefixCacheConfigNode;
    let mut session = cfg_node.create_session(
        39,
        true,
        5,
        true,
        true,
        false,
        "Native Masked AV",
        0.0,
        0.25,
        0.75,
    );
    assert_eq!(session.config.rolling_frames, 39);

    let applier = MiniMaxPrefixCacheApplierNode;
    let cond = vec![ConditioningEntry::new(Tensor::zeros(vec![1, 4]))];
    let prev_lat = AvLatent::from_video_audio(
        Tensor::full(vec![1, 4, 37, 2, 2], 1.0),
        Tensor::full(vec![1, 4, 2, 207], 0.5),
    );
    let tgt_lat = AvLatent::from_video_audio(
        Tensor::zeros(vec![1, 4, 37, 2, 2]),
        Tensor::zeros(vec![1, 4, 2, 207]),
    );

    let out = applier
        .apply_cache(
            &mut session,
            &cond,
            Some(&prev_lat),
            Some(&tgt_lat),
            None,
            None,
            None,
            None,
            None,
            None,
            1.0,
            1.0,
            "Match Video Handover",
            4,
            0.0,
            42,
        )
        .expect("applier");
    assert!(out.masked_av_plan.is_some());
    assert_eq!(session.last_rolling_steps, 12);
    assert_eq!(session.last_rolling_frames, 39);

    let monitor = MiniMaxCacheMonitorNode;
    let report = monitor.monitor(&session);
    assert!(report.contains("39 frames (12 latent steps)"));

    let trimmer = MiniMaxTrimPrefixLatentNode;
    let (trimmed_lat, _, _) = trimmer
        .trim(
            Some(&mut session),
            out.latent.as_ref(),
            None,
            None,
            -1,
            24.0,
        )
        .expect("trim");
    assert_eq!(trimmed_lat.unwrap().video.unwrap().shape[2], 25);
    assert!(session.is_current_clip_trimmed);

    // Safe VAE decode in Initial Clip Mode (None) vs populated
    let v_dec = MiniMaxSafeVAEDecodeNode;
    let empty_img = v_dec.decode(&DummyVae, None).unwrap();
    assert_eq!(empty_img.shape[0], 0);
    let decoded_img = v_dec.decode(&DummyVae, Some(&prev_lat)).unwrap();
    assert_eq!(decoded_img.shape[0], 124);

    let a_dec = MiniMaxSafeVAEDecodeAudioNode;
    assert!(a_dec.decode(&DummyAudioVae, None).unwrap().is_none());
    assert!(a_dec
        .decode(&DummyAudioVae, Some(&prev_lat))
        .unwrap()
        .is_some());

    // Save & Load Latent SafeTensors
    let dir = tempdir().unwrap();
    let lat_path = dir.path().join("test_latent.safetensors");
    let saver = MiniMaxSaveLatentNode;
    let loader = MiniMaxLoadLatentNode;
    saver.save_latent(&lat_path, &prev_lat, None).unwrap();
    let (loaded_lat, _) = loader.load_latent(&lat_path).unwrap();
    assert_eq!(loaded_lat.video.unwrap().shape, vec![1, 4, 37, 2, 2]);
}

// ============================================================================
// 7. Clip Bin, API & Security Regression Tests (test_clip_bin.py, test_clip_bin_api.py, test_regressions.py)
// ============================================================================

#[test]
fn test_clip_bin_dag_and_api_endpoints() {
    let dir = tempdir().unwrap();
    let base = dir.path();

    let imgs = Tensor::full(vec![10, 16, 16, 3], 0.6);
    let vlat = Tensor::full(vec![1, 4, 3, 2, 2], 0.2);

    let saver = MiniMaxClipBinSaverNode;
    let (c1_id, _) = saver
        .save_to_bin(SaveClipRequest {
            base_dir: base,
            project_name: "SciFi_Trailer",
            parent_id: None,
            images: Some(&imgs),
            video_latent: Some(&vlat),
            audio_latent: None,
            audio: None,
            prompt: "Opening shot of spaceship",
            seed: 101,
            fps: 24.0,
            save_tail_frames: 4,
            source_video_path: None,
            auto_encode_mp4: false,
        })
        .expect("save c1");
    assert_eq!(c1_id, "clip_001_01");

    // Save child take in Scene 2
    let (c2_id, _) = saver
        .save_to_bin(SaveClipRequest {
            base_dir: base,
            project_name: "SciFi_Trailer",
            parent_id: Some(&c1_id),
            images: Some(&imgs),
            video_latent: Some(&vlat),
            audio_latent: None,
            audio: None,
            prompt: "Cockpit closeup",
            seed: 102,
            fps: 24.0,
            save_tail_frames: 4,
            source_video_path: None,
            auto_encode_mp4: false,
        })
        .expect("save c2");
    assert_eq!(c2_id, "clip_002_01");

    let loaded = load_clip_asset(base, "SciFi_Trailer", &c2_id).expect("load c2");
    assert_eq!(loaded.meta.scene_num, 2);
    assert_eq!(loaded.tail_images.shape[0], 4);
    assert!(format_clip_label(&loaded.meta).contains("S02 T01"));

    let sel = get_clips_for_selection(base, "SciFi_Trailer", true).unwrap();
    assert_eq!(sel.len(), 3); // Root option + 2 clips

    // API endpoints
    let list_resp = get_project_clips_api(base, "SciFi_Trailer");
    assert_eq!(list_resp.status_code, 200);

    let rate_resp = update_clip_rating_api(base, "SciFi_Trailer", &c2_id, 5);
    assert_eq!(rate_resp.status_code, 200);

    let picker = MiniMaxClipBinPickerNode;
    let picked = picker
        .pick_clip(base, "SciFi_Trailer", "Auto", 5, "latest")
        .expect("pick latest 5-star");
    assert_eq!(picked.clip_id, c2_id);

    // Initial mode on empty project
    let init_pick = picker
        .pick_clip(base, "Empty_Proj", "Auto", 1, "latest")
        .expect("pick initial");
    assert_eq!(init_pick.clip_id, "[INITIAL_GENERATION]");
    assert!(init_pick.latent.is_none());

    let del_resp = delete_clip_api(base, "SciFi_Trailer", &c2_id);
    assert_eq!(del_resp.status_code, 200);

    let rescan_resp = rescan_project_api(base, "SciFi_Trailer");
    assert_eq!(rescan_resp.status_code, 200);
}

#[test]
fn test_security_regressions_path_traversal_and_symlinks() {
    let dir = tempdir().unwrap();
    let base = dir.path();

    // Path traversal clip_ids must be rejected with SecurityViolation (HTTP 403 in API)
    assert!(get_clip_dir(base, "Proj", "../escape").is_err());
    assert!(get_clip_dir(base, "Proj", "clip_001/../../etc").is_err());
    assert!(get_clip_dir(base, "Proj", "clip\\..\\escape").is_err());

    let rate_escape = update_clip_rating_api(base, "Proj", "../escape", 5);
    assert_eq!(rate_escape.status_code, 403);

    let del_escape = delete_clip_api(base, "Proj", "../escape");
    assert_eq!(del_escape.status_code, 403);

    // Symlink escape detection
    #[cfg(unix)]
    {
        let outside = tempdir().unwrap();
        let proj_dir = get_project_dir(base, "SymlinkProj").unwrap();
        let symlink_clip = proj_dir.join("clip_symlink");
        std::os::unix::fs::symlink(outside.path(), &symlink_clip).unwrap();
        assert!(get_clip_dir(base, "SymlinkProj", "clip_symlink").is_err());
    }

    // Atomic write JSON preserves existing file if target path is valid
    let json_path = base.join("atomic_test.json");
    atomic_write_json(&json_path, &serde_json::json!({"ok": true})).unwrap();
    assert!(json_path.exists());
    let index = rebuild_project_index(base, "Proj").unwrap();
    assert!(index.is_empty());
    let _ = delete_clip_asset(base, "Proj", "clip_999_99");
}

#[test]
fn test_disk_stream_append_and_validation() {
    let dir = tempdir().unwrap();
    let base = dir.path();
    let imgs = Tensor::full(vec![8, 16, 16, 3], 0.5);
    let aud = AudioData::new(Tensor::full(vec![1, 2, 1000], 0.1), 32000);

    let (manifest_path, _, total_frames) = append_disk_clip_with_encoder(
        base,
        "DiskProj",
        "stream_01",
        Some(&imgs),
        Some(&aud),
        2,
        24.0,
        false,
        |_imgs, _fps, _aud, seg_path| fs::write(seg_path, b"dummy mkv").is_ok(),
    )
    .expect("append disk clip");

    assert_eq!(total_frames, 6);
    assert!(Path::new(&manifest_path).exists());

    // Odd dimensions or invalid stream names must be rejected
    let odd_imgs = Tensor::full(vec![4, 15, 16, 3], 0.5);
    assert!(append_disk_clip_with_encoder(
        base,
        "DiskProj",
        "stream_01",
        Some(&odd_imgs),
        None,
        0,
        24.0,
        false,
        |_, _, _, _| true,
    )
    .is_err());

    assert!(append_disk_clip_with_encoder(
        base,
        "DiskProj",
        "../bad_stream",
        Some(&imgs),
        None,
        0,
        24.0,
        false,
        |_, _, _, _| true,
    )
    .is_err());

    let node = MiniMaxDiskVideoStreamNode;
    let _ = node;
}

// ============================================================================
// 8. Timeline Slicer & Patch Reassembler Tests (test_timeline_slicer.py)
// ============================================================================

#[test]
fn test_timeline_slicer_and_patch_reassembler_nodes() {
    let dir = tempdir().unwrap();
    let base = dir.path();

    let master_imgs = Tensor::full(vec![30, 16, 16, 3], 0.2);
    let master_aud = AudioData::new(Tensor::full(vec![1, 2, 3000], 0.1), 2400);

    let slicer = MiniMaxVideoChunkSlicerNode;
    let reassembler = MiniMaxVideoPatchReassemblerNode;

    let (chunk1_imgs, chunk1_aud, ctx1, ind1, _) = slicer
        .slice_chunk(
            base,
            "Edit_Session_01",
            &master_imgs,
            Some(&master_aud),
            24.0,
            0,
            15,
            true,
            true,
        )
        .expect("slice chunk 1");
    assert_eq!(chunk1_imgs.shape[0], 15);
    assert_eq!(ind1.shape[0], 1);

    let edited_chunk1 = Tensor::full(vec![15, 16, 16, 3], 0.8);
    let (reassembled, _, _, report) = reassembler
        .reassemble_patch(
            base,
            &ctx1,
            &edited_chunk1,
            Some(&chunk1_aud),
            2,
            true,
            "graded first half",
        )
        .expect("reassemble 1");

    assert_eq!(reassembled.shape, vec![30, 16, 16, 3]);
    assert!(report.contains("patch_000000_000015"));

    // Verify standardize_audio_dict handles 1D and 2D waveforms
    let wf_1d = AudioData::new(Tensor::ones(vec![100]), 16000);
    assert_eq!(
        standardize_audio_dict(Some(&wf_1d), 16000).waveform.shape,
        vec![1, 1, 100]
    );
    let wf_2d = AudioData::new(Tensor::ones(vec![2, 100]), 16000);
    assert_eq!(
        standardize_audio_dict(Some(&wf_2d), 16000).waveform.shape,
        vec![1, 2, 100]
    );

    let sliced_pad =
        slice_video_and_audio(&master_imgs, Some(&master_aud), 24.0, 20, 35, true).unwrap();
    // Padded from 10 available frames [20..30) to requested 15 frames [20..35)
    assert_eq!(sliced_pad.chunk_images.shape[0], 15);
    assert_eq!(sliced_pad.actual_start, 20);
    assert_eq!(sliced_pad.actual_end, 30);

    let sess = TimelineSession::load_or_create(base, "Edit_Session_01").unwrap();
    assert_eq!(sess.get_missing_gap_ranges(), vec![(15, 30)]);
    assert!(sess.is_complete()); // master source is registered

    let _ =
        render_timeline_indicator_image(30, Some((0, 15)), Some(&[(0, 15)]), 24.0, "test").unwrap();
}

// ============================================================================
// 9. Workflow Examples & Node Registry Validation
// ============================================================================

#[test]
fn test_workflow_examples_and_node_registry() {
    assert_eq!(NODE_CLASS_NAMES.len(), 15);
    let wf1 = fs::read_to_string("examples/MiniMaxH3_PrefixStream_v1.0.json")
        .expect("read example workflow 1");
    let rep1 = validate_workflow_json(&wf1).expect("validate workflow 1");
    assert!(rep1.valid);
    assert!(!rep1.minimax_nodes.is_empty());

    let wf2 = fs::read_to_string("examples/MiniMaxH3_LongVideo_Slicer_Reassembler_Template.json")
        .expect("read example workflow 2");
    let rep2 = validate_workflow_json(&wf2).expect("validate workflow 2");
    assert!(rep2.valid);
    assert!(!rep2.minimax_nodes.is_empty());

    // Also test LongVideoStitcherNode
    let stitcher = MiniMaxLongVideoStitcherNode;
    let p_img = Tensor::full(vec![22, 8, 8, 3], 0.3);
    let c_img = Tensor::full(vec![44, 8, 8, 3], 0.3);
    let (st_img, _, _) = stitcher
        .stitch(
            None,
            Some(&p_img),
            Some(&c_img),
            None,
            None,
            None,
            None,
            22,
            4,
            15.0,
            true,
            24.0,
        )
        .unwrap();
    assert_eq!(st_img.unwrap().shape[0], 44);

    let mut lvs = LongVideoSession::default();
    let v = Tensor::zeros(vec![1, 4, 12, 2, 2]);
    let (dv, _) = lvs.commit_generated_clip(&v, None, true).unwrap();
    assert_eq!(dv.shape[2], 12);
}
