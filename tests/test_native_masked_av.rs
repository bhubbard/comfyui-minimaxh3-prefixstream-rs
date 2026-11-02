//! Rust port of `tests/test_native_masked_av.py`.

use std::collections::HashMap;
use std::f32::consts::PI;

use minimaxh3_prefixstream::engine::cache_manager::KVCacheConfig;
use minimaxh3_prefixstream::engine::continuation_state::{
    decode_state, encode_state, prepare_identity_references, ContinuationState,
};
use minimaxh3_prefixstream::nodes::{
    MiniMaxPrefixCacheApplierNode, MiniMaxPrefixCacheConfigNode, MiniMaxTrimPrefixLatentNode,
};
use minimaxh3_prefixstream::pipeline::long_video_director::LongVideoSession;
use minimaxh3_prefixstream::pipeline::native_masked_av::{
    apply_native_masked_av, is_exact_masked_av_context, latent_boundaries, masked_av_tail_slice,
    snap_masked_av_context_length, MASKED_AV_CONTEXTS,
};
use minimaxh3_prefixstream::tensor::{
    AvLatent, ConditioningEntry, IdentityReference, MinimaxKeyframe, Tensor,
};
use serde_json::json;

#[test]
fn test_all_native_images_survive_missing_references_without_growth() {
    let ref_a = Tensor::full([1, 8, 8, 3], 0.25);
    let ref_b = Tensor::full([1, 8, 8, 3], 0.75);
    let refs = vec![
        IdentityReference {
            image: ref_a.clone(),
            strength: 1.0,
            source: "native_a".to_string(),
        },
        IdentityReference {
            image: ref_b.clone(),
            strength: 1.0,
            source: "native_b".to_string(),
        },
    ];

    let (anchor, prepared) = prepare_identity_references(Some(&ref_a), Some(&refs), 1.0, 1.0);
    assert!(anchor.unwrap().allclose(&ref_a, 1e-6));
    assert_eq!(prepared.len(), 2);

    // Repeating with the same references must deduplicate without growing
    let mut dup_refs = prepared.clone();
    dup_refs.extend(refs);
    let (_, prepared_again) = prepare_identity_references(Some(&ref_a), Some(&dup_refs), 1.0, 1.0);
    assert_eq!(prepared_again.len(), 2);
}

#[test]
fn test_soft_video_context_is_opt_in_and_leaves_audio_independent() {
    let src_v = Tensor::ones([1, 4, 12, 4, 4]);
    let src_a = Tensor::ones([1, 4, 2, 65]).scale(2.0);
    let tgt_v = Tensor::zeros([1, 4, 17, 4, 4]);
    let tgt_a = Tensor::zeros([1, 4, 2, 92]);

    let res = apply_native_masked_av(
        &tgt_v,
        &tgt_a,
        &src_v,
        &src_a,
        39,
        "Match Video Handover",
        0,
        0,
        0,
        0.04,
    )
    .unwrap();

    assert!((res.plan.video_context_denoise - 0.04).abs() < 1e-6);
    let v_head = res.video_mask.slice_axis(2, 0, 12).unwrap();
    let v_tail = res.video_mask.slice_axis(2, 12, 17).unwrap();
    let a_head = res.audio_mask.slice_axis(3, 0, 65).unwrap();
    let a_tail = res.audio_mask.slice_axis(3, 65, 92).unwrap();

    assert!(v_head.allclose(&Tensor::full([1, 1, 12, 4, 4], 0.04), 1e-6));
    assert!(v_tail.allclose(&Tensor::ones([1, 1, 5, 4, 4]), 1e-6));
    assert!(a_head.allclose(&Tensor::zeros([1, 1, 2, 65]), 1e-6));
    assert!(a_tail.allclose(&Tensor::ones([1, 1, 2, 27]), 1e-6));
}

#[test]
fn test_continuation_model_and_soft_mask_reach_native_node_output() {
    let (_, mut sess) = MiniMaxPrefixCacheConfigNode.configure(
        "Native Masked AV",
        22,
        5,
        true,
        true,
        0.0,
        0.5,
        0.0,
    );
    let prev =
        AvLatent::from_video_audio(Tensor::ones([1, 4, 12, 4, 4]), Tensor::ones([1, 4, 2, 65]));
    let target = AvLatent::from_video_audio(
        Tensor::zeros([1, 4, 12, 4, 4]),
        Tensor::zeros([1, 4, 2, 65]),
    );

    let out = MiniMaxPrefixCacheApplierNode
        .apply_cache(
            &mut sess,
            &[],
            Some(&prev),
            Some(&target),
            None,
            None,
            None,
            None,
            None,
            None,
            1.0,
            1.0,
            "Match Video Handover",
            0,
            0.06,
            0,
        )
        .unwrap();

    let masked = out.latent.unwrap();
    let v_mask = masked.video_mask.unwrap();
    let a_mask = masked.audio_mask.unwrap();
    assert!(v_mask
        .slice_axis(2, 0, 7)
        .unwrap()
        .allclose(&Tensor::full([1, 1, 7, 4, 4], 0.06), 1e-6));
    assert!(v_mask
        .slice_axis(2, 7, 12)
        .unwrap()
        .allclose(&Tensor::ones([1, 1, 5, 4, 4]), 1e-6));
    assert!(a_mask
        .slice_axis(3, 0, 37)
        .unwrap()
        .allclose(&Tensor::zeros([1, 1, 2, 37]), 1e-6));
}

#[test]
fn test_trimmed_context_matches_full_context_without_tail_loss() {
    let mut v_seq = Vec::with_capacity(37);
    for i in 0..37 {
        v_seq.push((i + 1) as f32);
    }
    let full_v = Tensor::from_vec(vec![1, 1, 37, 1, 1], v_seq).unwrap();

    let mut a_seq = Vec::with_capacity(207);
    for i in 0..207 {
        a_seq.push((i + 100) as f32);
    }
    let full_a = Tensor::from_vec(vec![1, 1, 1, 207], a_seq).unwrap();
    let full_latent = AvLatent::from_video_audio(full_v.clone(), full_a.clone());

    let trimmed = MiniMaxTrimPrefixLatentNode
        .trim_frames(39, None, None, Some(&full_latent), None, None, 24.0, true)
        .unwrap()
        .latent
        .unwrap();

    assert_eq!(trimmed.metadata["minimax_source_start_step"], json!(12));
    assert_eq!(
        trimmed.metadata["minimax_source_audio_start_tick"],
        json!(65)
    );

    let target_v = Tensor::zeros([1, 1, 37, 1, 1]);
    let target_a = Tensor::full([1, 1, 1, 207], -1.0);

    for frames in [22, 39, 56, 73] {
        let from_full = apply_native_masked_av(
            &target_v,
            &target_a,
            &full_v,
            &full_a,
            frames,
            "Match Video Handover",
            0,
            0,
            0,
            0.0,
        )
        .unwrap();

        let from_trimmed = apply_native_masked_av(
            &target_v,
            &target_a,
            trimmed.video.as_ref().unwrap(),
            trimmed.audio.as_ref().unwrap(),
            frames,
            "Match Video Handover",
            0,
            12,
            65,
            0.0,
        )
        .unwrap();

        assert!(from_trimmed
            .video_latent
            .allclose(&from_full.video_latent, 1e-6));
        assert!(from_trimmed
            .audio_latent
            .allclose(&from_full.audio_latent, 1e-6));
        assert!(from_trimmed
            .video_mask
            .allclose(&from_full.video_mask, 1e-6));
        assert!(from_trimmed
            .audio_mask
            .allclose(&from_full.audio_mask, 1e-6));
        assert_eq!(from_trimmed.plan.ignored_tail_frames, 0);
    }
}

#[test]
fn test_repeated_latent_trims_track_actual_boundaries() {
    let v = Tensor::from_vec(vec![1, 1, 37, 1, 1], (0..37).map(|i| i as f32).collect()).unwrap();
    let a = Tensor::from_vec(vec![1, 1, 1, 207], (0..207).map(|i| i as f32).collect()).unwrap();
    let full = AvLatent::from_video_audio(v.clone(), a.clone());

    let trim_node = MiniMaxTrimPrefixLatentNode;
    let first = trim_node
        .trim_frames(39, None, None, Some(&full), None, None, 24.0, true)
        .unwrap()
        .latent
        .unwrap();
    let second = trim_node
        .trim_frames(22, None, None, Some(&first), None, None, 24.0, true)
        .unwrap()
        .latent
        .unwrap();

    // Second cut starts at phase 2, where 7 latent steps span 25 frames (total 64 frames -> round(64 * 40 / 24) = 107 audio ticks)
    assert_eq!(second.metadata["minimax_source_start_step"], json!(19));
    assert_eq!(
        second.metadata["minimax_source_audio_start_tick"],
        json!(107)
    );
    assert!(second
        .video
        .as_ref()
        .unwrap()
        .allclose(&v.slice_axis(2, 19, 37).unwrap(), 1e-6));
    assert!(second
        .audio
        .as_ref()
        .unwrap()
        .allclose(&a.slice_axis(3, 107, 207).unwrap(), 1e-6));
}

#[test]
fn test_matches_official_22_frame_cut_concat_masks() {
    let src_v = Tensor::from_vec(
        vec![1, 2, 37, 2, 2],
        (0..(2 * 37 * 2 * 2)).map(|i| i as f32).collect(),
    )
    .unwrap();
    let src_a = Tensor::from_vec(
        vec![1, 2, 3, 207],
        (0..(2 * 3 * 207)).map(|i| i as f32).collect(),
    )
    .unwrap();
    let tgt_v = Tensor::full([1, 2, 37, 2, 2], -1.0);
    let tgt_a = Tensor::full([1, 2, 3, 207], -2.0);

    let res = apply_native_masked_av(
        &tgt_v,
        &tgt_a,
        &src_v,
        &src_a,
        22,
        "Match Video Handover",
        0,
        0,
        0,
        0.0,
    )
    .unwrap();

    assert_eq!(res.plan.start_t, 30);
    assert_eq!(res.plan.end_t, 37);
    assert_eq!(res.plan.context_steps, 7);
    assert_eq!(res.plan.actual_context_frames, 22);
    assert_eq!(res.plan.ignored_tail_frames, 0);
    assert_eq!(res.plan.audio_start_tick, 170);
    assert_eq!(res.plan.audio_end_tick, 207);
    assert_eq!(res.plan.audio_steps, 37);

    assert!(res
        .video_latent
        .slice_axis(2, 0, 7)
        .unwrap()
        .allclose(&src_v.slice_axis(2, 30, 37).unwrap(), 1e-6));
    assert!(res
        .audio_latent
        .slice_axis(3, 0, 37)
        .unwrap()
        .allclose(&src_a.slice_axis(3, 170, 207).unwrap(), 1e-6));
}

#[test]
fn test_short_target_stays_masked_and_rounds_audio_duration_once() {
    let src_v = Tensor::ones([1, 2, 37, 2, 2]);
    let src_a = Tensor::from_vec(vec![1, 1, 1, 207], (0..207).map(|i| i as f32).collect()).unwrap();
    let tgt_v = Tensor::zeros([1, 2, 8, 2, 2]);
    let tgt_a = Tensor::full([1, 1, 1, 42], -1.0);

    let res = apply_native_masked_av(
        &tgt_v,
        &tgt_a,
        &src_v,
        &src_a,
        39,
        "Match Video Handover",
        0,
        0,
        0,
        0.0,
    )
    .unwrap();

    assert_eq!(res.plan.actual_context_frames, 22);
    assert_eq!(res.plan.context_steps, 7);
    assert_eq!(res.plan.audio_steps, 37);

    let res2 = apply_native_masked_av(
        &Tensor::zeros([1, 2, 7, 2, 2]),
        &Tensor::full([1, 1, 1, 37], -1.0),
        &src_v,
        &src_a,
        39,
        "Match Video Handover",
        0,
        0,
        0,
        0.0,
    )
    .unwrap();

    assert_eq!(res2.plan.actual_context_frames, 5);
    assert_eq!(res2.plan.context_steps, 2);
    assert_eq!(res2.plan.audio_steps, 8);
    assert_eq!(res2.plan.audio_start_tick, 199);
    assert_eq!(res2.plan.audio_end_tick, 207);
}

#[test]
fn test_safe_native_phase_alignment_and_commit_use_actual_prefix() {
    let cfg = KVCacheConfig {
        continuation_mode: "Safe Native Keyframes".to_string(),
        rolling_frames: 39,
        anchor_frames: 5,
        ..Default::default()
    };
    let mut sess = LongVideoSession::new(cfg);

    let prev_v = Tensor::randn([1, 4, 35, 2, 2], 77);
    let cond = vec![ConditioningEntry::default()];

    // start=23 advances to 25 (phase-aligned), leaving 10 steps
    let out_cond = sess
        .prepare_next_clip(&cond, Some(&prev_v), None, None, 0)
        .unwrap();

    assert_eq!(sess.last_rolling_steps, 10);
    let kfs = &out_cond[0].minimax_keyframes;
    assert_eq!(kfs.len(), 10);
    assert!(kfs[0]
        .latent
        .as_ref()
        .unwrap()
        .allclose(&prev_v.slice_axis(2, 25, 26).unwrap(), 1e-6));

    let (delivered, _) = sess.commit_generated_clip(&prev_v, None, true).unwrap();
    assert_eq!(delivered.shape[2], 25);
}

#[test]
fn test_default_mode_and_exact_context_grid() {
    let cfg = KVCacheConfig::default();
    assert!(cfg.continuation_mode.contains("Native Masked AV"));
    assert_eq!(cfg.rolling_frames, 39);
    assert_eq!(
        MASKED_AV_CONTEXTS,
        [5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192]
    );
    assert!(is_exact_masked_av_context(39));
    assert!(is_exact_masked_av_context(90));
    assert!(is_exact_masked_av_context(141));
    assert!(is_exact_masked_av_context(192));
    assert!(!is_exact_masked_av_context(22));
    assert!(!is_exact_masked_av_context(56));
    assert_eq!(snap_masked_av_context_length(40, 124, 124).unwrap(), 39);
    assert_eq!(snap_masked_av_context_length(39, 30, 124).unwrap(), 22);
    assert_eq!(snap_masked_av_context_length(39, 124, 39).unwrap(), 22);
}

#[test]
fn test_canonical_tail_slice() {
    assert_eq!(
        latent_boundaries(12, 0),
        vec![0, 1, 5, 9, 13, 17, 18, 22, 26, 30, 34, 35, 39]
    );
    let plan = masked_av_tail_slice(37, 39, 37, 0).unwrap();
    assert_eq!(plan.context_steps, 12);
    assert_eq!(plan.actual_context_frames, 39);
    assert_eq!(plan.start_t % 5, 0);
    assert_eq!(plan.end_t % 5, 2);
    assert_eq!(plan.ignored_tail_frames, 0);
}

#[test]
fn test_copies_av_and_builds_independent_masks() {
    let src_v = Tensor::from_vec(
        vec![1, 2, 12, 2, 2],
        (0..(2 * 12 * 2 * 2)).map(|i| i as f32).collect(),
    )
    .unwrap();
    let src_a = Tensor::from_vec(
        vec![1, 2, 3, 65],
        (0..(2 * 3 * 65)).map(|i| (i + 1000) as f32).collect(),
    )
    .unwrap();
    let tgt_v = Tensor::full([1, 2, 17, 2, 2], -1.0);
    let tgt_a = Tensor::full([1, 2, 3, 92], -2.0);

    let res = apply_native_masked_av(
        &tgt_v,
        &tgt_a,
        &src_v,
        &src_a,
        39,
        "Match Video Handover",
        0,
        0,
        0,
        0.0,
    )
    .unwrap();

    assert_eq!(res.plan.context_steps, 12);
    assert_eq!(res.plan.audio_steps, 65);
    assert!(res
        .video_latent
        .slice_axis(2, 0, 12)
        .unwrap()
        .allclose(&src_v, 1e-6));
    assert!(res
        .video_latent
        .slice_axis(2, 12, 17)
        .unwrap()
        .allclose(&Tensor::full([1, 2, 5, 2, 2], -1.0), 1e-6));
    assert!(res
        .audio_latent
        .slice_axis(3, 0, 65)
        .unwrap()
        .allclose(&src_a, 1e-6));
    assert!(res
        .audio_latent
        .slice_axis(3, 65, 92)
        .unwrap()
        .allclose(&Tensor::full([1, 2, 3, 27], -2.0), 1e-6));
}

#[test]
fn test_audio_feather_releases_only_audio_tail() {
    let src_v = Tensor::ones([1, 2, 12, 2, 2]);
    let src_a = Tensor::ones([1, 2, 3, 65]);
    let tgt_v = Tensor::zeros([1, 2, 17, 2, 2]);
    let tgt_a = Tensor::zeros([1, 2, 3, 92]);

    let res = apply_native_masked_av(
        &tgt_v,
        &tgt_a,
        &src_v,
        &src_a,
        39,
        "Match Video Handover",
        4,
        0,
        0,
        0.0,
    )
    .unwrap();

    assert!(res
        .video_mask
        .slice_axis(2, 0, 12)
        .unwrap()
        .allclose(&Tensor::zeros([1, 1, 12, 2, 2]), 1e-6));
    assert!(res
        .audio_mask
        .slice_axis(3, 0, 61)
        .unwrap()
        .allclose(&Tensor::zeros([1, 1, 3, 61]), 1e-6));

    let ramp = res.audio_mask.slice_axis(3, 61, 65).unwrap();
    for i in 1..=4 {
        let expected = 0.5 - 0.5 * (PI * (i as f32) / 4.0).cos();
        assert!((ramp.data[i - 1] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_applier_outputs_masked_latent_and_removes_head_keyframe() {
    let cfg = KVCacheConfig {
        continuation_mode: "Native Masked AV".to_string(),
        rolling_frames: 39,
        ..Default::default()
    };
    let mut sess = LongVideoSession::new(cfg);

    let prev = AvLatent::from_video_audio(
        Tensor::ones([1, 2, 12, 2, 2]).scale(3.0),
        Tensor::ones([1, 2, 3, 65]).scale(4.0),
    );
    let target = AvLatent::from_video_audio(
        Tensor::zeros([1, 2, 17, 2, 2]),
        Tensor::zeros([1, 2, 3, 92]),
    );

    let entry = ConditioningEntry {
        minimax_keyframes: vec![
            MinimaxKeyframe {
                resolved_frame_index: 0.0,
                latent: Some(Tensor::ones([1, 2, 1, 2, 2])),
                audio_latent: None,
            },
            MinimaxKeyframe {
                resolved_frame_index: 48.0,
                latent: Some(Tensor::ones([1, 2, 1, 2, 2])),
                audio_latent: None,
            },
        ],
        ..Default::default()
    };

    let out = MiniMaxPrefixCacheApplierNode
        .apply_cache(
            &mut sess,
            &[entry],
            Some(&prev),
            Some(&target),
            None,
            None,
            None,
            None,
            None,
            None,
            1.0,
            1.0,
            "Match Video Handover",
            0,
            0.0,
            0,
        )
        .unwrap();

    assert_eq!(sess.last_rolling_frames, 39);
    assert_eq!(sess.last_rolling_steps, 12);
    assert_eq!(out.conditioning[0].minimax_keyframes.len(), 1);
    assert!((out.conditioning[0].minimax_keyframes[0].resolved_frame_index - 48.0).abs() < 1e-6);
}

#[test]
fn test_applier_falls_back_for_a_short_target() {
    let cfg = KVCacheConfig {
        continuation_mode: "Native Masked AV".to_string(),
        rolling_frames: 39,
        ..Default::default()
    };
    let mut sess = LongVideoSession::new(cfg);

    let prev = AvLatent::from_video_audio(
        Tensor::ones([1, 2, 12, 2, 2]).scale(3.0),
        Tensor::ones([1, 2, 3, 65]).scale(4.0),
    );
    let short_target =
        AvLatent::from_video_audio(Tensor::zeros([1, 2, 2, 2, 2]), Tensor::zeros([1, 2, 3, 9]));
    let cond = vec![ConditioningEntry::default()];

    let out = MiniMaxPrefixCacheApplierNode
        .apply_cache(
            &mut sess,
            &cond,
            Some(&prev),
            Some(&short_target),
            None,
            None,
            None,
            None,
            None,
            None,
            1.0,
            1.0,
            "Match Video Handover",
            0,
            0.0,
            0,
        )
        .unwrap();

    assert!(out.masked_av_plan.is_none());
    assert!(!out.conditioning[0].minimax_keyframes.is_empty());

    // Also verify ContinuationState roundtrip
    let anchor = Tensor::full([1, 8, 8, 3], 0.5);
    let mut manifest = HashMap::new();
    manifest.insert("video_start_step".to_string(), json!(12));
    let state = ContinuationState {
        anchor_image: Some(anchor.clone()),
        identity_references: vec![IdentityReference {
            image: anchor.clone(),
            strength: 1.0,
            source: "anchor_image".to_string(),
        }],
        manifest,
    };
    let (tensors, meta) = encode_state(&state);
    let decoded = decode_state(&tensors, Some(&meta)).unwrap().unwrap();
    assert!(decoded.anchor_image.unwrap().allclose(&anchor, 1e-6));
    assert_eq!(decoded.manifest["video_start_step"], json!(12));
}
