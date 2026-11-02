//! Rust port of `tests/test_context_noise.py`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use minimaxh3_prefixstream::engine::cache_manager::{latent_steps_to_pixel_frames, KVCacheConfig};
use minimaxh3_prefixstream::error::Result;
use minimaxh3_prefixstream::nodes::{MiniMaxPrefixCacheApplierNode, MiniMaxPrefixCacheConfigNode};
use minimaxh3_prefixstream::pipeline::context_noise::{
    encode_noisy_prefix_with_plan, noise_reference_frames, reference_noise_cosine_strengths,
};
use minimaxh3_prefixstream::pipeline::long_video_director::LongVideoSession;
use minimaxh3_prefixstream::pipeline::native_masked_av::{
    apply_native_masked_av, MASKED_AV_CONTEXTS,
};
use minimaxh3_prefixstream::tensor::{
    AvLatent, ConditioningEntry, IdentityReference, Tensor, VideoVae,
};
use serde_json::json;

struct BiasedTestVae {
    bias: f32,
    decode_calls: Arc<AtomicUsize>,
    encode_calls: Arc<AtomicUsize>,
}

impl BiasedTestVae {
    fn new(bias: f32) -> Self {
        Self {
            bias,
            decode_calls: Arc::new(AtomicUsize::new(0)),
            encode_calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl VideoVae for BiasedTestVae {
    fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        self.decode_calls.fetch_add(1, Ordering::SeqCst);
        let frames = latent_steps_to_pixel_frames(latent.shape[2]);
        let mut out = vec![0.0_f32; frames * 4 * 4 * 3];
        let mut cursor = 0usize;
        for step in 0..latent.shape[2] {
            let count = if step % 5 == 0 { 1 } else { 4 };
            let c_dim = latent.shape[1];
            let t_dim = latent.shape[2];
            let h_dim = latent.shape[3];
            let w_dim = latent.shape[4];
            let mut sum = 0.0_f32;
            let mut n = 0usize;
            for c in 0..c_dim {
                for h in 0..h_dim {
                    for w in 0..w_dim {
                        let idx = (((c * t_dim) + step) * h_dim + h) * w_dim + w;
                        sum += latent.data[idx];
                        n += 1;
                    }
                }
            }
            let val = (sum / (n as f32)).clamp(0.0, 1.0);
            for offset in 0..count {
                let f = cursor + offset;
                for pix in 0..(4 * 4 * 3) {
                    out[f * (4 * 4 * 3) + pix] = val;
                }
            }
            cursor += count;
        }
        Tensor::from_vec(vec![frames, 4, 4, 3], out)
    }

    fn encode(&self, images: &Tensor) -> Result<Tensor> {
        self.encode_calls.fetch_add(1, Ordering::SeqCst);
        let frames = images.shape[0];
        assert!(
            frames >= 5 && (frames - 5).is_multiple_of(17),
            "Expected canonical 17k+5 frame window, got {}",
            frames
        );
        let steps = 2 + 5 * ((frames - 5) / 17);
        let mut out = vec![0.0_f32; 2 * steps * 2 * 2];
        let mut cursor = 0usize;
        for step in 0..steps {
            let count = if step % 5 == 0 { 1 } else { 4 };
            let slice = images.slice_dim0(cursor, cursor + count)?;
            let mean = slice.mean() + self.bias;
            for c in 0..2 {
                for h in 0..2 {
                    for w in 0..2 {
                        let idx = (((c * steps) + step) * 2 + h) * 2 + w;
                        out[idx] = mean;
                    }
                }
            }
            cursor += count;
        }
        Tensor::from_vec(vec![1, 2, steps, 2, 2], out)
    }
}

#[test]
fn test_recipe_reproducible_without_mutating_images_or_global_rng() {
    let images = Tensor::full([12, 8, 8, 4], 0.5);
    let before = images.clone();

    let first = noise_reference_frames(&images, 0.4, 0, 0.0, 0.0).unwrap();
    let second = noise_reference_frames(&images, 0.4, 0, 0.0, 0.0).unwrap();

    assert!(images.allclose(&before, 1e-7));
    assert!(first.allclose(&second, 1e-7));

    // Alpha channel (channel 3) must remain untouched (0.5)
    let alpha = first.slice_axis(3, 3, 4).unwrap();
    assert!(alpha.allclose(&before.slice_axis(3, 3, 4).unwrap(), 1e-7));

    // Final pixel frame [-1] must remain untouched
    assert!(first
        .slice_dim0(11, 12)
        .unwrap()
        .allclose(&before.slice_dim0(11, 12).unwrap(), 1e-7));

    // Earlier frames must be modified and bounded in [0, 1]
    assert!(!first
        .slice_dim0(0, 11)
        .unwrap()
        .allclose(&before.slice_dim0(0, 11).unwrap(), 1e-5));
    assert!(first.data.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_every_context_has_monotone_noise_and_clean_tail() {
    for &count in &MASKED_AV_CONTEXTS {
        let profile = reference_noise_cosine_strengths(count);
        assert_eq!(profile.len(), count);
        assert!((profile[0] - 0.4).abs() < 1e-6);
        assert!(profile[count - 1].abs() < 1e-6);
        for pair in profile.windows(2) {
            assert!(
                pair[0] > pair[1],
                "Expected strict monotone decrease: {:?}",
                pair
            );
        }
    }
}

#[test]
fn test_five_frame_context_noises_first_four_and_preserves_final_pixel_frame() {
    let profile = reference_noise_cosine_strengths(5);
    let expected = [0.4_f32, 0.341_421_36, 0.2, 0.058_578_644, 0.0];
    for (p, e) in profile.iter().zip(expected.iter()) {
        assert!((p - e).abs() < 1e-5, "got {}, expected {}", p, e);
    }

    let images = Tensor::full([5, 8, 8, 3], 0.5);
    let noisy = noise_reference_frames(&images, 0.4, 0, 0.0, 0.0).unwrap();
    for frame_idx in 0..4 {
        let f_slice = noisy.slice_dim0(frame_idx, frame_idx + 1).unwrap();
        let orig_slice = images.slice_dim0(frame_idx, frame_idx + 1).unwrap();
        assert!(!f_slice.allclose(&orig_slice, 1e-5));
    }
    assert!(noisy
        .slice_dim0(4, 5)
        .unwrap()
        .allclose(&images.slice_dim0(4, 5).unwrap(), 1e-7));
}

#[test]
fn test_all_context_choices_keep_masks_and_audio_unchanged() {
    let vae = BiasedTestVae::new(0.0);
    let source_video = Tensor::full([1, 2, 57, 2, 2], 0.45);
    let source_audio = Tensor::full([1, 2, 3, 330], 0.25);
    let target_video = Tensor::zeros([1, 2, 57, 2, 2]);
    let target_audio = Tensor::zeros([1, 2, 3, 330]);

    for &frames in &MASKED_AV_CONTEXTS {
        let clean_res = apply_native_masked_av(
            &target_video,
            &target_audio,
            &source_video,
            &source_audio,
            frames,
            "Match Video Handover",
            0,
            0,
            0,
            0.03,
        )
        .unwrap();

        let noisy_prefix =
            encode_noisy_prefix_with_plan(&vae, &source_video, &clean_res.plan, 0.4, 0).unwrap();
        let orig_tail = source_video
            .slice_axis(2, clean_res.plan.start_t, clean_res.plan.end_t)
            .unwrap();
        assert!(!noisy_prefix.allclose(&orig_tail, 1e-6));
        assert_eq!(noisy_prefix.shape, orig_tail.shape);
    }
}

#[test]
fn test_zero_noise_cancels_codec_bias_instead_of_replacing_original() {
    let vae = BiasedTestVae::new(10.0);
    let mut seq = Vec::with_capacity(2 * 12 * 2 * 2);
    for i in 0..(2 * 12 * 2 * 2) {
        seq.push((i as f32) / 96.0);
    }
    let source_video = Tensor::from_vec(vec![1, 2, 12, 2, 2], seq).unwrap();
    let target_video = Tensor::zeros([1, 2, 17, 2, 2]);
    let source_audio = Tensor::ones([1, 2, 3, 65]);
    let target_audio = Tensor::zeros([1, 2, 3, 92]);

    let res = apply_native_masked_av(
        &target_video,
        &target_audio,
        &source_video,
        &source_audio,
        39,
        "Match Video Handover",
        0,
        0,
        0,
        0.0,
    )
    .unwrap();

    let encoded = encode_noisy_prefix_with_plan(&vae, &source_video, &res.plan, 0.0, 0).unwrap();
    assert!(encoded.allclose(&source_video, 1e-6));
}

#[test]
fn test_noise_hard_protects_audio_even_with_saved_feather_setting() {
    let vae = BiasedTestVae::new(0.0);
    let (_, mut sess) = MiniMaxPrefixCacheConfigNode.configure(
        "Native Masked AV",
        39,
        5,
        true,
        true,
        0.4,
        0.5,
        0.0,
    );
    let prev = AvLatent::from_video_audio(
        Tensor::full([1, 2, 12, 2, 2], 0.4),
        Tensor::ones([1, 2, 3, 65]),
    );
    let target = AvLatent::from_video_audio(
        Tensor::zeros([1, 2, 17, 2, 2]),
        Tensor::zeros([1, 2, 3, 92]),
    );

    let out = MiniMaxPrefixCacheApplierNode
        .apply_cache(
            &mut sess,
            &[],
            Some(&prev),
            Some(&target),
            None,
            None,
            Some(&vae),
            None,
            None,
            None,
            1.0,
            1.0,
            "Match Video Handover",
            6, // Saved feather ticks > 0, but reference_noise is enabled so it must be forced to 0!
            0.0,
            0,
        )
        .unwrap();

    let masked = out.latent.unwrap();
    let a_mask = masked.audio_mask.unwrap();
    let protected_audio = a_mask.slice_axis(3, 0, 65).unwrap();
    assert!(protected_audio.allclose(&Tensor::zeros([1, 1, 3, 65]), 1e-6));
}

#[test]
fn test_requested_context_snaps_to_actual_target_duration() {
    let vae = BiasedTestVae::new(0.0);
    let cfg = KVCacheConfig {
        continuation_mode: "Native Masked AV".to_string(),
        rolling_frames: 192,
        reference_noise_strength: 0.4,
        ..Default::default()
    };
    let mut sess = LongVideoSession::new(cfg);
    let prev = AvLatent::from_video_audio(
        Tensor::full([1, 2, 36, 2, 2], 0.5),
        Tensor::ones([1, 2, 3, 207]),
    );
    let target = AvLatent::from_video_audio(
        Tensor::zeros([1, 2, 17, 2, 2]),
        Tensor::zeros([1, 2, 3, 92]),
    );

    let out = MiniMaxPrefixCacheApplierNode
        .apply_cache(
            &mut sess,
            &[],
            Some(&prev),
            Some(&target),
            None,
            None,
            Some(&vae),
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
    let masked = out.latent.unwrap();
    assert_eq!(masked.metadata["minimax_prefix_frames"], json!(39));
}

#[test]
fn test_initial_clip_needs_no_vae_and_off_switch_never_calls_it() {
    let vae = BiasedTestVae::new(0.0);
    let target = AvLatent::from_video_audio(
        Tensor::zeros([1, 2, 17, 2, 2]),
        Tensor::zeros([1, 2, 3, 92]),
    );

    let cfg_on = KVCacheConfig {
        continuation_mode: "Native Masked AV".to_string(),
        rolling_frames: 39,
        reference_noise_strength: 0.4,
        ..Default::default()
    };
    let mut sess_on = LongVideoSession::new(cfg_on);
    let out_init = MiniMaxPrefixCacheApplierNode
        .apply_cache(
            &mut sess_on,
            &[],
            None,
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
    assert!(out_init.latent.is_some());

    let cfg_off = KVCacheConfig {
        continuation_mode: "Native Masked AV".to_string(),
        rolling_frames: 39,
        reference_noise_strength: 0.0,
        ..Default::default()
    };
    let mut sess_off = LongVideoSession::new(cfg_off);
    let prev = AvLatent::from_video_audio(
        Tensor::full([1, 2, 12, 2, 2], 0.4),
        Tensor::ones([1, 2, 3, 65]),
    );
    MiniMaxPrefixCacheApplierNode
        .apply_cache(
            &mut sess_off,
            &[],
            Some(&prev),
            Some(&target),
            None,
            None,
            Some(&vae),
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
    assert_eq!(vae.decode_calls.load(Ordering::SeqCst), 0);
    assert_eq!(vae.encode_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn test_original_image_references_are_not_noised() {
    let vae = BiasedTestVae::new(0.0);
    let ref_img = Tensor::full([1, 8, 8, 3], 0.9);
    let refs = vec![IdentityReference {
        image: ref_img.clone(),
        strength: 1.0,
        source: "native".to_string(),
    }];
    let cond = vec![ConditioningEntry::default()];

    let cfg = KVCacheConfig {
        continuation_mode: "Native Masked AV".to_string(),
        rolling_frames: 39,
        reference_noise_strength: 0.4,
        ..Default::default()
    };
    let mut sess = LongVideoSession::new(cfg);
    let prev = AvLatent::from_video_audio(
        Tensor::full([1, 2, 12, 2, 2], 0.4),
        Tensor::ones([1, 2, 3, 65]),
    );
    let target = AvLatent::from_video_audio(
        Tensor::zeros([1, 2, 17, 2, 2]),
        Tensor::zeros([1, 2, 3, 92]),
    );

    let out = MiniMaxPrefixCacheApplierNode
        .apply_cache(
            &mut sess,
            &cond,
            Some(&prev),
            Some(&target),
            None,
            None,
            Some(&vae),
            Some(&ref_img),
            Some(&refs),
            None,
            1.0,
            1.0,
            "Match Video Handover",
            0,
            0.0,
            0,
        )
        .unwrap();

    assert!(out.anchor_image.unwrap().allclose(&ref_img, 1e-7));
    assert!(out.identity_references[0].image.allclose(&ref_img, 1e-7));
}

#[test]
fn test_missing_vae_and_phase_shifted_sources_are_explicit() {
    let prev = AvLatent::from_video_audio(
        Tensor::full([1, 2, 12, 2, 2], 0.4),
        Tensor::ones([1, 2, 3, 65]),
    );
    let target = AvLatent::from_video_audio(
        Tensor::zeros([1, 2, 17, 2, 2]),
        Tensor::zeros([1, 2, 3, 92]),
    );

    let cfg = KVCacheConfig {
        continuation_mode: "Native Masked AV".to_string(),
        rolling_frames: 39,
        reference_noise_strength: 0.4,
        ..Default::default()
    };
    let mut sess = LongVideoSession::new(cfg.clone());
    assert!(MiniMaxPrefixCacheApplierNode
        .apply_cache(
            &mut sess,
            &[],
            Some(&prev),
            Some(&target),
            None,
            None,
            None, // Missing VAE
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
        .is_err());

    let vae = BiasedTestVae::new(0.0);
    let mut trimmed_prev = prev.clone();
    trimmed_prev
        .metadata
        .insert("minimax_source_start_step".to_string(), json!(7));
    let mut sess2 = LongVideoSession::new(cfg);
    assert!(MiniMaxPrefixCacheApplierNode
        .apply_cache(
            &mut sess2,
            &[],
            Some(&trimmed_prev),
            Some(&target),
            None,
            None,
            Some(&vae),
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
        .is_err());
}
