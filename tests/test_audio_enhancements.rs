//! Rust port of `tests/test_audio_enhancements.py`.

use std::f32::consts::PI;

use minimaxh3_prefixstream::engine::cache_manager::KVCacheConfig;
use minimaxh3_prefixstream::nodes::{MiniMaxLongVideoStitcherNode, MiniMaxTrimPrefixLatentNode};
use minimaxh3_prefixstream::pipeline::long_video_director::LongVideoSession;
use minimaxh3_prefixstream::pipeline::native_masked_av::apply_native_masked_av;
use minimaxh3_prefixstream::pipeline::seam_protector::{fit_audio_length, stitch_audio_waveforms};
use minimaxh3_prefixstream::tensor::{AudioData, Tensor};

#[test]
fn test_fit_audio_length_no_dc_plateau() {
    let a = Tensor::ones([1, 16, 100]);
    let fitted = fit_audio_length(&a, 120).unwrap();
    assert_eq!(fitted.shape, vec![1, 16, 120]);
    // First 68 samples (100 - 32 micro-fade) remain 1.0, last 20 padded samples are 0.0
    let head = fitted.slice_axis(2, 0, 68).unwrap();
    let tail = fitted.slice_axis(2, 100, 120).unwrap();
    assert!(head.allclose(&Tensor::ones([1, 16, 68]), 1e-6));
    assert!(tail.allclose(&Tensor::zeros([1, 16, 20]), 1e-6));
}

#[test]
fn test_fit_audio_length_truncate_and_exact() {
    let mut seq = Vec::with_capacity(100);
    for i in 0..100 {
        seq.push(i as f32);
    }
    let a = Tensor::from_vec(vec![1, 4, 25], seq).unwrap();
    let trunc = fit_audio_length(&a, 15).unwrap();
    assert_eq!(trunc.shape, vec![1, 4, 15]);
    assert!(trunc.allclose(&a.slice_axis(2, 0, 15).unwrap(), 1e-6));

    let exact = fit_audio_length(&a, 25).unwrap();
    assert!(exact.allclose(&a, 1e-6));
}

#[test]
fn test_identical_audio_overlap_does_not_amplify_or_clip() {
    let sr = 32000u32;
    let fps = 24.0_f32;
    let samples = 64000usize; // 2.0s
    let wave = Tensor::full([1, 2, samples], 0.8);
    let prev_audio = AudioData {
        waveform: wave.clone(),
        sample_rate: sr,
    };
    let curr_audio = AudioData {
        waveform: wave,
        sample_rate: sr,
    };

    let stitched = stitch_audio_waveforms(Some(&prev_audio), Some(&curr_audio), 48, 22, 15.0, fps)
        .unwrap()
        .unwrap();
    let w = &stitched.waveform;
    let max_val = w.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_val = w.data.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        max_val <= 0.8 + 1e-5,
        "Coherent audio crossfade amplified signal above original level: {}",
        max_val
    );
    assert!(
        min_val >= 0.8 - 1e-5,
        "Coherent audio crossfade dipped below original level: {}",
        min_val
    );
}

#[test]
fn test_identical_sine_overlap_preserves_waveform_and_timeline() {
    let sr = 24000u32;
    let fps = 24.0_f32;
    let total_samples = 72000usize; // 3 seconds
    let mut full_data = vec![0.0_f32; 2 * total_samples];
    for i in 0..total_samples {
        let t = (i as f32) / (sr as f32);
        let ch0 = 0.6 * (2.0 * PI * 440.0 * t).sin();
        let ch1 = 0.4 * (2.0 * PI * 660.0 * t).cos();
        full_data[i] = ch0;
        full_data[total_samples + i] = ch1;
    }
    let full_wave = Tensor::from_vec(vec![1, 2, total_samples], full_data).unwrap();

    // Clip 1: frames 0..48 (0.0s .. 2.0s -> 48000 samples)
    // Clip 2: frames 26..72 (22-frame overlap with Clip 1 -> samples 26000..72000)
    let prev_audio = AudioData {
        waveform: full_wave.slice_axis(2, 0, 48000).unwrap(),
        sample_rate: sr,
    };
    let curr_audio = AudioData {
        waveform: full_wave.slice_axis(2, 26000, 72000).unwrap(),
        sample_rate: sr,
    };

    let stitched = stitch_audio_waveforms(Some(&prev_audio), Some(&curr_audio), 46, 22, 15.0, fps)
        .unwrap()
        .unwrap();
    assert_eq!(stitched.waveform.shape[2], total_samples);
    assert!(stitched.waveform.allclose(&full_wave, 1e-5));
}

#[test]
fn test_audio_handover_vs_full_previous_tail() {
    // Source video with 65 latent steps (226 frames).
    // Snap finds canonical end_t = 62 (209 frames), leaving 17 ignored tail frames.
    let src_v = Tensor::randn([1, 4, 65, 2, 2], 10);
    let src_a = Tensor::randn([1, 2, 2, 377], 11);
    let tgt_v = Tensor::zeros([1, 4, 37, 2, 2]);
    let tgt_a = Tensor::zeros([1, 2, 2, 207]);

    let handover = apply_native_masked_av(
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
    let full_tail = apply_native_masked_av(
        &tgt_v,
        &tgt_a,
        &src_v,
        &src_a,
        39,
        "Full Previous Tail",
        0,
        0,
        0,
        0.0,
    )
    .unwrap();

    assert!(handover.plan.audio_steps < full_tail.plan.audio_steps);
    assert_eq!(handover.plan.audio_steps, 65);
}

#[test]
fn test_stitcher_double_trim_guard() {
    let config = KVCacheConfig {
        rolling_frames: 22,
        ..Default::default()
    };
    let mut sess = LongVideoSession::new(config.clone());
    sess.last_rolling_frames = 22;

    let trim_node = MiniMaxTrimPrefixLatentNode;
    let stitch_node = MiniMaxLongVideoStitcherNode;

    let prev_imgs = Tensor::ones([50, 32, 32, 3]).scale(0.5);
    let curr_imgs = Tensor::ones([50, 32, 32, 3]).scale(0.6);
    let prev_aud = AudioData {
        waveform: Tensor::ones([1, 2, 66667]).scale(0.1),
        sample_rate: 32000,
    };
    let curr_aud = AudioData {
        waveform: Tensor::ones([1, 2, 66667]).scale(0.1),
        sample_rate: 32000,
    };

    // 1. Upstream Trim node trims 22 frames -> 28 frames left, and marks session.is_current_clip_trimmed = true
    let trim_out = trim_node
        .trim_frames(
            0,
            Some(&curr_imgs),
            Some(&curr_aud),
            None,
            Some(&mut sess),
            Some(&config),
            24.0,
            true,
        )
        .unwrap();
    let trimmed_imgs = trim_out.images.unwrap();
    let trimmed_aud = trim_out.audio.unwrap();
    assert_eq!(trimmed_imgs.shape[0], 28);
    assert!(sess.is_current_clip_trimmed);

    // 2. Stitcher receives already-trimmed clip + same session: must NOT trim another 22 frames!
    let (stitched_imgs, _, _) = stitch_node
        .stitch(
            Some(&sess),
            Some(&prev_imgs),
            Some(&trimmed_imgs),
            Some(&prev_aud),
            Some(&trimmed_aud),
            None,
            None,
            -1,
            4,
            15.0,
            true,
            24.0,
        )
        .unwrap();
    assert_eq!(stitched_imgs.unwrap().shape[0], 50 + 28);
}
