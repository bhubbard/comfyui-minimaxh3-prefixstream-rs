//! Rust port of `tests/test_fused_attention.py`.

use minimaxh3_prefixstream::engine::fused_attention::asymmetric_cached_attention;
use minimaxh3_prefixstream::engine::rope_aligner::{
    total_span_for_steps, video_t_grid, video_t_spans, TemporalCursorTracker,
};
use minimaxh3_prefixstream::pipeline::seam_protector::{
    audio_equal_power_crossfade_concat, latent_soft_blend, stitch_audio_latents,
    stitch_video_latents,
};
use minimaxh3_prefixstream::tensor::Tensor;

#[test]
fn test_asymmetric_cached_attention() {
    let b = 1;
    let h = 4;
    let n_curr = 32;
    let n_cache = 24;
    let d = 16;

    let q_curr = Tensor::randn([b, h, n_curr, d], 1);
    let k_curr = Tensor::randn([b, h, n_curr, d], 2);
    let v_curr = Tensor::randn([b, h, n_curr, d], 3);

    let k_cache = Tensor::randn([b, h, n_cache, d], 4);
    let v_cache = Tensor::randn([b, h, n_cache, d], 5);

    let out_no_cache = asymmetric_cached_attention(&q_curr, &k_curr, &v_curr, None, None).unwrap();
    assert_eq!(out_no_cache.shape, vec![b, h, n_curr, d]);

    let k_full = Tensor::concat(&[&k_cache, &k_curr], 2).unwrap();
    let v_full = Tensor::concat(&[&v_cache, &v_curr], 2).unwrap();
    let out_with_cache =
        asymmetric_cached_attention(&q_curr, &k_full, &v_full, None, None).unwrap();
    assert_eq!(out_with_cache.shape, vec![b, h, n_curr, d]);
    assert!(!out_no_cache.allclose(&out_with_cache, 1e-4));
}

#[test]
fn test_rope_video_grids() {
    assert_eq!(video_t_spans(7, 0), vec![1, 4, 4, 4, 4, 1, 4]);
    assert_eq!(
        video_t_grid(7, 0),
        vec![0.0, 1.0, 5.0, 9.0, 13.0, 17.0, 18.0]
    );
    assert_eq!(total_span_for_steps(7, 0), 22);
    assert_eq!(total_span_for_steps(12, 0), 39);
}

#[test]
fn test_cursor_tracker() {
    let mut tracker = TemporalCursorTracker::new();
    assert_eq!(tracker.current_global_step, 0);

    // Clip 1: 32 latent steps, no rolling overlap trimmed
    let info1 = tracker.advance_clip(32, Some(0));
    assert_eq!(info1.start_step, 0);
    assert_eq!(info1.next_step, 32);
    assert_eq!(tracker.current_global_step, 32);

    // Clip 2: 32 latent steps, 7 rolling overlap steps -> net 25 steps
    let info2 = tracker.advance_clip(32, Some(7));
    assert_eq!(info2.start_step, 32);
    assert_eq!(info2.next_step, 57);
    assert_eq!(tracker.current_global_step, 57);
}

#[test]
fn test_seam_protector_audio_and_latent() {
    let a1 = Tensor::ones([1, 64, 10]);
    let a2 = Tensor::zeros([1, 64, 10]);
    let faded = audio_equal_power_crossfade_concat(&a1, &a2, 4).unwrap();
    assert_eq!(faded.shape, vec![1, 64, 16]);

    let v1 = Tensor::ones([1, 16, 8, 8, 8]);
    let v2 = Tensor::zeros([1, 16, 8, 8, 8]);
    let blended = latent_soft_blend(&v1, &v2, 4).unwrap();
    assert_eq!(blended.shape, vec![1, 16, 12, 8, 8]);
}

#[test]
fn test_stitch_video_and_audio() {
    let v_prev = Tensor::ones([1, 16, 20, 8, 8]);
    let v_curr = Tensor::full([1, 16, 20, 8, 8], 2.0);
    // Overlap = 7 steps -> total = 20 + 20 - 7 = 33 steps
    let v_stitched = stitch_video_latents(&v_prev, &v_curr, 7, 3).unwrap();
    assert_eq!(v_stitched.shape, vec![1, 16, 33, 8, 8]);

    let a_prev = Tensor::ones([1, 64, 50]);
    let a_curr = Tensor::full([1, 64, 50], 2.0);
    // Overlap = 15 steps -> total = 50 + 50 - 15 = 85 steps
    let a_stitched = stitch_audio_latents(Some(&a_prev), &a_curr, 15).unwrap();
    assert_eq!(a_stitched.shape, vec![1, 64, 85]);
}
