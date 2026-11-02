//! Rust port of `tests/test_cache_manager.py`.

use minimaxh3_prefixstream::engine::cache_manager::{
    latent_steps_to_pixel_frames, pixel_frames_to_latent_steps, snap_to_run_grid, KVCacheConfig,
    PrefixKVCacheManager,
};
use minimaxh3_prefixstream::tensor::Tensor;

#[test]
fn test_cache_initialization() {
    let config = KVCacheConfig {
        anchor_frames: 5,
        rolling_frames: 22,
        num_layers: 12,
        ..Default::default()
    };
    let manager = PrefixKVCacheManager::new(config);
    assert!(!manager.has_cache(0));
    assert!(!manager.has_cache(11));
}

#[test]
fn test_set_and_get_kv() {
    let config = KVCacheConfig {
        anchor_frames: 5,
        rolling_frames: 22,
        num_layers: 4,
        ..Default::default()
    };
    let mut manager = PrefixKVCacheManager::new(config);

    // Shape: [B=1, H=8, N=100, D=64]
    let k = Tensor::randn([1, 8, 100, 64], 101);
    let v = Tensor::randn([1, 8, 100, 64], 102);

    manager.set_anchor_kv(0, k.clone(), v.clone());
    assert!(manager.has_cache(0));

    let (cached_k, cached_v) = manager.get_concat_kv(0, "video").unwrap();
    assert_eq!(cached_k.shape, vec![1, 8, 100, 64]);
    assert_eq!(cached_v.shape, vec![1, 8, 100, 64]);
    assert!(cached_k.allclose(&k, 1e-6));
    assert!(cached_v.allclose(&v, 1e-6));
}

#[test]
fn test_anchor_plus_rolling_concatenation() {
    let config = KVCacheConfig {
        anchor_frames: 5,
        rolling_frames: 22,
        num_layers: 2,
        ..Default::default()
    };
    let mut manager = PrefixKVCacheManager::new(config);

    let k_anchor = Tensor::ones([1, 4, 50, 32]);
    let v_anchor = Tensor::ones([1, 4, 50, 32]);
    let k_rolling = Tensor::full([1, 4, 80, 32], 2.0);
    let v_rolling = Tensor::full([1, 4, 80, 32], 2.0);

    manager.set_anchor_kv(0, k_anchor, v_anchor);
    manager.set_rolling_kv(0, k_rolling, v_rolling);

    let (cat_k, cat_v) = manager.get_concat_kv(0, "video").unwrap();
    assert_eq!(cat_k.shape, vec![1, 4, 130, 32]);
    assert_eq!(cat_v.shape, vec![1, 4, 130, 32]);

    let first_50 = cat_k.slice_axis(2, 0, 50).unwrap();
    let last_80 = cat_k.slice_axis(2, 50, 130).unwrap();
    assert!(first_50.allclose(&Tensor::ones([1, 4, 50, 32]), 1e-6));
    assert!(last_80.allclose(&Tensor::full([1, 4, 80, 32], 2.0), 1e-6));
}

#[test]
fn test_prefetch_slot_mechanism() {
    let config = KVCacheConfig {
        num_layers: 2,
        ..Default::default()
    };
    let mut manager = PrefixKVCacheManager::new(config);

    let k = Tensor::randn([1, 4, 30, 16], 201);
    let v = Tensor::randn([1, 4, 30, 16], 202);
    manager.set_anchor_kv(0, k.clone(), v.clone());

    manager.prefetch_layer(0, "video");
    let (pk, pv) = manager.consume_prefetched_kv().unwrap();
    assert!(pk.allclose(&k, 1e-6));
    assert!(pv.allclose(&v, 1e-6));

    // Slot must be cleared after consumption
    assert!(manager.consume_prefetched_kv().is_none());
}

#[test]
fn test_memory_tracking() {
    let config = KVCacheConfig {
        num_layers: 2,
        ..Default::default()
    };
    let mut manager = PrefixKVCacheManager::new(config);
    assert_eq!(manager.total_cached_bytes(), 0);

    // 1 * 4 * 100 * 64 * 4 bytes = 102400 bytes per tensor -> 204800 bytes for (K, V)
    let k = Tensor::zeros([1, 4, 100, 64]);
    let v = Tensor::zeros([1, 4, 100, 64]);
    manager.set_anchor_kv(0, k, v);
    assert_eq!(manager.total_cached_bytes(), 204_800);

    manager.clear_all();
    assert_eq!(manager.total_cached_bytes(), 0);
}

#[test]
fn test_grid_math_utilities() {
    assert_eq!(snap_to_run_grid(5, "nearest"), 5);
    assert_eq!(snap_to_run_grid(20, "nearest"), 22);
    assert_eq!(snap_to_run_grid(22, "nearest"), 22);
    assert_eq!(snap_to_run_grid(39, "nearest"), 39);
    assert_eq!(snap_to_run_grid(124, "nearest"), 124);

    assert_eq!(pixel_frames_to_latent_steps(1), 1);
    assert_eq!(pixel_frames_to_latent_steps(5), 2);
    assert_eq!(pixel_frames_to_latent_steps(22), 7);
    assert_eq!(pixel_frames_to_latent_steps(39), 12);

    assert_eq!(latent_steps_to_pixel_frames(1), 1);
    assert_eq!(latent_steps_to_pixel_frames(2), 5);
    assert_eq!(latent_steps_to_pixel_frames(7), 22);
    assert_eq!(latent_steps_to_pixel_frames(12), 39);
}
