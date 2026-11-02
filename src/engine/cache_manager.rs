//! Prefix KV Cache Manager and MiniMax H3 VAE Spatiotemporal Run Grid utilities.
//!
//! Rust port of `engine/cache_manager.py`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::tensor::Tensor;

/// Per-token pixel frame spans across each 5-step MiniMax H3 cycle (`1 + 4 + 4 + 4 + 4 = 17` frames).
pub const FRAME_PER_TOKEN: [usize; 5] = [1, 4, 4, 4, 4];

/// Canonical MiniMax H3 VAE run grid (`17k + 5`, ascending).
pub const VIDEO_RUN_GRID: [usize; 8] = [5, 22, 39, 56, 73, 90, 107, 124];

/// Snap a requested pixel frame count to [`VIDEO_RUN_GRID`] using `"nearest"`, `"lower"`, or `"higher"`.
///
/// Mirrors `snap_to_run_grid` in `engine/cache_manager.py`.
pub fn snap_to_run_grid(frames: usize, prefer: &str) -> usize {
    let target = frames.max(1);
    match prefer.trim().to_lowercase().as_str() {
        "lower" => VIDEO_RUN_GRID
            .iter()
            .copied()
            .rfind(|&g| g <= target)
            .unwrap_or(VIDEO_RUN_GRID[0]),
        "higher" => VIDEO_RUN_GRID
            .iter()
            .copied()
            .find(|&g| g >= target)
            .unwrap_or(*VIDEO_RUN_GRID.last().unwrap()),
        _ => VIDEO_RUN_GRID
            .iter()
            .copied()
            .min_by_key(|&g| (g as isize - target as isize).unsigned_abs())
            .unwrap_or(VIDEO_RUN_GRID[0]),
    }
}

/// Convert a pixel frame count to the exact or minimum MiniMax H3 latent step count (`T`).
///
/// Mirrors `pixel_frames_to_latent_steps` in `engine/cache_manager.py`.
pub fn pixel_frames_to_latent_steps(pixel_frames: usize) -> usize {
    if pixel_frames == 0 {
        return 0;
    }
    let mut k = 0usize;
    let mut covered = 0usize;
    while covered < pixel_frames {
        covered += FRAME_PER_TOKEN[k % 5];
        k += 1;
    }
    k.max(1)
}

/// Alias for [`pixel_frames_to_latent_steps`].
pub fn steps_for_frames(frames: usize) -> usize {
    pixel_frames_to_latent_steps(frames)
}

/// Convert MiniMax H3 latent steps (`T`) starting at phase 0 to pixel frames.
///
/// Mirrors `latent_steps_to_pixel_frames` in `engine/cache_manager.py`.
pub fn latent_steps_to_pixel_frames(latent_steps: usize) -> usize {
    (0..latent_steps).map(|k| FRAME_PER_TOKEN[k % 5]).sum()
}

/// Configuration for MiniMax H3 Prefix KV Cache & Continuation.
///
/// Mirrors `KVCacheConfig` in `engine/cache_manager.py`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KVCacheConfig {
    /// Whether prefix caching / continuation is enabled.
    pub enabled: bool,
    /// Whether to pin the initial anchor clip.
    pub use_anchor: bool,
    /// Number of anchor pixel frames (snapped to [`VIDEO_RUN_GRID`]).
    pub anchor_frames: usize,
    /// Number of rolling context pixel frames (snapped to [`VIDEO_RUN_GRID`]).
    pub rolling_frames: usize,
    /// Whether to maintain continuous 3D-RoPE temporal coordinates across clips.
    pub rope_continuity: bool,
    /// Whether to offload cached KV tensors to CPU RAM between steps.
    pub offload_to_cpu: bool,
    /// Continuation mode (`"Native Masked AV"` or `"Safe Native Keyframes"`).
    pub continuation_mode: String,
    /// Pixel-space reference noise strength (`0.0` disables).
    pub reference_noise_strength: f32,
    /// End-of-window scale for reference noise ramp (`[0.0, 1.0]`).
    pub reference_noise_end_scale: f32,
    /// Temporal AR(1) correlation for reference noise (`[0.0, 0.98]`).
    pub reference_noise_correlation: f32,
    /// Number of transformer layers in MiniMax H3 (`50`).
    pub num_layers: usize,
    /// Number of attention heads (`56`).
    pub num_heads: usize,
    /// Head dimension (`128`).
    pub head_dim: usize,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            use_anchor: true,
            anchor_frames: 5,
            rolling_frames: 39,
            rope_continuity: true,
            offload_to_cpu: true,
            continuation_mode: "Native Masked AV".to_string(),
            reference_noise_strength: 0.0,
            reference_noise_end_scale: 0.25,
            reference_noise_correlation: 0.75,
            num_layers: 50,
            num_heads: 56,
            head_dim: 128,
        }
    }
}

/// Cached Key/Value pair for a single transformer block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedLayerKV {
    /// Key tensor `[B, H, N, D]` or `[B, N, C]`.
    pub k: Tensor,
    /// Value tensor `[B, H, N, D]` or `[B, N, C]`.
    pub v: Tensor,
    /// Optional token start index.
    pub start_token: Option<usize>,
    /// Optional token end index.
    pub end_token: Option<usize>,
}

/// Telemetry statistics returned by [`PrefixKVCacheManager::get_telemetry_stats`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheTelemetryStats {
    /// Number of transformer layers with active cached KV states.
    pub cached_layers: usize,
    /// Whether anchor cache is populated.
    pub has_anchor_cache: bool,
    /// Whether rolling cache is populated.
    pub has_rolling_cache: bool,
    /// Estimated memory footprint in MiB.
    pub estimated_vram_mb: f64,
}

/// Dual-tier (Anchor + Rolling) Prefix KV Cache Manager for MiniMax H3.
///
/// Mirrors `PrefixKVCacheManager` in `engine/cache_manager.py`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrefixKVCacheManager {
    /// Cache configuration.
    pub config: KVCacheConfig,
    /// Anchor KV cache indexed by `layer_idx`.
    pub anchor_cache: HashMap<usize, CachedLayerKV>,
    /// Rolling tail KV cache indexed by `layer_idx`.
    pub rolling_cache: HashMap<usize, CachedLayerKV>,
    /// Current diffusion step counter.
    pub step_counter: usize,
    /// Optional prefetched `(K, V)` slot for double-buffered async layer prefetching.
    #[serde(skip)]
    pub prefetched_kv: Option<(Tensor, Tensor)>,
}

impl Default for PrefixKVCacheManager {
    fn default() -> Self {
        Self::new(KVCacheConfig::default())
    }
}

impl PrefixKVCacheManager {
    /// Create a new [`PrefixKVCacheManager`].
    pub fn new(config: KVCacheConfig) -> Self {
        Self {
            config,
            anchor_cache: HashMap::new(),
            rolling_cache: HashMap::new(),
            step_counter: 0,
            prefetched_kv: None,
        }
    }

    /// Store Anchor KV tensors for `layer_idx` (by value, mirroring Python `set_anchor_kv`).
    pub fn set_anchor_kv(&mut self, layer_idx: usize, k: Tensor, v: Tensor) {
        self.store_anchor_kv(layer_idx, &k, &v, None, None);
    }

    /// Store Rolling KV tensors for `layer_idx` (by value, mirroring Python `set_rolling_kv`).
    pub fn set_rolling_kv(&mut self, layer_idx: usize, k: Tensor, v: Tensor) {
        self.store_rolling_kv(layer_idx, &k, &v, None, None);
    }

    /// Check whether `layer_idx` has any cached prefix (mirroring Python `has_cache`).
    pub fn has_cache(&self, layer_idx: usize) -> bool {
        self.has_active_prefix(layer_idx)
    }

    /// Concatenate cached `[Anchor, Rolling]` KV tensors for `layer_idx` (mirroring Python `get_concat_kv`).
    pub fn get_concat_kv(&self, layer_idx: usize, _stream: &str) -> Option<(Tensor, Tensor)> {
        let mut k_parts: Vec<&Tensor> = Vec::new();
        let mut v_parts: Vec<&Tensor> = Vec::new();

        if self.config.use_anchor {
            if let Some(anc) = self.anchor_cache.get(&layer_idx) {
                k_parts.push(&anc.k);
                v_parts.push(&anc.v);
            }
        }
        if let Some(roll) = self.rolling_cache.get(&layer_idx) {
            k_parts.push(&roll.k);
            v_parts.push(&roll.v);
        }
        if k_parts.is_empty() {
            return None;
        }
        if k_parts.len() == 1 {
            return Some((k_parts[0].clone(), v_parts[0].clone()));
        }
        let seq_dim = if k_parts[0].ndim() == 4 { 2 } else { 1 };
        let cat_k = Tensor::concat(&k_parts, seq_dim).ok()?;
        let cat_v = Tensor::concat(&v_parts, seq_dim).ok()?;
        Some((cat_k, cat_v))
    }

    /// Stage concatenated KV for `layer_idx` into the prefetch slot.
    pub fn prefetch_layer(&mut self, layer_idx: usize, stream: &str) {
        self.prefetched_kv = self.get_concat_kv(layer_idx, stream);
    }

    /// Consume and clear the prefetched KV slot.
    pub fn consume_prefetched_kv(&mut self) -> Option<(Tensor, Tensor)> {
        self.prefetched_kv.take()
    }

    /// Total cached bytes across all anchor and rolling KV entries (assuming `f32` = 4 bytes/element).
    pub fn total_cached_bytes(&self) -> usize {
        let mut total_elems = 0usize;
        for entry in self.anchor_cache.values() {
            total_elems += entry.k.numel() + entry.v.numel();
        }
        for entry in self.rolling_cache.values() {
            total_elems += entry.k.numel() + entry.v.numel();
        }
        total_elems * 4
    }

    /// Store Anchor KV tensors for `layer_idx`.
    pub fn store_anchor_kv(
        &mut self,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        start_token: Option<usize>,
        end_token: Option<usize>,
    ) {
        if !self.config.use_anchor {
            return;
        }
        self.anchor_cache.insert(
            layer_idx,
            CachedLayerKV {
                k: k.clone(),
                v: v.clone(),
                start_token,
                end_token,
            },
        );
    }

    /// Store Rolling context KV tensors for `layer_idx`.
    pub fn store_rolling_kv(
        &mut self,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        start_token: Option<usize>,
        end_token: Option<usize>,
    ) {
        self.rolling_cache.insert(
            layer_idx,
            CachedLayerKV {
                k: k.clone(),
                v: v.clone(),
                start_token,
                end_token,
            },
        );
    }

    /// Check whether `layer_idx` has any cached prefix (anchor or rolling).
    pub fn has_active_prefix(&self, layer_idx: usize) -> bool {
        self.anchor_cache.contains_key(&layer_idx) || self.rolling_cache.contains_key(&layer_idx)
    }

    /// Concatenate `[Anchor (optional), Rolling (optional), Current]` Key and Value tensors along the sequence axis.
    ///
    /// For 4D tensors `[B, H, N, D]`, sequence axis is `dim = 2`; for 3D tensors `[B, N, C]`, `dim = 1`.
    pub fn get_concatenated_kv(
        &self,
        layer_idx: usize,
        current_k: &Tensor,
        current_v: &Tensor,
        include_anchor: bool,
    ) -> Result<(Tensor, Tensor)> {
        let seq_dim = if current_k.ndim() == 4 { 2 } else { 1 };
        let mut k_parts: Vec<&Tensor> = Vec::new();
        let mut v_parts: Vec<&Tensor> = Vec::new();

        if include_anchor && self.config.use_anchor {
            if let Some(anc) = self.anchor_cache.get(&layer_idx) {
                k_parts.push(&anc.k);
                v_parts.push(&anc.v);
            }
        }
        if let Some(roll) = self.rolling_cache.get(&layer_idx) {
            k_parts.push(&roll.k);
            v_parts.push(&roll.v);
        }
        k_parts.push(current_k);
        v_parts.push(current_v);

        if k_parts.len() == 1 {
            return Ok((current_k.clone(), current_v.clone()));
        }

        let full_k = Tensor::concat(&k_parts, seq_dim)?;
        let full_v = Tensor::concat(&v_parts, seq_dim)?;
        Ok((full_k, full_v))
    }

    /// Reset step counter for the next clip while preserving anchor and rolling caches.
    pub fn reset_for_next_clip(&mut self) {
        self.step_counter = 0;
    }

    /// Clear both anchor and rolling caches.
    pub fn clear_all(&mut self) {
        self.anchor_cache.clear();
        self.rolling_cache.clear();
        self.step_counter = 0;
    }

    /// Compute telemetry statistics for the cache monitor node.
    pub fn get_telemetry_stats(&self) -> CacheTelemetryStats {
        let mut total_elems = 0usize;
        for entry in self.anchor_cache.values() {
            total_elems += entry.k.numel() + entry.v.numel();
        }
        for entry in self.rolling_cache.values() {
            total_elems += entry.k.numel() + entry.v.numel();
        }
        let cached_layers = self
            .anchor_cache
            .keys()
            .chain(self.rolling_cache.keys())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let bytes = total_elems * 4;
        CacheTelemetryStats {
            cached_layers,
            has_anchor_cache: !self.anchor_cache.is_empty(),
            has_rolling_cache: !self.rolling_cache.is_empty(),
            estimated_vram_mb: (bytes as f64) / (1024.0 * 1024.0),
        }
    }
}
