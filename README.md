# ComfyUI-MiniMaxH3-PrefixStream-RS

[![CI](https://github.com/bhubbard/comfyui-minimaxh3-prefixstream-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/bhubbard/comfyui-minimaxh3-prefixstream-rs/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A complete, pure-Rust port of [`knoic/ComfyUI-MiniMaxH3-PrefixStream`](https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream) — an infinite video & audio continuation engine, 3D-RoPE temporal cursor aligner, Native Masked AV planner, frequency-band texture drift analyzer, DAG Clip Bin manager, and non-destructive timeline slicer/reassembler for **MiniMaxHailuo-03 (H3)**.

---

## Key Features

- **H3 Spatiotemporal Grid & 3D-RoPE Aligner (`engine::cache_manager`, `engine::rope_aligner`)**:
  - Exact modeling of MiniMax H3's `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)` cycle (`5` latent steps = `17` pixel frames) and canonical `17k + 5` run grid (`5, 22, 39, 56, 73, 90, 107, 124`).
  - Continuous 3D-RoPE temporal coordinate tracking via `TemporalCursorTracker` across multi-clip streams.
  - Dual-tier (Anchor + Rolling) `PrefixKVCacheManager` and `asymmetric_cached_attention` (`engine::fused_attention`).
- **Native Masked AV Continuation (`pipeline::native_masked_av`)**:
  - Phase-aligned source tail slicing (`(source_start_step + start_t) % 5 == 0` and `(source_start_step + end_t) % 5 == 2`).
  - Synchronized 24 fps video / 40 Hz audio tick calculation, exact context detection (`39, 90, 141, 192` frames), and half-cosine feathered audio denoise masks.
- **Anti-Texture-Hardening & Frequency Diagnostics (`pipeline::context_noise`, `engine::texture_diagnostics`)**:
  - Temporally correlated (`AR(1)`) ramped pixel-space reference noise (`noise_reference_frames`, `encode_noisy_prefix`) to prevent static texture carryover.
  - 2D linear plane detrending + 2D Hann-windowed FFT radial wavelength power spectrum analysis across `2_8px`, `8_32px`, and `32_128px` bands (`texture_band_rms`, `summarize_clip_texture_drift`).
- **Seamless Audio/Video Handover (`pipeline::seam_protector`)**:
  - Subsampled median RGB luminance gain estimation (`estimate_luminance_gain`), cosine fade-out (`apply_luminance_gain_fade`), pixel-space S-curve video stitching (`stitch_video_images`), and complementary `sin^2`/`cos^2` audio waveform stitching (`stitch_audio_waveforms`).
- **DAG Clip Bin & Non-Destructive Timeline Slicer (`engine::clip_bin_manager`, `engine::timeline_session_manager`)**:
  - SafeTensors-backed multi-take branching DAG asset storage (`ClipMeta`, `save_clip_asset`, `load_clip_asset`) with strict path-traversal and symlink escape protection.
  - Non-destructive `Dtype::U8` SafeTensors chunk patch storage, gap analysis, and full-timeline crossfaded reassembly (`TimelineSession`).
- **15 ComfyUI Nodes & Workflow Validator (`nodes`)**:
  - Full Rust implementation of all 15 custom nodes plus JSON workflow validation against official ComfyUI workflow templates (`examples/`).

---

## Architecture Mapping (Python -> Rust)

| Upstream Python Module | Rust Module | Core Types & Functions |
| :--- | :--- | :--- |
| `engine/cache_manager.py` | `src/engine/cache_manager.rs` | `KVCacheConfig`, `PrefixKVCacheManager`, `snap_to_run_grid`, `pixel_frames_to_latent_steps`, `latent_steps_to_pixel_frames` |
| `engine/rope_aligner.py` | `src/engine/rope_aligner.rs` | `TemporalCursorTracker`, `ClipCursorInfo`, `video_t_spans`, `video_t_grid`, `total_span_for_steps` |
| `engine/fused_attention.py` | `src/engine/fused_attention.rs` | `asymmetric_cached_attention` |
| `engine/continuation_state.py` | `src/engine/continuation_state.rs` | `ContinuationState`, `prepare_identity_references`, `encode_state`, `decode_state` |
| `engine/texture_diagnostics.py` & `tools/analyze_texture_drift.py` | `src/engine/texture_diagnostics.rs` | `texture_band_rms`, `summarize_clip_texture_drift`, `remove_linear_trend` |
| `engine/clip_bin_manager.py` | `src/engine/clip_bin_manager.rs` | `ClipMeta`, `save_clip_asset`, `load_clip_asset`, `atomic_write_json`, `get_clip_dir` |
| `engine/clip_bin_api.py` | `src/engine/clip_bin_api.rs` | `get_project_clips_api`, `update_clip_rating_api`, `delete_clip_api`, `rescan_project_api` |
| `engine/timeline_session_manager.py` | `src/engine/timeline_session_manager.rs` | `TimelineSession`, `slice_video_and_audio`, `render_timeline_indicator_image` |
| `pipeline/native_masked_av.py` | `src/pipeline/native_masked_av.rs` | `apply_native_masked_av`, `masked_av_tail_slice`, `is_exact_masked_av_context` |
| `pipeline/context_noise.py` | `src/pipeline/context_noise.rs` | `reference_noise_strengths`, `noise_reference_frames`, `encode_noisy_prefix` |
| `pipeline/seam_protector.py` | `src/pipeline/seam_protector.rs` | `stitch_video_images`, `stitch_audio_waveforms`, `estimate_luminance_gain`, `trim_images_and_audio` |
| `pipeline/long_video_director.py` | `src/pipeline/long_video_director.rs` | `LongVideoSession`, `inject_minimax_keyframes`, `step_offsets` |
| `pipeline/disk_stream.py` | `src/pipeline/disk_stream.rs` | `append_disk_clip`, `DiskStreamManifest` |
| `nodes.py` | `src/nodes.rs` | All 15 `MiniMax*Node` structs + `validate_workflow_json` |

---

## Quick Start

### Build & Run Tests

```bash
cargo build --release
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

### CLI Usage (`minimax-prefixstream`)

```bash
# 1. Inspect the H3 (1,4,4,4,4) spatiotemporal grid and snap a frame count
cargo run --bin minimax-prefixstream -- grid --frames 42 --prefer nearest

# 2. Plan a phase-aligned Native Masked AV tail slice & 40Hz audio tick window
cargo run --bin minimax-prefixstream -- masked-av --source-video-t 37 --context-frames 39 --target-video-t 37

# 3. Simulate multi-clip 3D-RoPE temporal cursor progression
cargo run --bin minimax-prefixstream -- rope-plan --clips 5 --latent-steps 37

# 4. Run 2D FFT spatial texture drift analysis across (2-8px, 8-32px, 32-128px) bands
cargo run --bin minimax-prefixstream -- texture-drift --window 4

# 5. Validate official ComfyUI workflow templates
cargo run --bin minimax-prefixstream -- workflow --file examples/MiniMaxH3_PrefixStream_v1.0.json

# 6. List all 15 registered ComfyUI custom nodes
cargo run --bin minimax-prefixstream -- nodes
```

---

## License

MIT License — see [LICENSE](LICENSE).
