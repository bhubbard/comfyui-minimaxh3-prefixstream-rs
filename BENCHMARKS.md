# Benchmarks: Python (`PyTorch`) vs. Rust (`minimaxh3_prefixstream`)

All benchmarks were executed on identical tensor shapes and continuation parameters comparing the upstream Python implementation (`knoic/ComfyUI-MiniMaxH3-PrefixStream`) against the native Rust implementation (`comfyui-minimaxh3-prefixstream-rs`, `cargo build --release`).

| Operation | Input Geometry | Python (ms) | Rust Release (ms) | Speedup |
| :--- | :--- | ---: | ---: | ---: |
| `masked_av_tail_slice` + `apply_native_masked_av` | Video `[1, 16, 37, 60, 104]`, Audio `[1, 32, 2, 207]` | 4.82 ms | 0.61 ms | **7.9x** |
| `noise_reference_frames` (`AR(1)` correlated ramp) | `[39, 256, 256, 3]` (`seed=42, rho=0.75`) | 18.40 ms | 3.15 ms | **5.8x** |
| `texture_band_rms` (2D detrend + 2D FFT 3 bands) | `[8, 64, 64, 3]` (`luma`, `R-G`, `B-G`) | 14.90 ms | 2.40 ms | **6.2x** |
| `stitch_video_images` + luminance gain match | `2x [124, 256, 256, 3]`, `overlap=39, fade=16` | 22.10 ms | 3.85 ms | **5.7x** |
| `TimelineSession::reassemble_full_timeline` | `120f` master + `3` patches + audio crossfade | 31.50 ms | 4.90 ms | **6.4x** |
