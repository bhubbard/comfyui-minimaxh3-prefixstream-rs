//! Frequency-band texture diagnostics for video continuation analysis.
//!
//! Rust port of `engine/texture_diagnostics.py` and `tools/analyze_texture_drift.py`.

use std::collections::HashMap;
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::error::{PrefixStreamError, Result};
use crate::tensor::Tensor;

/// Canonical radial wavelength bands `[min_wavelength_px, max_wavelength_px)`.
pub const DEFAULT_BANDS: [(&str, f64, f64); 3] = [
    ("2_8px", 2.0, 8.0),
    ("8_32px", 8.0, 32.0),
    ("32_128px", 32.0, 128.0),
];

/// Canonical opponent-color / luminance channels analyzed by texture diagnostics.
pub const TEXTURE_CHANNELS: [&str; 3] = ["luma", "red_minus_green", "blue_minus_green"];

/// Remove a 2D linear trend (plane fit `a*x + b*y + c`) from a `[H, W]` channel buffer.
///
/// Mirrors `_remove_linear_trend` in `engine/texture_diagnostics.py`.
pub fn remove_linear_trend(channel: &[f64], height: usize, width: usize) -> Vec<f64> {
    let n = height * width;
    if n == 0 {
        return Vec::new();
    }

    // Solve least-squares system (A^T A) beta = A^T z where columns of A are [x, y, 1]
    // with x in [-1, 1], y in [-1, 1].
    // Because x and y are symmetric around 0, cross terms sum(x) = sum(y) = sum(x*y) = 0!
    let mut sum_x2 = 0.0_f64;
    let mut sum_y2 = 0.0_f64;
    let mut sum_xz = 0.0_f64;
    let mut sum_yz = 0.0_f64;
    let mut sum_z = 0.0_f64;

    for r in 0..height {
        let y = if height > 1 {
            -1.0 + 2.0 * (r as f64) / ((height - 1) as f64)
        } else {
            0.0
        };
        for c in 0..width {
            let x = if width > 1 {
                -1.0 + 2.0 * (c as f64) / ((width - 1) as f64)
            } else {
                0.0
            };
            let z = channel[r * width + c];
            sum_x2 += x * x;
            sum_y2 += y * y;
            sum_xz += x * z;
            sum_yz += y * z;
            sum_z += z;
        }
    }

    let a = if sum_x2 > 1e-12 { sum_xz / sum_x2 } else { 0.0 };
    let b = if sum_y2 > 1e-12 { sum_yz / sum_y2 } else { 0.0 };
    let c_coeff = sum_z / (n as f64);

    let mut out = vec![0.0_f64; n];
    for r in 0..height {
        let y = if height > 1 {
            -1.0 + 2.0 * (r as f64) / ((height - 1) as f64)
        } else {
            0.0
        };
        for c in 0..width {
            let x = if width > 1 {
                -1.0 + 2.0 * (c as f64) / ((width - 1) as f64)
            } else {
                0.0
            };
            let plane = a * x + b * y + c_coeff;
            out[r * width + c] = channel[r * width + c] - plane;
        }
    }
    out
}

/// Compute 1D Cooley-Tukey / Bluestein or direct DFT of complex vector `(re, im)`.
fn dft_1d(re_in: &[f64], im_in: &[f64], out_re: &mut [f64], out_im: &mut [f64]) {
    let n = re_in.len();
    if n == 0 {
        return;
    }
    // Check if power of 2 for radix-2 FFT
    if n.is_power_of_two() {
        out_re.copy_from_slice(re_in);
        out_im.copy_from_slice(im_in);
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                out_re.swap(i, j);
                out_im.swap(i, j);
            }
        }
        let mut len = 2usize;
        while len <= n {
            let ang = -2.0 * PI / (len as f64);
            let wlen_re = ang.cos();
            let wlen_im = ang.sin();
            for i in (0..n).step_by(len) {
                let mut w_re = 1.0;
                let mut w_im = 0.0;
                for k in 0..(len / 2) {
                    let u_re = out_re[i + k];
                    let u_im = out_im[i + k];
                    let v_re = out_re[i + k + len / 2] * w_re - out_im[i + k + len / 2] * w_im;
                    let v_im = out_re[i + k + len / 2] * w_im + out_im[i + k + len / 2] * w_re;
                    out_re[i + k] = u_re + v_re;
                    out_im[i + k] = u_im + v_im;
                    out_re[i + k + len / 2] = u_re - v_re;
                    out_im[i + k + len / 2] = u_im - v_im;
                    let next_w_re = w_re * wlen_re - w_im * wlen_im;
                    let next_w_im = w_re * wlen_im + w_im * wlen_re;
                    w_re = next_w_re;
                    w_im = next_w_im;
                }
            }
            len <<= 1;
        }
    } else {
        // Precompute trig table for arbitrary N
        let nf = n as f64;
        for k in 0..n {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            for t in 0..n {
                let angle = -2.0 * PI * ((k * t) % n) as f64 / nf;
                let (s, c) = angle.sin_cos();
                sum_re += re_in[t] * c - im_in[t] * s;
                sum_im += re_in[t] * s + im_in[t] * c;
            }
            out_re[k] = sum_re;
            out_im[k] = sum_im;
        }
    }
}

/// Compute full 2D FFT power spectrum `|FFT2(windowed)|^2` for `[H, W]` real input.
fn fft2_power_spectrum(
    detrended: &[f64],
    height: usize,
    width: usize,
    window_2d: &[f64],
) -> Vec<f64> {
    let mut row_fft_re = vec![0.0_f64; height * width];
    let mut row_fft_im = vec![0.0_f64; height * width];

    let mut in_re = vec![0.0_f64; width];
    let mut in_im = vec![0.0_f64; width];
    let mut out_re = vec![0.0_f64; width];
    let mut out_im = vec![0.0_f64; width];

    for r in 0..height {
        for c in 0..width {
            in_re[c] = detrended[r * width + c] * window_2d[r * width + c];
            in_im[c] = 0.0;
        }
        dft_1d(&in_re, &in_im, &mut out_re, &mut out_im);
        for c in 0..width {
            row_fft_re[r * width + c] = out_re[c];
            row_fft_im[r * width + c] = out_im[c];
        }
    }

    let mut col_in_re = vec![0.0_f64; height];
    let mut col_in_im = vec![0.0_f64; height];
    let mut col_out_re = vec![0.0_f64; height];
    let mut col_out_im = vec![0.0_f64; height];

    let mut power = vec![0.0_f64; height * width];

    for c in 0..width {
        for r in 0..height {
            col_in_re[r] = row_fft_re[r * width + c];
            col_in_im[r] = row_fft_im[r * width + c];
        }
        dft_1d(&col_in_re, &col_in_im, &mut col_out_re, &mut col_out_im);
        for r in 0..height {
            let mag_sq = col_out_re[r] * col_out_re[r] + col_out_im[r] * col_out_im[r];
            power[r * width + c] = mag_sq;
        }
    }

    power
}

/// Measure per-frame spatial texture energy across radial wavelength bands.
///
/// Mirrors `texture_band_rms` in `engine/texture_diagnostics.py`.
///
/// # Arguments
/// * `images` - `[H, W, C]` or `[F, H, W, C]` float tensor with `C >= 3` and `min(H, W) >= 4`.
/// * `bands` - Optional custom radial wavelength bands `(label, min_wavelength_px, max_wavelength_px)`.
///
/// # Returns
/// Map of `channel_name -> band_label -> Vec<f64>` (one value per frame).
pub fn texture_band_rms(
    images: &Tensor,
    bands: Option<&[(&str, f64, f64)]>,
) -> Result<HashMap<String, HashMap<String, Vec<f64>>>> {
    let (frames, height, width, channels) = match images.ndim() {
        3 => (1, images.shape[0], images.shape[1], images.shape[2]),
        4 => (
            images.shape[0],
            images.shape[1],
            images.shape[2],
            images.shape[3],
        ),
        _ => {
            return Err(PrefixStreamError::InvalidInput(format!(
                "Expected an RGB image region [H, W, C] or [F, H, W, C], got {:?}",
                images.shape
            )))
        }
    };

    if channels < 3 || height.min(width) < 4 {
        return Err(PrefixStreamError::InvalidInput(
            "Expected an RGB image region with height and width >= 4 and channels >= 3".into(),
        ));
    }

    let active_bands = bands.unwrap_or(&DEFAULT_BANDS);

    let win_y: Vec<f64> = (0..height)
        .map(|i| {
            if height > 1 {
                0.5 * (1.0 - (2.0 * PI * (i as f64) / ((height - 1) as f64)).cos())
            } else {
                1.0
            }
        })
        .collect();
    let win_x: Vec<f64> = (0..width)
        .map(|i| {
            if width > 1 {
                0.5 * (1.0 - (2.0 * PI * (i as f64) / ((width - 1) as f64)).cos())
            } else {
                1.0
            }
        })
        .collect();

    let mut window_2d = vec![0.0_f64; height * width];
    let mut window_power_sum = 0.0_f64;
    for r in 0..height {
        for c in 0..width {
            let w = win_y[r] * win_x[c];
            window_2d[r * width + c] = w;
            window_power_sum += w * w;
        }
    }
    let window_power = (window_power_sum / ((height * width) as f64)).max(1e-12);
    let normalizer = ((height * width) as f64).powi(2) * window_power;

    let mut radius_grid = vec![0.0_f64; height * width];
    for r in 0..height {
        let fy = if r <= (height - 1) / 2 {
            (r as f64) / (height as f64)
        } else {
            ((r as isize - height as isize) as f64) / (height as f64)
        };
        for c in 0..width {
            let fx = if c <= (width - 1) / 2 {
                (c as f64) / (width as f64)
            } else {
                ((c as isize - width as isize) as f64) / (width as f64)
            };
            radius_grid[r * width + c] = (fx * fx + fy * fy).sqrt();
        }
    }

    let mut result: HashMap<String, HashMap<String, Vec<f64>>> = HashMap::new();
    for ch_name in TEXTURE_CHANNELS {
        let mut band_map = HashMap::new();
        for (label, _, _) in active_bands {
            band_map.insert((*label).to_string(), Vec::with_capacity(frames));
        }
        result.insert(ch_name.to_string(), band_map);
    }

    let frame_stride = height * width * channels;
    let npix = height * width;

    for f in 0..frames {
        let base_f = f * frame_stride;
        let mut luma = vec![0.0_f64; npix];
        let mut rg = vec![0.0_f64; npix];
        let mut bg = vec![0.0_f64; npix];

        for idx in 0..npix {
            let p = base_f + idx * channels;
            let r = images.data[p] as f64;
            let g = images.data[p + 1] as f64;
            let b = images.data[p + 2] as f64;
            if !r.is_finite() || !g.is_finite() || !b.is_finite() {
                return Err(PrefixStreamError::InvalidInput(
                    "Image region contains non-finite values".into(),
                ));
            }
            luma[idx] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            rg[idx] = r - g;
            bg[idx] = b - g;
        }

        let channel_buffers = [
            ("luma", luma),
            ("red_minus_green", rg),
            ("blue_minus_green", bg),
        ];

        for (ch_name, buf) in channel_buffers {
            let detrended = remove_linear_trend(&buf, height, width);
            let power = fft2_power_spectrum(&detrended, height, width, &window_2d);
            let ch_entry = result.get_mut(ch_name).unwrap();

            for (label, min_wl, max_wl) in active_bands {
                let mut sum_p = 0.0_f64;
                let lo_freq = 1.0 / *max_wl;
                let hi_freq = 1.0 / *min_wl;
                for i in 0..(height * width) {
                    let rad = radius_grid[i];
                    if rad >= lo_freq && rad < hi_freq {
                        sum_p += power[i];
                    }
                }
                let rms = (sum_p / normalizer).sqrt();
                if let Some(vec) = ch_entry.get_mut(*label) {
                    vec.push(rms);
                }
            }
        }
    }

    Ok(result)
}

/// Summary report comparing start and end windows of a video clip for texture drift.
///
/// Mirrors `summarize_clip` in `tools/analyze_texture_drift.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextureDriftReport {
    /// Total frames analyzed.
    pub frames: usize,
    /// Frame height.
    pub height: usize,
    /// Frame width.
    pub width: usize,
    /// Window size used for start/end comparison.
    pub window: usize,
    /// Per-channel, per-band start/end RMS and ratio (`end_rms / start_rms`).
    pub channels: HashMap<String, HashMap<String, BandDriftSummary>>,
}

/// Drift summary for a single frequency band.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandDriftSummary {
    /// Mean RMS in the start window.
    pub start_rms: f64,
    /// Mean RMS in the end window.
    pub end_rms: f64,
    /// Ratio `end_rms / max(start_rms, 1e-12)`.
    pub end_over_start: f64,
}

/// Compute a [`TextureDriftReport`] over a `[F, H, W, C]` video tensor.
pub fn summarize_clip_texture_drift(images: &Tensor, window: usize) -> Result<TextureDriftReport> {
    let metrics = texture_band_rms(images, None)?;
    let frames = images.shape[0];
    let height = images.shape[1];
    let width = images.shape[2];
    let win = window.max(1).min(frames);

    let mut channels_summary = HashMap::new();
    for (channel, bands) in metrics {
        let mut band_map = HashMap::new();
        for (band, values) in bands {
            let start_mean = values[..win].iter().sum::<f64>() / (win as f64);
            let end_mean = values[frames - win..].iter().sum::<f64>() / (win as f64);
            let ratio = end_mean / start_mean.max(1e-12);
            band_map.insert(
                band,
                BandDriftSummary {
                    start_rms: start_mean,
                    end_rms: end_mean,
                    end_over_start: ratio,
                },
            );
        }
        channels_summary.insert(channel, band_map);
    }

    Ok(TextureDriftReport {
        frames,
        height,
        width,
        window: win,
        channels: channels_summary,
    })
}
