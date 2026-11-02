//! Pixel-space reference noise for continuation prefixes.
//!
//! Adds small temporally ramped noise to reference frames before VAE encoding
//! to suppress static texture carryover across continuation boundaries.
//!
//! Rust port of `pipeline/context_noise.py`.

use std::f32::consts::PI;

use crate::error::{PrefixStreamError, Result};
use crate::pipeline::native_masked_av::MaskedAvPlan;
use crate::pipeline::seam_protector::standardize_image_tensor;
use crate::tensor::{Tensor, VideoVae};

/// Build per-frame cosine noise strengths `[0.4, ..., 0.0]` across a reference frame window.
///
/// Mirrors `reference_noise_strengths(frame_count)` in `pipeline/context_noise.py`:
/// `0.2 * (1.0 + cos(pi * i / (frame_count - 1)))`, or `[0.0; frame_count]` if `frame_count <= 1`.
pub fn reference_noise_cosine_strengths(frame_count: usize) -> Vec<f32> {
    if frame_count <= 1 {
        return vec![0.0; frame_count];
    }
    (0..frame_count)
        .map(|i| 0.2 * (1.0 + (PI * (i as f32) / ((frame_count - 1) as f32)).cos()))
        .collect()
}

/// Build per-frame noise strengths across a reference frame window with configurable `strength` and `end_scale`.
///
/// When `strength == 0.4` and `end_scale == 0.0`, uses the exact cosine taper `0.2 * (1 + cos(pi * t))`
/// from `pipeline/context_noise.py`.
pub fn reference_noise_strengths(frame_count: usize, strength: f32, end_scale: f32) -> Vec<f32> {
    if frame_count == 0 {
        return Vec::new();
    }
    let s = strength.max(0.0);
    let e = end_scale.clamp(0.0, 1.0);
    if frame_count == 1 {
        return vec![s * e];
    }
    let mut weights = Vec::with_capacity(frame_count);
    for i in 0..frame_count {
        let t = (i as f32) / ((frame_count - 1) as f32);
        let cosine_w = 0.5 * (1.0 + (PI * t).cos());
        let w = e + (1.0 - e) * cosine_w;
        weights.push(s * w);
    }
    weights
}

/// Add deterministic cosine-tapered pixel noise to `[F, H, W, C]` frames in `[0, 1]`.
///
/// Preserves the final pixel frame (`frames[-1]`) when `end_scale == 0.0` and leaves any alpha channel (`c >= 3`) untouched.
/// Mirrors `noise_reference_frames` in `pipeline/context_noise.py`.
pub fn noise_reference_frames(
    images: &Tensor,
    strength: f32,
    seed: u64,
    temporal_correlation: f32,
    end_scale: f32,
) -> Result<Tensor> {
    if images.ndim() != 4 {
        return Err(PrefixStreamError::InvalidInput(format!(
            "Expected images with shape [F, H, W, C], got {:?}",
            images.shape
        )));
    }
    if strength <= 0.0 || images.shape[0] == 0 || (images.shape[0] == 1 && end_scale <= 0.0) {
        return Ok(images.clone());
    }

    let frames = images.shape[0];
    let h = images.shape[1];
    let w = images.shape[2];
    let c = images.shape[3];
    let rgb_c = c.min(3);
    let frame_stride = h * w * c;

    let rho = temporal_correlation.clamp(0.0, 0.98);
    let innovation_scale = (1.0 - rho * rho).max(0.0).sqrt();

    let mut raw_noise = Tensor::randn(&images.shape, seed).data;

    if rho > 0.0 && frames > 1 {
        for f in 1..frames {
            let prev_base = (f - 1) * frame_stride;
            let cur_base = f * frame_stride;
            for i in 0..frame_stride {
                raw_noise[cur_base + i] =
                    rho * raw_noise[prev_base + i] + innovation_scale * raw_noise[cur_base + i];
            }
        }
    }

    let weights = reference_noise_strengths(frames, strength, end_scale);
    let mut out_data = images.data.clone();
    let n_rgb_elem = (h * w * rgb_c) as f32;

    for (f, &weight) in weights.iter().enumerate() {
        if weight <= 0.0 {
            continue;
        }
        let base = f * frame_stride;
        let slice = &mut raw_noise[base..base + frame_stride];
        // Compute mean and std over RGB channels only
        let mut sum = 0.0_f32;
        for pix in 0..(h * w) {
            for ch in 0..rgb_c {
                sum += slice[pix * c + ch];
            }
        }
        let mean = sum / n_rgb_elem.max(1.0);
        let mut var_sum = 0.0_f32;
        for pix in 0..(h * w) {
            for ch in 0..rgb_c {
                let centered = slice[pix * c + ch] - mean;
                slice[pix * c + ch] = centered;
                var_sum += centered * centered;
            }
        }
        let std = (var_sum / n_rgb_elem.max(1.0)).sqrt().max(1e-6);
        for pix in 0..(h * w) {
            for ch in 0..rgb_c {
                let idx = base + pix * c + ch;
                let noisy = out_data[idx] + (slice[pix * c + ch] / std) * weight;
                out_data[idx] = noisy.clamp(0.0, 1.0);
            }
        }
    }

    Tensor::from_vec(images.shape.clone(), out_data)
}

/// Apply `E(noisy pixels) - E(clean pixels)` to the original selected latent slice,
/// cancelling clean decode/encode round-trip codec bias.
///
/// Mirrors `encode_noisy_prefix(vae, source_video, plan)` in `pipeline/context_noise.py`.
pub fn encode_noisy_prefix_with_plan(
    vae: &dyn VideoVae,
    source_video: &Tensor,
    plan: &MaskedAvPlan,
    strength: f32,
    seed: u64,
) -> Result<Tensor> {
    let decoded = standardize_image_tensor(&vae.decode(source_video)?)?;
    if decoded.shape[0] < plan.source_end_frame {
        return Err(PrefixStreamError::InvalidInput(
            "Reference noise: VAE decoded fewer frames than the selected source window".into(),
        ));
    }
    let window = decoded.slice_dim0(plan.source_start_frame, plan.source_end_frame)?;
    let pixels = if window.shape[3] > 3 {
        window.slice_axis(3, 0, 3)?
    } else {
        window
    };
    let clean_encoded = vae.encode(&pixels)?;
    let noisy_pixels = noise_reference_frames(&pixels, strength, seed, 0.0, 0.0)?;
    let noisy_encoded = vae.encode(&noisy_pixels)?;
    let expected = vec![
        source_video.shape[0],
        source_video.shape[1],
        plan.context_steps,
        source_video.shape[3],
        source_video.shape[4],
    ];
    if clean_encoded.shape != expected || noisy_encoded.shape != expected {
        return Err(PrefixStreamError::InvalidInput(format!(
            "Reference noise requires the matching H3 video VAE: expected {:?}, got clean={:?}, noisy={:?}",
            expected, clean_encoded.shape, noisy_encoded.shape
        )));
    }
    let original = source_video.slice_axis(2, plan.start_t, plan.end_t)?;
    let mut out_data = Vec::with_capacity(original.numel());
    for i in 0..original.numel() {
        out_data.push(original.data[i] + (noisy_encoded.data[i] - clean_encoded.data[i]));
    }
    Tensor::from_vec(expected, out_data)
}

/// VAE-encode noisy reference frames into a `5D` `[B, C, T, H, W]` latent prefix.
///
/// Mirrors `encode_noisy_prefix` in `pipeline/context_noise.py`.
pub fn encode_noisy_prefix(
    vae: &dyn VideoVae,
    images: &Tensor,
    frame_count: usize,
    strength: f32,
    seed: u64,
    temporal_correlation: f32,
    end_scale: f32,
) -> Result<(Tensor, Tensor)> {
    if images.ndim() != 4 {
        return Err(PrefixStreamError::InvalidInput(format!(
            "Expected images with shape [F, H, W, C], got {:?}",
            images.shape
        )));
    }
    let count = frame_count.max(1).min(images.shape[0]);
    let total_f = images.shape[0];
    let clip = images.slice_dim0(total_f - count, total_f)?;
    let noisy = noise_reference_frames(&clip, strength, seed, temporal_correlation, end_scale)?;
    let rgb = if noisy.shape[3] > 3 {
        noisy.slice_axis(3, 0, 3)?
    } else {
        noisy.clone()
    };

    let mut encoded = vae.encode(&rgb)?;
    if encoded.ndim() == 4 {
        encoded = Tensor::from_vec(
            vec![
                1,
                encoded.shape[0],
                encoded.shape[1],
                encoded.shape[2],
                encoded.shape[3],
            ],
            encoded.data,
        )?;
    }
    if encoded.ndim() != 5 {
        return Err(PrefixStreamError::InvalidInput(format!(
            "VAE.encode returned unsupported latent shape {:?}",
            encoded.shape
        )));
    }
    Ok((encoded, noisy))
}
