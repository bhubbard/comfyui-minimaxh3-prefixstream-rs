//! Pixel-space reference noise for continuation prefixes.
//!
//! Adds small temporally ramped noise to reference frames before VAE encoding
//! to suppress static texture carryover across continuation boundaries.
//!
//! Rust port of `pipeline/context_noise.py`.

use crate::error::{PrefixStreamError, Result};
use crate::tensor::{Tensor, VideoVae};

/// Build per-frame noise strengths across a reference frame window.
///
/// Earlier context frames receive `strength`, ramping down to `strength * end_scale` at the handover seam.
///
/// Mirrors `reference_noise_strengths` in `pipeline/context_noise.py`.
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
        let w = 1.0 + (e - 1.0) * t;
        weights.push(s * w);
    }
    weights
}

/// Add deterministic zero-mean temporally correlated pixel noise to `[F, H, W, C]` frames in `[0, 1]`.
///
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
    if strength <= 0.0 || images.shape[0] == 0 {
        return Ok(images.clone());
    }

    let frames = images.shape[0];
    let h = images.shape[1];
    let w = images.shape[2];
    let c = images.shape[3];
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

    // Center and normalize each frame across (H, W, C)
    let weights = reference_noise_strengths(frames, strength, end_scale);
    let mut out_data = images.data.clone();
    let n_elem = frame_stride as f32;

    for (f, &weight) in weights.iter().enumerate() {
        let base = f * frame_stride;
        let slice = &mut raw_noise[base..base + frame_stride];
        let mean: f32 = slice.iter().sum::<f32>() / n_elem;
        for v in slice.iter_mut() {
            *v -= mean;
        }
        let var: f32 = slice.iter().map(|&v| v * v).sum::<f32>() / n_elem;
        let std = var.sqrt().max(1e-6);
        for i in 0..frame_stride {
            let noisy = out_data[base + i] + (slice[i] / std) * weight;
            out_data[base + i] = noisy.clamp(0.0, 1.0);
        }
    }

    Tensor::from_vec(images.shape.clone(), out_data)
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
