//! Seam Protector & Audio-Video Seamless Handover for Long Video Chaining.
//!
//! Provides:
//! 1. Pixel-space video stitching with luminance matching and cosine S-curve crossfade.
//! 2. Waveform-space audio stitching with sample-accurate alignment and complementary cosine crossfade.
//! 3. Synchronous frame & sample trimming for decoded `IMAGE` and `AUDIO` streams.
//! 4. Latent-space soft overlap blending.
//!
//! Rust port of `pipeline/seam_protector.py`.

use std::f32::consts::PI;

use crate::error::{PrefixStreamError, Result};
use crate::tensor::{AudioData, Tensor};

/// Round a floating-point value using Python's round-half-to-even (`round()`) semantics.
fn py_round(val: f64) -> usize {
    val.round_ties_even().max(0.0) as usize
}

/// Generate `n` linearly spaced points in `[start, end]`, matching `torch.linspace`.
fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / ((n - 1) as f32);
    (0..n).map(|i| start + (i as f32) * step).collect()
}

/// Blend two equal-length audio slices `(tail1, head2)` using an equal-power cosine/sine curve.
pub fn audio_equal_power_crossfade(tail1: &Tensor, head2: &Tensor) -> Result<Tensor> {
    let last_dim = tail1.ndim().saturating_sub(1);
    let c = tail1.shape[last_dim].min(head2.shape[last_dim]);
    if c == 0 {
        return Tensor::concat(&[tail1, head2], last_dim);
    }
    let t = linspace(0.0, PI / 2.0, c);
    let mut out = tail1.clone();
    let outer = tail1.numel() / c;
    for o in 0..outer {
        let base = o * c;
        for (i, &ti) in t.iter().enumerate().take(c) {
            let fade_out = ti.cos();
            let fade_in = ti.sin();
            out.data[base + i] = tail1.data[base + i] * fade_out + head2.data[base + i] * fade_in;
        }
    }
    Ok(out)
}

/// Concatenate two audio waveforms with an equal-power cosine crossfade over `crossfade_samples`.
///
/// Mirrors `audio_equal_power_crossfade(wave1, wave2, crossfade_samples)` in `pipeline/seam_protector.py`.
pub fn audio_equal_power_crossfade_concat(
    wave1: &Tensor,
    wave2: &Tensor,
    crossfade_samples: usize,
) -> Result<Tensor> {
    let last_dim = wave1.ndim().saturating_sub(1);
    let n1 = wave1.shape[last_dim];
    let n2 = wave2.shape[last_dim];
    let c = crossfade_samples.min(n1).min(n2);
    if c == 0 {
        return Tensor::concat(&[wave1, wave2], last_dim);
    }

    let tail1 = wave1.slice_axis(last_dim, n1 - c, n1)?;
    let head2 = wave2.slice_axis(last_dim, 0, c)?;
    let blended = audio_equal_power_crossfade(&tail1, &head2)?;

    let mut parts = Vec::new();
    if n1 > c {
        parts.push(wave1.slice_axis(last_dim, 0, n1 - c)?);
    }
    parts.push(blended);
    if n2 > c {
        parts.push(wave2.slice_axis(last_dim, c, n2)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::concat(&refs, last_dim)
}

/// Pad or truncate an audio waveform along its last axis to `target_samples` with a 32-sample micro fade-out
/// before zero padding to avoid DC clicks.
///
/// Mirrors `fit_audio_length` in `pipeline/seam_protector.py`.
pub fn fit_audio_length(waveform: &Tensor, target_samples: usize) -> Result<Tensor> {
    if waveform.ndim() == 0 {
        return Ok(waveform.clone());
    }
    let last_dim = waveform.ndim() - 1;
    let current = waveform.shape[last_dim];
    if current == target_samples {
        return Ok(waveform.clone());
    }
    if current > target_samples {
        return waveform.slice_axis(last_dim, 0, target_samples);
    }
    if target_samples == 0 {
        return waveform.slice_axis(last_dim, 0, 0);
    }
    if current == 0 {
        let mut shape = waveform.shape.clone();
        shape[last_dim] = target_samples;
        return Ok(Tensor::zeros(shape));
    }

    let fade_len = current.min(32);
    let mut faded = waveform.clone();
    if fade_len > 0 {
        let t = linspace(0.0, PI, fade_len);
        let fade_curve: Vec<f32> = t.into_iter().map(|x| 0.5 + 0.5 * x.cos()).collect();
        let outer = faded.numel() / current;
        for o in 0..outer {
            let base = o * current + (current - fade_len);
            for (i, &fc) in fade_curve.iter().enumerate().take(fade_len) {
                faded.data[base + i] *= fc;
            }
        }
    }

    let mut pad_shape = waveform.shape.clone();
    pad_shape[last_dim] = target_samples - current;
    let zero_pad = Tensor::zeros(pad_shape);
    Tensor::concat(&[&faded, &zero_pad], last_dim)
}

/// Estimate conservative global RGB luminance gain between time-overlapping `[F, H, W, C]` clips.
///
/// Mirrors `estimate_luminance_gain` in `pipeline/seam_protector.py`.
pub fn estimate_luminance_gain(
    previous_context: &Tensor,
    next_context: &Tensor,
    max_correction_percent: f32,
) -> f32 {
    if previous_context.shape != next_context.shape
        || previous_context.ndim() != 4
        || previous_context.shape[0] == 0
        || previous_context.shape[3] < 3
    {
        return 1.0;
    }

    let frames = previous_context.shape[0];
    let h = previous_context.shape[1];
    let w = previous_context.shape[2];
    let c = previous_context.shape[3];
    let f_stride = h * w * c;

    let sub_h = h.div_ceil(4);
    let sub_w = w.div_ceil(4);
    let min_valid = ((sub_h * sub_w) as f32 * 0.01) as usize;
    let min_valid = min_valid.max(4);

    let mut ratios = Vec::new();
    for f in 0..frames {
        let base = f * f_stride;
        let mut a_sum = 0.0_f64;
        let mut b_sum = 0.0_f64;
        let mut valid_count = 0usize;

        for y in (0..h).step_by(4) {
            for x in (0..w).step_by(4) {
                let p = base + (y * w + x) * c;
                let a = 0.2126 * previous_context.data[p]
                    + 0.7152 * previous_context.data[p + 1]
                    + 0.0722 * previous_context.data[p + 2];
                let b = 0.2126 * next_context.data[p]
                    + 0.7152 * next_context.data[p + 1]
                    + 0.0722 * next_context.data[p + 2];
                if a > 0.04 && a < 0.96 && b > 0.04 && b < 0.96 {
                    a_sum += a as f64;
                    b_sum += b as f64;
                    valid_count += 1;
                }
            }
        }

        if valid_count >= min_valid {
            let a_mean = a_sum / (valid_count as f64);
            let b_mean = b_sum / (valid_count as f64);
            if b_mean > 1e-6 {
                ratios.push((a_mean / b_mean) as f32);
            }
        }
    }

    if ratios.is_empty() {
        return 1.0;
    }

    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Match torch.median() which returns the lower of the two middle elements for even length
    let mid_idx = (ratios.len() - 1) / 2;
    let measured = ratios[mid_idx];
    let limit = max_correction_percent.max(0.0) / 100.0;
    measured.clamp((1.0 - limit).max(0.01), 1.0 + limit)
}

/// Smoothly apply luminance `gain` at the start of `images` and fade back to `1.0` over `fade_frames`.
///
/// Mirrors `apply_luminance_gain_fade` in `pipeline/seam_protector.py`.
pub fn apply_luminance_gain_fade(images: &Tensor, gain: f32, fade_frames: usize) -> Result<Tensor> {
    if images.ndim() != 4 {
        return Err(PrefixStreamError::InvalidInput(format!(
            "Expected 4D images [F, H, W, C], got {:?}",
            images.shape
        )));
    }
    let n = fade_frames.min(images.shape[0]);
    if n == 0 || (gain - 1.0).abs() < 1e-6 {
        return Ok(images.clone());
    }

    let mut out = images.clone();
    let h = out.shape[1];
    let w = out.shape[2];
    let c = out.shape[3];
    let rgb_c = c.min(3);
    let f_stride = h * w * c;

    let t = linspace(0.0, 1.0, n);
    for (f, &ti) in t.iter().enumerate() {
        let weight = 0.5 + 0.5 * (PI * ti).cos();
        let scale = 1.0 + (gain - 1.0) * weight;
        let base = f * f_stride;
        for pix in 0..(h * w) {
            let p = base + pix * c;
            for ch in 0..rgb_c {
                out.data[p + ch] = (out.data[p + ch] * scale).clamp(0.0, 1.0);
            }
        }
    }
    Ok(out)
}

/// Standardize a video image tensor to canonical ComfyUI 4D shape `[F, H, W, C]`.
///
/// Mirrors `_standardize_image_tensor` in `pipeline/seam_protector.py`.
pub fn standardize_image_tensor(images: &Tensor) -> Result<Tensor> {
    match images.ndim() {
        4 => Ok(images.clone()),
        3 => Tensor::from_vec(
            vec![1, images.shape[0], images.shape[1], images.shape[2]],
            images.data.clone(),
        ),
        5 => {
            let b = images.shape[0];
            let d1 = images.shape[1];
            let d2 = images.shape[2];
            let d3 = images.shape[3];
            let d4 = images.shape[4];
            if b == 1 && matches!(d4, 1 | 3 | 4) {
                Tensor::from_vec(vec![d1, d2, d3, d4], images.data.clone())
            } else if matches!(d1, 1 | 3 | 4) {
                // [B, C, F, H, W] -> [B*F, H, W, C]
                let c = d1;
                let f = d2;
                let h = d3;
                let w = d4;
                let mut out = vec![0.0_f32; b * f * h * w * c];
                for bi in 0..b {
                    for ci in 0..c {
                        for fi in 0..f {
                            for hi in 0..h {
                                for wi in 0..w {
                                    let src = ((((bi * c + ci) * f + fi) * h + hi) * w) + wi;
                                    let dst = ((((bi * f + fi) * h + hi) * w + wi) * c) + ci;
                                    out[dst] = images.data[src];
                                }
                            }
                        }
                    }
                }
                Tensor::from_vec(vec![b * f, h, w, c], out)
            } else {
                Tensor::from_vec(vec![b * d1, d2, d3, d4], images.data.clone())
            }
        }
        _ => Err(PrefixStreamError::InvalidInput(format!(
            "Unsupported image tensor shape {:?}",
            images.shape
        ))),
    }
}

/// Seamlessly join two decoded video `IMAGE` tensors `[F, H, W, C]` in pixel space.
///
/// Mirrors `stitch_video_images` in `pipeline/seam_protector.py`.
pub fn stitch_video_images(
    prev_images: Option<&Tensor>,
    curr_images: &Tensor,
    trim_frames: usize,
    crossfade_frames: usize,
    luminance_match: bool,
    luminance_fade_frames: usize,
) -> Result<Tensor> {
    let curr = standardize_image_tensor(curr_images)?;
    let prev_opt = match prev_images {
        Some(p) => Some(standardize_image_tensor(p)?),
        None => None,
    };

    let head = trim_frames;
    if head >= curr.shape[0] {
        return match prev_opt {
            Some(p) => Ok(p),
            None => curr.slice_dim0(0, 0),
        };
    }

    let Some(prev) = prev_opt.filter(|p| p.shape[0] > 0) else {
        return curr.slice_dim0(head, curr.shape[0]);
    };

    let mut curr_body = curr.slice_dim0(head, curr.shape[0])?;
    let mut gain = 1.0_f32;

    if luminance_match && head > 0 && prev.shape[0] > 0 {
        let analysis_n = 8usize.min(head).min(prev.shape[0]);
        let prev_ctx = prev.slice_dim0(prev.shape[0] - analysis_n, prev.shape[0])?;
        let curr_ctx = curr.slice_dim0(head - analysis_n, head)?;
        gain = estimate_luminance_gain(&prev_ctx, &curr_ctx, 10.0);
        curr_body = apply_luminance_gain_fade(&curr_body, gain, luminance_fade_frames)?;
    }

    let n = crossfade_frames.min(head).min(prev.shape[0]);
    if n == 0 {
        return Tensor::concat(&[&prev, &curr_body], 0);
    }

    let prev_prefix = prev.slice_dim0(0, prev.shape[0] - n)?;
    let prev_tail = prev.slice_dim0(prev.shape[0] - n, prev.shape[0])?;
    let mut curr_overlap = curr.slice_dim0(head - n, head)?;
    if luminance_match && (gain - 1.0).abs() > 1e-6 {
        let h = curr_overlap.shape[1];
        let w = curr_overlap.shape[2];
        let c = curr_overlap.shape[3];
        let rgb_c = c.min(3);
        for pix in 0..(n * h * w) {
            let base = pix * c;
            for ch in 0..rgb_c {
                curr_overlap.data[base + ch] =
                    (curr_overlap.data[base + ch] * gain).clamp(0.0, 1.0);
            }
        }
    }

    let t = linspace(0.0, 1.0, n);
    let f_stride = prev_tail.shape[1] * prev_tail.shape[2] * prev_tail.shape[3];
    let mut blended_data = vec![0.0_f32; n * f_stride];
    for (f, &ti) in t.iter().enumerate() {
        let alpha = 0.5 - 0.5 * (PI * ti).cos();
        let base = f * f_stride;
        for i in 0..f_stride {
            blended_data[base + i] =
                prev_tail.data[base + i] * (1.0 - alpha) + curr_overlap.data[base + i] * alpha;
        }
    }
    let blended = Tensor::from_vec(prev_tail.shape.clone(), blended_data)?;

    let mut parts = Vec::new();
    if prev_prefix.shape[0] > 0 {
        parts.push(&prev_prefix);
    }
    parts.push(&blended);
    if curr_body.shape[0] > 0 {
        parts.push(&curr_body);
    }
    Tensor::concat(&parts, 0)
}

/// Synchronously trim leading overlap frames from `IMAGE` and matching samples from `AUDIO`.
///
/// Mirrors `trim_images_and_audio` in `pipeline/seam_protector.py`.
pub fn trim_images_and_audio(
    images: &Tensor,
    audio: Option<&AudioData>,
    trim_frames: usize,
    fps: f32,
    match_tail: bool,
) -> Result<(Tensor, Option<AudioData>)> {
    let std_images = standardize_image_tensor(images)?;
    let total_frames = std_images.shape[0];
    let n = trim_frames;

    let trimmed_images = if n >= total_frames {
        std_images.slice_dim0(0, 0)?
    } else if n > 0 {
        std_images.slice_dim0(n, total_frames)?
    } else {
        std_images
    };

    let trimmed_audio = if let Some(aud) = audio {
        let sr = if aud.sample_rate > 0 {
            aud.sample_rate
        } else {
            32000
        };
        let wf = &aud.waveform;
        let last_dim = wf.ndim().saturating_sub(1);
        let total_samples = wf.shape[last_dim];
        let head_samples = py_round((n as f64) / (fps as f64) * (sr as f64));
        let kept_frames = total_frames.saturating_sub(n);

        let w = if match_tail {
            let want_samples = py_round((kept_frames as f64) / (fps as f64) * (sr as f64));
            let s0 = head_samples.min(total_samples);
            let s1 = (head_samples + want_samples).min(total_samples);
            let sliced = wf.slice_axis(last_dim, s0, s1)?;
            fit_audio_length(&sliced, want_samples)?
        } else {
            let s0 = head_samples.min(total_samples);
            wf.slice_axis(last_dim, s0, total_samples)?
        };
        Some(AudioData::new(w, sr))
    } else {
        None
    };

    Ok((trimmed_images, trimmed_audio))
}

/// Context-aligned audio stitch with complementary `sin^2` / `cos^2` crossfade and sample-accurate timeline sync.
///
/// Mirrors `stitch_audio_waveforms` in `pipeline/seam_protector.py`.
pub fn stitch_audio_waveforms(
    prev_audio: Option<&AudioData>,
    curr_audio: Option<&AudioData>,
    curr_total_frames: usize,
    trim_frames: usize,
    crossfade_ms: f32,
    fps: f32,
) -> Result<Option<AudioData>> {
    let Some(curr) = curr_audio else {
        return Ok(prev_audio.cloned());
    };

    let sr = if curr.sample_rate > 0 {
        curr.sample_rate
    } else {
        32000
    };
    let head_samples = py_round((trim_frames as f64) / (fps as f64) * (sr as f64));
    let kept_frames = curr_total_frames.saturating_sub(trim_frames);
    let want_samples = py_round((kept_frames as f64) / (fps as f64) * (sr as f64));

    let curr_w = &curr.waveform;
    let last_dim = curr_w.ndim().saturating_sub(1);
    let curr_len = curr_w.shape[last_dim];
    let s0 = head_samples.min(curr_len);
    let s1 = (head_samples + want_samples).min(curr_len);
    let curr_sliced = curr_w.slice_axis(last_dim, s0, s1)?;
    let curr_body = fit_audio_length(&curr_sliced, want_samples)?;

    let Some(prev) = prev_audio else {
        return Ok(Some(AudioData::new(curr_body, sr)));
    };

    let prev_sr = if prev.sample_rate > 0 {
        prev.sample_rate
    } else {
        32000
    };
    if prev_sr != sr {
        return Err(PrefixStreamError::InvalidInput(
            "Audio sample rates must match; resample clips before stitching".to_string(),
        ));
    }

    let prev_w = &prev.waveform;
    if prev_w.ndim() != curr_w.ndim() || prev_w.shape[..last_dim] != curr_w.shape[..last_dim] {
        return Err(PrefixStreamError::InvalidInput(
            "Audio batch size and channel count must match before stitching".to_string(),
        ));
    }

    let prev_len = prev_w.shape[last_dim];
    let mut c = py_round(((crossfade_ms as f64) / 1000.0) * (sr as f64))
        .min(head_samples)
        .min(prev_len);
    if head_samples > curr_len {
        c = 0;
    }

    let mut stitched = if c == 0 {
        Tensor::concat(&[prev_w, &curr_body], last_dim)?
    } else {
        let prev_tail = prev_w.slice_axis(last_dim, prev_len - c, prev_len)?;
        let curr_overlap = curr_w.slice_axis(last_dim, head_samples - c, head_samples)?;

        let t = linspace(0.0, PI / 2.0, c);
        let outer = prev_tail.numel() / c;
        let mut blended_data = vec![0.0_f32; prev_tail.numel()];
        for o in 0..outer {
            let base = o * c;
            for (i, &ti) in t.iter().enumerate() {
                let s = ti.sin();
                let fade_in = s * s;
                let fade_out = 1.0 - fade_in;
                blended_data[base + i] =
                    prev_tail.data[base + i] * fade_out + curr_overlap.data[base + i] * fade_in;
            }
        }
        let blended = Tensor::from_vec(prev_tail.shape.clone(), blended_data)?;

        let mut parts = Vec::new();
        if prev_len > c {
            parts.push(prev_w.slice_axis(last_dim, 0, prev_len - c)?);
        }
        parts.push(blended);
        if curr_body.shape[last_dim] > 0 {
            parts.push(curr_body);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::concat(&refs, last_dim)?
    };

    for v in &mut stitched.data {
        *v = v.clamp(-1.0, 1.0);
    }

    Ok(Some(AudioData::new(stitched, sr)))
}

/// Softly blend the tail of `latent1` with the head of `latent2` along the temporal axis `dim = 2`.
///
/// Mirrors `latent_soft_blend` in `pipeline/seam_protector.py`.
pub fn latent_soft_blend(latent1: &Tensor, latent2: &Tensor, blend_steps: usize) -> Result<Tensor> {
    if latent1.ndim() != 5 || latent2.ndim() != 5 {
        return Err(PrefixStreamError::InvalidInput(
            "latent_soft_blend expects 5D video latents [B, C, T, H, W]".to_string(),
        ));
    }
    let t1 = latent1.shape[2];
    let t2 = latent2.shape[2];
    let b = blend_steps.min(t1).min(t2);
    if b == 0 {
        return Tensor::concat(&[latent1, latent2], 2);
    }

    let alpha = linspace(0.0, 1.0, b);
    let overlap1 = latent1.slice_axis(2, t1 - b, t1)?;
    let overlap2 = latent2.slice_axis(2, 0, b)?;

    let batch_ch = overlap1.shape[0] * overlap1.shape[1];
    let spatial = overlap1.shape[3] * overlap1.shape[4];
    let mut blended_data = vec![0.0_f32; overlap1.numel()];
    for bc in 0..batch_ch {
        for (ti, &a) in alpha.iter().enumerate() {
            let base = (bc * b + ti) * spatial;
            for s in 0..spatial {
                blended_data[base + s] =
                    (1.0 - a) * overlap1.data[base + s] + a * overlap2.data[base + s];
            }
        }
    }
    let blended = Tensor::from_vec(overlap1.shape.clone(), blended_data)?;

    let mut parts = Vec::new();
    if t1 > b {
        parts.push(latent1.slice_axis(2, 0, t1 - b)?);
    }
    parts.push(blended);
    if t2 > b {
        parts.push(latent2.slice_axis(2, b, t2)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::concat(&refs, 2)
}

/// Stitch two contiguous 5D video latents `[B, C, T, H, W]`, seamlessly blending `overlap_steps`.
///
/// Mirrors `stitch_video_latents` in `pipeline/seam_protector.py`.
pub fn stitch_video_latents(
    prev_latent: &Tensor,
    curr_latent: &Tensor,
    overlap_steps: usize,
    blend_steps: usize,
) -> Result<Tensor> {
    if overlap_steps == 0 {
        return latent_soft_blend(prev_latent, curr_latent, blend_steps);
    }
    let t1 = prev_latent.shape[2];
    let t2 = curr_latent.shape[2];
    let ov = overlap_steps.min(t1).min(t2);
    if ov == 0 {
        return Tensor::concat(&[prev_latent, curr_latent], 2);
    }

    let prev_overlap = prev_latent.slice_axis(2, t1 - ov, t1)?;
    let curr_overlap = curr_latent.slice_axis(2, 0, ov)?;

    let t = linspace(0.0, PI / 2.0, ov);
    let batch_ch = prev_overlap.shape[0] * prev_overlap.shape[1];
    let spatial = prev_overlap.shape[3] * prev_overlap.shape[4];
    let mut blended_data = vec![0.0_f32; prev_overlap.numel()];

    for bc in 0..batch_ch {
        for (ti, &ang) in t.iter().enumerate() {
            let s = ang.sin();
            let alpha = s * s;
            let base = (bc * ov + ti) * spatial;
            for sp in 0..spatial {
                blended_data[base + sp] = (1.0 - alpha) * prev_overlap.data[base + sp]
                    + alpha * curr_overlap.data[base + sp];
            }
        }
    }
    let blended_overlap = Tensor::from_vec(prev_overlap.shape.clone(), blended_data)?;

    let mut parts = Vec::new();
    if t1 > ov {
        parts.push(prev_latent.slice_axis(2, 0, t1 - ov)?);
    }
    parts.push(blended_overlap);
    if t2 > ov {
        parts.push(curr_latent.slice_axis(2, ov, t2)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::concat(&refs, 2)
}

/// Stitch two contiguous 4D audio latents `[B, 32, 2, T]`, seamlessly blending `overlap_steps`.
///
/// Mirrors `stitch_audio_latents` in `pipeline/seam_protector.py`.
pub fn stitch_audio_latents(
    prev_latent: Option<&Tensor>,
    curr_latent: &Tensor,
    overlap_steps: usize,
) -> Result<Tensor> {
    let Some(prev) = prev_latent else {
        return Ok(curr_latent.clone());
    };
    if overlap_steps == 0 {
        return Ok(curr_latent.clone());
    }
    let last_dim = curr_latent.ndim() - 1;
    let t1 = prev.shape[last_dim];
    let t2 = curr_latent.shape[last_dim];
    let ov = overlap_steps.min(t1).min(t2);
    if ov == 0 {
        return Tensor::concat(&[prev, curr_latent], last_dim);
    }

    let prev_overlap = prev.slice_axis(last_dim, t1 - ov, t1)?;
    let curr_overlap = curr_latent.slice_axis(last_dim, 0, ov)?;

    let t = linspace(0.0, PI / 2.0, ov);
    let outer = prev_overlap.numel() / ov;
    let mut blended_data = vec![0.0_f32; prev_overlap.numel()];
    for o in 0..outer {
        let base = o * ov;
        for (i, &ang) in t.iter().enumerate() {
            let s = ang.sin();
            let alpha = s * s;
            blended_data[base + i] =
                (1.0 - alpha) * prev_overlap.data[base + i] + alpha * curr_overlap.data[base + i];
        }
    }
    let blended = Tensor::from_vec(prev_overlap.shape.clone(), blended_data)?;

    let mut parts = Vec::new();
    if t1 > ov {
        parts.push(prev.slice_axis(last_dim, 0, t1 - ov)?);
    }
    parts.push(blended);
    if t2 > ov {
        parts.push(curr_latent.slice_axis(last_dim, ov, t2)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::concat(&refs, last_dim)
}

/// Trim leading prefix/condition latent steps from a 5D video latent `[B, C, T, H, W]`.
pub fn trim_prefix_frames(
    full_video_latent: &Tensor,
    prefix_latent_steps: usize,
) -> Result<Tensor> {
    if prefix_latent_steps == 0 {
        return Ok(full_video_latent.clone());
    }
    let t = full_video_latent.shape[2];
    full_video_latent.slice_axis(2, prefix_latent_steps.min(t), t)
}

/// Trim leading samples from an audio waveform along its last axis.
pub fn trim_audio_waveform(waveform: &Tensor, trim_samples: usize) -> Result<Tensor> {
    if trim_samples == 0 {
        return Ok(waveform.clone());
    }
    let last_dim = waveform.ndim() - 1;
    let total = waveform.shape[last_dim];
    waveform.slice_axis(last_dim, trim_samples.min(total), total)
}

/// Trim leading overlap steps from a 4D audio latent `[B, 32, 2, T]`.
pub fn trim_audio_latents(full_audio_latent: &Tensor, prefix_audio_steps: usize) -> Result<Tensor> {
    if prefix_audio_steps == 0 {
        return Ok(full_audio_latent.clone());
    }
    let last_dim = full_audio_latent.ndim() - 1;
    let total = full_audio_latent.shape[last_dim];
    full_audio_latent.slice_axis(last_dim, prefix_audio_steps.min(total), total)
}
