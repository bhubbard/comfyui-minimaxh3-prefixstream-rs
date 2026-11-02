//! Non-destructive video chunk slicing, incremental SafeTensors patch storage, and timeline reassembly.
//!
//! Rust port of `engine/timeline_session_manager.py`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::Local;
use image::{Rgb, RgbImage};
use regex::Regex;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};

use crate::engine::clip_bin_manager::{
    atomic_write_json, load_multi_safetensors, save_multi_safetensors,
};
use crate::error::{PrefixStreamError, Result};
use crate::pipeline::seam_protector::{
    apply_luminance_gain_fade, audio_equal_power_crossfade, estimate_luminance_gain,
};
use crate::tensor::{AudioData, Tensor};

/// Resolve the base directory for Timeline Sessions (`MINIMAX_TIMELINE_DIR` or `./output/MiniMax_TimelineSessions`).
pub fn get_timeline_sessions_base_dir() -> PathBuf {
    let base = if let Ok(env_dir) = std::env::var("MINIMAX_TIMELINE_DIR") {
        PathBuf::from(env_dir)
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("output")
            .join("MiniMax_TimelineSessions")
    };
    let _ = fs::create_dir_all(&base);
    base
}

/// Sanitize a session ID and resolve its directory under `base_dir`, rejecting path traversal and symlinks.
pub fn get_session_dir(base_dir: &Path, session_id: &str) -> Result<PathBuf> {
    let re = Regex::new(r"[^a-zA-Z0-9_\-]").unwrap();
    let safe_id = re.replace_all(session_id.trim(), "_").into_owned();
    let safe_id = if safe_id.is_empty() || safe_id == "." || safe_id == ".." {
        "Default_Timeline_Session".to_string()
    } else {
        safe_id
    };

    fs::create_dir_all(base_dir)?;
    let base_real = fs::canonicalize(base_dir)?;
    let sess_dir = base_dir.join(&safe_id);

    if let Ok(meta) = fs::symlink_metadata(&sess_dir) {
        if meta.file_type().is_symlink() {
            return Err(PrefixStreamError::SecurityViolation(format!(
                "Security Violation: Session directory is a symlink ({safe_id})"
            )));
        }
    }

    fs::create_dir_all(&sess_dir)?;
    let sess_real = fs::canonicalize(&sess_dir)?;
    if !sess_real.starts_with(&base_real) || sess_real == base_real {
        return Err(PrefixStreamError::SecurityViolation(format!(
            "Security Violation: Session directory escaped base ({safe_id})"
        )));
    }
    Ok(sess_dir)
}

/// Standardize an [`AudioData`] waveform to `[1, C, S]` float32.
///
/// Mirrors `standardize_audio_dict` in `engine/timeline_session_manager.py`.
pub fn standardize_audio_dict(audio: Option<&AudioData>, fallback_sr: u32) -> AudioData {
    let Some(aud) = audio else {
        return AudioData::new(Tensor::zeros(vec![1, 2, 1]), fallback_sr);
    };
    let sr = if aud.sample_rate > 0 {
        aud.sample_rate
    } else {
        fallback_sr
    };
    let wf = &aud.waveform;
    let standardized = match wf.ndim() {
        1 => {
            let s = wf.shape[0];
            Tensor::from_vec(vec![1, 1, s], wf.data.clone())
                .unwrap_or_else(|_| Tensor::zeros(vec![1, 2, 1]))
        }
        2 => {
            let d0 = wf.shape[0];
            let d1 = wf.shape[1];
            if d0 <= 8 && d1 > d0 {
                // [C, S] -> [1, C, S]
                Tensor::from_vec(vec![1, d0, d1], wf.data.clone())
                    .unwrap_or_else(|_| Tensor::zeros(vec![1, 2, 1]))
            } else {
                // [S, C] -> transpose to [1, C, S]
                let mut out = vec![0.0_f32; d0 * d1];
                for s in 0..d0 {
                    for c in 0..d1 {
                        out[c * d0 + s] = wf.data[s * d1 + c];
                    }
                }
                Tensor::from_vec(vec![1, d1, d0], out)
                    .unwrap_or_else(|_| Tensor::zeros(vec![1, 2, 1]))
            }
        }
        3 => wf.clone(),
        _ => Tensor::zeros(vec![1, 2, 1]),
    };

    AudioData::new(standardized, sr)
}

/// Result of [`slice_video_and_audio`].
#[derive(Debug, Clone)]
pub struct SlicedChunk {
    /// Sliced video frames `[chunk_F, H, W, C]`.
    pub chunk_images: Tensor,
    /// Synchronized audio slice `[1, C, chunk_S]`.
    pub chunk_audio: AudioData,
    /// Clamped start frame index.
    pub actual_start: usize,
    /// Clamped exclusive end frame index.
    pub actual_end: usize,
    /// Total frames in the source video.
    pub total_frames: usize,
}

/// Slice a `[F, H, W, C]` video tensor and its synchronized audio between `[start_frame, end_frame)`.
///
/// Mirrors `slice_video_and_audio` in `engine/timeline_session_manager.py`.
pub fn slice_video_and_audio(
    images: &Tensor,
    audio: Option<&AudioData>,
    fps: f32,
    start_frame: isize,
    end_frame: isize,
    pad_to_length: bool,
) -> Result<SlicedChunk> {
    if images.ndim() != 4 {
        return Err(PrefixStreamError::InvalidInput(format!(
            "images must have shape [F, H, W, C], got {:?}",
            images.shape
        )));
    }
    let total_frames = images.shape[0];
    let safe_fps = fps.max(1.0);

    let s_frame = start_frame
        .max(0)
        .min(total_frames.saturating_sub(1) as isize) as usize;
    let e_frame = if end_frame <= 0 || (end_frame as usize) > total_frames {
        total_frames
    } else {
        end_frame as usize
    };
    let e_frame = e_frame.max(s_frame + 1).min(total_frames);

    let mut chunk_images = images.slice_dim0(s_frame, e_frame)?;
    let requested_len = if end_frame > start_frame {
        (end_frame - start_frame) as usize
    } else {
        chunk_images.shape[0]
    };

    if pad_to_length && chunk_images.shape[0] < requested_len && chunk_images.shape[0] > 0 {
        let pad_needed = requested_len - chunk_images.shape[0];
        let last_f = chunk_images.last_frame();
        let mut parts = Vec::with_capacity(1 + pad_needed);
        parts.push(chunk_images);
        for _ in 0..pad_needed {
            parts.push(last_f.clone());
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        chunk_images = Tensor::concat(&refs, 0)?;
    }

    let std_aud = standardize_audio_dict(audio, 44100);
    let sr = std_aud.sample_rate;
    let wf = &std_aud.waveform;
    let total_samples = wf.shape[2];

    let start_sample = (((s_frame as f64) / (safe_fps as f64)) * (sr as f64)).round() as usize;
    let end_sample = (((e_frame as f64) / (safe_fps as f64)) * (sr as f64)).round() as usize;
    let target_samples =
        (((chunk_images.shape[0] as f64) / (safe_fps as f64)) * (sr as f64)).round() as usize;
    let target_samples = target_samples.max(1);

    let s_smp = start_sample.min(total_samples);
    let e_smp = end_sample.max(s_smp).min(total_samples);

    let mut chunk_wf = if s_smp < e_smp {
        wf.slice_axis(2, s_smp, e_smp)?
    } else {
        Tensor::zeros(vec![wf.shape[0], wf.shape[1], 1])
    };

    let cur_samples = chunk_wf.shape[2];
    if cur_samples < target_samples {
        let pad_s = target_samples - cur_samples;
        let pad_zeros = Tensor::zeros(vec![chunk_wf.shape[0], chunk_wf.shape[1], pad_s]);
        chunk_wf = Tensor::concat(&[&chunk_wf, &pad_zeros], 2)?;
    } else if cur_samples > target_samples {
        chunk_wf = chunk_wf.slice_axis(2, 0, target_samples)?;
    }

    Ok(SlicedChunk {
        chunk_images,
        chunk_audio: AudioData::new(chunk_wf, sr),
        actual_start: s_frame,
        actual_end: e_frame,
        total_frames,
    })
}

/// Render a timeline progress / coverage bar as a `[1, H, W, 3]` float32 tensor in `[0, 1]`.
///
/// Mirrors `render_timeline_indicator_image` in `engine/timeline_session_manager.py`.
pub fn render_timeline_indicator_image(
    total_frames: usize,
    active_range: Option<(usize, usize)>,
    patched_ranges: Option<&[(usize, usize)]>,
    fps: f32,
    title: &str,
) -> Result<Tensor> {
    let _ = (fps, title);
    let width = 768usize;
    let height = 140usize;
    let mut img = RgbImage::from_pixel(width as u32, height as u32, Rgb([20, 22, 28]));

    let bar_x0 = 24usize;
    let bar_x1 = width - 24;
    let bar_y0 = 44usize;
    let bar_y1 = 84usize;
    let bar_w = (bar_x1 - bar_x0).max(1);
    let tf = total_frames.max(1);

    // Base bar background
    for y in bar_y0..=bar_y1 {
        for x in bar_x0..=bar_x1 {
            img.put_pixel(x as u32, y as u32, Rgb([42, 46, 58]));
        }
    }

    let frame_to_x = |f: usize| -> usize {
        let ratio = (f as f64 / tf as f64).clamp(0.0, 1.0);
        bar_x0 + (ratio * (bar_w as f64)).round() as usize
    };

    // Draw patched ranges in emerald green
    if let Some(ranges) = patched_ranges {
        for &(ps, pe) in ranges {
            let x0 = frame_to_x(ps);
            let x1 = frame_to_x(pe).max(x0 + 2).min(bar_x1);
            for y in (bar_y0 + 2)..=(bar_y1 - 2) {
                for x in x0..=x1 {
                    img.put_pixel(x as u32, y as u32, Rgb([34, 197, 94]));
                }
            }
        }
    }

    // Draw active range in amber/cyan
    if let Some((as_f, ae_f)) = active_range {
        let x0 = frame_to_x(as_f);
        let x1 = frame_to_x(ae_f).max(x0 + 3).min(bar_x1);
        for y in bar_y0..=bar_y1 {
            for x in x0..=x1 {
                img.put_pixel(x as u32, y as u32, Rgb([245, 158, 11]));
            }
        }
    }

    let mut data = Vec::with_capacity(height * width * 3);
    for y in 0..height {
        for x in 0..width {
            let p = img.get_pixel(x as u32, y as u32);
            data.push(p[0] as f32 / 255.0);
            data.push(p[1] as f32 / 255.0);
            data.push(p[2] as f32 / 255.0);
        }
    }

    Tensor::from_vec(vec![1, height, width, 3], data)
}

/// Metadata for a single patched chunk inside a [`TimelineSession`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchMeta {
    /// Patch identifier (`patch_000000_000081`).
    pub patch_id: String,
    /// Start frame in master timeline.
    pub start_frame: usize,
    /// End frame (exclusive) in master timeline.
    pub end_frame: usize,
    /// Number of frames in patch.
    pub frames: usize,
    /// Whether synchronized audio is saved with this patch.
    pub has_audio: bool,
    /// Recommended crossfade overlap in frames.
    pub blend_overlap_frames: usize,
    /// Notes or prompt.
    pub notes: String,
    /// Timestamp.
    pub timestamp: String,
}

/// Session metadata stored in `session_meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    /// Session ID.
    pub session_id: String,
    /// Total frames in master video.
    pub total_frames: usize,
    /// Master video height.
    pub height: usize,
    /// Master video width.
    pub width: usize,
    /// Master video FPS.
    pub fps: f32,
    /// Audio sample rate.
    pub sample_rate: u32,
    /// Whether a master source video is registered.
    pub has_master_source: bool,
    /// Optional path to an external master video file.
    pub master_video_file_path: Option<String>,
    /// Registered patches keyed by `patch_id`.
    pub patches: HashMap<String, PatchMeta>,
    /// Last updated timestamp.
    pub updated_at: String,
}

impl Default for SessionMeta {
    fn default() -> Self {
        Self {
            session_id: "Default_Timeline_Session".to_string(),
            total_frames: 0,
            height: 0,
            width: 0,
            fps: 24.0,
            sample_rate: 44100,
            has_master_source: false,
            master_video_file_path: None,
            patches: HashMap::new(),
            updated_at: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        }
    }
}

/// Non-destructive timeline session manager storing uint8 RGB patches in SafeTensors on disk.
///
/// Mirrors `TimelineSession` in `engine/timeline_session_manager.py`.
#[derive(Debug, Clone)]
pub struct TimelineSession {
    /// Base timeline sessions directory.
    pub base_dir: PathBuf,
    /// Session ID.
    pub session_id: String,
    /// Resolved session directory.
    pub session_dir: PathBuf,
    /// Path to `session_meta.json`.
    pub meta_path: PathBuf,
    /// Session metadata.
    pub meta: SessionMeta,
}

impl TimelineSession {
    /// Load or create a timeline session under `base_dir`.
    pub fn load_or_create(base_dir: &Path, session_id: &str) -> Result<Self> {
        let session_dir = get_session_dir(base_dir, session_id)?;
        let meta_path = session_dir.join("session_meta.json");
        let meta = if meta_path.exists() {
            fs::read_to_string(&meta_path)
                .ok()
                .and_then(|s| serde_json::from_str::<SessionMeta>(&s).ok())
                .unwrap_or_else(|| SessionMeta {
                    session_id: session_id.to_string(),
                    ..Default::default()
                })
        } else {
            SessionMeta {
                session_id: session_id.to_string(),
                ..Default::default()
            }
        };

        Ok(Self {
            base_dir: base_dir.to_path_buf(),
            session_id: session_id.to_string(),
            session_dir,
            meta_path,
            meta,
        })
    }

    /// Persist `session_meta.json` atomically.
    pub fn save_meta(&mut self) -> Result<()> {
        self.meta.updated_at = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        atomic_write_json(&self.meta_path, &self.meta)
    }

    /// Save a `[F, H, W, C]` float32 tensor (`[0, 1]`) as compact `Dtype::U8` SafeTensors.
    fn save_u8_images_safetensors(path: &Path, key: &str, images: &Tensor) -> Result<()> {
        let u8_bytes: Vec<u8> = images
            .data
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        let view = TensorView::new(Dtype::U8, images.shape.clone(), &u8_bytes)
            .map_err(|e| PrefixStreamError::SafeTensors(e.to_string()))?;
        let mut map = HashMap::new();
        map.insert(key.to_string(), view);
        let serialized = safetensors::serialize(&map, None)
            .map_err(|e| PrefixStreamError::SafeTensors(e.to_string()))?;
        fs::write(path, serialized)?;
        Ok(())
    }

    /// Load a `Dtype::U8` or `Dtype::F32` image tensor from SafeTensors as float32 in `[0, 1]`.
    fn load_u8_images_safetensors(path: &Path, key: &str) -> Result<Tensor> {
        let raw = fs::read(path)?;
        let st = SafeTensors::deserialize(&raw)
            .map_err(|e| PrefixStreamError::SafeTensors(e.to_string()))?;
        let view = st
            .tensor(key)
            .map_err(|e| PrefixStreamError::SafeTensors(e.to_string()))?;
        let shape = view.shape().to_vec();
        let data: Vec<f32> = match view.dtype() {
            Dtype::U8 => view.data().iter().map(|&b| (b as f32) / 255.0).collect(),
            Dtype::F32 => view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect(),
            other => {
                return Err(PrefixStreamError::SafeTensors(format!(
                    "Unsupported dtype: {other:?}"
                )))
            }
        };
        Tensor::from_vec(shape, data)
    }

    /// Register a master source video tensor and optional audio into the session.
    pub fn register_master_source(
        &mut self,
        master_images: Option<&Tensor>,
        master_audio: Option<&AudioData>,
        fps: f32,
        overwrite_master: bool,
        master_video_path: Option<&str>,
    ) -> Result<()> {
        let master_img_path = self.session_dir.join("master_video.safetensors");
        let master_aud_path = self.session_dir.join("master_audio.safetensors");

        if let Some(vpath) = master_video_path {
            if Path::new(vpath).is_file() {
                self.meta.master_video_file_path = Some(vpath.to_string());
                self.meta.has_master_source = true;
            }
        }

        if let Some(imgs) = master_images {
            if imgs.ndim() == 4 {
                self.meta.total_frames = imgs.shape[0];
                self.meta.height = imgs.shape[1];
                self.meta.width = imgs.shape[2];
                self.meta.fps = fps;
                if overwrite_master || !master_img_path.exists() {
                    Self::save_u8_images_safetensors(&master_img_path, "images", imgs)?;
                    self.meta.has_master_source = true;
                }
            }
        }

        if let Some(aud) = master_audio {
            let std_aud = standardize_audio_dict(Some(aud), 44100);
            self.meta.sample_rate = std_aud.sample_rate;
            if overwrite_master || !master_aud_path.exists() {
                let mut tensors = HashMap::new();
                tensors.insert("waveform".to_string(), std_aud.waveform);
                tensors.insert(
                    "sample_rate".to_string(),
                    Tensor::from_vec(vec![1], vec![std_aud.sample_rate as f32])?,
                );
                save_multi_safetensors(&master_aud_path, &tensors, None)?;
            }
        }

        self.save_meta()
    }

    /// Save a processed chunk patch (`[start_frame, end_frame)`) to disk.
    #[allow(clippy::too_many_arguments)]
    pub fn save_chunk_patch(
        &mut self,
        chunk_images: &Tensor,
        chunk_audio: Option<&AudioData>,
        start_frame: usize,
        end_frame: Option<usize>,
        fps: f32,
        blend_overlap_frames: usize,
        notes: &str,
    ) -> Result<String> {
        if chunk_images.ndim() != 4 || chunk_images.shape[0] == 0 {
            return Err(PrefixStreamError::InvalidInput(
                "chunk_images must be a non-empty [F, H, W, C] tensor".to_string(),
            ));
        }

        let f_len = chunk_images.shape[0];
        let actual_end = match end_frame {
            Some(e) if e > start_frame => (start_frame + f_len).min(e),
            _ => start_frame + f_len,
        };

        let patch_id = format!("patch_{start_frame:06}_{actual_end:06}");
        let patch_img_path = self.session_dir.join(format!("{patch_id}_img.safetensors"));
        let patch_aud_path = self.session_dir.join(format!("{patch_id}_aud.safetensors"));

        Self::save_u8_images_safetensors(&patch_img_path, "images", chunk_images)?;

        let mut has_aud = false;
        if let Some(aud) = chunk_audio {
            let std_aud = standardize_audio_dict(Some(aud), self.meta.sample_rate);
            let mut map = HashMap::new();
            map.insert("waveform".to_string(), std_aud.waveform);
            map.insert(
                "sample_rate".to_string(),
                Tensor::from_vec(vec![1], vec![std_aud.sample_rate as f32])?,
            );
            save_multi_safetensors(&patch_aud_path, &map, None)?;
            has_aud = true;
            self.meta.sample_rate = std_aud.sample_rate;
        }

        if self.meta.height == 0 {
            self.meta.height = chunk_images.shape[1];
            self.meta.width = chunk_images.shape[2];
            self.meta.fps = fps;
        }
        if actual_end > self.meta.total_frames {
            self.meta.total_frames = actual_end;
        }

        self.meta.patches.insert(
            patch_id.clone(),
            PatchMeta {
                patch_id: patch_id.clone(),
                start_frame,
                end_frame: actual_end,
                frames: f_len,
                has_audio: has_aud,
                blend_overlap_frames,
                notes: notes.to_string(),
                timestamp: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            },
        );
        self.save_meta()?;
        Ok(patch_id)
    }

    /// Load a single patch's `[F, H, W, C]` float32 images and optional [`AudioData`].
    pub fn load_chunk_patch(&self, patch_id: &str) -> Result<(Option<Tensor>, Option<AudioData>)> {
        let patch_img_path = self.session_dir.join(format!("{patch_id}_img.safetensors"));
        let patch_aud_path = self.session_dir.join(format!("{patch_id}_aud.safetensors"));

        let imgs = if patch_img_path.exists() {
            Some(Self::load_u8_images_safetensors(&patch_img_path, "images")?)
        } else {
            None
        };

        let aud = if patch_aud_path.exists() {
            let (mut map, _) = load_multi_safetensors(&patch_aud_path)?;
            if let Some(wf) = map.remove("waveform") {
                let sr = map
                    .get("sample_rate")
                    .and_then(|t| t.data.first().copied())
                    .unwrap_or(self.meta.sample_rate as f32) as u32;
                Some(AudioData::new(wf, sr))
            } else {
                None
            }
        } else {
            None
        };

        Ok((imgs, aud))
    }

    /// Compute unpatched frame ranges `[(gap_start, gap_end), ...]` across `[0, total_frames)`.
    pub fn get_missing_gap_ranges(&self) -> Vec<(usize, usize)> {
        let total_f = self.meta.total_frames;
        if total_f == 0 {
            return Vec::new();
        }
        let mut covered = vec![false; total_f];
        for p in self.meta.patches.values() {
            let s = p.start_frame.min(total_f);
            let e = p.end_frame.min(total_f);
            for flag in &mut covered[s..e] {
                *flag = true;
            }
        }

        let mut gaps = Vec::new();
        let mut in_gap = false;
        let mut gap_start = 0usize;
        for (i, &cov) in covered.iter().enumerate() {
            if !cov && !in_gap {
                in_gap = true;
                gap_start = i;
            } else if cov && in_gap {
                in_gap = false;
                gaps.push((gap_start, i));
            }
        }
        if in_gap {
            gaps.push((gap_start, total_f));
        }
        gaps
    }

    /// Return `true` when `total_frames > 0` and either a master source exists or all frames are covered by patches.
    pub fn is_complete(&self) -> bool {
        if self.meta.total_frames == 0 {
            return false;
        }
        if self.meta.has_master_source {
            return true;
        }
        self.get_missing_gap_ranges().is_empty()
    }

    /// Reassemble the full timeline by overlaying all saved patches onto the master source (or blank canvas)
    /// with optional linear crossfade and luminance matching at boundaries.
    ///
    /// Mirrors `reassemble_full_timeline` in `engine/timeline_session_manager.py`.
    pub fn reassemble_full_timeline(
        &self,
        fallback_master_images: Option<&Tensor>,
        fallback_master_audio: Option<&AudioData>,
        override_blend_frames: Option<usize>,
        luminance_match: bool,
    ) -> Result<(Tensor, AudioData, String)> {
        let master_img_path = self.session_dir.join("master_video.safetensors");
        let master_aud_path = self.session_dir.join("master_audio.safetensors");

        let mut canvas_imgs = if let Some(fb) = fallback_master_images {
            fb.clone()
        } else if master_img_path.exists() {
            Self::load_u8_images_safetensors(&master_img_path, "images")?
        } else {
            let tf = self.meta.total_frames.max(1);
            let h = if self.meta.height > 0 {
                self.meta.height
            } else {
                480
            };
            let w = if self.meta.width > 0 {
                self.meta.width
            } else {
                832
            };
            Tensor::zeros(vec![tf, h, w, 3])
        };

        let total_f = canvas_imgs.shape[0];
        let h = canvas_imgs.shape[1];
        let w = canvas_imgs.shape[2];
        let c = canvas_imgs.shape[3];
        let fps = self.meta.fps.max(1.0);
        let sr = self.meta.sample_rate.max(8000);
        let target_total_samples =
            (((total_f as f64) / (fps as f64)) * (sr as f64)).round() as usize;
        let target_total_samples = target_total_samples.max(1);

        let mut canvas_wf = if let Some(fb_aud) = fallback_master_audio {
            standardize_audio_dict(Some(fb_aud), sr).waveform
        } else if master_aud_path.exists() {
            let (mut map, _) = load_multi_safetensors(&master_aud_path)?;
            map.remove("waveform")
                .unwrap_or_else(|| Tensor::zeros(vec![1, 2, target_total_samples]))
        } else {
            Tensor::zeros(vec![1, 2, target_total_samples])
        };

        if canvas_wf.shape[2] < target_total_samples {
            let pad = Tensor::zeros(vec![
                canvas_wf.shape[0],
                canvas_wf.shape[1],
                target_total_samples - canvas_wf.shape[2],
            ]);
            canvas_wf = Tensor::concat(&[&canvas_wf, &pad], 2)?;
        } else if canvas_wf.shape[2] > target_total_samples {
            canvas_wf = canvas_wf.slice_axis(2, 0, target_total_samples)?;
        }

        let mut sorted_patches: Vec<&PatchMeta> = self.meta.patches.values().collect();
        sorted_patches.sort_by_key(|p| p.start_frame);

        let mut applied_log = Vec::new();
        let frame_stride = h * w * c;

        for p_info in sorted_patches {
            let (p_imgs_opt, p_aud_opt) = self.load_chunk_patch(&p_info.patch_id)?;
            let Some(mut p_imgs) = p_imgs_opt else {
                continue;
            };

            let s_f = p_info.start_frame;
            if s_f >= total_f {
                continue;
            }
            let p_len = p_imgs.shape[0].min(total_f - s_f);
            let e_f = s_f + p_len;
            p_imgs = p_imgs.slice_dim0(0, p_len)?;

            let blend_f = override_blend_frames.unwrap_or(p_info.blend_overlap_frames);

            if luminance_match && s_f > 0 && blend_f > 0 {
                let ref_len = blend_f.min(s_f).min(p_len);
                if ref_len > 0 {
                    let ref_tail = canvas_imgs.slice_dim0(s_f - ref_len, s_f)?;
                    let cur_head = p_imgs.slice_dim0(0, ref_len)?;
                    let gain = estimate_luminance_gain(&ref_tail, &cur_head, 10.0);
                    p_imgs = apply_luminance_gain_fade(&p_imgs, gain, p_len.min(16))?;
                }
            }

            let left_blend = if s_f > 0 { blend_f.min(p_len / 3) } else { 0 };
            let right_blend = if e_f < total_f && self.meta.has_master_source {
                blend_f.min(p_len / 3)
            } else {
                0
            };

            for f in 0..p_len {
                let dst_f = s_f + f;
                let alpha = if f < left_blend && left_blend > 0 {
                    if left_blend == 1 {
                        0.5
                    } else {
                        (f as f32) / ((left_blend - 1) as f32)
                    }
                } else if f >= p_len - right_blend && right_blend > 0 {
                    let rem = p_len - 1 - f;
                    if right_blend == 1 {
                        0.5
                    } else {
                        (rem as f32) / ((right_blend - 1) as f32)
                    }
                } else {
                    1.0
                };

                let dst_base = dst_f * frame_stride;
                let src_base = f * frame_stride;
                for idx in 0..frame_stride {
                    let orig = canvas_imgs.data[dst_base + idx];
                    let patch_v = p_imgs.data[src_base + idx];
                    canvas_imgs.data[dst_base + idx] = orig * (1.0 - alpha) + patch_v * alpha;
                }
            }

            if let Some(p_aud) = p_aud_opt {
                let mut pwf = standardize_audio_dict(Some(&p_aud), sr).waveform;
                if pwf.shape[1] != canvas_wf.shape[1] {
                    if pwf.shape[1] == 1 && canvas_wf.shape[1] > 1 {
                        let mut reps = Vec::with_capacity(canvas_wf.shape[1]);
                        for _ in 0..canvas_wf.shape[1] {
                            reps.push(&pwf);
                        }
                        pwf = Tensor::concat(&reps, 1)?;
                    } else {
                        pwf = pwf.slice_axis(1, 0, canvas_wf.shape[1].min(pwf.shape[1]))?;
                    }
                }

                let s_smp = (((s_f as f64) / (fps as f64)) * (sr as f64)).round() as usize;
                let e_smp = (((e_f as f64) / (fps as f64)) * (sr as f64)).round() as usize;
                let s_smp = s_smp.min(target_total_samples);
                let e_smp = e_smp.min(target_total_samples);
                let seg_len = e_smp.saturating_sub(s_smp);

                if seg_len > 0 {
                    if pwf.shape[2] < seg_len {
                        let pad =
                            Tensor::zeros(vec![pwf.shape[0], pwf.shape[1], seg_len - pwf.shape[2]]);
                        pwf = Tensor::concat(&[&pwf, &pad], 2)?;
                    } else if pwf.shape[2] > seg_len {
                        pwf = pwf.slice_axis(2, 0, seg_len)?;
                    }

                    let blend_smp =
                        (((left_blend as f64) / (fps as f64)) * (sr as f64)).round() as usize;
                    let blend_smp = blend_smp.min(seg_len / 3);

                    if blend_smp > 0 && s_smp > 0 {
                        let base_head = canvas_wf.slice_axis(2, s_smp, s_smp + blend_smp)?;
                        let patch_head = pwf.slice_axis(2, 0, blend_smp)?;
                        let faded = audio_equal_power_crossfade(&base_head, &patch_head)?;
                        let rest = pwf.slice_axis(2, blend_smp, seg_len)?;
                        pwf = Tensor::concat(&[&faded, &rest], 2)?;
                    }

                    // Splice pwf into canvas_wf along axis 2
                    let mut parts: Vec<Tensor> = Vec::new();
                    if s_smp > 0 {
                        parts.push(canvas_wf.slice_axis(2, 0, s_smp)?);
                    }
                    parts.push(pwf);
                    if e_smp < canvas_wf.shape[2] {
                        parts.push(canvas_wf.slice_axis(2, e_smp, canvas_wf.shape[2])?);
                    }
                    let refs: Vec<&Tensor> = parts.iter().collect();
                    canvas_wf = Tensor::concat(&refs, 2)?;
                }
            }

            applied_log.push(format!("{} [{}..{}]", p_info.patch_id, s_f, e_f));
        }

        let gaps = self.get_missing_gap_ranges();
        let report = format!(
            "Session '{}' | Total: {}f @ {:.1}fps | Patches Applied ({}): {} | Remaining Gaps: {:?}",
            self.session_id,
            total_f,
            fps,
            applied_log.len(),
            if applied_log.is_empty() {
                "None".to_string()
            } else {
                applied_log.join(", ")
            },
            gaps
        );

        Ok((canvas_imgs, AudioData::new(canvas_wf, sr), report))
    }
}
