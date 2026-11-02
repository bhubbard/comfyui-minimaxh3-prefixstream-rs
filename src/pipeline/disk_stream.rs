//! Append decoded clips to disk and export without retaining the full `IMAGE` timeline in RAM.
//!
//! Rust port of `pipeline/disk_stream.py`.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use regex::Regex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::engine::clip_bin_manager::{atomic_write_json, encode_images_to_mp4, get_project_dir};
use crate::error::{PrefixStreamError, Result};
use crate::pipeline::seam_protector::trim_images_and_audio;
use crate::tensor::{AudioData, Tensor};

/// Stream geometry stored in `manifest.json` to enforce consistency across appended segments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiskStreamGeometry {
    /// Video frame rate.
    pub fps: f32,
    /// Frame height (must be even).
    pub height: usize,
    /// Frame width (must be even).
    pub width: usize,
    /// Optional audio sample rate.
    pub sample_rate: Option<u32>,
    /// Optional audio channel count.
    pub channels: Option<usize>,
}

/// Entry for a single segment file in `manifest.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiskStreamClipEntry {
    /// Segment filename (`segment_<32-hex>.mkv`).
    pub file: String,
    /// Number of video frames in this segment.
    pub frames: usize,
}

/// Persistent manifest state for a disk stream (`manifest.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiskStreamManifest {
    /// Schema version (`1`).
    pub version: u32,
    /// Appended clip entries.
    pub clips: Vec<DiskStreamClipEntry>,
    /// Total frames across all clips.
    pub total_frames: usize,
    /// Stream geometry (set on first clip).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometry: Option<DiskStreamGeometry>,
}

impl Default for DiskStreamManifest {
    fn default() -> Self {
        Self {
            version: 1,
            clips: Vec::new(),
            total_frames: 0,
            geometry: None,
        }
    }
}

/// Append a decoded clip to a disk stream and optionally export the concatenated MP4.
///
/// Mirrors `append_disk_clip` in `pipeline/disk_stream.py`.
///
/// `encode_fn` allows injecting a custom segment encoder (or using [`encode_images_to_mp4`]).
#[allow(clippy::too_many_arguments)]
pub fn append_disk_clip_with_encoder<F>(
    base_dir: &Path,
    project_name: &str,
    stream_name: &str,
    images: Option<&Tensor>,
    audio: Option<&AudioData>,
    trim_frames: usize,
    fps: f32,
    export: bool,
    mut encode_fn: F,
) -> Result<(String, String, usize)>
where
    F: FnMut(&Tensor, f32, Option<&AudioData>, &Path) -> bool,
{
    let name_re = Regex::new(r"^[\w-]+$").unwrap();
    if !name_re.is_match(stream_name) {
        return Err(PrefixStreamError::InvalidInput(
            "Stream name must contain only letters, digits, underscores or hyphens".to_string(),
        ));
    }

    let proj_dir = get_project_dir(base_dir, project_name)?;
    let streams_root = proj_dir.join(".streams");
    if let Ok(meta) = fs::symlink_metadata(&streams_root) {
        if meta.file_type().is_symlink() {
            return Err(PrefixStreamError::SecurityViolation(
                "Linked stream directories are not supported".to_string(),
            ));
        }
    }
    fs::create_dir_all(&streams_root)?;

    let directory = streams_root.join(stream_name);
    if let Ok(meta) = fs::symlink_metadata(&directory) {
        if meta.file_type().is_symlink() {
            return Err(PrefixStreamError::SecurityViolation(
                "Linked stream directories are not supported".to_string(),
            ));
        }
    }
    fs::create_dir_all(&directory)?;

    let proj_real = fs::canonicalize(&proj_dir)?;
    let dir_real = fs::canonicalize(&directory)?;
    if !dir_real.starts_with(&proj_real) {
        return Err(PrefixStreamError::SecurityViolation(
            "Linked stream directories are not supported".to_string(),
        ));
    }

    let manifest_path = directory.join("manifest.json");
    let mut state: DiskStreamManifest = if manifest_path.is_file() {
        let raw = fs::read_to_string(&manifest_path)?;
        serde_json::from_str(&raw)?
    } else {
        DiskStreamManifest::default()
    };

    if let Some(raw_imgs) = images {
        let (trimmed_imgs, trimmed_aud) =
            trim_images_and_audio(raw_imgs, audio, trim_frames, fps, true)?;
        if trimmed_imgs.shape[0] == 0 {
            return Err(PrefixStreamError::InvalidInput(
                "No frames remain after trimming".to_string(),
            ));
        }
        let h = trimmed_imgs.shape[1];
        let w = trimmed_imgs.shape[2];
        let c = trimmed_imgs.shape[3];
        if !matches!(c, 3 | 4) || h % 2 != 0 || w % 2 != 0 {
            return Err(PrefixStreamError::InvalidInput(
                "Disk streams require RGB/RGBA images with even height and width".to_string(),
            ));
        }
        if let Some(ref aud) = trimmed_aud {
            if aud.waveform.ndim() != 3 || aud.waveform.shape[0] != 1 {
                return Err(PrefixStreamError::InvalidInput(
                    "Disk streams require audio shape [1, channels, samples]".to_string(),
                ));
            }
        }

        let geometry = DiskStreamGeometry {
            fps,
            height: h,
            width: w,
            sample_rate: trimmed_aud.as_ref().map(|a| a.sample_rate),
            channels: trimmed_aud.as_ref().map(|a| a.waveform.shape[1]),
        };

        if !state.clips.is_empty() && state.geometry.as_ref() != Some(&geometry) {
            return Err(PrefixStreamError::InvalidInput(
                "Stream resolution, fps, sample rate and channels must remain consistent"
                    .to_string(),
            ));
        }

        let filename = format!("segment_{}.mkv", Uuid::new_v4().simple());
        let seg_path = directory.join(&filename);
        if !encode_fn(&trimmed_imgs, fps, trimmed_aud.as_ref(), &seg_path) {
            return Err(PrefixStreamError::Internal(
                "Segment encoding failed; stream manifest was not changed".to_string(),
            ));
        }

        state.geometry = Some(geometry);
        state.clips.push(DiskStreamClipEntry {
            file: filename,
            frames: trimmed_imgs.shape[0],
        });
        state.total_frames += trimmed_imgs.shape[0];
        atomic_write_json(&manifest_path, &state)?;
    }

    let mut output_path = String::new();
    if export {
        if state.clips.is_empty() {
            return Err(PrefixStreamError::InvalidInput(
                "Cannot export an empty stream".to_string(),
            ));
        }
        let seg_re = Regex::new(r"^segment_[0-9a-f]{32}\.mkv$").unwrap();
        let concat_path = directory.join("concat.txt");
        let geom_fps = state
            .geometry
            .as_ref()
            .map(|g| g.fps)
            .unwrap_or(fps)
            .max(1.0);

        let mut concat_content = String::new();
        for clip in &state.clips {
            if !seg_re.is_match(&clip.file) {
                return Err(PrefixStreamError::SecurityViolation(
                    "Invalid segment filename in stream manifest".to_string(),
                ));
            }
            concat_content.push_str(&format!("file '{}'\n", clip.file));
            concat_content.push_str(&format!(
                "duration {:.12}\n",
                (clip.frames as f64) / (geom_fps as f64)
            ));
        }
        let mut f = File::create(&concat_path)?;
        f.write_all(concat_content.as_bytes())?;
        drop(f);

        let temp_output = directory.join(format!("export_{}.mp4", Uuid::new_v4().simple()));
        let final_output = directory.join("video.mp4");

        let status = Command::new("ffmpeg")
            .args(["-v", "error", "-f", "concat", "-safe", "1", "-i"])
            .arg(&concat_path)
            .args([
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
            ])
            .arg(&temp_output)
            .status();

        match status {
            Ok(s) if s.success() && temp_output.exists() => {
                fs::rename(&temp_output, &final_output)?;
                output_path = final_output.to_string_lossy().into_owned();
            }
            _ => {
                if temp_output.exists() {
                    let _ = fs::remove_file(&temp_output);
                }
                return Err(PrefixStreamError::Internal(
                    "ffmpeg export failed".to_string(),
                ));
            }
        }
    }

    Ok((
        manifest_path.to_string_lossy().into_owned(),
        output_path,
        state.total_frames,
    ))
}

/// Append a decoded clip to a disk stream using the default [`encode_images_to_mp4`] encoder.
#[allow(clippy::too_many_arguments)]
pub fn append_disk_clip(
    base_dir: &Path,
    project_name: &str,
    stream_name: &str,
    images: Option<&Tensor>,
    audio: Option<&AudioData>,
    trim_frames: usize,
    fps: f32,
    export: bool,
) -> Result<(String, String, usize)> {
    append_disk_clip_with_encoder(
        base_dir,
        project_name,
        stream_name,
        images,
        audio,
        trim_frames,
        fps,
        export,
        encode_images_to_mp4,
    )
}
