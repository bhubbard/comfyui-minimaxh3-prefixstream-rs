//! Clip Bin Manager: persistent DAG storage, SafeTensors serialization, and metadata indexing.
//!
//! Rust port of `engine/clip_bin_manager.py`.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::Local;
use image::{Rgb, RgbImage};
use regex::Regex;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::error::{PrefixStreamError, Result};
use crate::tensor::{AudioData, Tensor};

/// Clip metadata entry stored in `meta.json` and `index.json`.
///
/// Mirrors `ClipMeta` in `engine/clip_bin_manager.py`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClipMeta {
    /// Clip identifier (e.g. `clip_001_01`).
    pub clip_id: String,
    /// Project name.
    pub project_name: String,
    /// Optional parent clip ID in the continuation DAG.
    pub parent_id: Option<String>,
    /// Scene number (`>= 1`).
    pub scene_num: usize,
    /// Take number (`>= 1`).
    pub take_num: usize,
    /// ISO-like timestamp `YYYY-MM-DD HH:MM:SS`.
    pub timestamp: String,
    /// Total pixel frames.
    pub frames: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Frame rate.
    pub fps: f32,
    /// Prompt or notes.
    pub prompt: String,
    /// Generation seed.
    pub seed: i64,
    /// User rating (`-1` rejected, `0` unrated, `1..=5` stars).
    pub rating: i32,
    /// Whether `video_latent.safetensors` is present.
    pub has_video_latent: bool,
    /// Whether `audio_latent.safetensors` is present.
    pub has_audio_latent: bool,
    /// Whether `audio_waveform.safetensors` is present.
    pub has_audio_waveform: bool,
    /// Number of tail images saved in `tail_images.safetensors`.
    pub tail_frames_saved: usize,
    /// Whether `source_video.mp4` is saved.
    pub has_source_video: bool,
}

impl Default for ClipMeta {
    fn default() -> Self {
        Self {
            clip_id: String::new(),
            project_name: "Default_Project".to_string(),
            parent_id: None,
            scene_num: 1,
            take_num: 1,
            timestamp: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            frames: 0,
            width: 0,
            height: 0,
            fps: 24.0,
            prompt: String::new(),
            seed: 0,
            rating: 0,
            has_video_latent: false,
            has_audio_latent: false,
            has_audio_waveform: false,
            tail_frames_saved: 0,
            has_source_video: false,
        }
    }
}

/// Write JSON data atomically via a temporary sibling file, `sync_all`, and `fs::rename`.
///
/// Cleans up the temporary file if serialization or flushing fails so existing files remain intact.
pub fn atomic_write_json<T: Serialize + ?Sized>(file_path: &Path, data: &T) -> Result<()> {
    let dir_name = file_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    fs::create_dir_all(&dir_name)?;

    let tmp_name = format!(".tmp_{}.json", Uuid::new_v4().simple());
    let temp_path = dir_name.join(tmp_name);

    let write_res = (|| -> Result<()> {
        let serialized = serde_json::to_string_pretty(data)?;
        let mut f = File::create(&temp_path)?;
        f.write_all(serialized.as_bytes())?;
        f.flush()?;
        f.sync_all()?;
        drop(f);
        fs::rename(&temp_path, file_path)?;
        Ok(())
    })();

    if write_res.is_err() && temp_path.exists() {
        let _ = fs::remove_file(&temp_path);
    }

    write_res
}

/// Save a single `f32` or `u8` [`Tensor`] to a `.safetensors` file under `key`.
pub fn save_tensor_safetensors(path: &Path, key: &str, tensor: &Tensor) -> Result<()> {
    let mut byte_buf = Vec::with_capacity(tensor.data.len() * 4);
    for &v in &tensor.data {
        byte_buf.extend_from_slice(&v.to_le_bytes());
    }
    let view = TensorView::new(Dtype::F32, tensor.shape.clone(), &byte_buf)
        .map_err(|e| PrefixStreamError::SafeTensors(e.to_string()))?;
    let mut map = HashMap::new();
    map.insert(key.to_string(), view);
    let bytes = safetensors::serialize(&map, None)
        .map_err(|e| PrefixStreamError::SafeTensors(e.to_string()))?;
    fs::write(path, bytes)?;
    Ok(())
}

/// Save multiple named `f32` [`Tensor`]s and optional string metadata to a `.safetensors` file.
pub fn save_multi_safetensors(
    path: &Path,
    tensors: &HashMap<String, Tensor>,
    metadata: Option<HashMap<String, String>>,
) -> Result<()> {
    let mut raw_buffers: HashMap<String, Vec<u8>> = HashMap::new();
    for (k, t) in tensors {
        let mut buf = Vec::with_capacity(t.data.len() * 4);
        for &v in &t.data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        raw_buffers.insert(k.clone(), buf);
    }

    let mut views = HashMap::new();
    for (k, t) in tensors {
        let buf = raw_buffers.get(k).unwrap();
        let view = TensorView::new(Dtype::F32, t.shape.clone(), buf)
            .map_err(|e| PrefixStreamError::SafeTensors(e.to_string()))?;
        views.insert(k.clone(), view);
    }

    let bytes = safetensors::serialize(&views, metadata)
        .map_err(|e| PrefixStreamError::SafeTensors(e.to_string()))?;
    fs::write(path, bytes)?;
    Ok(())
}

/// Load all tensors and optional string metadata from a `.safetensors` file.
pub fn load_multi_safetensors(
    path: &Path,
) -> Result<(HashMap<String, Tensor>, HashMap<String, String>)> {
    let raw = fs::read(path)?;
    let (_, header) = SafeTensors::read_metadata(&raw)
        .map_err(|e| PrefixStreamError::SafeTensors(e.to_string()))?;
    let metadata = header.metadata().clone().unwrap_or_default();

    let st = SafeTensors::deserialize(&raw)
        .map_err(|e| PrefixStreamError::SafeTensors(e.to_string()))?;

    let mut tensors = HashMap::new();
    for (name, view) in st.tensors() {
        let shape = view.shape().to_vec();
        let bytes = view.data();
        let data: Vec<f32> = match view.dtype() {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect(),
            Dtype::U8 => bytes.iter().map(|&b| b as f32).collect(),
            other => {
                return Err(PrefixStreamError::SafeTensors(format!(
                    "Unsupported dtype in safetensors: {other:?}"
                )))
            }
        };
        tensors.insert(name, Tensor::from_vec(shape, data)?);
    }

    Ok((tensors, metadata))
}

/// Load a single tensor by key from a `.safetensors` file.
pub fn load_tensor_safetensors(path: &Path, key: &str) -> Result<Tensor> {
    let (mut map, _) = load_multi_safetensors(path)?;
    map.remove(key).ok_or_else(|| {
        PrefixStreamError::SafeTensors(format!("Key '{key}' not found in {}", path.display()))
    })
}

/// Resolve the base directory for MiniMax Clip Bin storage.
///
/// Honors `MINIMAX_CLIP_BIN_DIR` environment variable if set; otherwise defaults to `./output/MiniMax_ClipBin`.
pub fn get_base_bin_dir() -> PathBuf {
    let base = if let Ok(env_dir) = std::env::var("MINIMAX_CLIP_BIN_DIR") {
        PathBuf::from(env_dir)
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("output")
            .join("MiniMax_ClipBin")
    };
    let _ = fs::create_dir_all(&base);
    base
}

/// Sanitize a project name and return its directory inside [`get_base_bin_dir`],
/// preventing path traversal and symlink escapes.
pub fn get_project_dir(base_dir: &Path, project_name: &str) -> Result<PathBuf> {
    let re = Regex::new(r"[^a-zA-Z0-9_\-]").unwrap();
    let safe_trimmed = project_name.trim();
    let safe_name = re.replace_all(safe_trimmed, "_").into_owned();
    let safe_name = if safe_name.is_empty() || safe_name == "." || safe_name == ".." {
        "Default_Project".to_string()
    } else {
        safe_name
    };

    fs::create_dir_all(base_dir)?;
    let base_real = fs::canonicalize(base_dir)?;
    let proj_dir = base_dir.join(&safe_name);

    // Check symlink before creating
    if let Ok(meta) = fs::symlink_metadata(&proj_dir) {
        if meta.file_type().is_symlink() {
            return Err(PrefixStreamError::SecurityViolation(format!(
                "Security Violation: Project directory is a symlink ({safe_name})"
            )));
        }
    }

    fs::create_dir_all(&proj_dir)?;
    let proj_real = fs::canonicalize(&proj_dir)?;
    if !proj_real.starts_with(&base_real) || proj_real == base_real {
        return Err(PrefixStreamError::SecurityViolation(format!(
            "Security Violation: Project directory escaped base bin ({safe_name})"
        )));
    }

    Ok(proj_dir)
}

/// Resolve and validate a clip directory inside a project directory, rejecting path traversal and symlinks.
///
/// Mirrors `get_clip_dir` in `engine/clip_bin_manager.py`.
pub fn get_clip_dir(base_dir: &Path, project_name: &str, clip_id: &str) -> Result<PathBuf> {
    if clip_id.is_empty()
        || clip_id.contains("..")
        || clip_id.contains('/')
        || clip_id.contains('\\')
    {
        return Err(PrefixStreamError::SecurityViolation(format!(
            "Invalid clip_id requested: {clip_id}"
        )));
    }

    let proj_dir = get_project_dir(base_dir, project_name)?;
    let proj_real = fs::canonicalize(&proj_dir)?;
    let target_dir = proj_dir.join(clip_id);

    if let Ok(meta) = fs::symlink_metadata(&target_dir) {
        if meta.file_type().is_symlink() {
            return Err(PrefixStreamError::SecurityViolation(format!(
                "Security Violation: Clip directory is a symlink ({clip_id})"
            )));
        }
    }

    // Check parent or existing target canonical path
    if target_dir.exists() {
        let target_real = fs::canonicalize(&target_dir)?;
        if !target_real.starts_with(&proj_real) || target_real == proj_real {
            return Err(PrefixStreamError::SecurityViolation(format!(
                "Security Violation: Path traversal detected for clip_id '{clip_id}'"
            )));
        }
    }

    Ok(target_dir)
}

/// List all valid project directories inside `base_dir`.
pub fn list_projects(base_dir: &Path) -> Vec<String> {
    let mut projects = Vec::new();
    if let Ok(entries) = fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            if let Ok(ft) = entry.file_type() {
                if ft.is_dir() && !ft.is_symlink() {
                    if let Some(name) = entry.file_name().to_str() {
                        if !name.starts_with('.') {
                            projects.push(name.to_string());
                        }
                    }
                }
            }
        }
    }
    projects.sort();
    if projects.is_empty() {
        projects.push("Default_Project".to_string());
    }
    projects
}

/// Load `index.json` for a project, rebuilding it if missing or corrupted.
pub fn load_project_index(
    base_dir: &Path,
    project_name: &str,
) -> Result<HashMap<String, ClipMeta>> {
    let proj_dir = get_project_dir(base_dir, project_name)?;
    let index_path = proj_dir.join("index.json");
    if index_path.exists() {
        if let Ok(raw) = fs::read_to_string(&index_path) {
            if let Ok(parsed) = serde_json::from_str::<HashMap<String, ClipMeta>>(&raw) {
                return Ok(parsed);
            }
        }
        return rebuild_project_index(base_dir, project_name);
    }
    Ok(HashMap::new())
}

/// Save `index.json` for a project atomically.
pub fn save_project_index(
    base_dir: &Path,
    project_name: &str,
    index_data: &HashMap<String, ClipMeta>,
) -> Result<()> {
    let proj_dir = get_project_dir(base_dir, project_name)?;
    let index_path = proj_dir.join("index.json");
    atomic_write_json(&index_path, index_data)
}

/// Scan all `clip_*` subdirectories in a project and rebuild `index.json`.
pub fn rebuild_project_index(
    base_dir: &Path,
    project_name: &str,
) -> Result<HashMap<String, ClipMeta>> {
    let proj_dir = get_project_dir(base_dir, project_name)?;
    let mut index_data = HashMap::new();

    if let Ok(entries) = fs::read_dir(&proj_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().into_owned();
            if path.is_dir() && file_name.starts_with("clip_") {
                let meta_path = path.join("meta.json");
                if meta_path.exists() {
                    if let Ok(raw) = fs::read_to_string(&meta_path) {
                        if let Ok(meta) = serde_json::from_str::<ClipMeta>(&raw) {
                            index_data.insert(file_name, meta);
                        }
                    }
                }
            }
        }
    }

    save_project_index(base_dir, project_name, &index_data)?;
    Ok(index_data)
}

/// Allocate the next `(clip_id, scene_num, take_num)` based on `parent_id` and existing clips.
pub fn allocate_next_clip_id(
    base_dir: &Path,
    project_name: &str,
    parent_id: Option<&str>,
) -> Result<(String, usize, usize)> {
    let index = load_project_index(base_dir, project_name)?;
    let mut target_scene = 1usize;

    if let Some(pid) = parent_id {
        if !pid.is_empty() && pid != "None (Root / Scene 1)" {
            if let Some(p_meta) = index.get(pid) {
                target_scene = p_meta.scene_num + 1;
            } else {
                let re = Regex::new(r"clip_(\d+)_\d+").unwrap();
                if let Some(caps) = re.captures(pid) {
                    if let Ok(sc) = caps[1].parse::<usize>() {
                        target_scene = sc + 1;
                    }
                }
            }
        }
    }

    let max_take = index
        .values()
        .filter(|m| m.scene_num == target_scene)
        .map(|m| m.take_num)
        .max()
        .unwrap_or(0);

    let next_take = max_take + 1;
    let clip_id = format!("clip_{target_scene:03}_{next_take:02}");
    Ok((clip_id, target_scene, next_take))
}

/// Create a placeholder RGB image `[H, W, 3]` with a dark slate background and status accent bar.
pub fn create_placeholder_card(width: u32, height: u32, _text: &str) -> RgbImage {
    let mut img = RgbImage::from_pixel(width, height, Rgb([26, 28, 35]));
    // Draw a subtle cyan header bar to visually distinguish placeholder cards
    let bar_h = (height / 12).max(4);
    for y in 0..bar_h.min(height) {
        for x in 0..width {
            img.put_pixel(x, y, Rgb([56, 189, 248]));
        }
    }
    img
}

/// Create a side-by-side preview PNG (`first_frame | last_frame`) from a `[F, H, W, C]` tensor.
pub fn create_side_by_side_preview(images: Option<&Tensor>, save_path: &Path) -> Result<()> {
    let target_h = 256u32;
    let combined = match images {
        Some(t) if t.ndim() == 4 && t.shape[0] > 0 && t.shape[3] >= 3 => {
            let f_count = t.shape[0];
            let h = t.shape[1];
            let w = t.shape[2];
            let c = t.shape[3];
            let target_w = ((w as f32) * (target_h as f32 / h.max(1) as f32)).round() as u32;
            let target_w = target_w.max(16);

            let frame_to_rgb = |frame_idx: usize| -> RgbImage {
                let mut raw_img = RgbImage::new(w as u32, h as u32);
                let base = frame_idx * h * w * c;
                for y in 0..h {
                    for x in 0..w {
                        let p = base + (y * w + x) * c;
                        let r = (t.data[p].clamp(0.0, 1.0) * 255.0) as u8;
                        let g = (t.data[p + 1].clamp(0.0, 1.0) * 255.0) as u8;
                        let b = (t.data[p + 2].clamp(0.0, 1.0) * 255.0) as u8;
                        raw_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                    }
                }
                image::imageops::resize(
                    &raw_img,
                    target_w,
                    target_h,
                    image::imageops::FilterType::Lanczos3,
                )
            };

            let img_first = frame_to_rgb(0);
            let img_last = if f_count > 1 {
                frame_to_rgb(f_count - 1)
            } else {
                create_placeholder_card(target_w, target_h, "Single Frame")
            };

            let total_w = target_w * 2 + 4;
            let mut canvas = RgbImage::from_pixel(total_w, target_h, Rgb([15, 15, 20]));
            image::imageops::overlay(&mut canvas, &img_first, 0, 0);
            image::imageops::overlay(&mut canvas, &img_last, (target_w + 4) as i64, 0);
            canvas
        }
        _ => {
            let p1 = create_placeholder_card(256, 256, "Latent Only (No RGB)");
            let p2 = create_placeholder_card(256, 256, "Connect images to Saver");
            let mut canvas = RgbImage::from_pixel(516, 256, Rgb([255, 255, 255]));
            image::imageops::overlay(&mut canvas, &p1, 0, 0);
            image::imageops::overlay(&mut canvas, &p2, 260, 0);
            canvas
        }
    };

    combined
        .save(save_path)
        .map_err(|e| PrefixStreamError::Image(e.to_string()))?;
    Ok(())
}

/// Resolve an optional source video path strictly inside allowed directories (`MINIMAX_ALLOWED_VIDEO_DIR`,
/// `output`, or `input`), rejecting symlinks and path traversal.
pub fn resolve_source_video_path(source_video_path: Option<&str>) -> Option<PathBuf> {
    let raw = source_video_path?.trim();
    if raw.is_empty() {
        return None;
    }

    let p = Path::new(raw);
    if let Ok(meta) = fs::symlink_metadata(p) {
        if meta.file_type().is_symlink() {
            return None;
        }
    }

    let candidate = fs::canonicalize(p).ok()?;
    let mut allowed_roots = Vec::new();
    if let Ok(env_root) = std::env::var("MINIMAX_ALLOWED_VIDEO_DIR") {
        if let Ok(canon) = fs::canonicalize(env_root) {
            allowed_roots.push(canon);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        if let Ok(out_dir) = fs::canonicalize(cwd.join("output")) {
            allowed_roots.push(out_dir);
        }
        if let Ok(in_dir) = fs::canonicalize(cwd.join("input")) {
            allowed_roots.push(in_dir);
        }
    }
    if let Ok(bin_root) = fs::canonicalize(get_base_bin_dir()) {
        allowed_roots.push(bin_root);
    }

    let is_allowed = allowed_roots
        .iter()
        .any(|root| candidate.starts_with(root) && &candidate != root);
    if !is_allowed {
        return None;
    }

    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}

/// Encode `[F, H, W, C]` float tensor images to an H.264 MP4 file using `ffmpeg` if available.
pub fn encode_images_to_mp4(
    images: &Tensor,
    fps: f32,
    audio: Option<&AudioData>,
    dest_mp4_path: &Path,
) -> bool {
    if images.ndim() != 4 || images.shape[0] == 0 || images.shape[3] < 3 {
        return false;
    }

    let f_count = images.shape[0];
    let h = images.shape[1];
    let w = images.shape[2];
    let c = images.shape[3];

    let mut raw_rgb = Vec::with_capacity(f_count * h * w * 3);
    for f in 0..f_count {
        let base = f * h * w * c;
        for idx in 0..(h * w) {
            let p = base + idx * c;
            raw_rgb.push((images.data[p].clamp(0.0, 1.0) * 255.0) as u8);
            raw_rgb.push((images.data[p + 1].clamp(0.0, 1.0) * 255.0) as u8);
            raw_rgb.push((images.data[p + 2].clamp(0.0, 1.0) * 255.0) as u8);
        }
    }

    let _ = audio; // Audio muxing optional if wav is not needed
    let child = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            &format!("{w}x{h}"),
            "-pix_fmt",
            "rgb24",
            "-r",
            &format!("{fps}"),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "18",
        ])
        .arg(dest_mp4_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();

    if let Ok(mut proc) = child {
        if let Some(mut stdin) = proc.stdin.take() {
            let _ = stdin.write_all(&raw_rgb);
        }
        if let Ok(status) = proc.wait() {
            return status.success() && dest_mp4_path.exists();
        }
    }
    false
}

/// Parameters for [`save_clip_asset`].
#[derive(Debug, Clone)]
pub struct SaveClipRequest<'a> {
    /// Base clip bin directory.
    pub base_dir: &'a Path,
    /// Project name.
    pub project_name: &'a str,
    /// Optional parent clip ID.
    pub parent_id: Option<&'a str>,
    /// Optional `[F, H, W, C]` RGB float tensor.
    pub images: Option<&'a Tensor>,
    /// Optional video latent tensor `[B, C, T, H, W]`.
    pub video_latent: Option<&'a Tensor>,
    /// Optional audio latent tensor.
    pub audio_latent: Option<&'a Tensor>,
    /// Optional decoded audio waveform + sample rate.
    pub audio: Option<&'a AudioData>,
    /// Prompt or generation notes.
    pub prompt: &'a str,
    /// Generation seed.
    pub seed: i64,
    /// Frame rate.
    pub fps: f32,
    /// Number of tail RGB frames to persist (`0` = save all frames).
    pub save_tail_frames: usize,
    /// Optional path to a source MP4 file.
    pub source_video_path: Option<&'a str>,
    /// Whether to auto-encode a preview MP4 when `source_video_path` is absent.
    pub auto_encode_mp4: bool,
}

/// Save a clip asset and its `.safetensors` payloads into the project Clip Bin.
///
/// Mirrors `save_clip_asset` in `engine/clip_bin_manager.py`.
pub fn save_clip_asset(req: SaveClipRequest<'_>) -> Result<ClipMeta> {
    let normalized_parent = req.parent_id.and_then(|p| {
        if p.is_empty() || p == "None (Root / Scene 1)" {
            None
        } else {
            Some(p.split(' ').next().unwrap_or(p))
        }
    });

    let (clip_id, scene_num, take_num) =
        allocate_next_clip_id(req.base_dir, req.project_name, normalized_parent)?;
    let clip_dir = get_clip_dir(req.base_dir, req.project_name, &clip_id)?;
    fs::create_dir_all(&clip_dir)?;

    let mut frames_count = 0usize;
    let mut height = 0usize;
    let mut width = 0usize;

    if let Some(imgs) = req.images {
        if imgs.ndim() == 4 {
            frames_count = imgs.shape[0];
            height = imgs.shape[1];
            width = imgs.shape[2];
        }
    } else if let Some(vlat) = req.video_latent {
        if vlat.ndim() == 5 {
            frames_count = (vlat.shape[2].saturating_sub(1)) * 4 + 1;
            height = vlat.shape[3] * 8;
            width = vlat.shape[4] * 8;
        }
    }

    let mut has_v_lat = false;
    if let Some(vlat) = req.video_latent {
        save_tensor_safetensors(&clip_dir.join("video_latent.safetensors"), "samples", vlat)?;
        has_v_lat = true;
    }

    let mut has_a_lat = false;
    if let Some(alat) = req.audio_latent {
        save_tensor_safetensors(&clip_dir.join("audio_latent.safetensors"), "samples", alat)?;
        has_a_lat = true;
    }

    let mut has_a_wav = false;
    if let Some(aud) = req.audio {
        let mut tensors = HashMap::new();
        tensors.insert("waveform".to_string(), aud.waveform.clone());
        tensors.insert(
            "sample_rate".to_string(),
            Tensor::from_vec(vec![1], vec![aud.sample_rate as f32])?,
        );
        save_multi_safetensors(&clip_dir.join("audio_waveform.safetensors"), &tensors, None)?;
        has_a_wav = true;
    }

    let mut tail_saved = 0usize;
    if let Some(imgs) = req.images {
        if imgs.ndim() == 4 && imgs.shape[0] > 0 {
            let total_f = imgs.shape[0];
            let tail_tensor = if req.save_tail_frames > 0 && total_f > req.save_tail_frames {
                imgs.slice_dim0(total_f - req.save_tail_frames, total_f)?
            } else {
                imgs.clone()
            };
            tail_saved = tail_tensor.shape[0];
            save_tensor_safetensors(
                &clip_dir.join("tail_images.safetensors"),
                "images",
                &tail_tensor,
            )?;
        }
    }

    let _ = create_side_by_side_preview(req.images, &clip_dir.join("preview.png"));

    let mut has_src_vid = false;
    let dest_vid = clip_dir.join("source_video.mp4");
    if let Some(src_path) = resolve_source_video_path(req.source_video_path) {
        if fs::copy(&src_path, &dest_vid).is_ok() {
            has_src_vid = true;
        }
    }
    if !has_src_vid && req.auto_encode_mp4 {
        if let Some(imgs) = req.images {
            has_src_vid = encode_images_to_mp4(imgs, req.fps, req.audio, &dest_vid);
        }
    }

    let meta = ClipMeta {
        clip_id: clip_id.clone(),
        project_name: req.project_name.to_string(),
        parent_id: normalized_parent.map(|s| s.to_string()),
        scene_num,
        take_num,
        timestamp: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        frames: frames_count,
        width,
        height,
        fps: req.fps,
        prompt: req.prompt.to_string(),
        seed: req.seed,
        rating: 0,
        has_video_latent: has_v_lat,
        has_audio_latent: has_a_lat,
        has_audio_waveform: has_a_wav,
        tail_frames_saved: tail_saved,
        has_source_video: has_src_vid,
    };

    atomic_write_json(&clip_dir.join("meta.json"), &meta)?;

    let mut index = load_project_index(req.base_dir, req.project_name)?;
    index.insert(clip_id, meta.clone());
    save_project_index(req.base_dir, req.project_name, &index)?;

    Ok(meta)
}

/// Loaded clip assets returned by [`load_clip_asset`].
#[derive(Debug, Clone)]
pub struct LoadedClipAsset {
    /// Clip metadata.
    pub meta: ClipMeta,
    /// Tail RGB images (`[F, H, W, C]`), or a 1-frame zero fallback if none saved.
    pub tail_images: Tensor,
    /// Optional video latent tensor (`[B, C, T, H, W]`).
    pub video_latent: Option<Tensor>,
    /// Optional audio latent tensor.
    pub audio_latent: Option<Tensor>,
    /// Optional decoded audio waveform + sample rate.
    pub audio: Option<AudioData>,
}

/// Load a clip asset and its `.safetensors` payloads from disk.
///
/// Mirrors `load_clip_asset` in `engine/clip_bin_manager.py`.
pub fn load_clip_asset(
    base_dir: &Path,
    project_name: &str,
    clip_id_or_label: &str,
) -> Result<LoadedClipAsset> {
    let actual_id = clip_id_or_label
        .split(' ')
        .next()
        .unwrap_or(clip_id_or_label)
        .trim();
    let clip_dir = get_clip_dir(base_dir, project_name, actual_id)?;
    if !clip_dir.exists() {
        return Err(PrefixStreamError::NotFound(format!(
            "Clip asset '{actual_id}' not found in project '{project_name}'"
        )));
    }

    let meta_path = clip_dir.join("meta.json");
    let meta: ClipMeta = if meta_path.exists() {
        let raw = fs::read_to_string(&meta_path)?;
        serde_json::from_str(&raw)?
    } else {
        ClipMeta {
            clip_id: actual_id.to_string(),
            project_name: project_name.to_string(),
            ..Default::default()
        }
    };

    let vlat_path = clip_dir.join("video_latent.safetensors");
    let video_latent = if vlat_path.exists() {
        Some(load_tensor_safetensors(&vlat_path, "samples")?)
    } else {
        None
    };

    let alat_path = clip_dir.join("audio_latent.safetensors");
    let audio_latent = if alat_path.exists() {
        Some(load_tensor_safetensors(&alat_path, "samples")?)
    } else {
        None
    };

    let awav_path = clip_dir.join("audio_waveform.safetensors");
    let audio = if awav_path.exists() {
        let (mut map, _) = load_multi_safetensors(&awav_path)?;
        if let Some(waveform) = map.remove("waveform") {
            let sr = map
                .get("sample_rate")
                .and_then(|t| t.data.first().copied())
                .unwrap_or(44100.0) as u32;
            Some(AudioData::new(waveform, sr))
        } else {
            None
        }
    } else {
        None
    };

    let timg_path = clip_dir.join("tail_images.safetensors");
    let tail_images = if timg_path.exists() {
        load_tensor_safetensors(&timg_path, "images")?
    } else {
        Tensor::zeros(vec![1, 256, 256, 3])
    };

    Ok(LoadedClipAsset {
        meta,
        tail_images,
        video_latent,
        audio_latent,
        audio,
    })
}

/// Format a human-readable clip label for UI dropdowns.
///
/// Mirrors `format_clip_label` in `engine/clip_bin_manager.py`.
pub fn format_clip_label(meta: &ClipMeta) -> String {
    let parent_str = match meta.parent_id.as_deref() {
        Some(p) if !p.is_empty() => format!("<- {p}"),
        _ => "Root".to_string(),
    };
    let stars = if meta.rating > 0 {
        format!(" [{}]", "*".repeat(meta.rating as usize))
    } else if meta.rating < 0 {
        " [Rejected]".to_string()
    } else {
        String::new()
    };
    let prompt_snip = if meta.prompt.len() > 25 {
        format!("{}...", &meta.prompt[..25])
    } else {
        meta.prompt.clone()
    };
    format!(
        "{} (S{:02} T{:02} | {}f | {}{} | \"{}\")",
        meta.clip_id, meta.scene_num, meta.take_num, meta.frames, parent_str, stars, prompt_snip
    )
}

/// Return formatted clip selection labels for a project, sorted newest-first.
pub fn get_clips_for_selection(
    base_dir: &Path,
    project_name: &str,
    include_root_option: bool,
) -> Result<Vec<String>> {
    let index = load_project_index(base_dir, project_name)?;
    let mut clips: Vec<&ClipMeta> = index.values().filter(|c| c.rating >= 0).collect();
    clips.sort_by(|a, b| {
        b.scene_num
            .cmp(&a.scene_num)
            .then_with(|| b.take_num.cmp(&a.take_num))
    });

    let mut options = Vec::new();
    if include_root_option {
        options.push("None (Root / Scene 1)".to_string());
    }
    for c in clips {
        options.push(format_clip_label(c));
    }
    if options.is_empty() {
        options.push("No clips found in project".to_string());
    }
    Ok(options)
}

/// Delete a clip directory from disk and remove it from `index.json`.
pub fn delete_clip_asset(base_dir: &Path, project_name: &str, clip_id: &str) -> Result<bool> {
    let actual_id = clip_id.split(' ').next().unwrap_or(clip_id).trim();
    let clip_dir = get_clip_dir(base_dir, project_name, actual_id)?;
    let mut deleted = false;
    if clip_dir.exists() {
        fs::remove_dir_all(&clip_dir)?;
        deleted = true;
    }
    let mut index = load_project_index(base_dir, project_name)?;
    if index.remove(actual_id).is_some() {
        save_project_index(base_dir, project_name, &index)?;
        deleted = true;
    }
    Ok(deleted)
}

/// Update a clip's rating in both `meta.json` and `index.json`.
pub fn update_clip_rating(
    base_dir: &Path,
    project_name: &str,
    clip_id: &str,
    rating: i32,
) -> Result<ClipMeta> {
    let clip_dir = get_clip_dir(base_dir, project_name, clip_id)?;
    let meta_path = clip_dir.join("meta.json");
    if !meta_path.exists() {
        return Err(PrefixStreamError::NotFound(format!(
            "Clip '{clip_id}' not found"
        )));
    }
    let raw = fs::read_to_string(&meta_path)?;
    let mut meta: ClipMeta = serde_json::from_str(&raw)?;
    meta.rating = rating;
    atomic_write_json(&meta_path, &meta)?;

    let mut index = load_project_index(base_dir, project_name)?;
    index.insert(clip_id.to_string(), meta.clone());
    save_project_index(base_dir, project_name, &index)?;
    Ok(meta)
}

/// Build a JSON summary of a project's clips and project list (mirrors `get_project_clips` API).
pub fn get_project_clips_summary(base_dir: &Path, project_name: &str) -> Result<Value> {
    let projects = list_projects(base_dir);
    let active_proj = if project_name.is_empty() {
        projects
            .first()
            .cloned()
            .unwrap_or_else(|| "Default_Project".to_string())
    } else {
        project_name.to_string()
    };
    let index = load_project_index(base_dir, &active_proj)?;
    Ok(json!({
        "status": "success",
        "projects": projects,
        "active_project": active_proj,
        "clips": index,
    }))
}
