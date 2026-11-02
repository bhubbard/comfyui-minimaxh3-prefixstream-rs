//! HTTP / JSON API handlers for the interactive Clip Bin Tree Picker.
//!
//! Rust port of `engine/clip_bin_api.py`.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::engine::clip_bin_manager::{
    delete_clip_asset, get_clip_dir, get_project_clips_summary, rebuild_project_index,
    update_clip_rating,
};
use crate::error::{PrefixStreamError, Result};

/// Structured API response mirroring aiohttp `web.json_response(payload, status=...)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    /// HTTP status code (`200`, `400`, `403`, `404`, `500`).
    pub status_code: u16,
    /// JSON response body.
    pub body: Value,
}

/// Handle `GET /minimax/clip_bin/projects?project=...`.
pub fn get_project_clips_api(base_dir: &Path, project_name: &str) -> ApiResponse {
    match get_project_clips_summary(base_dir, project_name) {
        Ok(body) => ApiResponse {
            status_code: 200,
            body,
        },
        Err(e) => ApiResponse {
            status_code: 500,
            body: json!({ "status": "error", "message": e.to_string() }),
        },
    }
}

/// Handle `GET /minimax/clip_bin/preview?project=...&clip_id=...&type=image|video`.
pub fn get_clip_preview_path_api(
    base_dir: &Path,
    project_name: &str,
    clip_id: &str,
    media_type: &str,
) -> Result<std::path::PathBuf> {
    let clip_dir = get_clip_dir(base_dir, project_name, clip_id)?;
    let file_path = if media_type == "video" {
        clip_dir.join("source_video.mp4")
    } else {
        clip_dir.join("preview.png")
    };
    if !file_path.exists() {
        return Err(PrefixStreamError::NotFound(format!(
            "Media file not found: {}",
            file_path.display()
        )));
    }
    Ok(file_path)
}

/// Handle `POST /minimax/clip_bin/rate`.
pub fn update_clip_rating_api(
    base_dir: &Path,
    project_name: &str,
    clip_id: &str,
    rating: i32,
) -> ApiResponse {
    match update_clip_rating(base_dir, project_name, clip_id, rating) {
        Ok(_) => ApiResponse {
            status_code: 200,
            body: json!({
                "status": "success",
                "clip_id": clip_id,
                "rating": rating,
            }),
        },
        Err(PrefixStreamError::SecurityViolation(msg)) => ApiResponse {
            status_code: 403,
            body: json!({ "status": "error", "message": msg }),
        },
        Err(PrefixStreamError::NotFound(msg)) => ApiResponse {
            status_code: 404,
            body: json!({ "status": "error", "message": msg }),
        },
        Err(e) => ApiResponse {
            status_code: 500,
            body: json!({ "status": "error", "message": e.to_string() }),
        },
    }
}

/// Handle `POST /minimax/clip_bin/delete`.
pub fn delete_clip_api(base_dir: &Path, project_name: &str, clip_id: &str) -> ApiResponse {
    match delete_clip_asset(base_dir, project_name, clip_id) {
        Ok(deleted) => ApiResponse {
            status_code: 200,
            body: json!({ "status": "success", "deleted": deleted }),
        },
        Err(PrefixStreamError::SecurityViolation(msg)) => ApiResponse {
            status_code: 403,
            body: json!({ "status": "error", "message": msg }),
        },
        Err(e) => ApiResponse {
            status_code: 500,
            body: json!({ "status": "error", "message": e.to_string() }),
        },
    }
}

/// Handle `POST /minimax/clip_bin/rescan`.
pub fn rescan_project_api(base_dir: &Path, project_name: &str) -> ApiResponse {
    match rebuild_project_index(base_dir, project_name) {
        Ok(index) => ApiResponse {
            status_code: 200,
            body: json!({ "status": "success", "clips": index }),
        },
        Err(PrefixStreamError::SecurityViolation(msg)) => ApiResponse {
            status_code: 403,
            body: json!({ "status": "error", "message": msg }),
        },
        Err(e) => ApiResponse {
            status_code: 500,
            body: json!({ "status": "error", "message": e.to_string() }),
        },
    }
}
