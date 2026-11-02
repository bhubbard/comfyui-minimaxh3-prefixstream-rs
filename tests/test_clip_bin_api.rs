//! Rust port of `tests/test_clip_bin_api.py`.

use minimaxh3_prefixstream::engine::clip_bin_api::{
    delete_clip_api, get_clip_preview_path_api, get_project_clips_api, rescan_project_api,
    update_clip_rating_api,
};
use minimaxh3_prefixstream::engine::clip_bin_manager::{save_clip_asset, SaveClipRequest};
use minimaxh3_prefixstream::tensor::{AudioData, Tensor};
use serde_json::json;

#[test]
fn test_get_project_clips_api_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let resp = get_project_clips_api(tmp.path(), "NonExistentProject_XYZ");
    assert_eq!(resp.status_code, 200);
    assert_eq!(resp.body["status"], json!("success"));
    assert_eq!(resp.body["active_project"], json!("NonExistentProject_XYZ"));
    assert_eq!(resp.body["clips"], json!({}));
}

#[test]
fn test_get_project_clips_api_with_assets_and_ratings() {
    let tmp = tempfile::tempdir().unwrap();
    let v = Tensor::randn([1, 16, 4, 8, 8], 11);
    let a = Tensor::randn([1, 64, 10], 12);
    let images = Tensor::rand([4, 32, 32, 3], 13);
    let audio = AudioData {
        waveform: Tensor::zeros([1, 2, 16000]),
        sample_rate: 32000,
    };

    let c1 = save_clip_asset(SaveClipRequest {
        base_dir: tmp.path(),
        project_name: "APITestProj",
        parent_id: None,
        images: Some(&images),
        video_latent: Some(&v),
        audio_latent: Some(&a),
        audio: Some(&audio),
        prompt: "First prompt",
        seed: 100,
        fps: 24.0,
        save_tail_frames: 4,
        source_video_path: None,
        auto_encode_mp4: false,
    })
    .unwrap();

    let c2 = save_clip_asset(SaveClipRequest {
        base_dir: tmp.path(),
        project_name: "APITestProj",
        parent_id: Some(&c1.clip_id),
        images: Some(&images),
        video_latent: Some(&v),
        audio_latent: None,
        audio: None,
        prompt: "Second prompt",
        seed: 101,
        fps: 24.0,
        save_tail_frames: 4,
        source_video_path: None,
        auto_encode_mp4: false,
    })
    .unwrap();

    let api_resp = get_project_clips_api(tmp.path(), "APITestProj");
    assert_eq!(api_resp.status_code, 200);
    let clips = api_resp.body["clips"].as_object().unwrap();
    assert_eq!(clips.len(), 2);
    assert_eq!(clips[&c1.clip_id]["prompt"], json!("First prompt"));
    assert_eq!(clips[&c1.clip_id]["scene_num"], json!(1));
    assert_eq!(clips[&c2.clip_id]["parent_id"], json!(c1.clip_id));
    assert_eq!(clips[&c2.clip_id]["scene_num"], json!(2));

    // Preview PNG path resolution
    let preview_path =
        get_clip_preview_path_api(tmp.path(), "APITestProj", &c1.clip_id, "image").unwrap();
    assert!(preview_path.is_file());

    // Update rating via API
    let rate_resp = update_clip_rating_api(tmp.path(), "APITestProj", &c1.clip_id, 5);
    assert_eq!(rate_resp.status_code, 200);
    let after_resp = get_project_clips_api(tmp.path(), "APITestProj");
    assert_eq!(after_resp.body["clips"][&c1.clip_id]["rating"], json!(5));
}

#[test]
fn test_register_routes_safe_without_server() {
    let tmp = tempfile::tempdir().unwrap();
    let rescan = rescan_project_api(tmp.path(), "EmptyRescanProj");
    assert_eq!(rescan.status_code, 200);
    let del = delete_clip_api(tmp.path(), "EmptyRescanProj", "clip_001_01");
    assert_eq!(del.status_code, 200);
    assert_eq!(del.body["deleted"], json!(false));
}
