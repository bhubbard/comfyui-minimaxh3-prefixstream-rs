//! Rust port of `tests/test_clip_bin.py`.

use std::fs;

use minimaxh3_prefixstream::engine::clip_bin_api::get_project_clips_api;
use minimaxh3_prefixstream::engine::clip_bin_manager::{
    delete_clip_asset, get_clip_dir, get_clips_for_selection, list_projects, load_project_index,
    resolve_source_video_path, save_clip_asset, update_clip_rating, SaveClipRequest,
};
use minimaxh3_prefixstream::nodes::{MiniMaxClipBinPickerNode, MiniMaxClipBinSaverNode};
use minimaxh3_prefixstream::tensor::{AudioData, Tensor};
use serde_json::{json, Value};

#[test]
fn test_save_and_load_clip_bin_asset() {
    let tmp = tempfile::tempdir().unwrap();
    let v = Tensor::randn([1, 16, 4, 8, 8], 1);
    let a = Tensor::randn([1, 64, 10], 2);
    let images = Tensor::rand([4, 32, 32, 3], 3);
    let audio = AudioData {
        waveform: Tensor::zeros([1, 2, 16000]),
        sample_rate: 32000,
    };

    let saver = MiniMaxClipBinSaverNode;
    let (clip_id, _summary) = saver
        .save_to_bin(SaveClipRequest {
            base_dir: tmp.path(),
            project_name: "MyProject",
            parent_id: None,
            images: Some(&images),
            video_latent: Some(&v),
            audio_latent: Some(&a),
            audio: Some(&audio),
            prompt: "Cinematic sunrise",
            seed: 42,
            fps: 24.0,
            save_tail_frames: 4,
            source_video_path: None,
            auto_encode_mp4: false,
        })
        .unwrap();

    assert!(clip_id.starts_with("clip_"));
    let clip_dir = get_clip_dir(tmp.path(), "MyProject", &clip_id).unwrap();
    assert!(clip_dir.is_dir());

    let picker = MiniMaxClipBinPickerNode;
    let loaded = picker
        .pick_clip(tmp.path(), "MyProject", "Continue", 0, &clip_id)
        .unwrap();

    assert_eq!(loaded.clip_id, clip_id);
    let lat = loaded.latent.unwrap();
    assert!(lat.video.unwrap().allclose(&v, 1e-6));
    assert!(lat.audio.unwrap().allclose(&a, 1e-6));
    assert!(loaded.tail_frame.allclose(&images, 1e-6));
    assert_eq!(loaded.prompt, "Cinematic sunrise");
}

#[test]
fn test_dag_branching_and_chain_assembly() {
    let tmp = tempfile::tempdir().unwrap();
    let v1 = Tensor::ones([1, 16, 4, 8, 8]);
    let v2 = Tensor::full([1, 16, 4, 8, 8], 2.0);
    let v3 = Tensor::full([1, 16, 4, 8, 8], 3.0);

    let img1 = Tensor::full([10, 16, 16, 3], 0.2);
    let img2 = Tensor::full([10, 16, 16, 3], 0.5);
    let img3 = Tensor::full([10, 16, 16, 3], 0.8);

    // 1. Root clip (Scene 1 Take 1)
    let r1 = save_clip_asset(SaveClipRequest {
        base_dir: tmp.path(),
        project_name: "DAGProject",
        parent_id: None,
        images: Some(&img1),
        video_latent: Some(&v1),
        audio_latent: None,
        audio: None,
        prompt: "Root",
        seed: 1,
        fps: 24.0,
        save_tail_frames: 10,
        source_video_path: None,
        auto_encode_mp4: false,
    })
    .unwrap();
    assert_eq!(r1.scene_num, 1);
    assert_eq!(r1.take_num, 1);

    // 2. Child take A (branched from r1 -> Scene 2 Take 1)
    let r2 = save_clip_asset(SaveClipRequest {
        base_dir: tmp.path(),
        project_name: "DAGProject",
        parent_id: Some(&r1.clip_id),
        images: Some(&img2),
        video_latent: Some(&v2),
        audio_latent: None,
        audio: None,
        prompt: "Branch A",
        seed: 2,
        fps: 24.0,
        save_tail_frames: 10,
        source_video_path: None,
        auto_encode_mp4: false,
    })
    .unwrap();
    assert_eq!(r2.scene_num, 2);
    assert_eq!(r2.take_num, 1);

    // 3. Child take B (also branched from r1 -> Scene 2 Take 2)
    let r3 = save_clip_asset(SaveClipRequest {
        base_dir: tmp.path(),
        project_name: "DAGProject",
        parent_id: Some(&r1.clip_id),
        images: Some(&img3),
        video_latent: Some(&v3),
        audio_latent: None,
        audio: None,
        prompt: "Branch B",
        seed: 3,
        fps: 24.0,
        save_tail_frames: 10,
        source_video_path: None,
        auto_encode_mp4: false,
    })
    .unwrap();
    assert_eq!(r3.scene_num, 2);
    assert_eq!(r3.take_num, 2);

    let projects = list_projects(tmp.path());
    assert!(projects.contains(&"DAGProject".to_string()));

    let clips = get_clips_for_selection(tmp.path(), "DAGProject", true).unwrap();
    assert_eq!(clips.len(), 4); // "None (Root / Scene 1)" + 3 clips

    update_clip_rating(tmp.path(), "DAGProject", &r3.clip_id, 5).unwrap();
    let best = MiniMaxClipBinPickerNode
        .pick_clip(tmp.path(), "DAGProject", "Continue", 5, "latest")
        .unwrap();
    assert_eq!(best.clip_id, r3.clip_id);
}

#[test]
fn test_clip_bin_saver_video_file_name_from_vhs() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("MINIMAX_ALLOWED_VIDEO_DIR", tmp.path());
    let vhs_subfolder = tmp.path().join("my_subfolder");
    fs::create_dir_all(&vhs_subfolder).unwrap();
    let mock_vhs_mp4 = vhs_subfolder.join("my_video_0001.mp4");
    fs::write(&mock_vhs_mp4, b"MOCK_MP4_CONTENT_12345").unwrap();

    let v = Tensor::randn([1, 16, 4, 8, 8], 77);
    let meta = save_clip_asset(SaveClipRequest {
        base_dir: tmp.path(),
        project_name: "VHSTest",
        parent_id: None,
        images: None,
        video_latent: Some(&v),
        audio_latent: None,
        audio: None,
        prompt: "A majestic lion",
        seed: 7,
        fps: 24.0,
        save_tail_frames: 4,
        source_video_path: Some(mock_vhs_mp4.to_str().unwrap()),
        auto_encode_mp4: false,
    })
    .unwrap();

    let bin_dir = get_clip_dir(tmp.path(), "VHSTest", &meta.clip_id).unwrap();
    let meta_json_path = bin_dir.join("meta.json");
    assert!(meta_json_path.is_file());
    let data: Value = serde_json::from_str(&fs::read_to_string(&meta_json_path).unwrap()).unwrap();
    assert_eq!(data["prompt"], json!("A majestic lion"));
    assert_eq!(data["has_source_video"], json!(true));

    let archived_video = bin_dir.join("source_video.mp4");
    assert!(archived_video.is_file());
    assert_eq!(
        fs::read(&archived_video).unwrap(),
        b"MOCK_MP4_CONTENT_12345"
    );

    let api_resp = get_project_clips_api(tmp.path(), "VHSTest");
    assert_eq!(api_resp.status_code, 200);
    assert_eq!(
        api_resp.body["clips"][&meta.clip_id]["has_source_video"],
        json!(true)
    );
}

#[test]
fn test_clip_bin_saver_prefers_vhs_audio_video() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("MINIMAX_ALLOWED_VIDEO_DIR", tmp.path());
    let muxed = tmp.path().join("clip-audio.mp4");
    fs::write(&muxed, b"video with audio").unwrap();

    let resolved = resolve_source_video_path(muxed.to_str()).unwrap();
    assert_eq!(resolved, fs::canonicalize(&muxed).unwrap());
}

#[test]
fn test_clip_bin_saver_auto_encode_video_from_images() {
    let tmp = tempfile::tempdir().unwrap();
    let v = Tensor::randn([1, 16, 4, 8, 8], 91);
    let images = Tensor::rand([12, 32, 32, 3], 92);
    let audio = AudioData {
        waveform: Tensor::zeros([1, 2, 16000]),
        sample_rate: 32000,
    };

    let meta = save_clip_asset(SaveClipRequest {
        base_dir: tmp.path(),
        project_name: "AutoEncodeTest",
        parent_id: None,
        images: Some(&images),
        video_latent: Some(&v),
        audio_latent: None,
        audio: Some(&audio),
        prompt: "A dancing robot",
        seed: 12,
        fps: 24.0,
        save_tail_frames: 4,
        source_video_path: None,
        auto_encode_mp4: true,
    })
    .unwrap();

    let bin_dir = get_clip_dir(tmp.path(), "AutoEncodeTest", &meta.clip_id).unwrap();
    let archived_video = bin_dir.join("source_video.mp4");
    if archived_video.is_file() {
        assert!(fs::metadata(&archived_video).unwrap().len() > 0);
        assert!(meta.has_source_video);
    }
}

#[test]
fn test_delete_clip_asset() {
    let tmp = tempfile::tempdir().unwrap();
    let v = Tensor::randn([1, 16, 4, 8, 8], 99);

    let meta = save_clip_asset(SaveClipRequest {
        base_dir: tmp.path(),
        project_name: "DeleteTest",
        parent_id: None,
        images: None,
        video_latent: Some(&v),
        audio_latent: None,
        audio: None,
        prompt: "",
        seed: 0,
        fps: 24.0,
        save_tail_frames: 4,
        source_video_path: None,
        auto_encode_mp4: false,
    })
    .unwrap();

    let bin_dir = get_clip_dir(tmp.path(), "DeleteTest", &meta.clip_id).unwrap();
    assert!(bin_dir.is_dir());

    let deleted = delete_clip_asset(tmp.path(), "DeleteTest", &meta.clip_id).unwrap();
    assert!(deleted);
    assert!(!bin_dir.is_dir());

    let idx = load_project_index(tmp.path(), "DeleteTest").unwrap();
    assert!(idx.is_empty());
}
