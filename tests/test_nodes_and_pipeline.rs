//! Rust port of `tests/test_nodes_and_pipeline.py`.

use minimaxh3_prefixstream::engine::clip_bin_manager::{save_clip_asset, SaveClipRequest};
use minimaxh3_prefixstream::nodes::{
    node_display_name_mappings, validate_workflow_json, MiniMaxCacheMonitorNode,
    MiniMaxClipBinPickerNode, MiniMaxLoadLatentNode, MiniMaxLongVideoStitcherNode,
    MiniMaxPrefixCacheApplierNode, MiniMaxPrefixCacheConfigNode, MiniMaxSaveLatentNode,
    MiniMaxTrimPrefixLatentNode, NODE_CLASS_NAMES,
};
use minimaxh3_prefixstream::tensor::{AudioData, AvLatent, ConditioningEntry, Tensor};

#[test]
fn test_node_mappings_consistency() {
    let display = node_display_name_mappings();
    assert_eq!(NODE_CLASS_NAMES.len(), 15);
    assert_eq!(display.len(), 15);
    for class_name in NODE_CLASS_NAMES {
        assert!(
            display.contains_key(class_name),
            "Missing display name for {}",
            class_name
        );
    }
}

#[test]
fn test_config_node() {
    let node = MiniMaxPrefixCacheConfigNode;
    let (config, _sess) = node.configure("Native Masked AV", 22, 5, true, true, 0.0, 0.5, 0.0);
    assert_eq!(config.continuation_mode, "Native Masked AV");
    assert_eq!(config.rolling_frames, 22);
    assert_eq!(config.anchor_frames, 5);
}

#[test]
fn test_applier_exposes_only_continuation_inputs() {
    let (_, mut sess) = MiniMaxPrefixCacheConfigNode.configure(
        "Native Masked AV",
        22,
        5,
        true,
        true,
        0.0,
        0.5,
        0.0,
    );
    let applier = MiniMaxPrefixCacheApplierNode;
    let cond = vec![ConditioningEntry::default()];
    let prev = AvLatent::from_video_audio(
        Tensor::randn([1, 4, 12, 4, 4], 1),
        Tensor::randn([1, 4, 2, 65], 2),
    );
    let tgt = AvLatent::from_video_audio(
        Tensor::zeros([1, 4, 12, 4, 4]),
        Tensor::zeros([1, 4, 2, 65]),
    );
    let out = applier
        .apply_cache(
            &mut sess,
            &cond,
            Some(&prev),
            Some(&tgt),
            None,
            None,
            None,
            None,
            None,
            None,
            1.0,
            1.0,
            "Match Video Handover",
            0,
            0.0,
            0,
        )
        .unwrap();
    assert!(out.masked_av_plan.is_some());
    assert!(out.latent.is_some());
}

#[test]
fn test_av_latent_pack_unpack() {
    let v = Tensor::randn([1, 16, 8, 8, 8], 10);
    let a = Tensor::randn([1, 64, 20], 11);
    let packed = AvLatent::from_video_audio(v.clone(), a.clone());
    assert!(packed.video.as_ref().unwrap().allclose(&v, 1e-6));
    assert!(packed.audio.as_ref().unwrap().allclose(&a, 1e-6));

    let v_only = AvLatent::from_video(v.clone());
    assert!(v_only.video.as_ref().unwrap().allclose(&v, 1e-6));
    assert!(v_only.audio.is_none());
}

#[test]
fn test_save_and_load_latent_node() {
    let tmp = tempfile::tempdir().unwrap();
    let v = Tensor::randn([1, 16, 8, 8, 8], 20);
    let a = Tensor::randn([1, 64, 20], 21);
    let latent = AvLatent::from_video_audio(v.clone(), a.clone());

    let saver = MiniMaxSaveLatentNode;
    let saved_path = tmp.path().join("test_av_latent.safetensors");
    saver.save_latent(&saved_path, &latent, None).unwrap();
    assert!(saved_path.is_file());

    let loader = MiniMaxLoadLatentNode;
    let (loaded, _state) = loader.load_latent(&saved_path).unwrap();
    assert!(loaded.video.as_ref().unwrap().allclose(&v, 1e-6));
    assert!(loaded.audio.as_ref().unwrap().allclose(&a, 1e-6));
}

#[test]
fn test_trim_node_pixel_and_waveform() {
    let trim_node = MiniMaxTrimPrefixLatentNode;
    let images = Tensor::rand([50, 32, 32, 3], 30);
    let audio = AudioData {
        waveform: Tensor::randn([1, 2, 66667], 31),
        sample_rate: 32000,
    };

    let out = trim_node
        .trim_frames(
            22,
            Some(&images),
            Some(&audio),
            None,
            None,
            None,
            24.0,
            true,
        )
        .unwrap();
    let trimmed_imgs = out.images.unwrap();
    let trimmed_aud = out.audio.unwrap();
    assert_eq!(trimmed_imgs.shape, vec![28, 32, 32, 3]);
    let expected_samples = ((28.0_f64 / 24.0) * 32000.0).round() as usize;
    assert_eq!(trimmed_aud.waveform.shape[2], expected_samples);
}

#[test]
fn test_trim_node_initial_clip_guard() {
    let (config, mut sess) = MiniMaxPrefixCacheConfigNode.configure(
        "Native Masked AV",
        22,
        5,
        true,
        true,
        0.0,
        0.5,
        0.0,
    );
    assert_eq!(sess.last_rolling_frames, 0);

    let trim_node = MiniMaxTrimPrefixLatentNode;
    let images = Tensor::rand([50, 32, 32, 3], 40);
    let audio = AudioData {
        waveform: Tensor::randn([1, 2, 66667], 41),
        sample_rate: 32000,
    };

    // Session connected with last_rolling_frames == 0 -> must preserve all 50 frames!
    let out1 = trim_node
        .trim_frames(
            0,
            Some(&images),
            Some(&audio),
            None,
            Some(&mut sess),
            Some(&config),
            24.0,
            true,
        )
        .unwrap();
    assert_eq!(out1.images.unwrap().shape[0], 50);

    // No session connected, only cache_config connected, and no noise_mask in latent -> must preserve all 50 frames!
    let initial_latent = AvLatent::from_video(Tensor::zeros([1, 16, 15, 8, 8]));
    let out2 = trim_node
        .trim_frames(
            0,
            Some(&images),
            Some(&audio),
            Some(&initial_latent),
            None,
            Some(&config),
            24.0,
            true,
        )
        .unwrap();
    assert_eq!(out2.images.unwrap().shape[0], 50);
}

#[test]
fn test_long_video_stitcher_node() {
    let stitcher = MiniMaxLongVideoStitcherNode;
    let prev_imgs = Tensor::ones([50, 32, 32, 3]).scale(0.4);
    let curr_imgs = Tensor::ones([50, 32, 32, 3]).scale(0.5);
    let prev_aud = AudioData {
        waveform: Tensor::ones([1, 2, 66667]).scale(0.1),
        sample_rate: 32000,
    };
    let curr_aud = AudioData {
        waveform: Tensor::ones([1, 2, 66667]).scale(0.1),
        sample_rate: 32000,
    };

    // Initial clip mode (previous_images = None)
    let (init_imgs, _, _) = stitcher
        .stitch(
            None,
            None,
            Some(&curr_imgs),
            None,
            Some(&curr_aud),
            None,
            None,
            0,
            4,
            15.0,
            true,
            24.0,
        )
        .unwrap();
    assert_eq!(init_imgs.unwrap().shape[0], 50);

    // Continuation clip mode (trim 22 frames -> 50 + 28 = 78 frames)
    let (cont_imgs, _, _) = stitcher
        .stitch(
            None,
            Some(&prev_imgs),
            Some(&curr_imgs),
            Some(&prev_aud),
            Some(&curr_aud),
            None,
            None,
            22,
            4,
            15.0,
            true,
            24.0,
        )
        .unwrap();
    assert_eq!(cont_imgs.unwrap().shape[0], 78);

    // Telemetry monitor
    let (_, sess) = MiniMaxPrefixCacheConfigNode.configure(
        "Native Masked AV",
        22,
        5,
        true,
        true,
        0.0,
        0.5,
        0.0,
    );
    let report = MiniMaxCacheMonitorNode.monitor(&sess);
    assert!(report.contains("MiniMax H3 PrefixStream Telemetry"));
}

#[test]
fn test_clip_bin_tree_picker_node_and_workflow_integrity() {
    let tmp = tempfile::tempdir().unwrap();
    let v = Tensor::randn([1, 16, 4, 8, 8], 50);
    let a = Tensor::randn([1, 64, 10], 51);
    let imgs = Tensor::rand([8, 32, 32, 3], 52);

    let saved = save_clip_asset(SaveClipRequest {
        base_dir: tmp.path(),
        project_name: "TreePickerTest",
        parent_id: None,
        images: Some(&imgs),
        video_latent: Some(&v),
        audio_latent: Some(&a),
        audio: None,
        prompt: "Tree prompt",
        seed: 42,
        fps: 24.0,
        save_tail_frames: 8,
        source_video_path: None,
        auto_encode_mp4: false,
    })
    .unwrap();

    let picker = MiniMaxClipBinPickerNode;
    let out = picker
        .pick_clip(tmp.path(), "TreePickerTest", "Continue", 0, &saved.clip_id)
        .unwrap();
    assert_eq!(out.clip_id, saved.clip_id);
    assert_eq!(out.tail_frame.shape, vec![8, 32, 32, 3]);

    let wf_json = r#"{"nodes":[{"id":1,"type":"MiniMaxPrefixCacheConfig"},{"id":2,"type":"MiniMaxPrefixCacheApplier"}]}"#;
    let report = validate_workflow_json(wf_json).unwrap();
    assert!(report.valid);
    assert_eq!(report.minimax_nodes.len(), 2);
}
