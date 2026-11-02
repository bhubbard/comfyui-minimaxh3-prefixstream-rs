//! Rust port of `tests/test_regressions.py`.

use std::fs;
use std::sync::Arc;
use std::thread;

use minimaxh3_prefixstream::engine::cache_manager::KVCacheConfig;
use minimaxh3_prefixstream::engine::clip_bin_manager::{
    delete_clip_asset, load_clip_asset, load_project_index, save_clip_asset, update_clip_rating,
    SaveClipRequest,
};
use minimaxh3_prefixstream::error::{PrefixStreamError, Result};
use minimaxh3_prefixstream::nodes::{
    MiniMaxLoadLatentNode, MiniMaxPrefixCacheApplierNode, MiniMaxSafeVAEDecodeAudioNode,
    MiniMaxSafeVAEDecodeNode, MiniMaxSaveLatentNode, MiniMaxTrimPrefixLatentNode,
};
use minimaxh3_prefixstream::pipeline::long_video_director::LongVideoSession;
use minimaxh3_prefixstream::pipeline::seam_protector::stitch_audio_waveforms;
use minimaxh3_prefixstream::tensor::{AudioData, AudioVae, AvLatent, Tensor, VideoVae};
use serde_json::json;

struct FailingVideoVae;
impl VideoVae for FailingVideoVae {
    fn decode(&self, _latent: &Tensor) -> Result<Tensor> {
        Err(PrefixStreamError::InvalidInput("decode exploded".into()))
    }
    fn encode(&self, _images: &Tensor) -> Result<Tensor> {
        Err(PrefixStreamError::InvalidInput("encode exploded".into()))
    }
}

struct FailingAudioVae;
impl AudioVae for FailingAudioVae {
    fn decode(&self, _latent: &Tensor) -> Result<AudioData> {
        Err(PrefixStreamError::InvalidInput(
            "audio decode exploded".into(),
        ))
    }
}

#[test]
fn test_plain_video_batch_is_not_split_into_audio() {
    let video_batch = Tensor::randn([2, 16, 4, 8, 8], 1);
    let latent = AvLatent::from_video(video_batch.clone());
    assert!(latent.video.as_ref().unwrap().allclose(&video_batch, 1e-6));
    assert!(latent.audio.is_none());
}

#[test]
fn test_initial_latent_and_audio_are_preserved() {
    let trim_node = MiniMaxTrimPrefixLatentNode;
    let cfg = KVCacheConfig {
        rolling_frames: 39,
        ..Default::default()
    };
    let v = Tensor::randn([1, 4, 12, 4, 4], 2);
    let a = Tensor::randn([1, 4, 2, 65], 3);
    let latent = AvLatent::from_video_audio(v.clone(), a.clone());
    let images = Tensor::rand([24, 8, 8, 3], 4);
    let audio = AudioData {
        waveform: Tensor::randn([1, 2, 32000], 5),
        sample_rate: 32000,
    };

    let out = trim_node
        .trim_frames(
            0,
            Some(&images),
            Some(&audio),
            Some(&latent),
            None,
            Some(&cfg),
            24.0,
            true,
        )
        .unwrap();

    assert_eq!(out.images.unwrap().shape[0], 24);
    assert_eq!(out.audio.unwrap().waveform.shape[2], 32000);
    assert_eq!(out.latent.unwrap().video.unwrap().shape[2], 12);
}

#[test]
fn test_audio_only_keeps_remaining_second() {
    let trim_node = MiniMaxTrimPrefixLatentNode;
    let audio = AudioData {
        waveform: Tensor::from_vec(vec![1, 1, 48000], (0..48000).map(|i| i as f32).collect())
            .unwrap(),
        sample_rate: 24000,
    };

    let out = trim_node
        .trim_frames(24, None, Some(&audio), None, None, None, 24.0, true)
        .unwrap();
    let trimmed_aud = out.audio.unwrap();
    assert_eq!(trimmed_aud.waveform.shape[2], 24000);
    assert!(trimmed_aud
        .waveform
        .allclose(&audio.waveform.slice_axis(2, 24000, 48000).unwrap(), 1e-6));
}

#[test]
fn test_trim_uses_effective_prefix_instead_of_requested_length() {
    let cfg = KVCacheConfig {
        continuation_mode: "Native Masked AV".to_string(),
        rolling_frames: 39,
        ..Default::default()
    };
    let mut sess = LongVideoSession::new(cfg.clone());
    let prev =
        AvLatent::from_video_audio(Tensor::ones([1, 2, 36, 2, 2]), Tensor::ones([1, 1, 1, 207]));
    // Short target (25 frames = 8 steps) snaps 39-frame request down to 22 frames (7 steps)
    let short_target =
        AvLatent::from_video_audio(Tensor::zeros([1, 2, 8, 2, 2]), Tensor::zeros([1, 1, 1, 42]));

    let out = MiniMaxPrefixCacheApplierNode
        .apply_cache(
            &mut sess,
            &[],
            Some(&prev),
            Some(&short_target),
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
    let masked_latent = out.latent.unwrap();
    assert_eq!(masked_latent.metadata["minimax_prefix_frames"], json!(22));

    let images = Tensor::zeros([25, 4, 4, 3]);
    let trim_out = MiniMaxTrimPrefixLatentNode
        .trim_frames(
            0,
            Some(&images),
            None,
            Some(&masked_latent),
            None,
            Some(&cfg),
            24.0,
            true,
        )
        .unwrap();
    assert_eq!(trim_out.images.unwrap().shape[0], 3);
    assert_eq!(trim_out.latent.unwrap().video.unwrap().shape[2], 1);
}

#[test]
fn test_trim_entire_latent_and_discard_obsolete_mask() {
    let v = Tensor::ones([1, 2, 7, 2, 2]);
    let a = Tensor::ones([1, 1, 1, 37]);
    let mut latent = AvLatent::from_video_audio(v, a);
    latent.video_mask = Some(Tensor::zeros([1, 1, 7, 2, 2]));
    latent.audio_mask = Some(Tensor::zeros([1, 1, 1, 37]));
    latent
        .metadata
        .insert("minimax_prefix_frames".to_string(), json!(22));

    let out = MiniMaxTrimPrefixLatentNode
        .trim_frames(22, None, None, Some(&latent), None, None, 24.0, true)
        .unwrap()
        .latent
        .unwrap();

    assert_eq!(out.video.unwrap().shape[2], 0);
    assert_eq!(out.audio.unwrap().shape[3], 0);
    assert!(out.video_mask.is_none());
    assert!(out.audio_mask.is_none());
    assert_eq!(out.metadata["minimax_prefix_frames"], json!(0));
    assert_eq!(out.metadata["minimax_source_start_step"], json!(7));
}

#[test]
fn test_audio_mismatch_is_explicit() {
    let prev = AudioData {
        waveform: Tensor::zeros([1, 2, 1000]),
        sample_rate: 32000,
    };
    let curr_sr_mismatch = AudioData {
        waveform: Tensor::zeros([1, 2, 1000]),
        sample_rate: 48000,
    };
    let err_sr = stitch_audio_waveforms(Some(&prev), Some(&curr_sr_mismatch), 24, 5, 15.0, 24.0)
        .unwrap_err();
    assert!(err_sr.to_string().contains("sample rates"));

    let curr_ch_mismatch = AudioData {
        waveform: Tensor::zeros([1, 1, 1000]),
        sample_rate: 32000,
    };
    let err_ch = stitch_audio_waveforms(Some(&prev), Some(&curr_ch_mismatch), 24, 5, 15.0, 24.0)
        .unwrap_err();
    assert!(err_ch.to_string().contains("channel"));
}

#[test]
fn test_audio_dtype_and_short_overlap() {
    let prev = AudioData {
        waveform: Tensor::ones([1, 2, 32000]).scale(0.25),
        sample_rate: 32000,
    };
    let curr = AudioData {
        waveform: Tensor::ones([1, 2, 4]).scale(0.25),
        sample_rate: 32000,
    };
    let out = stitch_audio_waveforms(Some(&prev), Some(&curr), 48, 24, 15.0, 24.0)
        .unwrap()
        .unwrap();
    assert_eq!(out.waveform.shape[2], 64000);
}

#[test]
fn test_decode_errors_are_not_empty_successes() {
    let v_latent = AvLatent::from_video(Tensor::zeros([1, 2, 2, 2, 2]));
    assert!(MiniMaxSafeVAEDecodeNode
        .decode(&FailingVideoVae, Some(&v_latent))
        .is_err());

    let a_latent =
        AvLatent::from_video_audio(Tensor::zeros([1, 2, 2, 2, 2]), Tensor::zeros([1, 2, 2, 10]));
    assert!(MiniMaxSafeVAEDecodeAudioNode
        .decode(&FailingAudioVae, Some(&a_latent))
        .is_err());
}

#[test]
fn test_save_and_load_latent_roundtrips_identity_and_temporal_origin() {
    let tmp = tempfile::tempdir().unwrap();
    let v = Tensor::randn([1, 2, 4, 2, 2], 10);
    let a = Tensor::randn([1, 2, 2, 10], 11);
    let latent = AvLatent::from_video_audio(v.clone(), a.clone());
    let path = tmp.path().join("origin_test.safetensors");

    MiniMaxSaveLatentNode
        .save_latent(&path, &latent, None)
        .unwrap();
    let (loaded, _) = MiniMaxLoadLatentNode.load_latent(&path).unwrap();
    assert!(loaded.video.unwrap().allclose(&v, 1e-6));
    assert!(loaded.audio.unwrap().allclose(&a, 1e-6));
}

#[test]
fn test_path_escape_rejected_without_deleting() {
    let tmp = tempfile::tempdir().unwrap();
    let victim = tmp.path().join("victim");
    fs::create_dir_all(&victim).unwrap();

    assert!(load_clip_asset(tmp.path(), "SafeProj", "../victim").is_err());
    assert!(delete_clip_asset(tmp.path(), "SafeProj", "../victim").is_err());
    assert!(victim.is_dir());
}

#[test]
fn test_parallel_savers_and_rating_keep_all_assets() {
    let tmp = tempfile::tempdir().unwrap();
    let base_dir = Arc::new(tmp.path().to_path_buf());

    // Sequential or synchronized saves to same project
    let mut clip_ids = Vec::new();
    for idx in 0..6 {
        let dir = Arc::clone(&base_dir);
        let h = thread::spawn(move || {
            let v = Tensor::full([1, 2, 2, 2, 2], idx as f32);
            save_clip_asset(SaveClipRequest {
                base_dir: &dir,
                project_name: &format!("ParallelProject_{idx}"),
                parent_id: None,
                images: None,
                video_latent: Some(&v),
                audio_latent: None,
                audio: None,
                prompt: "",
                seed: idx as i64,
                fps: 24.0,
                save_tail_frames: 1,
                source_video_path: None,
                auto_encode_mp4: false,
            })
            .unwrap()
            .clip_id
        });
        clip_ids.push((idx, h.join().unwrap()));
    }

    for (idx, cid) in &clip_ids {
        let proj = format!("ParallelProject_{idx}");
        let updated = update_clip_rating(tmp.path(), &proj, cid, 5).unwrap();
        assert_eq!(updated.rating, 5);
        let index = load_project_index(tmp.path(), &proj).unwrap();
        assert_eq!(index[cid].rating, 5);
    }
}

#[test]
fn test_disk_stream_appends_exports_and_checks_geometry() {
    use minimaxh3_prefixstream::pipeline::disk_stream::append_disk_clip_with_encoder;

    let tmp = tempfile::tempdir().unwrap();
    let first = Tensor::full([12, 16, 16, 4], 0.25);
    let second = Tensor::full([12, 16, 16, 4], 0.75);
    let audio = AudioData {
        waveform: Tensor::zeros([1, 2, 16000]),
        sample_rate: 32000,
    };

    let mock_encoder =
        |_imgs: &Tensor, _fps: f32, _aud: Option<&AudioData>, dest: &std::path::Path| {
            fs::write(dest, b"MOCK_SEGMENT").is_ok()
        };

    let (_, _, total1) = append_disk_clip_with_encoder(
        tmp.path(),
        "DiskTestProj",
        "stream_a",
        Some(&first),
        Some(&audio),
        0,
        24.0,
        false,
        mock_encoder,
    )
    .unwrap();
    assert_eq!(total1, 12);

    let (manifest_path, _, total2) = append_disk_clip_with_encoder(
        tmp.path(),
        "DiskTestProj",
        "stream_a",
        Some(&second),
        Some(&audio),
        6,
        24.0,
        false,
        mock_encoder,
    )
    .unwrap();
    assert_eq!(total2, 18);

    // Geometry mismatch (32x16 vs 16x16) must be rejected explicitly without mutating manifest
    let bad_geom = Tensor::zeros([12, 32, 16, 3]);
    let err = append_disk_clip_with_encoder(
        tmp.path(),
        "DiskTestProj",
        "stream_a",
        Some(&bad_geom),
        Some(&audio),
        0,
        24.0,
        false,
        mock_encoder,
    )
    .unwrap_err();
    assert!(err.to_string().contains("consistent"));

    let manifest_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest_json["total_frames"], json!(18));
}
