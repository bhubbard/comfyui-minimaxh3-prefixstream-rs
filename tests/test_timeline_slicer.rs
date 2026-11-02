//! Rust port of `tests/test_timeline_slicer.py`.

use minimaxh3_prefixstream::engine::timeline_session_manager::{
    render_timeline_indicator_image, slice_video_and_audio, TimelineSession,
};
use minimaxh3_prefixstream::nodes::{
    MiniMaxVideoChunkSlicerNode, MiniMaxVideoPatchReassemblerNode,
};
use minimaxh3_prefixstream::tensor::{AudioData, Tensor};

#[test]
fn test_slice_video_and_audio_basic() {
    let images = Tensor::rand([100, 32, 32, 3], 1);
    let audio = AudioData {
        waveform: Tensor::randn([1, 2, 133333], 2),
        sample_rate: 32000,
    };

    let sliced = slice_video_and_audio(&images, Some(&audio), 24.0, 20, 50, false).unwrap();
    assert_eq!(sliced.actual_start, 20);
    assert_eq!(sliced.actual_end, 50);
    assert_eq!(sliced.chunk_images.shape, vec![30, 32, 32, 3]);
    let expected_samples = ((30.0_f64 / 24.0) * 32000.0).round() as usize;
    assert_eq!(sliced.chunk_audio.waveform.shape[2], expected_samples);
}

#[test]
fn test_timeline_session_initialize_and_patch() {
    let tmp = tempfile::tempdir().unwrap();
    let master_img = Tensor::zeros([100, 32, 32, 3]);
    let mut session =
        TimelineSession::load_or_create(tmp.path(), "UnitTest_TimelinePatch").unwrap();
    session
        .register_master_source(Some(&master_img), None, 24.0, true, None)
        .unwrap();

    assert_eq!(session.meta.total_frames, 100);
    assert!(session.meta.has_master_source);

    let patch = Tensor::ones([50, 32, 32, 3]);
    let patch_id = session
        .save_chunk_patch(&patch, None, 0, Some(50), 24.0, 4, "chunk 0")
        .unwrap();
    assert_eq!(patch_id, "patch_000000_000050");

    let indicator = render_timeline_indicator_image(
        100,
        Some((0, 50)),
        Some(&[(0, 50)]),
        24.0,
        "UnitTest_TimelinePatch",
    )
    .unwrap();
    assert_eq!(indicator.shape, vec![1, 140, 768, 3]);
}

#[test]
fn test_slicer_and_reassembler_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let master_img = Tensor::zeros([100, 48, 64, 3]);

    let slicer = MiniMaxVideoChunkSlicerNode;
    let reassembler = MiniMaxVideoPatchReassemblerNode;

    // 1. Slice Chunk 0 (0..50)
    let (chunk0_imgs, _chunk0_aud, ctx0, _ind0, info0) = slicer
        .slice_chunk(
            tmp.path(),
            "UnitTest_EndToEnd",
            &master_img,
            None,
            24.0,
            0,
            50,
            false,
            true,
        )
        .unwrap();
    assert_eq!(chunk0_imgs.shape, vec![50, 48, 64, 3]);
    assert!(info0.contains("Sliced [0..50]"));

    // 2. Patch Chunk 0 with 0.8
    let patch0 = Tensor::full([50, 48, 64, 3], 0.8);
    let (assembled0, _, _, report0) = reassembler
        .reassemble_patch(tmp.path(), &ctx0, &patch0, None, 4, true, "take 1")
        .unwrap();
    assert_eq!(assembled0.shape, vec![100, 48, 64, 3]);
    assert!(report0.contains("patch_000000_000050"));

    // 3. Slice Chunk 1 (50..100) and patch with 0.6
    let (_chunk1_imgs, _chunk1_aud, ctx1, _, _) = slicer
        .slice_chunk(
            tmp.path(),
            "UnitTest_EndToEnd",
            &master_img,
            None,
            24.0,
            50,
            100,
            false,
            false,
        )
        .unwrap();
    let patch1 = Tensor::full([50, 48, 64, 3], 0.6);
    let (assembled1, _, _, report1) = reassembler
        .reassemble_patch(tmp.path(), &ctx1, &patch1, None, 4, true, "take 2")
        .unwrap();
    assert_eq!(assembled1.shape, vec![100, 48, 64, 3]);
    assert!(report1.contains("Remaining Gaps: []"));
}

#[test]
fn test_padding_and_trimming_on_reassembly() {
    let tmp = tempfile::tempdir().unwrap();
    let master_img = Tensor::zeros([80, 32, 32, 3]);
    let slicer = MiniMaxVideoChunkSlicerNode;
    let reassembler = MiniMaxVideoPatchReassemblerNode;

    // Request 50..100 (50 frames) on an 80-frame video with pad_to_length = true
    let (chunk_imgs, _, ctx, _, _) = slicer
        .slice_chunk(
            tmp.path(),
            "UnitTest_PadTrim",
            &master_img,
            None,
            24.0,
            50,
            100,
            true,
            true,
        )
        .unwrap();
    assert_eq!(chunk_imgs.shape, vec![50, 32, 32, 3]);
    assert_eq!(ctx.valid_frames, 30);

    // Reassembler automatically trims the 20 padded frames back to 30 valid frames
    let edited = Tensor::full([50, 32, 32, 3], 0.9);
    let (assembled, _, _, _) = reassembler
        .reassemble_patch(tmp.path(), &ctx, &edited, None, 0, false, "")
        .unwrap();
    assert_eq!(assembled.shape, vec![80, 32, 32, 3]);
    assert!(assembled
        .slice_dim0(50, 80)
        .unwrap()
        .allclose(&Tensor::full([30, 32, 32, 3], 0.9), 0.02));
}

#[test]
fn test_gap_detection_and_incremental_patch_storage() {
    let tmp = tempfile::tempdir().unwrap();
    let mut session = TimelineSession::load_or_create(tmp.path(), "UnitTest_Gaps").unwrap();
    session.meta.total_frames = 100;
    session.meta.height = 32;
    session.meta.width = 32;
    session.save_meta().unwrap();

    assert_eq!(session.get_missing_gap_ranges(), vec![(0, 100)]);
    assert!(!session.is_complete());

    let p0 = Tensor::full([30, 32, 32, 3], 0.88);
    let pid0 = session
        .save_chunk_patch(&p0, None, 0, Some(30), 24.0, 0, "")
        .unwrap();
    assert_eq!(session.get_missing_gap_ranges(), vec![(30, 100)]);

    let (loaded_p0, _) = session.load_chunk_patch(&pid0).unwrap();
    assert!(loaded_p0
        .unwrap()
        .allclose(&Tensor::full([30, 32, 32, 3], 0.88), 0.02));
}
