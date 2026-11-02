//! CLI entrypoint for `minimax-prefixstream` (`comfyui-minimaxh3-prefixstream-rs`).

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use minimaxh3_prefixstream::engine::cache_manager::{
    latent_steps_to_pixel_frames, pixel_frames_to_latent_steps, snap_to_run_grid, VIDEO_RUN_GRID,
};
use minimaxh3_prefixstream::engine::clip_bin_api::{
    delete_clip_api, get_project_clips_api, rescan_project_api, update_clip_rating_api,
};
use minimaxh3_prefixstream::engine::clip_bin_manager::{
    get_base_bin_dir, load_tensor_safetensors, save_clip_asset, SaveClipRequest,
};
use minimaxh3_prefixstream::engine::rope_aligner::TemporalCursorTracker;
use minimaxh3_prefixstream::engine::texture_diagnostics::summarize_clip_texture_drift;
use minimaxh3_prefixstream::engine::timeline_session_manager::{
    get_timeline_sessions_base_dir, TimelineSession,
};
use minimaxh3_prefixstream::nodes::{
    node_display_name_mappings, validate_workflow_json, NODE_CLASS_NAMES,
};
use minimaxh3_prefixstream::pipeline::native_masked_av::{
    is_exact_masked_av_context, masked_av_tail_slice,
};
use minimaxh3_prefixstream::tensor::Tensor;

#[derive(Parser, Debug)]
#[command(
    name = "minimax-prefixstream",
    version,
    about = "Pure Rust CLI & Engine for ComfyUI-MiniMaxH3-PrefixStream (MiniMaxHailuo-03 Video/Audio Continuation)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Inspect the H3 (1,4,4,4,4) spatiotemporal grid and snap a frame count
    Grid {
        /// Requested frame count to snap to the 17k + 5 run grid
        #[arg(short, long, default_value_t = 39)]
        frames: usize,
        /// Rounding preference: nearest, lower, higher
        #[arg(short, long, default_value = "nearest")]
        prefer: String,
    },
    /// Plan a Native Masked AV tail slice and check 24fps / 40Hz exactness
    MaskedAv {
        /// Source video latent temporal length (T)
        #[arg(long, default_value_t = 37)]
        source_video_t: usize,
        /// Requested rolling context frames (e.g. 22, 39, 90)
        #[arg(long, default_value_t = 39)]
        context_frames: usize,
        /// Target video latent temporal length (T)
        #[arg(long, default_value_t = 37)]
        target_video_t: usize,
        /// Source start step phase offset
        #[arg(long, default_value_t = 0)]
        source_start_step: usize,
    },
    /// Simulate multi-clip 3D-RoPE temporal cursor progression
    RopePlan {
        /// Number of clips to simulate
        #[arg(short, long, default_value_t = 4)]
        clips: usize,
        /// Latent temporal steps per clip
        #[arg(short, long, default_value_t = 32)]
        latent_steps: usize,
    },
    /// Analyze spatial texture RMS across wavelength bands (2-8px, 8-32px, 32-128px)
    TextureDrift {
        /// Optional path to a .safetensors file containing an `images` [F, H, W, C] tensor
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Window size for start/end comparison
        #[arg(short, long, default_value_t = 4)]
        window: usize,
    },
    /// Manage the DAG Clip Bin for a project
    ClipBin {
        /// Project name
        #[arg(short, long, default_value = "Default_Project")]
        project: String,
        /// Action: list, save-demo, rate, delete, rescan
        #[arg(short, long, default_value = "list")]
        action: String,
        /// Optional clip ID for rate/delete/parent
        #[arg(short, long)]
        clip_id: Option<String>,
        /// Rating (-1 to 5) when action=rate
        #[arg(short, long, default_value_t = 5)]
        rating: i32,
    },
    /// Inspect or run a demo non-destructive timeline session
    Timeline {
        /// Session ID
        #[arg(short, long, default_value = "Default_Timeline_Session")]
        session_id: String,
        /// Run a synthetic 2-patch reassembly test
        #[arg(long)]
        demo: bool,
    },
    /// Validate a ComfyUI workflow JSON file
    Workflow {
        /// Path to workflow JSON file
        #[arg(short, long)]
        file: PathBuf,
    },
    /// List all 15 registered MiniMaxH3 PrefixStream ComfyUI nodes
    Nodes,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Grid { frames, prefer } => {
            let snapped = snap_to_run_grid(frames, &prefer);
            let steps = pixel_frames_to_latent_steps(snapped);
            let roundtrip = latent_steps_to_pixel_frames(steps);
            let exact_av = is_exact_masked_av_context(snapped);
            println!("Canonical H3 Run Grid (17k + 5): {:?}", VIDEO_RUN_GRID);
            println!("Requested Frames : {frames}");
            println!("Snapped Frames   : {snapped} (prefer={prefer})");
            println!("Latent Steps (T) : {steps}");
            println!("Roundtrip Frames : {roundtrip}");
            println!("Exact 40Hz Audio : {exact_av}");
        }
        Commands::MaskedAv {
            source_video_t,
            context_frames,
            target_video_t,
            source_start_step,
        } => {
            let plan = masked_av_tail_slice(
                source_video_t,
                context_frames,
                target_video_t,
                source_start_step,
            )?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
        Commands::RopePlan {
            clips,
            latent_steps,
        } => {
            let mut tracker = TemporalCursorTracker::new();
            for _ in 0..clips {
                let info = tracker.advance_clip(latent_steps, None);
                println!(
                    "Clip #{:02}: start_step={:3} -> next_step={:3} | span_frames={:3} | cumulative_frames={:4}",
                    info.clip_index,
                    info.start_step,
                    info.next_step,
                    info.span_frames,
                    info.cumulative_frames
                );
            }
        }
        Commands::TextureDrift { input, window } => {
            let images = if let Some(path) = input {
                load_tensor_safetensors(&path, "images")
                    .with_context(|| format!("Failed to load 'images' from {}", path.display()))?
            } else {
                // Generate a synthetic 8-frame [8, 32, 32, 3] checkerboard clip
                let f_cnt = 8usize;
                let h = 32usize;
                let w = 32usize;
                let mut data = Vec::with_capacity(f_cnt * h * w * 3);
                for f in 0..f_cnt {
                    let amp = 1.0 - 0.05 * (f as f32);
                    for y in 0..h {
                        for x in 0..w {
                            let pat = if ((x / 2) + (y / 2)) % 2 == 0 {
                                0.7 * amp
                            } else {
                                0.3 * amp
                            };
                            data.push(pat);
                            data.push(pat * 0.9);
                            data.push(pat * 1.05_f32.min(1.0));
                        }
                    }
                }
                Tensor::from_vec(vec![f_cnt, h, w, 3], data)?
            };
            let report = summarize_clip_texture_drift(&images, window)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::ClipBin {
            project,
            action,
            clip_id,
            rating,
        } => {
            let base_dir = get_base_bin_dir();
            match action.as_str() {
                "list" => {
                    let resp = get_project_clips_api(&base_dir, &project);
                    println!("{}", serde_json::to_string_pretty(&resp.body)?);
                }
                "save-demo" => {
                    let imgs = Tensor::full(vec![5, 32, 32, 3], 0.5);
                    let vlat = Tensor::zeros(vec![1, 16, 2, 4, 4]);
                    let meta = save_clip_asset(SaveClipRequest {
                        base_dir: &base_dir,
                        project_name: &project,
                        parent_id: clip_id.as_deref(),
                        images: Some(&imgs),
                        video_latent: Some(&vlat),
                        audio_latent: None,
                        audio: None,
                        prompt: "CLI demo clip",
                        seed: 42,
                        fps: 24.0,
                        save_tail_frames: 5,
                        source_video_path: None,
                        auto_encode_mp4: false,
                    })?;
                    println!("{}", serde_json::to_string_pretty(&meta)?);
                }
                "rate" => {
                    let id = clip_id.context("--clip-id is required for action=rate")?;
                    let resp = update_clip_rating_api(&base_dir, &project, &id, rating);
                    println!("{}", serde_json::to_string_pretty(&resp.body)?);
                }
                "delete" => {
                    let id = clip_id.context("--clip-id is required for action=delete")?;
                    let resp = delete_clip_api(&base_dir, &project, &id);
                    println!("{}", serde_json::to_string_pretty(&resp.body)?);
                }
                "rescan" => {
                    let resp = rescan_project_api(&base_dir, &project);
                    println!("{}", serde_json::to_string_pretty(&resp.body)?);
                }
                other => anyhow::bail!("Unknown clip-bin action: {other}"),
            }
        }
        Commands::Timeline { session_id, demo } => {
            let base_dir = get_timeline_sessions_base_dir();
            let mut session = TimelineSession::load_or_create(&base_dir, &session_id)?;
            if demo {
                let master = Tensor::full(vec![24, 16, 16, 3], 0.25);
                session.register_master_source(Some(&master), None, 24.0, true, None)?;
                let patch = Tensor::full(vec![8, 16, 16, 3], 0.75);
                session.save_chunk_patch(&patch, None, 8, Some(16), 24.0, 2, "demo patch")?;
                let (_, _, report) = session.reassemble_full_timeline(None, None, Some(2), true)?;
                println!("{report}");
            } else {
                println!("{}", serde_json::to_string_pretty(&session.meta)?);
                println!("Missing Gaps: {:?}", session.get_missing_gap_ranges());
            }
        }
        Commands::Workflow { file } => {
            let raw = fs::read_to_string(&file)
                .with_context(|| format!("Failed to read {}", file.display()))?;
            let report = validate_workflow_json(&raw)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::Nodes => {
            let display_map = node_display_name_mappings();
            for &cls in &NODE_CLASS_NAMES {
                let disp = display_map.get(cls).copied().unwrap_or(cls);
                println!("{cls:<34} -> {disp}");
            }
        }
    }

    Ok(())
}
