//! Multi-dimensional tensor primitives for MiniMax H3 video latents, audio latents,
//! pixel image frames, and audio waveforms.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{PrefixStreamError, Result};

/// Generic N-dimensional contiguous `f32` tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Tensor {
    /// Dimension sizes (row-major).
    pub shape: Vec<usize>,
    /// Flat row-major `f32` elements.
    pub data: Vec<f32>,
}

impl Tensor {
    /// Creates a new tensor of the given shape filled with `fill_val`.
    pub fn full<S: AsRef<[usize]>>(shape: S, fill_val: f32) -> Self {
        let s = shape.as_ref().to_vec();
        let numel: usize = s.iter().product();
        Self {
            shape: s,
            data: vec![fill_val; numel],
        }
    }

    /// Creates a new tensor of the given shape filled with zeros.
    pub fn zeros<S: AsRef<[usize]>>(shape: S) -> Self {
        Self::full(shape, 0.0)
    }

    /// Creates a new tensor of the given shape filled with ones.
    pub fn ones<S: AsRef<[usize]>>(shape: S) -> Self {
        Self::full(shape, 1.0)
    }

    /// Creates a new tensor of the same shape filled with zeros.
    pub fn zeros_like(&self) -> Self {
        Self::zeros(&self.shape)
    }

    /// Creates a new tensor of the same shape filled with ones.
    pub fn ones_like(&self) -> Self {
        Self::ones(&self.shape)
    }

    /// Creates a tensor from `shape` and `data`, validating element count.
    pub fn from_vec(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if numel != data.len() {
            return Err(PrefixStreamError::InvalidInput(format!(
                "Shape {:?} requires {} elements, got {}",
                shape,
                numel,
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Creates a deterministic standard normal (`N(0, 1)`) tensor using Box-Muller.
    pub fn randn<S: AsRef<[usize]>>(shape: S, seed: u64) -> Self {
        let s = shape.as_ref().to_vec();
        let numel: usize = s.iter().product();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            let u1: f32 = rng.gen::<f32>().clamp(1e-7, 1.0);
            let u2: f32 = rng.gen::<f32>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            data.push(z);
        }
        Self { shape: s, data }
    }

    /// Creates a deterministic uniform `[0.0, 1.0)` tensor.
    pub fn rand<S: AsRef<[usize]>>(shape: S, seed: u64) -> Self {
        let s = shape.as_ref().to_vec();
        let numel: usize = s.iter().product();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            data.push(rng.gen::<f32>());
        }
        Self { shape: s, data }
    }

    /// Arithmetic mean of all elements (or `0.0` if empty).
    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            0.0
        } else {
            self.data.iter().sum::<f32>() / (self.data.len() as f32)
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of scalar elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the tensor has 0 elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the first slice `[0..1]` along dimension 0 (or clone if empty).
    pub fn first_frame(&self) -> Self {
        if self.ndim() == 0 || self.shape[0] <= 1 {
            return self.clone();
        }
        self.slice_axis(0, 0, 1).unwrap_or_else(|_| self.clone())
    }

    /// Returns the last slice `[N-1..N]` along dimension 0 (or clone if empty).
    pub fn last_frame(&self) -> Self {
        if self.ndim() == 0 || self.shape[0] <= 1 {
            return self.clone();
        }
        let n = self.shape[0];
        self.slice_axis(0, n - 1, n)
            .unwrap_or_else(|_| self.clone())
    }

    /// Slice along dimension 0 `[start..end]`.
    pub fn slice_dim0(&self, start: usize, end: usize) -> Result<Self> {
        self.slice_axis(0, start, end)
    }

    /// Slice along a specific axis `[start..end]`.
    pub fn slice_axis(&self, dim: usize, start: usize, end: usize) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(PrefixStreamError::IndexError(format!(
                "dim {} out of bounds for ndim {}",
                dim,
                self.ndim()
            )));
        }
        let dim_len = self.shape[dim];
        let s = start.min(dim_len);
        let e = end.min(dim_len).max(s);
        let new_dim_len = e - s;

        let mut new_shape = self.shape.clone();
        new_shape[dim] = new_dim_len;
        if new_dim_len == 0 || self.numel() == 0 {
            return Ok(Self::zeros(&new_shape));
        }

        let outer_size: usize = self.shape[..dim].iter().product();
        let inner_size: usize = self.shape[dim + 1..].iter().product();
        let old_stride = dim_len * inner_size;
        let copy_len = new_dim_len * inner_size;

        let mut new_data = Vec::with_capacity(outer_size * copy_len);
        for o in 0..outer_size {
            let src_offset = o * old_stride + s * inner_size;
            new_data.extend_from_slice(&self.data[src_offset..src_offset + copy_len]);
        }

        Ok(Self {
            shape: new_shape,
            data: new_data,
        })
    }

    /// Alias for [`Self::slice_axis`].
    pub fn slice_dim(&self, dim: usize, start: usize, end: usize) -> Result<Self> {
        self.slice_axis(dim, start, end)
    }

    /// Return a new tensor with every element multiplied by `factor`.
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&v| v * factor).collect(),
        }
    }

    /// Concatenate multiple tensors along `dim`.
    pub fn concat(tensors: &[&Tensor], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Ok(Self::zeros([0]));
        }
        let first = tensors[0];
        let ndim = first.ndim();
        if dim >= ndim {
            return Err(PrefixStreamError::IndexError(
                "concat dim out of range".into(),
            ));
        }
        let mut total_dim = 0usize;
        for t in tensors {
            if t.ndim() != ndim {
                return Err(PrefixStreamError::InvalidInput(
                    "concat rank mismatch".into(),
                ));
            }
            for d in 0..ndim {
                if d != dim && t.shape[d] != first.shape[d] {
                    return Err(PrefixStreamError::InvalidInput(format!(
                        "concat shape mismatch at dim {}: {:?} vs {:?}",
                        d, first.shape, t.shape
                    )));
                }
            }
            total_dim += t.shape[dim];
        }

        let mut out_shape = first.shape.clone();
        out_shape[dim] = total_dim;
        if total_dim == 0 {
            return Ok(Self::zeros(&out_shape));
        }

        let outer_size: usize = first.shape[..dim].iter().product();
        let inner_size: usize = first.shape[dim + 1..].iter().product();
        let mut out_data = Vec::with_capacity(outer_size * total_dim * inner_size);

        for o in 0..outer_size {
            for t in tensors {
                let d_len = t.shape[dim];
                if d_len > 0 {
                    let chunk = d_len * inner_size;
                    let start = o * chunk;
                    out_data.extend_from_slice(&t.data[start..start + chunk]);
                }
            }
        }

        Ok(Self {
            shape: out_shape,
            data: out_data,
        })
    }

    /// Alias for [`Self::concat`].
    pub fn cat(tensors: &[&Tensor], dim: usize) -> Result<Self> {
        Self::concat(tensors, dim)
    }

    /// Checks approximate equality within `atol`.
    pub fn approx_eq(&self, other: &Self, atol: f32) -> bool {
        if self.shape != other.shape {
            return false;
        }
        self.data
            .iter()
            .zip(&other.data)
            .all(|(&a, &b)| (a - b).abs() <= atol)
    }

    /// Alias for [`Self::approx_eq`].
    pub fn allclose(&self, other: &Self, atol: f32) -> bool {
        self.approx_eq(other, atol)
    }
}

/// Standard ComfyUI `AUDIO` dictionary representation: `{"waveform": Tensor [1, C, N], "sample_rate": u32}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioData {
    /// Waveform tensor `[1, C, S]`.
    pub waveform: Tensor,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl AudioData {
    /// Construct a new [`AudioData`].
    pub fn new(waveform: Tensor, sample_rate: u32) -> Self {
        Self {
            waveform,
            sample_rate,
        }
    }
}

/// Identity reference entry carried in continuation state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityReference {
    /// Single-frame image tensor `[1, H, W, C]`.
    pub image: Tensor,
    /// Conditioning strength in `[0.0, 1.0]`.
    pub strength: f32,
    /// Reference source tag (e.g. `"anchor_image"`, `"reference"`).
    pub source: String,
}

/// Keyframe conditioning entry in `minimax_keyframes`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinimaxKeyframe {
    /// Resolved pixel frame coordinate (`0.0`, `1.0`, `5.0`, ...).
    pub resolved_frame_index: f32,
    /// Optional 1-step video latent slice `[1, C, 1, H, W]`.
    pub latent: Option<Tensor>,
    /// Optional audio latent tail `[1, C_a, 2, T_a]`.
    pub audio_latent: Option<Tensor>,
}

/// Conditioning entry `[embedding, extra_dict]` used in ComfyUI conditioning lists.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConditioningEntry {
    /// Text / prompt embedding tensor.
    pub embedding: Tensor,
    /// Injected MiniMax H3 keyframes.
    pub minimax_keyframes: Vec<MinimaxKeyframe>,
    /// Additional conditioning metadata.
    pub extra: HashMap<String, Value>,
}

impl ConditioningEntry {
    /// Create a new [`ConditioningEntry`] with the given embedding.
    pub fn new(embedding: Tensor) -> Self {
        Self {
            embedding,
            minimax_keyframes: Vec::new(),
            extra: HashMap::new(),
        }
    }
}

/// Unified MiniMax H3 AV Latent payload.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AvLatent {
    /// Optional 5D video latent `[B, C, T, H, W]`.
    pub video: Option<Tensor>,
    /// Optional 4D audio latent `[B, 32, 2, T_a]`.
    pub audio: Option<Tensor>,
    /// Optional 5D video denoise mask `[1, 1, T, H, W]`.
    pub video_mask: Option<Tensor>,
    /// Optional 4D audio denoise mask `[1, 1, 2, T_a]`.
    pub audio_mask: Option<Tensor>,
    /// Arbitrary metadata dictionary.
    pub metadata: HashMap<String, Value>,
}

impl AvLatent {
    /// Create an [`AvLatent`] containing only a video latent.
    pub fn from_video(video: Tensor) -> Self {
        Self {
            video: Some(video),
            audio: None,
            video_mask: None,
            audio_mask: None,
            metadata: HashMap::new(),
        }
    }

    /// Create an [`AvLatent`] containing both video and audio latents.
    pub fn from_video_audio(video: Tensor, audio: Tensor) -> Self {
        Self {
            video: Some(video),
            audio: Some(audio),
            video_mask: None,
            audio_mask: None,
            metadata: HashMap::new(),
        }
    }
}

/// Trait for Video VAE encode/decode operations.
pub trait VideoVae {
    /// Encode `[F, H, W, 3]` RGB pixels in `[0, 1]` to a 5D video latent `[1, C, T, H_l, W_l]`.
    fn encode(&self, images: &Tensor) -> Result<Tensor>;
    /// Decode a 5D video latent `[1, C, T, H_l, W_l]` to `[F, H, W, 3]` RGB pixels.
    fn decode(&self, latent: &Tensor) -> Result<Tensor>;
}

/// Trait for Audio VAE decode operations.
pub trait AudioVae {
    /// Decode a 4D audio latent `[1, C_a, 2, T_a]` to [`AudioData`].
    fn decode(&self, latent: &Tensor) -> Result<AudioData>;
}
