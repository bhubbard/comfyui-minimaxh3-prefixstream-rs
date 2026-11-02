//! Asymmetric Fused Cross/Self Attention for Prefix KV Caching.
//!
//! Computes scaled dot-product attention where Query is target-only (`N_q`),
//! while Key and Value contain `[Prefix + Target]` (`N_kv`).
//!
//! Rust port of `engine/fused_attention.py`.

use crate::error::{PrefixStreamError, Result};
use crate::tensor::Tensor;

/// Compute scaled dot-product attention where `q` has length `N_q` and `k`/`v` have length `N_kv >= N_q`.
///
/// Supports:
/// - 4D tensors `[B, H, N, D]`
/// - 3D tensors `[B, N, H * D]` (when `heads` is provided, or defaults to `1` head)
///
/// Mirrors `asymmetric_cached_attention` in `engine/fused_attention.py`.
pub fn asymmetric_cached_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    heads: Option<usize>,
) -> Result<Tensor> {
    if k.shape != v.shape {
        return Err(PrefixStreamError::InvalidInput(format!(
            "Key and Value shapes must match, got {:?} vs {:?}",
            k.shape, v.shape
        )));
    }

    match (q.ndim(), k.ndim()) {
        (4, 4) => {
            let b = q.shape[0];
            let h = q.shape[1];
            let n_q = q.shape[2];
            let d = q.shape[3];
            let n_kv = k.shape[2];
            if k.shape[0] != b || k.shape[1] != h || k.shape[3] != d {
                return Err(PrefixStreamError::InvalidInput(format!(
                    "Incompatible 4D Q {:?} and K {:?}",
                    q.shape, k.shape
                )));
            }
            let out_data = sdpa_4d(&q.data, &k.data, &v.data, attn_mask, b, h, n_q, n_kv, d);
            Tensor::from_vec(vec![b, h, n_q, d], out_data)
        }
        (3, 3) => {
            let b = q.shape[0];
            let n_q = q.shape[1];
            let c = q.shape[2];
            let n_kv = k.shape[1];
            let h = heads.unwrap_or(1).max(1);
            if k.shape[0] != b || k.shape[2] != c || !c.is_multiple_of(h) {
                return Err(PrefixStreamError::InvalidInput(format!(
                    "Incompatible 3D Q {:?}, K {:?}, heads={h}",
                    q.shape, k.shape
                )));
            }
            let d = c / h;
            let q_4d = transpose_bnhd_to_bhnd(&q.data, b, n_q, h, d);
            let k_4d = transpose_bnhd_to_bhnd(&k.data, b, n_kv, h, d);
            let v_4d = transpose_bnhd_to_bhnd(&v.data, b, n_kv, h, d);
            let out_4d = sdpa_4d(&q_4d, &k_4d, &v_4d, attn_mask, b, h, n_q, n_kv, d);
            let out_3d = transpose_bhnd_to_bnhd(&out_4d, b, h, n_q, d);
            Tensor::from_vec(vec![b, n_q, c], out_3d)
        }
        _ => Err(PrefixStreamError::InvalidInput(format!(
            "asymmetric_cached_attention expects both 3D or both 4D tensors, got Q {:?} and K {:?}",
            q.shape, k.shape
        ))),
    }
}

fn transpose_bnhd_to_bhnd(data: &[f32], b: usize, n: usize, h: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * h * n * d];
    for bi in 0..b {
        for ni in 0..n {
            for hi in 0..h {
                let src = ((bi * n + ni) * h + hi) * d;
                let dst = ((bi * h + hi) * n + ni) * d;
                out[dst..dst + d].copy_from_slice(&data[src..src + d]);
            }
        }
    }
    out
}

fn transpose_bhnd_to_bnhd(data: &[f32], b: usize, h: usize, n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * n * h * d];
    for bi in 0..b {
        for hi in 0..h {
            for ni in 0..n {
                let src = ((bi * h + hi) * n + ni) * d;
                let dst = ((bi * n + ni) * h + hi) * d;
                out[dst..dst + d].copy_from_slice(&data[src..src + d]);
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn sdpa_4d(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attn_mask: Option<&Tensor>,
    b: usize,
    h: usize,
    n_q: usize,
    n_kv: usize,
    d: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0_f32; b * h * n_q * d];

    for bi in 0..b {
        for hi in 0..h {
            let bh = bi * h + hi;
            let q_bh = bh * n_q * d;
            let kv_bh = bh * n_kv * d;

            for qi in 0..n_q {
                let q_row = &q[q_bh + qi * d..q_bh + (qi + 1) * d];
                let mut scores = vec![0.0_f32; n_kv];
                let mut max_s = f32::NEG_INFINITY;

                for ki in 0..n_kv {
                    let k_row = &k[kv_bh + ki * d..kv_bh + (ki + 1) * d];
                    let mut dot = 0.0_f32;
                    for di in 0..d {
                        dot += q_row[di] * k_row[di];
                    }
                    let mut sc = dot * scale;
                    if let Some(m) = attn_mask {
                        if !m.data.is_empty() {
                            let idx = (qi * n_kv + ki) % m.data.len();
                            sc += m.data[idx];
                        }
                    }
                    scores[ki] = sc;
                    if sc > max_s {
                        max_s = sc;
                    }
                }

                let mut sum_exp = 0.0_f32;
                for sc in &mut scores {
                    *sc = (*sc - max_s).exp();
                    sum_exp += *sc;
                }
                let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

                let out_row = &mut out[q_bh + qi * d..q_bh + (qi + 1) * d];
                for ki in 0..n_kv {
                    let w = scores[ki] * inv_sum;
                    let v_row = &v[kv_bh + ki * d..kv_bh + (ki + 1) * d];
                    for di in 0..d {
                        out_row[di] += w * v_row[di];
                    }
                }
            }
        }
    }
    out
}
