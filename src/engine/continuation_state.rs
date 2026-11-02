//! Continuation state packing and identity reference preparation.
//!
//! Rust port of `engine/continuation_state.py`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::Result;
use crate::tensor::{IdentityReference, Tensor};

/// Key used to store continuation state in a latent dictionary.
pub const STATE_KEY: &str = "minimax_continuation_state";
/// Key for the primary anchor image inside continuation state.
pub const ANCHOR_KEY: &str = "anchor_image";
/// Key for identity references inside continuation state.
pub const REFERENCES_KEY: &str = "identity_references";
/// Key for metadata manifest inside continuation state.
pub const MANIFEST_KEY: &str = "manifest";

/// Canonical continuation state keys serialized into `.safetensors` metadata or latent payloads.
pub const STATE_KEYS: [&str; 4] = [
    STATE_KEY,
    "minimax_anchor_image",
    "minimax_identity_references",
    "minimax_manifest",
];

/// Continuation state bundled with a saved or live latent payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContinuationState {
    /// Optional primary anchor image `[1, H, W, C]`.
    pub anchor_image: Option<Tensor>,
    /// Optional list of identity references.
    pub identity_references: Vec<IdentityReference>,
    /// Arbitrary JSON metadata manifest.
    pub manifest: HashMap<String, Value>,
}

/// Check if two tensors have identical shapes and element-wise values within `1e-6`.
pub fn same_image(a: &Tensor, b: &Tensor) -> bool {
    a.approx_eq(b, 1e-6)
}

/// Deduplicate and prepare identity reference entries alongside an optional primary anchor image.
///
/// Mirrors `prepare_identity_references` in `engine/continuation_state.py`.
pub fn prepare_identity_references(
    anchor_image: Option<&Tensor>,
    extra_references: Option<&[IdentityReference]>,
    anchor_strength: f32,
    extra_strength_scale: f32,
) -> (Option<Tensor>, Vec<IdentityReference>) {
    let mut prepared: Vec<IdentityReference> = Vec::new();
    let mut primary = anchor_image.map(|img| img.first_frame());

    if let Some(ref p) = primary {
        prepared.push(IdentityReference {
            image: p.clone(),
            strength: anchor_strength,
            source: "anchor_image".to_string(),
        });
    }

    if let Some(extras) = extra_references {
        for item in extras {
            let image = item.image.first_frame();
            let raw_strength = item.strength;
            let strength = if item.source == "anchor_image" {
                raw_strength.clamp(0.0, 1.0)
            } else {
                (raw_strength * extra_strength_scale).clamp(0.0, 1.0)
            };

            if primary.is_none() {
                primary = Some(image.clone());
            }

            if prepared
                .iter()
                .any(|existing| same_image(&existing.image, &image))
            {
                continue;
            }

            prepared.push(IdentityReference {
                image,
                strength,
                source: if item.source.is_empty() {
                    "reference".to_string()
                } else {
                    item.source.clone()
                },
            });
        }
    }

    (primary, prepared)
}

/// Encode a [`ContinuationState`] into flat named tensors and a JSON metadata manifest.
///
/// Mirrors `encode_state` in `engine/continuation_state.py`.
pub fn encode_state(
    state: &ContinuationState,
) -> (HashMap<String, Tensor>, HashMap<String, String>) {
    let mut tensors = HashMap::new();
    let mut metadata = HashMap::new();

    if let Some(ref anchor) = state.anchor_image {
        tensors.insert(
            "minimax_state__anchor_image".to_string(),
            anchor.first_frame(),
        );
    }

    let mut ref_meta = Vec::new();
    for (idx, item) in state.identity_references.iter().enumerate() {
        let tensor_key = format!("minimax_state__ref_{idx:02}");
        tensors.insert(tensor_key.clone(), item.image.first_frame());
        ref_meta.push(json!({
            "tensor_key": tensor_key,
            "strength": item.strength,
            "source": if item.source.is_empty() { "reference" } else { &item.source },
        }));
    }

    let manifest_payload = json!({
        "has_anchor": tensors.contains_key("minimax_state__anchor_image"),
        "references": ref_meta,
        "manifest": state.manifest,
    });
    if let Ok(serialized) = serde_json::to_string(&manifest_payload) {
        metadata.insert("minimax_continuation_manifest".to_string(), serialized);
    }

    (tensors, metadata)
}

/// Reconstruct a [`ContinuationState`] from flat named tensors and metadata.
///
/// Mirrors `decode_state` in `engine/continuation_state.py`.
pub fn decode_state(
    tensors: &HashMap<String, Tensor>,
    metadata: Option<&HashMap<String, String>>,
) -> Result<Option<ContinuationState>> {
    let mut manifest_payload: Value = json!({});
    if let Some(meta) = metadata {
        if let Some(raw_manifest) = meta.get("minimax_continuation_manifest") {
            if let Ok(parsed) = serde_json::from_str::<Value>(raw_manifest) {
                manifest_payload = parsed;
            }
        }
    }

    let anchor = tensors
        .get("minimax_state__anchor_image")
        .map(|t| t.first_frame());

    let mut references = Vec::new();
    if let Some(ref_items) = manifest_payload
        .get("references")
        .and_then(|v| v.as_array())
    {
        for item in ref_items {
            let tensor_key = item
                .get("tensor_key")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            if let Some(img) = tensors.get(tensor_key) {
                let strength = item.get("strength").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                let source = item
                    .get("source")
                    .and_then(|v| v.as_str())
                    .unwrap_or("reference")
                    .to_string();
                references.push(IdentityReference {
                    image: img.first_frame(),
                    strength,
                    source,
                });
            }
        }
    }

    let mut manifest_map = HashMap::new();
    if let Some(obj) = manifest_payload.get("manifest").and_then(|v| v.as_object()) {
        for (k, v) in obj {
            manifest_map.insert(k.clone(), v.clone());
        }
    }

    if anchor.is_none() && references.is_empty() && manifest_map.is_empty() {
        return Ok(None);
    }

    Ok(Some(ContinuationState {
        anchor_image: anchor,
        identity_references: references,
        manifest: manifest_map,
    }))
}
