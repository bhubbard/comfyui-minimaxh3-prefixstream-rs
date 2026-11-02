//! Rust port of `tests/test_texture_diagnostics.py`.

use std::f32::consts::PI;

use minimaxh3_prefixstream::engine::texture_diagnostics::texture_band_rms;
use minimaxh3_prefixstream::tensor::Tensor;

#[test]
fn test_removes_flat_color_and_linear_lighting() {
    let h = 64usize;
    let w = 64usize;
    let mut data = vec![0.0_f32; h * w * 3];
    for r in 0..h {
        let y = (r as f32) / ((h - 1) as f32) - 0.5;
        for c in 0..w {
            let x = (c as f32) / ((w - 1) as f32) - 0.5;
            let idx = (r * w + c) * 3;
            data[idx] = 0.4 + 0.2 * x - 0.1 * y;
            data[idx + 1] = 0.4 + 0.2 * x - 0.1 * y + 0.05 * x;
            data[idx + 2] = 0.4 + 0.2 * x - 0.1 * y - 0.03 * y;
        }
    }
    let image = Tensor::from_vec(vec![h, w, 3], data).unwrap();
    let result = texture_band_rms(&image, None).unwrap();
    for bands in result.values() {
        for values in bands.values() {
            assert!(
                values[0] < 1e-6,
                "Expected linear trend to be removed (< 1e-6), got {}",
                values[0]
            );
        }
    }
}

#[test]
fn test_detects_mid_band_pattern_and_scales_with_amplitude() {
    let h = 64usize;
    let w = 64usize;
    let mut low_data = vec![0.5_f32; h * w * 3];
    let mut high_data = vec![0.5_f32; h * w * 3];
    for r in 0..h {
        for c in 0..w {
            // Wavelength 16 px falls in 8_32px band
            let pattern = (2.0 * PI * (c as f32) / 16.0).sin();
            let idx = (r * w + c) * 3;
            for ch in 0..3 {
                low_data[idx + ch] += 0.02 * pattern;
                high_data[idx + ch] += 0.06 * pattern;
            }
        }
    }
    let low_img = Tensor::from_vec(vec![h, w, 3], low_data).unwrap();
    let high_img = Tensor::from_vec(vec![h, w, 3], high_data).unwrap();
    let low = texture_band_rms(&low_img, None).unwrap();
    let high = texture_band_rms(&high_img, None).unwrap();

    let low_8_32 = low["luma"]["8_32px"][0];
    let low_2_8 = low["luma"]["2_8px"][0];
    let high_8_32 = high["luma"]["8_32px"][0];

    assert!(
        low_8_32 > low_2_8 * 10.0,
        "Expected 8_32px ({}) > 10x 2_8px ({})",
        low_8_32,
        low_2_8
    );
    let ratio = high_8_32 / low_8_32;
    assert!(
        (ratio - 3.0).abs() < 1e-4,
        "Expected 3x amplitude scaling ratio, got {}",
        ratio
    );
}

#[test]
fn test_measures_chroma_pattern_separately() {
    let h = 64usize;
    let w = 64usize;
    let mut data = vec![0.5_f32; h * w * 3];
    for r in 0..h {
        for c in 0..w {
            let pattern = 0.04 * (2.0 * PI * (r as f32) / 16.0).cos();
            let idx = (r * w + c) * 3;
            // Cancel Rec.709 luma (0.2126 R + 0.7152 G + 0.0722 B == 0)
            data[idx] += pattern;
            data[idx + 1] -= pattern * (0.2126 / 0.7152);
        }
    }
    let image = Tensor::from_vec(vec![h, w, 3], data).unwrap();
    let result = texture_band_rms(&image, None).unwrap();
    let rg_8_32 = result["red_minus_green"]["8_32px"][0];
    let luma_8_32 = result["luma"]["8_32px"][0];
    assert!(
        rg_8_32 > luma_8_32 * 20.0,
        "Expected red_minus_green 8_32px ({}) > 20x luma 8_32px ({})",
        rg_8_32,
        luma_8_32
    );
}

#[test]
fn test_non_finite_image_is_explicit() {
    let mut data = vec![0.0_f32; 8 * 8 * 3];
    data[0] = f32::NAN;
    let image = Tensor::from_vec(vec![8, 8, 3], data).unwrap();
    let err = texture_band_rms(&image, None).unwrap_err();
    assert!(
        err.to_string().contains("non-finite"),
        "Expected non-finite error message, got {}",
        err
    );
}
