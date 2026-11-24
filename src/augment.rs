//! Physically Valid Acoustic Data Augmentation
//!
//! This module implements 4 train-only physically valid data augmentations:
//! - Onset jitter (±40ms temporal shift)
//! - Background noise (Gaussian white noise, SNR 15-35 dB)
//! - Spectral tilt (±3 dB linear slope across Mel bands)
//! - SpecAugment (frequency & time masking)
//!
//! Pitch Shifting and Time Stretching are strictly prohibited as they distort
//! cavity resonance frequency and temporal decay slope, corrupting ripeness labels.

use ndarray::Array2;

use crate::dsp::{normalize_peak, ImpulseConfig};

/// Apply temporal onset jitter to a tap window.
///
/// Samples a random jitter offset delta in [-max_jitter_samples, +max_jitter_samples],
/// shifts the window extraction center by delta, and re-normalizes to peak amplitude 1.0.
/// Preserves exact window length (17,640 samples) and [-1.0, 1.0] bounds.
///
/// # Arguments
///
/// * `full_samples` - Full recording waveform containing the tap
/// * `center_sample` - Original onset center sample index
/// * `max_jitter_samples` - Maximum jitter in samples (880 samples ≈ ±40ms at 22,050 Hz)
/// * `cfg` - Impulse configuration for window parameters
///
/// # Returns
///
/// Jittered and peak-normalized tap window of exact length window_samples()
///
/// # Examples
///
/// ```
/// use durian_sonic::augment::apply_onset_jitter;
/// use durian_sonic::dsp::ImpulseConfig;
///
/// let cfg = ImpulseConfig::default();
/// let full_signal = vec![0.0; 50_000];
/// let jittered = apply_onset_jitter(&full_signal, 10_000, 880, &cfg);
/// assert_eq!(jittered.len(), 17_640);
/// ```
pub fn apply_onset_jitter(
    full_samples: &[f32],
    center_sample: usize,
    max_jitter_samples: usize,
    cfg: &ImpulseConfig,
) -> Vec<f32> {
    use rand::{rng, RngExt};

    let win_len = cfg.window_samples();
    let pre_len = cfg.pre_onset_samples();

    if win_len == 0 {
        return Vec::new();
    }

    let delta = if max_jitter_samples > 0 {
        let max_jitter = max_jitter_samples as i32;
        rng().random_range(-max_jitter..=max_jitter) as isize
    } else {
        0
    };

    let jittered_center = (center_sample as isize + delta).max(0) as usize;
    let raw_start = jittered_center.saturating_sub(pre_len);

    if full_samples.len() < win_len {
        let copy_start = raw_start.min(full_samples.len());
        let available = full_samples.len() - copy_start;
        let mut padded = vec![0.0; win_len];
        padded[..available].copy_from_slice(&full_samples[copy_start..]);
        normalize_peak(&padded)
    } else {
        let start = raw_start.min(full_samples.len() - win_len);
        normalize_peak(&full_samples[start..start + win_len])
    }
}

/// Add Gaussian white noise at a random SNR.
///
/// Generates Box-Muller Gaussian white noise N(0, sigma_noise^2) and adds it to the
/// input window, then re-normalizes to peak amplitude 1.0. SNR is uniformly sampled
/// from [min_snr_db, max_snr_db]. Preserves exact window length and [-1.0, 1.0] bounds.
///
/// # Arguments
///
/// * `window` - Input tap window (typically 17,640 samples)
/// * `min_snr_db` - Minimum SNR in dB (default 15.0)
/// * `max_snr_db` - Maximum SNR in dB (default 35.0)
///
/// # Returns
///
/// Noisy window with added Gaussian white noise, peak-normalized to [-1.0, 1.0]
///
/// # Examples
///
/// ```
/// use durian_sonic::augment::add_background_noise;
///
/// let window = vec![0.5; 17_640];
/// let noisy = add_background_noise(&window, 15.0, 35.0);
/// assert_eq!(noisy.len(), 17_640);
/// assert!(noisy.iter().all(|&x| x.is_finite() && x >= -1.0 && x <= 1.0));
/// ```
pub fn add_background_noise(window: &[f32], min_snr_db: f32, max_snr_db: f32) -> Vec<f32> {
    use rand::{rng, RngExt};

    if window.is_empty() {
        return Vec::new();
    }

    let (lo, hi) = (min_snr_db.min(max_snr_db), min_snr_db.max(max_snr_db));
    let snr_db = if (hi - lo).abs() < 1e-6 {
        lo
    } else {
        rng().random_range(lo..=hi)
    };

    let p_sig: f32 = window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32;

    if p_sig <= 1e-12 {
        return normalize_peak(window);
    }

    let sigma_noise = (p_sig * 10.0_f32.powf(-snr_db / 10.0)).sqrt();

    let mut noisy = Vec::with_capacity(window.len());
    for &sample in window {
        let u1: f32 = rng().random_range(1e-7..1.0);
        let u2: f32 = rng().random_range(0.0..(2.0 * std::f32::consts::PI));
        let z = (-2.0 * u1.ln()).sqrt() * u2.cos();
        noisy.push(sample + sigma_noise * z);
    }

    normalize_peak(&noisy)
}

/// Apply linear spectral tilt across Mel frequency bands.
///
/// Samples a random tilt_db in [-max_tilt_db, +max_tilt_db] and adds a linear
/// dB offset across Mel rows: Delta_m = (tilt_db / top_db) * (2m / (M-1) - 1).
/// All values are clamped to [0.0, 1.0] after modification. Operates in-place.
///
/// # Arguments
///
/// * `mel_spec` - Log-Mel spectrogram (128, 128) with values in [0.0, 1.0]
/// * `max_tilt_db` - Maximum tilt magnitude in dB (default 3.0)
/// * `top_db` - Dynamic range in dB (default 80.0)
///
/// # Examples
///
/// ```
/// use durian_sonic::augment::apply_spectral_tilt;
/// use ndarray::Array2;
///
/// let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
/// apply_spectral_tilt(&mut mel_spec, 3.0, 80.0);
/// assert!(mel_spec.iter().all(|&x| x >= 0.0 && x <= 1.0));
/// ```
pub fn apply_spectral_tilt(mel_spec: &mut Array2<f32>, max_tilt_db: f32, top_db: f32) {
    use rand::{rng, RngExt};

    let (n_mels, _n_frames) = mel_spec.dim();

    if max_tilt_db == 0.0 || !max_tilt_db.is_finite() || top_db <= 0.0 || n_mels <= 1 {
        for val in mel_spec.iter_mut() {
            *val = val.clamp(0.0, 1.0);
        }
        return;
    }

    let bound = max_tilt_db.abs();
    let tilt_db = rng().random_range(-bound..=bound);

    for m in 0..n_mels {
        let delta = (tilt_db / top_db) * (2.0 * m as f32 / (n_mels - 1) as f32 - 1.0);
        for t in 0..mel_spec.ncols() {
            mel_spec[[m, t]] = (mel_spec[[m, t]] + delta).clamp(0.0, 1.0);
        }
    }
}

/// Apply SpecAugment frequency and time masking.
///
/// Masks a random contiguous frequency band (up to max_freq_mask or n_mels/4)
/// and a random contiguous time band (up to max_time_mask or n_frames/4) by
/// setting masked elements to noise_floor_val (0.0 = -80 dB floor). Operates in-place.
///
/// # Arguments
///
/// * `mel_spec` - Log-Mel spectrogram (128, 128) with values in [0.0, 1.0]
/// * `max_freq_mask` - Maximum frequency mask width (default 12)
/// * `max_time_mask` - Maximum time mask width (default 16)
/// * `noise_floor_val` - Replacement value for masked regions (default 0.0)
///
/// # Examples
///
/// ```
/// use durian_sonic::augment::apply_spec_augment;
/// use ndarray::Array2;
///
/// let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
/// apply_spec_augment(&mut mel_spec, 12, 16, 0.0);
/// assert!(mel_spec.iter().all(|&x| x >= 0.0 && x <= 1.0));
/// ```
pub fn apply_spec_augment(
    mel_spec: &mut Array2<f32>,
    max_freq_mask: usize,
    max_time_mask: usize,
    noise_floor_val: f32,
) {
    use rand::{rng, RngExt};

    let (n_mels, n_frames) = mel_spec.dim();

    // Frequency masking
    let f_bound = max_freq_mask.min(n_mels / 4);
    if f_bound > 0 {
        let f_mask = rng().random_range(0..=f_bound);
        if f_mask > 0 && f_mask < n_mels {
            let f0 = rng().random_range(0..(n_mels - f_mask));
            for f in f0..(f0 + f_mask) {
                for t in 0..n_frames {
                    mel_spec[[f, t]] = noise_floor_val;
                }
            }
        }
    }

    // Time masking
    let t_bound = max_time_mask.min(n_frames / 4);
    if t_bound > 0 {
        let t_mask = rng().random_range(0..=t_bound);
        if t_mask > 0 && t_mask < n_frames {
            let t0 = rng().random_range(0..(n_frames - t_mask));
            for f in 0..n_mels {
                for t in t0..(t0 + t_mask) {
                    mel_spec[[f, t]] = noise_floor_val;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    // T019: Tests for waveform augmentations (apply_onset_jitter, add_background_noise)

    #[test]
    fn test_onset_jitter_preserves_length() {
        let cfg = ImpulseConfig::default();
        let win_len = cfg.window_samples();
        let full_samples = vec![0.5; 50_000];

        let jittered = apply_onset_jitter(&full_samples, 10_000, 880, &cfg);
        assert_eq!(jittered.len(), win_len);
    }

    #[test]
    fn test_onset_jitter_preserves_bounds() {
        let cfg = ImpulseConfig::default();
        let full_samples = vec![0.3; 50_000];

        let jittered = apply_onset_jitter(&full_samples, 10_000, 880, &cfg);
        assert!(jittered
            .iter()
            .all(|&x| x.is_finite() && (-1.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_onset_jitter_near_start() {
        let cfg = ImpulseConfig::default();
        let win_len = cfg.window_samples();
        let full_samples = vec![0.5; 50_000];

        let jittered = apply_onset_jitter(&full_samples, 1_000, 880, &cfg);
        assert_eq!(jittered.len(), win_len);
        assert!(jittered.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_onset_jitter_near_end() {
        let cfg = ImpulseConfig::default();
        let win_len = cfg.window_samples();
        let full_samples = vec![0.5; 30_000];

        let jittered = apply_onset_jitter(&full_samples, 25_000, 880, &cfg);
        assert_eq!(jittered.len(), win_len);
        assert!(jittered.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_onset_jitter_short_input_zero_pads() {
        let cfg = ImpulseConfig::default();
        let win_len = cfg.window_samples();
        let full_samples = vec![0.5; 10_000];

        let jittered = apply_onset_jitter(&full_samples, 5_000, 880, &cfg);
        assert_eq!(jittered.len(), win_len);
        assert!(jittered.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_onset_jitter_zero_jitter() {
        let cfg = ImpulseConfig::default();
        let win_len = cfg.window_samples();
        let full_samples = vec![0.5; 50_000];

        let jittered = apply_onset_jitter(&full_samples, 10_000, 0, &cfg);
        assert_eq!(jittered.len(), win_len);
        assert!(jittered.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_onset_jitter_peak_normalized() {
        let cfg = ImpulseConfig::default();
        let mut full_samples = vec![0.3; 50_000];
        full_samples[10_000] = 0.5;

        let jittered = apply_onset_jitter(&full_samples, 10_000, 880, &cfg);
        let peak = jittered.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert_relative_eq!(peak, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_add_background_noise_preserves_length() {
        let window = vec![0.5; 17_640];
        let noisy = add_background_noise(&window, 15.0, 35.0);
        assert_eq!(noisy.len(), 17_640);
    }

    #[test]
    fn test_add_background_noise_preserves_bounds() {
        let window = vec![0.3; 17_640];
        let noisy = add_background_noise(&window, 15.0, 35.0);
        assert!(noisy
            .iter()
            .all(|&x| x.is_finite() && (-1.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_add_background_noise_inverted_snr_range() {
        let window = vec![0.5; 17_640];
        let noisy = add_background_noise(&window, 35.0, 15.0);
        assert_eq!(noisy.len(), 17_640);
        assert!(noisy.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_add_background_noise_equal_snr() {
        let window = vec![0.5; 17_640];
        let noisy = add_background_noise(&window, 25.0, 25.0);
        assert_eq!(noisy.len(), 17_640);
        assert!(noisy.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_add_background_noise_silent_input() {
        let window = vec![0.0; 17_640];
        let noisy = add_background_noise(&window, 15.0, 35.0);
        assert_eq!(noisy.len(), 17_640);
        assert!(noisy.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_add_background_noise_empty_input() {
        let window: Vec<f32> = vec![];
        let noisy = add_background_noise(&window, 15.0, 35.0);
        assert_eq!(noisy.len(), 0);
    }

    #[test]
    fn test_add_background_noise_preserves_frequency() {
        use crate::dsp::{extract_physics_features, SpectrogramConfig};

        let sr = 22_050;
        let freq = 600.0;
        let n = 17_640;
        let mut window = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let decay = (-t * 2.0).exp();
            window.push((2.0 * std::f32::consts::PI * freq * t).sin() * decay * 0.5);
        }

        let cfg = SpectrogramConfig::default();
        let features_orig = extract_physics_features(&window, &cfg);
        let f_peak_orig = features_orig[0];

        let noisy = add_background_noise(&window, 20.0, 30.0);
        let features_noisy = extract_physics_features(&noisy, &cfg);
        let f_peak_noisy = features_noisy[0];

        assert_relative_eq!(f_peak_orig, f_peak_noisy, epsilon = 0.1);
    }

    // T020: Tests for spectrogram augmentations (apply_spectral_tilt, apply_spec_augment)

    #[test]
    fn test_spectral_tilt_preserves_shape() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        apply_spectral_tilt(&mut mel_spec, 3.0, 80.0);
        assert_eq!(mel_spec.dim(), (128, 128));
    }

    #[test]
    fn test_spectral_tilt_preserves_bounds() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        apply_spectral_tilt(&mut mel_spec, 3.0, 80.0);
        assert!(mel_spec.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_spectral_tilt_zero_tilt() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        let original = mel_spec.clone();
        apply_spectral_tilt(&mut mel_spec, 0.0, 80.0);
        assert_eq!(mel_spec, original);
    }

    #[test]
    fn test_spectral_tilt_negative_tilt() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        apply_spectral_tilt(&mut mel_spec, -3.0, 80.0);
        assert!(mel_spec.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_spectral_tilt_non_finite_tilt() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        apply_spectral_tilt(&mut mel_spec, f32::NAN, 80.0);
        assert!(mel_spec.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_spectral_tilt_zero_top_db() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        apply_spectral_tilt(&mut mel_spec, 3.0, 0.0);
        assert!(mel_spec.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_spectral_tilt_negative_top_db() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        apply_spectral_tilt(&mut mel_spec, 3.0, -80.0);
        assert!(mel_spec.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_spectral_tilt_single_row() {
        let mut mel_spec = Array2::<f32>::from_elem((1, 128), 0.5);
        apply_spectral_tilt(&mut mel_spec, 3.0, 80.0);
        assert!(mel_spec.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_spec_augment_preserves_shape() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        apply_spec_augment(&mut mel_spec, 12, 16, 0.0);
        assert_eq!(mel_spec.dim(), (128, 128));
    }

    #[test]
    fn test_spec_augment_preserves_bounds() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        apply_spec_augment(&mut mel_spec, 12, 16, 0.0);
        assert!(mel_spec.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_spec_augment_contains_floor_value() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        apply_spec_augment(&mut mel_spec, 12, 16, 0.0);
        let has_floor = mel_spec.iter().any(|&x| x == 0.0);
        assert!(has_floor || mel_spec.iter().all(|&x| x == 0.5));
    }

    #[test]
    fn test_spec_augment_zero_freq_mask() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        apply_spec_augment(&mut mel_spec, 0, 16, 0.0);
        assert_eq!(mel_spec.dim(), (128, 128));
    }

    #[test]
    fn test_spec_augment_zero_time_mask() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        apply_spec_augment(&mut mel_spec, 12, 0, 0.0);
        assert_eq!(mel_spec.dim(), (128, 128));
    }

    #[test]
    fn test_spec_augment_zero_both_masks() {
        let mut mel_spec = Array2::<f32>::from_elem((128, 128), 0.5);
        let original = mel_spec.clone();
        apply_spec_augment(&mut mel_spec, 0, 0, 0.0);
        assert_eq!(mel_spec, original);
    }

    #[test]
    fn test_spec_augment_small_matrix() {
        let mut mel_spec = Array2::<f32>::from_elem((3, 3), 0.5);
        apply_spec_augment(&mut mel_spec, 12, 16, 0.0);
        assert!(mel_spec.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }
}
