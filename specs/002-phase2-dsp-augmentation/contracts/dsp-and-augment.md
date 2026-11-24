# Phase 2 Interface Contracts: Rust DSP & Augmentation Library API

**Feature**: `002-phase2-dsp-augmentation`
**Date**: 2026-11-24

---

## 1. Crate Root Exports (`src/lib.rs`)

Phase 2 registers two new public modules in `src/lib.rs` (`FR-010`):

```rust
pub mod augment;
pub mod config;
pub mod dataset;
pub mod dsp;
pub mod training;
```

---

## 2. Module `durian_sonic::dsp` (`src/dsp.rs`)

### 2.1. Configuration & Feature Structs

```rust
use crate::config::DspConfig;
use ndarray::Array2;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct ImpulseConfig {
    pub target_sr: u32,
    pub window_sec: f32,
    pub pre_onset_sec: f32,
    pub min_interval_sec: f32,
    pub threshold_ratio: f32,
}

impl Default for ImpulseConfig { /* target_sr: 22_050, window_sec: 0.8, pre_onset_sec: 0.1, min_interval_sec: 0.4, threshold_ratio: 0.25 */ }
impl From<&DspConfig> for ImpulseConfig { /* maps from DspConfig */ }

#[derive(Debug, Clone, PartialEq)]
pub struct SpectrogramConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub top_db: f32,
}

impl Default for SpectrogramConfig { /* sample_rate: 22_050, n_fft: 1024, hop_length: 128, n_mels: 128, fmin: 50.0, fmax: 8_000.0, top_db: 80.0 */ }
impl From<&DspConfig> for SpectrogramConfig { /* maps from DspConfig */ }

#[derive(Debug, Clone, PartialEq)]
pub struct TapFeatures {
    pub mel_spec: Array2<f32>,
    pub physics_features: [f32; 4],
}
```

### 2.2. Audio Decoding, Downmixing & Resampling Functions (`FR-001`, `FR-002`)

```rust
/// Decodes an audio file (.wav PCM or .m4a AAC-LC), downmixes all channels to Mono
/// via arithmetic mean, and resamples to `target_sr` (22,050 Hz).
///
/// Errors if:
/// - `target_sr == 0`
/// - `path` does not exist, cannot be opened, or has an unsupported/corrupted container
/// - Any audio packet fails to decode
/// - The decoded audio stream contains 0 samples
pub fn decode_audio_to_mono(path: &Path, target_sr: u32) -> anyhow::Result<Vec<f32>>;

/// Resamples a single-channel (Mono) f32 slice from `src_sr` to `target_sr` using
/// Rubato's synchronous FFT resampler and truncates/pads to exact rounded length
/// `round(samples.len() * target_sr / src_sr)`.
///
/// Returns `Ok(samples.to_vec())` directly when `src_sr == target_sr`.
/// Errors if `samples.is_empty()`, `src_sr == 0`, or `target_sr == 0`.
pub fn resample_mono(samples: &[f32], src_sr: u32, target_sr: u32) -> anyhow::Result<Vec<f32>>;
```

### 2.3. Transient Impulse Onset Detection & Windowing Functions (`FR-004`, `FR-005`)

```rust
/// Scans a 22,050 Hz Mono waveform using 256-sample short-time RMS energy frames and
/// returns the sample indices (`center_sample = frame_idx * 256`) of valid tap onsets.
///
/// Edge cases:
/// - Returns `Vec::new()` if `samples.is_empty()`.
/// - Returns `vec![0]` if `0 < samples.len() < win_len`.
/// - For `samples.len() >= win_len`, only retains local peaks >= `max(max_rms * threshold_ratio, 1e-4)`
///   separated by at least `min_gap` samples whose `[start..start + win_len]` window fits within `samples`.
pub fn detect_onset_centers(samples: &[f32], cfg: &ImpulseConfig) -> Vec<usize>;

/// Extracts peak-normalized 0.8s (`win_len = 17,640` samples) tap impulse windows
/// using the onset centers detected by `detect_onset_centers`.
///
/// Edge cases:
/// - Returns `Vec::new()` if `samples.is_empty()`.
/// - Zero-pads to `win_len` and returns 1 peak-normalized window if `0 < samples.len() < win_len`.
pub fn extract_tap_impulses(samples: &[f32], cfg: &ImpulseConfig) -> Vec<Vec<f32>>;

/// Normalizes a waveform slice by dividing every sample by `max(max_n |x[n]|, 1e-8)`,
/// guaranteeing all output values lie in `[-1.0, 1.0]` without division by zero.
pub fn normalize_peak(window: &[f32]) -> Vec<f32>;
```

### 2.4. Mel Filterbank, STFT, Log-Mel Spectrogram & Physics Feature Functions (`FR-006` – `FR-008`, `FR-010`)

```rust
/// Converts frequency in Hz to Mel scale: `2595.0 * (1.0 + hz / 700.0).log10()`.
pub fn hz_to_mel(hz: f32) -> f32;

/// Converts Mel scale value back to frequency in Hz: `700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)`.
pub fn mel_to_hz(mel: f32) -> f32;

/// Constructs a Slaney area-normalized triangular Mel filterbank matrix of shape
/// `(n_mels, n_fft / 2 + 1)` (`(128, 513)`) spanning `[fmin, fmax]` Hz.
pub fn build_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Array2<f32>;

/// Computes the 128-frame RealFFT power spectrogram `|X(k, t)|^2` with a Hann window
/// of length `cfg.n_fft` and step `cfg.hop_length`, returning shape `(n_fft / 2 + 1, 128)` (`(513, 128)`).
pub fn compute_power_spectrogram(audio: &[f32], cfg: &SpectrogramConfig) -> Array2<f32>;

/// Projects the STFT power spectrogram onto `mel_basis`, converts to Log-dB relative to
/// window maximum power (`top_db = 80.0`), and linearly scales to `[0.0, 1.0]` with shape `(n_mels, 128)`.
pub fn compute_log_mel_spectrogram(
    audio: &[f32],
    cfg: &SpectrogramConfig,
    mel_basis: &Array2<f32>,
) -> Array2<f32>;

/// Computes the 4 normalized acoustic physics features `[\tilde{f}_peak, \tilde{C}_s, \tilde{R}_85, \tilde{\beta}_decay]`
/// in `[0.0, 1.0]^4` directly from a precomputed STFT power spectrogram of shape `(n_fft / 2 + 1, 128)`.
/// Returns `[0.0, 0.0, 0.0, 0.0]` if total spectrogram energy is `<= 1e-10`.
pub fn extract_physics_features_from_power_spec(
    power_spec: &Array2<f32>,
    cfg: &SpectrogramConfig,
) -> [f32; 4];

/// Standalone convenience wrapper that computes `compute_power_spectrogram(audio, cfg)` and
/// delegates to `extract_physics_features_from_power_spec`.
pub fn extract_physics_features(audio: &[f32], cfg: &SpectrogramConfig) -> [f32; 4];

/// Unified single-pass extractor that computes the STFT power spectrogram once and produces
/// both the `(128, 128)` Log-Mel spectrogram and the `[f32; 4]` physics features in `< 4.0 ms`.
pub fn extract_tap_features(
    audio: &[f32],
    cfg: &SpectrogramConfig,
    mel_basis: &Array2<f32>,
) -> TapFeatures;
```

---

## 3. Module `durian_sonic::augment` (`src/augment.rs`)

All functions in `durian_sonic::augment` are strictly for `Train`-split augmentation (`FR-009`), preserve cavity resonance frequency and damping rate (zero Pitch Shift / Time Stretch), and guard all random range bounds against degenerate inputs.

```rust
use crate::dsp::ImpulseConfig;
use ndarray::Array2;

/// Randomly shifts `center_sample` by `delta in [-max_jitter_samples, +max_jitter_samples]`
/// (default `880` samples = `+-40 ms` at `22,050 Hz`), clamps window boundaries within `full_samples`
/// (or zero-pads if `full_samples.len() < win_len`), and returns a peak-normalized window of length `win_len`.
pub fn apply_onset_jitter(
    full_samples: &[f32],
    center_sample: usize,
    max_jitter_samples: usize,
    cfg: &ImpulseConfig,
) -> Vec<f32>;

/// Adds Gaussian white noise N(0, sigma_noise^2) via Box-Muller transform at a uniformly sampled
/// `SNR_dB in [min(min_snr_db, max_snr_db), max(min_snr_db, max_snr_db)]` (default `[15.0, 35.0] dB`)
/// and re-normalizes peak amplitude to `[-1.0, 1.0]`.
pub fn add_background_noise(window: &[f32], min_snr_db: f32, max_snr_db: f32) -> Vec<f32>;

/// Applies a random linear frequency-response slope `tilt_db in [-|max_tilt_db|, +|max_tilt_db|]`
/// (default `+-3.0 dB` relative to `top_db = 80.0 dB`) across Mel rows `0..M` and clamps values to `[0.0, 1.0]`.
pub fn apply_spectral_tilt(mel_spec: &mut Array2<f32>, max_tilt_db: f32, top_db: f32);

/// Masks up to `min(max_freq_mask, n_mels / 4)` contiguous Mel bands (default `12`) and up to
/// `min(max_time_mask, n_frames / 4)` contiguous time frames (default `16`) in-place with
/// `noise_floor_val` (`0.0`, corresponding to `-80 dB`).
pub fn apply_spec_augment(
    mel_spec: &mut Array2<f32>,
    max_freq_mask: usize,
    max_time_mask: usize,
    noise_floor_val: f32,
);
```
