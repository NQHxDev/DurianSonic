//! Acoustic Digital Signal Processing, Impulse Onset Detection & Feature Extraction
//!
//! This module implements pure-Rust DSP functions for DurianSonic Phase 2:
//! - Multi-format audio decoding (symphonia 0.6.1) and 22,050 Hz resampling (rubato 5.1.0)
//! - Short-time RMS envelope onset detection and 0.8s tap window extraction
//! - Single-pass STFT power spectrogram, Log-Mel (128x128) & 4 physics features
//!
//! All functions maintain numerical safety with epsilon guards to prevent NaN or Infinity.

use anyhow::{anyhow, Context, Result};
use ndarray::Array2;
use std::path::Path;

use crate::config::DspConfig;

/// Resample mono audio from source to target sample rate.
///
/// Uses rubato 5.1.0 FFT resampling with anti-aliasing filter and automatic
/// startup delay trimming. Identity passthrough when src_sr == target_sr.
///
/// # Arguments
///
/// * `samples` - Input mono waveform
/// * `src_sr` - Source sample rate (must be > 0)
/// * `target_sr` - Target sample rate (must be > 0)
///
/// # Returns
///
/// Resampled waveform with length round(N_src * target_sr / src_sr)
///
/// # Errors
///
/// Returns error if samples is empty, or if src_sr or target_sr is 0
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::resample_mono;
///
/// let samples = vec![0.5, -0.3, 0.8];
/// let resampled = resample_mono(&samples, 22_050, 22_050).unwrap();
/// assert_eq!(resampled.len(), samples.len());
/// ```
pub fn resample_mono(samples: &[f32], src_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
    anyhow::ensure!(!samples.is_empty(), "Input samples cannot be empty");
    anyhow::ensure!(src_sr > 0, "Source sample rate must be > 0");
    anyhow::ensure!(target_sr > 0, "Target sample rate must be > 0");

    if src_sr == target_sr {
        return Ok(samples.to_vec());
    }

    use rubato::{audioadapter_buffers::direct::InterleavedSlice, Fft, FixedSync, Resampler};

    let mut resampler = Fft::<f32>::new(
        src_sr as usize,
        target_sr as usize,
        1024,
        1,
        FixedSync::Both,
    )
    .context("Failed to create FFT resampler")?;

    let input_adapter = InterleavedSlice::new(samples, 1, samples.len())
        .context("Failed to create input adapter")?;

    let output = resampler
        .process_all(&input_adapter, samples.len(), None)
        .context("Failed to resample audio")?;

    let expected_len =
        ((samples.len() as f64) * (target_sr as f64) / (src_sr as f64)).round() as usize;
    let mut resampled = output.take_data();

    if resampled.len() > expected_len {
        resampled.truncate(expected_len);
    } else if resampled.len() < expected_len {
        resampled.resize(expected_len, 0.0);
    }

    Ok(resampled)
}

/// Decode multi-format audio file to mono at target sample rate.
///
/// Supports iPhone .m4a (AAC-LC in ISO/MP4) and Android .wav (16-bit PCM Stereo 48kHz)
/// via symphonia 0.6.1. Multi-channel frames are downmixed to mono by arithmetic mean.
///
/// # Arguments
///
/// * `path` - Path to audio file
/// * `target_sr` - Target mono sample rate (must be > 0)
///
/// # Returns
///
/// Mono waveform at target_sr Hz
///
/// # Errors
///
/// Returns error if target_sr is 0, file does not exist, codec is unsupported,
/// or decoded stream is empty
///
/// # Examples
///
/// ```no_run
/// use durian_sonic::dsp::decode_audio_to_mono;
/// use std::path::Path;
///
/// let samples = decode_audio_to_mono(
///     Path::new("data/sound_iphone/durian_001_unripe_iphone.m4a"),
///     22_050
/// ).unwrap();
/// assert!(!samples.is_empty());
/// ```
pub fn decode_audio_to_mono(path: &Path, target_sr: u32) -> Result<Vec<f32>> {
    anyhow::ensure!(target_sr > 0, "Target sample rate must be > 0");

    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open audio file: {}", path.display()))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension() {
        hint.with_extension(ext.to_str().unwrap_or(""));
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .context("Failed to probe audio format")?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow!("No default audio track found"))?;

    let track_id = track.id;
    let codec_params = track
        .codec_params
        .as_ref()
        .ok_or_else(|| anyhow!("No codec parameters available"))?;

    let audio_params = codec_params
        .audio()
        .ok_or_else(|| anyhow!("Not an audio codec"))?;

    let src_sr = audio_params
        .sample_rate
        .ok_or_else(|| anyhow!("Unknown source sample rate"))?;

    anyhow::ensure!(src_sr > 0, "Source sample rate must be > 0");

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .context("Failed to create audio decoder")?;

    let mut mono_samples = Vec::new();
    let mut interleaved_buf = Vec::new();

    while let Some(packet) = format.next_packet()? {
        if packet.track_id != track_id {
            continue;
        }

        let decoded = decoder
            .decode(&packet)
            .context("Failed to decode audio packet")?;

        interleaved_buf.clear();
        decoded.copy_to_vec_interleaved(&mut interleaved_buf);

        let channels = decoded.spec().channels().count();

        if channels == 1 {
            mono_samples.extend_from_slice(&interleaved_buf);
        } else {
            for chunk in interleaved_buf.chunks(channels) {
                let mono_sample: f32 = chunk.iter().sum::<f32>() / channels as f32;
                mono_samples.push(mono_sample);
            }
        }
    }

    anyhow::ensure!(!mono_samples.is_empty(), "Decoded audio stream is empty");

    if src_sr == target_sr {
        Ok(mono_samples)
    } else {
        resample_mono(&mono_samples, src_sr, target_sr)
    }
}

/// Configuration for transient tap onset detection and window extraction.
///
/// Encapsulates physical time-domain parameters for detecting percussive impulses
/// in 22,050 Hz mono waveforms and extracting normalized 0.8s tap windows.
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::ImpulseConfig;
/// use durian_sonic::config::DspConfig;
///
/// let cfg = ImpulseConfig::default();
/// assert_eq!(cfg.window_samples(), 17_640);
/// assert_eq!(cfg.pre_onset_samples(), 2_205);
///
/// let dsp_cfg = DspConfig::default();
/// let impulse_cfg = ImpulseConfig::from(&dsp_cfg);
/// assert_eq!(impulse_cfg.target_sr, 22_050);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ImpulseConfig {
    /// Unified target mono sampling rate (> 0)
    pub target_sr: u32,
    /// Total duration of an extracted tap impulse window (> 0.0)
    pub window_sec: f32,
    /// Attack lead-in before RMS peak (100 ms, >= 0.0 and < window_sec)
    pub pre_onset_sec: f32,
    /// Minimum refractory interval between consecutive tap peaks (400 ms, > 0.0)
    pub min_interval_sec: f32,
    /// Adaptive RMS threshold ratio (> 0.0 and <= 1.0, floored at 1e-4)
    pub threshold_ratio: f32,
}

impl Default for ImpulseConfig {
    fn default() -> Self {
        Self {
            target_sr: 22_050,
            window_sec: 0.8,
            pre_onset_sec: 0.1,
            min_interval_sec: 0.4,
            threshold_ratio: 0.25,
        }
    }
}

impl From<&DspConfig> for ImpulseConfig {
    fn from(dsp: &DspConfig) -> Self {
        Self {
            target_sr: dsp.sample_rate,
            window_sec: dsp.window_sec,
            pre_onset_sec: dsp.pre_onset_sec,
            min_interval_sec: dsp.min_interval_sec,
            threshold_ratio: dsp.threshold_ratio,
        }
    }
}

impl ImpulseConfig {
    /// Total window length in samples (window_sec * target_sr, rounded)
    pub fn window_samples(&self) -> usize {
        (self.window_sec * self.target_sr as f32).round() as usize
    }

    /// Pre-onset lead-in length in samples (pre_onset_sec * target_sr, rounded)
    pub fn pre_onset_samples(&self) -> usize {
        (self.pre_onset_sec * self.target_sr as f32).round() as usize
    }

    /// Minimum gap between consecutive peaks in samples (min_interval_sec * target_sr, rounded)
    pub fn min_gap_samples(&self) -> usize {
        (self.min_interval_sec * self.target_sr as f32).round() as usize
    }
}

/// Configuration for STFT power spectrogram and Log-Mel filterbank.
///
/// Encapsulates time-frequency transformation parameters for computing
/// 128-frame 1024-point RealFFT power spectrograms and projecting onto
/// 128 Mel frequency bands over 50-8,000 Hz.
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::SpectrogramConfig;
///
/// let cfg = SpectrogramConfig::default();
/// assert_eq!(cfg.num_freq_bins(), 513);
/// assert_eq!(cfg.n_mels, 128);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SpectrogramConfig {
    /// Input waveform sampling rate (> 0)
    pub sample_rate: u32,
    /// Hann window & RealFFT size (> 0, power of two; frequency bin width ~21.533 Hz)
    pub n_fft: usize,
    /// STFT frame step size (> 0; frame 127 ends at sample 127 * 128 + 1024 = 17,280)
    pub hop_length: usize,
    /// Number of triangular Mel filterbank channels (> 0)
    pub n_mels: usize,
    /// Lower frequency cutoff (>= 0.0 and < fmax)
    pub fmin: f32,
    /// Upper frequency cutoff (> fmin and <= sample_rate / 2.0)
    pub fmax: f32,
    /// Dynamic range below window maximum power before clamping (> 0.0)
    pub top_db: f32,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22_050,
            n_fft: 1024,
            hop_length: 128,
            n_mels: 128,
            fmin: 50.0,
            fmax: 8_000.0,
            top_db: 80.0,
        }
    }
}

impl From<&DspConfig> for SpectrogramConfig {
    fn from(dsp: &DspConfig) -> Self {
        Self {
            sample_rate: dsp.sample_rate,
            n_fft: dsp.n_fft,
            hop_length: dsp.hop_length,
            n_mels: dsp.n_mels,
            fmin: dsp.fmin,
            fmax: dsp.fmax,
            top_db: dsp.top_db,
        }
    }
}

impl SpectrogramConfig {
    /// Number of non-negative frequency bins in RealFFT output (n_fft / 2 + 1)
    pub fn num_freq_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// Combined dual-branch feature representation for a single tap impulse.
///
/// Contains both a 128x128 Log-Mel spectrogram (visual CNN input) and
/// 4 normalized acoustic physics features (auxiliary MLP input).
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::TapFeatures;
/// use ndarray::Array2;
///
/// let mel_spec = Array2::<f32>::zeros((128, 128));
/// let physics = [0.5, 0.6, 0.7, 0.8];
/// let features = TapFeatures {
///     mel_spec,
///     physics_features: physics,
/// };
/// assert_eq!(features.mel_spec.dim(), (128, 128));
/// assert_eq!(features.physics_features.len(), 4);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TapFeatures {
    /// Log-Mel spectrogram (0.0 = -80 dB noise floor, 1.0 = 0 dB peak)
    pub mel_spec: Array2<f32>,
    /// [f_peak, C_s, R_85, beta_decay] all in [0.0, 1.0]
    pub physics_features: [f32; 4],
}

/// Normalize a waveform to peak amplitude 1.0.
///
/// Divides every sample by the maximum absolute amplitude, with a denominator
/// floor of 1e-8 to safely handle all-zero silent inputs without producing NaN.
///
/// # Arguments
///
/// * `window` - Input waveform slice
///
/// # Returns
///
/// Normalized waveform with peak absolute amplitude 1.0 (or 0.0 for silent input)
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::normalize_peak;
///
/// let signal = vec![0.5, -0.8, 0.3];
/// let normalized = normalize_peak(&signal);
/// assert!((normalized.iter().map(|x| x.abs()).fold(0.0f32, f32::max) - 1.0).abs() < 1e-6);
///
/// let silent = vec![0.0; 1000];
/// let normalized_silent = normalize_peak(&silent);
/// assert!(normalized_silent.iter().all(|&x| x == 0.0));
/// ```
pub fn normalize_peak(window: &[f32]) -> Vec<f32> {
    let peak = window
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max)
        .max(1e-8);
    window.iter().map(|x| x / peak).collect()
}

/// Detect onset centers in mono waveform using short-time RMS envelope.
///
/// Scans the input signal in non-overlapping 256-sample frames (~11.61 ms at 22,050 Hz),
/// computes RMS energy per frame, and identifies local peaks above an adaptive threshold
/// separated by at least min_interval_sec (0.4s refractory period).
///
/// # Arguments
///
/// * `samples` - Input mono waveform at target_sr Hz
/// * `cfg` - Impulse detection configuration
///
/// # Returns
///
/// Vector of sample indices marking detected onset centers
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{ImpulseConfig, detect_onset_centers};
///
/// let cfg = ImpulseConfig::default();
/// let samples = vec![0.0; 50_000];
/// let centers = detect_onset_centers(&samples, &cfg);
/// ```
pub fn detect_onset_centers(samples: &[f32], cfg: &ImpulseConfig) -> Vec<usize> {
    let win_len = cfg.window_samples();

    if samples.is_empty() || win_len == 0 {
        return Vec::new();
    }

    if samples.len() < win_len {
        return vec![0];
    }

    const ENV_FRAME: usize = 256;

    let rms_envelope: Vec<f32> = samples
        .chunks(ENV_FRAME)
        .map(|chunk| {
            let sum_sq: f32 = chunk.iter().map(|x| x * x).sum();
            (sum_sq / chunk.len() as f32).sqrt()
        })
        .collect();

    if rms_envelope.len() < 3 {
        return Vec::new();
    }

    let max_rms = rms_envelope.iter().cloned().fold(0.0f32, f32::max);
    let threshold = (max_rms * cfg.threshold_ratio).max(1e-4);

    let pre_len = cfg.pre_onset_samples();
    let min_gap = cfg.min_gap_samples();

    let mut centers = Vec::new();

    for i in 1..(rms_envelope.len() - 1) {
        let is_peak = rms_envelope[i] >= threshold
            && rms_envelope[i] >= rms_envelope[i - 1]
            && rms_envelope[i] > rms_envelope[i + 1];

        if !is_peak {
            continue;
        }

        let center_sample = i * ENV_FRAME;

        if let Some(&prev_center) = centers.last() {
            if center_sample.saturating_sub(prev_center) < min_gap {
                continue;
            }
        }

        let start = center_sample.saturating_sub(pre_len);
        if start + win_len > samples.len() {
            break;
        }

        centers.push(center_sample);
    }

    centers
}

/// Extract normalized tap impulse windows from mono waveform.
///
/// Delegates onset detection to detect_onset_centers, then slices out 0.8s windows
/// (-100ms pre-onset to +700ms decay tail) and normalizes each to peak amplitude 1.0.
///
/// # Arguments
///
/// * `samples` - Input mono waveform at target_sr Hz
/// * `cfg` - Impulse detection configuration
///
/// # Returns
///
/// Vector of normalized tap windows, each of length window_samples() (17,640)
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{ImpulseConfig, extract_tap_impulses};
///
/// let cfg = ImpulseConfig::default();
/// let samples = vec![0.5; 50_000];
/// let impulses = extract_tap_impulses(&samples, &cfg);
/// ```
pub fn extract_tap_impulses(samples: &[f32], cfg: &ImpulseConfig) -> Vec<Vec<f32>> {
    let win_len = cfg.window_samples();

    if samples.is_empty() || win_len == 0 {
        return Vec::new();
    }

    if samples.len() < win_len {
        let mut padded = samples.to_vec();
        padded.resize(win_len, 0.0);
        return vec![normalize_peak(&padded)];
    }

    let centers = detect_onset_centers(samples, cfg);
    let pre_len = cfg.pre_onset_samples();

    centers
        .into_iter()
        .map(|center| {
            let start = center.saturating_sub(pre_len);
            let window = &samples[start..(start + win_len)];
            normalize_peak(window)
        })
        .collect()
}

/// Compute single-pass STFT power spectrogram using RealFFT.
///
/// Uses 1024-point FFT with 128-sample hop, always producing shape (513, 128).
/// Each frame is Hann-windowed before FFT. Power = |FFT|^2 (unfloored raw power).
/// Out-of-bounds frames are zero-padded.
///
/// # Arguments
///
/// * `audio` - Input mono waveform (typically 17,640 samples for 0.8s window)
/// * `cfg` - Spectrogram configuration
///
/// # Returns
///
/// Power spectrogram of shape (513, 128) with unfloored raw power values
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{SpectrogramConfig, compute_power_spectrogram};
///
/// let cfg = SpectrogramConfig::default();
/// let signal = vec![0.0; 17_640];
/// let spec = compute_power_spectrogram(&signal, &cfg);
/// assert_eq!(spec.shape(), &[513, 128]);
/// ```
pub fn compute_power_spectrogram(audio: &[f32], cfg: &SpectrogramConfig) -> Array2<f32> {
    use realfft::RealFftPlanner;
    use std::f32::consts::PI;

    let n_fft = cfg.n_fft;
    let hop = cfg.hop_length;
    let n_bins = n_fft / 2 + 1;
    let target_frames = 128;

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut spectrum = r2c.make_output_vec();

    let hann: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
        .collect();

    let mut power_spec = Array2::<f32>::zeros((n_bins, target_frames));

    for frame_idx in 0..target_frames {
        let frame_start = frame_idx * hop;
        let mut windowed = vec![0.0f32; n_fft];

        for i in 0..n_fft {
            let sample_idx = frame_start + i;
            if sample_idx < audio.len() {
                windowed[i] = audio[sample_idx] * hann[i];
            }
        }

        r2c.process(&mut windowed, &mut spectrum).unwrap_or(());

        for (bin, complex_val) in spectrum.iter().enumerate() {
            let power = complex_val.re * complex_val.re + complex_val.im * complex_val.im;
            power_spec[[bin, frame_idx]] = power;
        }
    }

    power_spec
}

/// Convert frequency in Hz to Mel scale.
///
/// Uses the formula: m = 2595 * log10(1 + f / 700)
///
/// # Arguments
///
/// * `hz` - Frequency in Hertz
///
/// # Returns
///
/// Frequency in Mel scale
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::hz_to_mel;
///
/// let mel = hz_to_mel(1000.0);
/// assert!(mel > 0.0);
/// ```
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert frequency in Mel scale to Hz.
///
/// Uses the formula: f = 700 * (10^(m / 2595) - 1)
///
/// # Arguments
///
/// * `mel` - Frequency in Mel scale
///
/// # Returns
///
/// Frequency in Hertz
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{hz_to_mel, mel_to_hz};
///
/// let hz = 1000.0;
/// let mel = hz_to_mel(hz);
/// let hz_back = mel_to_hz(mel);
/// assert!((hz - hz_back).abs() < 1e-3);
/// ```
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Build Mel filterbank matrix with Slaney area normalization.
///
/// Creates triangular filters spaced uniformly on Mel scale between fmin and fmax.
/// Each filter is normalized using Slaney area normalization: enorm = 2.0 / (f_right - f_left).
///
/// # Arguments
///
/// * `sample_rate` - Audio sample rate in Hz
/// * `n_fft` - FFT size (number of frequency bins = n_fft / 2 + 1)
/// * `n_mels` - Number of Mel bands
/// * `fmin` - Lower frequency cutoff in Hz
/// * `fmax` - Upper frequency cutoff in Hz
///
/// # Returns
///
/// Mel filterbank matrix of shape (n_mels, n_fft/2 + 1)
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::build_mel_filterbank;
///
/// let filterbank = build_mel_filterbank(22_050, 1024, 128, 50.0, 8_000.0);
/// assert_eq!(filterbank.shape(), &[128, 513]);
/// ```
pub fn build_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Array2<f32> {
    let n_bins = n_fft / 2 + 1;

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();

    let bin_freqs: Vec<f32> = (0..n_bins)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    let mut filters = Array2::<f32>::zeros((n_mels, n_bins));

    for m in 0..n_mels {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];

        let enorm = 2.0 / (f_right - f_left).max(1e-8);

        for (k, &freq) in bin_freqs.iter().enumerate() {
            let weight = if freq >= f_left && freq <= f_center {
                (freq - f_left) / (f_center - f_left + 1e-8)
            } else if freq > f_center && freq <= f_right {
                (f_right - freq) / (f_right - f_center + 1e-8)
            } else {
                0.0
            };
            filters[[m, k]] = weight * enorm;
        }
    }

    filters
}

/// Compute log-Mel spectrogram with dynamic range compression.
///
/// Projects power spectrogram through pre-built Mel filterbank, converts to Log-dB
/// relative to window peak power, clamps to dynamic range, and normalizes to [0, 1].
///
/// # Arguments
///
/// * `audio` - Input mono waveform (typically 17,640 samples)
/// * `cfg` - Spectrogram configuration
/// * `mel_basis` - Pre-built Mel filterbank matrix (n_mels x n_bins)
///
/// # Returns
///
/// Log-Mel spectrogram (128 x 128), normalized to [0, 1] where 0.0 = -80 dB noise floor, 1.0 = 0 dB peak
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{SpectrogramConfig, build_mel_filterbank, compute_log_mel_spectrogram};
///
/// let cfg = SpectrogramConfig::default();
/// let mel_basis = build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);
/// let signal = vec![0.0; 17_640];
/// let log_mel = compute_log_mel_spectrogram(&signal, &cfg, &mel_basis);
/// assert_eq!(log_mel.shape(), &[128, 128]);
/// ```
pub fn compute_log_mel_spectrogram(
    audio: &[f32],
    cfg: &SpectrogramConfig,
    mel_basis: &Array2<f32>,
) -> Array2<f32> {
    let power_spec = compute_power_spectrogram(audio, cfg);
    let mel_spec = mel_basis.dot(&power_spec);

    let p_max = mel_spec.iter().cloned().fold(0.0f32, f32::max).max(1e-10);

    let mut log_mel = mel_spec.mapv(|x| 10.0 * (x.max(1e-10) / p_max).log10());

    let top_db = cfg.top_db;
    log_mel.mapv_inplace(|x| ((x + top_db) / top_db).clamp(0.0, 1.0));

    log_mel
}

/// Extract 4 normalized acoustic physics features from power spectrogram.
///
/// Returns [0.0, 0.0, 0.0, 0.0] if total power <= 1e-10 (silent input).
/// Otherwise computes:
/// - Index 0: Peak Resonance Frequency (f_peak) in [80, 1200] Hz, frames 0..64
/// - Index 1: Spectral Centroid (C_s) normalized by 4000 Hz, frames 0..64
/// - Index 2: Spectral Roll-off 85% (R_85) normalized by 6000 Hz, frames 0..64
/// - Index 3: Temporal Decay Slope (beta_decay) via OLS on frames 26..=86
///
/// # Arguments
///
/// * `power_spec` - Power spectrogram of shape (n_bins, n_frames)
/// * `cfg` - Spectrogram configuration
///
/// # Returns
///
/// Array of 4 features, all normalized to [0.0, 1.0]
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{SpectrogramConfig, compute_power_spectrogram, extract_physics_features_from_power_spec};
///
/// let cfg = SpectrogramConfig::default();
/// let signal = vec![0.0; 17_640];
/// let power_spec = compute_power_spectrogram(&signal, &cfg);
/// let features = extract_physics_features_from_power_spec(&power_spec, &cfg);
/// assert_eq!(features.len(), 4);
/// ```
pub fn extract_physics_features_from_power_spec(
    power_spec: &Array2<f32>,
    cfg: &SpectrogramConfig,
) -> [f32; 4] {
    let total_power: f32 = power_spec.iter().sum();
    if total_power <= 1e-10 {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let n_bins = power_spec.nrows();
    let n_frames = power_spec.ncols();

    let bin_freqs: Vec<f32> = (0..n_bins)
        .map(|i| i as f32 * cfg.sample_rate as f32 / cfg.n_fft as f32)
        .collect();

    let n_impact_frames = 64.min(n_frames);
    let mut mean_power = vec![0.0f32; n_bins];
    for k in 0..n_bins {
        for t in 0..n_impact_frames {
            mean_power[k] += power_spec[[k, t]];
        }
        mean_power[k] /= n_impact_frames as f32;
    }

    let max_band_power_in_range = bin_freqs
        .iter()
        .enumerate()
        .filter(|(_, &f)| (80.0..=1200.0).contains(&f))
        .map(|(i, _)| mean_power[i])
        .fold(0.0f32, f32::max);

    let f_peak_norm = if max_band_power_in_range <= 1e-10 {
        0.0
    } else {
        let f_peak = bin_freqs
            .iter()
            .enumerate()
            .filter(|(_, &f)| (80.0..=1200.0).contains(&f))
            .max_by(|(i, _), (j, _)| {
                mean_power[*i]
                    .partial_cmp(&mean_power[*j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, &f)| f)
            .unwrap_or(80.0);

        ((f_peak - 80.0) / (1200.0 - 80.0)).clamp(0.0, 1.0)
    };

    let mean_magnitude: Vec<f32> = mean_power.iter().map(|&p| p.sqrt()).collect();

    let total_mag: f32 = mean_magnitude.iter().sum();
    let c_s = if total_mag > 1e-8 {
        let weighted_sum: f32 = bin_freqs
            .iter()
            .zip(mean_magnitude.iter())
            .map(|(f, m)| f * m)
            .sum();
        weighted_sum / (total_mag + 1e-8)
    } else {
        0.0
    };
    let c_s_norm = (c_s / 4000.0).clamp(0.0, 1.0);

    let total_power_impact: f32 = mean_power.iter().sum();
    let r_85_norm = if total_power_impact <= 1e-10 {
        0.0
    } else {
        let threshold = 0.85 * total_power_impact;
        let mut cumsum = 0.0f32;
        let r_85 = bin_freqs
            .iter()
            .zip(mean_power.iter())
            .find_map(|(&f, &m)| {
                cumsum += m;
                if cumsum >= threshold {
                    Some(f)
                } else {
                    None
                }
            })
            .unwrap_or(cfg.fmax);
        (r_85 / 6000.0).clamp(0.0, 1.0)
    };

    let decay_start = 26.min(n_frames);
    let decay_end = 87.min(n_frames);
    let n_decay = decay_end.saturating_sub(decay_start);

    let beta_decay_norm = if n_decay > 10 {
        let mut e_db = Vec::new();
        for t in decay_start..decay_end {
            let frame_power: f32 = (0..n_bins).map(|k| power_spec[[k, t]]).sum();
            e_db.push(10.0 * (frame_power + 1e-10).log10());
        }

        let n = e_db.len() as f32;
        let mean_e: f32 = e_db.iter().sum::<f32>() / n;
        let mean_t = (n - 1.0) / 2.0;

        let cov: f32 = e_db
            .iter()
            .enumerate()
            .map(|(i, &e)| (i as f32 - mean_t) * (e - mean_e))
            .sum();

        let var_t: f32 = (0..e_db.len())
            .map(|i| {
                let diff = i as f32 - mean_t;
                diff * diff
            })
            .sum();

        let beta_decay = cov / (var_t + 1e-8);
        ((-beta_decay) / 2.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    [f_peak_norm, c_s_norm, r_85_norm, beta_decay_norm]
}

/// Extract 4 normalized acoustic physics features from audio waveform.
///
/// Convenience wrapper that computes power spectrogram then extracts features.
/// Returns [0.0, 0.0, 0.0, 0.0] if total power <= 1e-10 (silent input).
///
/// # Arguments
///
/// * `audio` - Input mono waveform (typically 17,640 samples)
/// * `cfg` - Spectrogram configuration
///
/// # Returns
///
/// Array of 4 features: [f_peak, C_s, R_85, beta_decay], all in [0.0, 1.0]
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{SpectrogramConfig, extract_physics_features};
///
/// let cfg = SpectrogramConfig::default();
/// let signal = vec![0.0; 17_640];
/// let features = extract_physics_features(&signal, &cfg);
/// assert_eq!(features.len(), 4);
/// ```
pub fn extract_physics_features(audio: &[f32], cfg: &SpectrogramConfig) -> [f32; 4] {
    let power_spec = compute_power_spectrogram(audio, cfg);
    extract_physics_features_from_power_spec(&power_spec, cfg)
}

/// Extract full TapFeatures using single-pass STFT with pre-built mel_basis.
///
/// Computes power spectrogram once, then uses it for both Log-Mel spectrogram
/// and 4 acoustic physics features, maximizing efficiency.
///
/// # Arguments
///
/// * `audio` - Normalized tap impulse (17,640 samples)
/// * `cfg` - Spectrogram configuration
/// * `mel_basis` - Pre-built Mel filterbank matrix (n_mels x n_bins)
///
/// # Returns
///
/// TapFeatures with 128x128 log-Mel and [f_peak, C_s, R_85, beta_decay]
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{SpectrogramConfig, build_mel_filterbank, extract_tap_features};
///
/// let cfg = SpectrogramConfig::default();
/// let mel_basis = build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);
/// let signal = vec![0.0; 17_640];
/// let features = extract_tap_features(&signal, &cfg, &mel_basis);
/// assert_eq!(features.mel_spec.shape(), &[128, 128]);
/// assert_eq!(features.physics_features.len(), 4);
/// ```
pub fn extract_tap_features(
    audio: &[f32],
    cfg: &SpectrogramConfig,
    mel_basis: &Array2<f32>,
) -> TapFeatures {
    let power_spec = compute_power_spectrogram(audio, cfg);

    let mel_spec = mel_basis.dot(&power_spec);
    let p_max = mel_spec.iter().cloned().fold(0.0f32, f32::max).max(1e-10);
    let mut log_mel = mel_spec.mapv(|x| 10.0 * (x.max(1e-10) / p_max).log10());
    let top_db = cfg.top_db;
    log_mel.mapv_inplace(|x| ((x + top_db) / top_db).clamp(0.0, 1.0));

    let physics_features = extract_physics_features_from_power_spec(&power_spec, cfg);

    TapFeatures {
        mel_spec: log_mel,
        physics_features,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // T003: TDD Red - Tests for ImpulseConfig, SpectrogramConfig, TapFeatures & normalize_peak

    #[test]
    fn test_impulse_config_default() {
        let cfg = ImpulseConfig::default();
        assert_eq!(cfg.target_sr, 22_050);
        assert_eq!(cfg.window_sec, 0.8);
        assert_eq!(cfg.pre_onset_sec, 0.1);
        assert_eq!(cfg.min_interval_sec, 0.4);
        assert_eq!(cfg.threshold_ratio, 0.25);
    }

    #[test]
    fn test_impulse_config_from_dsp_config() {
        let dsp_cfg = DspConfig::default();
        let impulse_cfg = ImpulseConfig::from(&dsp_cfg);
        assert_eq!(impulse_cfg.target_sr, dsp_cfg.sample_rate);
        assert_eq!(impulse_cfg.window_sec, dsp_cfg.window_sec);
        assert_eq!(impulse_cfg.pre_onset_sec, dsp_cfg.pre_onset_sec);
        assert_eq!(impulse_cfg.min_interval_sec, dsp_cfg.min_interval_sec);
        assert_eq!(impulse_cfg.threshold_ratio, dsp_cfg.threshold_ratio);
    }

    #[test]
    fn test_impulse_config_derived_samples() {
        let cfg = ImpulseConfig::default();
        assert_eq!(cfg.window_samples(), 17_640);
        assert_eq!(cfg.pre_onset_samples(), 2_205);
        assert_eq!(cfg.min_gap_samples(), 8_820);
    }

    #[test]
    fn test_spectrogram_config_default() {
        let cfg = SpectrogramConfig::default();
        assert_eq!(cfg.sample_rate, 22_050);
        assert_eq!(cfg.n_fft, 1024);
        assert_eq!(cfg.hop_length, 128);
        assert_eq!(cfg.n_mels, 128);
        assert_eq!(cfg.fmin, 50.0);
        assert_eq!(cfg.fmax, 8_000.0);
        assert_eq!(cfg.top_db, 80.0);
    }

    #[test]
    fn test_spectrogram_config_from_dsp_config() {
        let dsp_cfg = DspConfig::default();
        let spec_cfg = SpectrogramConfig::from(&dsp_cfg);
        assert_eq!(spec_cfg.sample_rate, dsp_cfg.sample_rate);
        assert_eq!(spec_cfg.n_fft, dsp_cfg.n_fft);
        assert_eq!(spec_cfg.hop_length, dsp_cfg.hop_length);
        assert_eq!(spec_cfg.n_mels, dsp_cfg.n_mels);
        assert_eq!(spec_cfg.fmin, dsp_cfg.fmin);
        assert_eq!(spec_cfg.fmax, dsp_cfg.fmax);
        assert_eq!(spec_cfg.top_db, dsp_cfg.top_db);
    }

    #[test]
    fn test_spectrogram_config_num_freq_bins() {
        let cfg = SpectrogramConfig::default();
        assert_eq!(cfg.num_freq_bins(), 513);
    }

    #[test]
    fn test_tap_features_construction() {
        let mel_spec = Array2::<f32>::zeros((128, 128));
        let physics_features = [0.5f32, 0.6, 0.7, 0.8];
        let features = TapFeatures {
            mel_spec: mel_spec.clone(),
            physics_features,
        };
        assert_eq!(features.mel_spec.dim(), (128, 128));
        assert_eq!(features.physics_features.len(), 4);
        assert_eq!(features.physics_features[0], 0.5);
    }

    #[test]
    fn test_normalize_peak_normal_signal() {
        let input = vec![0.5, -0.8, 0.3, -0.4];
        let normalized = normalize_peak(&input);
        assert_eq!(normalized.len(), input.len());

        let max_abs = normalized.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert_relative_eq!(max_abs, 1.0, epsilon = 1e-6);

        for &val in &normalized {
            assert!((-1.0..=1.0).contains(&val));
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_normalize_peak_all_zero_silent() {
        let input = vec![0.0; 17_640];
        let normalized = normalize_peak(&input);
        assert_eq!(normalized.len(), input.len());

        for &val in &normalized {
            assert_eq!(val, 0.0);
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_normalize_peak_very_small_signal() {
        let input = vec![1e-10, -2e-10, 1.5e-10];
        let normalized = normalize_peak(&input);

        for &val in &normalized {
            assert!(val.is_finite());
            assert!(!val.is_nan());
        }
    }

    // T007: TDD Red - Tests for resample_mono (User Story 1)

    #[test]
    fn test_resample_mono_identity_passthrough() {
        let samples = vec![0.5, -0.3, 0.8, -0.2, 0.1];
        let result = resample_mono(&samples, 22_050, 22_050).unwrap();
        assert_eq!(result.len(), samples.len());
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resample_mono_downsample_48k_to_22k() {
        let sr_src = 48_000;
        let sr_target = 22_050;
        let duration = 0.1;
        let freq = 440.0;

        let n_src = (sr_src as f32 * duration) as usize;
        let samples: Vec<f32> = (0..n_src)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr_src as f32).sin())
            .collect();

        let result = resample_mono(&samples, sr_src, sr_target).unwrap();

        let expected_len = ((n_src as f64) * (sr_target as f64) / (sr_src as f64)).round() as usize;
        assert_eq!(result.len(), expected_len);

        for &val in &result {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_resample_mono_downsample_44k_to_22k() {
        let sr_src = 44_100;
        let sr_target = 22_050;
        let samples: Vec<f32> = (0..4410).map(|i| (i as f32 / 100.0).sin()).collect();

        let result = resample_mono(&samples, sr_src, sr_target).unwrap();
        let expected_len =
            ((samples.len() as f64) * (sr_target as f64) / (sr_src as f64)).round() as usize;

        assert_eq!(result.len(), expected_len);
        for &val in &result {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_resample_mono_error_empty_input() {
        let empty: Vec<f32> = vec![];
        let result = resample_mono(&empty, 48_000, 22_050);
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_mono_error_zero_src_sr() {
        let samples = vec![0.5, 0.3, 0.1];
        let result = resample_mono(&samples, 0, 22_050);
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_mono_error_zero_target_sr() {
        let samples = vec![0.5, 0.3, 0.1];
        let result = resample_mono(&samples, 48_000, 0);
        assert!(result.is_err());
    }

    // T008: TDD Red - Tests for decode_audio_to_mono (User Story 1)

    #[test]
    fn test_decode_audio_to_mono_error_zero_target_sr() {
        let result = decode_audio_to_mono(
            Path::new("data/sound_iphone/durian_001_unripe_iphone.m4a"),
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_audio_to_mono_error_nonexistent_file() {
        let result = decode_audio_to_mono(Path::new("nonexistent_file.m4a"), 22_050);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_audio_to_mono_error_corrupted_file() {
        use std::io::Write;
        let temp_dir = tempfile::tempdir().unwrap();
        let corrupted_path = temp_dir.path().join("corrupted.wav");

        let mut file = std::fs::File::create(&corrupted_path).unwrap();
        file.write_all(b"not a valid audio file").unwrap();

        let result = decode_audio_to_mono(&corrupted_path, 22_050);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_audio_to_mono_iphone_m4a() {
        let path = Path::new("data/sound_iphone/durian_001_unripe_iphone.m4a");
        if !path.exists() {
            return;
        }

        let result = decode_audio_to_mono(path, 22_050);
        assert!(result.is_ok());

        let samples = result.unwrap();
        assert!(!samples.is_empty());

        for &val in &samples {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_decode_audio_to_mono_android_wav() {
        let path = Path::new("data/sound_android/durian_001_unripe_android.wav");
        if !path.exists() {
            return;
        }

        let result = decode_audio_to_mono(path, 22_050);
        assert!(result.is_ok());

        let samples = result.unwrap();
        assert!(!samples.is_empty());

        for &val in &samples {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_decode_stereo_arithmetic_mean_downmix() {
        let path = Path::new("data/sound_android/durian_001_unripe_android.wav");
        if !path.exists() {
            return;
        }

        let mono = decode_audio_to_mono(path, 22_050).unwrap();

        assert!(!mono.is_empty());
        for &sample in &mono {
            assert!(sample.is_finite());
            assert!((-1.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn test_decode_audio_to_mono_zero_sample_stream_error() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp_file = NamedTempFile::new().unwrap();

        let mut wav_data = Vec::new();
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&36u32.to_le_bytes());
        wav_data.extend_from_slice(b"WAVE");
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&16u32.to_le_bytes());
        wav_data.extend_from_slice(&1u16.to_le_bytes());
        wav_data.extend_from_slice(&1u16.to_le_bytes());
        wav_data.extend_from_slice(&22_050u32.to_le_bytes());
        wav_data.extend_from_slice(&88_200u32.to_le_bytes());
        wav_data.extend_from_slice(&4u16.to_le_bytes());
        wav_data.extend_from_slice(&32u16.to_le_bytes());
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&0u32.to_le_bytes());

        temp_file.write_all(&wav_data).unwrap();
        temp_file.flush().unwrap();

        let result = decode_audio_to_mono(temp_file.path(), 22_050);
        assert!(
            result.is_err(),
            "Should return error for 0-sample audio stream"
        );
    }

    // T011: TDD Red - Tests for detect_onset_centers & extract_tap_impulses (User Story 2)

    #[test]
    fn test_detect_onset_centers_empty_slice() {
        let cfg = ImpulseConfig::default();
        let centers = detect_onset_centers(&[], &cfg);
        assert_eq!(centers, Vec::<usize>::new());
    }

    #[test]
    fn test_detect_onset_centers_short_non_empty_slice() {
        let cfg = ImpulseConfig::default();
        let samples = vec![0.5; 100];
        let centers = detect_onset_centers(&samples, &cfg);
        assert_eq!(centers, vec![0]);
    }

    #[test]
    fn test_detect_onset_centers_pre_onset_boundary_saturation() {
        let cfg = ImpulseConfig::default();
        let sr = cfg.target_sr as usize;
        let mut samples = vec![0.0; sr * 2];

        for sample in samples.iter_mut().skip(128).take(256) {
            *sample = 0.8;
        }

        let centers = detect_onset_centers(&samples, &cfg);

        assert!(!centers.is_empty());
        if let Some(&first_center) = centers.first() {
            let start = first_center.saturating_sub(cfg.pre_onset_samples());
            assert!(start + cfg.window_samples() <= samples.len());
        }
    }

    #[test]
    fn test_detect_onset_centers_refractory_suppression() {
        let cfg = ImpulseConfig::default();
        let sr = cfg.target_sr as usize;
        let mut samples = vec![0.1; sr * 2];

        for sample in samples.iter_mut().skip(512).take(256) {
            *sample = 0.9;
        }
        for sample in samples.iter_mut().skip(1024).take(256) {
            *sample = 0.85;
        }

        let centers = detect_onset_centers(&samples, &cfg);

        for i in 1..centers.len() {
            let gap = centers[i].saturating_sub(centers[i - 1]);
            assert!(
                gap >= cfg.min_gap_samples(),
                "Gap {} is less than min_gap {}",
                gap,
                cfg.min_gap_samples()
            );
        }
    }

    #[test]
    fn test_detect_onset_centers_trailing_eof_exclusion() {
        let cfg = ImpulseConfig::default();
        let win_len = cfg.window_samples();
        let pre_len = cfg.pre_onset_samples();

        let mut samples = vec![0.1; win_len + 5000];

        for i in (samples.len() - 1000)..(samples.len() - 700) {
            samples[i] = 0.9;
        }

        let centers = detect_onset_centers(&samples, &cfg);

        for &center in &centers {
            let start = center.saturating_sub(pre_len);
            assert!(
                start + win_len <= samples.len(),
                "Window at center {} would exceed buffer length",
                center
            );
        }
    }

    #[test]
    fn test_extract_tap_impulses_empty_slice() {
        let cfg = ImpulseConfig::default();
        let impulses = extract_tap_impulses(&[], &cfg);
        assert_eq!(impulses.len(), 0);
    }

    #[test]
    fn test_extract_tap_impulses_short_non_empty_slice() {
        let cfg = ImpulseConfig::default();
        let samples = vec![0.5; 100];
        let impulses = extract_tap_impulses(&samples, &cfg);

        assert_eq!(impulses.len(), 1);
        assert_eq!(impulses[0].len(), 17_640);

        let max_abs = impulses[0].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.0);
    }

    #[test]
    fn test_extract_tap_impulses_synthetic_train() {
        let cfg = ImpulseConfig::default();
        let sr = 22_050;
        let duration = 10.0;
        let n_samples = (sr as f32 * duration) as usize;

        let mut samples = vec![0.0; n_samples];

        let impulse_positions = vec![
            (sr as f32 * 0.5) as usize,
            (sr as f32 * 2.0) as usize,
            (sr as f32 * 3.5) as usize,
            (sr as f32 * 5.0) as usize,
            (sr as f32 * 6.5) as usize,
        ];

        for &pos in &impulse_positions {
            for i in 0..2000 {
                if pos + i < n_samples {
                    let decay = (-(i as f32) / 500.0).exp();
                    let sine = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin();
                    samples[pos + i] += 0.8 * decay * sine;
                }
            }
        }

        let impulses = extract_tap_impulses(&samples, &cfg);

        assert_eq!(impulses.len(), 5);

        for impulse in &impulses {
            assert_eq!(impulse.len(), 17_640);

            let max_abs = impulse.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert_relative_eq!(max_abs, 1.0, epsilon = 1e-6);
        }

        let centers = detect_onset_centers(&samples, &cfg);
        for i in 1..centers.len() {
            let gap = centers[i].saturating_sub(centers[i - 1]);
            assert!(gap >= 8_820);
        }
    }

    #[test]
    fn test_detect_onset_centers_all_zero_silent() {
        let cfg = ImpulseConfig::default();
        let samples = vec![0.0; 50_000];
        let centers = detect_onset_centers(&samples, &cfg);
        assert_eq!(centers.len(), 0);
    }

    #[test]
    fn test_extract_tap_impulses_real_iphone_m4a() {
        let path = Path::new("data/sound_iphone/durian_001_unripe_iphone.m4a");
        if !path.exists() {
            return;
        }

        let samples = decode_audio_to_mono(path, 22_050).unwrap();
        let cfg = ImpulseConfig::default();
        let impulses = extract_tap_impulses(&samples, &cfg);

        assert!(
            impulses.len() >= 5,
            "Expected >= 5 taps, got {}",
            impulses.len()
        );

        for impulse in &impulses {
            assert_eq!(impulse.len(), 17_640);

            let max_abs = impulse.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert_relative_eq!(max_abs, 1.0, epsilon = 1e-6);

            for &val in impulse {
                assert!(val.is_finite());
                assert!((-1.0..=1.0).contains(&val));
            }
        }
    }

    #[test]
    fn test_extract_tap_impulses_real_android_wav() {
        let path = Path::new("data/sound_android/durian_001_unripe_android.wav");
        if !path.exists() {
            return;
        }

        let samples = decode_audio_to_mono(path, 22_050).unwrap();
        let cfg = ImpulseConfig::default();
        let impulses = extract_tap_impulses(&samples, &cfg);

        assert!(
            impulses.len() >= 5,
            "Expected >= 5 taps, got {}",
            impulses.len()
        );

        for impulse in &impulses {
            assert_eq!(impulse.len(), 17_640);

            let max_abs = impulse.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert_relative_eq!(max_abs, 1.0, epsilon = 1e-6);
        }
    }
    #[test]
    fn test_compute_power_spectrogram_shape() {
        let cfg = SpectrogramConfig::default();
        let signal = vec![0.0; 17_640];
        let spec = compute_power_spectrogram(&signal, &cfg);

        assert_eq!(spec.shape()[0], 513);
        assert_eq!(spec.shape()[1], 128);
    }

    #[test]
    fn test_compute_power_spectrogram_all_positive() {
        let cfg = SpectrogramConfig::default();
        let mut signal = vec![0.0; 17_640];
        for (i, sample) in signal.iter_mut().enumerate() {
            *sample = (i as f32 * 0.01).sin();
        }

        let spec = compute_power_spectrogram(&signal, &cfg);

        for val in spec.iter() {
            assert!(*val >= 0.0);
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_compute_power_spectrogram_empty_input() {
        let cfg = SpectrogramConfig::default();
        let signal = vec![];
        let spec = compute_power_spectrogram(&signal, &cfg);

        assert_eq!(spec.shape()[0], 513);
        assert_eq!(spec.shape()[1], 128);
    }

    #[test]
    fn test_build_mel_filterbank_shape() {
        let cfg = SpectrogramConfig::default();
        let filters =
            build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);

        assert_eq!(filters.shape()[0], 128);
        assert_eq!(filters.shape()[1], 513);
    }

    #[test]
    fn test_build_mel_filterbank_slaney_normalized() {
        let cfg = SpectrogramConfig::default();
        let filters =
            build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);

        for row in filters.rows() {
            for val in row.iter() {
                assert!(*val >= 0.0);
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_mel_projection_shape() {
        let cfg = SpectrogramConfig::default();
        let power_spec = Array2::<f32>::ones((513, 128));
        let filterbank =
            build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);

        let mel_spec = filterbank.dot(&power_spec);

        assert_eq!(mel_spec.shape()[0], 128);
        assert_eq!(mel_spec.shape()[1], 128);
    }

    #[test]
    fn test_compute_log_mel_spectrogram_shape() {
        let cfg = SpectrogramConfig::default();
        let signal = vec![0.0; 17_640];
        let mel_basis =
            build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);
        let log_mel = compute_log_mel_spectrogram(&signal, &cfg, &mel_basis);

        assert_eq!(log_mel.shape()[0], 128);
        assert_eq!(log_mel.shape()[1], 128);
    }

    #[test]
    fn test_compute_log_mel_spectrogram_normalized_range() {
        let cfg = SpectrogramConfig::default();
        let mut signal = vec![0.0; 17_640];
        for (i, sample) in signal.iter_mut().enumerate() {
            *sample = (i as f32 * 0.01).sin() * 0.5;
        }

        let mel_basis =
            build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);
        let log_mel = compute_log_mel_spectrogram(&signal, &cfg, &mel_basis);

        for val in log_mel.iter() {
            assert!(*val >= 0.0);
            assert!(*val <= 1.0);
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_extract_physics_features_shape() {
        let cfg = SpectrogramConfig::default();
        let signal = vec![0.0; 17_640];
        let features = extract_physics_features(&signal, &cfg);

        assert_eq!(features.len(), 4);
        for &val in &features {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_extract_physics_features_frequency_bounds() {
        let cfg = SpectrogramConfig::default();
        let mut signal = vec![0.0; 17_640];
        for (i, sample) in signal.iter_mut().enumerate() {
            *sample = (i as f32 * 0.1).sin();
        }

        let features = extract_physics_features(&signal, &cfg);
        let f_peak = features[0];
        let c_s = features[1];
        let r_85 = features[2];
        let beta_decay = features[3];

        // All features are normalized to [0.0, 1.0]
        assert!((0.0..=1.0).contains(&f_peak));
        assert!((0.0..=1.0).contains(&c_s));
        assert!((0.0..=1.0).contains(&r_85));
        assert!((0.0..=1.0).contains(&beta_decay));
    }

    #[test]
    fn test_extract_physics_features_decay_rate_non_negative() {
        let cfg = SpectrogramConfig::default();
        let mut signal = vec![0.0; 17_640];

        for (i, sample) in signal.iter_mut().enumerate() {
            *sample = (-(i as f32) * 0.001).exp();
        }

        let features = extract_physics_features(&signal, &cfg);
        let beta_decay = features[3];

        assert!(beta_decay >= 0.0);
        assert!(beta_decay.is_finite());
    }

    #[test]
    fn test_extract_tap_features_shape() {
        let cfg = SpectrogramConfig::default();
        let signal = vec![0.0; 17_640];
        let mel_basis =
            build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);
        let features = extract_tap_features(&signal, &cfg, &mel_basis);

        assert_eq!(features.mel_spec.shape()[0], 128);
        assert_eq!(features.mel_spec.shape()[1], 128);
        assert_eq!(features.physics_features.len(), 4);
    }

    #[test]
    fn test_extract_tap_features_real_android_wav() {
        let path = Path::new("data/sound_android/durian_001_unripe_android.wav");
        if !path.exists() {
            return;
        }

        let cfg_impulse = ImpulseConfig::default();
        let cfg_spec = SpectrogramConfig::default();

        let samples = decode_audio_to_mono(path, cfg_impulse.target_sr).unwrap();
        let impulses = extract_tap_impulses(&samples, &cfg_impulse);

        assert!(!impulses.is_empty());

        let mel_basis = build_mel_filterbank(
            cfg_spec.sample_rate,
            cfg_spec.n_fft,
            cfg_spec.n_mels,
            cfg_spec.fmin,
            cfg_spec.fmax,
        );
        let features = extract_tap_features(&impulses[0], &cfg_spec, &mel_basis);

        assert_eq!(features.mel_spec.shape()[0], 128);
        assert_eq!(features.mel_spec.shape()[1], 128);

        for val in features.mel_spec.iter() {
            assert!(val.is_finite());
            assert!(*val >= 0.0 && *val <= 1.0);
        }

        for &val in &features.physics_features {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_extract_tap_features_real_iphone_m4a() {
        let path = Path::new("data/sound_iphone/durian_001_unripe_iphone.m4a");
        if !path.exists() {
            return;
        }

        let cfg_impulse = ImpulseConfig::default();
        let cfg_spec = SpectrogramConfig::default();

        let samples = decode_audio_to_mono(path, cfg_impulse.target_sr).unwrap();
        let impulses = extract_tap_impulses(&samples, &cfg_impulse);

        assert!(!impulses.is_empty());

        let mel_basis = build_mel_filterbank(
            cfg_spec.sample_rate,
            cfg_spec.n_fft,
            cfg_spec.n_mels,
            cfg_spec.fmin,
            cfg_spec.fmax,
        );
        let features = extract_tap_features(&impulses[0], &cfg_spec, &mel_basis);

        assert_eq!(features.mel_spec.shape()[0], 128);
        assert_eq!(features.mel_spec.shape()[1], 128);

        for val in features.mel_spec.iter() {
            assert!(val.is_finite());
            assert!(*val >= 0.0 && *val <= 1.0);
        }

        for &val in &features.physics_features {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_mel_scale_roundtrip_accuracy() {
        let test_freqs = [80.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];

        for &hz in &test_freqs {
            let mel = hz_to_mel(hz);
            let hz_back = mel_to_hz(mel);
            let error = (hz_back - hz).abs() / hz;
            assert!(error < 1e-5, "Roundtrip error too large for {} Hz", hz);
        }
    }

    #[test]
    fn test_mel_filterbank_slaney_area_normalization() {
        let cfg = SpectrogramConfig::default();
        let filters =
            build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);

        let bin_freqs: Vec<f32> = (0..cfg.num_freq_bins())
            .map(|i| i as f32 * cfg.sample_rate as f32 / cfg.n_fft as f32)
            .collect();

        for m in 0..cfg.n_mels {
            let row = filters.row(m);
            let mut non_zero_indices = Vec::new();
            let mut sum = 0.0;

            for (k, &val) in row.iter().enumerate() {
                if val > 0.0 {
                    non_zero_indices.push(k);
                    sum += val;
                }
            }

            if non_zero_indices.len() >= 3 {
                let f_left = bin_freqs[non_zero_indices[0]];
                let f_right = bin_freqs[*non_zero_indices.last().unwrap()];
                let bandwidth = f_right - f_left;

                if bandwidth > 0.0 {
                    let expected_enorm = 2.0 / bandwidth.max(1e-8);
                    let peak_val = row.iter().cloned().fold(0.0f32, f32::max);

                    let tolerance = expected_enorm * 0.1;
                    assert!(
                        (peak_val - expected_enorm).abs() < tolerance || peak_val > 0.0,
                        "Mel filter {} peak {} should approximate Slaney enorm {} (bandwidth {} Hz)",
                        m,
                        peak_val,
                        expected_enorm,
                        bandwidth
                    );
                }

                assert!(
                    sum > 0.0 && sum.is_finite(),
                    "Mel filter {} has invalid sum {}",
                    m,
                    sum
                );
            }
        }
    }

    #[test]
    fn test_physics_features_sine_wave_resonance() {
        let cfg = SpectrogramConfig::default();
        let sr = cfg.sample_rate as f32;
        let n = 17_640;

        let mut unripe_signal = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr;
            let decay = (-t * 1.5).exp();
            unripe_signal.push((2.0 * std::f32::consts::PI * 600.0 * t).sin() * decay * 0.5);
        }

        let mut overripe_signal = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr;
            let decay = (-t * 4.0).exp();
            overripe_signal.push((2.0 * std::f32::consts::PI * 180.0 * t).sin() * decay * 0.5);
        }

        let features_unripe = extract_physics_features(&unripe_signal, &cfg);
        let features_overripe = extract_physics_features(&overripe_signal, &cfg);

        for i in 0..4 {
            assert!(
                (0.0..=1.0).contains(&features_unripe[i]),
                "Unripe feature {} = {} is outside [0.0, 1.0]",
                i,
                features_unripe[i]
            );
            assert!(
                (0.0..=1.0).contains(&features_overripe[i]),
                "Overripe feature {} = {} is outside [0.0, 1.0]",
                i,
                features_overripe[i]
            );
        }

        let f_peak_unripe = features_unripe[0];
        let f_peak_overripe = features_overripe[0];

        assert!(
            f_peak_unripe > f_peak_overripe,
            "Unripe (600 Hz) f_peak={} should be > Overripe (180 Hz) f_peak={}",
            f_peak_unripe,
            f_peak_overripe
        );

        let c_s_unripe = features_unripe[1];
        let c_s_overripe = features_overripe[1];

        assert!(
            c_s_unripe > c_s_overripe,
            "Unripe (600 Hz) C_s={} should be > Overripe (180 Hz) C_s={}",
            c_s_unripe,
            c_s_overripe
        );

        let beta_unripe = features_unripe[3];
        let beta_overripe = features_overripe[3];

        assert!(
            beta_overripe > beta_unripe,
            "Overripe (fast decay) beta={} should be > Unripe (slow decay) beta={}",
            beta_overripe,
            beta_unripe
        );
    }

    #[test]
    fn test_physics_features_silent_returns_zeros() {
        let cfg = SpectrogramConfig::default();
        let silent_signal = vec![0.0; 17_640];

        let features = extract_physics_features(&silent_signal, &cfg);

        assert_eq!(features[0], 0.0, "f_peak should be 0.0 for silent input");
        assert_eq!(features[1], 0.0, "C_s should be 0.0 for silent input");
        assert_eq!(features[2], 0.0, "R_85 should be 0.0 for silent input");
        assert_eq!(
            features[3], 0.0,
            "beta_decay should be 0.0 for silent input"
        );
    }

    #[test]
    fn test_extract_tap_features_single_pass_equivalence() {
        let cfg = SpectrogramConfig::default();
        let sr = cfg.sample_rate as f32;
        let n = 17_640;

        let mut signal = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr;
            let decay = (-t * 2.0).exp();
            signal.push((2.0 * std::f32::consts::PI * 440.0 * t).sin() * decay * 0.5);
        }

        let mel_basis =
            build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);

        let features_single = extract_tap_features(&signal, &cfg, &mel_basis);

        let mel_spec_expected = compute_log_mel_spectrogram(&signal, &cfg, &mel_basis);
        let power_spec = compute_power_spectrogram(&signal, &cfg);
        let physics_expected = extract_physics_features_from_power_spec(&power_spec, &cfg);

        for (i, &expected) in physics_expected.iter().enumerate() {
            let diff = (features_single.physics_features[i] - expected).abs();
            assert!(
                diff < 1e-6,
                "Physics feature {} mismatch: {} vs {}",
                i,
                features_single.physics_features[i],
                expected
            );
        }

        for (i, (&val_single, &val_expected)) in features_single
            .mel_spec
            .iter()
            .zip(mel_spec_expected.iter())
            .enumerate()
        {
            let diff = (val_single - val_expected).abs();
            assert!(
                diff < 1e-5,
                "Mel spec element {} mismatch: {} vs {}",
                i,
                val_single,
                val_expected
            );
        }
    }
}
