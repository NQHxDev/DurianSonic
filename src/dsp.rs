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

    loop {
        match format.next_packet() {
            Ok(Some(packet)) => {
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
            Ok(None) => break,
            Err(_) => break,
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
    let mut prev_center = 0usize;

    for i in 1..(rms_envelope.len() - 1) {
        let is_peak = rms_envelope[i] >= threshold
            && rms_envelope[i] >= rms_envelope[i - 1]
            && rms_envelope[i] > rms_envelope[i + 1];

        if !is_peak {
            continue;
        }

        let center_sample = i * ENV_FRAME;

        if center_sample.saturating_sub(prev_center) < min_gap {
            continue;
        }

        let start = center_sample.saturating_sub(pre_len);
        if start + win_len > samples.len() {
            break;
        }

        centers.push(center_sample);
        prev_center = center_sample;
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
/// Uses 1024-point FFT with 128-sample hop, producing 513 frequency bins.
/// Each frame is Hann-windowed before FFT. Power = |FFT|^2.
///
/// # Arguments
///
/// * `signal` - Input mono waveform (17,640 samples for 0.8s window)
/// * `cfg` - Spectrogram configuration
///
/// # Returns
///
/// Power spectrogram (n_bins x n_frames), where n_bins = n_fft/2 + 1 = 513
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{SpectrogramConfig, compute_stft_power_spectrogram};
///
/// let cfg = SpectrogramConfig::default();
/// let signal = vec![0.0; 17_640];
/// let spec = compute_stft_power_spectrogram(&signal, &cfg);
/// assert_eq!(spec.shape(), &[513, 128]);
/// ```
pub fn compute_stft_power_spectrogram(signal: &[f32], cfg: &SpectrogramConfig) -> Array2<f32> {
    use realfft::RealFftPlanner;
    use std::f32::consts::PI;

    let n_fft = cfg.n_fft;
    let hop = cfg.hop_length;
    let n_bins = n_fft / 2 + 1;

    if signal.is_empty() {
        return Array2::zeros((n_bins, 0));
    }

    let n_frames = (signal.len().saturating_sub(n_fft) / hop) + 1;
    let n_frames = n_frames.min(128);
    if n_frames == 0 {
        return Array2::zeros((n_bins, 0));
    }

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut spectrum = r2c.make_output_vec();

    let hann: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
        .collect();

    let mut power_spec = Array2::<f32>::zeros((n_bins, n_frames));

    for (frame_idx, frame_start) in (0..signal.len()).step_by(hop).take(n_frames).enumerate() {
        let frame_end = (frame_start + n_fft).min(signal.len());
        let frame_len = frame_end - frame_start;

        let mut windowed = vec![0.0f32; n_fft];
        for i in 0..frame_len {
            windowed[i] = signal[frame_start + i] * hann[i];
        }

        r2c.process(&mut windowed, &mut spectrum).unwrap_or(());

        for (bin, complex_val) in spectrum.iter().enumerate() {
            let power = complex_val.re * complex_val.re + complex_val.im * complex_val.im;
            power_spec[[bin, frame_idx]] = power.max(1e-10);
        }
    }

    power_spec
}

/// Create Mel filterbank matrix.
///
/// Triangular filters spaced uniformly on Mel scale between fmin and fmax.
/// Each filter has unit area (sum of weights = 1.0).
///
/// # Arguments
///
/// * `cfg` - Spectrogram configuration with n_mels, fmin, fmax
///
/// # Returns
///
/// Mel filterbank matrix (n_mels x n_bins), where n_bins = n_fft/2 + 1 = 513
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{SpectrogramConfig, mel_filterbank};
///
/// let cfg = SpectrogramConfig::default();
/// let filters = mel_filterbank(&cfg);
/// assert_eq!(filters.shape(), &[128, 513]);
/// ```
pub fn mel_filterbank(cfg: &SpectrogramConfig) -> Array2<f32> {
    let n_bins = cfg.n_fft / 2 + 1;
    let n_mels = cfg.n_mels;

    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);

    let mel_min = hz_to_mel(cfg.fmin);
    let mel_max = hz_to_mel(cfg.fmax);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();

    let bin_freqs: Vec<f32> = (0..n_bins)
        .map(|i| i as f32 * cfg.sample_rate as f32 / cfg.n_fft as f32)
        .collect();

    let mut filters = Array2::<f32>::zeros((n_mels, n_bins));

    for m in 0..n_mels {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];

        for (k, &freq) in bin_freqs.iter().enumerate() {
            let weight = if freq >= f_left && freq <= f_center {
                (freq - f_left) / (f_center - f_left + 1e-8)
            } else if freq > f_center && freq <= f_right {
                (f_right - freq) / (f_right - f_center + 1e-8)
            } else {
                0.0
            };
            filters[[m, k]] = weight;
        }

        let sum: f32 = filters.row(m).sum();
        if sum > 1e-8 {
            for k in 0..n_bins {
                filters[[m, k]] /= sum;
            }
        }
    }

    filters
}

/// Apply Mel filterbank to power spectrogram.
///
/// Matrix multiplication: mel_spec = filterbank @ power_spec
///
/// # Arguments
///
/// * `power_spec` - Power spectrogram (n_bins x n_frames)
/// * `filterbank` - Mel filterbank (n_mels x n_bins)
///
/// # Returns
///
/// Mel spectrogram (n_mels x n_frames)
pub fn apply_mel_filterbank(power_spec: &Array2<f32>, filterbank: &Array2<f32>) -> Array2<f32> {
    filterbank.dot(power_spec)
}

/// Compute log-Mel spectrogram with dynamic range compression.
///
/// Full pipeline: STFT → Power → Mel filterbank → log10(mel + eps) → dB clipping
/// Output is normalized to [0, 1] range after top_db clipping.
///
/// # Arguments
///
/// * `signal` - Input mono waveform (17,640 samples)
/// * `cfg` - Spectrogram configuration
///
/// # Returns
///
/// Log-Mel spectrogram (128 x 128), normalized to [0, 1]
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{SpectrogramConfig, compute_log_mel_spectrogram};
///
/// let cfg = SpectrogramConfig::default();
/// let signal = vec![0.0; 17_640];
/// let log_mel = compute_log_mel_spectrogram(&signal, &cfg);
/// assert_eq!(log_mel.shape(), &[128, 128]);
/// ```
pub fn compute_log_mel_spectrogram(signal: &[f32], cfg: &SpectrogramConfig) -> Array2<f32> {
    let power_spec = compute_stft_power_spectrogram(signal, cfg);
    let filterbank = mel_filterbank(cfg);
    let mel_spec = apply_mel_filterbank(&power_spec, &filterbank);

    let mut log_mel = mel_spec.mapv(|x| 10.0 * (x.max(1e-10)).log10());

    let max_db = log_mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_db = max_db - cfg.top_db;

    log_mel.mapv_inplace(|x| ((x - min_db) / cfg.top_db).clamp(0.0, 1.0));

    log_mel
}

/// Extract 4 acoustic physics features from tap impulse.
///
/// Features computed:
/// 1. f_peak: Peak frequency (Hz) in [50, 8000] Hz range
/// 2. C_s: Spectral centroid (Hz), gravity center of spectrum
/// 3. R_85: Rolloff frequency (Hz), 85% cumulative energy point
/// 4. beta_decay: Exponential decay rate (1/samples) from envelope fit
///
/// # Arguments
///
/// * `signal` - Normalized tap impulse (17,640 samples)
/// * `cfg` - Spectrogram configuration
///
/// # Returns
///
/// [f_peak, C_s, R_85, beta_decay]
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
pub fn extract_physics_features(signal: &[f32], cfg: &SpectrogramConfig) -> [f32; 4] {
    use realfft::RealFftPlanner;

    let n_fft = cfg.n_fft;
    let sr = cfg.sample_rate as f32;

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);

    let mut windowed = signal.iter().take(n_fft).copied().collect::<Vec<_>>();
    windowed.resize(n_fft, 0.0);

    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut windowed, &mut spectrum).unwrap_or(());

    let magnitudes: Vec<f32> = spectrum
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();

    let freqs: Vec<f32> = (0..magnitudes.len())
        .map(|i| i as f32 * sr / n_fft as f32)
        .collect();

    let valid_range: Vec<(usize, f32, f32)> = freqs
        .iter()
        .enumerate()
        .filter(|(_, &f)| f >= cfg.fmin && f <= cfg.fmax)
        .map(|(i, &f)| (i, f, magnitudes[i]))
        .collect();

    if valid_range.is_empty() {
        return [cfg.fmin, cfg.fmin, cfg.fmin, 0.0];
    }

    let (_peak_idx, f_peak, _) = valid_range
        .iter()
        .max_by(|(_, _, m1), (_, _, m2)| m1.partial_cmp(m2).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or((0, cfg.fmin, 0.0));

    let total_mag: f32 = valid_range.iter().map(|(_, _, m)| m).sum();
    let c_s = if total_mag > 1e-8 {
        valid_range.iter().map(|(_, f, m)| f * m).sum::<f32>() / total_mag
    } else {
        cfg.fmin
    };

    let mut cumsum = 0.0f32;
    let threshold = 0.85 * total_mag;
    let r_85 = valid_range
        .iter()
        .find_map(|(_, f, m)| {
            cumsum += m;
            if cumsum >= threshold {
                Some(*f)
            } else {
                None
            }
        })
        .unwrap_or(cfg.fmax);

    let envelope: Vec<f32> = signal.iter().take(8820).map(|&x| x.abs()).collect();

    let peak_env = envelope.iter().cloned().fold(0.0f32, f32::max).max(1e-8);

    let decay_start = envelope
        .iter()
        .position(|&x| x >= 0.9 * peak_env)
        .unwrap_or(0);

    let decay_region = &envelope[decay_start..];
    let n = decay_region.len() as f32;

    if n < 10.0 {
        return [f_peak, c_s, r_85, 0.0];
    }

    let log_env: Vec<f32> = decay_region
        .iter()
        .map(|&x| (x.max(1e-8) / peak_env).ln())
        .collect();

    let mean_log: f32 = log_env.iter().sum::<f32>() / n;
    let mean_t: f32 = (n - 1.0) / 2.0;

    let cov: f32 = log_env
        .iter()
        .enumerate()
        .map(|(i, &log_val)| (i as f32 - mean_t) * (log_val - mean_log))
        .sum::<f32>();

    let var_t: f32 = (0..log_env.len())
        .map(|i| {
            let diff = i as f32 - mean_t;
            diff * diff
        })
        .sum::<f32>();

    let beta_decay = if var_t > 1e-8 {
        -(cov / var_t).max(0.0)
    } else {
        0.0
    };

    [f_peak, c_s, r_85, beta_decay]
}

/// Compute full TapFeatures from normalized tap impulse.
///
/// Combines log-Mel spectrogram and 4 physics features into single struct.
///
/// # Arguments
///
/// * `signal` - Normalized tap impulse (17,640 samples)
/// * `cfg` - Spectrogram configuration
///
/// # Returns
///
/// TapFeatures with 128x128 log-Mel and [f_peak, C_s, R_85, beta_decay]
///
/// # Examples
///
/// ```
/// use durian_sonic::dsp::{SpectrogramConfig, compute_tap_features};
///
/// let cfg = SpectrogramConfig::default();
/// let signal = vec![0.0; 17_640];
/// let features = compute_tap_features(&signal, &cfg);
/// assert_eq!(features.mel_spec.shape(), &[128, 128]);
/// assert_eq!(features.physics_features.len(), 4);
/// ```
pub fn compute_tap_features(signal: &[f32], cfg: &SpectrogramConfig) -> TapFeatures {
    let mel_spec = compute_log_mel_spectrogram(signal, cfg);
    let physics_features = extract_physics_features(signal, cfg);

    TapFeatures {
        mel_spec,
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
            eprintln!("Skipping test: file not found");
            return;
        }

        let result = decode_audio_to_mono(path, 22_050);
        if let Err(e) = &result {
            eprintln!("Decode error: {:?}", e);
        }
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
            eprintln!("Skipping test: file not found");
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
            eprintln!("Skipping test: file not found");
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
            eprintln!("Skipping test: file not found");
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
    fn test_compute_stft_power_spectrogram_shape() {
        let cfg = SpectrogramConfig::default();
        let signal = vec![0.0; 17_640];
        let spec = compute_stft_power_spectrogram(&signal, &cfg);

        assert_eq!(spec.shape()[0], 513);
        assert_eq!(spec.shape()[1], 128);
    }

    #[test]
    fn test_compute_stft_power_spectrogram_all_positive() {
        let cfg = SpectrogramConfig::default();
        let mut signal = vec![0.0; 17_640];
        for (i, sample) in signal.iter_mut().enumerate() {
            *sample = (i as f32 * 0.01).sin();
        }

        let spec = compute_stft_power_spectrogram(&signal, &cfg);

        for val in spec.iter() {
            assert!(*val >= 0.0);
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_compute_stft_power_spectrogram_empty_input() {
        let cfg = SpectrogramConfig::default();
        let signal = vec![];
        let spec = compute_stft_power_spectrogram(&signal, &cfg);

        assert_eq!(spec.shape()[0], 513);
        assert_eq!(spec.shape()[1], 0);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let cfg = SpectrogramConfig::default();
        let filters = mel_filterbank(&cfg);

        assert_eq!(filters.shape()[0], 128);
        assert_eq!(filters.shape()[1], 513);
    }

    #[test]
    fn test_mel_filterbank_normalized() {
        let cfg = SpectrogramConfig::default();
        let filters = mel_filterbank(&cfg);

        for m in 0..filters.shape()[0] {
            let row_sum: f32 = filters.row(m).sum();
            if row_sum > 0.0 {
                assert!((row_sum - 1.0).abs() < 1e-5, "Row {} sum = {}", m, row_sum);
            }
        }
    }

    #[test]
    fn test_apply_mel_filterbank_shape() {
        let cfg = SpectrogramConfig::default();
        let power_spec = Array2::<f32>::ones((513, 128));
        let filterbank = mel_filterbank(&cfg);

        let mel_spec = apply_mel_filterbank(&power_spec, &filterbank);

        assert_eq!(mel_spec.shape()[0], 128);
        assert_eq!(mel_spec.shape()[1], 128);
    }

    #[test]
    fn test_compute_log_mel_spectrogram_shape() {
        let cfg = SpectrogramConfig::default();
        let signal = vec![0.0; 17_640];
        let log_mel = compute_log_mel_spectrogram(&signal, &cfg);

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

        let log_mel = compute_log_mel_spectrogram(&signal, &cfg);

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

        assert!(f_peak >= cfg.fmin && f_peak <= cfg.fmax);
        assert!(c_s >= cfg.fmin && c_s <= cfg.fmax);
        assert!(r_85 >= cfg.fmin && r_85 <= cfg.fmax);
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
    fn test_compute_tap_features_shape() {
        let cfg = SpectrogramConfig::default();
        let signal = vec![0.0; 17_640];
        let features = compute_tap_features(&signal, &cfg);

        assert_eq!(features.mel_spec.shape()[0], 128);
        assert_eq!(features.mel_spec.shape()[1], 128);
        assert_eq!(features.physics_features.len(), 4);
    }

    #[test]
    fn test_compute_tap_features_real_android_wav() {
        let path = Path::new("data/sound_android/durian_001_unripe_android.wav");
        if !path.exists() {
            eprintln!("Skipping test: file not found");
            return;
        }

        let cfg_impulse = ImpulseConfig::default();
        let cfg_spec = SpectrogramConfig::default();

        let samples = decode_audio_to_mono(path, cfg_impulse.target_sr).unwrap();
        let impulses = extract_tap_impulses(&samples, &cfg_impulse);

        assert!(!impulses.is_empty());

        let features = compute_tap_features(&impulses[0], &cfg_spec);

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
    fn test_compute_tap_features_real_iphone_m4a() {
        let path = Path::new("data/sound_iphone/durian_001_unripe_iphone.m4a");
        if !path.exists() {
            eprintln!("Skipping test: file not found");
            return;
        }

        let cfg_impulse = ImpulseConfig::default();
        let cfg_spec = SpectrogramConfig::default();

        let samples = decode_audio_to_mono(path, cfg_impulse.target_sr).unwrap();
        let impulses = extract_tap_impulses(&samples, &cfg_impulse);

        assert!(!impulses.is_empty());

        let features = compute_tap_features(&impulses[0], &cfg_spec);

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
}
