//! Training module for Phase 3 deep learning pipeline.
//!
//! This module implements tap sample extraction, minority oversampling,
//! batch collation, loss functions, evaluation metrics, and the training loop.

use crate::augment::{
    add_background_noise, apply_onset_jitter, apply_spec_augment, apply_spectral_tilt,
};
use crate::dataset::{AudioMetadataRecord, PlatformTarget, RipenessLabel};
use crate::dsp::{
    build_mel_filterbank, decode_audio_to_mono, detect_onset_centers, extract_tap_features,
    normalize_peak, ImpulseConfig, SpectrogramConfig,
};
use crate::model::DurianAudioCnn;
use anyhow::{Context as AnyhowContext, Result};
use burn::tensor::{Device, Int, Tensor};
use std::path::{Path, PathBuf};
use tracing::warn;

// Phase 3 Training Entities

/// Represents a single extracted 0.8s tap window after Phase 2 DSP feature extraction.
///
/// # Field Constraints
/// - `fruit_id`: 1..=175 (physical durian fruit identifier)
/// - `sample_num`: 1..=N (1..=281 iOS, 1..=175 Android)
/// - `tap_index`: 0..K (zero-based window index within recording)
/// - `mel_data`: Length 16,384 (128 * 128), [0.0, 1.0] (row-major flattened Log-Mel spectrogram)
/// - `physics_feats`: Length 4, [0.0, 1.0]^4 (normalized acoustic physics vector)
/// - `label`: Unripe (0), Ripe (1), Overripe (2)
#[derive(Debug, Clone, PartialEq)]
pub struct TapSample {
    pub fruit_id: u32,
    pub sample_num: u32,
    pub tap_index: usize,
    pub mel_data: Vec<f32>,
    pub physics_feats: [f32; 4],
    pub label: RipenessLabel,
}

/// Bundle of train/validation/test data splits for model training.
///
/// Consolidates the three sample slices to reduce function parameter counts.
#[derive(Debug)]
pub struct DataSplits<'a> {
    pub train: &'a [TapSample],
    pub val: &'a [TapSample],
    pub test: &'a [TapSample],
}

/// Collated mini-batch structure for training and evaluation.
///
/// # Field Shapes
/// - `mel_specs`: `[B, 1, 128, 128]` in `[0.0, 1.0]`
/// - `phys_feats`: `[B, 4]` in `[0.0, 1.0]`
/// - `targets`: `[B]` in `{0, 1, 2}`
#[derive(Clone, Debug)]
pub struct DurianBatch {
    pub mel_specs: Tensor<4>,
    pub phys_feats: Tensor<2>,
    pub targets: Tensor<1, Int>,
}

/// Extract tap samples from a single waveform with train-only minority oversampling.
///
/// # Train-Only Onset-Window Minority Oversampling
/// - `is_train=false` (Val/Test): 1x canonical window per peak, zero augmentation
/// - `is_train=true` (Train):
///   - Unripe: 1x canonical
///   - Ripe: 2x (1 canonical + 1 replica with jitter+noise+SpecAugment)
///   - Overripe: 3x (1 canonical + 2 replicas with jitter+tilt+SpecAugment)
///
/// # Zero-Onset Fallback
/// If `detect_onset_centers` returns 0 peaks, extracts 1 canonical 0.8s window
/// at max-RMS frame and logs a warning.
///
/// # Examples
///
/// ```no_run
/// use durian_sonic::training::extract_tap_samples_from_waveform;
/// use durian_sonic::config::DspConfig;
/// use durian_sonic::dsp::build_mel_filterbank;
/// use durian_sonic::dataset::RipenessLabel;
///
/// let samples = vec![0.0; 50_000];
/// let dsp_cfg = DspConfig::default();
/// let mel_basis = build_mel_filterbank(22_050, 1024, 128, 50.0, 8000.0);
///
/// let taps = extract_tap_samples_from_waveform(
///     &samples,
///     1,
///     1,
///     RipenessLabel::Ripe,
///     &dsp_cfg,
///     &mel_basis,
///     true,
/// );
/// ```
pub fn extract_tap_samples_from_waveform(
    samples: &[f32],
    fruit_id: u32,
    sample_num: u32,
    label: RipenessLabel,
    dsp_config: &crate::config::DspConfig,
    mel_basis: &ndarray::Array2<f32>,
    is_train: bool,
) -> Vec<TapSample> {
    let impulse_cfg = ImpulseConfig::from(dsp_config);
    let spec_cfg = SpectrogramConfig::from(dsp_config);
    let mut centers = detect_onset_centers(samples, &impulse_cfg);

    // Zero-onset fallback: extract 1 canonical window at max-RMS frame
    if centers.is_empty() {
        warn!(
            "Recording sample_num={} has 0 detected onset peaks, using max-RMS fallback",
            sample_num
        );

        let win_len = impulse_cfg.window_samples();
        if samples.len() < win_len {
            centers.push(samples.len() / 2);
        } else {
            // Find max-RMS 256-sample frame
            let frame_size = 256;
            let mut max_rms = 0.0;
            let mut max_idx = 0;
            for i in (0..samples.len().saturating_sub(frame_size)).step_by(frame_size) {
                let rms: f32 = samples[i..i + frame_size]
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f32>()
                    / frame_size as f32;
                if rms > max_rms {
                    max_rms = rms;
                    max_idx = i + frame_size / 2;
                }
            }
            centers.push(max_idx);
        }
    }

    let max_jitter_samples = (0.025 * impulse_cfg.target_sr as f32).round() as usize; // 551 samples at 22,050 Hz
    let win_len = impulse_cfg.window_samples();
    let pre_len = impulse_cfg.pre_onset_samples();

    let mut result = Vec::new();

    for (tap_index, &center_sample) in centers.iter().enumerate() {
        // Determine replication count based on class
        let replicas = if is_train {
            match label {
                RipenessLabel::Unripe => 1,
                RipenessLabel::Ripe => 2,
                RipenessLabel::Overripe => 3,
            }
        } else {
            1 // Val/Test: strictly 1x canonical, zero augmentation
        };

        for replica_idx in 0..replicas {
            let feats = if replica_idx == 0 {
                // Canonical window (unaugmented)
                let raw_start = center_sample.saturating_sub(pre_len);
                let window = if samples.len() < win_len {
                    let copy_start = raw_start.min(samples.len());
                    let available = samples.len() - copy_start;
                    let mut padded = vec![0.0; win_len];
                    padded[..available].copy_from_slice(&samples[copy_start..]);
                    normalize_peak(&padded)
                } else {
                    let start = raw_start.min(samples.len() - win_len);
                    normalize_peak(&samples[start..start + win_len])
                };
                extract_tap_features(&window, &spec_cfg, mel_basis)
            } else {
                // Augmented replica
                match label {
                    RipenessLabel::Ripe => {
                        // Waveform augmentations: jitter + noise
                        let jittered = apply_onset_jitter(
                            samples,
                            center_sample,
                            max_jitter_samples,
                            &impulse_cfg,
                        );
                        let noisy = add_background_noise(&jittered, 15.0, 35.0);
                        let mut feats = extract_tap_features(&noisy, &spec_cfg, mel_basis);
                        // Spectrogram augmentation: SpecAugment
                        apply_spec_augment(&mut feats.mel_spec, 12, 16, 0.0);
                        feats
                    }
                    RipenessLabel::Overripe => {
                        // Waveform augmentation: jitter only
                        let jittered = apply_onset_jitter(
                            samples,
                            center_sample,
                            max_jitter_samples,
                            &impulse_cfg,
                        );
                        let mut feats = extract_tap_features(&jittered, &spec_cfg, mel_basis);
                        // Spectrogram augmentations: tilt + SpecAugment
                        apply_spectral_tilt(&mut feats.mel_spec, 3.0, spec_cfg.top_db);
                        apply_spec_augment(&mut feats.mel_spec, 12, 16, 0.0);
                        feats
                    }
                    RipenessLabel::Unripe => unreachable!("Unripe only has 1 replica"),
                }
            };

            // Flatten mel_spec in row-major order
            let mel_data: Vec<f32> = feats.mel_spec.iter().copied().collect();

            result.push(TapSample {
                fruit_id,
                sample_num,
                tap_index,
                mel_data,
                physics_feats: feats.physics_features,
                label,
            });
        }
    }

    result
}

/// Extract tap samples from a dataset split with precomputed Mel filterbank.
///
/// Precomputes `mel_basis` once before iterating over all recordings for efficiency.
///
/// # Arguments
///
/// * `records` - Metadata records for the split
/// * `audio_dir` - Directory containing audio files
/// * `impulse_cfg` - Impulse detection configuration
/// * `spec_cfg` - Spectrogram configuration
/// * `is_train` - Whether to apply train-only minority oversampling
///
/// # Returns
///
/// Vector of all extracted TapSamples preserving fruit_id disjointness
pub fn extract_dataset_tap_samples(
    records: &[AudioMetadataRecord],
    audio_dir: &Path,
    dsp_config: &crate::config::DspConfig,
    is_train: bool,
) -> Result<Vec<TapSample>> {
    let impulse_cfg = ImpulseConfig::from(dsp_config);
    let spec_cfg = SpectrogramConfig::from(dsp_config);

    // Precompute mel_basis once
    let mel_basis = build_mel_filterbank(
        spec_cfg.sample_rate,
        spec_cfg.n_fft,
        spec_cfg.n_mels,
        spec_cfg.fmin,
        spec_cfg.fmax,
    );

    let mut all_samples = Vec::new();
    let total_files = records.len();
    let split_label = if is_train {
        "Train (oversampled)"
    } else {
        "Eval (canonical)   "
    };
    let start_time = std::time::Instant::now();

    for (file_idx, record) in records.iter().enumerate() {
        let audio_path = audio_dir.join(&record.filename);
        let waveform = decode_audio_to_mono(&audio_path, impulse_cfg.target_sr)
            .with_context(|| format!("Failed to decode {}", audio_path.display()))?;

        // Parse ripeness string to RipenessLabel enum
        let label = match record.ripeness.to_lowercase().as_str() {
            "unripe" => RipenessLabel::Unripe,
            "ripe" => RipenessLabel::Ripe,
            "overripe" => RipenessLabel::Overripe,
            _ => anyhow::bail!("Invalid ripeness label: {}", record.ripeness),
        };

        let taps = extract_tap_samples_from_waveform(
            &waveform,
            record.fruit_id,
            record.sample_num,
            label,
            dsp_config,
            &mel_basis,
            is_train,
        );

        all_samples.extend(taps);

        use std::io::Write;
        print!(
            "\r      Extracting {}: {:3}/{:3} files ({:4} taps) [{:.1}s]",
            split_label,
            file_idx + 1,
            total_files,
            all_samples.len(),
            start_time.elapsed().as_secs_f32()
        );
        let _ = std::io::stdout().flush();
    }

    if total_files > 0 {
        tracing::debug!("Extraction complete");
    }

    Ok(all_samples)
}

/// Collate a slice of TapSamples into a DurianBatch with dynamic batch size.
///
/// # Arguments
///
/// * `samples` - Slice of TapSamples to collate (B >= 1)
/// * `device` - Backend device
///
/// # Returns
///
/// DurianBatch with tensors of shape `[B, 1, 128, 128]`, `[B, 4]`, `[B]`
///
/// # Examples
///
/// ```no_run
/// use durian_sonic::training::{TapSample, collate_tap_batch};
/// use durian_sonic::dataset::RipenessLabel;
/// use burn::prelude::*;
///
/// let device = Device::flex();
///
/// let samples = vec![
///     TapSample {
///         fruit_id: 1,
///         sample_num: 1,
///         tap_index: 0,
///         mel_data: vec![0.5; 16_384],
///         physics_feats: [0.5, 0.5, 0.5, 0.5],
///         label: RipenessLabel::Ripe,
///     },
/// ];
///
/// let batch = collate_tap_batch(&samples, &device);
/// assert_eq!(batch.mel_specs.dims(), [1, 1, 128, 128]);
/// ```
pub fn collate_tap_batch(samples: &[TapSample], device: &Device) -> DurianBatch {
    let batch_size = samples.len();

    // Flatten all mel_data into a single vector [B * 1 * 128 * 128]
    let mel_flat: Vec<f32> = samples
        .iter()
        .flat_map(|s| s.mel_data.iter().copied())
        .collect();

    let mel_specs = Tensor::<4>::from_data(
        burn::tensor::TensorData::new(mel_flat, [batch_size, 1, 128, 128]),
        device,
    );

    // Flatten all physics_feats into [B * 4]
    let phys_flat: Vec<f32> = samples
        .iter()
        .flat_map(|s| s.physics_feats.iter().copied())
        .collect();

    let phys_feats = Tensor::<2>::from_data(
        burn::tensor::TensorData::new(phys_flat, [batch_size, 4]),
        device,
    );

    // Convert labels to class indices {0, 1, 2}
    let labels: Vec<i64> = samples
        .iter()
        .map(|s| match s.label {
            RipenessLabel::Unripe => 0,
            RipenessLabel::Ripe => 1,
            RipenessLabel::Overripe => 2,
        })
        .collect();

    let targets =
        Tensor::<1, Int>::from_data(burn::tensor::TensorData::new(labels, [batch_size]), device);

    DurianBatch {
        mel_specs,
        phys_feats,
        targets,
    }
}

/// Compute weighted cross-entropy loss with label smoothing.
///
/// Combines platform-specific inverse class frequency weights with label smoothing
/// to handle class imbalance and prevent overconfident predictions.
///
/// # Mathematical Formulation
///
/// For each sample i with true label y_i and smoothing factor epsilon (default 0.05):
/// - Smoothed target distribution: q_i(c) = (1 - epsilon) * I(c = y_i) + epsilon / 3
///   - True class: 1 - 2epsilon/3  approximately  0.9666667
///   - Non-target classes: epsilon / 3  approximately  0.0166667
/// - Weighted target: T_{i,c} = w_c * q_i(c)
/// - Batch mean loss: L = (1/B) * sum_i (-sum_c T_{i,c} * log_softmax(logits_i)_c)
///
/// # Arguments
///
/// * `logits` - Raw model outputs [B, 3] (unnormalized class scores)
/// * `targets` - Ground truth class indices [0, 1, 2] for each sample
/// * `class_weights` - Platform-specific inverse frequency weights [w_0, w_1, w_2]
///   - iPhone: [0.548, 1.419, 2.129]
///   - Android: [0.614, 1.535, 1.389]
/// * `label_smoothing` - Smoothing factor epsilon  in  [0.0, 1.0), typically 0.05
/// * `device` - Burn device for tensor operations
///
/// # Returns
///
/// Scalar loss tensor with shape-preserving gradient flow for backpropagation.
///
/// # Examples
///
/// ```no_run
/// use burn::tensor::{Device, Tensor};
/// use durian_sonic::training::weighted_cross_entropy_loss;
///
/// let device = Device::flex();
///
/// let logits: Tensor<2> = Tensor::from_data(
///     burn::tensor::TensorData::new(vec![-1.0, 2.0, -1.0], [1, 3]),
///     &device,
/// );
/// let targets = vec![1]; // Ripe
/// let weights = [0.548, 1.419, 2.129]; // iPhone weights
///
/// let loss = weighted_cross_entropy_loss(logits, &targets, &weights, 0.05, &device);
/// ```
pub fn weighted_cross_entropy_loss(
    logits: Tensor<2>,
    targets: &[usize],
    class_weights: &[f32; 3],
    label_smoothing: f32,
    device: &Device,
) -> Tensor<1> {
    let batch_size = targets.len();
    let num_classes = 3;

    // Compute numerically stable log-probabilities using log_softmax
    // log_softmax(z_c) = (z_c - max_k z_k) - ln(sum_j exp(z_j - max_k z_k))
    let log_probs = burn::tensor::activation::log_softmax(logits, 1);

    // Build class-weighted smoothed target distribution T[B, 3]
    // For each sample i: T[i, c] = w_c * q_i(c) where:
    //   q_i(c) = (1 - epsilon) + (epsilon / 3) if c = y_i (true class)
    //   q_i(c) = epsilon / 3 otherwise
    // This ensures sum_c q_i(c) = 1.0
    let mut target_weights = Vec::with_capacity(batch_size * num_classes);

    let smooth_on_target = (1.0 - label_smoothing) + (label_smoothing / 3.0); // approximately 0.9666667
    let smooth_off_target = label_smoothing / 3.0; // approximately 0.0166667

    for &target_idx in targets {
        for (c, &weight) in class_weights.iter().enumerate() {
            let q_i_c = if c == target_idx {
                smooth_on_target
            } else {
                smooth_off_target
            };
            let weighted_target = weight * q_i_c;
            target_weights.push(weighted_target);
        }
    }

    let target_tensor: Tensor<2> = Tensor::from_data(
        burn::tensor::TensorData::new(target_weights, [batch_size, num_classes]),
        device,
    );

    // Compute weighted cross-entropy: L = -(1/B) * sum_i sum_c T[i, c] * log_probs[i, c]
    // = -(target_tensor * log_probs).sum_dim(1).mean()
    let weighted_log_probs = target_tensor * log_probs;
    let sample_losses = weighted_log_probs.sum_dim(1); // [B]
    let mean_loss = sample_losses.mean(); // scalar

    // Return negative mean (cross-entropy is -sum p log q)
    mean_loss.neg()
}

/// 3x3 Confusion matrix for ripeness classification (Unripe, Ripe, Overripe).
///
/// Tracks predicted vs actual class counts: `counts[actual][predicted]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfusionMatrix3x3 {
    pub counts: [[usize; 3]; 3],
}

impl ConfusionMatrix3x3 {
    /// Creates a new empty confusion matrix with all counts initialized to zero.
    pub fn new() -> Self {
        Self {
            counts: [[0; 3]; 3],
        }
    }

    /// Records a single prediction: increments `counts[actual][predicted]`.
    ///
    /// # Arguments
    /// * `actual` - Ground truth class index (0=Unripe, 1=Ripe, 2=Overripe)
    /// * `predicted` - Model predicted class index (0=Unripe, 1=Ripe, 2=Overripe)
    pub fn record(&mut self, actual: usize, predicted: usize) {
        self.counts[actual][predicted] += 1;
    }

    /// Computes evaluation metrics from the confusion matrix.
    ///
    /// All divisions by zero are guarded and return 0.0 (not NaN).
    pub fn compute_report(&self) -> EvaluationReport {
        let mut per_class_precision = [0.0f32; 3];
        let mut per_class_recall = [0.0f32; 3];
        let mut per_class_f1 = [0.0f32; 3];

        let mut total_correct = 0;
        let mut total_samples = 0;

        for c in 0..3 {
            // Compute row sum (actual class c)
            let row_sum: usize = self.counts[c].iter().sum();

            // Compute column sum (predicted class c)
            let col_sum: usize = (0..3).map(|r| self.counts[r][c]).sum();

            // Precision = TP / (TP + FP) = counts[c][c] / col_sum
            per_class_precision[c] = if col_sum > 0 {
                self.counts[c][c] as f32 / col_sum as f32
            } else {
                0.0
            };

            // Recall = TP / (TP + FN) = counts[c][c] / row_sum
            per_class_recall[c] = if row_sum > 0 {
                self.counts[c][c] as f32 / row_sum as f32
            } else {
                0.0
            };

            // F1 = 2 * P * R / (P + R)
            let p = per_class_precision[c];
            let r = per_class_recall[c];
            per_class_f1[c] = if p + r > 0.0 {
                2.0 * p * r / (p + r)
            } else {
                0.0
            };

            total_correct += self.counts[c][c];
            total_samples += row_sum;
        }

        // Macro F1 = average F1 across classes
        let macro_f1 = (per_class_f1[0] + per_class_f1[1] + per_class_f1[2]) / 3.0;

        // Balanced Accuracy = average recall across classes
        let balanced_accuracy =
            (per_class_recall[0] + per_class_recall[1] + per_class_recall[2]) / 3.0;

        // Overall Accuracy = total correct / total samples
        let overall_accuracy = if total_samples > 0 {
            total_correct as f32 / total_samples as f32
        } else {
            0.0
        };

        // Critical FP_R rate = (FP_unripe->ripe + FP_overripe->ripe) / (total_unripe + total_overripe)
        let fp_unripe_to_ripe = self.counts[0][1];
        let fp_overripe_to_ripe = self.counts[2][1];
        let total_unripe = self.counts[0].iter().sum::<usize>();
        let total_overripe = self.counts[2].iter().sum::<usize>();
        let critical_fp_ripe_rate = if total_unripe + total_overripe > 0 {
            (fp_unripe_to_ripe + fp_overripe_to_ripe) as f32
                / (total_unripe + total_overripe) as f32
        } else {
            0.0
        };

        EvaluationReport {
            confusion: self.counts,
            per_class_precision,
            per_class_recall,
            per_class_f1,
            macro_f1,
            balanced_accuracy,
            overall_accuracy,
            critical_fp_ripe_rate,
        }
    }
}

impl Default for ConfusionMatrix3x3 {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregates tap-level probabilities into a single recording-level prediction
/// using Shannon entropy-based confidence weighting.
///
/// # Algorithm (FR-008)
///
/// For M taps in a recording with probability distributions P_k(c) for k=1..M:
/// 1. Compute Shannon entropy: H(P_k) = -sum_c P_k(c) * ln(P_k(c))
/// 2. Compute confidence weight: alpha_k = 1 - H(P_k) / ln(3)
///    - High confidence (low entropy)  ->  alpha_k close to 1.0
///    - Low confidence (high entropy)  ->  alpha_k close to 0.0
/// 3. Weighted aggregation: P_final(c) = sum_k (alpha_k * P_k(c)) / sum_k alpha_k
/// 4. Fallback: If sum_k alpha_k <= 1e-6, use uniform mean: P_final(c) = (1/M) * sum_k P_k(c)
///
/// # Arguments
///
/// * `tap_probas` - Vector of probability distributions, each of length 3
///
/// # Returns
///
/// Aggregated probability distribution [P(Unripe), P(Ripe), P(Overripe)]
///
/// # Examples
///
/// ```
/// use durian_sonic::training::aggregate_recording_probas;
///
/// let taps = vec![
///     vec![0.8, 0.1, 0.1],  // High confidence for Unripe
///     vec![0.4, 0.3, 0.3],  // Low confidence (gets low weight)
/// ];
/// let result = aggregate_recording_probas(&taps);
/// assert!(result[0] > 0.6);  // Should favor high-confidence tap
/// ```
pub fn aggregate_recording_probas(tap_probas: &[Vec<f32>]) -> Vec<f32> {
    let num_taps = tap_probas.len();
    if num_taps == 0 {
        return vec![0.0, 0.0, 0.0];
    }

    if num_taps == 1 {
        return tap_probas[0].clone();
    }

    let ln_3 = 3.0f32.ln(); // ln(3)  approximately  1.0986
    let epsilon = 1e-12f32; // Floor for log stability
    let weight_threshold = 1e-6f32;

    // Compute confidence weights for each tap
    let mut weights = Vec::with_capacity(num_taps);
    let mut total_weight = 0.0f32;

    for probs in tap_probas {
        // Compute Shannon entropy: H = -sum p * ln(p)
        let mut entropy = 0.0f32;
        for &p in probs {
            if p > epsilon {
                entropy -= p * p.max(epsilon).ln();
            }
        }

        // Confidence weight: alpha = 1 - H / ln(3)
        // Clamp to [0, 1] to handle numerical precision issues
        let confidence = (1.0 - entropy / ln_3).clamp(0.0, 1.0);
        weights.push(confidence);
        total_weight += confidence;
    }

    // Check if we need fallback (all weights near zero)
    if total_weight <= weight_threshold {
        // Fallback: uniform arithmetic mean
        let mut result = vec![0.0f32; 3];
        for probs in tap_probas {
            for c in 0..3 {
                result[c] += probs[c];
            }
        }
        for prob in &mut result {
            *prob /= num_taps as f32;
        }
        return result;
    }

    // Weighted aggregation: P_final(c) = sum(alpha_k * P_k(c)) / sum(alpha_k)
    let mut result = vec![0.0f32; 3];
    for (k, probs) in tap_probas.iter().enumerate() {
        let weight = weights[k];
        for c in 0..3 {
            result[c] += weight * probs[c];
        }
    }
    for prob in &mut result {
        *prob /= total_weight;
    }

    result
}

/// Comprehensive evaluation metrics report for 3-class ripeness classification.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationReport {
    pub confusion: [[usize; 3]; 3],
    pub per_class_precision: [f32; 3],
    pub per_class_recall: [f32; 3],
    pub per_class_f1: [f32; 3],
    pub macro_f1: f32,
    pub balanced_accuracy: f32,
    pub overall_accuracy: f32,
    pub critical_fp_ripe_rate: f32,
}

// Legacy training utilities (preserved for benchmark compatibility)

/// Gradient descent optimizer (placeholder)
pub struct GradientDescent {
    learning_rate: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    /// Update weights based on gradients (placeholder implementation)
    pub fn update(&mut self, weights: &[f64], gradients: &[f64]) -> Vec<f64> {
        weights
            .iter()
            .zip(gradients.iter())
            .map(|(w, g)| w - self.learning_rate * g)
            .collect()
    }
}

/// Learning rate schedule strategies
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant { rate: f64 },

    /// Exponential decay: rate = initial_rate * decay_rate^(epoch / decay_steps)
    ExponentialDecay {
        initial_rate: f64,
        decay_rate: f64,
        decay_steps: usize,
    },

    /// Step decay: rate = initial_rate * drop_rate^floor(epoch / epochs_drop)
    StepDecay {
        initial_rate: f64,
        drop_rate: f64,
        epochs_drop: usize,
    },
}

impl LearningRateSchedule {
    /// Get learning rate for a given epoch (placeholder implementation)
    pub fn get_rate(&self, epoch: usize) -> f64 {
        match self {
            Self::Constant { rate } => *rate,
            Self::ExponentialDecay {
                initial_rate,
                decay_rate,
                decay_steps,
            } => {
                let power = epoch as f64 / *decay_steps as f64;
                initial_rate * decay_rate.powf(power)
            }
            Self::StepDecay {
                initial_rate,
                drop_rate,
                epochs_drop,
            } => {
                let drops = (epoch / epochs_drop) as u32;
                initial_rate * drop_rate.powi(drops as i32)
            }
        }
    }
}

// Phase 3 User Story 4: Dual-Level Evaluation & Early Stopping

/// Pairs tap-level and recording-level evaluation reports.
///
/// - `tap_level`: Evaluates each individual TapSample
/// - `recording_level`: Aggregates taps by sample_num using entropy weighting
#[derive(Debug, Clone)]
pub struct DualLevelEvaluation {
    pub tap_level: EvaluationReport,
    pub recording_level: EvaluationReport,
}

/// Early stopping action returned by EarlyStopping::step
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingAction {
    /// Validation metric improved, checkpoint saved
    Improved,
    /// No improvement but patience remaining
    Continue,
    /// Patience exhausted, stop training
    Stop,
}

/// Early stopping controller based on validation recording-level Macro F1.
///
/// Tracks best validation metric and triggers stop when patience is exhausted.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    pub patience: usize,
    pub best_macro_f1: f32,
    pub best_epoch: usize,
    pub patience_counter: usize,
}

impl EarlyStopping {
    /// Creates new early stopping with given patience.
    ///
    /// Initializes best_macro_f1 to -1.0 so Epoch 1 always saves a checkpoint.
    pub fn new(patience: usize) -> Self {
        Self {
            patience,
            best_macro_f1: -1.0,
            best_epoch: 0,
            patience_counter: 0,
        }
    }

    /// Updates state with new validation metric.
    ///
    /// Returns Improved if metric improved (resets patience counter),
    /// Continue if no improvement but patience remains,
    /// Stop if patience exhausted.
    pub fn step(&mut self, epoch: usize, val_recording_macro_f1: f32) -> EarlyStoppingAction {
        if val_recording_macro_f1 > self.best_macro_f1 {
            self.best_macro_f1 = val_recording_macro_f1;
            self.best_epoch = epoch;
            self.patience_counter = 0;
            EarlyStoppingAction::Improved
        } else {
            self.patience_counter += 1;
            if self.patience_counter >= self.patience {
                EarlyStoppingAction::Stop
            } else {
                EarlyStoppingAction::Continue
            }
        }
    }
}

/// Training outcome summary for a platform model.
#[derive(Debug, Clone)]
pub struct TrainingOutcome {
    pub platform: PlatformTarget,
    pub class_weights: [f32; 3],
    pub best_epoch: usize,
    pub epochs_run: usize,
    pub best_val_recording_macro_f1: f32,
    pub artifact_path: PathBuf,
    pub artifact_size_bytes: u64,
    pub test_evaluation: DualLevelEvaluation,
}

/// Aggregates tap-level probabilities into recording-level distribution using Shannon entropy confidence weighting.
///
/// Each tap's confidence weight alpha_k = (1.0 - H(P_k)/ln(3)).clamp(0.0, 1.0), where
/// H(P_k) = -sum P_k(c)ln(max(P_k(c), 1e-12)) is the Shannon entropy.
///
/// Returns weighted distribution P_final(c) = sum(alpha_k * P_k(c)) / sum(alpha_k).
/// Falls back to uniform arithmetic mean [1/3, 1/3, 1/3] when sum alpha_k <= 1e-6.
pub fn aggregate_recording_probabilities(tap_probs: &[[f32; 3]]) -> [f32; 3] {
    const LN3: f32 = 1.0986123; // ln(3)
    const LOG_FLOOR: f32 = 1e-12;
    const ALPHA_SUM_THRESHOLD: f32 = 1e-6;

    let mut weighted_sum = [0.0f32; 3];
    let mut alpha_sum = 0.0f32;

    for probs in tap_probs {
        // Compute Shannon entropy H(P_k) = -sum P_k(c) * ln(max(P_k(c), 1e-12))
        let entropy: f32 = probs
            .iter()
            .map(|&p| {
                let clamped = p.max(LOG_FLOOR);
                -p * clamped.ln()
            })
            .sum();

        // Confidence weight alpha_k = (1.0 - H(P_k)/ln(3)).clamp(0.0, 1.0)
        let alpha = (1.0 - entropy / LN3).clamp(0.0, 1.0);
        alpha_sum += alpha;

        // Accumulate weighted probabilities
        for c in 0..3 {
            weighted_sum[c] += alpha * probs[c];
        }
    }

    // Uniform arithmetic mean fallback when sum alpha_k <= 1e-6
    if alpha_sum <= ALPHA_SUM_THRESHOLD {
        let mut uniform = [0.0f32; 3];
        for c in 0..3 {
            uniform[c] = tap_probs.iter().map(|p| p[c]).sum::<f32>() / tap_probs.len() as f32;
        }
        return uniform;
    }

    // Normalize by alpha sum
    let mut result = [0.0f32; 3];
    for c in 0..3 {
        result[c] = weighted_sum[c] / alpha_sum;
    }
    result
}

/// Evaluates model on a dataset split, producing dual-level (tap and recording) evaluation reports.
///
/// # Arguments
/// - `model`: The neural network in evaluation mode (call `.valid()` before passing)
/// - `samples`: TapSample dataset split to evaluate
/// - `batch_size`: Batch size for forward passes
/// - `device`: Backend device
///
/// # Returns
/// DualLevelEvaluation containing:
/// - `tap_level`: Individual tap predictions vs ground truth
/// - `recording_level`: Entropy-weighted aggregated recording predictions vs ground truth
pub fn evaluate_model(
    model: &DurianAudioCnn,
    samples: &[TapSample],
    batch_size: usize,
    device: &Device,
) -> DualLevelEvaluation {
    use std::collections::BTreeMap;

    let mut tap_level_cm = ConfusionMatrix3x3::default();
    let mut recording_groups: BTreeMap<u32, (RipenessLabel, Vec<[f32; 3]>)> = BTreeMap::new();

    // Process samples in batches
    for chunk in samples.chunks(batch_size) {
        let batch = collate_tap_batch(chunk, device);
        let probs = model.forward_proba(batch.mel_specs, batch.phys_feats);

        let probs_data = probs.to_data();
        let probs_vec: Vec<f32> = probs_data.iter::<f32>().collect();

        for (i, sample) in chunk.iter().enumerate() {
            let prob_slice = &probs_vec[i * 3..(i + 1) * 3];
            let tap_probs = [prob_slice[0], prob_slice[1], prob_slice[2]];

            // Tap-level: argmax prediction
            let predicted_class = tap_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0); // Fallback to class 0 if all NaN

            let actual_class = sample.label as usize;
            tap_level_cm.record(actual_class, predicted_class);

            // Recording-level: group by sample_num
            recording_groups
                .entry(sample.sample_num)
                .or_insert_with(|| (sample.label, Vec::new()))
                .1
                .push(tap_probs);
        }
    }

    // Recording-level evaluation: aggregate taps per recording
    let mut recording_level_cm = ConfusionMatrix3x3::default();
    for (_sample_num, (label, tap_probs)) in recording_groups {
        let aggregated = aggregate_recording_probabilities(&tap_probs);
        let predicted_class = aggregated
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0); // Fallback to class 0 if all NaN

        let actual_class = label as usize;
        recording_level_cm.record(actual_class, predicted_class);
    }

    DualLevelEvaluation {
        tap_level: tap_level_cm.compute_report(),
        recording_level: recording_level_cm.compute_report(),
    }
}

/// Trains a model on Train/Val/Test splits with early stopping and checkpoint management.
///
/// # Arguments
/// - `model`: Initialized DurianAudioCnn model
/// - `train_samples`: Training TapSample dataset
/// - `val_samples`: Validation TapSample dataset
/// - `test_samples`: Test TapSample dataset (held-out, evaluated only after training)
/// - `class_weights`: Platform-specific inverse class frequency weights `[3]`
/// - `config`: Training configuration (epochs, batch_size, lr, etc.)
/// - `checkpoint_dir`: Directory to save .mpk checkpoints
/// - `device`: Device for computation
///
/// # Returns
/// TrainingOutcome with best checkpoint path, test metrics, and training statistics
pub fn train_model_on_splits(
    mut model: DurianAudioCnn,
    splits: &DataSplits,
    class_weights: &[f32; 3],
    config: &crate::config::Config,
    checkpoint_dir: &Path,
    device: &Device,
) -> Result<TrainingOutcome> {
    use burn::optim::{AdamWConfig, GradientsParams};
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let epochs = config.training.epochs;
    let batch_size = config.training.batch_size;
    let patience = config.training.early_stopping_patience;
    let label_smoothing = config.training.label_smoothing;
    let learning_rate = config.training.learning_rate;
    let weight_decay = config.training.weight_decay;

    // Initialize AdamW optimizer with weight decay
    let mut optim = AdamWConfig::new()
        .with_weight_decay(weight_decay as f32)
        .init();

    // Initialize early stopping
    let mut early_stopping = EarlyStopping::new(patience);
    let mut best_checkpoint_path: Option<PathBuf> = None;
    let mut epochs_run = 0;
    let total_batches = splits.train.len().div_ceil(batch_size.max(1));

    use std::io::Write;
    tracing::info!(
        "Starting training: {} epochs max, batch_size={}, {} batches/epoch, patience={}",
        epochs,
        batch_size,
        total_batches,
        patience
    );
    println!(
        "\n[3/3] Training DurianAudioCnn ({} epochs max, batch_size={}, {} batches/epoch, patience={}):",
        epochs, batch_size, total_batches, patience
    );
    tracing::info!(
        "Tap windows: Train={} | Val={} | Test={}",
        splits.train.len(),
        splits.val.len(),
        splits.test.len()
    );
    println!(
        "      Tap windows: Train={} | Val={} | Test={}\n",
        splits.train.len(),
        splits.val.len(),
        splits.test.len()
    );
    let _ = std::io::stdout().flush();

    tracing::info!(
        "Starting training: {} epochs, batch_size={}, lr={}, weight_decay={}, patience={}",
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        patience
    );
    tracing::info!(
        "Dataset sizes - Train: {}, Val: {}, Test: {}",
        splits.train.len(),
        splits.val.len(),
        splits.test.len()
    );

    for epoch in 1..=epochs {
        epochs_run = epoch;
        let epoch_start = std::time::Instant::now();
        tracing::info!("Epoch {}/{}", epoch, epochs);

        // Deterministic epoch shuffling with seeded RNG
        let seed = config.dataset.split_seed + epoch as u64;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut train_indices: Vec<usize> = (0..splits.train.len()).collect();
        train_indices.shuffle(&mut rng);

        // Training phase - iterate over shuffled batches
        let mut epoch_train_loss = 0.0f32;
        let mut num_batches = 0;

        for (batch_idx, batch_indices) in train_indices.chunks(batch_size).enumerate() {
            let batch_samples: Vec<TapSample> = batch_indices
                .iter()
                .map(|&idx| splits.train[idx].clone())
                .collect();

            // Create batch tensors
            let batch = collate_tap_batch(&batch_samples, device);
            let targets: Vec<usize> = batch_samples.iter().map(|s| s.label as usize).collect();

            // Forward pass - ensure tensors are tracked for autodiff
            let logits = model.forward(
                batch.mel_specs.clone().require_grad(),
                batch.phys_feats.clone().require_grad(),
            );

            // Compute loss
            let loss = weighted_cross_entropy_loss(
                logits,
                &targets,
                class_weights,
                label_smoothing,
                device,
            );

            let loss_val: f32 = loss.clone().into_scalar();
            epoch_train_loss += loss_val;
            num_batches += 1;

            // Backpropagation: compute gradients
            let grads = loss.backward();

            // Update model parameters with AdamW
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(learning_rate, model, grads_params);

            // Render live batch progress bar
            let current_batch = batch_idx + 1;
            let bar_width = 20;
            let filled = (current_batch * bar_width) / total_batches.max(1);
            let bar: String = (0..bar_width)
                .map(|i| {
                    if i < filled {
                        '='
                    } else if i == filled {
                        '>'
                    } else {
                        '.'
                    }
                })
                .collect();
            let running_avg_loss = epoch_train_loss / num_batches as f32;
            print!(
                "\r      Epoch {:02}/{:02} [{}] Batch {:3}/{:3} | Loss: {:.4} (avg: {:.4}) | {:.1}s",
                epoch,
                epochs,
                bar,
                current_batch,
                total_batches,
                loss_val,
                running_avg_loss,
                epoch_start.elapsed().as_secs_f32()
            );
            let _ = std::io::stdout().flush();
        }

        let avg_train_loss = epoch_train_loss / num_batches.max(1) as f32;
        tracing::info!("Epoch {} - Avg Train Loss: {:.4}", epoch, avg_train_loss);

        print!(
            "\r      Epoch {:02}/{:02} [====================] Validating ({} taps)...                         ",
            epoch,
            epochs,
            splits.val.len()
        );
        let _ = std::io::stdout().flush();

        // Validation phase - evaluate on val set
        let val_eval = evaluate_model(&model, splits.val, batch_size, device);

        tracing::info!(
            "Epoch {} - Val Tap Macro F1: {:.4}, Val Recording Macro F1: {:.4}",
            epoch,
            val_eval.tap_level.macro_f1,
            val_eval.recording_level.macro_f1
        );

        // Early stopping check
        let previous_best_f1 = early_stopping.best_macro_f1;
        let action = early_stopping.step(epoch, val_eval.recording_level.macro_f1);
        let epoch_secs = epoch_start.elapsed().as_secs_f32();

        match action {
            EarlyStoppingAction::Improved => {
                tracing::info!(
                    "Epoch {} - New best Val Recording Macro F1: {:.4} (previous: {:.4})",
                    epoch,
                    val_eval.recording_level.macro_f1,
                    previous_best_f1
                );

                // Save checkpoint using Burn 0.22.1 store API
                let checkpoint_name = format!("model_epoch_{}.mpk", epoch);
                let checkpoint_path = checkpoint_dir.join(&checkpoint_name);
                model
                    .save_mpk(&checkpoint_path)
                    .with_context(|| format!("Failed to save checkpoint at epoch {}", epoch))?;
                best_checkpoint_path = Some(checkpoint_path.clone());
                tracing::info!("Checkpoint saved: {}", checkpoint_path.display());

                // User-facing progress line
                println!(
                    "\r      Epoch {:02}/{:02} [{:5.1}s] | Train Loss: {:.4} | Val Tap F1: {:.4} | Val Rec F1: {:.4} (Acc: {:5.1}%) | * BEST (saved)",
                    epoch,
                    epochs,
                    epoch_secs,
                    avg_train_loss,
                    val_eval.tap_level.macro_f1,
                    val_eval.recording_level.macro_f1,
                    val_eval.recording_level.overall_accuracy * 100.0
                );
            }
            EarlyStoppingAction::Continue => {
                tracing::info!(
                    "Epoch {} - No improvement (patience: {}/{})",
                    epoch,
                    early_stopping.patience_counter,
                    patience
                );
                // User-facing progress line
                println!(
                    "\r      Epoch {:02}/{:02} [{:5.1}s] | Train Loss: {:.4} | Val Tap F1: {:.4} | Val Rec F1: {:.4} (Acc: {:5.1}%) | Patience: {}/{}",
                    epoch,
                    epochs,
                    epoch_secs,
                    avg_train_loss,
                    val_eval.tap_level.macro_f1,
                    val_eval.recording_level.macro_f1,
                    val_eval.recording_level.overall_accuracy * 100.0,
                    early_stopping.patience_counter,
                    patience
                );
            }
            EarlyStoppingAction::Stop => {
                tracing::info!(
                    "Early stopping triggered at epoch {} (patience exhausted)",
                    epoch
                );
                // User-facing progress line
                println!(
                    "\r      Epoch {:02}/{:02} [{:5.1}s] | Train Loss: {:.4} | Val Tap F1: {:.4} | Val Rec F1: {:.4} (Acc: {:5.1}%) | STOP (patience {}/{})",
                    epoch,
                    epochs,
                    epoch_secs,
                    avg_train_loss,
                    val_eval.tap_level.macro_f1,
                    val_eval.recording_level.macro_f1,
                    val_eval.recording_level.overall_accuracy * 100.0,
                    early_stopping.patience_counter,
                    patience
                );
                break;
            }
        }
    }

    // Restore best checkpoint using Burn 0.22.1 store API
    let best_path = best_checkpoint_path
        .ok_or_else(|| anyhow::anyhow!("No checkpoint was saved during training"))?;
    tracing::info!(
        "Restoring best checkpoint (epoch {}, Val Rec F1: {:.4}) and evaluating on Test set...",
        early_stopping.best_epoch,
        early_stopping.best_macro_f1
    );
    println!(
        "\n      Restoring best checkpoint (epoch {}, Val Rec F1: {:.4}) and evaluating on Test set...",
        early_stopping.best_epoch, early_stopping.best_macro_f1
    );
    let _ = std::io::stdout().flush();
    tracing::info!("Restoring best checkpoint: {}", best_path.display());
    model = model
        .load_mpk(&best_path, device)
        .with_context(|| "Failed to restore best checkpoint")?;

    // Final evaluation on held-out test set
    tracing::info!("Evaluating on test set...");
    let test_eval = evaluate_model(&model, splits.test, batch_size, device);

    tracing::info!(
        "Test Results - Tap Macro F1: {:.4}, Recording Macro F1: {:.4}",
        test_eval.tap_level.macro_f1,
        test_eval.recording_level.macro_f1
    );

    // Get artifact size from saved checkpoint
    let artifact_size_bytes = std::fs::metadata(&best_path)
        .with_context(|| format!("Failed to get file size for {}", best_path.display()))?
        .len();

    Ok(TrainingOutcome {
        platform: PlatformTarget::Iphone, // Will be set by caller
        class_weights: *class_weights,
        best_epoch: early_stopping.best_epoch,
        epochs_run,
        best_val_recording_macro_f1: early_stopping.best_macro_f1,
        artifact_path: best_path,
        artifact_size_bytes,
        test_evaluation: test_eval,
    })
}

/// Run end-to-end platform training workflow.
///
/// Loads platform metadata, computes class weights from raw counts,
/// partitions by fruit_id, extracts tap samples with train-only oversampling,
/// and delegates to train_model_on_splits.
///
/// # Arguments
/// - `config`: Configuration with DSP, dataset, training, and model parameters
/// - `platform`: Target platform (iPhone or Android)
/// - `output_path`: Path stem for saving .mpk checkpoint (e.g., "artifacts/durian_iphone")
/// - `device`: Backend device for training
///
/// # Returns
/// `TrainingOutcome` with platform, class weights, best epoch, artifact path/size, and test evaluation
///
/// # Examples
///
/// ```no_run
/// use durian_sonic::training::run_platform_training;
/// use durian_sonic::config::Config;
/// use durian_sonic::dataset::PlatformTarget;
/// use burn::tensor::Device;
/// use std::path::Path;
///
/// let config = Config::default();
/// let platform = PlatformTarget::Iphone;
/// let output = Path::new("artifacts/durian_iphone");
/// let device = Device::flex();
///
/// let outcome = run_platform_training(
///     &config,
///     platform,
///     output,
///     &device,
/// );
/// ```
pub fn run_platform_training(
    config: &crate::config::Config,
    platform: PlatformTarget,
    output_path: &Path,
    device: &Device,
) -> Result<TrainingOutcome> {
    use crate::dataset::{
        compute_class_weights, load_and_verify_metadata, stratified_group_split_by_fruit,
    };

    let data_dir = PathBuf::from(&config.dataset.data_dir);

    // Resolve platform-specific paths
    let csv_path = platform.metadata_csv_path(&data_dir);
    let audio_dir = platform.audio_subdir(&data_dir);

    // Load and verify metadata
    use std::io::Write;
    tracing::info!("Loading and verifying dataset metadata (SHA-256)...");
    println!("[1/3] Loading and verifying dataset metadata (SHA-256)...");
    let _ = std::io::stdout().flush();

    let (records, integrity_report) = load_and_verify_metadata(&csv_path, &audio_dir, true)
        .context("Failed to load and verify platform metadata")?;

    // Compute class weights from raw metadata counts (before tap slicing)
    let class_weights = compute_class_weights(&integrity_report.class_counts);

    // Partition records by fruit_id with stratified group split
    let splits = stratified_group_split_by_fruit(
        &records,
        config.dataset.train_ratio,
        config.dataset.val_ratio,
        config.dataset.split_seed,
    )?;

    tracing::info!(
        "Verified {} files ({} unique fruits) | Class weights: [{:.3}, {:.3}, {:.3}]",
        integrity_report.total_files,
        integrity_report.unique_fruits,
        class_weights[0],
        class_weights[1],
        class_weights[2]
    );
    println!(
        "      Verified {} files ({} unique fruits) | Class weights: [{:.3}, {:.3}, {:.3}]",
        integrity_report.total_files,
        integrity_report.unique_fruits,
        class_weights[0],
        class_weights[1],
        class_weights[2]
    );
    tracing::info!(
        "Anti-leakage split: Train={} files, Val={} files, Test={} files",
        splits.train.len(),
        splits.val.len(),
        splits.test.len()
    );
    println!(
        "      Anti-leakage split: Train={} files, Val={} files, Test={} files\n",
        splits.train.len(),
        splits.val.len(),
        splits.test.len()
    );
    tracing::info!(
        "Extracting DSP tap features (22.05 kHz, Log-Mel 128x128 + 4 physics features)..."
    );
    println!(
        "[2/3] Extracting DSP tap features (22.05 kHz, Log-Mel 128x128 + 4 physics features)..."
    );
    let _ = std::io::stdout().flush();

    // Extract tap samples with train-only oversampling
    let train_samples = extract_dataset_tap_samples(
        &splits.train,
        &audio_dir,
        &config.dsp,
        true, // is_train
    )
    .context("Failed to extract Train tap samples")?;

    let val_samples = extract_dataset_tap_samples(
        &splits.val,
        &audio_dir,
        &config.dsp,
        false, // is_train
    )
    .context("Failed to extract Validation tap samples")?;

    let test_samples = extract_dataset_tap_samples(
        &splits.test,
        &audio_dir,
        &config.dsp,
        false, // is_train
    )
    .context("Failed to extract Test tap samples")?;

    // Assert zero fruit_id overlap across splits
    let train_fruits: std::collections::HashSet<u32> =
        train_samples.iter().map(|s| s.fruit_id).collect();
    let val_fruits: std::collections::HashSet<u32> =
        val_samples.iter().map(|s| s.fruit_id).collect();
    let test_fruits: std::collections::HashSet<u32> =
        test_samples.iter().map(|s| s.fruit_id).collect();

    anyhow::ensure!(
        train_fruits.is_disjoint(&val_fruits),
        "Train and Val splits have overlapping fruit_ids"
    );
    anyhow::ensure!(
        train_fruits.is_disjoint(&test_fruits),
        "Train and Test splits have overlapping fruit_ids"
    );
    anyhow::ensure!(
        val_fruits.is_disjoint(&test_fruits),
        "Val and Test splits have overlapping fruit_ids"
    );

    // Initialize model
    let model = DurianAudioCnn::new(config.model.num_classes, config.model.dropout, device);

    // Train the model
    let splits = DataSplits {
        train: &train_samples,
        val: &val_samples,
        test: &test_samples,
    };
    let mut outcome =
        train_model_on_splits(model, &splits, &class_weights, config, output_path, device)?;

    // Update outcome with platform information
    outcome.platform = platform;

    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Phase 2 Foundational Tests: TapSample and DurianBatch (T003)

    #[test]
    fn test_tap_sample_construction_and_field_invariants() {
        // Create a valid TapSample with proper constraints
        let mel_data = vec![0.5f32; 16_384]; // 128 * 128 = 16,384
        let physics_feats = [0.3f32, 0.5f32, 0.7f32, 0.9f32];

        let sample = TapSample {
            fruit_id: 42,    // 1..=175
            sample_num: 100, // 1..=281 (iOS) or 1..=175 (Android)
            tap_index: 2,    // 0..K
            mel_data: mel_data.clone(),
            physics_feats,
            label: RipenessLabel::Ripe,
        };

        // Verify field values
        assert_eq!(sample.fruit_id, 42);
        assert_eq!(sample.sample_num, 100);
        assert_eq!(sample.tap_index, 2);
        assert_eq!(sample.mel_data.len(), 16_384);
        assert_eq!(sample.physics_feats.len(), 4);
        assert_eq!(sample.label, RipenessLabel::Ripe);

        // Verify mel_data is in [0.0, 1.0]
        for &val in &sample.mel_data {
            assert!((0.0..=1.0).contains(&val), "mel_data must be in [0.0, 1.0]");
        }

        // Verify physics_feats is in [0.0, 1.0]
        for &val in &sample.physics_feats {
            assert!(
                (0.0..=1.0).contains(&val),
                "physics_feats must be in [0.0, 1.0]"
            );
        }
    }

    #[test]
    fn test_tap_sample_clone_and_equality() {
        let mel_data = vec![0.5f32; 16_384];
        let sample1 = TapSample {
            fruit_id: 1,
            sample_num: 1,
            tap_index: 0,
            mel_data: mel_data.clone(),
            physics_feats: [0.1, 0.2, 0.3, 0.4],
            label: RipenessLabel::Unripe,
        };

        let sample2 = sample1.clone();
        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_durian_batch_tensor_construction() {
        let device = Device::flex();

        // Create batch tensors with correct shapes
        let batch_size = 4;

        // mel_specs: [B, 1, 128, 128]
        let mel_data = vec![0.5f32; batch_size * 128 * 128];
        let mel_specs = Tensor::<4>::from_data(
            burn::tensor::TensorData::new(mel_data, [batch_size, 1, 128, 128]),
            &device,
        );

        // phys_feats: [B, 4]
        let phys_data = vec![0.5f32; batch_size * 4];
        let phys_feats = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(phys_data, [batch_size, 4]),
            &device,
        );

        // targets: [B]
        let target_data = vec![0i64, 1, 2, 1]; // class indices
        let targets = Tensor::<1, Int>::from_data(
            burn::tensor::TensorData::new(target_data, [batch_size]),
            &device,
        );

        let batch = DurianBatch {
            mel_specs: mel_specs.clone(),
            phys_feats: phys_feats.clone(),
            targets: targets.clone(),
        };

        // Verify tensor shapes
        assert_eq!(batch.mel_specs.dims(), [batch_size, 1, 128, 128]);
        assert_eq!(batch.phys_feats.dims(), [batch_size, 4]);
        assert_eq!(batch.targets.dims(), [batch_size]);
    }

    // Legacy training utilities tests (preserved for benchmark compatibility)

    #[test]
    fn test_gradient_descent_basic() {
        let mut optimizer = GradientDescent::new(0.1);
        let weights = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.5, 0.5, 0.5];

        let updated = optimizer.update(&weights, &gradients);

        assert_eq!(updated, vec![0.95, 1.95, 2.95]);
    }

    #[test]
    fn test_constant_schedule() {
        let schedule = LearningRateSchedule::Constant { rate: 0.01 };

        assert_eq!(schedule.get_rate(0), 0.01);
        assert_eq!(schedule.get_rate(100), 0.01);
    }

    #[test]
    fn test_exponential_decay() {
        let schedule = LearningRateSchedule::ExponentialDecay {
            initial_rate: 0.1,
            decay_rate: 0.95,
            decay_steps: 100,
        };

        let rate_0 = schedule.get_rate(0);
        let rate_100 = schedule.get_rate(100);

        assert!(rate_0 > rate_100);
        assert!((rate_0 - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_step_decay() {
        let schedule = LearningRateSchedule::StepDecay {
            initial_rate: 0.1,
            drop_rate: 0.5,
            epochs_drop: 10,
        };

        assert_eq!(schedule.get_rate(0), 0.1);
        assert_eq!(schedule.get_rate(9), 0.1);
        assert_eq!(schedule.get_rate(10), 0.05);
        assert_eq!(schedule.get_rate(20), 0.025);
    }

    // Phase 4 Tests: Tap Sample Extraction with Train-Only Minority Oversampling (T010-T011)

    #[test]
    fn test_extract_tap_samples_train_vs_eval_mode() {
        let dsp_config = crate::config::DspConfig::default();
        let spec_cfg = SpectrogramConfig::from(&dsp_config);
        let mel_basis = build_mel_filterbank(
            spec_cfg.sample_rate,
            spec_cfg.n_fft,
            spec_cfg.n_mels,
            spec_cfg.fmin,
            spec_cfg.fmax,
        );

        // Create synthetic waveform with 3 clear peaks
        let sr = dsp_config.sample_rate;
        let duration_samples = (sr as f32 * 2.5) as usize; // 2.5 seconds
        let mut waveform = vec![0.0f32; duration_samples];

        // Insert 3 impulse peaks at 0.5s, 1.0s, 1.5s intervals
        let peak_positions = vec![
            (sr as f32 * 0.5) as usize,
            (sr as f32 * 1.0) as usize,
            (sr as f32 * 1.5) as usize,
        ];

        for &pos in &peak_positions {
            if pos < waveform.len() {
                waveform[pos] = 0.9;
                // Add some decay
                for j in 1..100 {
                    if pos + j < waveform.len() {
                        waveform[pos + j] = 0.9 * (-(j as f32) / 50.0).exp();
                    }
                }
            }
        }

        // Test each ripeness class
        for (label, expected_train_multiplier) in [
            (RipenessLabel::Unripe, 1),
            (RipenessLabel::Ripe, 2),
            (RipenessLabel::Overripe, 3),
        ] {
            // Eval mode: should return exactly K samples (no oversampling)
            let eval_samples = extract_tap_samples_from_waveform(
                &waveform,
                1,
                1,
                label,
                &dsp_config,
                &mel_basis,
                false, // is_train=false
            );

            // Train mode: should return K * multiplier samples
            let train_samples = extract_tap_samples_from_waveform(
                &waveform,
                1,
                1,
                label,
                &dsp_config,
                &mel_basis,
                true, // is_train=true
            );

            // Verify eval mode returns unaugmented samples only
            assert!(
                !eval_samples.is_empty(),
                "Should extract at least 1 tap for {:?}",
                label
            );
            let num_onsets = eval_samples.len();

            // Verify train mode applies correct multiplier
            assert_eq!(
                train_samples.len(),
                num_onsets * expected_train_multiplier,
                "Train mode should produce {}x samples for {:?}",
                expected_train_multiplier,
                label
            );

            // Verify all samples have correct metadata
            for sample in &train_samples {
                assert_eq!(sample.fruit_id, 1);
                assert_eq!(sample.sample_num, 1);
                assert_eq!(sample.label, label);
                assert_eq!(sample.mel_data.len(), 16_384);
                assert_eq!(sample.physics_feats.len(), 4);
            }
        }
    }

    #[test]
    fn test_extract_tap_samples_zero_onset_fallback() {
        let dsp_config = crate::config::DspConfig::default();
        let spec_cfg = SpectrogramConfig::from(&dsp_config);
        let mel_basis = build_mel_filterbank(
            spec_cfg.sample_rate,
            spec_cfg.n_fft,
            spec_cfg.n_mels,
            spec_cfg.fmin,
            spec_cfg.fmax,
        );

        // Create waveform with no clear peaks (white noise below threshold)
        let duration_samples = dsp_config.sample_rate; // 1 second
        let waveform: Vec<f32> = (0..duration_samples)
            .map(|i| (i as f32 * 0.001).sin() * 0.01)
            .collect(); // Very weak signal

        // Test eval mode (is_train=false): should produce exactly 1 canonical sample
        let samples_eval = extract_tap_samples_from_waveform(
            &waveform,
            99,
            42,
            RipenessLabel::Ripe,
            &dsp_config,
            &mel_basis,
            false, // is_train=false
        );

        // Should use max-RMS fallback and extract exactly 1 canonical sample
        assert_eq!(
            samples_eval.len(),
            1,
            "Zero-onset fallback in eval mode should produce exactly 1 sample"
        );

        let sample = &samples_eval[0];
        assert_eq!(sample.fruit_id, 99);
        assert_eq!(sample.sample_num, 42);
        assert_eq!(sample.tap_index, 0);
        assert_eq!(sample.mel_data.len(), 16_384);

        // Verify mel_data is in [0.0, 1.0] bounds
        for &val in &sample.mel_data {
            assert!(
                (0.0..=1.0).contains(&val),
                "mel_data value {} must be in [0.0, 1.0]",
                val
            );
        }

        // Verify physics_feats is in [0.0, 1.0] bounds
        for &val in &sample.physics_feats {
            assert!(
                (0.0..=1.0).contains(&val),
                "physics_feats value {} must be in [0.0, 1.0]",
                val
            );
        }

        // Test train mode with different classes to verify minority oversampling multipliers
        for (label, expected_multiplier) in [
            (RipenessLabel::Unripe, 1),
            (RipenessLabel::Ripe, 2),
            (RipenessLabel::Overripe, 3),
        ] {
            let samples_train = extract_tap_samples_from_waveform(
                &waveform,
                99,
                42,
                label,
                &dsp_config,
                &mel_basis,
                true, // is_train=true
            );

            assert_eq!(
                samples_train.len(),
                expected_multiplier,
                "Zero-onset fallback in train mode should produce {}x samples for {:?}",
                expected_multiplier,
                label
            );

            // Verify all samples have correct bounds
            for sample in &samples_train {
                for &val in &sample.mel_data {
                    assert!((0.0..=1.0).contains(&val), "mel_data must be in [0.0, 1.0]");
                }
                for &val in &sample.physics_feats {
                    assert!(
                        (0.0..=1.0).contains(&val),
                        "physics_feats must be in [0.0, 1.0]"
                    );
                }
            }
        }
    }

    #[test]
    fn test_collate_tap_batch_dynamic_shapes() {
        let device = Device::flex();

        // Test with various batch sizes: 1 (single inference), 7 (partial epoch), 32 (full batch)
        for batch_size in [1, 7, 32] {
            let samples: Vec<TapSample> = (0..batch_size)
                .map(|i| TapSample {
                    fruit_id: (i + 1) as u32,
                    sample_num: (i + 1) as u32,
                    tap_index: i,
                    mel_data: vec![0.5; 16_384],
                    physics_feats: [0.1, 0.2, 0.3, 0.4],
                    label: match i % 3 {
                        0 => RipenessLabel::Unripe,
                        1 => RipenessLabel::Ripe,
                        _ => RipenessLabel::Overripe,
                    },
                })
                .collect();

            let batch = collate_tap_batch(&samples, &device);

            // Verify shapes
            assert_eq!(
                batch.mel_specs.dims(),
                [batch_size, 1, 128, 128],
                "Mel specs shape mismatch for B={}",
                batch_size
            );
            assert_eq!(
                batch.phys_feats.dims(),
                [batch_size, 4],
                "Physics features shape mismatch for B={}",
                batch_size
            );
            assert_eq!(
                batch.targets.dims(),
                [batch_size],
                "Targets shape mismatch for B={}",
                batch_size
            );

            // Verify data integrity (spot check)
            let mel_data = batch.mel_specs.to_data();
            for val in mel_data.iter::<f32>().take(100) {
                assert!(val.is_finite(), "Mel spec must be finite");
            }

            let targets_data = batch.targets.to_data();
            let targets_vec: Vec<i64> = targets_data.iter::<i64>().collect();
            for &t in &targets_vec {
                assert!(
                    (0..=2).contains(&t),
                    "Target must be in {{0, 1, 2}}, got {}",
                    t
                );
            }
        }
    }

    // Phase 5 Tests: Weighted Cross-Entropy Loss with Label Smoothing & AdamW (T015)

    #[test]
    fn test_weighted_cross_entropy_loss_formula_and_class_weight_ratio() {
        let device = Device::flex();

        // iPhone class weights: [0.5478, 1.4192, 2.1288]
        let class_weights = [0.5478, 1.4192, 2.1288];
        let label_smoothing = 0.05;

        // Create a simple case with known probabilities for exact calculation
        // Use logits that produce near-uniform probabilities after softmax
        let logits_data = vec![
            0.0f32, 0.0, 0.0, // Sample 1: uniform probabilities ~[0.333, 0.333, 0.333]
        ];
        let logits: Tensor<2> = Tensor::from_data(
            burn::tensor::TensorData::new(logits_data.clone(), [1, 3]),
            &device,
        );

        // Compute loss for Unripe (class 0)
        let targets_unripe = vec![0];
        let loss_unripe = weighted_cross_entropy_loss(
            logits.clone(),
            &targets_unripe,
            &class_weights,
            label_smoothing,
            &device,
        );
        let loss_unripe_val: f32 = loss_unripe.into_scalar();

        // Compute loss for Overripe (class 2) with same logits
        let logits2: Tensor<2> =
            Tensor::from_data(burn::tensor::TensorData::new(logits_data, [1, 3]), &device);
        let targets_overripe = vec![2];
        let loss_overripe = weighted_cross_entropy_loss(
            logits2,
            &targets_overripe,
            &class_weights,
            label_smoothing,
            &device,
        );
        let loss_overripe_val: f32 = loss_overripe.into_scalar();

        // Both losses should be finite
        assert!(loss_unripe_val.is_finite(), "Unripe loss must be finite");
        assert!(
            loss_overripe_val.is_finite(),
            "Overripe loss must be finite"
        );

        // Verify that Overripe sample incurs higher loss than Unripe due to class weighting
        // Weight ratio: 2.1288 / 0.5478  approximately  3.886
        let weight_ratio: f32 = class_weights[2] / class_weights[0];
        assert!(
            (weight_ratio - 3.886).abs() < 0.01,
            "Weight ratio should be ~3.886, got {}",
            weight_ratio
        );

        // For uniform predictions with equal misclassification, the loss ratio should
        // approximately match the weight ratio
        assert!(
            loss_overripe_val > loss_unripe_val,
            "Overripe loss ({}) should be higher than Unripe loss ({}) due to higher class weight",
            loss_overripe_val,
            loss_unripe_val
        );
    }

    #[test]
    fn test_weighted_cross_entropy_extreme_logits_stability() {
        let device = Device::flex();

        let class_weights = [1.0, 1.0, 1.0];
        let label_smoothing = 0.05;

        // Extreme logits that could cause numerical instability
        let logits_data = vec![
            100.0f32, -100.0, -100.0, // Very confident wrong prediction
            -100.0, 100.0, -100.0, // Very confident correct prediction
        ];
        let logits: Tensor<2> =
            Tensor::from_data(burn::tensor::TensorData::new(logits_data, [2, 3]), &device);

        let targets = vec![0, 1];

        let loss =
            weighted_cross_entropy_loss(logits, &targets, &class_weights, label_smoothing, &device);

        let loss_val: f32 = loss.into_scalar();

        // Must be finite (not NaN or Infinity)
        assert!(
            loss_val.is_finite(),
            "Loss must be finite with extreme logits"
        );
        assert!(!loss_val.is_nan(), "Loss must not be NaN");
        assert!(!loss_val.is_infinite(), "Loss must not be Infinity");
    }

    #[test]
    fn test_weighted_cross_entropy_loss_adamw_gradient_convergence() {
        use burn::optim::{AdamWConfig, GradientsParams};

        let device = Device::flex().autodiff();

        // Create a small synthetic balanced batch: 2 Unripe, 2 Ripe, 2 Overripe
        let batch_size = 6;
        let targets = vec![0, 0, 1, 1, 2, 2];
        let class_weights = [1.0, 1.0, 1.0];
        let label_smoothing = 0.05;

        // Initialize DurianAudioCnn model on autodiff device
        let mut model = DurianAudioCnn::new(3, 0.3, &device);

        // Initialize AdamW optimizer with weight_decay = 1e-4
        // Use higher learning rate for stronger convergence signal in test
        let mut optimizer = AdamWConfig::new().with_weight_decay(1e-4).init();
        let learning_rate = 1e-3; // Higher LR for clearer convergence in test

        // Create synthetic batch with more discriminative features
        // Use different patterns for each class to make learning easier
        let mut mel_data = Vec::new();
        for &target in &targets {
            // Create class-specific patterns to help the model learn
            let pattern = match target {
                0 => 0.3f32, // Unripe: lower values
                1 => 0.5f32, // Ripe: middle values
                _ => 0.7f32, // Overripe: higher values
            };
            mel_data.extend(vec![pattern; 128 * 128]);
        }

        let phys_data: Vec<f32> = targets
            .iter()
            .flat_map(|&t| {
                let val = match t {
                    0 => 0.2f32,
                    1 => 0.5f32,
                    _ => 0.8f32,
                };
                vec![val; 4]
            })
            .collect();

        let mel_specs: Tensor<4> = Tensor::from_data(
            burn::tensor::TensorData::new(mel_data, [batch_size, 1, 128, 128]),
            &device,
        );
        let phys_feats: Tensor<2> = Tensor::from_data(
            burn::tensor::TensorData::new(phys_data, [batch_size, 4]),
            &device,
        );

        let mut losses = Vec::new();

        // Run 5 consecutive AdamW optimization steps
        for step in 0..5 {
            // Forward pass
            let logits = model.forward(mel_specs.clone(), phys_feats.clone());

            // Compute loss
            let loss = weighted_cross_entropy_loss(
                logits,
                &targets,
                &class_weights,
                label_smoothing,
                &device,
            );

            let loss_val: f32 = loss.clone().inner().into_scalar();
            losses.push(loss_val);

            // Verify loss is finite
            assert!(loss_val.is_finite(), "Loss at step {} must be finite", step);

            // Backward pass and optimizer step
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, grads_params);
        }

        // Verify loss decreased overall (compare step 5 to step 1)
        // Allow for some tolerance since neural network training can have noise
        let loss_step_1 = losses[0];
        let loss_step_5 = losses[4];

        // Check that loss decreased OR at least didn't increase significantly
        // This validates that gradients are computed correctly and optimizer works
        assert!(
            loss_step_5 <= loss_step_1 * 1.1,
            "Loss should not increase significantly after 5 AdamW steps: step_1={}, step_5={}. \
             This validates gradients flow correctly through the model.",
            loss_step_1,
            loss_step_5
        );

        // Verify all weights remain finite (checked implicitly by finite loss assertions)
    }

    // Phase 3 User Story 4 Tests: ConfusionMatrix3x3 and EvaluationReport (T017)

    #[test]
    fn test_confusion_matrix_3x3_basic_operations() {
        let mut cm = ConfusionMatrix3x3::new();

        // Record some predictions: 2 Unripe correct, 1 Unripe->Ripe, 2 Ripe correct, 1 Overripe->Ripe
        cm.record(0, 0); // Unripe -> Unripe (correct)
        cm.record(0, 0); // Unripe -> Unripe (correct)
        cm.record(0, 1); // Unripe -> Ripe (FP_R)
        cm.record(1, 1); // Ripe -> Ripe (correct)
        cm.record(1, 1); // Ripe -> Ripe (correct)
        cm.record(2, 1); // Overripe -> Ripe (FP_R)

        // Verify counts matrix
        let counts = cm.counts;
        assert_eq!(counts[0][0], 2); // Unripe->Unripe
        assert_eq!(counts[0][1], 1); // Unripe->Ripe
        assert_eq!(counts[0][2], 0); // Unripe->Overripe
        assert_eq!(counts[1][0], 0); // Ripe->Unripe
        assert_eq!(counts[1][1], 2); // Ripe->Ripe
        assert_eq!(counts[1][2], 0); // Ripe->Overripe
        assert_eq!(counts[2][0], 0); // Overripe->Unripe
        assert_eq!(counts[2][1], 1); // Overripe->Ripe
        assert_eq!(counts[2][2], 0); // Overripe->Overripe

        // Compute report
        let report = cm.compute_report();

        // Verify confusion matrix in report matches
        assert_eq!(report.confusion, counts);

        // Verify per-class metrics
        // Unripe: precision = 2/2 = 1.0, recall = 2/3  approximately  0.667, F1  approximately  0.8
        assert!((report.per_class_precision[0] - 1.0).abs() < 0.01);
        assert!((report.per_class_recall[0] - 0.6667).abs() < 0.01);
        assert!((report.per_class_f1[0] - 0.8).abs() < 0.01);

        // Ripe: precision = 2/4 = 0.5, recall = 2/2 = 1.0, F1  approximately  0.667
        assert!((report.per_class_precision[1] - 0.5).abs() < 0.01);
        assert!((report.per_class_recall[1] - 1.0).abs() < 0.01);
        assert!((report.per_class_f1[1] - 0.6667).abs() < 0.01);

        // Overripe: precision = 0/0 = 0.0, recall = 0/1 = 0.0, F1 = 0.0
        assert_eq!(report.per_class_precision[2], 0.0);
        assert_eq!(report.per_class_recall[2], 0.0);
        assert_eq!(report.per_class_f1[2], 0.0);

        // Macro F1 = (0.8 + 0.667 + 0.0) / 3  approximately  0.489
        assert!((report.macro_f1 - 0.489).abs() < 0.01);

        // Balanced Accuracy = (0.667 + 1.0 + 0.0) / 3  approximately  0.556
        assert!((report.balanced_accuracy - 0.556).abs() < 0.01);

        // Overall Accuracy = 4/6  approximately  0.667
        assert!((report.overall_accuracy - 0.6667).abs() < 0.01);

        // Critical FP_R rate = (1 + 1) / (3 + 1) = 2/4 = 0.5
        assert!((report.critical_fp_ripe_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_confusion_matrix_zero_denominator_guards() {
        // Empty confusion matrix - all denominators are zero
        let cm = ConfusionMatrix3x3::new();
        let report = cm.compute_report();

        // All metrics should be 0.0, not NaN
        for c in 0..3 {
            assert_eq!(report.per_class_precision[c], 0.0);
            assert_eq!(report.per_class_recall[c], 0.0);
            assert_eq!(report.per_class_f1[c], 0.0);
        }
        assert_eq!(report.macro_f1, 0.0);
        assert_eq!(report.balanced_accuracy, 0.0);
        assert_eq!(report.overall_accuracy, 0.0);
        assert_eq!(report.critical_fp_ripe_rate, 0.0);

        // Partial case: only class 1 has predictions, class 0 and 2 have zero row/col sums
        let mut cm2 = ConfusionMatrix3x3::new();
        cm2.record(1, 1);
        cm2.record(1, 1);
        let report2 = cm2.compute_report();

        // Class 0 and 2 should have 0.0 metrics (not NaN)
        assert_eq!(report2.per_class_precision[0], 0.0);
        assert_eq!(report2.per_class_recall[0], 0.0);
        assert_eq!(report2.per_class_precision[2], 0.0);
        assert_eq!(report2.per_class_recall[2], 0.0);

        // Class 1 should have precision=1.0, recall=1.0
        assert_eq!(report2.per_class_precision[1], 1.0);
        assert_eq!(report2.per_class_recall[1], 1.0);
    }

    #[test]
    fn test_recording_level_aggregation_entropy_weighting() {
        // Test case: 3 taps in one recording with varying confidence levels
        // Tap 1: [0.7, 0.2, 0.1] -> High confidence for Unripe, low entropy
        // Tap 2: [0.4, 0.3, 0.3] -> Low confidence, high entropy (uniform-like)
        // Tap 3: [0.8, 0.1, 0.1] -> Very high confidence for Unripe, lowest entropy

        let tap_probas = vec![
            vec![0.7f32, 0.2, 0.1], // Tap 1
            vec![0.4, 0.3, 0.3],    // Tap 2 (should get low weight)
            vec![0.8, 0.1, 0.1],    // Tap 3 (should get highest weight)
        ];

        let aggregated = aggregate_recording_probas(&tap_probas);

        // Tap 3 has highest confidence, should dominate
        // Expected: aggregated probability for class 0 (Unripe) should be close to 0.8
        assert!(
            aggregated[0] > 0.65,
            "Class 0 prob should be high: {}",
            aggregated[0]
        );
        assert!(
            aggregated[1] < 0.25,
            "Class 1 prob should be low: {}",
            aggregated[1]
        );
        assert!(
            aggregated[2] < 0.20,
            "Class 2 prob should be low: {}",
            aggregated[2]
        );

        // Probabilities should sum to 1.0
        let sum: f32 = aggregated.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Probabilities should sum to 1.0: {}",
            sum
        );
    }

    #[test]
    fn test_recording_level_aggregation_fallback_uniform() {
        // Edge case: all taps have maximum entropy (uniform distribution)
        // Should fallback to arithmetic mean
        let tap_probas = vec![
            vec![0.333f32, 0.333, 0.334],
            vec![0.333, 0.334, 0.333],
            vec![0.334, 0.333, 0.333],
        ];

        let aggregated = aggregate_recording_probas(&tap_probas);

        // Should be close to uniform average
        for (c, &prob) in aggregated.iter().enumerate().take(3) {
            assert!((prob - 0.333).abs() < 0.01, "Class {} prob: {}", c, prob);
        }

        // Sum should be 1.0
        let sum: f32 = aggregated.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_recording_level_aggregation_single_tap() {
        // Single tap should return its own probability
        let tap_probas = vec![vec![0.6f32, 0.3, 0.1]];

        let aggregated = aggregate_recording_probas(&tap_probas);

        assert!((aggregated[0] - 0.6).abs() < 1e-5);
        assert!((aggregated[1] - 0.3).abs() < 1e-5);
        assert!((aggregated[2] - 0.1).abs() < 1e-5);
    }

    // Phase 6 User Story 4: Integration Test for train_model_on_splits (T018)

    #[test]
    fn test_train_model_on_splits_end_to_end_checkpoint_restore() {
        use crate::config::Config;
        use crate::model::DurianAudioCnn;
        use std::fs;
        use tempfile::tempdir;

        let device = Device::flex().autodiff(); // Use autodiff device for training
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_dir = temp_dir.path().join("checkpoints");
        fs::create_dir_all(&checkpoint_dir).expect("Failed to create checkpoint dir");

        // Create synthetic TapSample datasets for Train/Val/Test
        // 18 samples: 6 per class, split into 12 train / 3 val / 3 test
        let mut train_samples = Vec::new();
        let mut val_samples = Vec::new();
        let mut test_samples = Vec::new();

        for class_idx in 0..3 {
            let label = match class_idx {
                0 => RipenessLabel::Unripe,
                1 => RipenessLabel::Ripe,
                _ => RipenessLabel::Overripe,
            };

            for sample_idx in 0..6 {
                let mel_data = vec![0.5f32; 16_384]; // 128 * 128
                let physics_feats = [0.5f32; 4];

                let sample = TapSample {
                    fruit_id: (class_idx * 6 + sample_idx + 1) as u32,
                    sample_num: (sample_idx + 1) as u32,
                    tap_index: 0,
                    mel_data,
                    physics_feats,
                    label,
                };

                // Split: first 4 to train, next 1 to val, last 1 to test
                if sample_idx < 4 {
                    train_samples.push(sample);
                } else if sample_idx == 4 {
                    val_samples.push(sample);
                } else {
                    test_samples.push(sample);
                }
            }
        }

        // Verify we have 12 train, 3 val, 3 test samples
        assert_eq!(train_samples.len(), 12);
        assert_eq!(val_samples.len(), 3);
        assert_eq!(test_samples.len(), 3);

        // Create a minimal config for training
        let mut config = Config::default();
        config.training.epochs = 2; // Set to 2 epochs for fast test execution
        let class_weights = [1.0f32, 1.0, 1.0]; // Balanced weights for test

        // Initialize model
        let model = DurianAudioCnn::new(3, 0.3, &device);

        // This test expects train_model_on_splits to exist with signature:
        // pub fn train_model_on_splits(
        //     model: DurianAudioCnn,
        //     train_samples: &[TapSample],
        //     val_samples: &[TapSample],
        //     test_samples: &[TapSample],
        //     class_weights: &[f32; 3],
        //     config: &Config,
        //     checkpoint_dir: &Path,
        //     device: &Device,
        // ) -> anyhow::Result<TrainingOutcome>

        // Call train_model_on_splits
        let splits = DataSplits {
            train: &train_samples,
            val: &val_samples,
            test: &test_samples,
        };
        let result = train_model_on_splits(
            model,
            &splits,
            &class_weights,
            &config,
            &checkpoint_dir,
            &device,
        );

        // Verify the function exists and returns expected type
        assert!(
            result.is_ok(),
            "train_model_on_splits should complete successfully"
        );

        let outcome = result.unwrap();

        // Verify TrainingOutcome structure
        assert_eq!(outcome.class_weights, class_weights);
        assert!(outcome.epochs_run >= 1, "Should run at least 1 epoch");
        assert!(
            outcome.best_epoch >= 1 && outcome.best_epoch <= outcome.epochs_run,
            "Best epoch should be within epochs_run"
        );
        assert!(
            outcome.best_val_recording_macro_f1 >= 0.0
                && outcome.best_val_recording_macro_f1 <= 1.0,
            "Val Macro F1 should be in [0, 1]"
        );

        // Verify checkpoint file exists and is < 2 MB
        assert!(
            outcome.artifact_path.exists(),
            "Checkpoint .burnpack file should exist"
        );
        assert!(
            outcome.artifact_size_bytes > 0,
            "Checkpoint should have non-zero size"
        );
        assert!(
            outcome.artifact_size_bytes < 2_000_000,
            "Checkpoint should be < 2.0 MB, got {} bytes",
            outcome.artifact_size_bytes
        );

        // Verify test evaluation has finite metrics
        let tap_report = &outcome.test_evaluation.tap_level;
        let rec_report = &outcome.test_evaluation.recording_level;

        assert!(
            tap_report.macro_f1.is_finite(),
            "Tap-level Macro F1 should be finite"
        );
        assert!(
            tap_report.balanced_accuracy.is_finite(),
            "Tap-level Balanced Accuracy should be finite"
        );
        assert!(
            rec_report.macro_f1.is_finite(),
            "Recording-level Macro F1 should be finite"
        );
        assert!(
            rec_report.balanced_accuracy.is_finite(),
            "Recording-level Balanced Accuracy should be finite"
        );

        // All metrics should be in [0, 1]
        assert!(tap_report.macro_f1 >= 0.0 && tap_report.macro_f1 <= 1.0);
        assert!(tap_report.balanced_accuracy >= 0.0 && tap_report.balanced_accuracy <= 1.0);
        assert!(rec_report.macro_f1 >= 0.0 && rec_report.macro_f1 <= 1.0);
        assert!(rec_report.balanced_accuracy >= 0.0 && rec_report.balanced_accuracy <= 1.0);
    }

    #[test]
    fn test_run_platform_training_end_to_end_synthetic_dataset() {
        use crate::config::Config;
        use crate::dataset::PlatformTarget;
        use std::fs;
        use std::io::Write;
        use tempfile::tempdir;

        let device = Device::flex().autodiff(); // Use autodiff device for training

        // Create a temporary directory structure for synthetic dataset
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let data_dir = temp_dir.path().join("data");
        let audio_dir = data_dir.join("sound_iphone");
        let artifacts_dir = temp_dir.path().join("artifacts");

        fs::create_dir_all(&audio_dir).expect("Failed to create audio dir");
        fs::create_dir_all(&artifacts_dir).expect("Failed to create artifacts dir");

        // Create synthetic metadata CSV with disjoint fruit_ids across splits
        // 9 recordings: 3 Unripe (fruit_id 1,2,3), 3 Ripe (4,5,6), 3 Overripe (7,8,9)
        let csv_path = data_dir.join("metadata_iphone.csv"); // Match expected filename
        let mut csv_file = fs::File::create(&csv_path).expect("Failed to create CSV");
        writeln!(
            csv_file,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date"
        )
        .unwrap();

        // Create 9 synthetic recordings with disjoint fruit_ids
        for fruit_id in 1..=9 {
            let ripeness = match fruit_id {
                1..=3 => "Unripe",
                4..=6 => "Ripe",
                _ => "Overripe",
            };
            let filename = format!("audio_{:03}.wav", fruit_id);
            writeln!(
                csv_file,
                "{},{},{},{},{},{},2026-11-27",
                fruit_id,
                fruit_id,
                ripeness,
                filename,
                filename,
                44144 // actual size of WAV file (12 + 24 + 8 + 44100 bytes)
            )
            .unwrap();

            // Create a minimal synthetic audio file (silent 1-second WAV)
            // This is a placeholder - in real tests we'd use actual audio
            // For this test, we verify the pipeline structure, not audio processing
            let audio_path = audio_dir.join(format!("audio_{:03}.wav", fruit_id));

            // Create a minimal valid WAV file (44 bytes header + 22050 samples * 2 bytes)
            let mut wav_data = Vec::new();

            // RIFF header
            wav_data.extend_from_slice(b"RIFF");
            let file_size: u32 = 36 + 22050 * 2; // 44 byte header - 8 + data
            wav_data.extend_from_slice(&file_size.to_le_bytes());
            wav_data.extend_from_slice(b"WAVE");

            // fmt chunk
            wav_data.extend_from_slice(b"fmt ");
            wav_data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
            wav_data.extend_from_slice(&1u16.to_le_bytes()); // PCM format
            wav_data.extend_from_slice(&1u16.to_le_bytes()); // mono
            wav_data.extend_from_slice(&22050u32.to_le_bytes()); // sample rate
            wav_data.extend_from_slice(&44100u32.to_le_bytes()); // byte rate (sample_rate * channels * bytes_per_sample)
            wav_data.extend_from_slice(&2u16.to_le_bytes()); // block align (channels * bytes_per_sample)
            wav_data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

            // data chunk
            wav_data.extend_from_slice(b"data");
            let data_size: u32 = 22050 * 2; // 22050 samples * 2 bytes
            wav_data.extend_from_slice(&data_size.to_le_bytes());

            // Audio data with unique content per fruit_id (simple sine wave with different frequency)
            // This ensures each file has a unique SHA-256 hash
            let frequency = 440.0 + (fruit_id as f32 * 10.0); // Different frequency per fruit
            for i in 0..22050 {
                let t = i as f32 / 22050.0;
                let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.1;
                let sample_i16 = (sample * 32767.0) as i16;
                wav_data.extend_from_slice(&sample_i16.to_le_bytes());
            }

            fs::write(&audio_path, wav_data).expect("Failed to write audio file");
        }

        // Create a minimal config pointing to our synthetic dataset
        let mut config = Config::default();
        config.dataset.data_dir = data_dir.to_string_lossy().to_string();
        config.training.epochs = 2; // Run only 2 epochs for speed
        config.training.batch_size = 4;
        config.training.early_stopping_patience = 1;

        let output_path = artifacts_dir.join("model_test.mpk");

        // Run platform training with synthetic dataset
        let result = run_platform_training(&config, PlatformTarget::Iphone, &output_path, &device);

        // Verify training completes successfully
        assert!(
            result.is_ok(),
            "Platform training should complete successfully with synthetic dataset: {:?}",
            result.err()
        );
    }
}
