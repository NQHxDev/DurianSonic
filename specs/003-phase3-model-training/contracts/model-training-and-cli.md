# Phase 3 Interface Contracts: Rust Model, Training Pipeline & CLI API

**Feature**: `003-phase3-model-training`
**Date**: 2026-11-26

---

## 1. Crate Root Exports (`src/lib.rs`)

Phase 3 registers `pub mod model;` in `src/lib.rs` alongside the existing modules (`FR-001`, `CHK001`):

```rust
pub mod augment;
pub mod config;
pub mod dataset;
pub mod dsp;
pub mod model;
pub mod training;
```

---

## 2. Module `durian_sonic::model` (`src/model.rs`)

Pure neural network architecture and `.mpk` serialization helpers with zero dependency on training loop state, allowing direct reuse by Phase 4 mobile inference (`CHK031`).

```rust
use crate::config::ModelConfig;
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
        Relu,
    },
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::activation::softmax,
};
use std::path::{Path, PathBuf};

/// 2D Convolutional block: Conv2d(3x3, stride=2, pad=Same) -> BatchNorm -> ReLU.
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    pub conv: Conv2d<B>,
    pub norm: BatchNorm<B>,
    pub activation: Relu,
}

impl<B: Backend> ConvBlock<B> {
    /// Initializes a 3x3 stride-2 Same-padded ConvBlock halving spatial resolution.
    pub fn new(in_ch: usize, out_ch: usize, device: &B::Device) -> Self;

    /// Maps `[B, in_ch, H, W]` -> `[B, out_ch, H / 2, W / 2]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4>;
}

/// Hybrid 2D Log-Mel Spectrogram CNN + 4-D Acoustic Physics neural network (~420K params).
#[derive(Module, Debug)]
pub struct DurianAudioCnn<B: Backend> {
    pub block1: ConvBlock<B>,
    pub block2: ConvBlock<B>,
    pub block3: ConvBlock<B>,
    pub block4: ConvBlock<B>,
    pub global_pool: AdaptiveAvgPool2d,
    pub phys_proj: Linear<B>,
    pub dropout: Dropout,
    pub fc1: Linear<B>,
    pub fc_out: Linear<B>,
    pub activation: Relu,
}

impl<B: Backend> DurianAudioCnn<B> {
    /// Initializes DurianAudioCnn with `num_classes` (default 3) and `dropout_rate` (default 0.3).
    pub fn new(num_classes: usize, dropout_rate: f64, device: &B::Device) -> Self;

    /// Convenience constructor initializing from `ModelConfig` (`config.num_classes`, `config.dropout`).
    pub fn from_config(config: &ModelConfig, device: &B::Device) -> Self;

    /// Raw logit forward pass:
    /// - `mel_spec`: `[B, 1, 128, 128]` in `[0.0, 1.0]`
    /// - `phys_feats`: `[B, 4]` in `[0.0, 1.0]`
    /// Returns raw unnormalized logits `[B, 3]` for `[Unripe, Ripe, Overripe]`.
    pub fn forward(&self, mel_spec: Tensor<B, 4>, phys_feats: Tensor<B, 2>) -> Tensor<B, 2>;

    /// Normalized probability forward pass applying `softmax(logits, 1)` along class dimension `dim = 1`.
    /// Returns `[B, 3]` probabilities in `[0.0, 1.0]` where each row sums to `1.0` (tolerance `< 1e-5`).
    pub fn forward_proba(&self, mel_spec: Tensor<B, 4>, phys_feats: Tensor<B, 2>) -> Tensor<B, 2>;

    /// Saves full-precision FP32 model weights to `<stem>.mpk` using `NamedMpkFileRecorder::<FullPrecisionSettings>`.
    /// Automatically creates missing parent directories (`std::fs::create_dir_all`) and normalizes
    /// the path so passing either `"artifacts/durian_iphone"` or `"artifacts/durian_iphone.mpk"`
    /// writes deterministically to `"artifacts/durian_iphone.mpk"` and returns the final `.mpk` `PathBuf`.
    pub fn save_mpk<P: AsRef<Path>>(&self, output_path: P) -> anyhow::Result<PathBuf>;

    /// Loads full-precision FP32 model weights from `<stem>.mpk` into `self` on `device`.
    pub fn load_mpk<P: AsRef<Path>>(self, input_path: P, device: &B::Device) -> anyhow::Result<Self>;
}
```

---

## 3. Module `durian_sonic::training` (`src/training.rs`)

Preserves existing `GradientDescent` and `LearningRateSchedule` symbols for benchmark compatibility (`CHK016`, `CHK040`) and provides the complete Phase 3 dataset extraction, minority oversampling, batch collation, weighted loss, dual-level evaluation, early stopping, and training loop APIs.

```rust
use crate::config::{Config, DspConfig};
use crate::dataset::{AudioMetadataRecord, PlatformTarget, RipenessLabel};
use crate::model::DurianAudioCnn;
use burn::{
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use std::path::{Path, PathBuf};

/// Single 0.8s tap window feature sample with fruit/recording lineage.
#[derive(Debug, Clone, PartialEq)]
pub struct TapSample {
    pub fruit_id: u32,
    pub sample_num: u32,
    pub tap_index: usize,
    pub mel_data: Vec<f32>,      // Length 16,384 (row-major 128 x 128) in [0.0, 1.0]
    pub physics_feats: [f32; 4], // [f_peak, C_s, R_85, beta_decay] in [0.0, 1.0]
    pub label: RipenessLabel,
}

/// Collated tensor batch on device for DurianAudioCnn.
#[derive(Clone, Debug)]
pub struct DurianBatch<B: Backend> {
    pub mel_specs: Tensor<B, 4>,    // [B, 1, 128, 128]
    pub phys_feats: Tensor<B, 2>,   // [B, 4]
    pub targets: Tensor<B, 1, Int>, // [B]
}

/// Extracts `TapSample` items from a slice of `AudioMetadataRecord` items.
/// - Precomputes `mel_basis` once across the entire split.
/// - When `is_train == true`: applies Onset-Window Minority Oversampling
///   (`1x Unripe`, `2x Ripe` with ±25ms jitter + SNR 15-35dB noise + SpecAugment(12,16,0.0),
///   `3x Overripe` with ±25ms jitter + ±3dB spectral tilt + SpecAugment(12,16,0.0)).
/// - When `is_train == false`: extracts strictly `1` canonical window per detected peak with zero augmentation.
/// - Zero-onset fallback: if a recording yields `0` detected peaks, logs a warning and extracts `1`
///   fallback 0.8s window centered at the recording's maximum RMS frame.
pub fn extract_dataset_tap_samples(
    records: &[AudioMetadataRecord],
    audio_dir: &Path,
    dsp_config: &DspConfig,
    is_train: bool,
) -> anyhow::Result<Vec<TapSample>>;

/// Extracts `TapSample` items from an in-memory mono `22,050 Hz` waveform for a single recording.
/// Useful for deterministic unit testing of `is_train = true` (1x/2x/3x) vs `is_train = false` (1x/1x/1x)
/// and zero-onset fallback without disk I/O.
pub fn extract_tap_samples_from_waveform(
    samples: &[f32],
    fruit_id: u32,
    sample_num: u32,
    label: RipenessLabel,
    dsp_config: &DspConfig,
    mel_basis: &ndarray::Array2<f32>,
    is_train: bool,
) -> Vec<TapSample>;

/// Collates a dynamic-length slice `&[TapSample]` (`B = samples.len() >= 1`) into a `DurianBatch<B>`.
pub fn collate_tap_batch<B: Backend>(
    samples: &[TapSample],
    device: &B::Device,
) -> DurianBatch<B>;

/// Computes platform-specific Weighted Cross-Entropy Loss with Label Smoothing (`epsilon = 0.05`):
/// `q_i(c) = (1 - epsilon) * I(c == y_i) + epsilon / 3`
/// `L = (1 / B) * sum_i ( - sum_c w_c * q_i(c) * log_softmax(logits_i)_c )`
/// Returns a 1-element scalar tensor `Tensor<B, 1>`.
pub fn weighted_cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 2>,
    targets: &[usize],
    class_weights: &[f32; 3],
    label_smoothing: f32,
    device: &B::Device,
) -> Tensor<B, 1>;

/// 3x3 Confusion Matrix for `[0: Unripe, 1: Ripe, 2: Overripe]`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfusionMatrix3x3 {
    pub counts: [[usize; 3]; 3],
}

/// Complete classification quality report derived from `ConfusionMatrix3x3`.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationReport {
    pub confusion: ConfusionMatrix3x3,
    pub per_class_precision: [f32; 3],
    pub per_class_recall: [f32; 3],
    pub per_class_f1: [f32; 3],
    pub macro_f1: f32,
    pub balanced_accuracy: f32,
    pub overall_accuracy: f32,
    pub critical_fp_ripe_rate: f32,
}

impl ConfusionMatrix3x3 {
    pub fn record(&mut self, actual: usize, predicted: usize);
    pub fn compute_report(&self) -> EvaluationReport;
}

/// Combined Tap-Level and Entropy-Weighted Recording-Level evaluation reports.
#[derive(Debug, Clone, PartialEq)]
pub struct DualLevelEvaluation {
    pub tap_level: EvaluationReport,
    pub recording_level: EvaluationReport,
}

/// Aggregates tap probability vectors `[P(Unripe), P(Ripe), P(Overripe)]` for a single recording
/// using Entropy confidence weights `alpha_k = 1 - H(P_k) / ln(3)` with uniform arithmetic mean
/// fallback when `sum(alpha_k) <= 1e-6`.
pub fn aggregate_recording_probabilities(tap_probs: &[[f32; 3]]) -> [f32; 3];

/// Evaluates `DurianAudioCnn<B>` on a slice of `TapSample` items in inference mode (`batch_size`),
/// returning both `Tap-Level` and `Recording-Level` (grouped by `sample_num`) reports.
pub fn evaluate_model<B: Backend>(
    model: &DurianAudioCnn<B>,
    samples: &[TapSample],
    batch_size: usize,
    device: &B::Device,
) -> DualLevelEvaluation;

/// Action returned by `EarlyStopping::step` after each validation epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingAction {
    Improved,
    Continue,
    Stop,
}

/// Tracks validation Recording-Level Macro F1-Score and patience counter.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    pub patience: usize,
    pub best_macro_f1: f32,
    pub best_epoch: usize,
    pub patience_counter: usize,
}

impl EarlyStopping {
    /// Initializes with `best_macro_f1 = -1.0` so Epoch 1 always saves a valid checkpoint.
    pub fn new(patience: usize) -> Self;
    pub fn step(&mut self, epoch: usize, val_recording_macro_f1: f32) -> EarlyStoppingAction;
}

/// Summary artifact and metrics returned upon completion of platform training.
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

/// Runs the generic `<B: AutodiffBackend>` training loop over pre-extracted `Train`, `Val`, and `Test`
/// `TapSample` splits using `AdamW`, saving the best checkpoint to `output_path` whenever
/// `val_recording_macro_f1` improves, stopping early after `early_stopping_patience` non-improving epochs,
/// restoring the best `.mpk` checkpoint, and evaluating on `test_samples`.
pub fn train_model_on_splits<B: AutodiffBackend>(
    config: &Config,
    platform: PlatformTarget,
    class_weights: [f32; 3],
    train_samples: &[TapSample],
    val_samples: &[TapSample],
    test_samples: &[TapSample],
    output_path: &Path,
    device: &B::Device,
) -> anyhow::Result<TrainingOutcome>;

/// End-to-end platform training pipeline: verifies platform metadata (`load_and_verify_metadata`),
/// computes raw inverse class frequency weights (`compute_class_weights(&report.class_counts)`),
/// performs anti-leakage fruit split (`stratified_group_split_by_fruit`), extracts `TapSample` splits,
/// and delegates to `train_model_on_splits`.
pub fn run_platform_training<B: AutodiffBackend>(
    config: &Config,
    platform: PlatformTarget,
    output_path: &Path,
    device: &B::Device,
) -> anyhow::Result<TrainingOutcome>;
```

---

## 4. CLI Contract: `train` Subcommand (`src/main.rs`) (`FR-010`, `CHK006`, `CHK021`, `CHK030`)

### 4.1. Command Syntax & Flags

```text
durian-sonic train --platform <iphone|android> [--config <PATH>] [--data-dir <PATH>] [--output <PATH>]
```

| Flag / Option | Required | Default Value                                           | Validation / Behavior                                                                                     |
| :------------ | :------: | :------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------- |
| `--platform`  |   Yes    | —                                                       | Case-insensitive `iphone` (`PlatformTarget::Iphone`) or `android` (`PlatformTarget::Android`)             |
| `--config`    |    No    | `configs/default.toml`                                  | Path to TOML configuration file validated via `Config::from_file`                                         |
| `--data-dir`  |    No    | `config.dataset.data_dir` (`data`)                      | Overrides `config.dataset.data_dir` before resolving `metadata_<platform>.csv` and `sound_<platform>/`    |
| `--output`    |    No    | `artifacts/durian_iphone` or `artifacts/durian_android` | Output stem or `.mpk` path; creates parent directories (`create_dir_all`) and saves `<stem>.mpk` (`<2MB`) |

### 4.2. Default Compute Backend & Output Report

- Initializes `type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;` with `NdArrayDevice::Cpu` by default for 100% deterministic execution across developer machines and CI (`Clarifications`, `FR-010`).
- Prints an emoji-free console report containing:
    - Platform name, verified file/fruit counts, and computed class weights $[w_0, w_1, w_2]$.
    - Fruit-disjoint split counts (`Train`, `Val`, `Test` files, fruits, and extracted `TapSample` counts).
    - Per-epoch training loss, validation `Tap-Level` Macro F1, validation `Recording-Level` Macro F1, and checkpoint save indicator.
    - Best checkpoint path (`artifacts/durian_iphone.mpk` or `artifacts/durian_android.mpk`) and file size in bytes/MB (`< 2.0 MB`).
    - Final `Test` split evaluation tables (`3x3 Confusion Matrix`, `Per-Class Precision / Recall / F1`, `Macro F1-Score`, `Balanced Accuracy`, `Overall Accuracy`, and `Critical FP_Ripe Rate`) at both **`Tap-Level`** and **`Recording-Level`**.
