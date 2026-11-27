# Implementation Plan: Phase 3 - Hybrid CNN Architecture & Training Pipeline in Burn

**Branch**: `003-phase3-model-training` | **Date**: 2026-11-26 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-phase3-model-training/spec.md` and requirements checklists (`checklists/model.md`, `checklists/training.md`, `checklists/requirements.md`)

## Summary

Phase 3 implements the pure-Rust deep learning architecture (`DurianAudioCnn`), imbalance-aware training pipeline, dual-level evaluation (`Tap-Level` and Entropy-weighted `Recording-Level`), Macro-F1 early stopping, and dual-platform CLI workflow (`durian_iphone.mpk` and `durian_android.mpk`) using the `burn 0.22.1` framework:

1. **Hybrid 2D Spectrogram CNN & Acoustic Physics Model (`src/model.rs`, `src/lib.rs`)**: Implement `ConvBlock<B: Backend>` (`Conv2d(3x3, stride=2, pad=Same) -> BatchNorm -> ReLU`) and `DurianAudioCnn<B: Backend>` (~`408,019` trainable parameters, `< 2.0 MB` FP32 `.mpk`), combining a 4-stage 2D Spectrogram Branch (`1 -> 32 -> 64 -> 128 -> 256` + `AdaptiveAvgPool2d([1, 1])` $\to$ `[B, 256]`) with a Physics Embedding Branch (`Linear(4 -> 16)` + `ReLU` $\to$ `[B, 16]`), concatenated via `Tensor::cat(dim=1)` into a `[B, 272]` Late-Fusion head (`Dropout(0.3) -> Linear(272 -> 64) + ReLU -> Linear(64 -> 3)`), exposing both raw logit `forward` and Softmax `forward_proba` along with `.mpk` save/load helpers (`NamedMpkFileRecorder::<FullPrecisionSettings>`).
2. **Tap Sample Extractor, Train-Only Minority Oversampling & Dynamic Batcher (`src/training.rs`)**: Transform anti-leakage `Train / Val / Test` `AudioMetadataRecord` partitions (grouped by `fruit_id 1..=175` in Phase 1) through the Phase 2 DSP pipeline (reusing a single precomputed `mel_basis`) into `TapSample` items (`fruit_id`, `sample_num`, `tap_index`, flattened `16,384` `mel_data`, `4` `physics_feats`, `RipenessLabel`). Enforce **Train-Only Onset-Window Minority Oversampling** (`1x Unripe` canonical; `2x Ripe` with $\pm 25\text{ ms}$ / `551`-sample onset jitter + $\text{SNR} \in [15, 35]\text{ dB}$ noise before `extract_tap_features` and `apply_spec_augment(12, 16, 0.0)` after; `3x Overripe` with $\pm 25\text{ ms}$ onset jitter before `extract_tap_features` and $\pm 3\text{ dB}$ spectral tilt + `apply_spec_augment(12, 16, 0.0)` after), keep `Val` and `Test` strictly at `1` canonical window per peak with zero augmentation, provide a max-RMS `0.8s` fallback window if a recording has `0` detected peaks, and collate dynamic slices `&[TapSample]` ($B \ge 1$) into `DurianBatch<B>`.
3. **Platform-Specific Weighted Cross-Entropy Loss with Label Smoothing & `AdamW` Optimization (`src/training.rs`)**: Implement `weighted_cross_entropy_loss<B: Backend>` with label smoothing $\epsilon = 0.05$ ($q_i(c) = (1-\epsilon)\mathbb{I}(c=y_i) + \epsilon/3$), platform-specific inverse class frequency weights computed from raw metadata reports prior to tap slicing (`compute_class_weights(&report.class_counts)`: `[0.548, 1.419, 2.129]` on iPhone, `[0.614, 1.535, 1.389]` on Android), and numerically stable `log_softmax(logits, 1)` paired with `AdamW` (`lr = 3e-4`, `weight_decay = 1e-4`).
4. **Dual-Level Evaluator (`Tap-Level` & Entropy-Weighted `Recording-Level`), Macro-F1 Early Stopping & Dual-Platform CLI (`src/training.rs`, `src/main.rs`)**: Evaluate after every epoch in inference mode (`model.valid()`) at both `Tap-Level` and `Recording-Level` (grouping taps by `sample_num` and weighting via Shannon entropy confidence $\alpha_k = 1 - H(P_k)/\ln 3$ with uniform fallback if $\sum \alpha_k \le 10^{-6}$), compute `ConfusionMatrix3x3` and `EvaluationReport` (including `macro_f1`, `balanced_accuracy`, and `critical_fp_ripe_rate` with zero-denominator guards), checkpoint on `val_recording_macro_f1` with `EarlyStopping` (`patience = 12`, `best_macro_f1 = -1.0`), and expose `cargo run --release -- train --platform <iphone|android>` defaulting to `Autodiff<NdArray<f32>>`.

## Technical Context

**Language/Version**: Rust (Edition 2021)

**Primary Dependencies**: `burn 0.22.1` (`train`, `wgpu`, `ndarray`, `autodiff`, `metrics`), `ndarray 0.17.2`, `rand 0.10.4`, `clap 4.6.9`, `serde 1.0.229`, `anyhow 1.0.105`, `tracing 0.1.44` (plus Phase 1/2 modules `symphonia 0.6.1`, `rubato 5.1.0`, `realfft 3.5.1`)

**Storage**: Local filesystem audio datasets (`data/metadata_iphone.csv` + `data/sound_iphone/*.m4a`, `data/metadata_android.csv` + `data/sound_android/*.wav`), configuration (`configs/default.toml`), and FP32 MessagePack model weight artifacts (`artifacts/durian_iphone.mpk`, `artifacts/durian_android.mpk`)

**Testing**: `cargo test` (deterministic CPU unit and integration tests using `burn::backend::NdArray<f32>` and `burn::backend::Autodiff<NdArray<f32>>`, `approx 0.5.1`, `tempfile 3.27.0`) and `cargo bench` (`criterion 0.8.2`)

**Target Platform**: Cross-platform developer workstation & CI (macOS, Linux, Windows) via deterministic `Autodiff<NdArray<f32>>` default backend (generic `<B: AutodiffBackend>` compatible with `Wgpu`), producing portable `.mpk` artifacts for Phase 4 iOS/Android inference

**Project Type**: Rust Library + CLI + Criterion Benchmark Suite (`src/lib.rs`, `src/model.rs`, `src/training.rs`, `src/main.rs`, `benches/training_benchmarks.rs`)

**Performance Goals**:

- **Accuracy & Generalization (`SC-004`)**: Both independent platform models (`durian_iphone.mpk` and `durian_android.mpk`) MUST achieve **Recording-Level Macro F1-Score $\ge 0.85$** and **Recording-Level Balanced Accuracy $\ge 0.85$** on their held-out `Test` split.
- **Artifact Footprint (`SC-005`)**: Each saved FP32 `.mpk` weight file MUST be **`< 2.0 MB`** on disk (~`408K` parameters $\approx 1.63\text{ MB}$).
- **Forward Latency (`Constitution §IV`)**: Single-tap `DurianAudioCnn::<NdArray<f32>>::forward` pass benchmarked in `benches/training_benchmarks.rs` runs in near-real-time (`< 8.0 ms/tap` on CPU).

**Constraints**:

- **Zero Fruit Leakage (`SC-003`)**: Disjoint `fruit_id` partitions (`Train`, `Val`, `Test`) from Phase 1 are preserved across all extracted `TapSample` sets; `Validation` and `Test` splits NEVER receive minority oversampling or data augmentation.
- **Physical Acoustic Integrity (`Constitution §III`, `CHK033`)**: Train-only oversampling composes only Phase 2's physically valid transforms (`apply_onset_jitter`, `add_background_noise`, `apply_spectral_tilt`, `apply_spec_augment`); Pitch Shifting and Time Stretching remain strictly banned.
- **Numerical & Dynamic-Batch Safety (`CHK021`, `CHK026`–`CHK029`)**: Dynamic `batch_size` extraction in `DurianAudioCnn::forward` and `collate_tap_batch`; `log_softmax` for loss calculation; `1e-12` probability floor in Shannon entropy $H(P_k)$; zero-denominator guards returning `0.0` (never `NaN`) in `ConfusionMatrix3x3::compute_report()`; `best_macro_f1 = -1.0` initialization ensuring Epoch 1 always saves a valid checkpoint.
- **Constitutional Compliance**: Strict TDD Red-Green-Refactor, 100% emoji-free codebase and CLI output, no decorative comment separators (`===`, `---`), no unguarded `unwrap()`/`expect()` in library code, and zero clippy warnings (`cargo clippy --all-targets -- -D warnings`).

**Scale/Scope**: Trains 2 independent platform models across `456` recordings (`281` iPhone `.m4a` $\to$ `197 Train / 41 Val / 43 Test` recordings, ~`3,370` canonical taps; `175` Android `.wav` $\to$ `123 Train / 26 Val / 26 Test` recordings, ~`2,100` canonical taps) over up to `80` epochs (`batch_size = 32`, `early_stopping_patience = 12`).

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design._

| Constitutional Principle               | Pre-Research Status | Post-Design Status | Compliance Evidence                                                                                                                                                                                                                                                                                                     |
| :------------------------------------- | :------------------ | :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **I. Clean Code Excellence**           | PASS                | PASS               | No bypass patterns; 100% emoji-free code, comments, and CLI output; explicit `anyhow::Result` propagation for all file I/O, audio decoding, and `.mpk` checkpoint operations; zero unguarded `unwrap()`/`expect()` in `src/model.rs` or `src/training.rs`.                                                              |
| **II. Test-First Development**         | PASS                | PASS               | Failing unit and integration tests specified first for `ConvBlock`, `DurianAudioCnn` forward/proba/save-load, `TapSample` oversampling (`1x/2x/3x` vs `1x/1x/1x`), `collate_tap_batch`, `weighted_cross_entropy_loss` + `AdamW` convergence, `ConfusionMatrix3x3`, Entropy voting, and `EarlyStopping`.                 |
| **III. Audio Quality Standards**       | PASS                | PASS               | Waveform augmentations (`apply_onset_jitter`, `add_background_noise`) run prior to single-pass STFT `extract_tap_features` so `physics_feats` remain consistent with the waveform; spectrogram augmentations (`apply_spectral_tilt`, `apply_spec_augment`) preserve `[0.0, 1.0]` bounds; zero Pitch Shift/Time Stretch. |
| **IV. Performance Requirements**       | PASS                | PASS               | `mel_basis` (`(128, 513)`) is precomputed once per dataset extraction; `DurianAudioCnn` uses ~`408K` parameters (`< 2.0 MB` FP32); `benches/training_benchmarks.rs` benchmarks `DurianAudioCnn::forward` while preserving existing benchmark groups.                                                                    |
| **V. Documentation & Maintainability** | PASS                | PASS               | Rustdoc with runnable examples on all public structs and functions in `src/model.rs` and `src/training.rs`; concise `//` comments explaining _why_ without decorative block separators (`===`, `---`); formatted via `rustfmt`.                                                                                         |

## Project Structure

### Documentation (this feature)

```text
specs/003-phase3-model-training/
├── plan.md                                # This implementation plan (/speckit-plan command output)
├── research.md                            # Phase 0 technical research & Burn 0.22.1 design decisions
├── data-model.md                          # Phase 1 neural network, batch, evaluation & checkpoint schemas
├── quickstart.md                          # Phase 1 runnable validation, CLI training & benchmark guide
├── contracts/
│   └── model-training-and-cli.md          # Public Rust API contracts for model, training & CLI train subcommand
├── checklists/
│   ├── requirements.md                    # Specification quality checklist (16/16 passing)
│   ├── model.md                           # Model architecture requirements checklist (CHK001-CHK032)
│   └── training.md                        # Training pipeline, loss, evaluation & CLI checklist (CHK001-CHK040)
└── tasks.md                               # Phase 2 task breakdown (/speckit-tasks command - created next)
```

### Source Code (repository root)

```text
Cargo.toml                                 # Locked workspace dependencies (burn 0.22.1, ndarray 0.17.2, rand 0.10.4, clap 4.6.9)
configs/
└── default.toml                           # Centralized [dsp], [dataset], [training], [model] parameters
data/
├── metadata_iphone.csv                    # 281 iPhone metadata records (171 Unripe, 66 Ripe, 44 Overripe)
├── metadata_android.csv                   # 175 Android metadata records (95 Unripe, 38 Ripe, 42 Overripe)
├── sound_iphone/                          # 281 .m4a AAC-LC recordings
└── sound_android/                         # 175 .wav PCM 16-bit 48kHz Stereo recordings
artifacts/                                 # Generated at training time via create_dir_all
├── durian_iphone.mpk                      # Dedicated iOS FP32 model weights (< 2.0 MB)
└── durian_android.mpk                     # Dedicated Android FP32 model weights (< 2.0 MB)
src/
├── lib.rs                                 # Exports pub mod augment; pub mod config; pub mod dataset; pub mod dsp; pub mod model; pub mod training;
├── main.rs                                # CLI entry point (VerifyDataset + NEW Train subcommand)
├── config.rs                              # [Phase 1] Config, DspConfig, DatasetConfig, TrainingConfig, ModelConfig
├── dataset.rs                             # [Phase 1] CSV verifier, fruit_id splitter, compute_class_weights
├── dsp.rs                                 # [Phase 2] Audio decode, resample, onset detection, Log-Mel & 4 physics features
├── augment.rs                             # [Phase 2] Onset jitter, SNR noise, spectral tilt, SpecAugment
├── model.rs                               # [Phase 3 NEW] ConvBlock<B>, DurianAudioCnn<B>, forward, forward_proba, save_mpk, load_mpk
└── training.rs                            # [Phase 3 EXPANDED] TapSample, oversampling, DurianBatch, weighted CE loss, metrics, EarlyStopping, training loop
benches/
└── training_benchmarks.rs                 # Criterion benchmarks (retains existing + benchmarks DurianAudioCnn::forward)
```

**Structure Decision**: Retain the single-crate library and CLI layout in `src/`, adding `src/model.rs` (exported via `pub mod model;` in `src/lib.rs`), expanding `src/training.rs` while preserving existing `GradientDescent` and `LearningRateSchedule` symbols for benchmark compatibility, extending `src/main.rs` with the `Train` subcommand, and updating `benches/training_benchmarks.rs`. This keeps `src/model.rs` decoupled from training state so Phase 4 mobile inference can import `DurianAudioCnn` directly (`model.md CHK031`).

## Implementation Architecture & Phases

### Phase 3A: Hybrid CNN Model Architecture & `.mpk` Serialization (`src/model.rs`, `src/lib.rs`)

Covers `FR-001`, `FR-002`, `US-1`, `SC-001`, `SC-005`, and `checklists/model.md` `CHK001`–`CHK032`.

1. **TDD Red Phase (`model.md CHK024`)**:
    - Export `pub mod model;` in `src/lib.rs`.
    - Write unit tests in `src/model.rs` using `type TestBackend = burn::backend::NdArray<f32>;` covering:
        - `test_conv_block_spatial_downsampling`: verify `ConvBlock::new(1, 32, &device)` maps `[2, 1, 128, 128]` to `[2, 32, 64, 64]` with finite values.
        - `test_durian_audio_cnn_forward_shape_and_proba_simplex`: verify `DurianAudioCnn::new(3, 0.3, &device)` and `DurianAudioCnn::from_config(&ModelConfig::default(), &device)` map `mel_spec: [B, 1, 128, 128]` and `phys_feats: [B, 4]` to `[B, 3]` across dynamic batch sizes ($B = 1$, $B = 4$, $B = 7$), with all logits finite and every `forward_proba` row in `[0.0, 1.0]` summing to `1.0` within `< 1e-5` (`SC-001`, `CHK015`, `CHK016`, `CHK021`).
        - `test_durian_audio_cnn_boundary_inputs_stability`: verify boundary inputs (all-zero `0.0` noise-floor spectrograms and all-one `1.0` peak tensors) produce finite logits and probabilities without `NaN` or `±Infinity` (`CHK022`).
        - `test_durian_audio_cnn_mpk_save_load_and_size_budget`: save model weights to a temporary nested path (where parent directory does not yet exist, testing both stem `"nested/durian_test"` and explicit `".mpk"` suffix) using `save_mpk` and reload via `load_mpk`; assert file exists at `"nested/durian_test.mpk"`, file size on disk is `> 0` and `< 2_000_000` bytes (`< 2.0 MB`, `SC-005`, `CHK017`, `CHK023`, `CHK030`, `CHK032`), and reloaded model produces identical `forward` logits (`approx::assert_relative_eq!`).
2. **TDD Green & Refactor Phase**:
    - Implement `ConvBlock<B: Backend>` and `DurianAudioCnn<B: Backend>` with `new`, `from_config`, `forward`, `forward_proba`, `save_mpk`, and `load_mpk` in `src/model.rs` with full rustdoc examples and zero emojis or decorative separators (`CHK025`, `CHK027`).

### Phase 3B: Tap Sample Extraction, Train-Only Minority Oversampling & Batch Collation (`src/training.rs`)

Covers `FR-003`–`FR-005`, `US-2`, `SC-003`, and `checklists/training.md` `CHK001`, `CHK002`, `CHK007`, `CHK013`, `CHK014`, `CHK017`, `CHK025`, `CHK026`, `CHK033`, `CHK035`–`CHK037`, `CHK039`.

1. **TDD Red Phase (`training.md CHK031`)**:
    - Write unit tests in `src/training.rs` covering:
        - `test_extract_tap_samples_train_minority_oversampling_vs_eval`: synthesize a multi-tap `22,050 Hz` waveform with $K = 3$ clear impulses; verify `extract_tap_samples_from_waveform` with `is_train = true` returns $1 \times K = 3$ samples for `RipenessLabel::Unripe`, $2 \times K = 6$ samples for `RipenessLabel::Ripe`, and $3 \times K = 9$ samples for `RipenessLabel::Overripe`, whereas `is_train = false` returns strictly $1 \times K = 3$ unaugmented canonical samples for all 3 classes (`CHK007`, `CHK013`, `CHK017`).
        - `test_extract_tap_samples_zero_onset_fallback`: pass a silent/near-silent waveform where `detect_onset_centers` yields `0` peaks; verify the extractor logs a warning and extracts `1` canonical `0.8s` fallback window (and `1/2/3` windows in `is_train = true`) with finite `mel_data` (`16,384` floats in `[0.0, 1.0]`) and `physics_feats` in `[0.0, 1.0]^4` (`CHK025`).
        - `test_collate_tap_batch_dynamic_shapes`: verify `collate_tap_batch` produces `mel_specs: [B, 1, 128, 128]`, `phys_feats: [B, 4]`, and `targets: [B]` with class indices `{0, 1, 2}` for both full batches and partial final batches ($B = 1$, $B = 5$) (`CHK002`, `CHK026`, `CHK036`).
2. **TDD Green & Refactor Phase**:
    - Implement `TapSample`, `DurianBatch<B>`, `extract_tap_samples_from_waveform`, `extract_dataset_tap_samples` (reusing a single precomputed `mel_basis`), and `collate_tap_batch` in `src/training.rs`.

### Phase 3C: Weighted Cross-Entropy Loss with Label Smoothing & `AdamW` Convergence (`src/training.rs`)

Covers `FR-006`, `US-3`, `SC-002`, and `checklists/training.md` `CHK003`, `CHK008`, `CHK009`, `CHK018`, `CHK019`, `CHK028`, `CHK038`.

1. **TDD Red Phase**:
    - Write unit tests in `src/training.rs` covering:
        - `test_weighted_cross_entropy_loss_formula_and_class_weight_ratio`: verify exact mathematical value of `weighted_cross_entropy_loss` against hand-calculated smoothed distribution $q_i(c) = (1 - 0.05)\mathbb{I}(c = y_i) + 0.05 / 3$; verify that for two identically misclassified samples using iPhone class weights `[0.5478, 1.4192, 2.1288]`, the `Overripe` sample incurs $\approx 2.1288 / 0.5478 \approx 3.886\times$ the loss of the `Unripe` sample (`US-3 Scenario 2`, `CHK008`, `CHK019`).
        - `test_weighted_cross_entropy_extreme_logits_stability`: verify extreme logits (e.g., `[100.0, -100.0, -100.0]`) produce finite loss without `NaN` or `Infinity` (`CHK028`).
        - `test_weighted_cross_entropy_loss_adamw_gradient_convergence`: construct a 6-sample synthetic mini-batch (`2 Unripe, 2 Ripe, 2 Overripe`), initialize `DurianAudioCnn<Autodiff<NdArray<f32>>>` and `AdamWConfig::new().with_weight_decay(1e-4).init()`, run 5 consecutive forward-loss-backward-step iterations (`lr = 3e-4`), and assert `loss_step_5 < loss_step_1` with all weights finite (`SC-002`, `CHK018`, `CHK038`).
2. **TDD Green & Refactor Phase**:
    - Implement `weighted_cross_entropy_loss<B: Backend>` in `src/training.rs`.

### Phase 3D: Dual-Level Evaluator (`Tap-Level` & `Recording-Level`), Confusion Matrix & Early Stopping (`src/training.rs`)

Covers `FR-007`–`FR-009`, `US-4`, and `checklists/training.md` `CHK004`, `CHK005`, `CHK010`–`CHK012`, `CHK015`, `CHK016`, `CHK022`–`CHK024`, `CHK027`–`CHK029`.

1. **TDD Red Phase**:
    - Write unit tests in `src/training.rs` covering:
        - `test_confusion_matrix_and_macro_f1_calculation`: populate a known $3 \times 3$ confusion matrix and verify `per_class_precision`, `per_class_recall`, `per_class_f1`, `macro_f1`, `balanced_accuracy`, `overall_accuracy`, and `critical_fp_ripe_rate` match hand calculations (`CHK004`, `CHK010`).
        - `test_confusion_matrix_zero_denominator_guards`: verify an empty `ConfusionMatrix3x3::default()` and a single-class matrix (`row_sum == 0` / `col_sum == 0`) return `0.0` for all zero-denominator metrics and never `NaN` (`CHK027`).
        - `test_recording_level_entropy_weighted_aggregation_and_uniform_fallback`: verify `aggregate_recording_probabilities` assigns higher weight $\alpha_k = 1 - H(P_k)/\ln 3$ to a confident tap (`[0.02, 0.96, 0.02]`) than to uncertain taps (`[0.35, 0.31, 0.34]`), handles `0.0` probabilities safely via the `1e-12` log floor, and falls back to uniform arithmetic mean `[1/3, 1/3, 1/3]` when all taps are maximally uncertain ($\sum \alpha_k \le 10^{-6}$) (`CHK011`, `CHK028`).
        - `test_early_stopping_state_machine`: verify `EarlyStopping::new(3)` initializes `best_macro_f1 = -1.0`, returns `Improved` on Epoch 1 even when `val_recording_macro_f1 == 0.0` (`CHK029`), resets `patience_counter = 0` on subsequent improvements, and returns `Stop` after `3` consecutive non-improving epochs (`CHK012`).
        - `test_train_model_on_splits_end_to_end_checkpoint_restore`: run `train_model_on_splits<Autodiff<NdArray<f32>>>` for 2 epochs on a small synthetic split in a `tempdir`, verifying epoch shuffling, `model.valid()` dual-level evaluation, `.mpk` checkpoint creation (`< 2.0 MB`), checkpoint restoration, and final `Test` evaluation report generation (`CHK005`, `CHK022`, `CHK023`).
2. **TDD Green & Refactor Phase**:
    - Implement `ConfusionMatrix3x3`, `EvaluationReport`, `DualLevelEvaluation`, `aggregate_recording_probabilities`, `evaluate_model`, `EarlyStopping`, `EarlyStoppingAction`, `TrainingOutcome`, `train_model_on_splits`, and `run_platform_training` in `src/training.rs` while keeping `GradientDescent` and `LearningRateSchedule` intact (`CHK016`).

### Phase 3E: Dual-Platform CLI `train` Subcommand & Criterion Forward Benchmark (`src/main.rs`, `benches/training_benchmarks.rs`)

Covers `FR-010`, `US-5`, `SC-004`, `SC-005`, `model.md CHK026`, and `training.md CHK006`, `CHK020`, `CHK021`, `CHK030`, `CHK034`, `CHK040`.

1. **TDD Red & Implementation Phase**:
    - Extend `Commands` in `src/main.rs` with `Train { platform: String, config: PathBuf, data_dir: Option<PathBuf>, output: Option<PathBuf> }`.
    - Parse `--platform` (`iphone` or `android`, failing fast on unknown platform names, missing config files, or invalid dataset paths per `CHK030`), resolve default output path (`artifacts/durian_iphone` or `artifacts/durian_android`), run `run_platform_training::<Autodiff<NdArray<f32>>>`, and format an emoji-free console summary displaying platform class weights, fruit split counts, best validation epoch/Macro-F1, saved `.mpk` path and size (`< 2.0 MB`), and both `Tap-Level` and `Recording-Level` test reports (`ConfusionMatrix3x3`, per-class metrics, `Macro F1`, `Balanced Accuracy`, `Critical FP_Ripe Rate`).
    - Update `benchmark_model_training` in `benches/training_benchmarks.rs` to benchmark `DurianAudioCnn::<NdArray<f32>>::forward` on a `[1, 1, 128, 128]` Log-Mel tensor and `[1, 4]` physics vector (`model.md CHK026`, `training.md CHK040`).
    - Run full constitutional quality gates: `cargo test`, `cargo bench --bench training_benchmarks -- --test`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check`.

## Complexity Tracking

No constitutional violations exist. All Phase 3 components use the locked `burn = "0.22.1"` crate and standard workspace dependencies (`ndarray 0.17.2`, `rand 0.10.4`, `clap 4.6.9`, `anyhow 1.0.105`) without adding new external crates.
