# Quickstart Validation Guide: Phase 3 - Hybrid CNN Architecture & Training Pipeline in Burn

**Feature**: `003-phase3-model-training`
**Date**: 2026-11-26

---

## 1. Prerequisites

- **Rust Toolchain**: `rustc` and `cargo` (Edition 2021)
- **Completed Phase 1 & Phase 2 Modules**:
    - `src/config.rs` (`Config`, `DspConfig`, `DatasetConfig`, `TrainingConfig`, `ModelConfig`)
    - `src/dataset.rs` (`load_and_verify_metadata`, `stratified_group_split_by_fruit`, `compute_class_weights`, `PlatformTarget`, `RipenessLabel`)
    - `src/dsp.rs` (`decode_audio_to_mono`, `detect_onset_centers`, `extract_tap_impulses`, `build_mel_filterbank`, `extract_tap_features`)
    - `src/augment.rs` (`apply_onset_jitter`, `add_background_noise`, `apply_spectral_tilt`, `apply_spec_augment`)
- **Verified Audio Dataset (`data/`)**:
    - `data/metadata_iphone.csv` + `data/sound_iphone/` (`281` `.m4a` files across `175` fruits)
    - `data/metadata_android.csv` + `data/sound_android/` (`175` `.wav` files across `175` fruits)

---

## 2. Step-by-Step Validation Commands

### Step 1: Run Phase 3 Unit & Integration Tests

Execute the full test suite verifying `ConvBlock`, `DurianAudioCnn` forward shapes and probability invariants, `.mpk` serialization roundtrip (`< 2.0 MB`), train-only minority oversampling (`1x/2x/3x` vs `1x/1x/1x`), dynamic batch collation, weighted cross-entropy loss with label smoothing, `AdamW` gradient descent convergence, `ConfusionMatrix3x3`, Entropy-weighted `Recording-Level` aggregation, `EarlyStopping`, and CLI integration:

```bash
cargo test
```

**Expected Outcome**:

- All Phase 1, Phase 2, and Phase 3 tests pass (`0` failures).
- Key validation scenarios proven by tests (see [contracts/model-training-and-cli.md](./contracts/model-training-and-cli.md) and [data-model.md](./data-model.md)):
    - **Hybrid CNN Forward & Probability Simplex (`US1`, `SC-001`)**: `DurianAudioCnn::forward` maps `[B, 1, 128, 128]` and `[B, 4]` to finite `[B, 3]` logits for any $B \ge 1$ (including $B = 1$, $B = 4$, and partial batches $1 \le B < 32$), and `forward_proba` outputs probabilities in `[0.0, 1.0]` summing to `1.0` per row (`< 1e-5` error).
    - **FP32 `.mpk` Weight Serialization & Size Budget (`US1`, `SC-005`)**: Saving `DurianAudioCnn` via `NamedMpkFileRecorder::<FullPrecisionSettings>` creates missing parent directories, produces an `.mpk` artifact strictly `< 2.0 MB` (~`1.63 MB`), and reloading reproduces identical inference logits.
    - **Train-Only Minority Oversampling & Zero Leakage (`US2`, `SC-003`)**: `is_train = true` produces `1x` (`Unripe`), `2x` (`Ripe`), and `3x` (`Overripe`) `TapSample` windows per onset peak with physically valid augmentations, while `is_train = false` (`Validation`/`Test`) produces strictly `1x` canonical window per peak with zero augmentation and zero `fruit_id` overlap across splits; `0`-onset recordings trigger the max-RMS `0.8s` fallback window so no recording is dropped.
    - **Weighted Cross-Entropy Loss & `AdamW` Convergence (`US3`, `SC-002`)**: Misclassified minority `Overripe` samples on iPhone ($w_2 \approx 2.129$) incur $\approx 3.88\times$ the loss of `Unripe` samples ($w_0 \approx 0.548$), and 5 consecutive `AdamW` optimization steps on a 6-sample mini-batch strictly reduce loss (`loss_step_5 < loss_step_1`).
    - **Dual-Level Evaluator & Early Stopping (`US4`)**: `ConfusionMatrix3x3::compute_report()` accurately computes per-class Precision/Recall/F1, Macro F1, Balanced Accuracy, Overall Accuracy, and `critical_fp_ripe_rate` with zero-denominator guards (`0.0`, never `NaN`); `aggregate_recording_probabilities` applies Entropy confidence weights $\alpha_k = 1 - H(P_k)/\ln 3$ with uniform fallback when $\sum \alpha_k \le 10^{-6}$.

---

### Step 2: Train Dedicated iPhone & Android Platform Models via CLI (`US5`, `SC-004`, `SC-005`)

Train and evaluate the two independent platform models using the `train` CLI subcommand:

```bash
# 1. Train & evaluate iPhone model (data/metadata_iphone.csv -> artifacts/durian_iphone.mpk)
cargo run --release -- train \
    --platform iphone \
    --config configs/default.toml \
    --output artifacts/durian_iphone

# 2. Train & evaluate Android model (data/metadata_android.csv -> artifacts/durian_android.mpk)
cargo run --release -- train \
    --platform android \
    --config configs/default.toml \
    --output artifacts/durian_android
```

**Expected Outcome**:

- Both runs verify dataset integrity, compute platform-specific class weights (`[0.548, 1.419, 2.129]` for iPhone; `[0.614, 1.535, 1.389]` for Android), confirm zero `fruit_id` overlap across `Train / Val / Test`, checkpoint best weights on `val_recording_macro_f1`, and output `Tap-Level` and `Recording-Level` test reports.
- `artifacts/durian_iphone.mpk` and `artifacts/durian_android.mpk` are each `< 2.0 MB` on disk (`SC-005`) and achieve **Recording-Level Macro F1-Score $\ge 0.85$** and **Recording-Level Balanced Accuracy $\ge 0.85$** on their held-out `Test` split (`SC-004`).

---

### Step 3: Run Criterion Model & DSP Benchmarks and Constitution Quality Gates

Verify forward-pass benchmark compilation and execution alongside strict linting and formatting gates:

```bash
cargo bench --bench training_benchmarks -- --test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

**Expected Outcome**:

- All Criterion benchmark groups (`gradient_descent`, `learning_rate_schedules`, `model_training::forward_pass`, `dsp_pipeline`) pass in test mode (`0` errors).
- `0` compiler or clippy warnings and `0` formatting diffs.
