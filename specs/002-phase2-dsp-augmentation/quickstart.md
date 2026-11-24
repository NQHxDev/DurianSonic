# Quickstart Validation Guide: Phase 2 - Acoustic DSP, Impulse Onset Detection & Feature Extraction

**Feature**: `002-phase2-dsp-augmentation`
**Date**: 2026-11-24

---

## 1. Prerequisites

- **Rust Toolchain**: `rustc` and `cargo` (Edition 2021)
- **Completed Phase 1 Foundation**:
    - `src/config.rs` (`DspConfig` with physical validation)
    - `src/dataset.rs` (`AudioMetadataRecord`, `PlatformTarget`, verified `/data/` directory)
- **Audio Dataset Fixtures (`data/`)**:
    - `data/sound_iphone/durian_001_unripe_iphone.m4a` (AAC-LC in ISO/MP4 container)
    - `data/sound_android/durian_001_unripe_android.wav` (16-bit PCM Stereo 48,000 Hz)

---

## 2. Step-by-Step Validation Commands

### Step 1: Run Phase 2 Unit & Integration Tests

Execute the full test suite verifying multi-format decoding, resampling, onset detection, Log-Mel spectrogram generation, physics feature extraction, and physically valid data augmentation:

```bash
cargo test
```

**Expected Outcome**:

- All existing Phase 1 tests and all new Phase 2 tests in `src/dsp.rs` and `src/augment.rs` pass (`0` failures).
- Key validation scenarios proven by tests:
    - **Multi-Format Decoding & Resampling (`US1`)**: Both `durian_001_unripe_iphone.m4a` and `durian_001_unripe_android.wav` decode cleanly to finite Mono `22,050 Hz` waveforms and yield `>= 5` valid tap windows (`SC-001`). Stereo buffers downmix to exact arithmetic mean per frame, and invalid/missing files or `0 Hz` rates return descriptive errors.
    - **Transient Impulse Onset Detection (`US2`)**: A synthetic `10s` (`220,500`-sample) signal containing `5` decaying sine impulses spaced `1.5s` apart yields exactly `5` windows of `17,640` samples each with peak amplitude `1.0` (`< 1e-6` error) and inter-onset distance `>= 8,820` samples (`0.4s`) (`SC-002`). Empty `&[]` returns `Vec::new()`, and short non-empty slices (`1..17,640` samples) zero-pad to `1` window of `17,640` samples.
    - **Log-Mel Spectrogram & 4 Physics Features (`US3`)**: Every `17,640`-sample window produces a `(128, 128)` Log-Mel matrix and `[f32; 4]` physics vector with all values finite and within `[0.0, 1.0]` (`SC-003`). Synthetic `Unripe` (`600 Hz`, slow decay) vs `Overripe` (`180 Hz`, fast decay) impulses confirm $\tilde{f}_{\text{peak}}(\text{Unripe}) > \tilde{f}_{\text{peak}}(\text{Overripe})$, $\tilde{C}_s(\text{Unripe}) > \tilde{C}_s(\text{Overripe})$, and $\tilde{\beta}_{\text{decay}}(\text{Overripe}) > \tilde{\beta}_{\text{decay}}(\text{Unripe})$ (`SC-004`).
    - **Physically Valid Augmentation (`US4`)**: `apply_onset_jitter`, `add_background_noise`, `apply_spectral_tilt`, and `apply_spec_augment` preserve output dimensions (`17,640` or `(128, 128)`), stay bounded in `[-1.0, 1.0]` or `[0.0, 1.0]`, preserve peak resonance frequency $f_{\text{peak}}$, mask regions to `0.0`, and never panic on degenerate bounds (`0`, inverted SNR range, or tiny matrices).

---

### Step 2: Run Criterion DSP Latency Benchmark (`< 4.0 ms / tap`)

Execute the Criterion benchmark suite in `benches/training_benchmarks.rs` to verify single-threaded per-tap DSP extraction latency (`SC-005`):

```bash
cargo bench --bench training_benchmarks -- dsp_extract_tap_features
```

**Expected Outcome**:

- The `dsp_extract_tap_features` benchmark (measuring `extract_tap_features` on a `0.8s` / `17,640`-sample tap window with a pre-built `mel_basis` `(128, 513)`) completes in **`< 4.0 ms`** per tap on a single CPU thread.

---

### Step 3: Verify Constitution Quality Gates

Run strict compiler linting, formatting, and dataset verification checks required by `.specify/memory/constitution.md`:

```bash
cargo clippy --all-targets -- -D warnings && cargo fmt --check && cargo run -- verify-dataset --data-dir data
```

**Expected Outcome**:

- `0` compiler or clippy warnings across library, binary, tests, and benchmarks.
- `0` formatting differences.
- Codebase remains 100% emoji-free, uses no decorative comment separators (`===`, `---`), and contains zero unguarded `unwrap()` or `expect()` calls in library code.
