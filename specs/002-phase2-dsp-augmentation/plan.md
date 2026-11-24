# Implementation Plan: Phase 2 - Acoustic DSP, Impulse Onset Detection & Feature Extraction

**Branch**: `002-phase2-dsp-augmentation` | **Date**: 2026-11-24 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-phase2-dsp-augmentation/spec.md`

## Summary

Phase 2 implements the core pure-Rust acoustic Digital Signal Processing (DSP), transient tap onset detection, dual-branch feature extraction (`128 x 128` Log-Mel Spectrogram + `4` Acoustic Physics Features), and physically valid training-time data augmentation for DurianSonic:

1. **Multi-Format Audio Decoder, Mono Downmixer & `22,050 Hz` Resampler (`src/dsp.rs`)**: Decode both iPhone `.m4a` (`AAC-LC` in ISO/MP4) and Android `.wav` (`16-bit PCM` Stereo `48,000 Hz`) recordings using `symphonia 0.6.1`, downmix multi-channel frames to Mono via arithmetic mean ($x_{\text{mono}}[n] = \frac{1}{C}\sum_{c} x_c[n]$), and resample to `22,050 Hz` using `rubato 5.1.0` (`Fft::<f32>::new` + `Resampler::process_all` with automatic startup delay trimming and exact length rounding).
2. **Short-Time RMS Onset Detector & `0.8s` Tap Window Extractor (`src/dsp.rs`)**: Scan `22,050 Hz` Mono waveforms in non-overlapping `256`-sample (`~11.61 ms`) RMS frames (`detect_onset_centers` & `extract_tap_impulses`), applying an adaptive threshold $\theta = \max(0.25 \cdot \max_i E_{\text{rms}}[i], 10^{-4})$ and a `0.4s` (`8,820`-sample) refractory period to slice peak-normalized `0.8s` (`17,640`-sample) windows (`-100 ms` pre-onset lead-in to `+700 ms` decay tail).
3. **Single-Pass STFT Power Spectrogram, Log-Mel (`128 x 128`) & 4 Physics Features (`src/dsp.rs`)**: Compute the `128`-frame `1024`-point Hann-windowed RealFFT power spectrogram (`(513, 128)`, `hop_length = 128`) once per tap (`compute_power_spectrogram`), project onto a reusable Slaney-normalized Mel filterbank (`build_mel_filterbank`, `(128, 513)` over `50–8,000 Hz`) mapped from `[-80 dB, 0 dB]` to `[0.0, 1.0]` (`compute_log_mel_spectrogram`), and extract the `4` normalized `[0.0, 1.0]` acoustic physics features (`extract_physics_features_from_power_spec`, `extract_physics_features`, and unified `extract_tap_features` returning `TapFeatures`): Peak Resonance Frequency ($\tilde{f}_{\text{peak}}$), Spectral Centroid ($\tilde{C}_s$), Spectral Roll-off 85% ($\tilde{R}_{85}$) on frames `0..64`, and Temporal Decay Slope ($\tilde{\beta}_{\text{decay}}$) on inclusive frames `26..=86`.
4. **Physically Valid Data Augmentation (`src/augment.rs`) & Criterion Benchmarks (`benches/training_benchmarks.rs`)**: Provide 4 train-only augmentations on `rand 0.10.4` (`apply_onset_jitter` $\pm 40\text{ ms}$, `add_background_noise` with Box-Muller Gaussian white noise at $\text{SNR} \in [15, 35]\text{ dB}$, `apply_spectral_tilt` $\pm 3\text{ dB}$, and `apply_spec_augment` masking to `0.0`) with defensive range guards and zero Pitch Shift / Time Stretch, plus a Criterion benchmark verifying `< 4.0 ms` per-tap DSP extraction latency.

## Technical Context

**Language/Version**: Rust (Edition 2021)

**Primary Dependencies**: `symphonia 0.6.1` (`aac`, `isomp4`, `pcm`, `wav`), `rubato 5.1.0`, `realfft 3.5.1`, `ndarray 0.17.2`, `rand 0.10.4`, `anyhow 1.0.105`

**Storage**: Local filesystem audio fixtures (`data/sound_iphone/*.m4a`, `data/sound_android/*.wav`) and configuration (`configs/default.toml`)

**Testing**: `cargo test` (unit and real-audio integration tests with `approx 0.5.1` and `tempfile 3.27.0`) and `cargo bench` (`criterion 0.8.2`)

**Target Platform**: Cross-platform workstation/CI (macOS, Linux, Windows) and downstream mobile targets (`aarch64-apple-ios`, `aarch64-linux-android`) with zero C/FFmpeg dependencies

**Project Type**: Rust Library + CLI + Criterion Benchmark Suite (`src/lib.rs`, `src/dsp.rs`, `src/augment.rs`, `benches/training_benchmarks.rs`)

**Performance Goals**: Single-threaded per-tap feature extraction (`extract_tap_features` computing `(128, 128)` Log-Mel spectrogram + `4` physics features from a `0.8s` / `17,640`-sample window using a pre-built `mel_basis`) MUST complete in **`< 4.0 ms`** (`SC-005`)

**Constraints**:

- **Zero Training-Serving Skew**: Identical pure-Rust DSP functions used across training, evaluation, and mobile inference.
- **Physical Label Integrity**: Pitch Shifting and Time Stretching are strictly prohibited (`FR-009`) because they distort cavity resonance frequency $f_r$ and damping slope $\beta_{\text{decay}}$.
- **Numerical & Memory Safety**: Zero `NaN` or `±Infinity` outputs even on all-zero silent inputs (`1e-8` amplitude floor, `1e-10` power floor); zero panics on empty, short (`< 17,640` samples), or degenerate augmentation parameters.
- **Constitutional Compliance**: TDD Red-Green-Refactor, 100% emoji-free codebase, no decorative comment separators (`===`, `---`), no unguarded `unwrap()`/`expect()` in library code, and zero clippy warnings (`cargo clippy --all-targets -- -D warnings`).

**Scale/Scope**: Processes `456` multi-tap recordings (`281` iPhone `.m4a` + `175` Android `.wav`, ~`28–51s` each) into ~`5,470` normalized `0.8s` tap impulses (~`3,370` iPhone taps + ~`2,100` Android taps)

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design._

| Constitutional Principle               | Pre-Research Status | Post-Design Status | Compliance Evidence                                                                                                                                                                                                                                             |
| :------------------------------------- | :------------------ | :----------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **I. Clean Code Excellence**           | PASS                | PASS               | No bypass patterns; 100% emoji-free code and comments; explicit `anyhow::Result` error handling in `decode_audio_to_mono` and `resample_mono`; defensive bounds guards on all `rand 0.10.4` ranges so library functions never panic.                            |
| **II. Test-First Development**         | PASS                | PASS               | Comprehensive unit and integration tests specified prior to implementation for decoding, resampling, onset detection, Log-Mel shape/bounds, synthetic Unripe vs Overripe resonance physics features, and all 4 augmentation transforms.                         |
| **III. Audio Quality Standards**       | PASS                | PASS               | `rubato::Fft` with `process_all` applies anti-aliasing filtering and trims filter delay; Slaney area normalization preserves energy across Mel bands; DSP docstrings explicitly document signal transformations and numerical epsilon guards (`1e-8`, `1e-10`). |
| **IV. Performance Requirements**       | PASS                | PASS               | `extract_tap_features` computes the `(513, 128)` RealFFT power spectrogram once per tap and reuses a pre-allocated `mel_basis` `(128, 513)`, achieving `< 4.0 ms/tap` verified by Criterion benchmark in `benches/training_benchmarks.rs`.                      |
| **V. Documentation & Maintainability** | PASS                | PASS               | Rustdoc with examples on all public structs and functions in `src/dsp.rs` and `src/augment.rs`; concise `//` comments explaining _why_ without decorative block separators (`===` or `---`); formatted via `rustfmt`.                                           |

## Project Structure

### Documentation (this feature)

```text
specs/002-phase2-dsp-augmentation/
├── plan.md                        # This implementation plan (/speckit-plan command output)
├── research.md                    # Phase 0 technical research & crate API decisions
├── data-model.md                  # Phase 1 DSP & feature entity schemas and mathematical invariants
├── quickstart.md                  # Phase 1 runnable validation & benchmark guide
├── contracts/
│   └── dsp-and-augment.md         # Public Rust API contracts for durian_sonic::dsp & durian_sonic::augment
├── checklists/
│   └── requirements.md            # Specification quality checklist
└── tasks.md                       # Phase 2 task breakdown (/speckit-tasks command - created next)
```

### Source Code (repository root)

```text
Cargo.toml                         # Crate manifest (symphonia 0.6.1, rubato 5.1.0, realfft 3.5.1, ndarray 0.17.2, rand 0.10.4)
configs/
└── default.toml                   # Centralized [dsp], [dataset], [training], [model] parameters
data/
├── sound_iphone/                  # 281 .m4a AAC-LC recordings (including durian_001_unripe_iphone.m4a)
└── sound_android/                 # 175 .wav PCM 16-bit 48kHz Stereo recordings (including durian_001_unripe_android.wav)
src/
├── lib.rs                         # Exports pub mod config; pub mod dataset; pub mod dsp; pub mod augment; pub mod training;
├── main.rs                        # CLI entry point
├── config.rs                      # [Phase 1] DspConfig & Config validation
├── dataset.rs                     # [Phase 1] Metadata verifier & anti-leakage splitter
├── dsp.rs                         # [Phase 2 NEW] Audio decoder, resampler, onset detector, Log-Mel & 4 physics features
├── augment.rs                     # [Phase 2 NEW] Physically valid waveform & spectrogram augmentations
└── training.rs                    # Existing training utilities (to be expanded in Phase 3)
benches/
└── training_benchmarks.rs         # Criterion benchmarks (extended with DSP per-tap latency benchmark)
```

**Structure Decision**: Extend the single-crate library layout in `src/` by adding `src/dsp.rs` and `src/augment.rs`, exported publicly via `src/lib.rs`, and adding DSP latency benchmarks to `benches/training_benchmarks.rs`. This matches `RoadMap.md`, `docs/TECHNICAL_DESIGN.md`, and `docs/phases/PHASE_2_DSP_AND_AUGMENTATION.md`.

## Implementation Architecture & Phases

### Phase 2A: Config Bridges, Multi-Format Audio Decoding & `22,050 Hz` Resampling (`src/dsp.rs`, `src/lib.rs`)

1. **TDD Red Phase**:
    - Declare `pub mod dsp;` and `pub mod augment;` in `src/lib.rs`.
    - Write unit and integration tests in `src/dsp.rs` covering:
        - `ImpulseConfig::default()`, `SpectrogramConfig::default()`, and `From<&DspConfig>` conversions matching `DspConfig::default()`.
        - `resample_mono`: identity passthrough when `src_sr == target_sr` (`22,050 Hz`), downsampling a synthetic `48,000 Hz` or `44,100 Hz` sine wave to `22,050 Hz` with output length equal to $\text{round}(N_{\text{src}} \times 22050 / f_{\text{src}})$ and all finite values, and error returns on empty input or `0 Hz` sample rates.
        - `decode_audio_to_mono`: error returns on `target_sr == 0`, non-existent file path, or corrupted/empty file (`tempfile`), and real-file decoding of `data/sound_iphone/durian_001_unripe_iphone.m4a` and `data/sound_android/durian_001_unripe_android.wav` to `22,050 Hz` Mono.
2. **TDD Green & Refactor Phase**:
    - Implement `ImpulseConfig`, `SpectrogramConfig`, `TapFeatures`, `decode_audio_to_mono` (using `symphonia 0.6.1` `Probe::probe`, `TrackType::Audio`, `CodecRegistry::make_audio_decoder`, `GenericAudioBufferRef::copy_to_vec_interleaved`, and channel averaging), and `resample_mono` (using `rubato 5.1.0` `Fft::<f32>::new` + `InterleavedSlice` + `Resampler::process_all`).

### Phase 2B: Transient Impulse Onset Detection & `0.8s` Tap Window Extraction (`src/dsp.rs`)

1. **TDD Red Phase**:
    - Write unit tests in `src/dsp.rs` covering:
        - `normalize_peak`: normalizes arbitrary waveforms to peak `1.0` (`< 1e-6` tolerance) and handles all-zero silent arrays (`vec![0.0; 17_640]`) safely without `NaN` using the `1e-8` denominator floor.
        - `test_extract_tap_impulses_synthetic_train`: synthetic `10s` (`220,500`-sample) signal at `22,050 Hz` containing `5` exponentially decaying sine impulses spaced `1.5s` apart; verify `detect_onset_centers` and `extract_tap_impulses` return exactly `5` windows, each of exact length `17,640` samples with peak amplitude `1.0` and consecutive center distance `>= 8,820` samples (`0.4s`).
        - Edge cases: empty slice `&[]` returns `Vec::new()` for both `detect_onset_centers` and `extract_tap_impulses`; short non-empty slice (`1..17,640` samples) returns `vec![0]` from `detect_onset_centers` and `1` zero-padded `17,640`-sample normalized window from `extract_tap_impulses`; secondary reflection peaks within `< 0.4s` (`8,820` samples) are suppressed; trailing peaks near EOF where `start + 17,640 > samples.len()` are excluded; real iPhone `.m4a` and Android `.wav` recordings each yield `>= 5` valid tap windows (`SC-001`).
2. **TDD Green & Refactor Phase**:
    - Implement `normalize_peak`, `detect_onset_centers`, and `extract_tap_impulses` (delegating to `detect_onset_centers`).

### Phase 2C: Mel Filterbank, Shared STFT Power Spectrogram, Log-Mel (`128 x 128`) & 4 Physics Features (`src/dsp.rs`)

1. **TDD Red Phase**:
    - Write unit tests in `src/dsp.rs` covering:
        - `hz_to_mel` and `mel_to_hz` roundtrip accuracy (`approx::assert_relative_eq!`) and `build_mel_filterbank(22_050, 1024, 128, 50.0, 8_000.0)` shape `(128, 513)`, non-negativity, and Slaney area normalization.
        - `test_log_mel_spectrogram_shape_and_bounds`: verify `compute_power_spectrogram` returns shape `(513, 128)` and `compute_log_mel_spectrogram` returns shape `(128, 128)` with all elements finite and in `[0.0, 1.0]` for both normal tap windows and all-zero silent windows.
        - `test_physics_features_sine_wave_resonance`: synthesize Signal A (`Unripe`: `600 Hz` sine wave with slow exponential decay) and Signal B (`Overripe`: `180 Hz` sine wave with fast exponential decay); verify all 4 features lie in `[0.0, 1.0]`, $\tilde{f}_{\text{peak}}(A) > \tilde{f}_{\text{peak}}(B)$, $\tilde{C}_s(A) > \tilde{C}_s(B)$, and $\tilde{\beta}_{\text{decay}}(B) > \tilde{\beta}_{\text{decay}}(A)$ (`SC-004`).
        - Silent signal invariance: `extract_physics_features(&vec![0.0; 17_640], &cfg)` returns `[0.0, 0.0, 0.0, 0.0]`.
        - Single-pass equivalence: verify `extract_tap_features(audio, &cfg, &mel_basis)` produces identical `mel_spec` and `physics_features` to calling `compute_log_mel_spectrogram` and `extract_physics_features` separately.
2. **TDD Green & Refactor Phase**:
    - Implement `hz_to_mel`, `mel_to_hz`, `build_mel_filterbank`, `compute_power_spectrogram`, `compute_log_mel_spectrogram`, `extract_physics_features_from_power_spec`, `extract_physics_features`, and `extract_tap_features`.

### Phase 2D: Physically Valid Acoustic Data Augmentation (`src/augment.rs`)

1. **TDD Red Phase**:
    - Write unit tests in `src/augment.rs` covering:
        - `apply_onset_jitter`: preserves exact `17,640`-sample window length and `[-1.0, 1.0]` peak normalization when `center_sample` is near index `0`, near `full_samples.len()`, when `full_samples.len() < 17,640`, or when `max_jitter_samples == 0`.
        - `add_background_noise`: adds Box-Muller Gaussian white noise within `[15.0, 35.0] dB` SNR, maintains `17,640` length and `[-1.0, 1.0]` bounds, preserves peak resonance frequency $\tilde{f}_{\text{peak}}$ of a `600 Hz` tone, and never panics when `min_snr_db >= max_snr_db` or when input is silent/empty.
        - `apply_spectral_tilt`: applies linear dB tilt across `(128, 128)` Log-Mel rows, clamps all values to `[0.0, 1.0]`, and handles `max_tilt_db <= 0.0` or `nrows <= 1` without panicking.
        - `apply_spec_augment`: masks contiguous frequency and time bands to `noise_floor_val = 0.0`, keeps all values in `[0.0, 1.0]`, and safely handles `max_freq_mask == 0`, `max_time_mask == 0`, or small matrices (`< 4` rows/cols).
2. **TDD Green & Refactor Phase**:
    - Implement `apply_onset_jitter`, `add_background_noise`, `apply_spectral_tilt`, and `apply_spec_augment` in `src/augment.rs` using `rand 0.10.4` (`rand::rng()`, `RngExt::random_range`).

### Phase 2E: Criterion DSP Latency Benchmark & Quality Gate Verification (`benches/training_benchmarks.rs`)

1. **Benchmark & Verification**:
    - Extend `benches/training_benchmarks.rs` with a `dsp_pipeline` benchmark group measuring `extract_tap_features` (and `extract_tap_impulses`) on a `0.8s` (`17,640`-sample) tap window with a precomputed `mel_basis`, confirming single-threaded execution completes in `< 4.0 ms` (`SC-005`).
    - Run full constitutional quality gates: `cargo test`, `cargo bench --bench training_benchmarks -- --test`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check`.

## Complexity Tracking

No constitutional violations exist. All Phase 2 components use locked workspace dependencies (`symphonia 0.6.1`, `rubato 5.1.0`, `realfft 3.5.1`, `ndarray 0.17.2`, `rand 0.10.4`) and pure-Rust standard library primitives.
