# Implementation Plan: Phase 1 - Data Pipeline, Config Expansion & Anti-Leakage Dataset Splitter

**Branch**: `001-phase1-data-config` | **Date**: 2026-11-24 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-phase1-data-config/spec.md`

## Summary

Phase 1 establishes the configuration management, dataset integrity verification, and anti-leakage dataset splitting foundation for DurianSonic:

1. **Centralized Multi-Domain Configuration (`configs/default.toml` & `src/config.rs`)**: Expand `Config` to manage 4 domain blocks (`[dsp]`, `[dataset]`, `[training]`, `[model]`) with `#[serde(default)]` backward compatibility and automated physical/mathematical validation (Nyquist frequency bound `fmax <= sample_rate / 2.0`, power-of-two `n_fft`, split ratio sum tolerance `< 1e-4`, and hyperparameter bounds).
2. **Dataset Integrity Verifier (`src/dataset.rs`)**: Parse `data/metadata_iphone.csv` (`281` M4A records) and `data/metadata_android.csv` (`175` WAV records), verifying contiguous `sample_num (1..=N)`, physical `fruit_id (1..=175)`, valid `RipenessLabel (Unripe=0, Ripe=1, Overripe=2)`, file existence, exact byte size match, and streaming `64 KB` chunked `SHA-256` content uniqueness.
3. **Anti-Leakage Stratified Group Splitter & CLI (`src/dataset.rs` & `src/main.rs`)**: Partition each platform dataset independently into `Train (70%) / Validation (15%) / Test (15%)` grouped by physical `fruit_id (1..=175)` and stratified by rarest ripeness class (`Overripe > Ripe > Unripe`) with a deterministic seed (`split_seed = 42`), compute platform-specific inverse class frequency weights, and expose a `verify-dataset` CLI subcommand.

## Technical Context

**Language/Version**: Rust (Edition 2021)

**Primary Dependencies**: `serde 1.0`, `toml 1.2`, `csv 1.4`, `sha2 0.11`, `rand 0.10`, `clap 4.6`, `anyhow 1.0`, `thiserror 2.0`

**Storage**: Local filesystem (`configs/default.toml`, `data/metadata_iphone.csv`, `data/metadata_android.csv`, `data/sound_iphone/*.m4a`, `data/sound_android/*.wav`)

**Testing**: `cargo test` with `tempfile 3.27` and `approx 0.5`

**Target Platform**: Cross-platform developer workstation & CI (macOS / Linux / Windows) for dataset verification and model training preparation

**Project Type**: Rust Library + CLI (`src/lib.rs` + `src/main.rs`)

**Performance Goals**: Complete verification (including streaming `SHA-256` hashing of all `456` audio files totaling ~1.36 GB) and stratified group split auditing in `< 10 seconds` (`SC-005`)

**Constraints**:

- Zero physical fruit leakage across `Train`, `Validation`, and `Test` splits ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$)
- 100% deterministic splits for identical `split_seed`
- Strict adherence to DurianSonic Constitution (TDD Red-Green-Refactor, zero emojis, no decorative comment separators, no `unwrap()`/`expect()` in library code, zero clippy warnings)

**Scale/Scope**: `456` verified audio recordings (`281` iPhone `.m4a` + `175` Android `.wav`, ~1.36 GB total) across `175` unique physical durian fruits per platform

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design._

| Constitutional Principle               | Pre-Research Status | Post-Design Status | Compliance Evidence                                                                                                                                                                                                      |
| :------------------------------------- | :------------------ | :----------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **I. Clean Code Excellence**           | PASS                | PASS               | No bypass patterns; 100% emoji-free code and CLI output; explicit `ConfigError` (`thiserror`) in `src/config.rs` and `anyhow::Result` with context in `src/dataset.rs`; zero `unwrap()`/`expect()` in library functions. |
| **II. Test-First Development**         | PASS                | PASS               | Unit and integration tests defined first for every public struct and function in `src/config.rs` and `src/dataset.rs` before implementation.                                                                             |
| **III. Audio Quality Standards**       | PASS                | PASS               | `DspConfig::validate()` enforces physical audio invariants upfront: Nyquist limit (`fmax <= sample_rate / 2.0`), power-of-two `n_fft`, and valid onset window bounds (`pre_onset_sec < window_sec`).                     |
| **IV. Performance Requirements**       | PASS                | PASS               | `compute_file_sha256` uses a fixed `64 KB` stack buffer (`[0u8; 65536]`) avoiding heap allocations in the hashing loop and verifying 1.36 GB in ~2 seconds (`< 10s` gate).                                               |
| **V. Documentation & Maintainability** | PASS                | PASS               | Rustdoc on all public types/functions; concise `//` comments explaining _why_ without decorative block separators (`===` or `---`); formatted via `rustfmt`.                                                             |

## Project Structure

### Documentation (this feature)

```text
specs/001-phase1-data-config/
├── plan.md              # This implementation plan
├── research.md          # Phase 0 architecture & technical decisions
├── data-model.md        # Phase 1 entity schemas & validation rules
├── quickstart.md        # Phase 1 end-to-end validation guide
├── contracts/
│   └── cli-and-library.md # Public Rust library API & CLI contract
├── checklists/
│   └── requirements.md  # Specification quality checklist (16/16 passing)
└── tasks.md             # Phase 2 output (/speckit-tasks command)
```

### Source Code (repository root)

```text
Cargo.toml               # Crate manifest (dependencies already present)
configs/
└── default.toml         # Expanded 4-section configuration ([dsp], [dataset], [training], [model])
data/
├── metadata_iphone.csv  # 281 iPhone metadata records (fruit_id 1..=175)
├── metadata_android.csv # 175 Android metadata records (fruit_id 1..=175)
├── sound_iphone/        # 281 .m4a audio recordings
└── sound_android/       # 175 .wav audio recordings
src/
├── lib.rs               # Exports pub mod config; pub mod dataset; pub mod training;
├── main.rs              # CLI entry point with clap verify-dataset subcommand
├── config.rs            # Expanded Config, DspConfig, DatasetConfig, TrainingConfig, ModelConfig
├── dataset.rs           # NEW: PlatformTarget, RipenessLabel, AudioMetadataRecord, verifier & splitter
└── training.rs          # Existing training utilities
```

**Structure Decision**: Retain the single-crate Rust workspace layout (`durian-sonic` library crate in `src/lib.rs` paired with binary entry point in `src/main.rs`). This matches `docs/TECHNICAL_DESIGN.md` and `docs/phases/PHASE_1_DATA_AND_CONFIG.md`, ensuring zero code duplication between the CLI tool, training pipeline, and future FFI mobile bindings.

## Implementation Architecture & Phases

### Phase 1A: Configuration Expansion & Validation (`configs/default.toml`, `src/config.rs`)

1. **TDD Red Phase**:
    - Add unit tests in `src/config.rs` covering:
        - Loading the full 4-section `configs/default.toml` (`Config::load_default()`).
        - Loading legacy minimal TOML files (containing only `[training]` and `[model]`) and verifying `DspConfig::default()`, `DatasetConfig::default()`, and new training/model fields populate automatically.
        - Rejecting invalid `DspConfig` parameters: `n_fft` not a power of two (e.g., `1000`), `fmax` exceeding Nyquist (`fmax = 12000.0` with `sample_rate = 22050`), `fmin >= fmax`, `pre_onset_sec >= window_sec`, `threshold_ratio > 1.0` or `<= 0.0`.
        - Rejecting invalid `DatasetConfig` parameters: `train_ratio + val_ratio + test_ratio` differing from `1.0` by `>= 1e-4`, or any split ratio `<= 0.0`.
        - Rejecting invalid extended `TrainingConfig` and `ModelConfig` fields (`label_smoothing >= 1.0`, `dropout >= 1.0`, `num_classes < 2`).
2. **TDD Green & Refactor Phase**:
    - Update `configs/default.toml` with full `[dsp]`, `[dataset]`, `[training]`, and `[model]` blocks.
    - Implement `DspConfig`, `DatasetConfig`, and extended fields on `TrainingConfig` and `ModelConfig` in `src/config.rs` with `#[serde(default)]` and comprehensive checks in `Config::validate()`.

### Phase 1B: Dataset Metadata Loader, SHA-256 Verifier & Class Weights (`src/dataset.rs`, `src/lib.rs`)

1. **TDD Red Phase**:
    - Declare `pub mod dataset;` in `src/lib.rs`.
    - Write unit tests in `src/dataset.rs` covering:
        - `RipenessLabel::from_str_label` case-insensitive and whitespace-trimmed parsing (`"Unripe"`, `" ripe "`, `"OVERRIPE"`) and rejection of invalid labels.
        - `PlatformTarget` path resolution and model artifact naming (`durian_iphone`, `durian_android`).
        - `compute_class_weights(&[171, 66, 44])` and `compute_class_weights(&[95, 38, 42])` accuracy using `approx::assert_relative_eq!`.
        - `load_and_verify_metadata` failure cases on synthetic temporary directories (`tempfile`): non-sequential `sample_num` gap, `fruit_id` outside `1..=175`, missing audio file, `1`-byte file size mismatch, and duplicate `SHA-256` binary content across two differently named files.
2. **TDD Green & Refactor Phase**:
    - Implement `PlatformTarget`, `RipenessLabel`, `AudioMetadataRecord`, `IntegrityReport`, `load_and_verify_metadata`, `compute_file_sha256` (with 64 KB streaming buffer), and `compute_class_weights`.

### Phase 1C: Anti-Leakage Stratified Group Splitter (`src/dataset.rs`)

1. **TDD Red Phase**:
    - Write unit and integration tests in `src/dataset.rs` covering:
        - `test_zero_fruit_id_leakage_across_splits`: Synthetic multi-stage dataset where individual `fruit_id`s have `Unripe`, `Ripe`, and `Overripe` recordings; assert $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ and total unique fruits are preserved.
        - `test_deterministic_split_with_same_seed`: Running `stratified_group_split_by_fruit` twice with `seed = 42` returns identical record lists in identical order.
        - `test_real_metadata_csv_loading`: Integration test loading `data/metadata_iphone.csv` (`281` records, `175` fruits) and `data/metadata_android.csv` (`175` records, `175` fruits), verifying exact counts (`123` Train fruits, `26` Val fruits, `26` Test fruits) and zero leakage.
2. **TDD Green & Refactor Phase**:
    - Implement `DatasetSplits` and `stratified_group_split_by_fruit` using `BTreeMap<u32, Vec<AudioMetadataRecord>>`, rarity-priority stratum assignment (`Overripe > Ripe > Unripe`), and `StdRng::seed_from_u64(seed)`.

### Phase 1D: CLI `verify-dataset` Subcommand (`src/main.rs`)

1. **Implementation & Verification**:
    - Integrate `clap` derive CLI in `src/main.rs` supporting `verify-dataset --data-dir <PATH> --config <PATH>` while keeping default execution clean and warning-free.
    - Format console report output cleanly without emojis, displaying integrity metrics, class weights, split distributions, and overlap verification for both iPhone and Android datasets.
    - Run full quality gates: `cargo test`, `cargo run -- verify-dataset --data-dir data`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check`.

## Complexity Tracking

No constitutional violations exist. All design choices use standard library collections (`BTreeMap`, `HashMap`, `HashSet`) and already-installed workspace dependencies.
