# Phase 0 Research: Data Pipeline, Config Expansion & Anti-Leakage Dataset Splitter

**Feature**: `001-phase1-data-config`
**Date**: 2026-11-24
**Status**: Complete (All technical decisions resolved)

---

## 1. Backward-Compatible Multi-Domain Configuration (`src/config.rs`)

- **Decision**: Expand `Config` to include `dsp: DspConfig`, `dataset: DatasetConfig`, `training: TrainingConfig`, and `model: ModelConfig`, annotating new top-level sections (`dsp`, `dataset`) and newly added fields in `TrainingConfig` and `ModelConfig` with `#[serde(default)]`. Keep `ConfigError` with `thiserror::Error` for typed validation and I/O errors.
- **Rationale**:
    - `src/config.rs` already contains 9 unit tests that deserialize minimal TOML payloads containing only `[training]` (`epochs`, `batch_size`, `learning_rate`) and `[model]` (`hidden_layers`, `activation`).
    - Using `#[serde(default)]` at both the struct and field levels satisfies `FR-001` and `FR-002` by allowing partial TOML files to load cleanly with physically valid defaults while enforcing strict validation rules (`FR-003`) on all four configuration domains.
- **Alternatives Considered**:
    - _Require all fields in every TOML file without `#[serde(default)]`_: Rejected because it breaks existing tests (`test_load_from_file`) and violates `FR-002` backward-compatibility requirements.
    - _Separate configuration files for DSP, Dataset, and Training_: Rejected because a single source of truth (`configs/default.toml`) prevents parameter drift across DSP extraction, training, and mobile export phases.

---

## 2. Independent Dual-Platform Stratified Group Split (`src/dataset.rs`)

- **Decision**: Implement `stratified_group_split_by_fruit` independently per `PlatformTarget` (`Iphone` and `Android`) using a two-stage deterministic algorithm:
    1. Group all `AudioMetadataRecord` items by physical `fruit_id` (`1..=175`) using a `BTreeMap<u32, Vec<AudioMetadataRecord>>` so iteration order over fruit groups is strictly deterministic.
    2. Assign each `fruit_id` a single representative stratum label based on rarity priority (`Overripe (2) > Ripe (1) > Unripe (0)`), shuffle each stratum's `fruit_id` vector using `StdRng::seed_from_u64(split_seed)`, and partition the fruit IDs into `Train` (`round(n * train_ratio)`), `Validation` (`round(n * val_ratio)`), and `Test` (remaining fruits in stratum).
- **Rationale**:
    - In `metadata_iphone.csv` (`281` recordings across `175` physical fruits), individual durians were recorded across multiple ripening transitions (`Unripe -> Ripe -> Overripe`). Splitting at the recording or impulse level would leak the acoustic resonance signature of a physical fruit across `Train` and `Test`.
    - Prioritizing the rarest class (`Overripe > Ripe > Unripe`) when assigning a multi-stage fruit to a stratum prevents minority-class recordings from clustering into a single split.
    - Partitioning `175` physical fruits with `70% / 15% / 15%` yields `123` Train fruits, `26` Validation fruits, and `26` Test fruits (`197 / 41 / 43` files on iPhone; `123 / 26 / 26` files on Android) with zero overlap ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **Alternatives Considered**:
    - _Merged cross-platform dataset split_: Rejected because iPhone recordings use AAC-LC compressed `.m4a` with iOS microphone frequency response, whereas Android recordings use uncompressed 16-bit PCM `.wav` with Android microphone response; the architecture trains two dedicated models (`durian_iphone` and `durian_android`).
    - _HashMap for fruit grouping_: Rejected because `HashMap` iteration order is randomized per process in Rust, which would break `SC-003` (100% reproducibility with `split_seed = 42`).

---

## 3. Streaming SHA-256 Deduplication & Fast-Fail Metadata Verification

- **Decision**: Validate dataset records sequentially in `load_and_verify_metadata(csv_path, audio_dir, verify_sha256)` in increasing cost order:
    1. CSV schema & row parsing (`csv::Reader::from_path`).
    2. Contiguous sequential `sample_num` check (`expected_num = idx + 1`).
    3. Physical fruit identifier range check (`1..=175`).
    4. Case-insensitive, whitespace-trimmed ripeness label parsing (`Unripe`, `Ripe`, `Overripe`).
    5. File existence and exact byte size match (`fs::metadata(path)?.len() == record.file_size_bytes`).
    6. Optional streaming SHA-256 content hash calculation using a fixed `64 KB` stack buffer (`[0u8; 65536]`) and duplicate check via `HashMap<String, String>`.
- **Rationale**:
    - Checking metadata invariants and file size via `stat` syscall before reading file bytes ensures immediate failure on corrupted or missing files (`Edge Case 5`).
    - Using a fixed `64 KB` buffer with `sha2::Sha256` keeps heap allocations near zero (satisfying Constitution Principle IV) and hashes all `456` files (~1.36 GB total) in ~1.5–3.0 seconds on standard developer hardware, comfortably below the `< 10s` requirement (`SC-005`).
- **Alternatives Considered**:
    - _Reading full audio files into memory with `fs::read`_: Rejected due to unnecessary multi-megabyte heap allocations per file.
    - _Relying solely on filename or file size for deduplication_: Rejected because two distinct audio takes with identical duration/format in uncompressed PCM WAV can have identical byte sizes, and the raw dataset historically contained 106 duplicate files under different filenames.

---

## 4. Platform-Specific Inverse Class Frequency Weighting

- **Decision**: Compute normalized inverse class frequency weights per platform using:
  $$w_c = \frac{N_{\text{total}}}{3 \cdot N_c}$$
  for $c \in \{\text{Unripe}, \text{Ripe}, \text{Overripe}\}$, returning `[0.0; 3]` (or `0.0` for empty class) safely if any count is zero in synthetic edge-case tests to prevent division-by-zero panics.
- **Rationale**:
    - Both datasets exhibit class imbalance (`171 : 66 : 44` on iPhone; `95 : 38 : 42` on Android). Pre-computing exact weights in Phase 1 (`[0.5478, 1.4192, 2.1288]` for iPhone and `[0.6140, 1.5351, 1.3889]` for Android) provides the exact loss weights required by Phase 3 weighted cross-entropy training while upholding Constitution Principle I (no unguarded panics/div-by-zero).
- **Alternatives Considered**:
    - _Hardcoding static class weights in config_: Rejected because dynamically deriving weights from verified metadata records ensures weights stay accurate whether computed over the full dataset or the training split.

---

## 5. `rand 0.10` Compatibility Note

- **Decision**: Ensure slice shuffling in `src/dataset.rs` uses the `rand = "0.10.4"` API installed in `Cargo.toml` (`use rand::seq::SliceRandom;` and `use rand::SeedableRng;`, invoking `fruit_list.shuffle(&mut rng)` on `rand::rngs::StdRng::seed_from_u64(seed)`).
- **Rationale**: Aligns directly with the crate version locked in `Cargo.toml` without adding extra dependencies.
- **Alternatives Considered**:
    - _Downgrading `rand` to `0.8`_: Rejected because `Cargo.toml` is already standardized on `rand = "0.10.4"`.
