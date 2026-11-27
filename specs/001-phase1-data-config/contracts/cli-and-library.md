# Phase 1 Interface Contracts: Rust Library API & CLI

**Feature**: `001-phase1-data-config`
**Date**: 2026-11-24

---

## 1. Public Library API Contract (`durian_sonic`)

### 1.1. Module `durian_sonic::config`

```rust
pub enum ConfigError {
    IoError(std::io::Error),
    ParseError(toml::de::Error),
    ValidationError(String),
}

pub struct Config {
    pub dsp: DspConfig,
    pub dataset: DatasetConfig,
    pub training: TrainingConfig,
    pub model: ModelConfig,
}

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError>;
    pub fn load_default() -> Result<Self, ConfigError>;
    pub fn validate(&self) -> Result<(), ConfigError>;
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError>;
}
```

### 1.2. Module `durian_sonic::dataset`

```rust
pub enum PlatformTarget {
    Iphone,
    Android,
}

impl PlatformTarget {
    pub fn metadata_csv_path(self, data_dir: &Path) -> PathBuf;
    pub fn audio_subdir(self, data_dir: &Path) -> PathBuf;
    pub fn default_model_artifact_name(self) -> &'static str;
}

pub enum RipenessLabel {
    Unripe = 0,
    Ripe = 1,
    Overripe = 2,
}

impl RipenessLabel {
    pub fn from_str_label(s: &str) -> anyhow::Result<Self>;
    pub fn as_index(self) -> usize;
    pub fn as_str(self) -> &'static str;
}

pub struct AudioMetadataRecord {
    pub sample_num: u32,
    pub fruit_id: u32,
    pub ripeness: String,
    pub filename: String,
    pub original_filename: String,
    pub file_size_bytes: u64,
    pub file_date: String,
}

impl AudioMetadataRecord {
    pub fn label(&self) -> anyhow::Result<RipenessLabel>;
    pub fn audio_path(&self, data_dir: &Path, audio_subdir: &str) -> PathBuf;
}

pub struct IntegrityReport {
    pub total_files: usize,
    pub unique_fruits: usize,
    pub class_counts: [usize; 3],
    pub total_bytes: u64,
}

pub struct DatasetSplits {
    pub train: Vec<AudioMetadataRecord>,
    pub val: Vec<AudioMetadataRecord>,
    pub test: Vec<AudioMetadataRecord>,
}

pub fn load_and_verify_metadata(
    csv_path: &Path,
    audio_dir: &Path,
    verify_sha256: bool,
) -> anyhow::Result<(Vec<AudioMetadataRecord>, IntegrityReport)>;

pub fn stratified_group_split_by_fruit(
    records: &[AudioMetadataRecord],
    train_ratio: f32,
    val_ratio: f32,
    seed: u64,
) -> anyhow::Result<DatasetSplits>;

pub fn compute_class_weights(class_counts: &[usize; 3]) -> [f32; 3];
```

---

## 2. CLI Contract (`durian-sonic`)

### 2.1. Command Syntax (`FR-011`)

```text
durian-sonic verify-dataset [--data-dir <PATH>] [--config <PATH>]
```

### 2.2. Arguments & Flags

| Flag / Option | Type      | Default                                       | Description                                                                                                                |
| :------------ | :-------- | :-------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| `--data-dir`  | `PathBuf` | Value of `config.dataset.data_dir` (`"data"`) | Override path to directory containing `metadata_iphone.csv`, `metadata_android.csv`, `sound_iphone/`, and `sound_android/` |
| `--config`    | `PathBuf` | `"configs/default.toml"`                      | Path to TOML configuration file                                                                                            |

### 2.3. Behavioral Contract & Exit Codes

1. **Subcommand & Configuration Loading**: Requires an explicit subcommand via `clap` (invoking `durian-sonic` without a subcommand prints CLI help/usage). Loads and validates the configuration from `--config` (defaulting to `"configs/default.toml"`). If the configuration file does not exist, cannot be read, or fails validation, prints a descriptive error message to `stderr` and exits immediately with non-zero status (`1`). When `--data-dir` is provided, it overrides `config.dataset.data_dir` for resolving platform metadata and audio directories.
2. **Platform Verification Loop**: For both `PlatformTarget::Iphone` and `PlatformTarget::Android`:
    - Calls `load_and_verify_metadata(csv_path, audio_dir, true)` (with `verify_sha256 = true`).
    - Calls `stratified_group_split_by_fruit(&records, config.dataset.train_ratio, config.dataset.val_ratio, config.dataset.split_seed)`.
    - Verifies that the intersection of `fruit_id` sets across `Train`, `Validation`, and `Test` equals `0`.
    - Computes inverse class frequency weights via `compute_class_weights(&report.class_counts)`.
3. **Standard Output Summary**: Prints an emoji-free structured summary for each platform including:
    - Total verified files, unique physical fruits (`175`), total byte size, and `[Unripe, Ripe, Overripe]` counts.
    - Normalized class weights `[w_unripe, w_ripe, w_overripe]`.
    - `Train / Val / Test` file counts, unique fruit counts (`123 / 26 / 26`), and confirmation that `Overlap fruit_ids = 0`.
4. **Exit Status**:
    - `0`: All 456 files verified, 0 SHA-256 duplicates, 0 fruit leakage across splits.
    - Non-zero (`1`): Any missing file, byte size mismatch, sequential `sample_num` gap, invalid `fruit_id`, invalid label, duplicate SHA-256 hash, or fruit ID leakage.
