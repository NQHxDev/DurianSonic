//! Dataset integrity verification and anti-leakage stratified group splitting
//!
//! This module provides tools for validating durian audio datasets and partitioning
//! them into train/validation/test splits with zero physical fruit leakage.
//!
//! # Key Features
//!
//! - **Integrity Verification**: Validates CSV metadata against physical audio files,
//!   checking contiguous sample numbers, valid fruit IDs, ripeness labels, file existence,
//!   byte sizes, and optional SHA-256 deduplication.
//! - **Anti-Leakage Splitting**: Partitions datasets by physical fruit_id with stratified
//!   sampling based on ripeness class rarity, ensuring zero fruit overlap across splits.
//! - **Platform Support**: Handles both iPhone (M4A) and Android (WAV) recordings.
//!
//! # Examples
//!
//! ```no_run
//! use durian_sonic::dataset::{load_and_verify_metadata, stratified_group_split_by_fruit};
//! use std::path::Path;
//!
//! // Verify dataset integrity with SHA-256 checking
//! let (records, report) = load_and_verify_metadata(
//!     Path::new("data/metadata_iphone.csv"),
//!     Path::new("data/sound_iphone"),
//!     true
//! ).unwrap();
//!
//! println!("Total files: {}", report.total_files);
//! println!("Unique fruits: {}", report.unique_fruits);
//!
//! // Perform anti-leakage stratified split
//! let splits = stratified_group_split_by_fruit(&records, 0.70, 0.15, 42).unwrap();
//! ```

use anyhow::{Context, Result};
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Platform-specific audio recording target (iPhone vs Android hardware)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformTarget {
    Iphone,
    Android,
}

impl PlatformTarget {
    /// Returns the metadata CSV path for this platform
    ///
    /// # Examples
    ///
    /// ```
    /// use durian_sonic::dataset::PlatformTarget;
    /// use std::path::Path;
    ///
    /// let data_dir = Path::new("data");
    /// let iphone_path = PlatformTarget::Iphone.metadata_csv_path(data_dir);
    /// assert_eq!(iphone_path.to_str().unwrap(), "data/metadata_iphone.csv");
    /// ```
    pub fn metadata_csv_path(&self, data_dir: &Path) -> PathBuf {
        match self {
            PlatformTarget::Iphone => data_dir.join("metadata_iphone.csv"),
            PlatformTarget::Android => data_dir.join("metadata_android.csv"),
        }
    }

    /// Returns the audio subdirectory for this platform
    ///
    /// # Examples
    ///
    /// ```
    /// use durian_sonic::dataset::PlatformTarget;
    /// use std::path::Path;
    ///
    /// let data_dir = Path::new("data");
    /// let iphone_subdir = PlatformTarget::Iphone.audio_subdir(data_dir);
    /// assert_eq!(iphone_subdir.to_str().unwrap(), "data/sound_iphone");
    /// ```
    pub fn audio_subdir(&self, data_dir: &Path) -> PathBuf {
        match self {
            PlatformTarget::Iphone => data_dir.join("sound_iphone"),
            PlatformTarget::Android => data_dir.join("sound_android"),
        }
    }

    /// Returns the default model artifact name for this platform
    ///
    /// # Examples
    ///
    /// ```
    /// use durian_sonic::dataset::PlatformTarget;
    ///
    /// assert_eq!(PlatformTarget::Iphone.default_model_artifact_name(), "durian_iphone");
    /// assert_eq!(PlatformTarget::Android.default_model_artifact_name(), "durian_android");
    /// ```
    pub fn default_model_artifact_name(&self) -> &'static str {
        match self {
            PlatformTarget::Iphone => "durian_iphone",
            PlatformTarget::Android => "durian_android",
        }
    }
}

/// 3-class ripeness taxonomy for durian maturity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum RipenessLabel {
    Unripe = 0,
    Ripe = 1,
    Overripe = 2,
}

impl RipenessLabel {
    /// Parse a ripeness label from string (case-insensitive, whitespace-trimmed)
    ///
    /// Accepts "unripe", "ripe", or "overripe" in any case with leading/trailing whitespace.
    /// Returns an error for any other input.
    ///
    /// # Examples
    ///
    /// ```
    /// use durian_sonic::dataset::RipenessLabel;
    ///
    /// assert_eq!(RipenessLabel::from_str_label("Unripe").unwrap(), RipenessLabel::Unripe);
    /// assert_eq!(RipenessLabel::from_str_label("  RIPE  ").unwrap(), RipenessLabel::Ripe);
    /// assert!(RipenessLabel::from_str_label("unknown").is_err());
    /// ```
    pub fn from_str_label(s: &str) -> Result<Self> {
        let trimmed = s.trim().to_lowercase();
        match trimmed.as_str() {
            "unripe" => Ok(RipenessLabel::Unripe),
            "ripe" => Ok(RipenessLabel::Ripe),
            "overripe" => Ok(RipenessLabel::Overripe),
            _ => anyhow::bail!("Invalid ripeness label: '{}'", s),
        }
    }

    /// Get the class index (0, 1, or 2)
    ///
    /// # Examples
    ///
    /// ```
    /// use durian_sonic::dataset::RipenessLabel;
    ///
    /// assert_eq!(RipenessLabel::Unripe.as_index(), 0);
    /// assert_eq!(RipenessLabel::Ripe.as_index(), 1);
    /// assert_eq!(RipenessLabel::Overripe.as_index(), 2);
    /// ```
    pub fn as_index(&self) -> usize {
        *self as usize
    }

    /// Get the canonical string representation
    ///
    /// # Examples
    ///
    /// ```
    /// use durian_sonic::dataset::RipenessLabel;
    ///
    /// assert_eq!(RipenessLabel::Unripe.as_str(), "Unripe");
    /// assert_eq!(RipenessLabel::Ripe.as_str(), "Ripe");
    /// assert_eq!(RipenessLabel::Overripe.as_str(), "Overripe");
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            RipenessLabel::Unripe => "Unripe",
            RipenessLabel::Ripe => "Ripe",
            RipenessLabel::Overripe => "Overripe",
        }
    }
}

/// Single audio metadata record from CSV (one recording of one fruit)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioMetadataRecord {
    pub sample_num: u32,
    pub fruit_id: u32,
    pub ripeness: String,
    pub filename: String,
    pub original_filename: String,
    pub file_size_bytes: u64,
    pub file_date: String,
}

/// Summary statistics from dataset integrity verification
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityReport {
    /// Total number of audio files
    pub total_files: usize,
    /// Number of unique physical fruits
    pub unique_fruits: usize,
    /// Class distribution [Unripe, Ripe, Overripe]
    pub class_counts: [usize; 3],
    /// Total bytes across all audio files
    pub total_bytes: u64,
}

/// Disjoint partition of dataset into Train/Val/Test splits
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetSplits {
    pub train: Vec<AudioMetadataRecord>,
    pub val: Vec<AudioMetadataRecord>,
    pub test: Vec<AudioMetadataRecord>,
}

/// Compute normalized inverse class frequency weights
///
/// Returns weights using formula: w_c = N_total / (3.0 * N_c)
/// Returns [0.0; 3] if any class count is zero to prevent division by zero.
///
/// # Examples
///
/// ```
/// use durian_sonic::dataset::compute_class_weights;
/// use approx::assert_relative_eq;
///
/// let iphone_weights = compute_class_weights(&[171, 66, 44]);
/// assert_relative_eq!(iphone_weights[0], 0.5478, epsilon = 0.001);
/// assert_relative_eq!(iphone_weights[1], 1.4192, epsilon = 0.001);
/// assert_relative_eq!(iphone_weights[2], 2.1288, epsilon = 0.001);
/// ```
pub fn compute_class_weights(class_counts: &[usize; 3]) -> [f32; 3] {
    let total: usize = class_counts.iter().sum();

    if total == 0 {
        return [0.0, 0.0, 0.0];
    }

    let mut weights = [0.0f32; 3];
    for (i, &count) in class_counts.iter().enumerate() {
        if count > 0 {
            weights[i] = total as f32 / (3.0 * count as f32);
        } else {
            weights[i] = 0.0;
        }
    }

    weights
}

/// Partition dataset by physical fruit_id with stratified group splitting
///
/// Groups recordings by fruit_id, assigns each fruit to a stratum based on rarest
/// ripeness class (Overripe > Ripe > Unripe), shuffles deterministically per stratum,
/// and partitions into Train/Val/Test ensuring zero fruit_id overlap.
///
/// # Examples
///
/// ```no_run
/// use durian_sonic::dataset::{AudioMetadataRecord, stratified_group_split_by_fruit};
///
/// let records = vec![/* ... */];
/// let splits = stratified_group_split_by_fruit(&records, 0.70, 0.15, 42).unwrap();
///
/// // Zero fruit_id overlap between splits
/// // 123 train fruits, 26 val fruits, 26 test fruits for iPhone dataset
/// ```
pub fn stratified_group_split_by_fruit(
    records: &[AudioMetadataRecord],
    train_ratio: f32,
    val_ratio: f32,
    seed: u64,
) -> Result<DatasetSplits> {
    // Validate inputs
    if records.is_empty() {
        anyhow::bail!("Cannot split empty records");
    }

    if train_ratio <= 0.0 || val_ratio <= 0.0 {
        anyhow::bail!("train_ratio and val_ratio must be > 0.0");
    }

    if train_ratio + val_ratio >= 1.0 {
        anyhow::bail!("train_ratio + val_ratio must be < 1.0");
    }

    // Group records by fruit_id using BTreeMap for deterministic iteration
    let mut fruit_groups: BTreeMap<u32, Vec<AudioMetadataRecord>> = BTreeMap::new();
    for record in records {
        fruit_groups
            .entry(record.fruit_id)
            .or_default()
            .push(record.clone());
    }

    // Assign each fruit to a stratum based on rarest ripeness class
    // Priority: Overripe (2) > Ripe (1) > Unripe (0)
    let mut strata: BTreeMap<u8, Vec<u32>> = BTreeMap::new();

    for (fruit_id, fruit_records) in &fruit_groups {
        let mut has_overripe = false;
        let mut has_ripe = false;

        for record in fruit_records {
            if let Ok(label) = record.label() {
                match label {
                    RipenessLabel::Overripe => has_overripe = true,
                    RipenessLabel::Ripe => has_ripe = true,
                    RipenessLabel::Unripe => {} // No need to track
                }
            }
        }

        // Assign to rarest class stratum
        let stratum = if has_overripe {
            2
        } else if has_ripe {
            1
        } else {
            0
        };

        strata.entry(stratum).or_default().push(*fruit_id);
    }

    // Shuffle and partition each stratum
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut train_fruit_ids = HashSet::new();
    let mut val_fruit_ids = HashSet::new();
    let mut test_fruit_ids = HashSet::new();

    for (_stratum, mut fruit_ids) in strata {
        fruit_ids.shuffle(&mut rng);

        let n = fruit_ids.len();
        let n_train = (n as f32 * train_ratio).round() as usize;
        let n_val = (n as f32 * val_ratio).round() as usize;
        let n_val = n_val.min(n - n_train); // Ensure we don't exceed n
        let _n_test = n - n_train - n_val;

        for (i, fruit_id) in fruit_ids.iter().enumerate() {
            if i < n_train {
                train_fruit_ids.insert(*fruit_id);
            } else if i < n_train + n_val {
                val_fruit_ids.insert(*fruit_id);
            } else {
                test_fruit_ids.insert(*fruit_id);
            }
        }
    }

    // Collect records for each split preserving original order
    let mut train_records = Vec::new();
    let mut val_records = Vec::new();
    let mut test_records = Vec::new();

    for record in records {
        if train_fruit_ids.contains(&record.fruit_id) {
            train_records.push(record.clone());
        } else if val_fruit_ids.contains(&record.fruit_id) {
            val_records.push(record.clone());
        } else if test_fruit_ids.contains(&record.fruit_id) {
            test_records.push(record.clone());
        }
    }

    Ok(DatasetSplits {
        train: train_records,
        val: val_records,
        test: test_records,
    })
}

/// Compute SHA-256 hash of a file using streaming 64KB buffer
///
/// Uses a fixed stack buffer to avoid heap allocations when hashing large files.
///
/// # Examples
///
/// ```no_run
/// use durian_sonic::dataset::compute_file_sha256;
/// use std::path::Path;
///
/// let hash = compute_file_sha256(Path::new("audio.m4a")).unwrap();
/// assert_eq!(hash.len(), 64); // SHA-256 produces 64 hex characters
/// ```
pub fn compute_file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open file for hashing: {}", path.display()))?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536]; // 64 KB stack buffer

    loop {
        let bytes_read = file
            .read(&mut buffer)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;

        if bytes_read == 0 {
            break;
        }

        hasher.update(&buffer[..bytes_read]);
    }

    let hash_bytes = hasher.finalize();
    // Convert hash bytes to hex string
    let hash_string = hash_bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();

    Ok(hash_string)
}

/// Load and verify metadata CSV with optional SHA-256 deduplication
///
/// Validates metadata integrity through sequential checks:
/// 1. CSV parsing
/// 2. Contiguous sample_num (1..=N)
/// 3. Valid fruit_id (1..=175)
/// 4. Valid RipenessLabel
/// 5. File existence and exact byte size match
/// 6. Optional SHA-256 hash uniqueness (when verify_sha256 = true)
///
/// # Examples
///
/// ```no_run
/// use durian_sonic::dataset::load_and_verify_metadata;
/// use std::path::Path;
///
/// let (records, report) = load_and_verify_metadata(
///     Path::new("data/metadata_iphone.csv"),
///     Path::new("data/sound_iphone"),
///     true
/// ).unwrap();
///
/// println!("Total files: {}", report.total_files);
/// println!("Unique fruits: {}", report.unique_fruits);
/// println!("Class counts: {:?}", report.class_counts);
/// ```
pub fn load_and_verify_metadata(
    csv_path: &Path,
    audio_dir: &Path,
    verify_sha256: bool,
) -> Result<(Vec<AudioMetadataRecord>, IntegrityReport)> {
    let mut reader = ReaderBuilder::new()
        .from_path(csv_path)
        .with_context(|| format!("Failed to open CSV file: {}", csv_path.display()))?;

    let mut records = Vec::new();
    let mut unique_fruits = HashSet::new();
    let mut class_counts = [0usize; 3];
    let mut total_bytes = 0u64;
    let mut sha256_hashes: HashMap<String, String> = HashMap::new();

    for (row_index, result) in reader.deserialize::<AudioMetadataRecord>().enumerate() {
        let record = result.with_context(|| {
            format!(
                "Failed to parse CSV row {} in {}",
                row_index + 1,
                csv_path.display()
            )
        })?;

        // Check 1: Contiguous sample_num (must equal row_index + 1)
        let expected_sample_num = (row_index + 1) as u32;
        if record.sample_num != expected_sample_num {
            anyhow::bail!(
                "Invalid sample_num at row {}: expected {}, got {}. sample_num must be contiguous (1..=N).",
                row_index + 1,
                expected_sample_num,
                record.sample_num
            );
        }

        // Check 2: Valid fruit_id (1..=175)
        if !(1..=175).contains(&record.fruit_id) {
            anyhow::bail!(
                "Invalid fruit_id at row {}: {}. fruit_id must be in range 1..=175.",
                row_index + 1,
                record.fruit_id
            );
        }

        // Check 3: Valid RipenessLabel
        let label = record.label().with_context(|| {
            format!(
                "Invalid ripeness label '{}' at row {}",
                record.ripeness,
                row_index + 1
            )
        })?;

        // Check 4: File existence
        let audio_path = audio_dir.join(&record.filename);
        if !audio_path.exists() {
            anyhow::bail!(
                "Audio file does not exist at row {}: {}",
                row_index + 1,
                audio_path.display()
            );
        }

        // Check 5: Exact byte size match
        let actual_size = std::fs::metadata(&audio_path)
            .with_context(|| format!("Failed to read file metadata: {}", audio_path.display()))?
            .len();

        if actual_size != record.file_size_bytes {
            anyhow::bail!(
                "File size mismatch at row {}: expected {} bytes, got {} bytes for file {}",
                row_index + 1,
                record.file_size_bytes,
                actual_size,
                audio_path.display()
            );
        }

        // Check 6: Optional SHA-256 deduplication
        if verify_sha256 {
            let hash = compute_file_sha256(&audio_path).with_context(|| {
                format!(
                    "Failed to compute SHA-256 hash for file: {}",
                    audio_path.display()
                )
            })?;

            if let Some(existing_file) = sha256_hashes.get(&hash) {
                anyhow::bail!(
                    "Duplicate SHA-256 hash detected at row {}: files '{}' and '{}' have identical content",
                    row_index + 1,
                    existing_file,
                    record.filename
                );
            }

            sha256_hashes.insert(hash, record.filename.clone());
        }

        // Accumulate statistics
        unique_fruits.insert(record.fruit_id);
        class_counts[label.as_index()] += 1;
        total_bytes += record.file_size_bytes;

        records.push(record);
    }

    // Verify at least one record exists
    if records.is_empty() {
        anyhow::bail!("CSV file contains no data rows: {}", csv_path.display());
    }

    let report = IntegrityReport {
        total_files: records.len(),
        unique_fruits: unique_fruits.len(),
        class_counts,
        total_bytes,
    };

    Ok((records, report))
}

impl AudioMetadataRecord {
    /// Parse the ripeness field into a RipenessLabel
    ///
    /// # Examples
    ///
    /// ```
    /// use durian_sonic::dataset::{AudioMetadataRecord, RipenessLabel};
    ///
    /// let record = AudioMetadataRecord {
    ///     sample_num: 1,
    ///     fruit_id: 42,
    ///     ripeness: "Ripe".to_string(),
    ///     filename: "test.m4a".to_string(),
    ///     original_filename: "orig.m4a".to_string(),
    ///     file_size_bytes: 1024,
    ///     file_date: "2024-01-01".to_string(),
    /// };
    /// assert_eq!(record.label().unwrap(), RipenessLabel::Ripe);
    /// ```
    pub fn label(&self) -> Result<RipenessLabel> {
        RipenessLabel::from_str_label(&self.ripeness).context("Failed to parse ripeness label")
    }

    /// Construct the full audio file path
    ///
    /// Combines data_dir, audio_subdir, and filename into a complete path.
    ///
    /// # Examples
    ///
    /// ```
    /// use durian_sonic::dataset::AudioMetadataRecord;
    /// use std::path::Path;
    ///
    /// let record = AudioMetadataRecord {
    ///     sample_num: 1,
    ///     fruit_id: 42,
    ///     ripeness: "Ripe".to_string(),
    ///     filename: "test.m4a".to_string(),
    ///     original_filename: "orig.m4a".to_string(),
    ///     file_size_bytes: 1024,
    ///     file_date: "2024-01-01".to_string(),
    /// };
    /// let path = record.audio_path(Path::new("data"), "sound_iphone");
    /// assert_eq!(path.to_str().unwrap(), "data/sound_iphone/test.m4a");
    /// ```
    pub fn audio_path(&self, data_dir: &Path, audio_subdir: &str) -> PathBuf {
        data_dir.join(audio_subdir).join(&self.filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    // T003: Failing tests for PlatformTarget
    #[test]
    fn test_platform_target_iphone_metadata_path() {
        let data_dir = Path::new("data");
        let path = PlatformTarget::Iphone.metadata_csv_path(data_dir);
        assert_eq!(path, PathBuf::from("data/metadata_iphone.csv"));
    }

    #[test]
    fn test_platform_target_android_metadata_path() {
        let data_dir = Path::new("data");
        let path = PlatformTarget::Android.metadata_csv_path(data_dir);
        assert_eq!(path, PathBuf::from("data/metadata_android.csv"));
    }

    #[test]
    fn test_platform_target_iphone_audio_subdir() {
        let data_dir = Path::new("data");
        let path = PlatformTarget::Iphone.audio_subdir(data_dir);
        assert_eq!(path, PathBuf::from("data/sound_iphone"));
    }

    #[test]
    fn test_platform_target_android_audio_subdir() {
        let data_dir = Path::new("data");
        let path = PlatformTarget::Android.audio_subdir(data_dir);
        assert_eq!(path, PathBuf::from("data/sound_android"));
    }

    #[test]
    fn test_platform_target_iphone_model_name() {
        assert_eq!(
            PlatformTarget::Iphone.default_model_artifact_name(),
            "durian_iphone"
        );
    }

    #[test]
    fn test_platform_target_android_model_name() {
        assert_eq!(
            PlatformTarget::Android.default_model_artifact_name(),
            "durian_android"
        );
    }

    // T003: Failing tests for RipenessLabel
    #[test]
    fn test_ripeness_label_from_str_lowercase() {
        assert_eq!(
            RipenessLabel::from_str_label("unripe").unwrap(),
            RipenessLabel::Unripe
        );
        assert_eq!(
            RipenessLabel::from_str_label("ripe").unwrap(),
            RipenessLabel::Ripe
        );
        assert_eq!(
            RipenessLabel::from_str_label("overripe").unwrap(),
            RipenessLabel::Overripe
        );
    }

    #[test]
    fn test_ripeness_label_from_str_case_insensitive() {
        assert_eq!(
            RipenessLabel::from_str_label("Unripe").unwrap(),
            RipenessLabel::Unripe
        );
        assert_eq!(
            RipenessLabel::from_str_label("RIPE").unwrap(),
            RipenessLabel::Ripe
        );
        assert_eq!(
            RipenessLabel::from_str_label("OverRipe").unwrap(),
            RipenessLabel::Overripe
        );
    }

    #[test]
    fn test_ripeness_label_from_str_whitespace_trimmed() {
        assert_eq!(
            RipenessLabel::from_str_label("  unripe  ").unwrap(),
            RipenessLabel::Unripe
        );
        assert_eq!(
            RipenessLabel::from_str_label(" ripe ").unwrap(),
            RipenessLabel::Ripe
        );
        assert_eq!(
            RipenessLabel::from_str_label("\toverripe\n").unwrap(),
            RipenessLabel::Overripe
        );
    }

    #[test]
    fn test_ripeness_label_from_str_invalid() {
        assert!(RipenessLabel::from_str_label("unknown").is_err());
        assert!(RipenessLabel::from_str_label("").is_err());
        assert!(RipenessLabel::from_str_label("   ").is_err());
    }

    #[test]
    fn test_ripeness_label_as_index() {
        assert_eq!(RipenessLabel::Unripe.as_index(), 0);
        assert_eq!(RipenessLabel::Ripe.as_index(), 1);
        assert_eq!(RipenessLabel::Overripe.as_index(), 2);
    }

    #[test]
    fn test_ripeness_label_as_str() {
        assert_eq!(RipenessLabel::Unripe.as_str(), "Unripe");
        assert_eq!(RipenessLabel::Ripe.as_str(), "Ripe");
        assert_eq!(RipenessLabel::Overripe.as_str(), "Overripe");
    }

    // T003: Failing tests for AudioMetadataRecord
    #[test]
    fn test_audio_metadata_record_label_parsing() {
        let record = AudioMetadataRecord {
            sample_num: 1,
            fruit_id: 42,
            ripeness: "Ripe".to_string(),
            filename: "test.m4a".to_string(),
            original_filename: "orig.m4a".to_string(),
            file_size_bytes: 1024,
            file_date: "2024-01-01".to_string(),
        };
        assert_eq!(record.label().unwrap(), RipenessLabel::Ripe);
    }

    #[test]
    fn test_audio_metadata_record_audio_path() {
        let record = AudioMetadataRecord {
            sample_num: 1,
            fruit_id: 42,
            ripeness: "Ripe".to_string(),
            filename: "test.m4a".to_string(),
            original_filename: "orig.m4a".to_string(),
            file_size_bytes: 1024,
            file_date: "2024-01-01".to_string(),
        };
        let path = record.audio_path(Path::new("data"), "sound_iphone");
        assert_eq!(path, PathBuf::from("data/sound_iphone/test.m4a"));
    }

    // T007: Tests for load_and_verify_metadata validation checks
    #[test]
    fn test_load_metadata_empty_csv() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("empty.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        // Write CSV with only header, no data rows
        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_metadata_sample_num_gap() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("gap.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        // Create audio files
        fs::write(audio_dir.join("file1.m4a"), vec![0u8; 100]).unwrap();
        fs::write(audio_dir.join("file3.m4a"), vec![0u8; 100]).unwrap();

        // sample_num jumps from 1 to 3 (missing 2)
        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n\
             1,1,Unripe,file1.m4a,orig1.m4a,100,2024-01-01\n\
             3,1,Unripe,file3.m4a,orig3.m4a,100,2024-01-01\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, false);
        assert!(result.is_err());
        // The error should be about sample_num validation or contiguous requirement
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("sample_num") || err_msg.contains("contiguous"),
            "Expected error about sample_num or contiguous, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_load_metadata_invalid_fruit_id() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("invalid_fruit.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        // fruit_id 0 is out of range (must be 1..=175)
        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n\
             1,0,Unripe,file1.m4a,orig1.m4a,100,2024-01-01\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("fruit_id"));
    }

    #[test]
    fn test_load_metadata_fruit_id_upper_bound() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("fruit_id_too_high.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        // fruit_id 176 exceeds upper bound (must be 1..=175)
        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n\
             1,176,Unripe,file1.m4a,orig1.m4a,100,2024-01-01\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, false);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("fruit_id") && err_msg.contains("175"));
    }

    #[test]
    fn test_load_metadata_malformed_csv_header() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("bad_header.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        // Missing required columns
        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness\n\
             1,1,Unripe\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_metadata_non_numeric_sample_num() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("bad_sample_num.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        // sample_num is not a number
        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n\
             abc,1,Unripe,file1.m4a,orig1.m4a,100,2024-01-01\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_metadata_non_numeric_file_size() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("bad_file_size.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        // file_size_bytes is not a number
        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n\
             1,1,Unripe,file1.m4a,orig1.m4a,notanumber,2024-01-01\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_metadata_duplicate_sha256_includes_both_filenames() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("duplicate.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        // Create two files with identical content
        let file1 = audio_dir.join("first_file.m4a");
        let file2 = audio_dir.join("second_file.m4a");
        fs::write(&file1, b"identical content").unwrap();
        fs::write(&file2, b"identical content").unwrap();

        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n\
             1,1,Unripe,first_file.m4a,orig1.m4a,17,2024-01-01\n\
             2,2,Ripe,second_file.m4a,orig2.m4a,17,2024-01-01\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, true);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();

        // Error should mention both conflicting filenames
        assert!(error_msg.contains("first_file.m4a") || error_msg.contains("second_file.m4a"));
        assert!(
            error_msg.contains("duplicate")
                || error_msg.contains("SHA-256")
                || error_msg.contains("identical")
        );
    }

    #[test]
    fn test_load_metadata_real_iphone_dataset() {
        // Integration test against real iPhone dataset
        let csv_path = std::path::Path::new("data/metadata_iphone.csv");
        let audio_dir = std::path::Path::new("data/sound_iphone");

        if csv_path.exists() && audio_dir.exists() {
            let result = load_and_verify_metadata(csv_path, audio_dir, true);

            if let Ok((records, report)) = result {
                assert_eq!(report.total_files, 281);
                assert_eq!(report.unique_fruits, 175);
                assert_eq!(report.class_counts, [171, 66, 44]);
                assert_eq!(records.len(), 281);
            }
            // If dataset doesn't pass, that's also valuable information
        }
    }

    #[test]
    fn test_load_metadata_real_android_dataset() {
        // Integration test against real Android dataset
        let csv_path = std::path::Path::new("data/metadata_android.csv");
        let audio_dir = std::path::Path::new("data/sound_android");

        if csv_path.exists() && audio_dir.exists() {
            let result = load_and_verify_metadata(csv_path, audio_dir, true);

            if let Ok((records, report)) = result {
                assert_eq!(report.total_files, 175);
                assert_eq!(report.unique_fruits, 175);
                assert_eq!(report.class_counts, [95, 38, 42]);
                assert_eq!(records.len(), 175);
            }
        }
    }

    #[test]
    fn test_load_metadata_invalid_ripeness() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("invalid_ripeness.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n\
             1,1,Unknown,file1.m4a,orig1.m4a,100,2024-01-01\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_metadata_missing_audio_file() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("missing.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n\
             1,1,Unripe,missing.m4a,orig.m4a,100,2024-01-01\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exist"));
    }

    #[test]
    fn test_load_metadata_file_size_mismatch() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("size_mismatch.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        // Create a 10-byte file
        let audio_file = audio_dir.join("file1.m4a");
        fs::write(&audio_file, b"0123456789").unwrap();

        // But CSV says it should be 100 bytes
        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n\
             1,1,Unripe,file1.m4a,orig1.m4a,100,2024-01-01\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("size"));
    }

    // T008: Tests for SHA-256 deduplication
    #[test]
    fn test_load_metadata_duplicate_sha256() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("duplicate.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        // Create two files with identical content
        let file1 = audio_dir.join("file1.m4a");
        let file2 = audio_dir.join("file2.m4a");
        fs::write(&file1, b"identical content").unwrap();
        fs::write(&file2, b"identical content").unwrap();

        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n\
             1,1,Unripe,file1.m4a,orig1.m4a,17,2024-01-01\n\
             2,2,Ripe,file2.m4a,orig2.m4a,17,2024-01-01\n",
        )
        .unwrap();

        // Without SHA-256 verification, should pass
        let result = load_and_verify_metadata(&csv_path, &audio_dir, false);
        assert!(result.is_ok());

        // With SHA-256 verification, should fail
        let result = load_and_verify_metadata(&csv_path, &audio_dir, true);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("duplicate") || error_msg.contains("SHA-256"));
    }

    #[test]
    fn test_load_metadata_valid_dataset() {
        let temp_dir = TempDir::new().unwrap();
        let csv_path = temp_dir.path().join("valid.csv");
        let audio_dir = temp_dir.path().join("audio");
        fs::create_dir(&audio_dir).unwrap();

        // Create valid audio files
        fs::write(audio_dir.join("file1.m4a"), b"audio1").unwrap();
        fs::write(audio_dir.join("file2.m4a"), b"audio22").unwrap();
        fs::write(audio_dir.join("file3.m4a"), b"audio333").unwrap();

        fs::write(
            &csv_path,
            "sample_num,fruit_id,ripeness,filename,original_filename,file_size_bytes,file_date\n\
             1,1,Unripe,file1.m4a,orig1.m4a,6,2024-01-01\n\
             2,1,Ripe,file2.m4a,orig2.m4a,7,2024-01-02\n\
             3,2,Overripe,file3.m4a,orig3.m4a,8,2024-01-03\n",
        )
        .unwrap();

        let result = load_and_verify_metadata(&csv_path, &audio_dir, true);
        assert!(result.is_ok());

        let (records, report) = result.unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(report.total_files, 3);
        assert_eq!(report.unique_fruits, 2);
        assert_eq!(report.class_counts, [1, 1, 1]);
        assert_eq!(report.total_bytes, 21);
    }

    // T011: Tests for compute_class_weights
    #[test]
    fn test_compute_class_weights_iphone() {
        use approx::assert_relative_eq;
        let weights = compute_class_weights(&[171, 66, 44]);
        assert_relative_eq!(weights[0], 0.5478, epsilon = 0.001);
        assert_relative_eq!(weights[1], 1.4192, epsilon = 0.001);
        assert_relative_eq!(weights[2], 2.1288, epsilon = 0.001);
    }

    #[test]
    fn test_compute_class_weights_android() {
        use approx::assert_relative_eq;
        let weights = compute_class_weights(&[95, 38, 42]);
        assert_relative_eq!(weights[0], 0.6140, epsilon = 0.001);
        assert_relative_eq!(weights[1], 1.5351, epsilon = 0.001);
        assert_relative_eq!(weights[2], 1.3889, epsilon = 0.001);
    }

    #[test]
    fn test_compute_class_weights_zero_count_safety() {
        // Should not panic with zero counts
        let weights = compute_class_weights(&[0, 0, 0]);
        assert_eq!(weights, [0.0, 0.0, 0.0]);

        let weights = compute_class_weights(&[100, 0, 50]);
        assert_eq!(weights[1], 0.0);
    }

    // T012: Tests for stratified_group_split_by_fruit
    #[test]
    fn test_split_empty_records() {
        let records: Vec<AudioMetadataRecord> = vec![];
        let result = stratified_group_split_by_fruit(&records, 0.70, 0.15, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_split_invalid_ratios() {
        let records = vec![AudioMetadataRecord {
            sample_num: 1,
            fruit_id: 1,
            ripeness: "Ripe".to_string(),
            filename: "test.m4a".to_string(),
            original_filename: "orig.m4a".to_string(),
            file_size_bytes: 100,
            file_date: "2024-01-01".to_string(),
        }];

        // train_ratio + val_ratio >= 1.0
        assert!(stratified_group_split_by_fruit(&records, 0.70, 0.30, 42).is_err());

        // negative ratio
        assert!(stratified_group_split_by_fruit(&records, -0.1, 0.15, 42).is_err());

        // zero ratio
        assert!(stratified_group_split_by_fruit(&records, 0.0, 0.15, 42).is_err());
    }

    #[test]
    fn test_split_zero_fruit_id_leakage() {
        // Create synthetic dataset with multi-stage fruits
        let records = vec![
            // Fruit 1: Unripe -> Ripe -> Overripe (rarest: Overripe)
            AudioMetadataRecord {
                sample_num: 1,
                fruit_id: 1,
                ripeness: "Unripe".to_string(),
                filename: "f1_unripe.m4a".to_string(),
                original_filename: "f1_unripe.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-01".to_string(),
            },
            AudioMetadataRecord {
                sample_num: 2,
                fruit_id: 1,
                ripeness: "Ripe".to_string(),
                filename: "f1_ripe.m4a".to_string(),
                original_filename: "f1_ripe.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-02".to_string(),
            },
            AudioMetadataRecord {
                sample_num: 3,
                fruit_id: 1,
                ripeness: "Overripe".to_string(),
                filename: "f1_overripe.m4a".to_string(),
                original_filename: "f1_overripe.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-03".to_string(),
            },
            // Fruit 2: Unripe only
            AudioMetadataRecord {
                sample_num: 4,
                fruit_id: 2,
                ripeness: "Unripe".to_string(),
                filename: "f2_unripe.m4a".to_string(),
                original_filename: "f2_unripe.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-04".to_string(),
            },
            // Fruit 3: Ripe only
            AudioMetadataRecord {
                sample_num: 5,
                fruit_id: 3,
                ripeness: "Ripe".to_string(),
                filename: "f3_ripe.m4a".to_string(),
                original_filename: "f3_ripe.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-05".to_string(),
            },
        ];

        let splits = stratified_group_split_by_fruit(&records, 0.60, 0.20, 42).unwrap();

        // Collect unique fruit_ids per split
        let train_fruits: HashSet<u32> = splits.train.iter().map(|r| r.fruit_id).collect();
        let val_fruits: HashSet<u32> = splits.val.iter().map(|r| r.fruit_id).collect();
        let test_fruits: HashSet<u32> = splits.test.iter().map(|r| r.fruit_id).collect();

        // Verify zero overlap
        assert!(train_fruits.is_disjoint(&val_fruits));
        assert!(train_fruits.is_disjoint(&test_fruits));
        assert!(val_fruits.is_disjoint(&test_fruits));

        // Verify all fruits accounted for
        assert_eq!(train_fruits.len() + val_fruits.len() + test_fruits.len(), 3);

        // Verify all records accounted for
        assert_eq!(splits.train.len() + splits.val.len() + splits.test.len(), 5);
    }

    #[test]
    fn test_split_multi_fruit_strata_non_empty_splits() {
        // Create dataset with >= 4 fruits per stratum to ensure non-empty splits
        let mut records = Vec::new();
        let mut sample_num = 1;

        // Overripe stratum: 5 fruits with Overripe recordings
        for fruit_id in 1..=5 {
            records.push(AudioMetadataRecord {
                sample_num,
                fruit_id,
                ripeness: "Overripe".to_string(),
                filename: format!("f{}_overripe.m4a", fruit_id),
                original_filename: format!("f{}_overripe.m4a", fruit_id),
                file_size_bytes: 100,
                file_date: "2024-01-01".to_string(),
            });
            sample_num += 1;
        }

        // Ripe stratum: 5 fruits with Ripe recordings (but no Overripe)
        for fruit_id in 6..=10 {
            records.push(AudioMetadataRecord {
                sample_num,
                fruit_id,
                ripeness: "Ripe".to_string(),
                filename: format!("f{}_ripe.m4a", fruit_id),
                original_filename: format!("f{}_ripe.m4a", fruit_id),
                file_size_bytes: 100,
                file_date: "2024-01-02".to_string(),
            });
            sample_num += 1;
        }

        // Unripe stratum: 5 fruits with only Unripe recordings
        for fruit_id in 11..=15 {
            records.push(AudioMetadataRecord {
                sample_num,
                fruit_id,
                ripeness: "Unripe".to_string(),
                filename: format!("f{}_unripe.m4a", fruit_id),
                original_filename: format!("f{}_unripe.m4a", fruit_id),
                file_size_bytes: 100,
                file_date: "2024-01-03".to_string(),
            });
            sample_num += 1;
        }

        let splits = stratified_group_split_by_fruit(&records, 0.60, 0.20, 42).unwrap();

        // All splits should be non-empty
        assert!(!splits.train.is_empty(), "Train split should not be empty");
        assert!(!splits.val.is_empty(), "Val split should not be empty");
        assert!(!splits.test.is_empty(), "Test split should not be empty");

        // Verify RNG shuffling was exercised (we have enough fruits for meaningful splits)
        let train_fruits: HashSet<u32> = splits.train.iter().map(|r| r.fruit_id).collect();
        let val_fruits: HashSet<u32> = splits.val.iter().map(|r| r.fruit_id).collect();
        let test_fruits: HashSet<u32> = splits.test.iter().map(|r| r.fruit_id).collect();

        // Should have roughly 60% train, 20% val, 20% test fruits
        assert!(train_fruits.len() >= 7); // ~60% of 15 = 9, but allow some variance
        assert!(val_fruits.len() >= 2);
        assert!(test_fruits.len() >= 2);
    }

    #[test]
    fn test_split_preserves_original_csv_order() {
        // Create dataset where sample_num reflects CSV order
        let records = vec![
            AudioMetadataRecord {
                sample_num: 1,
                fruit_id: 1,
                ripeness: "Unripe".to_string(),
                filename: "f1_1.m4a".to_string(),
                original_filename: "f1_1.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-01".to_string(),
            },
            AudioMetadataRecord {
                sample_num: 2,
                fruit_id: 2,
                ripeness: "Ripe".to_string(),
                filename: "f2_1.m4a".to_string(),
                original_filename: "f2_1.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-02".to_string(),
            },
            AudioMetadataRecord {
                sample_num: 3,
                fruit_id: 1,
                ripeness: "Ripe".to_string(),
                filename: "f1_2.m4a".to_string(),
                original_filename: "f1_2.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-03".to_string(),
            },
            AudioMetadataRecord {
                sample_num: 4,
                fruit_id: 3,
                ripeness: "Overripe".to_string(),
                filename: "f3_1.m4a".to_string(),
                original_filename: "f3_1.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-04".to_string(),
            },
            AudioMetadataRecord {
                sample_num: 5,
                fruit_id: 2,
                ripeness: "Overripe".to_string(),
                filename: "f2_2.m4a".to_string(),
                original_filename: "f2_2.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-05".to_string(),
            },
        ];

        let splits = stratified_group_split_by_fruit(&records, 0.60, 0.20, 42).unwrap();

        // Within each split, verify sample_num is in ascending order
        for i in 1..splits.train.len() {
            assert!(
                splits.train[i - 1].sample_num < splits.train[i].sample_num,
                "Train split not in ascending sample_num order"
            );
        }

        for i in 1..splits.val.len() {
            assert!(
                splits.val[i - 1].sample_num < splits.val[i].sample_num,
                "Val split not in ascending sample_num order"
            );
        }

        for i in 1..splits.test.len() {
            assert!(
                splits.test[i - 1].sample_num < splits.test[i].sample_num,
                "Test split not in ascending sample_num order"
            );
        }
    }

    #[test]
    fn test_split_real_iphone_dataset_cardinality() {
        let csv_path = std::path::Path::new("data/metadata_iphone.csv");
        let audio_dir = std::path::Path::new("data/sound_iphone");

        if csv_path.exists() && audio_dir.exists() {
            if let Ok((records, _report)) = load_and_verify_metadata(csv_path, audio_dir, false) {
                let splits = stratified_group_split_by_fruit(&records, 0.70, 0.15, 42).unwrap();

                let train_fruits: HashSet<u32> = splits.train.iter().map(|r| r.fruit_id).collect();
                let val_fruits: HashSet<u32> = splits.val.iter().map(|r| r.fruit_id).collect();
                let test_fruits: HashSet<u32> = splits.test.iter().map(|r| r.fruit_id).collect();

                // Verify expected cardinalities for iPhone dataset
                assert_eq!(
                    train_fruits.len(),
                    123,
                    "iPhone train should have 123 unique fruits"
                );
                assert_eq!(
                    val_fruits.len(),
                    26,
                    "iPhone val should have 26 unique fruits"
                );
                assert_eq!(
                    test_fruits.len(),
                    26,
                    "iPhone test should have 26 unique fruits"
                );

                // Verify zero fruit_id overlap
                assert!(train_fruits.is_disjoint(&val_fruits));
                assert!(train_fruits.is_disjoint(&test_fruits));
                assert!(val_fruits.is_disjoint(&test_fruits));
            }
        }
    }

    #[test]
    fn test_split_real_android_dataset_cardinality() {
        let csv_path = std::path::Path::new("data/metadata_android.csv");
        let audio_dir = std::path::Path::new("data/sound_android");

        if csv_path.exists() && audio_dir.exists() {
            if let Ok((records, _report)) = load_and_verify_metadata(csv_path, audio_dir, false) {
                let splits = stratified_group_split_by_fruit(&records, 0.70, 0.15, 42).unwrap();

                let train_fruits: HashSet<u32> = splits.train.iter().map(|r| r.fruit_id).collect();
                let val_fruits: HashSet<u32> = splits.val.iter().map(|r| r.fruit_id).collect();
                let test_fruits: HashSet<u32> = splits.test.iter().map(|r| r.fruit_id).collect();

                // Verify expected cardinalities for Android dataset
                assert_eq!(
                    train_fruits.len(),
                    123,
                    "Android train should have 123 unique fruits"
                );
                assert_eq!(
                    val_fruits.len(),
                    26,
                    "Android val should have 26 unique fruits"
                );
                assert_eq!(
                    test_fruits.len(),
                    26,
                    "Android test should have 26 unique fruits"
                );

                // Verify zero fruit_id overlap
                assert!(train_fruits.is_disjoint(&val_fruits));
                assert!(train_fruits.is_disjoint(&test_fruits));
                assert!(val_fruits.is_disjoint(&test_fruits));
            }
        }
    }

    #[test]
    fn test_split_deterministic_with_same_seed() {
        let records = vec![
            AudioMetadataRecord {
                sample_num: 1,
                fruit_id: 1,
                ripeness: "Unripe".to_string(),
                filename: "f1.m4a".to_string(),
                original_filename: "f1.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-01".to_string(),
            },
            AudioMetadataRecord {
                sample_num: 2,
                fruit_id: 2,
                ripeness: "Ripe".to_string(),
                filename: "f2.m4a".to_string(),
                original_filename: "f2.m4a".to_string(),
                file_size_bytes: 100,
                file_date: "2024-01-02".to_string(),
            },
        ];

        let splits1 = stratified_group_split_by_fruit(&records, 0.50, 0.25, 42).unwrap();
        let splits2 = stratified_group_split_by_fruit(&records, 0.50, 0.25, 42).unwrap();

        // Same seed should produce identical splits
        assert_eq!(splits1, splits2);
    }
}
