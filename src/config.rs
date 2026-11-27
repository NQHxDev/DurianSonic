//! Configuration management for DurianSonic
//!
//! This module provides centralized configuration management across 4 domains:
//! - DSP (Digital Signal Processing): Audio processing parameters
//! - Dataset: Data paths and split configuration
//! - Training: Model training hyperparameters
//! - Model: Neural network architecture specification
//!
//! All configurations support TOML serialization with backward compatibility
//! and comprehensive validation of physical and mathematical constraints.
//!
//! # Examples
//!
//! ```no_run
//! use durian_sonic::config::Config;
//!
//! // Load from default path
//! let config = Config::load_default().unwrap();
//!
//! // Load from custom path
//! let config = Config::from_file("configs/custom.toml").unwrap();
//!
//! // Validation is automatic on load
//! assert!(config.validate().is_ok());
//! ```

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Configuration error types
///
/// Represents all possible errors during configuration loading, parsing, and validation.
///
/// # Examples
///
/// ```no_run
/// use durian_sonic::config::{Config, ConfigError};
///
/// match Config::from_file("config.toml") {
///     Ok(config) => { /* Use config */ },
///     Err(ConfigError::IoError(_)) => { /* Handle file not found */ },
///     Err(ConfigError::ParseError(_)) => { /* Handle invalid TOML */ },
///     Err(ConfigError::ValidationError(_)) => { /* Handle invalid values */ },
/// }
/// ```
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse config file: {0}")]
    ParseError(#[from] toml::de::Error),

    #[error("Invalid configuration: {0}")]
    ValidationError(String),
}

/// Main configuration structure combining all 4 configuration domains
///
/// Provides centralized management of DSP, dataset, training, and model parameters
/// with automatic validation and backward compatibility for legacy configurations.
///
/// # Examples
///
/// ```no_run
/// use durian_sonic::config::Config;
///
/// // Load default configuration
/// let config = Config::load_default().unwrap();
/// assert_eq!(config.dsp.sample_rate, 22_050);
///
/// // Create custom configuration
/// let mut config = Config::default();
/// config.training.epochs = 50;
/// config.save_to_file("configs/custom.toml").unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub dsp: DspConfig,
    #[serde(default)]
    pub dataset: DatasetConfig,
    pub training: TrainingConfig,
    pub model: ModelConfig,
}

/// Digital Signal Processing configuration
///
/// Defines parameters for audio feature extraction including sample rate, windowing,
/// mel-spectrogram generation, and onset detection. All parameters are validated
/// against physical constraints (Nyquist limit, power-of-two FFT, etc.).
///
/// # Examples
///
/// ```
/// use durian_sonic::config::DspConfig;
///
/// let dsp = DspConfig::default();
/// assert_eq!(dsp.sample_rate, 22_050);
/// assert_eq!(dsp.n_mels, 128);
/// assert!(dsp.n_fft.is_power_of_two());
///
/// // fmax must not exceed Nyquist frequency (sample_rate / 2)
/// assert!(dsp.fmax <= (dsp.sample_rate as f32) / 2.0);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DspConfig {
    pub sample_rate: u32,
    pub window_sec: f32,
    pub pre_onset_sec: f32,
    pub min_interval_sec: f32,
    pub threshold_ratio: f32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub top_db: f32,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            sample_rate: 22_050,
            window_sec: 0.8,
            pre_onset_sec: 0.1,
            min_interval_sec: 0.4,
            threshold_ratio: 0.25,
            n_fft: 1024,
            hop_length: 128,
            n_mels: 128,
            fmin: 50.0,
            fmax: 8_000.0,
            top_db: 80.0,
        }
    }
}

/// Dataset paths and split configuration
///
/// Specifies data directory locations and train/val/test split ratios.
/// All split ratios must sum to 1.0 within tolerance (< 1e-4).
///
/// # Examples
///
/// ```
/// use durian_sonic::config::DatasetConfig;
///
/// let dataset = DatasetConfig::default();
/// assert_eq!(dataset.data_dir, "data");
/// assert_eq!(dataset.train_ratio + dataset.val_ratio + dataset.test_ratio, 1.0);
/// assert_eq!(dataset.split_seed, 42); // Deterministic splits
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DatasetConfig {
    pub data_dir: String,
    pub iphone_metadata: String,
    pub android_metadata: String,
    pub train_ratio: f32,
    pub val_ratio: f32,
    pub test_ratio: f32,
    pub split_seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            data_dir: "data".to_string(),
            iphone_metadata: "data/metadata_iphone.csv".to_string(),
            android_metadata: "data/metadata_android.csv".to_string(),
            train_ratio: 0.70,
            val_ratio: 0.15,
            test_ratio: 0.15,
            split_seed: 42,
        }
    }
}

/// Training hyperparameters configuration
///
/// Specifies model training parameters including epochs, batch size, learning rate,
/// regularization, and early stopping criteria.
///
/// # Examples
///
/// ```
/// use durian_sonic::config::TrainingConfig;
///
/// let training = TrainingConfig {
///     epochs: 100,
///     batch_size: 32,
///     learning_rate: 0.001,
///     weight_decay: 0.0001,
///     label_smoothing: 0.05,
///     early_stopping_patience: 12,
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "default_label_smoothing")]
    pub label_smoothing: f32,
    #[serde(default = "default_early_stopping_patience")]
    pub early_stopping_patience: usize,
}

fn default_weight_decay() -> f64 {
    0.0001
}

fn default_label_smoothing() -> f32 {
    0.05
}

fn default_early_stopping_patience() -> usize {
    12
}

/// Model architecture configuration
///
/// Specifies neural network architecture including input dimensions, hidden layers,
/// activation functions, and regularization.
///
/// # Examples
///
/// ```
/// use durian_sonic::config::ModelConfig;
///
/// let model = ModelConfig {
///     hidden_layers: vec![128, 64],
///     activation: "relu".to_string(),
///     architecture: "durian_audio_cnn_small".to_string(),
///     input_channels: 1,
///     input_height: 128,
///     input_width: 128,
///     physics_features_dim: 4,
///     num_classes: 3,
///     dropout: 0.3,
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_layers: Vec<usize>,
    pub activation: String,
    #[serde(default = "default_architecture")]
    pub architecture: String,
    #[serde(default = "default_input_channels")]
    pub input_channels: usize,
    #[serde(default = "default_input_height")]
    pub input_height: usize,
    #[serde(default = "default_input_width")]
    pub input_width: usize,
    #[serde(default = "default_physics_features_dim")]
    pub physics_features_dim: usize,
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
}

fn default_architecture() -> String {
    "durian_audio_cnn_small".to_string()
}

fn default_input_channels() -> usize {
    1
}

fn default_input_height() -> usize {
    128
}

fn default_input_width() -> usize {
    128
}

fn default_physics_features_dim() -> usize {
    4
}

fn default_num_classes() -> usize {
    3
}

fn default_dropout() -> f64 {
    0.3
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![],
            activation: "relu".to_string(),
            architecture: default_architecture(),
            input_channels: default_input_channels(),
            input_height: default_input_height(),
            input_width: default_input_width(),
            physics_features_dim: default_physics_features_dim(),
            num_classes: default_num_classes(),
            dropout: default_dropout(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dsp: DspConfig::default(),
            dataset: DatasetConfig::default(),
            training: TrainingConfig {
                epochs: 100,
                batch_size: 32,
                learning_rate: 0.001,
                weight_decay: 0.0001,
                label_smoothing: 0.05,
                early_stopping_patience: 12,
            },
            model: ModelConfig {
                hidden_layers: vec![128, 64],
                activation: "relu".to_string(),
                architecture: "durian_audio_cnn_small".to_string(),
                input_channels: 1,
                input_height: 128,
                input_width: 128,
                physics_features_dim: 4,
                num_classes: 3,
                dropout: 0.3,
            },
        }
    }
}

impl Config {
    /// Load configuration from a TOML file with automatic validation
    ///
    /// Reads and parses a TOML configuration file, then validates all parameters
    /// against physical and mathematical constraints. Returns an error if the file
    /// cannot be read, parsed, or if validation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use durian_sonic::config::Config;
    ///
    /// let config = Config::from_file("configs/default.toml").unwrap();
    /// assert!(config.dsp.sample_rate > 0);
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load default configuration from configs/default.toml
    ///
    /// Convenience method for loading the default configuration file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use durian_sonic::config::Config;
    ///
    /// let config = Config::load_default().unwrap();
    /// ```
    pub fn load_default() -> Result<Self, ConfigError> {
        Self::from_file("configs/default.toml")
    }

    /// Save configuration to a TOML file
    ///
    /// Serializes the configuration to pretty-printed TOML format and writes
    /// it to the specified file path.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use durian_sonic::config::Config;
    ///
    /// let config = Config::default();
    /// config.save_to_file("configs/custom.toml").unwrap();
    /// ```
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| ConfigError::ValidationError(e.to_string()))?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Validate all configuration parameters
    ///
    /// Checks all DSP, dataset, training, and model parameters against their
    /// respective constraints. Returns ValidationError if any parameter is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use durian_sonic::config::Config;
    ///
    /// let mut config = Config::default();
    /// assert!(config.validate().is_ok());
    ///
    /// // Invalid configuration
    /// config.dsp.n_fft = 1000; // Not a power of two
    /// assert!(config.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.validate_dsp()?;
        self.validate_dataset()?;
        self.validate_training()?;
        self.validate_model()?;
        Ok(())
    }

    fn validate_dsp(&self) -> Result<(), ConfigError> {
        // Check finiteness for all f32 fields
        let floats = [
            ("window_sec", self.dsp.window_sec),
            ("pre_onset_sec", self.dsp.pre_onset_sec),
            ("min_interval_sec", self.dsp.min_interval_sec),
            ("threshold_ratio", self.dsp.threshold_ratio),
            ("fmin", self.dsp.fmin),
            ("fmax", self.dsp.fmax),
            ("top_db", self.dsp.top_db),
        ];

        for (name, value) in &floats {
            if !value.is_finite() {
                return Err(ConfigError::ValidationError(format!(
                    "{} must be finite (not NaN or infinity)",
                    name
                )));
            }
        }

        if self.dsp.sample_rate == 0 {
            return Err(ConfigError::ValidationError(
                "sample_rate must be > 0".to_string(),
            ));
        }

        if self.dsp.window_sec <= 0.0 {
            return Err(ConfigError::ValidationError(
                "window_sec must be > 0.0".to_string(),
            ));
        }

        if self.dsp.pre_onset_sec < 0.0 || self.dsp.pre_onset_sec >= self.dsp.window_sec {
            return Err(ConfigError::ValidationError(
                "pre_onset_sec must be >= 0.0 and < window_sec".to_string(),
            ));
        }

        if self.dsp.min_interval_sec <= 0.0 {
            return Err(ConfigError::ValidationError(
                "min_interval_sec must be > 0.0".to_string(),
            ));
        }

        if self.dsp.threshold_ratio <= 0.0 || self.dsp.threshold_ratio > 1.0 {
            return Err(ConfigError::ValidationError(
                "threshold_ratio must be > 0.0 and <= 1.0".to_string(),
            ));
        }

        if self.dsp.n_fft == 0 || !self.dsp.n_fft.is_power_of_two() {
            return Err(ConfigError::ValidationError(
                "n_fft must be > 0 and a power of two".to_string(),
            ));
        }

        if self.dsp.hop_length == 0 {
            return Err(ConfigError::ValidationError(
                "hop_length must be > 0".to_string(),
            ));
        }

        if self.dsp.n_mels == 0 {
            return Err(ConfigError::ValidationError(
                "n_mels must be > 0".to_string(),
            ));
        }

        if self.dsp.fmin < 0.0 || self.dsp.fmin >= self.dsp.fmax {
            return Err(ConfigError::ValidationError(
                "fmin must be >= 0.0 and < fmax".to_string(),
            ));
        }

        let nyquist = (self.dsp.sample_rate as f32) / 2.0;
        if self.dsp.fmax <= self.dsp.fmin || self.dsp.fmax > nyquist {
            return Err(ConfigError::ValidationError(format!(
                "fmax must be > fmin and <= sample_rate/2 (Nyquist limit: {})",
                nyquist
            )));
        }

        if self.dsp.top_db <= 0.0 {
            return Err(ConfigError::ValidationError(
                "top_db must be > 0.0".to_string(),
            ));
        }

        Ok(())
    }

    fn validate_dataset(&self) -> Result<(), ConfigError> {
        // Check finiteness
        if !self.dataset.train_ratio.is_finite()
            || !self.dataset.val_ratio.is_finite()
            || !self.dataset.test_ratio.is_finite()
        {
            return Err(ConfigError::ValidationError(
                "Split ratios must be finite".to_string(),
            ));
        }

        if self.dataset.data_dir.trim().is_empty() {
            return Err(ConfigError::ValidationError(
                "data_dir must be a non-empty string".to_string(),
            ));
        }

        if self.dataset.iphone_metadata.trim().is_empty() {
            return Err(ConfigError::ValidationError(
                "iphone_metadata must be a non-empty string".to_string(),
            ));
        }

        if self.dataset.android_metadata.trim().is_empty() {
            return Err(ConfigError::ValidationError(
                "android_metadata must be a non-empty string".to_string(),
            ));
        }

        if self.dataset.train_ratio <= 0.0 || self.dataset.train_ratio >= 1.0 {
            return Err(ConfigError::ValidationError(
                "train_ratio must be > 0.0 and < 1.0".to_string(),
            ));
        }

        if self.dataset.val_ratio <= 0.0 || self.dataset.val_ratio >= 1.0 {
            return Err(ConfigError::ValidationError(
                "val_ratio must be > 0.0 and < 1.0".to_string(),
            ));
        }

        if self.dataset.test_ratio <= 0.0 || self.dataset.test_ratio >= 1.0 {
            return Err(ConfigError::ValidationError(
                "test_ratio must be > 0.0 and < 1.0".to_string(),
            ));
        }

        let sum = self.dataset.train_ratio + self.dataset.val_ratio + self.dataset.test_ratio;
        if (sum - 1.0).abs() >= 1e-4 {
            return Err(ConfigError::ValidationError(format!(
                "train_ratio + val_ratio + test_ratio must sum to 1.0 (got {}, difference: {})",
                sum,
                (sum - 1.0).abs()
            )));
        }

        Ok(())
    }

    fn validate_training(&self) -> Result<(), ConfigError> {
        // Check finiteness
        if !self.training.learning_rate.is_finite() || !self.training.weight_decay.is_finite() {
            return Err(ConfigError::ValidationError(
                "Training float parameters must be finite".to_string(),
            ));
        }

        if !self.training.label_smoothing.is_finite() {
            return Err(ConfigError::ValidationError(
                "label_smoothing must be finite".to_string(),
            ));
        }

        if self.training.epochs == 0 {
            return Err(ConfigError::ValidationError(
                "epochs must be > 0".to_string(),
            ));
        }

        if self.training.batch_size == 0 {
            return Err(ConfigError::ValidationError(
                "batch_size must be > 0".to_string(),
            ));
        }

        if self.training.learning_rate <= 0.0 || self.training.learning_rate >= 1.0 {
            return Err(ConfigError::ValidationError(
                "learning_rate must be > 0.0 and < 1.0".to_string(),
            ));
        }

        if self.training.weight_decay < 0.0 {
            return Err(ConfigError::ValidationError(
                "weight_decay must be >= 0.0".to_string(),
            ));
        }

        if self.training.label_smoothing < 0.0 || self.training.label_smoothing >= 1.0 {
            return Err(ConfigError::ValidationError(
                "label_smoothing must be >= 0.0 and < 1.0".to_string(),
            ));
        }

        if self.training.early_stopping_patience == 0 {
            return Err(ConfigError::ValidationError(
                "early_stopping_patience must be > 0".to_string(),
            ));
        }

        Ok(())
    }

    fn validate_model(&self) -> Result<(), ConfigError> {
        // Check finiteness
        if !self.model.dropout.is_finite() {
            return Err(ConfigError::ValidationError(
                "dropout must be finite".to_string(),
            ));
        }

        if self.model.architecture.trim().is_empty() {
            return Err(ConfigError::ValidationError(
                "architecture must be a non-empty string".to_string(),
            ));
        }

        if self.model.input_channels == 0 {
            return Err(ConfigError::ValidationError(
                "input_channels must be > 0".to_string(),
            ));
        }

        if self.model.input_height == 0 {
            return Err(ConfigError::ValidationError(
                "input_height must be > 0".to_string(),
            ));
        }

        if self.model.input_width == 0 {
            return Err(ConfigError::ValidationError(
                "input_width must be > 0".to_string(),
            ));
        }

        if self.model.physics_features_dim == 0 {
            return Err(ConfigError::ValidationError(
                "physics_features_dim must be > 0".to_string(),
            ));
        }

        if self.model.num_classes < 2 {
            return Err(ConfigError::ValidationError(
                "num_classes must be >= 2".to_string(),
            ));
        }

        if self.model.dropout < 0.0 || self.model.dropout >= 1.0 {
            return Err(ConfigError::ValidationError(
                "dropout must be >= 0.0 and < 1.0".to_string(),
            ));
        }

        if self.model.hidden_layers.is_empty() {
            return Err(ConfigError::ValidationError(
                "hidden_layers cannot be empty".to_string(),
            ));
        }

        for &layer_size in &self.model.hidden_layers {
            if layer_size == 0 {
                return Err(ConfigError::ValidationError(
                    "hidden layer size must be > 0".to_string(),
                ));
            }
        }

        let valid_activations = ["relu", "sigmoid", "tanh", "leaky_relu"];
        if !valid_activations.contains(&self.model.activation.as_str()) {
            return Err(ConfigError::ValidationError(format!(
                "activation must be one of: {:?}",
                valid_activations
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.training.epochs, 100);
        assert_eq!(config.training.batch_size, 32);
        assert_eq!(config.training.learning_rate, 0.001);
        assert_eq!(config.model.hidden_layers, vec![128, 64]);
        assert_eq!(config.model.activation, "relu");
    }

    #[test]
    fn test_config_validation_valid() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_zero_epochs() {
        let mut config = Config::default();
        config.training.epochs = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_learning_rate() {
        let mut config = Config::default();
        config.training.learning_rate = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_empty_hidden_layers() {
        let mut config = Config::default();
        config.model.hidden_layers = vec![];
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_activation() {
        let mut config = Config::default();
        config.model.activation = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_load_from_file_legacy_2section() {
        let toml_content = r#"
            [training]
            epochs = 50
            batch_size = 16
            learning_rate = 0.01

            [model]
            hidden_layers = [256, 128]
            activation = "sigmoid"
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();

        let config = Config::from_file(temp_file.path()).unwrap();
        assert_eq!(config.training.epochs, 50);
        assert_eq!(config.training.batch_size, 16);
        assert_eq!(config.training.learning_rate, 0.01);
        assert_eq!(config.model.hidden_layers, vec![256, 128]);
        assert_eq!(config.model.activation, "sigmoid");

        // Check defaults populated for DSP and Dataset
        assert_eq!(config.dsp.sample_rate, 22_050);
        assert_eq!(config.dataset.train_ratio, 0.70);

        // Check that extended fields have defaults
        assert_eq!(config.training.weight_decay, 0.0001);
        assert_eq!(config.training.label_smoothing, 0.05);
        assert_eq!(config.training.early_stopping_patience, 12);
        assert_eq!(config.model.architecture, "durian_audio_cnn_small");
        assert_eq!(config.model.num_classes, 3);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let original_config = Config::default();
        let temp_file = NamedTempFile::new().unwrap();

        original_config.save_to_file(temp_file.path()).unwrap();
        let loaded_config = Config::from_file(temp_file.path()).unwrap();

        // Full struct equality check
        assert_eq!(original_config, loaded_config);
    }

    #[test]
    fn test_load_default_config() {
        // This test requires the actual configs/default.toml file to exist
        // Skip if file doesn't exist
        if std::path::Path::new("configs/default.toml").exists() {
            let config = Config::load_default().unwrap();

            // Verify specific values from default.toml
            assert_eq!(config.training.epochs, 80);
            assert_eq!(config.training.learning_rate, 0.0003);
            assert_eq!(config.dsp.sample_rate, 22_050);
            assert_eq!(config.dataset.split_seed, 42);
        }
    }

    #[test]
    fn test_forward_compatible_unknown_keys() {
        // TOML with extra unknown keys should be ignored, not cause errors
        let toml_content = r#"
            [dsp]
            sample_rate = 22_050
            window_sec = 0.8
            pre_onset_sec = 0.1
            min_interval_sec = 0.4
            threshold_ratio = 0.25
            n_fft = 1024
            hop_length = 128
            n_mels = 128
            fmin = 50.0
            fmax = 8_000.0
            top_db = 80.0
            unknown_future_field = "this should be ignored"

            [dataset]
            data_dir = "data"
            iphone_metadata = "data/metadata_iphone.csv"
            android_metadata = "data/metadata_android.csv"
            train_ratio = 0.70
            val_ratio = 0.15
            test_ratio = 0.15
            split_seed = 42
            future_feature = 123

            [training]
            epochs = 100
            batch_size = 32
            learning_rate = 0.001

            [model]
            hidden_layers = [128, 64]
            activation = "relu"
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();

        // Should succeed despite unknown keys
        let result = Config::from_file(temp_file.path());
        assert!(
            result.is_ok(),
            "Config with unknown keys should be accepted for forward compatibility"
        );
    }

    #[test]
    fn test_from_file_validates_automatically() {
        // Create invalid config (n_fft not power of two)
        let toml_content = r#"
            [dsp]
            sample_rate = 22_050
            window_sec = 0.8
            pre_onset_sec = 0.1
            min_interval_sec = 0.4
            threshold_ratio = 0.25
            n_fft = 1000
            hop_length = 128
            n_mels = 128
            fmin = 50.0
            fmax = 8_000.0
            top_db = 80.0

            [dataset]
            data_dir = "data"
            iphone_metadata = "data/metadata_iphone.csv"
            android_metadata = "data/metadata_android.csv"
            train_ratio = 0.70
            val_ratio = 0.15
            test_ratio = 0.15
            split_seed = 42

            [training]
            epochs = 100
            batch_size = 32
            learning_rate = 0.001

            [model]
            hidden_layers = [128, 64]
            activation = "relu"
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();

        // from_file should reject due to automatic validation
        let result = Config::from_file(temp_file.path());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ConfigError::ValidationError(_)
        ));
    }

    #[test]
    fn test_dsp_validation_nyquist_limit() {
        let mut config = Config::default();
        config.dsp.fmax = 15000.0; // Exceeds Nyquist for 22050 Hz
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dsp_validation_power_of_two() {
        let mut config = Config::default();
        config.dsp.n_fft = 1000; // Not a power of two
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dsp_validation_float_finiteness() {
        let mut config = Config::default();

        // Test NaN
        config.dsp.window_sec = f32::NAN;
        assert!(config.validate().is_err());
        config.dsp.window_sec = 0.8;

        // Test INFINITY
        config.dsp.fmax = f32::INFINITY;
        assert!(config.validate().is_err());
        config.dsp.fmax = 8_000.0;

        // Test -INFINITY
        config.dsp.fmin = f32::NEG_INFINITY;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dsp_validation_sample_rate_zero() {
        let mut config = Config::default();
        config.dsp.sample_rate = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dsp_validation_window_sec_bounds() {
        let mut config = Config::default();
        config.dsp.window_sec = 0.0;
        assert!(config.validate().is_err());

        config.dsp.window_sec = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dsp_validation_pre_onset_sec_bounds() {
        let mut config = Config::default();

        // pre_onset_sec must be < window_sec
        config.dsp.pre_onset_sec = config.dsp.window_sec;
        assert!(config.validate().is_err());

        config.dsp.pre_onset_sec = config.dsp.window_sec + 0.1;
        assert!(config.validate().is_err());

        // pre_onset_sec must be >= 0.0
        config.dsp.pre_onset_sec = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dsp_validation_min_interval_sec_bounds() {
        let mut config = Config::default();
        config.dsp.min_interval_sec = 0.0;
        assert!(config.validate().is_err());

        config.dsp.min_interval_sec = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dsp_validation_threshold_ratio_bounds() {
        let mut config = Config::default();

        config.dsp.threshold_ratio = 0.0;
        assert!(config.validate().is_err());

        config.dsp.threshold_ratio = 1.1;
        assert!(config.validate().is_err());

        config.dsp.threshold_ratio = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dsp_validation_hop_length_zero() {
        let mut config = Config::default();
        config.dsp.hop_length = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dsp_validation_n_mels_zero() {
        let mut config = Config::default();
        config.dsp.n_mels = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dsp_validation_fmin_bounds() {
        let mut config = Config::default();

        // fmin must be >= 0.0
        config.dsp.fmin = -1.0;
        assert!(config.validate().is_err());

        // fmin must be < fmax
        config.dsp.fmin = config.dsp.fmax;
        assert!(config.validate().is_err());

        config.dsp.fmin = config.dsp.fmax + 100.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dsp_validation_top_db_bounds() {
        let mut config = Config::default();
        config.dsp.top_db = 0.0;
        assert!(config.validate().is_err());

        config.dsp.top_db = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dataset_validation_whitespace_only_strings() {
        let mut config = Config::default();

        // data_dir whitespace-only
        config.dataset.data_dir = "   ".to_string();
        assert!(config.validate().is_err());
        config.dataset.data_dir = "data".to_string();

        // iphone_metadata whitespace-only
        config.dataset.iphone_metadata = "\t\n".to_string();
        assert!(config.validate().is_err());
        config.dataset.iphone_metadata = "data/metadata_iphone.csv".to_string();

        // android_metadata whitespace-only
        config.dataset.android_metadata = "".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dataset_validation_split_ratio_finiteness() {
        let mut config = Config::default();

        config.dataset.train_ratio = f32::NAN;
        assert!(config.validate().is_err());
        config.dataset.train_ratio = 0.70;

        config.dataset.val_ratio = f32::INFINITY;
        assert!(config.validate().is_err());
        config.dataset.val_ratio = 0.15;

        config.dataset.test_ratio = f32::NEG_INFINITY;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dataset_validation_individual_ratio_bounds() {
        let mut config = Config::default();

        // train_ratio bounds
        config.dataset.train_ratio = 0.0;
        assert!(config.validate().is_err());
        config.dataset.train_ratio = 1.0;
        assert!(config.validate().is_err());
        config.dataset.train_ratio = 0.70;

        // val_ratio bounds
        config.dataset.val_ratio = -0.1;
        assert!(config.validate().is_err());
        config.dataset.val_ratio = 0.99;
        config.dataset.test_ratio = 0.02; // Fix sum first
        assert!(config.validate().is_err());
        config.dataset.val_ratio = 0.15;
        config.dataset.test_ratio = 0.15;

        // test_ratio bounds
        config.dataset.test_ratio = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_training_validation_finiteness() {
        let mut config = Config::default();

        config.training.learning_rate = f64::NAN;
        assert!(config.validate().is_err());
        config.training.learning_rate = 0.001;

        config.training.weight_decay = f64::INFINITY;
        assert!(config.validate().is_err());
        config.training.weight_decay = 0.0001;

        config.training.label_smoothing = f32::NEG_INFINITY;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_training_validation_batch_size_zero() {
        let mut config = Config::default();
        config.training.batch_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_training_validation_weight_decay_negative() {
        let mut config = Config::default();
        config.training.weight_decay = -0.001;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_training_validation_label_smoothing_bounds() {
        let mut config = Config::default();

        config.training.label_smoothing = -0.1;
        assert!(config.validate().is_err());

        config.training.label_smoothing = 1.0;
        assert!(config.validate().is_err());

        config.training.label_smoothing = 1.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_training_validation_early_stopping_patience_zero() {
        let mut config = Config::default();
        config.training.early_stopping_patience = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_validation_finiteness() {
        let mut config = Config::default();

        config.model.dropout = f64::NAN;
        assert!(config.validate().is_err());
        config.model.dropout = 0.3;

        config.model.dropout = f64::INFINITY;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_validation_whitespace_only_architecture() {
        let mut config = Config::default();
        config.model.architecture = "   ".to_string();
        assert!(config.validate().is_err());

        config.model.architecture = "".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_validation_zero_dimensions() {
        let mut config = Config::default();

        config.model.input_channels = 0;
        assert!(config.validate().is_err());
        config.model.input_channels = 1;

        config.model.input_height = 0;
        assert!(config.validate().is_err());
        config.model.input_height = 128;

        config.model.input_width = 0;
        assert!(config.validate().is_err());
        config.model.input_width = 128;

        config.model.physics_features_dim = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_validation_num_classes_less_than_two() {
        let mut config = Config::default();

        config.model.num_classes = 1;
        assert!(config.validate().is_err());

        config.model.num_classes = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_validation_dropout_bounds() {
        let mut config = Config::default();

        config.model.dropout = -0.1;
        assert!(config.validate().is_err());

        config.model.dropout = 1.0;
        assert!(config.validate().is_err());

        config.model.dropout = 1.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_validation_zero_in_hidden_layers() {
        let mut config = Config::default();
        config.model.hidden_layers = vec![128, 0, 64];
        assert!(config.validate().is_err());

        config.model.hidden_layers = vec![0];
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_validation_uppercase_activation() {
        let mut config = Config::default();

        config.model.activation = "ReLU".to_string();
        assert!(config.validate().is_err());

        config.model.activation = "RELU".to_string();
        assert!(config.validate().is_err());

        config.model.activation = "Sigmoid".to_string();
        assert!(config.validate().is_err());

        // Lowercase should work
        config.model.activation = "relu".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_dataset_validation_split_ratios() {
        let mut config = Config::default();
        config.dataset.train_ratio = 0.80;
        config.dataset.val_ratio = 0.15;
        config.dataset.test_ratio = 0.10;
        // Sum is 1.05, should fail
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_parse_invalid_toml() {
        let invalid_toml = "this is not valid toml {{{}";
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(invalid_toml.as_bytes()).unwrap();

        let result = Config::from_file(temp_file.path());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ConfigError::ParseError(_)));
    }
}
