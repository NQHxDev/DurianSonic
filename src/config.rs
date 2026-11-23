/// Configuration management module
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse config file: {0}")]
    ParseError(#[from] toml::de::Error),

    #[error("Invalid configuration: {0}")]
    ValidationError(String),
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub training: TrainingConfig,
    pub model: ModelConfig,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,

    /// Batch size for training
    pub batch_size: usize,

    /// Learning rate
    pub learning_rate: f64,
}

/// Model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hidden layer sizes [layer1_size, layer2_size, ...]
    pub hidden_layers: Vec<usize>,

    /// Activation function name
    pub activation: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            training: TrainingConfig {
                epochs: 100,
                batch_size: 32,
                learning_rate: 0.001,
            },
            model: ModelConfig {
                hidden_layers: vec![128, 64],
                activation: "relu".to_string(),
            },
        }
    }
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load default configuration from configs/default.toml
    pub fn load_default() -> Result<Self, ConfigError> {
        Self::from_file("configs/default.toml")
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate training config
        if self.training.epochs == 0 {
            return Err(ConfigError::ValidationError(
                "epochs must be greater than 0".to_string(),
            ));
        }

        if self.training.batch_size == 0 {
            return Err(ConfigError::ValidationError(
                "batch_size must be greater than 0".to_string(),
            ));
        }

        if self.training.learning_rate <= 0.0 || self.training.learning_rate >= 1.0 {
            return Err(ConfigError::ValidationError(
                "learning_rate must be between 0 and 1".to_string(),
            ));
        }

        // Validate model config
        if self.model.hidden_layers.is_empty() {
            return Err(ConfigError::ValidationError(
                "hidden_layers cannot be empty".to_string(),
            ));
        }

        for &layer_size in &self.model.hidden_layers {
            if layer_size == 0 {
                return Err(ConfigError::ValidationError(
                    "hidden layer size must be greater than 0".to_string(),
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

    /// Save configuration to a TOML file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| ConfigError::ValidationError(e.to_string()))?;
        fs::write(path, content)?;
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
    fn test_load_from_file() {
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
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let original_config = Config::default();
        let temp_file = NamedTempFile::new().unwrap();

        // Save
        original_config.save_to_file(temp_file.path()).unwrap();

        // Load
        let loaded_config = Config::from_file(temp_file.path()).unwrap();

        assert_eq!(
            original_config.training.epochs,
            loaded_config.training.epochs
        );
        assert_eq!(
            original_config.training.batch_size,
            loaded_config.training.batch_size
        );
        assert_eq!(
            original_config.training.learning_rate,
            loaded_config.training.learning_rate
        );
        assert_eq!(
            original_config.model.hidden_layers,
            loaded_config.model.hidden_layers
        );
        assert_eq!(
            original_config.model.activation,
            loaded_config.model.activation
        );
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
