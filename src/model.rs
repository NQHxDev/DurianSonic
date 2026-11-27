//! Neural network architecture module for DurianSonic Phase 3.
//!
//! This module implements the hybrid 2D Log-Mel Spectrogram CNN combined with
//! a 4-dimensional acoustic physics embedding branch, producing a compact
//! (~408K parameters, < 2.0 MB FP32) model suitable for mobile deployment.

use crate::config::ModelConfig;
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
        Relu,
    },
    prelude::*,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::{activation::softmax, Device, Tensor},
};
use std::path::Path;

/// 2D Convolutional block: Conv2d(3x3, stride=2, pad=Same) -> BatchNorm -> ReLU.
///
/// Halves spatial dimensions at each application: [B, in_ch, H, W] -> [B, out_ch, H/2, W/2].
///
/// # Examples
///
/// ```
/// use durian_sonic::model::ConvBlock;
/// use burn::prelude::*;
///
/// let device = Device::flex();
/// let block = ConvBlock::new(1, 32, &device);
///
/// let input = Tensor::<4>::zeros([2, 1, 128, 128], &device);
/// let output = block.forward(input);
/// assert_eq!(output.dims(), [2, 32, 64, 64]);
/// ```
#[derive(Module, Debug)]
pub struct ConvBlock {
    pub conv: Conv2d,
    pub norm: BatchNorm,
    pub activation: Relu,
}

impl ConvBlock {
    /// Initializes a 3x3 stride-2 Same-padded ConvBlock halving spatial resolution.
    pub fn new(in_ch: usize, out_ch: usize, device: &Device) -> Self {
        let conv = Conv2dConfig::new([in_ch, out_ch], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let norm = BatchNormConfig::new(out_ch).init(device);
        let activation = Relu::new();

        Self {
            conv,
            norm,
            activation,
        }
    }

    /// Maps [B, in_ch, H, W] -> [B, out_ch, H / 2, W / 2].
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.conv.forward(x);
        let x = self.norm.forward(x);
        self.activation.forward(x)
    }
}

/// Hybrid 2D Log-Mel Spectrogram CNN + 4-D Acoustic Physics neural network (~408K params).
///
/// Combines a 4-stage Spectrogram Branch (1 -> 32 -> 64 -> 128 -> 256 channels) with
/// a Physics Embedding Branch (4 -> 16 features), late-fused into a 3-class classifier.
///
/// # Architecture
///
/// - **Spectrogram Branch**: 4 sequential ConvBlocks + AdaptiveAvgPool2d([1, 1])
///   - Input: [B, 1, 128, 128] Log-Mel spectrogram
///   - Output: [B, 256] global features
/// - **Physics Branch**: Linear(4 -> 16) + ReLU
///   - Input: [B, 4] acoustic physics features
///   - Output: [B, 16]
/// - **Late-Fusion Head**: Concatenation [B, 272] -> Dropout(0.3) -> Linear(272 -> 64) + ReLU -> Linear(64 -> 3)
///   - Output: [B, 3] class logits for [Unripe, Ripe, Overripe]
///
/// # Examples
///
/// ```
/// use durian_sonic::model::DurianAudioCnn;
/// use burn::prelude::*;
///
/// let device = Device::flex();
/// let model = DurianAudioCnn::new(3, 0.3, &device);
///
/// let mel = Tensor::<4>::zeros([1, 1, 128, 128], &device);
/// let phys = Tensor::<2>::zeros([1, 4], &device);
/// let logits = model.forward(mel, phys);
/// assert_eq!(logits.dims(), [1, 3]);
/// ```
#[derive(Module, Debug)]
pub struct DurianAudioCnn {
    pub block1: ConvBlock,
    pub block2: ConvBlock,
    pub block3: ConvBlock,
    pub block4: ConvBlock,
    pub global_pool: AdaptiveAvgPool2d,
    pub phys_proj: Linear,
    pub dropout: Dropout,
    pub fc1: Linear,
    pub fc_out: Linear,
    pub activation: Relu,
}

impl DurianAudioCnn {
    /// Initializes DurianAudioCnn with `num_classes` (default 3) and `dropout_rate` (default 0.3).
    pub fn new(num_classes: usize, dropout_rate: f64, device: &Device) -> Self {
        Self {
            block1: ConvBlock::new(1, 32, device),
            block2: ConvBlock::new(32, 64, device),
            block3: ConvBlock::new(64, 128, device),
            block4: ConvBlock::new(128, 256, device),
            global_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            phys_proj: LinearConfig::new(4, 16).init(device),
            dropout: DropoutConfig::new(dropout_rate).init(),
            fc1: LinearConfig::new(272, 64).init(device),
            fc_out: LinearConfig::new(64, num_classes).init(device),
            activation: Relu::new(),
        }
    }

    /// Convenience constructor initializing from `ModelConfig`.
    pub fn from_config(config: &ModelConfig, device: &Device) -> Self {
        Self::new(config.num_classes, config.dropout, device)
    }

    /// Raw logit forward pass.
    ///
    /// # Inputs
    /// - `mel_spec`: [B, 1, 128, 128] in [0.0, 1.0]
    /// - `phys_feats`: [B, 4] in [0.0, 1.0]
    ///
    /// # Output
    /// - Raw unnormalized logits [B, 3] for [Unripe, Ripe, Overripe]
    pub fn forward(&self, mel_spec: Tensor<4>, phys_feats: Tensor<2>) -> Tensor<2> {
        // Spectrogram branch: [B, 1, 128, 128] -> [B, 256]
        let x = self.block1.forward(mel_spec);
        let x = self.block2.forward(x);
        let x = self.block3.forward(x);
        let x = self.block4.forward(x);
        let x = self.global_pool.forward(x); // [B, 256, 1, 1]

        // Dynamic batch size extraction for reshape
        let [batch_size, _, _, _] = x.dims();
        let spec_emb = x.reshape([batch_size, 256]);

        // Physics branch: [B, 4] -> [B, 16]
        let phys_emb = self.phys_proj.forward(phys_feats);
        let phys_emb = self.activation.forward(phys_emb);

        // Late fusion: [B, 256] + [B, 16] -> [B, 272]
        let fused = Tensor::cat(vec![spec_emb, phys_emb], 1);

        // Classification head: [B, 272] -> [B, 3]
        let x = self.dropout.forward(fused);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        self.fc_out.forward(x)
    }

    /// Normalized probability forward pass applying softmax(logits, 1) along class dimension.
    ///
    /// # Output
    /// - [B, 3] probabilities in [0.0, 1.0] where each row sums to 1.0
    pub fn forward_proba(&self, mel_spec: Tensor<4>, phys_feats: Tensor<2>) -> Tensor<2> {
        let logits = self.forward(mel_spec, phys_feats);
        softmax(logits, 1)
    }

    /// Saves model weights to .mpk file (MessagePack format) with full precision.
    ///
    /// Creates parent directories if needed and normalizes .mpk suffix.
    /// Uses Burn's BurnpackStore internally for FP32 serialization.
    ///
    /// # Returns
    /// PathBuf to the saved .mpk file (should be < 2.0 MB)
    pub fn save_mpk<P: AsRef<Path>>(&self, output_path: P) -> anyhow::Result<std::path::PathBuf> {
        use anyhow::Context;
        use std::fs;

        let path = output_path.as_ref();

        // Normalize path: ensure .mpk extension
        let normalized_path = if path.extension().and_then(|s| s.to_str()) == Some("mpk") {
            path.to_path_buf()
        } else {
            path.with_extension("mpk")
        };

        // Create parent directories if needed
        if let Some(parent) = normalized_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
        }

        // Remove existing file if present so BurnpackStore does not fail with AlreadyExists
        if normalized_path.is_file() {
            fs::remove_file(&normalized_path).with_context(|| {
                format!(
                    "Failed to remove existing model file: {}",
                    normalized_path.display()
                )
            })?;
        }

        // Save model using BurnpackStore (FP32 full precision)
        let mut store = BurnpackStore::from_file(&normalized_path);
        self.save_into(&mut store)
            .map_err(|e| anyhow::anyhow!("Failed to save model: {:?}", e))?;

        Ok(normalized_path)
    }

    /// Loads model weights from .mpk file (MessagePack format).
    ///
    /// # Arguments
    /// - `input_path`: Path to .mpk file (with or without .mpk extension)
    /// - `device`: Device for model computation
    ///
    /// # Returns
    /// Loaded DurianAudioCnn model
    pub fn load_mpk<P: AsRef<Path>>(
        mut self,
        input_path: P,
        _device: &Device,
    ) -> anyhow::Result<Self> {
        use anyhow::Context;

        let path = input_path.as_ref();

        // Normalize path: ensure .mpk extension
        let normalized_path = if path.extension().and_then(|s| s.to_str()) == Some("mpk") {
            path.to_path_buf()
        } else {
            path.with_extension("mpk")
        };

        // Verify file exists
        if !normalized_path.exists() {
            anyhow::bail!("Model file not found: {}", normalized_path.display());
        }

        // Load model using BurnpackStore (FP32 full precision)
        let mut store = BurnpackStore::from_file(&normalized_path);
        self.load_from(&mut store)
            .with_context(|| format!("Failed to load model from {}", normalized_path.display()))?;

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Phase 3 User Story 1 Tests: ConvBlock and DurianAudioCnn (T005)

    #[test]
    fn test_conv_block_spatial_downsampling() {
        let device = Device::flex();

        // ConvBlock should map [B, in_ch, H, W] -> [B, out_ch, H/2, W/2]
        // Test with [2, 1, 128, 128] -> [2, 32, 64, 64]
        let batch_size = 2;
        let input_data = vec![0.5f32; batch_size * 128 * 128];
        let input = Tensor::<4>::from_data(
            burn::tensor::TensorData::new(input_data, [batch_size, 1, 128, 128]),
            &device,
        );

        let block = ConvBlock::new(1, 32, &device);
        let output = block.forward(input);

        // Verify output shape: [2, 32, 64, 64]
        assert_eq!(output.dims(), [2, 32, 64, 64]);

        // Verify all values are finite
        let output_data = output.to_data();
        for val in output_data.iter::<f32>() {
            assert!(val.is_finite(), "Output must be finite");
        }
    }

    #[test]
    fn test_durian_audio_cnn_forward_shape_and_proba_simplex() {
        let device = Device::flex();

        // Test with dynamic batch sizes: B = 1, 4, 7
        for batch_size in [1, 4, 7] {
            // Create input tensors
            let mel_data = vec![0.5f32; batch_size * 128 * 128];
            let mel_spec = Tensor::<4>::from_data(
                burn::tensor::TensorData::new(mel_data, [batch_size, 1, 128, 128]),
                &device,
            );

            let phys_data = vec![0.5f32; batch_size * 4];
            let phys_feats = Tensor::<2>::from_data(
                burn::tensor::TensorData::new(phys_data, [batch_size, 4]),
                &device,
            );

            // Test both constructors
            let model1 = DurianAudioCnn::new(3, 0.3, &device);
            let model2 = DurianAudioCnn::from_config(&ModelConfig::default(), &device);

            for model in [model1, model2] {
                // Test forward (raw logits)
                let logits = model.forward(mel_spec.clone(), phys_feats.clone());
                assert_eq!(
                    logits.dims(),
                    [batch_size, 3],
                    "Logits shape must be [B, 3]"
                );

                // Verify all logits are finite
                let logits_data = logits.to_data();
                for val in logits_data.iter::<f32>() {
                    assert!(val.is_finite(), "Logits must be finite");
                }

                // Test forward_proba (softmax probabilities)
                let probs = model.forward_proba(mel_spec.clone(), phys_feats.clone());
                assert_eq!(probs.dims(), [batch_size, 3], "Probs shape must be [B, 3]");

                let probs_data = probs.to_data();
                let probs_vec: Vec<f32> = probs_data.iter::<f32>().collect();

                // Verify each row is in [0.0, 1.0] and sums to 1.0
                for b in 0..batch_size {
                    let row_start = b * 3;
                    let row = &probs_vec[row_start..row_start + 3];

                    for &p in row {
                        assert!(
                            (0.0..=1.0).contains(&p),
                            "Probability must be in [0.0, 1.0]"
                        );
                    }

                    let row_sum: f32 = row.iter().sum();
                    assert_relative_eq!(row_sum, 1.0, epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_durian_audio_cnn_boundary_inputs_stability() {
        let device = Device::flex();
        let model = DurianAudioCnn::new(3, 0.3, &device);

        // Test with all-zero input (noise floor)
        let mel_zeros = Tensor::<4>::zeros([1, 1, 128, 128], &device);
        let phys_zeros = Tensor::<2>::zeros([1, 4], &device);

        let logits = model.forward(mel_zeros, phys_zeros);
        let logits_data = logits.to_data();
        for val in logits_data.iter::<f32>() {
            assert!(val.is_finite(), "Must handle zero inputs without NaN/Inf");
        }

        // Test with all-one input (peak values)
        let mel_ones = Tensor::<4>::ones([1, 1, 128, 128], &device);
        let phys_ones = Tensor::<2>::ones([1, 4], &device);

        let logits = model.forward(mel_ones, phys_ones);
        let logits_data = logits.to_data();
        for val in logits_data.iter::<f32>() {
            assert!(val.is_finite(), "Must handle max inputs without NaN/Inf");
        }

        let probs = model.forward_proba(
            Tensor::<4>::ones([1, 1, 128, 128], &device),
            Tensor::<2>::ones([1, 4], &device),
        );
        let probs_data = probs.to_data();
        for val in probs_data.iter::<f32>() {
            assert!(
                val.is_finite(),
                "Softmax must handle extreme inputs without NaN/Inf"
            );
        }
    }

    #[test]
    fn test_durian_audio_cnn_mpk_save_load_and_size_budget() {
        use std::fs;
        use tempfile::tempdir;

        let device = Device::flex();
        let model = DurianAudioCnn::new(3, 0.3, &device);

        // Create test inputs
        let mel_spec = Tensor::<4>::ones([2, 1, 128, 128], &device);
        let phys_feats = Tensor::<2>::ones([2, 4], &device);

        // Get original logits
        let original_logits = model.forward(mel_spec.clone(), phys_feats.clone());
        let original_data: Vec<f32> = original_logits.to_data().iter::<f32>().collect();

        // Save model to temporary file
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let model_path = temp_dir.path().join("test_model.mpk");

        model.save_mpk(&model_path).expect("Failed to save model");
        // Verify overwriting an existing .mpk file succeeds without AlreadyExists error
        model
            .save_mpk(&model_path)
            .expect("Failed to overwrite existing model file");

        // Verify file exists and check size budget (< 2.0 MB)
        assert!(model_path.exists(), "Model file should exist after save");
        let file_size = fs::metadata(&model_path)
            .expect("Failed to read model file metadata")
            .len();
        assert!(
            file_size < 2_000_000,
            "Model size {} bytes must be < 2,000,000 bytes (< 2.0 MB)",
            file_size
        );

        // Load model from file
        let loaded_model = DurianAudioCnn::new(3, 0.3, &device)
            .load_mpk(&model_path, &device)
            .expect("Failed to load model");

        // Get loaded model logits
        let loaded_logits = loaded_model.forward(mel_spec, phys_feats);
        let loaded_data: Vec<f32> = loaded_logits.to_data().iter::<f32>().collect();

        // Verify exact logit match after reload
        assert_eq!(
            original_data.len(),
            loaded_data.len(),
            "Loaded model must produce same output shape"
        );
        for (i, (&orig, &loaded)) in original_data.iter().zip(loaded_data.iter()).enumerate() {
            assert_relative_eq!(orig, loaded, epsilon = 1e-6);
            if (orig - loaded).abs() > 1e-6 {
                panic!("Logit {} mismatch: original={}, loaded={}", i, orig, loaded);
            }
        }
    }
}
