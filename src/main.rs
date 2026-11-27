use anyhow::Result;
use clap::{Parser, Subcommand};
use durian_sonic::config::Config;
use durian_sonic::dataset::{
    compute_class_weights, load_and_verify_metadata, stratified_group_split_by_fruit,
    PlatformTarget,
};
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "durian-sonic")]
#[command(about = "DurianSonic - Durian ripeness classification via acoustic analysis", long_about = None)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Verify dataset integrity and anti-leakage splits
    VerifyDataset {
        /// Override data directory path
        #[arg(long)]
        data_dir: Option<PathBuf>,

        /// Path to configuration file
        #[arg(long, default_value = "configs/default.toml")]
        config: PathBuf,
    },
    /// Train platform-specific model (durian_iphone.mpk or durian_android.mpk)
    Train {
        /// Target platform: "iphone" or "android"
        #[arg(long)]
        platform: String,

        /// Path to configuration file
        #[arg(long, default_value = "configs/default.toml")]
        config: PathBuf,

        /// Override data directory path
        #[arg(long)]
        data_dir: Option<PathBuf>,

        /// Output path stem for .mpk checkpoint (default: artifacts/durian_{platform})
        #[arg(long)]
        output: Option<PathBuf>,

        /// Compute device: "auto" (probe GPU, fallback CPU), "gpu" (force GPU), "cpu" (force CPU)
        #[arg(long, default_value = "auto")]
        device: String,
    },
}

fn print_confusion_matrix(cm: &[[usize; 3]; 3]) {
    println!("              Unripe  Ripe  Overripe");
    println!(
        "    Unripe    {:6}  {:4}  {:8}",
        cm[0][0], cm[0][1], cm[0][2]
    );
    println!(
        "    Ripe      {:6}  {:4}  {:8}",
        cm[1][0], cm[1][1], cm[1][2]
    );
    println!(
        "    Overripe  {:6}  {:4}  {:8}",
        cm[2][0], cm[2][1], cm[2][2]
    );
}

fn main() -> Result<()> {
    // Initialize tracing subscriber for logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::VerifyDataset { data_dir, config }) => {
            verify_dataset_command(data_dir, config)?;
        }
        Some(Commands::Train {
            platform,
            config,
            data_dir,
            output,
            device,
        }) => {
            train_command(platform, config, data_dir, output, device)?;
        }
        None => {
            // This branch is unreachable because arg_required_else_help = true
            // causes clap to display help and exit before reaching here
            unreachable!("clap should have exited due to arg_required_else_help")
        }
    }

    Ok(())
}

fn verify_dataset_command(data_dir_override: Option<PathBuf>, config_path: PathBuf) -> Result<()> {
    // Load configuration
    if !config_path.exists() {
        eprintln!(
            "Error: Configuration file does not exist: {}",
            config_path.display()
        );
        std::process::exit(1);
    }

    let mut config = match Config::from_file(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!(
                "Error: Failed to load or validate configuration file '{}': {}",
                config_path.display(),
                e
            );
            std::process::exit(1);
        }
    };

    // Override data_dir if provided
    if let Some(override_dir) = data_dir_override {
        config.dataset.data_dir = override_dir
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid data directory path"))?
            .to_string();
    }

    let data_dir = PathBuf::from(&config.dataset.data_dir);

    println!("DurianSonic Dataset Verification");
    println!("------------------------------------------\n");

    // Verify both platforms
    for platform in [PlatformTarget::Iphone, PlatformTarget::Android] {
        let platform_name = match platform {
            PlatformTarget::Iphone => "iPhone",
            PlatformTarget::Android => "Android",
        };

        println!("Platform: {}", platform_name);
        println!("--------\n");

        let csv_path = platform.metadata_csv_path(&data_dir);
        let audio_dir = platform.audio_subdir(&data_dir);

        // Load and verify metadata with SHA-256 checking
        println!(
            "Loading and verifying metadata from: {}",
            csv_path.display()
        );
        let (records, report) = load_and_verify_metadata(&csv_path, &audio_dir, true)?;

        println!("  Total files: {}", report.total_files);
        println!("  Unique fruits: {}", report.unique_fruits);
        println!(
            "  Class counts: [Unripe: {}, Ripe: {}, Overripe: {}]",
            report.class_counts[0], report.class_counts[1], report.class_counts[2]
        );
        println!("  Total bytes: {} bytes\n", report.total_bytes);

        // Compute class weights
        let weights = compute_class_weights(&report.class_counts);
        println!(
            "  Class weights: [{:.3}, {:.3}, {:.3}]\n",
            weights[0], weights[1], weights[2]
        );

        // Perform stratified group split
        tracing::info!("Performing anti-leakage stratified group split...");
        let splits = stratified_group_split_by_fruit(
            &records,
            config.dataset.train_ratio,
            config.dataset.val_ratio,
            config.dataset.split_seed,
        )?;

        // Verify zero fruit_id overlap
        let train_fruits: HashSet<u32> = splits.train.iter().map(|r| r.fruit_id).collect();
        let val_fruits: HashSet<u32> = splits.val.iter().map(|r| r.fruit_id).collect();
        let test_fruits: HashSet<u32> = splits.test.iter().map(|r| r.fruit_id).collect();

        let train_val_overlap = train_fruits.intersection(&val_fruits).count();
        let train_test_overlap = train_fruits.intersection(&test_fruits).count();
        let val_test_overlap = val_fruits.intersection(&test_fruits).count();

        println!(
            "  Train: {} files from {} fruits",
            splits.train.len(),
            train_fruits.len()
        );
        println!(
            "  Val:   {} files from {} fruits",
            splits.val.len(),
            val_fruits.len()
        );
        println!(
            "  Test:  {} files from {} fruits",
            splits.test.len(),
            test_fruits.len()
        );

        println!("\nAnti-leakage verification:");
        println!("  Train/Val overlap: {} fruits", train_val_overlap);
        println!("  Train/Test overlap: {} fruits", train_test_overlap);
        println!("  Val/Test overlap: {} fruits", val_test_overlap);

        if train_val_overlap == 0 && train_test_overlap == 0 && val_test_overlap == 0 {
            println!("  Status: PASS - Zero fruit_id leakage across splits\n");
        } else {
            eprintln!("  Status: FAIL - Fruit_id leakage detected!\n");
            std::process::exit(1);
        }

        println!();
    }

    println!("------------------------------------------");
    println!("Dataset verification complete: ALL CHECKS PASSED");

    Ok(())
}

fn train_command(
    platform_str: String,
    config_path: PathBuf,
    data_dir_override: Option<PathBuf>,
    output_override: Option<PathBuf>,
    device_str: String,
) -> Result<()> {
    use durian_sonic::training::run_platform_training;

    // Load configuration
    if !config_path.exists() {
        eprintln!(
            "Error: Configuration file does not exist: {}",
            config_path.display()
        );
        std::process::exit(1);
    }

    let mut config = match Config::from_file(&config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!(
                "Error: Failed to load or validate configuration file '{}': {}",
                config_path.display(),
                e
            );
            std::process::exit(1);
        }
    };

    // Override data_dir if provided
    if let Some(override_dir) = data_dir_override {
        config.dataset.data_dir = override_dir
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid data directory path"))?
            .to_string();
    }

    // Parse platform (case-insensitive)
    let platform = match platform_str.to_lowercase().as_str() {
        "iphone" => PlatformTarget::Iphone,
        "android" => PlatformTarget::Android,
        _ => {
            eprintln!(
                "Error: Invalid platform '{}'. Must be 'iphone' or 'android'.",
                platform_str
            );
            std::process::exit(1);
        }
    };

    // Determine output path
    let output_path = output_override.unwrap_or_else(|| {
        PathBuf::from(format!(
            "artifacts/durian_{}",
            match platform {
                PlatformTarget::Iphone => "iphone",
                PlatformTarget::Android => "android",
            }
        ))
    });

    println!("DurianSonic Model Training");
    println!("------------------------------------------\n");
    println!(
        "Platform: {}",
        match platform {
            PlatformTarget::Iphone => "iPhone",
            PlatformTarget::Android => "Android",
        }
    );
    println!("Configuration: {}", config_path.display());
    println!("Output: {}.mpk\n", output_path.display());

    // Device selection logic
    let device = match device_str.to_lowercase().as_str() {
        "cpu" => {
            tracing::info!("Device: CPU (forced)");
            println!("Device: CPU (forced)\n");
            burn::tensor::Device::flex().autodiff()
        }
        "gpu" => {
            tracing::info!("Device: GPU (forced)");
            println!("Device: GPU (forced)");
            match burn::tensor::Device::wgpu_options().init() {
                Ok(gpu_device) => {
                    tracing::info!("GPU device initialized successfully");
                    println!("GPU device initialized successfully\n");
                    gpu_device.autodiff()
                }
                Err(e) => {
                    eprintln!("Error: Failed to initialize GPU device: {:?}", e);
                    eprintln!("GPU was forced but not available. Use --device cpu or --device auto instead.");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            // Check for headless CI environment
            let is_ci = std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok();

            if is_ci {
                tracing::info!("Device: CPU (CI environment detected, skipping GPU probe)");
                println!("Device: CPU (CI environment detected, skipping GPU probe)\n");
                burn::tensor::Device::flex().autodiff()
            } else {
                // Try GPU first, fallback to CPU
                tracing::info!("Device: Auto-detecting...");
                println!("Device: Auto-detecting...");
                match burn::tensor::Device::wgpu_options().init() {
                    Ok(gpu_device) => {
                        tracing::info!("GPU device detected and initialized (Metal/Vulkan/DX12)");
                        println!("GPU device detected and initialized (Metal/Vulkan/DX12)\n");
                        gpu_device.autodiff()
                    }
                    Err(_) => {
                        tracing::info!("No GPU available, falling back to CPU");
                        println!("No GPU available, falling back to CPU\n");
                        burn::tensor::Device::flex().autodiff()
                    }
                }
            }
        }
    };

    tracing::info!("Starting training...");
    println!("Starting training...\n");

    // Run training
    let outcome = run_platform_training(&config, platform, &output_path, &device)?;

    // Print results
    println!("\n------------------------------------------");
    println!("Training Complete\n");

    println!("Platform Class Weights:");
    println!(
        "  Unripe: {:.3}, Ripe: {:.3}, Overripe: {:.3}\n",
        outcome.class_weights[0], outcome.class_weights[1], outcome.class_weights[2]
    );

    println!("Training Summary:");
    println!("  Best Epoch: {}", outcome.best_epoch);
    println!("  Epochs Run: {}", outcome.epochs_run);
    println!(
        "  Best Val Recording Macro F1: {:.4}\n",
        outcome.best_val_recording_macro_f1
    );

    println!("Model Artifact:");
    println!("  Path: {}", outcome.artifact_path.display());
    println!(
        "  Size: {} bytes ({:.2} MB)",
        outcome.artifact_size_bytes,
        outcome.artifact_size_bytes as f64 / 1_000_000.0
    );
    if outcome.artifact_size_bytes < 2_000_000 {
        println!("  Status: PASS (< 2.0 MB)\n");
    } else {
        println!("  Status: FAIL (>= 2.0 MB)\n");
    }

    println!("Test Evaluation (Tap-Level):");
    let tap = &outcome.test_evaluation.tap_level;
    println!("  Macro F1: {:.4}", tap.macro_f1);
    println!("  Balanced Accuracy: {:.4}", tap.balanced_accuracy);
    println!("  Overall Accuracy: {:.4}", tap.overall_accuracy);
    println!("  Critical FP Ripe Rate: {:.4}", tap.critical_fp_ripe_rate);

    println!("\n  Confusion Matrix (Actual \\ Predicted):");
    print_confusion_matrix(&tap.confusion);

    println!("\n  Per-Class Metrics:");
    println!(
        "    Unripe   - Precision: {:.4}, Recall: {:.4}, F1: {:.4}",
        tap.per_class_precision[0], tap.per_class_recall[0], tap.per_class_f1[0]
    );
    println!(
        "    Ripe     - Precision: {:.4}, Recall: {:.4}, F1: {:.4}",
        tap.per_class_precision[1], tap.per_class_recall[1], tap.per_class_f1[1]
    );
    println!(
        "    Overripe - Precision: {:.4}, Recall: {:.4}, F1: {:.4}\n",
        tap.per_class_precision[2], tap.per_class_recall[2], tap.per_class_f1[2]
    );

    println!("Test Evaluation (Recording-Level with Entropy Weighting):");
    let rec = &outcome.test_evaluation.recording_level;
    println!("  Macro F1: {:.4}", rec.macro_f1);
    println!("  Balanced Accuracy: {:.4}", rec.balanced_accuracy);
    println!("  Overall Accuracy: {:.4}", rec.overall_accuracy);
    println!("  Critical FP Ripe Rate: {:.4}", rec.critical_fp_ripe_rate);

    println!("\n  Confusion Matrix (Actual \\ Predicted):");
    print_confusion_matrix(&rec.confusion);

    println!("\n  Per-Class Metrics:");
    println!(
        "    Unripe   - Precision: {:.4}, Recall: {:.4}, F1: {:.4}",
        rec.per_class_precision[0], rec.per_class_recall[0], rec.per_class_f1[0]
    );
    println!(
        "    Ripe     - Precision: {:.4}, Recall: {:.4}, F1: {:.4}",
        rec.per_class_precision[1], rec.per_class_recall[1], rec.per_class_f1[1]
    );
    println!(
        "    Overripe - Precision: {:.4}, Recall: {:.4}, F1: {:.4}\n",
        rec.per_class_precision[2], rec.per_class_recall[2], rec.per_class_f1[2]
    );

    // Check quality thresholds
    if rec.macro_f1 >= 0.85 && rec.balanced_accuracy >= 0.85 {
        println!("Quality Status: PASS (Macro F1 >= 0.85 and Balanced Accuracy >= 0.85)");
    } else {
        println!("Quality Status: FAIL (Target: Macro F1 >= 0.85 and Balanced Accuracy >= 0.85)");
    }

    println!("\n------------------------------------------");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_train_subcommand_parsing_and_platform_validation() {
        // Test valid iPhone platform (lowercase)
        let args = vec!["durian-sonic", "train", "--platform", "iphone"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "Should parse 'train --platform iphone'");

        if let Ok(parsed) = cli {
            match parsed.command {
                Some(Commands::Train {
                    platform, config, ..
                }) => {
                    assert_eq!(platform, "iphone");
                    assert_eq!(config, PathBuf::from("configs/default.toml"));
                }
                _ => panic!("Expected Train command"),
            }
        }

        // Test valid Android platform (lowercase)
        let args = vec!["durian-sonic", "train", "--platform", "android"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "Should parse 'train --platform android'");

        // Test case-insensitive parsing (uppercase)
        let args = vec!["durian-sonic", "train", "--platform", "IPHONE"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "Should parse 'train --platform IPHONE'");

        // Test case-insensitive parsing (mixed case)
        let args = vec!["durian-sonic", "train", "--platform", "Android"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "Should parse 'train --platform Android'");

        // Test custom config path
        let args = vec![
            "durian-sonic",
            "train",
            "--platform",
            "iphone",
            "--config",
            "configs/custom.toml",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "Should parse with custom config");

        if let Ok(parsed) = cli {
            match parsed.command {
                Some(Commands::Train { config, .. }) => {
                    assert_eq!(config, PathBuf::from("configs/custom.toml"));
                }
                _ => panic!("Expected Train command"),
            }
        }

        // Test custom output path
        let args = vec![
            "durian-sonic",
            "train",
            "--platform",
            "iphone",
            "--output",
            "models/my_model",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "Should parse with custom output");

        // Test custom data directory
        let args = vec![
            "durian-sonic",
            "train",
            "--platform",
            "android",
            "--data-dir",
            "/path/to/data",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "Should parse with custom data directory");

        // Test missing required --platform flag
        let args = vec!["durian-sonic", "train"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_err(), "Should fail without --platform flag");

        // Note: Invalid platform strings like "banana" are accepted by clap parser
        // but rejected at runtime in train_command() with std::process::exit(1)
        // This test validates the CLI parsing layer, not runtime validation
    }
}
