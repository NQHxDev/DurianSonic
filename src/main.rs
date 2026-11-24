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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::VerifyDataset { data_dir, config }) => {
            verify_dataset_command(data_dir, config)?;
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
        println!("Performing anti-leakage stratified group split...");
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
