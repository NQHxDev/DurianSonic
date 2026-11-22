/// Example demonstrating how to use the configuration system
use durian_sonic::config::Config;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("DurianSonic Configuration Demo...\n");

    // Load from default config file
    println!("Loading from configs/default.toml:");
    match Config::load_default() {
        Ok(config) => {
            println!("   ✓ Successfully loaded config");
            print_config(&config);
        }
        Err(e) => {
            println!("   ✗ Failed to load: {}", e);
            println!("   → Using fallback default config instead");
            let config = Config::default();
            print_config(&config);
        }
    }

    // Create and save a custom config
    println!("\nCreating custom configuration:");
    let mut custom_config = Config::default();
    custom_config.training.epochs = 200;
    custom_config.training.batch_size = 64;
    custom_config.training.learning_rate = 0.005;
    custom_config.model.hidden_layers = vec![256, 128, 64];
    custom_config.model.activation = "tanh".to_string();

    println!("   Custom config created:");
    print_config(&custom_config);

    // Save to a file
    let custom_path = "configs/custom_experiment.toml";
    match custom_config.save_to_file(custom_path) {
        Ok(_) => println!("   ✓ Saved to {}", custom_path),
        Err(e) => println!("   ✗ Failed to save: {}", e),
    }

    // Validation demo
    println!("\nConfiguration validation:");
    let mut invalid_config = Config::default();
    invalid_config.training.learning_rate = 1.5; // Invalid: > 1.0

    match invalid_config.validate() {
        Ok(_) => println!("   ✓ Config is valid"),
        Err(e) => println!("   ✗ Validation failed: {}", e),
    }

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn print_config(config: &Config) {
    println!("   Training:");
    println!("     - Epochs: {}", config.training.epochs);
    println!("     - Batch size: {}", config.training.batch_size);
    println!("     - Learning rate: {}", config.training.learning_rate);
    println!("   Model:");
    println!("     - Hidden layers: {:?}", config.model.hidden_layers);
    println!("     - Activation: {}", config.model.activation);
}
