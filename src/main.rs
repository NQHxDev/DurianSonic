use durian_sonic::{mean, normalize, relu};

fn main() {
    println!("Hello World!");

    // Demo Test System
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    if let Some(avg) = mean(&data) {
        println!("\nMean of data: {}", avg);
    }

    let normalized = normalize(&data);
    println!("Normalized data: {:?}", normalized);

    println!("\nReLU activation:");
    for &x in &[-2.0, 0.0, 3.0] {
        println!("  relu({}) = {}", x, relu(x));
    }
}
