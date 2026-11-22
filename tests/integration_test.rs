use durian_sonic::{mean, normalize, relu};

#[test]
fn test_data_pipeline() {
    // Simulate a simple data preprocessing pipeline
    let raw_data = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    // Calculate mean
    let avg = mean(&raw_data).expect("Data should not be empty");
    assert_eq!(avg, 30.0);

    // Normalize
    let normalized = normalize(&raw_data);
    assert_eq!(normalized[0], 0.0);
    assert_eq!(normalized[4], 1.0);

    // Apply activation
    let activated: Vec<f64> = normalized.iter().map(|&x| relu(x - 0.5)).collect();
    assert!(activated[0] == 0.0); // 0.0 - 0.5 = -0.5 -> ReLU = 0
    assert!(activated[4] > 0.0); // 1.0 - 0.5 = 0.5 -> ReLU = 0.5
}

#[test]
fn test_empty_data_handling() {
    let empty_data: Vec<f64> = vec![];

    assert_eq!(mean(&empty_data), None);
    assert_eq!(normalize(&empty_data), vec![]);
}

#[test]
fn test_negative_values() {
    let data = vec![-10.0, -5.0, 0.0, 5.0, 10.0];

    let avg = mean(&data).unwrap();
    assert_eq!(avg, 0.0);

    let normalized = normalize(&data);
    assert_eq!(normalized[0], 0.0); // min value
    assert_eq!(normalized[4], 1.0); // max value
    assert_eq!(normalized[2], 0.5); // middle value
}
