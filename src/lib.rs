// Module declarations
pub mod config;
pub mod dataset;
pub mod training;

/// Calculate mean of a vector
pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let sum: f64 = data.iter().sum();
    Some(sum / data.len() as f64)
}

/// Normalize data to range [0, 1]
pub fn normalize(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }

    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range == 0.0 {
        return vec![0.0; data.len()];
    }

    data.iter().map(|&x| (x - min) / range).collect()
}

/// Simple activation function (ReLU)
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&data), Some(3.0));
    }

    #[test]
    fn test_mean_empty() {
        let data: Vec<f64> = vec![];
        assert_eq!(mean(&data), None);
    }

    #[test]
    fn test_normalize_basic() {
        let data = vec![0.0, 5.0, 10.0];
        let normalized = normalize(&data);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_normalize_empty() {
        let data: Vec<f64> = vec![];
        let normalized = normalize(&data);
        assert_eq!(normalized, vec![]);
    }

    #[test]
    fn test_normalize_constant() {
        let data = vec![5.0, 5.0, 5.0];
        let normalized = normalize(&data);
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_positive() {
        assert_eq!(relu(5.0), 5.0);
    }

    #[test]
    fn test_relu_negative() {
        assert_eq!(relu(-3.0), 0.0);
    }

    #[test]
    fn test_relu_zero() {
        assert_eq!(relu(0.0), 0.0);
    }
}
