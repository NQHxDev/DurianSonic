use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use durian_sonic::training::{GradientDescent, LearningRateSchedule};

fn benchmark_gradient_descent(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_descent");

    group.bench_function("update_weights_small", |b| {
        let mut optimizer = GradientDescent::new(0.01);
        let weights = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.1, 0.2, 0.3];

        b.iter(|| optimizer.update(black_box(&weights), black_box(&gradients)));
    });

    group.bench_function("update_weights_large", |b| {
        let mut optimizer = GradientDescent::new(0.01);
        let weights = vec![1.0; 1000];
        let gradients = vec![0.1; 1000];

        b.iter(|| optimizer.update(black_box(&weights), black_box(&gradients)));
    });

    group.finish();
}

fn benchmark_learning_rate_schedules(c: &mut Criterion) {
    let mut group = c.benchmark_group("learning_rate_schedules");

    group.bench_function("constant_schedule", |b| {
        let schedule = LearningRateSchedule::Constant { rate: 0.01 };

        b.iter(|| schedule.get_rate(black_box(100)));
    });

    group.bench_function("exponential_decay", |b| {
        let schedule = LearningRateSchedule::ExponentialDecay {
            initial_rate: 0.1,
            decay_rate: 0.95,
            decay_steps: 100,
        };

        b.iter(|| schedule.get_rate(black_box(500)));
    });

    group.bench_function("step_decay", |b| {
        let schedule = LearningRateSchedule::StepDecay {
            initial_rate: 0.1,
            drop_rate: 0.5,
            epochs_drop: 10,
        };

        b.iter(|| schedule.get_rate(black_box(50)));
    });

    group.finish();
}

fn benchmark_model_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_training");

    // Add more comprehensive training benchmarks here
    group.bench_function("forward_pass", |b| {
        // TODO: Implement when model structure is ready
        b.iter(|| black_box(42));
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_gradient_descent,
    benchmark_learning_rate_schedules,
    benchmark_model_training
);

criterion_main!(benches);
