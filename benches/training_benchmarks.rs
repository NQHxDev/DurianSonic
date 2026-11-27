use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use burn::tensor::{Device, Tensor};
use durian_sonic::dsp::{
    build_mel_filterbank, extract_tap_features, extract_tap_impulses, ImpulseConfig,
    SpectrogramConfig,
};
use durian_sonic::model::DurianAudioCnn;
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

    // Benchmark: DurianAudioCnn::forward (single-tap inference)
    // Target: < 8.0 ms per tap on CPU (Constitution §IV)
    group.bench_function("forward_pass", |b| {
        let device = Device::flex();

        // Initialize model
        let model = DurianAudioCnn::new(3, 0.3, &device);

        // Create synthetic input tensors [1, 1, 128, 128] and [1, 4]
        let mel_data = vec![0.5f32; 128 * 128];
        let phys_data = vec![0.3f32, 0.5f32, 0.7f32, 0.9f32];

        let mel_spec = Tensor::<4>::from_data(
            burn::tensor::TensorData::new(mel_data, [1, 1, 128, 128]),
            &device,
        );
        let phys_feats =
            Tensor::<2>::from_data(burn::tensor::TensorData::new(phys_data, [1, 4]), &device);

        b.iter(|| {
            let _logits = model.forward(black_box(mel_spec.clone()), black_box(phys_feats.clone()));
        });
    });

    group.finish();
}

fn benchmark_dsp_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("dsp_pipeline");

    // Create synthetic 0.8s tap window (17,640 samples at 22,050 Hz)
    let cfg = SpectrogramConfig::default();
    let impulse_cfg = ImpulseConfig::default();
    let win_len = impulse_cfg.window_samples();
    let sr = impulse_cfg.target_sr;
    let freq = 600.0;

    let mut tap_window = Vec::with_capacity(win_len);
    for i in 0..win_len {
        let t = i as f32 / sr as f32;
        let decay = (-t * 2.0).exp();
        tap_window.push((2.0 * std::f32::consts::PI * freq * t).sin() * decay * 0.5);
    }

    // Pre-build mel_basis for reuse across batch
    let mel_basis =
        build_mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax);

    // Benchmark: extract_tap_features (single-pass STFT + Log-Mel + 4 physics features)
    // Target: < 4.0 ms per tap (SC-005)
    group.bench_function("dsp_extract_tap_features", |b| {
        b.iter(|| {
            extract_tap_features(
                black_box(&tap_window),
                black_box(&cfg),
                black_box(&mel_basis),
            )
        });
    });

    // Benchmark: extract_tap_impulses (onset detection + windowing on full recording)
    let full_recording = vec![0.3; 220_500]; // 10s synthetic recording
    group.bench_function("dsp_extract_tap_impulses", |b| {
        b.iter(|| extract_tap_impulses(black_box(&full_recording), black_box(&impulse_cfg)));
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_gradient_descent,
    benchmark_learning_rate_schedules,
    benchmark_model_training,
    benchmark_dsp_pipeline
);

criterion_main!(benches);
