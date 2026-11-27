# Phase 0 Research: Hybrid CNN Architecture & Training Pipeline in Burn

**Feature**: `003-phase3-model-training`
**Date**: 2026-11-26
**Status**: Complete (All technical decisions and checklist requirements resolved)

---

## 1. Hybrid 2D Spectrogram CNN & Acoustic Physics Architecture in `burn 0.22.1` (`src/model.rs`)

- **Decision**: Implement `ConvBlock<B: Backend>` and `DurianAudioCnn<B: Backend>` in `src/model.rs` (exported via `pub mod model;` in `src/lib.rs`) using the locked `burn = "0.22.1"` neural network and tensor primitives:
    1. **`ConvBlock<B: Backend>` (`#[derive(Module, Debug)]`)**:
        - Fields: `conv: Conv2d<B>`, `norm: BatchNorm<B>`, `activation: Relu`.
        - Constructor: `pub fn new(in_ch: usize, out_ch: usize, device: &B::Device) -> Self` configuring `Conv2dConfig::new([in_ch, out_ch], [3, 3]).with_stride([2, 2]).with_padding(PaddingConfig2d::Same).init(device)`, `BatchNormConfig::new(out_ch).init(device)`, and `Relu::new()`.
        - Forward pass: `pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4>` applying `self.activation.forward(self.norm.forward(self.conv.forward(x)))`, halving spatial dimensions `[H, W] -> [H/2, W/2]` at each stage.
    2. **`DurianAudioCnn<B: Backend>` (`#[derive(Module, Debug)]`)**:
        - **Spectrogram Branch**: 4 sequential `ConvBlock<B>` layers (`block1: 1 -> 32`, `block2: 32 -> 64`, `block3: 64 -> 128`, `block4: 128 -> 256`) downsampling `[B, 1, 128, 128] -> [B, 32, 64, 64] -> [B, 64, 32, 32] -> [B, 128, 16, 16] -> [B, 256, 8, 8]`, followed by `global_pool: AdaptiveAvgPool2d` (`AdaptiveAvgPool2dConfig::new([1, 1]).init()`) and dynamic batch flattening `.reshape([batch_size, 256])` where `let [batch_size, _, _, _] = mel_spec.dims();`.
        - **Physics Embedding Branch**: `phys_proj: Linear<B>` (`LinearConfig::new(4, 16).init(device)`) followed by `self.activation.forward(...)`, projecting `[B, 4] -> [B, 16]`.
        - **Late-Fusion Classification Head**: Feature concatenation `Tensor::cat(vec![spec_emb, phys_emb], 1)` producing `[B, 272]`, followed by `dropout: Dropout` (`DropoutConfig::new(dropout_rate).init()`, default `0.3`), `fc1: Linear<B>` (`LinearConfig::new(272, 64).init(device)`) + `Relu`, and `fc_out: Linear<B>` (`LinearConfig::new(64, num_classes).init(device)`, default `3`) producing raw class logits `[B, 3]`.
        - **Constructors & Inference Methods**:
            - `pub fn new(num_classes: usize, dropout_rate: f64, device: &B::Device) -> Self`
            - `pub fn from_config(config: &ModelConfig, device: &B::Device) -> Self` (delegating `Self::new(config.num_classes, config.dropout, device)`)
            - `pub fn forward(&self, mel_spec: Tensor<B, 4>, phys_feats: Tensor<B, 2>) -> Tensor<B, 2>` (raw logits `[B, 3]`)
            - `pub fn forward_proba(&self, mel_spec: Tensor<B, 4>, phys_feats: Tensor<B, 2>) -> Tensor<B, 2>` (`burn::tensor::activation::softmax(self.forward(mel_spec, phys_feats), 1)` along class dimension `dim = 1`).
- **Rationale**:
    - Combining a 4-stage 2D CNN (~389K conv/BN parameters) with an explicit 4-feature acoustic physics branch (`f_peak`, `C_s`, `R_85`, `beta_decay`) constrains the hypothesis space to physically meaningful resonance patterns, preventing overfitting on ~3,370 iPhone and ~2,100 Android tap impulses.
    - Total trainable parameters equal ~`408,019` (~`420K` budget), which serializes to ~`1.63 MB` in full-precision FP32 MessagePack (`.mpk`), strictly satisfying the `< 2.0 MB` mobile artifact requirement (`SC-005`).
    - Extracting `batch_size` dynamically from `mel_spec.dims()` guarantees safe execution for single-tap inference (`B = 1`) and partial final epoch batches (`1 <= B < 32`).
- **Alternatives Considered**:
    - _Off-the-shelf 2D vision backbones (`ResNet50`, `MobileNetV4`)_: Rejected because `1.5M–25.6M` parameters overfit studio background noise on 175 unique fruits and exceed the `< 2.0 MB` FP32 mobile weight budget.
    - _Spectrogram-only CNN without the Physics Embedding Branch_: Rejected because explicit cavity resonance frequency ($f_{\text{peak}}$) and temporal damping slope ($\beta_{\text{decay}}$) provide strong inductive bias for distinguishing `Unripe` vs `Ripe` vs `Overripe` rinds.

---

## 2. Tap Sample Extraction, Train-Only Minority Oversampling & Zero-Onset Fallback (`src/training.rs`)

- **Decision**: Implement `extract_dataset_tap_samples` (and per-recording helper `extract_recording_tap_samples`) in `src/training.rs` to transform `AudioMetadataRecord` slices into `Vec<TapSample>` while enforcing a strict train/eval boundary and precomputing the Mel filterbank once:
    1. **Single Precomputed Mel Filterbank (`CHK039`)**: Construct `ImpulseConfig::from(&config.dsp)`, `SpectrogramConfig::from(&config.dsp)`, and `mel_basis = build_mel_filterbank(spec_cfg.sample_rate, spec_cfg.n_fft, spec_cfg.n_mels, spec_cfg.fmin, spec_cfg.fmax)` once before iterating over recordings.
    2. **Zero-Onset Fallback (`FR-004` & `CHK025`)**: Decode each recording via `decode_audio_to_mono(&audio_path, impulse_cfg.target_sr)?` and detect tap centers via `detect_onset_centers(&samples, &impulse_cfg)`. If `centers.is_empty()`, emit `tracing::warn!` containing `record.sample_num` and `audio_path.display()`, and compute a single fallback center at the 256-sample frame with maximum RMS energy (clamped so a `17,640`-sample window fits inside `samples`, or zero-padded from sample `0` if `samples.len() < 17,640`). This guarantees every `Validation` and `Test` recording yields at least `1` `TapSample` for `Recording-Level` evaluation.
    3. **Train-Only Onset-Window Minority Oversampling & Augmentation Ordering (`FR-004`, `CHK007`, `CHK013`, `CHK035`)**:
        - When `is_train == false` (`Validation` and `Test` splits): Extract strictly `1` canonical peak-normalized window per `center_sample` via `extract_tap_features(&window, &spec_cfg, &mel_basis)`. Never oversample or augment.
        - When `is_train == true` (`Train` split): Set `max_jitter_samples = ((0.025 * impulse_cfg.target_sr as f32).round() as usize)` (`551` samples $= \pm 25\text{ ms}$ at `22,050 Hz`). For each `center_sample`:
            - **`RipenessLabel::Unripe` (`1x`)**: Extract `1` canonical window (unaugmented).
            - **`RipenessLabel::Ripe` (`2x`)**:
                1. Window 1: Canonical window (unaugmented).
                2. Window 2 (Replica): Apply waveform augmentations first—`jittered = apply_onset_jitter(&samples, center_sample, 551, &impulse_cfg)` followed by `noisy = add_background_noise(&jittered, 15.0, 35.0)`—then run `mut feats = extract_tap_features(&noisy, &spec_cfg, &mel_basis)` (so `physics_feats` reflect the jittered/noisy waveform), and finally apply spectrogram masking `apply_spec_augment(&mut feats.mel_spec, 12, 16, 0.0)`.
            - **`RipenessLabel::Overripe` (`3x`)**:
                1. Window 1: Canonical window (unaugmented).
                2. Windows 2 & 3 (2 Replicas): For each replica, apply waveform onset jitter first—`jittered = apply_onset_jitter(&samples, center_sample, 551, &impulse_cfg)`—then run `mut feats = extract_tap_features(&jittered, &spec_cfg, &mel_basis)`, and finally apply spectrogram augmentations `apply_spectral_tilt(&mut feats.mel_spec, 3.0, spec_cfg.top_db)` and `apply_spec_augment(&mut feats.mel_spec, 12, 16, 0.0)`.
    4. **Row-Major Flattening & Sequential `tap_index` (`CHK036`, `CHK037`)**: Flatten each `(128, 128)` `Array2<f32>` spectrogram in standard row-major order (`mel_spec.iter().copied().collect::<Vec<f32>>()`, length `16,384`) and assign sequential `tap_index: 0..N` within each recording (`sample_num`).
- **Rationale**:
    - Running waveform augmentations (`apply_onset_jitter`, `add_background_noise`) before `extract_tap_features` ensures both the `(128, 128)` Log-Mel spectrogram and the `4` physics features remain physically coherent with the perturbed waveform, while spectrogram-domain regularizers (`apply_spectral_tilt`, `apply_spec_augment`) operate in-place on `feats.mel_spec` within `[0.0, 1.0]`.
    - Oversampling minority onset windows (`2x Ripe`, `3x Overripe`) balances mini-batch class density on the `Train` split while keeping `Val` and `Test` 100% pure and leakage-free (`SC-003`).
- **Alternatives Considered**:
    - _Duplicating identical unaugmented `TapSample` tensors in `Train`_: Rejected because exact tensor copies cause the CNN to memorize specific minority spectrograms rather than learning shift- and noise-invariant representations.
    - _Applying oversampling before `stratified_group_split_by_fruit`_: Rejected because any pre-split augmentation or tap slicing would risk data leakage across splits.

---

## 3. Platform-Specific Weighted Cross-Entropy Loss with Label Smoothing (`src/training.rs`)

- **Decision**: Implement `weighted_cross_entropy_loss<B: Backend>` in `src/training.rs` combining inverse class frequency weights $w_c$ with label smoothing $\epsilon = 0.05$:
    1. **Class Weight Source (`FR-010` & `CHK009`)**: Compute platform class weights directly from the raw metadata verification report via `compute_class_weights(&report.class_counts)` _before_ tap slicing and minority oversampling:
        - **iPhone (`171 Unripe : 66 Ripe : 44 Overripe`)**: $w = [0.5478, 1.4192, 2.1288] \approx [0.548, 1.419, 2.129]$.
        - **Android (`95 Unripe : 38 Ripe : 42 Overripe`)**: $w = [0.6140, 1.5351, 1.3889] \approx [0.614, 1.535, 1.389]$.
    2. **Mathematical Formulation (`FR-006` & `CHK008`)**:
        - For each sample $i \in \{0, \dots, B-1\}$ with true label $y_i \in \{0, 1, 2\}$ and smoothing factor $\epsilon \in [0.0, 1.0)$ (default `0.05`):
          $$q_i(c) = (1 - \epsilon)\mathbb{I}(c = y_i) + \frac{\epsilon}{3} = \begin{cases} 1 - \frac{2\epsilon}{3} \approx 0.9666667 & \text{if } c = y_i \\ \frac{\epsilon}{3} \approx 0.0166667 & \text{if } c \neq y_i \end{cases}$$
        - Construct the class-weighted smoothed target matrix $T_{i,c} = w_c \cdot q_i(c)$ of shape `[B, 3]`.
        - Compute numerically stable log-probabilities $\log P_i(c) = \text{log\_softmax}(\text{logits}, 1)_{i,c}$ using `burn::tensor::activation::log_softmax`.
        - Compute batch mean loss:
          $$\mathcal{L} = \frac{1}{B}\sum_{i=0}^{B-1} \left( -\sum_{c=0}^{2} T_{i,c} \cdot \text{log\_softmax}(\text{logits}_i)_c \right)$$
          via `-(target_tensor * log_probs).sum_dim(1).mean()`, returning a 1D scalar tensor `Tensor<B, 1>` of shape `[1]`.
- **Rationale**:
    - Using `burn::tensor::activation::log_softmax(logits, 1)` applies the log-sum-exp trick internally ($\log \text{softmax}(z_c) = (z_c - \max_k z_k) - \ln \sum_j e^{z_j - \max_k z_k}$), guaranteeing that extreme logit magnitudes never underflow to $\ln(0) = -\infty$ or produce `NaN` gradients during backpropagation (`CHK028`).
    - Label smoothing ($\epsilon = 0.05$) prevents overconfident logit saturation on the majority `Unripe` class, ensuring well-calibrated Softmax probabilities for Entropy-weighted multi-tap voting.
- **Alternatives Considered**:
    - _Unweighted Cross-Entropy Loss_: Rejected because predicting `Unripe` for every fruit achieves ~60.9% accuracy on iPhone while yielding `0.0` recall on `Ripe` and `Overripe`.
    - _Computing `softmax(logits, 1).log()` manually_: Rejected because direct `.log()` on tiny Softmax probabilities can underflow to `-Infinity` in FP32.

---

## 4. Dual-Level Evaluation (`Tap-Level` & Entropy-Weighted `Recording-Level`) & Early Stopping (`src/training.rs`)

- **Decision**: Implement `ConfusionMatrix3x3`, `EvaluationReport`, `DualLevelEvaluation`, `evaluate_model<B: Backend>`, and `EarlyStopping` in `src/training.rs`:
    1. **`ConfusionMatrix3x3` & Zero-Denominator Guards (`FR-007` & `CHK010`, `CHK027`)**:
        - `counts: [[usize; 3]; 3]` indexed as `counts[actual][predicted]`.
        - For each class $c \in \{0, 1, 2\}$:
            - `precision[c] = if col_sum > 0 { tp / col_sum as f32 } else { 0.0 }`
            - `recall[c] = if row_sum > 0 { tp / row_sum as f32 } else { 0.0 }`
            - `f1[c] = if precision[c] + recall[c] > 0.0 { 2.0 * precision[c] * recall[c] / (precision[c] + recall[c]) } else { 0.0 }`
        - `macro_f1 = (f1[0] + f1[1] + f1[2]) / 3.0`
        - `balanced_accuracy = (recall[0] + recall[1] + recall[2]) / 3.0`
        - `overall_accuracy = if total_samples > 0 { total_correct as f32 / total_samples as f32 } else { 0.0 }`
        - `critical_fp_ripe_rate = if row_sum(0) + row_sum(2) > 0 { (counts[0][1] + counts[2][1]) as f32 / (row_sum(0) + row_sum(2)) as f32 } else { 0.0 }` (tracking both `FP_R(1): Unripe -> Ripe` and `FP_R(2): Overripe -> Ripe`).
    2. **Entropy-Weighted Recording-Level Aggregation (`FR-008`, `CHK011`, `CHK028`, `CHK037`)**:
        - Group all evaluated `TapSample` probability predictions $P_k \in [0.0, 1.0]^3$ by `sample_num` (using `BTreeMap<u32, (RipenessLabel, Vec<[f32; 3]>)>` for deterministic iteration).
        - For each recording with $M \ge 1$ taps, compute Shannon entropy for tap $k$ with probability floor `1e-12`:
          $$H(P_k) = -\sum_{c=0}^{2} P_k(c) \ln\left(\max(P_k(c), 10^{-12})\right)$$
        - Compute confidence weight $\alpha_k = \left(1.0 - \frac{H(P_k)}{\ln 3}\right).\text{clamp}(0.0, 1.0)$.
        - If $\sum_{k=1}^{M} \alpha_k > 10^{-6}$, compute $P_{\text{final}}(c) = \frac{\sum_{k=1}^{M} \alpha_k P_k(c)}{\sum_{k=1}^{M} \alpha_k}$; otherwise fall back to uniform arithmetic mean $P_{\text{final}}(c) = \frac{1}{M}\sum_{k=1}^{M} P_k(c)$.
        - Select $\hat{y}_{\text{rec}} = \arg\max_{c \in \{0, 1, 2\}} P_{\text{final}}(c)$ and record `(actual_label.as_index(), predicted_cls)` in the recording-level `ConfusionMatrix3x3`.
    3. **Macro-F1 Early Stopping & `.mpk` Checkpointing (`FR-009`, `CHK005`, `CHK012`, `CHK023`, `CHK029`, `CHK030`)**:
        - Initialize `best_macro_f1 = -1.0_f32` and `patience_counter = 0usize` so Epoch 1 (`val_recording_macro_f1 >= 0.0 > -1.0`) is guaranteed to save a valid `.mpk` checkpoint even in 1-epoch or zero-F1 integration tests.
        - Normalize the checkpoint path stem (stripping any trailing `.mpk` extension if provided, ensuring parent directory exists via `std::fs::create_dir_all`), and save via `model.clone().save_file(&checkpoint_stem, &NamedMpkFileRecorder::<FullPrecisionSettings>::new())`.
        - When `patience_counter >= config.training.early_stopping_patience` (`12`) or all epochs finish, reload the best checkpoint via `model.load_file(&checkpoint_stem, &NamedMpkFileRecorder::<FullPrecisionSettings>::new(), device)` and evaluate on the held-out `Test` split.
- **Rationale**:
    - Aligning Phase 3's `Recording-Level` aggregation formula with Phase 4's on-device Entropy-weighted multi-tap voting eliminates evaluation-serving discrepancy and selects the model checkpoint that performs best on whole durian fruits after suppressing noisy/glancing taps.
- **Alternatives Considered**:
    - _Early stopping on Training or Validation Cross-Entropy Loss_: Rejected because weighted loss can continue decreasing on majority samples while minority class F1 degrades.
    - _Simple majority vote across taps_: Rejected because a high-confidence resonant tap (`[0.02, 0.96, 0.02]`, $\alpha_k \approx 0.83$) should outweigh two ambiguous noisy taps (`[0.35, 0.31, 0.34]`, $\alpha_k \approx 0.001$).

---

## 5. Generic `<B: AutodiffBackend>` Training Loop, Default `Autodiff<NdArray>` CLI Backend & Benchmark Compatibility (`src/training.rs`, `src/main.rs`, `benches/training_benchmarks.rs`)

- **Decision**:
    1. Keep the training loop `train_platform_model<B: AutodiffBackend>` generic over `<B: AutodiffBackend>` using `AdamWConfig::new().with_weight_decay(config.training.weight_decay as f32).init()` and `GradientsParams::from_grads(loss.backward(), &model)`.
    2. Default the `train` CLI subcommand in `src/main.rs` and automated test suites to `burn::backend::Autodiff<burn::backend::NdArray<f32>>` (and `NdArray<f32>` for inference tests/benchmarks) with deterministic `StdRng::seed_from_u64(config.dataset.split_seed + epoch as u64)` epoch shuffling (`Clarifications`, `FR-009`, `FR-010`, `CHK020`, `CHK022`, `CHK024`).
    3. Preserve the existing `GradientDescent` and `LearningRateSchedule` structs in `src/training.rs` and update `benchmark_model_training` in `benches/training_benchmarks.rs` to benchmark real `DurianAudioCnn::<NdArray<f32>>::forward` execution (`CHK016`, `CHK026`, `CHK040`).
- **Rationale**:
    - `Autodiff<NdArray<f32>>` requires zero GPU drivers or shaders, executes deterministically across macOS, Linux, Windows, and headless CI containers, and still allows switching to `Autodiff<Wgpu>` via generic type parameter when GPU acceleration is desired.
    - Retaining `GradientDescent` and `LearningRateSchedule` preserves 100% backward compatibility with existing Phase 1/2 unit tests and Criterion benchmarks.
- **Alternatives Considered**:
    - _Hardcoding `Wgpu` in `src/training.rs` and `src/main.rs`_: Rejected because headless CI runners and some developer environments lack Vulkan/Metal/DX12 GPU adapters, causing non-deterministic or failing tests.
