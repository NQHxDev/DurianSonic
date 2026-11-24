# Phase 0 Research: Acoustic DSP, Impulse Onset Detection & Feature Extraction

**Feature**: `002-phase2-dsp-augmentation`
**Date**: 2026-11-24
**Status**: Complete (All technical decisions resolved)

---

## 1. Multi-Format Audio Decoding & Mono Downmixing with `symphonia 0.6.1` (`src/dsp.rs`)

- **Decision**: Implement `decode_audio_to_mono(path: &Path, target_sr: u32) -> anyhow::Result<Vec<f32>>` using the locked `symphonia = "0.6.1"` API:
    1. Validate `target_sr > 0` upfront; return an explicit error if `target_sr == 0`.
    2. Open the file and wrap it in `MediaSourceStream::new(Box::new(file), Default::default())` with an extension `Hint` (`symphonia::core::formats::probe::Hint`).
    3. Probe the container via `symphonia::default::get_probe().probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())`.
    4. Select the default audio track via `format.default_track(TrackType::Audio)` and extract `AudioCodecParameters` from `track.codec_params` (`CodecParameters::Audio(params)`), verifying `sample_rate` is present and `> 0`.
    5. Instantiate the codec decoder via `symphonia::default::get_codecs().make_audio_decoder(&audio_params, &AudioDecoderOptions::default())`.
    6. Read packets sequentially with `while let Some(packet) = format.next_packet()?`, filtering by `packet.track_id() == track_id`. Propagate any packet read or decode error immediately (`decoder.decode(&packet)?`).
    7. Copy decoded samples into a reusable `Vec<f32>` buffer via `decoded.copy_to_vec_interleaved(&mut interleaved_buf)` and downmix multi-channel frames to Mono by taking the arithmetic mean across channels:
       $$x_{\text{mono}}[n] = \frac{1}{C}\sum_{c=0}^{C-1} x_c[n]$$
    8. Reject empty decoded streams (`mono_samples.is_empty()`), return `Ok(mono_samples)` directly when `src_sr == target_sr`, or delegate to `resample_mono(&mono_samples, src_sr, target_sr)`.
- **Rationale**:
    - In `symphonia 0.6.1`, `Probe::format` (from 0.5) was replaced by `Probe::probe` returning `Result<Box<dyn FormatReader>>`, `FormatReader::default_track` takes `TrackType::Audio`, `FormatReader::next_packet()` returns `Result<Option<Packet>>` (where `Ok(None)` cleanly signals EOF without matching `IoErrorKind::UnexpectedEof`), and `GenericAudioBufferRef::copy_to_vec_interleaved` handles sample format conversion (`S16` from Android `.wav`, `F32` from iPhone `.m4a` AAC-LC) into a reusable buffer with minimal allocations.
- **Alternatives Considered**:
    - _Using external C libraries (`ffmpeg`, `libsndfile`)_: Rejected because pure-Rust `symphonia` guarantees cross-platform reproducibility and seamless mobile cross-compilation (`aarch64-apple-ios`, `aarch64-linux-android`) with zero C toolchain dependencies.
    - _Ignoring packet decode errors inside the loop_: Rejected by Constitution Principle I (No Bypass Patterns) and `FR-001`, which require failing fast on corrupted audio streams.

---

## 2. Synchronous FFT Resampling with `rubato 5.1.0` (`src/dsp.rs`)

- **Decision**: Implement `resample_mono(samples: &[f32], src_sr: u32, target_sr: u32) -> anyhow::Result<Vec<f32>>` using `rubato = "5.1.0"`:
    1. Validate `!samples.is_empty()`, `src_sr > 0`, and `target_sr > 0`.
    2. If `src_sr == target_sr`, return `Ok(samples.to_vec())` without invoking the resampler.
    3. Construct a synchronous FFT resampler `Fft::<f32>::new(src_sr as usize, target_sr as usize, 1024, 1, FixedSync::Both)?`.
    4. Wrap `samples` in `rubato::audioadapter_buffers::direct::InterleavedSlice::new(samples, 1, samples.len())?` and invoke `resampler.process_all(&input_adapter, samples.len(), None)?`.
    5. Extract the owned `Vec<f32>` via `.take_data()` and adjust length to the exact mathematical target `expected_len = ((samples.len() as f64) * (target_sr as f64) / (src_sr as f64)).round() as usize` (truncating or zero-padding if differed by 1 boundary sample).
- **Rationale**:
    - In `rubato 5.1.0`, `FftFixedIn` (from `0.14`) was unified into `Fft<T>` with `FixedSync::Both` and `audioadapter` buffer traits (re-exported as `rubato::audioadapter_buffers`).
    - Calling `Resampler::process_all` automatically chunks the input, resets internal state, and trims the FFT anti-aliasing filter's startup delay (`output_delay()`), ensuring transient tap onsets are not shifted in time and preserving frequency content up to the `22,050 Hz` Nyquist limit (`11,025 Hz`), satisfying Constitution Principle III (Audio Quality Standards).
- **Alternatives Considered**:
    - _Linear or polynomial interpolation without anti-aliasing_: Rejected because downsampling from `48,000 Hz` or `44,100 Hz` to `22,050 Hz` without an anti-aliasing low-pass filter folds high-frequency noise into the `[50, 8000] Hz` acoustic resonance band (violating Constitution Principle III).
    - _Manual chunk loop with `Resampler::process` without trimming `output_delay()`_: Rejected because `rubato` documentation explicitly warns that manual `process()` leaves leading filter delay silence in the output.

---

## 3. Short-Time RMS Onset Detection & `0.8s` Tap Windowing (`src/dsp.rs`)

- **Decision**: Separate onset detection into `detect_onset_centers(samples: &[f32], cfg: &ImpulseConfig) -> Vec<usize>` and window slicing into `extract_tap_impulses(samples: &[f32], cfg: &ImpulseConfig) -> Vec<Vec<f32>>`:
    - Compute derived sample counts: `win_len = (cfg.window_sec * cfg.target_sr as f32).round() as usize` (`17,640`), `pre_len = (cfg.pre_onset_sec * cfg.target_sr as f32).round() as usize` (`2,205`), and `min_gap = (cfg.min_interval_sec * cfg.target_sr as f32).round() as usize` (`8,820`).
    - Handle short/empty inputs consistently (`FR-004` & Clarifications):
        - If `samples.is_empty()` (or `win_len == 0`): both `detect_onset_centers` and `extract_tap_impulses` return `Vec::new()`.
        - If `0 < samples.len() < win_len`: `detect_onset_centers` returns `vec![0]`, and `extract_tap_impulses` zero-pads `samples` to `win_len` (`17,640`) and returns `vec![normalize_peak(&padded)]`.
    - For `samples.len() >= win_len`:
        1. Partition `samples` into non-overlapping frames of `env_frame = 256` samples (~`11.61 ms` at `22,050 Hz`) and compute $E_{\text{rms}}[i] = \sqrt{\frac{1}{N_i}\sum_{x \in \text{chunk}_i} x^2}$.
        2. Compute global adaptive threshold $\theta = \max(\max_i E_{\text{rms}}[i] \times \text{threshold\_ratio}, 10^{-4})$.
        3. Scan interior frames $i \in [1, L_{\text{rms}} - 2]$ for local peaks satisfying $E_{\text{rms}}[i] \ge \theta \land E_{\text{rms}}[i] \ge E_{\text{rms}}[i-1] \land E_{\text{rms}}[i] > E_{\text{rms}}[i+1]$.
        4. Map peak frame $i$ to `center_sample = i * 256`, enforce `center_sample.saturating_sub(prev_center) >= min_gap`, and require `start + win_len <= samples.len()` where `start = center_sample.saturating_sub(pre_len)`.
        5. `extract_tap_impulses` iterates over `detect_onset_centers(samples, cfg)` and extracts `normalize_peak(&samples[start..start + win_len])`.
- **Rationale**:
    - Reusing `detect_onset_centers` directly inside `extract_tap_impulses` guarantees 100% consistency between validation/inference window extraction and training-time `apply_onset_jitter` (which jitters around `center_sample` returned by `detect_onset_centers`).
    - The `-100 ms` (`2,205` samples) pre-onset lead-in captures the full mechanical impact rise, while the `+700 ms` (`15,435` samples) tail captures the hull resonance decay, and the `0.4s` (`8,820` samples) refractory period suppresses secondary acoustic reflections.
- **Alternatives Considered**:
    - _Spectral flux onset detection_: Rejected because percussive knock transients on durian rinds exhibit sharp amplitude ratio jumps in the time-domain RMS envelope, which runs in `< 0.1 ms` per 30s file without requiring full-file STFT computation.

---

## 4. Single-Pass STFT Power Spectrogram, Log-Mel (`128 x 128`) & 4 Physics Features (`src/dsp.rs`)

- **Decision**: Factor the time-frequency pipeline so the `(513, 128)` STFT power spectrogram is computed once per tap and shared between Log-Mel projection and Physics Feature extraction:
    1. `compute_power_spectrogram(audio: &[f32], cfg: &SpectrogramConfig) -> Array2<f32>`:
        - Uses `RealFftPlanner::<f32>::new().plan_fft_forward(cfg.n_fft)` (`n_fft = 1024`, `n_freqs = 513`) with a periodic/symmetric Hann window $w[n] = 0.5 \cdot (1 - \cos(2\pi n / N_{\text{fft}}))$ over `target_frames = 128` frames (`hop_length = 128`), zero-padding out-of-bounds sample indices via `audio.get(start + i).copied().unwrap_or(0.0)`, and storing $P(k, t) = |X(k, t)|^2 = \text{re}^2 + \text{im}^2$ in shape `(513, 128)`.
    2. `compute_log_mel_spectrogram(audio: &[f32], cfg: &SpectrogramConfig, mel_basis: &Array2<f32>) -> Array2<f32>`:
        - Computes `power_spec = compute_power_spectrogram(audio, cfg)` (or shares via internal helper), multiplies `mel_spec = mel_basis.dot(&power_spec)` (`(128, 513) x (513, 128) -> (128, 128)`), references against window peak power $P_{\max} = \max(\max_{m,t} S_{\text{mel}}(m, t), 10^{-10})$, converts to decibels $10\log_{10}(\max(p, 10^{-10}) / P_{\max})$, clamps to `[-top_db, 0.0]`, and maps linearly to `[0.0, 1.0]`.
    3. `extract_physics_features_from_power_spec(power_spec: &Array2<f32>, cfg: &SpectrogramConfig) -> [f32; 4]` & `extract_physics_features(audio: &[f32], cfg: &SpectrogramConfig) -> [f32; 4]`:
        - If total power across `power_spec` is `<= 1e-10` (or `power_spec` is empty), return `[0.0, 0.0, 0.0, 0.0]`.
        - Use the primary impact window of the first `64` STFT frames (`t in 0..64`, clamped to `n_frames`) for the 3 spectral features:
            - **Peak Resonance Frequency ($\tilde{f}_{\text{peak}}$)**: Maximize mean power $P_{\text{mean}}(k) = \frac{1}{64}\sum_{t=0}^{63} P(k, t)$ over bins $k$ satisfying $80.0 \le f_k \le 1200.0\text{ Hz}$ ($f_k = k \cdot f_s / N_{\text{fft}}$). If max power in band is `<= 0.0`, return `0.0`; else $\text{clamp}((f_{\text{peak}} - 80.0) / 1120.0, 0.0, 1.0)$.
            - **Spectral Centroid ($\tilde{C}_s$)**: Compute mean magnitude $|X_{\text{mean}}(k)| = \frac{1}{64}\sum_{t=0}^{63} \sqrt{P(k, t)}$ across all frequency bins $k \in [0, n_{\text{freqs}})$, $C_s = \frac{\sum_k f_k |X_{\text{mean}}(k)|}{\sum_k |X_{\text{mean}}(k)| + 10^{-8}}$, normalized via $\text{clamp}(C_s / 4000.0, 0.0, 1.0)$.
            - **Spectral Roll-off 85% ($\tilde{R}_{85}$)**: Using $P_{\text{mean}}(k)$ on `t in 0..64`, find the smallest bin $K$ where cumulative power $\sum_{k=0}^{K} P_{\text{mean}}(k) \ge 0.85 \sum_{k=0}^{n_{\text{freqs}}-1} P_{\text{mean}}(k)$, normalized via $\text{clamp}(f_K / 6000.0, 0.0, 1.0)$.
        - Use the inclusive decay frame range `t in 26..=86` (`61` frames, corresponding to ~`50 ms` to ~`400 ms` after the `0.1s` onset peak) for **Temporal Decay Slope ($\tilde{\beta}_{\text{decay}}$)**:
            - Compute frame energy in dB: $E_{\text{dB}}(t) = 10 \log_{10}\left(\sum_{k} P(k, t) + 10^{-10}\right)$.
            - Compute the ordinary least-squares (OLS) linear regression slope $\beta_{\text{decay}} = \frac{\sum (t - \bar{t})(E_{\text{dB}}(t) - \bar{E})}{\sum (t - \bar{t})^2 + 10^{-8}}$ (in dB/frame) and normalize via $\text{clamp}(-\beta_{\text{decay}} / 2.0, 0.0, 1.0)$.
    4. `extract_tap_features(audio: &[f32], cfg: &SpectrogramConfig, mel_basis: &Array2<f32>) -> TapFeatures`:
        - Computes `power_spec` once, projects `mel_spec` with `mel_basis`, computes `physics_features` from `power_spec`, and returns `TapFeatures { mel_spec, physics_features }`.
- **Rationale**:
    - Computing the 128-frame 1024-point RealFFT once per tap instead of twice cuts STFT time by 50%, ensuring `extract_tap_features` easily runs in `< 1.0–2.0 ms` (well below the `< 4.0 ms` target in `SC-005`).
- **Alternatives Considered**:
    - _Recomputing STFT inside both `compute_log_mel_spectrogram` and `extract_physics_features` during batch extraction_: Rejected because it doubles FFT overhead on mobile and training hot paths.

---

## 5. Physically Valid Data Augmentation & `rand 0.10.4` Guards (`src/augment.rs`)

- **Decision**: Implement 4 physically valid augmentation functions in `src/augment.rs` using `rand = "0.10.4"` (`use rand::{rng, RngExt};`):
    1. `apply_onset_jitter(full_samples: &[f32], center_sample: usize, max_jitter_samples: usize, cfg: &ImpulseConfig) -> Vec<f32>`:
        - Compute `win_len` and `pre_len` from `cfg`. If `win_len == 0`, return `Vec::new()`.
        - Sample jitter offset $\delta \in [-\text{max\_jitter}, +\text{max\_jitter}]$ via `rng().random_range(-(max_jitter as isize)..=(max_jitter as isize))` when `max_jitter_samples > 0`, else `0`.
        - Compute jittered center `(center_sample as isize + delta).max(0) as usize` and `raw_start = jittered_center.saturating_sub(pre_len)`.
        - If `full_samples.len() < win_len`, copy from `raw_start.min(full_samples.len())`, zero-pad to `win_len`, and call `normalize_peak`. Otherwise clamp `start = raw_start.min(full_samples.len() - win_len)` and return `normalize_peak(&full_samples[start..start + win_len])`.
    2. `add_background_noise(window: &[f32], min_snr_db: f32, max_snr_db: f32) -> Vec<f32>`:
        - If `window.is_empty()`, return `Vec::new()`.
        - Order bounds `(lo, hi) = (min_snr_db.min(max_snr_db), min_snr_db.max(max_snr_db))`; sample `snr_db = if (hi - lo).abs() < 1e-6 { lo } else { rng.random_range(lo..=hi) }`.
        - Compute signal power $P_{\text{sig}} = \frac{1}{N}\sum x_i^2$. If $P_{\text{sig}} \le 10^{-12}$, return `normalize_peak(window)` safely.
        - Compute $\sigma_{\text{noise}} = \sqrt{P_{\text{sig}} \cdot 10^{-\text{snr\_db} / 10.0}}$.
        - Generate standard Gaussian white noise $z \sim \mathcal{N}(0, 1)$ using the Box-Muller transform ($u_1 \in [10^{-7}, 1.0)$, $u_2 \in [0.0, 2\pi)$, $z = \sqrt{-2 \ln u_1} \cos(u_2)$), add $\sigma_{\text{noise}} \cdot z$ to each sample, and call `normalize_peak`.
    3. `apply_spectral_tilt(mel_spec: &mut Array2<f32>, max_tilt_db: f32, top_db: f32)`:
        - Guard degenerate inputs: if `max_tilt_db == 0.0 || !max_tilt_db.is_finite() || top_db <= 0.0 || mel_spec.nrows() <= 1`, clamp existing elements to `[0.0, 1.0]` and return.
        - Let `bound = max_tilt_db.abs()`; sample `tilt_db = rng.random_range(-bound..=bound)`.
        - For each Mel row $m \in [0, M-1]$, add $\Delta_m = \frac{\text{tilt\_db}}{\text{top\_db}} \cdot \left(\frac{2m}{M - 1} - 1\right)$ to every frame in row $m$ and clamp to `[0.0, 1.0]`.
    4. `apply_spec_augment(mel_spec: &mut Array2<f32>, max_freq_mask: usize, max_time_mask: usize, noise_floor_val: f32)`:
        - Let `(n_mels, n_frames) = mel_spec.dim()`.
        - Frequency masking: let `f_bound = max_freq_mask.min(n_mels / 4)`. If `f_bound > 0`, sample `f_mask = rng.random_range(0..=f_bound)`; if `f_mask > 0 && f_mask < n_mels`, sample `f0 = rng.random_range(0..(n_mels - f_mask))` and set `mel_spec[[f, t]] = noise_floor_val` for `f in f0..(f0 + f_mask)`.
        - Time masking: let `t_bound = max_time_mask.min(n_frames / 4)`. If `t_bound > 0`, sample `t_mask = rng.random_range(0..=t_bound)`; if `t_mask > 0 && t_mask < n_frames`, sample `t0 = rng.random_range(0..(n_frames - t_mask))` and set `mel_spec[[f, t]] = noise_floor_val` for `t in t0..(t0 + t_mask)`.
- **Rationale**:
    - Using Box-Muller directly on `rand 0.10.4` produces true Gaussian white noise $\mathcal{N}(0, \sigma_{\text{noise}}^2)$ without adding a new crate to `Cargo.toml`.
    - Explicit range guards ensure `rng.random_range(...)` is never called with an empty or inverted range, eliminating all panic vectors in library code (Constitution Principle I).
- **Alternatives Considered**:
    - _Adding `rand_distr` crate to `Cargo.toml`_: Rejected because `Cargo.toml` dependencies are already locked and Box-Muller takes 5 lines of standard Rust math.
    - _Pitch Shifting / Time Stretching_: Strictly forbidden because they alter cavity resonance frequency $f_r$ and temporal damping slope $\beta_{\text{decay}}$, corrupting ground-truth ripeness labels.
