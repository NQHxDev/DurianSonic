# Quickstart Validation Guide: Phase 1 - Data Pipeline, Config Expansion & Anti-Leakage Dataset Splitter

**Feature**: `001-phase1-data-config`
**Date**: 2026-11-24

---

## 1. Prerequisites

- **Rust Toolchain**: `rustc` and `cargo` (Edition 2021)
- **Dataset Directory (`data/`)**:
    - `data/metadata_iphone.csv` (`281` rows) + `data/sound_iphone/*.m4a` (`281` files)
    - `data/metadata_android.csv` (`175` rows) + `data/sound_android/*.wav` (`175` files)
- **Configuration File**: `configs/default.toml`

---

## 2. Step-by-Step Validation Commands

### Step 1: Run Unit & Integration Tests

Verify configuration parsing, physical/mathematical constraint checks, metadata verification, SHA-256 deduplication, and zero-leakage stratified group splitting:

```bash
cargo test
```

**Expected Outcome**:

- All unit tests in `src/config.rs`, `src/dataset.rs`, `src/training.rs`, and `src/lib.rs` pass (`0` failures).
- Specific Phase 1 test coverage includes:
    - Full 4-section TOML loading and backward-compatible partial TOML loading.
    - Rejection of invalid Nyquist `fmax`, non-power-of-two `n_fft`, and invalid split ratio sums.
    - Ripeness label parsing (`Unripe = 0`, `Ripe = 1`, `Overripe = 2`) and class weight computation.
    - Detection of duplicate SHA-256 binary content, byte-size mismatches, and `sample_num` gaps.
    - Zero `fruit_id` overlap across `Train`, `Validation`, and `Test` splits (`Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅`) and 100% deterministic output for `split_seed = 42`.

---

### Step 2: Run Full Dataset Integrity & Anti-Leakage CLI Audit

Execute the `verify-dataset` command against the real `/data/` directory:

```bash
cargo run -- verify-dataset --data-dir data --config configs/default.toml
```

**Expected Outcome** (completes in `< 10 seconds`):

- **iPhone Dataset (`metadata_iphone.csv`)**:
    - Verified files: `281` (`0` duplicate SHA-256 hashes)
    - Unique physical fruits: `175` (`fruit_id` `1..=175`)
    - Class distribution: `171 Unripe, 66 Ripe, 44 Overripe`
    - Class weights: `[0.548, 1.419, 2.129]`
    - Stratified Group Split (`seed = 42`): `123` Train fruits (`197` files), `26` Val fruits (`41` files), `26` Test fruits (`43` files), `Overlap fruit_ids = 0`
- **Android Dataset (`metadata_android.csv`)**:
    - Verified files: `175` (`0` duplicate SHA-256 hashes)
    - Unique physical fruits: `175` (`fruit_id` `1..=175`)
    - Class distribution: `95 Unripe, 38 Ripe, 42 Overripe`
    - Class weights: `[0.614, 1.535, 1.389]`
    - Stratified Group Split (`seed = 42`): `123` Train fruits (`123` files), `26` Val fruits (`26` files), `26` Test fruits (`26` files), `Overlap fruit_ids = 0`

---

### Step 3: Verify Constitution Quality Gates

Run strict compiler linting and formatting checks required by `constitution.md`:

```bash
cargo clippy --all-targets -- -D warnings && cargo fmt --check
```

**Expected Outcome**:

- `0` compiler or clippy warnings.
- `0` formatting diffs.
- Codebase remains 100% emoji-free and adheres to Clean Code guidelines.
