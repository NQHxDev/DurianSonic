# Burn 0.22.1 Serialization API Blocker - RESOLVED ✅

## Issue (RESOLVED on 2026-11-26)

~~Cannot find the correct module path for model serialization APIs in Burn 0.22.1~~

**SOLUTION**: Burn 0.22.1 migrated from `burn::record` to `burn::store` with new API.

## Root Cause

Burn 0.22.1 introduced breaking changes:

- **Old API** (deprecated): `burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder}`
- **New API** (Burn 0.22.1): `burn::store::{BurnpackStore, ModuleSnapshot}`

The `store` feature was not enabled in Cargo.toml, and code was using deprecated API.

## Solution Applied

### 1. Enable `store` feature in Cargo.toml

```toml
burn = { version = "0.22.1", features = ["train", "wgpu", "ndarray", "autodiff", "metrics", "store"] }
```

### 2. Update model.rs API

- **Replaced**: `.mpk` format → `.burnpack` format (native Burn format)
- **Methods**: `save_mpk()` / `load_mpk()` → `save_burnpack()` / `load_burnpack()`
- **API**: Uses `BurnpackStore` and `ModuleSnapshot` trait methods `save_into()` / `load_from()`

### 3. Update training.rs checkpoint logic

- Checkpoint saving now uses `model.save_burnpack(path)`
- Checkpoint loading uses `model.load_burnpack(path)`
- File size tracking works correctly

## Impact - ALL RESOLVED ✅

- ✅ **T014**: `DurianAudioCnn::save_burnpack()` and `load_burnpack()` implemented
- ✅ **T021**: `train_model_on_splits()` checkpoint save/load working
- ✅ **T018**: Integration test `test_train_model_on_splits_end_to_end_checkpoint_restore` now passes
- ✅ **T023**: Artifact saving unblocked

## Verification

```bash
# Test passes (previously ignored)
cargo test test_train_model_on_splits_end_to_end_checkpoint_restore --lib
# Result: test training::tests::test_train_model_on_splits_end_to_end_checkpoint_restore ... ok

# No ignored tests remaining
cargo test --lib -- --ignored --list
# Result: 0 tests, 0 benchmarks
```

## References

- Burn 0.22.1 Store API: `burn::store::{BurnpackStore, SafetensorsStore, ModuleSnapshot}`
- Burn Book: [Model Storage](https://burn.dev/book/)
- Cargo.toml: `burn = { version = "0.22.1", features = ["store"] }`

## Migration Notes

For future reference, Burn 0.22.1 store API patterns:

```rust
use burn::store::{BurnpackStore, ModuleSnapshot};

// Save
let mut store = BurnpackStore::from_file("model.burnpack");
model.save_into(&mut store)?;

// Load
let mut store = BurnpackStore::from_file("model.burnpack");
model.load_from(&mut store)?;
```
