# Git Hooks

This directory contains git hooks for the DurianSonic project.

## Available Hooks

### pre-commit (Lightweight checks)

Runs before each commit to ensure code quality:

- Code formatting (`cargo fmt --check`)
- Lint checks (`cargo clippy`)
- Common issues detection (debug prints, TODOs)
- Git LFS auto-tracking for `.json` and `.md` files >= 500KB

### pre-push (Comprehensive checks)

Runs before pushing to ensure production readiness:

- Debug build (`cargo build`)
- Release build (`cargo build --release`)
- All tests (`cargo test`)
- Doc tests (`cargo test --doc`)
- Documentation (`cargo doc`)
- Benchmarks compilation (if benches/ exists)
- Git LFS push object verification

### post-checkout, post-commit, post-merge

- Git LFS state synchronization across git operations

## Installation

Run the installation script:

```bash
./.githooks/install.sh
```

Or manually:

````bash
chmod +x .githooks/*
git config core.hooksPath .githooks
git lfs install --skip-repo

## Usage

### Normal workflow

Hooks run automatically:

```bash
git commit -m "feat: add new feature"  # Runs pre-commit
git push                                # Runs pre-push
````

### Skip hooks temporarily

Use `--no-verify` (or `-n`) to bypass checks:

```bash
git commit -m "WIP: work in progress" --no-verify
git push --no-verify
```

### Run hooks manually

Test hooks before committing/pushing:

```bash
./.githooks/pre-commit
./.githooks/pre-push
```

## Requirements

- Rust toolchain (cargo, rustfmt, clippy)
- Optional: cargo-udeps for unused dependency checks

## Troubleshooting

### Hooks not running

Check if hooks path is configured:

```bash
git config core.hooksPath
# Should output: .githooks
```

### Permission denied

Make sure hooks are executable:

```bash
chmod +x .githooks/pre-commit .githooks/pre-push
```

### Hooks too slow

- Pre-commit is designed to be fast (< 30 seconds)
- Pre-push is comprehensive and may take longer (1-5 minutes)
- Use `--no-verify` for WIP commits if needed

## Customization

Edit the hook scripts directly in `.githooks/` to:

- Add more checks
- Remove unnecessary checks
- Adjust strictness levels
- Add project-specific validations

## CI/CD Integration

These hooks mirror CI checks locally, helping catch issues early:

- Faster feedback than waiting for CI
- Reduced failed CI builds
- Better development experience
