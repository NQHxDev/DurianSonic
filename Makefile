.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  make run               - Run the project"
	@echo "  make build             - Build debug version"
	@echo "  make release           - Build optimized release"
	@echo "  make watch             - Run with auto-reload"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test              - Run tests"
	@echo "  make test-all          - Run tests with output"
	@echo "  make verify-data       - Verify dataset"
	@echo "  make bench             - Run benchmarks"
	@echo "  make bench-save        - Save benchmark baseline"
	@echo "  make coverage          - Generate code coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make check             - Check code without building"
	@echo "  make clippy            - Run clippy linter"
	@echo "  make fmt               - Format code"
	@echo "  make fmt-check         - Check formatting"
	@echo "  make ci                - Run all CI checks locally"
	@echo ""
	@echo "Documentation:"
	@echo "  make doc               - Generate documentation"
	@echo "  make doc-open          - Generate and open docs"
	@echo ""
	@echo "Utilities:"
	@echo "  make install           - Install binary"
	@echo "  make install-hooks     - Install git hooks"
	@echo "  make uninstall         - Uninstall binary"
	@echo "  make clean             - Clean build artifacts"
	@echo "  make all               - Format, lint, test, and build"

.PHONY: run
run:
	cargo run

.PHONY: build
build:
	cargo build

.PHONY: release
release:
	cargo build --release

.PHONY: test
test:
	cargo test

.PHONY: test-all
test-all:
	cargo test -- --nocapture --test-threads=1

.PHONY: verify-data
verify-data:
	cargo run -- verify-dataset --config configs/default.toml --data-dir data

.PHONY: bench
bench:
	cargo bench

.PHONY: bench-save
bench-save:
	cargo bench -- --save-baseline main

.PHONY: coverage
coverage:
	cargo install cargo-llvm-cov --locked || true
	cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
	cargo llvm-cov report --html
	@echo "Coverage report generated in target/llvm-cov/html/index.html"

.PHONY: check
check:
	cargo check --all-targets

.PHONY: clippy
clippy:
	cargo clippy --all-targets -- -D warnings

.PHONY: fmt
fmt:
	cargo fmt --all

.PHONY: fmt-check
fmt-check:
	cargo fmt --all -- --check

.PHONY: ci
ci: fmt-check clippy test bench
	@echo "✓ All CI checks passed"

.PHONY: doc
doc:
	cargo doc --no-deps --document-private-items

.PHONY: doc-open
doc-open:
	cargo doc --no-deps --document-private-items --open

.PHONY: install
install:
	cargo install --path .

.PHONY: install-hooks
install-hooks:
	./.githooks/install.sh

.PHONY: uninstall
uninstall:
	cargo uninstall durian-sonic

.PHONY: clean
clean:
	cargo clean
	rm -f lcov.info
	rm -rf target/llvm-cov

.PHONY: watch
watch:
	cargo watch -x run

.PHONY: all
all: fmt clippy test build
