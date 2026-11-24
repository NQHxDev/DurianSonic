<!--
SYNC IMPACT REPORT
-------------------
Version Change: [UNVERSIONED] → 1.0.0
Rationale: Initial constitution ratification with comprehensive coding standards

Added Principles:
- I. Clean Code Excellence
- II. Test-First Development
- III. Audio Quality Standards
- IV. Performance Requirements
- V. Documentation & Maintainability

Added Sections:
- Code Style & Formatting Standards
- Development Workflow & Quality Gates

Modified/Removed: None (initial version)

Follow-up TODOs: None
-------------------
-->

# DurianSonic Constitution

## Core Principles

### I. Clean Code Excellence (NON-NEGOTIABLE)

All code MUST adhere to Clean Code principles without exception:

- **No Bypass Patterns**: Code MUST NOT use workarounds, shortcuts, or bypass mechanisms that circumvent proper architecture or error handling. Every solution must address the root cause, not symptoms.
- **No Emojis**: The codebase MUST be entirely emoji-free. Use descriptive names and proper documentation instead of visual markers.
- **Clear Intent**: Every function, struct, and module must have a single, well-defined purpose. Names must be self-documenting.
- **Explicit Over Implicit**: Favor explicitness in types, error handling, and data flow. Magic behavior is prohibited.

**Rationale**: Clean code is maintainable code. Bypass patterns create technical debt that compounds over time. Emojis introduce encoding issues and reduce professionalism. Explicit code is debuggable code.

### II. Test-First Development (NON-NEGOTIABLE)

TDD methodology is strictly enforced:

- Tests MUST be written before implementation code
- Test approval from stakeholders MUST occur before implementation begins
- Red-Green-Refactor cycle: Tests fail → Implementation → Tests pass → Refactor
- Every public API must have corresponding unit tests
- Integration tests required for audio pipeline components and ML model interactions

**Rationale**: Tests define contracts before implementation, preventing architectural mistakes. Audio processing and ML systems are complex; tests are the safety net.

### III. Audio Quality Standards

Audio processing code must maintain signal integrity:

- Sample rate conversions MUST preserve frequency content within audible range
- DSP operations MUST document their effect on signal characteristics
- Numerical precision requirements must be explicitly stated and tested
- Audio artifacts (clipping, aliasing, quantization noise) must be monitored and minimized

**Rationale**: DurianSonic processes audio signals. Quality degradation is unacceptable and difficult to debug without proactive standards.

### IV. Performance Requirements

Performance is a first-class concern:

- Audio processing MUST run in real-time or near-real-time for target use cases
- Memory allocations in hot paths must be justified and documented
- Benchmarks required for critical audio/ML pipeline components using Criterion
- Performance regressions caught in CI before merge

**Rationale**: Audio and ML workloads are computationally intensive. Performance problems compound across the pipeline.

### V. Documentation & Maintainability

Code must be maintainable by others:

- **Concise Comments**: Comments MUST be short, precise, and explain _why_, not _what_. No block-style separators (===, ---, etc.). The code itself explains what; comments explain rationale.
- Public APIs require rustdoc with examples
- Complex algorithms require references to papers/sources
- Architecture decisions documented in `docs/` with rationale

**Rationale**: Future maintainers (including future you) need context, not verbosity. Comment noise reduces signal.

## Code Style & Formatting Standards

**Rust Formatting**: Use `rustfmt` with project defaults. No manual formatting debates.

**Comment Guidelines**:

- Avoid obvious comments: `// Increment counter` is noise
- Good comments explain non-obvious decisions: `// Using f32 for SIMD alignment`
- No decorative separators: `// ========` or `// ---------` are banned
- Keep comments under 80 characters where possible
- Use `//` line comments, never `/* ... */` block comments except for temporary multi-line disabling during debugging

**Naming Conventions**:

- `snake_case` for functions, variables, modules
- `PascalCase` for types, traits, enums
- `SCREAMING_SNAKE_CASE` for constants
- Descriptive names over abbreviations: `sample_rate` not `sr`

**Error Handling**:

- Use `Result<T, E>` for recoverable errors with explicit error types
- Use `anyhow::Result` only at application boundaries
- Never use `unwrap()` or `expect()` in library code without documented invariants
- Panics MUST be documented and justified

## Development Workflow & Quality Gates

**Pre-Implementation**:

1. Specification review and approval
2. Test cases written and reviewed
3. API contracts defined
4. Performance expectations established

**Implementation**:

1. Tests fail (Red)
2. Implement minimal code to pass tests (Green)
3. Refactor for clarity and performance (Refactor)
4. Benchmark critical paths

**Pre-Merge Quality Gates**:

- All tests pass (`cargo test`)
- No clippy warnings (`cargo clippy -- -D warnings`)
- Formatting verified (`cargo fmt --check`)
- Benchmarks run and reviewed for regressions
- Code review approved with explicit Clean Code compliance check
- Documentation complete for public APIs

## Governance

This constitution supersedes all other development practices and coding preferences.

**Amendment Process**:

- Amendments require documented rationale explaining the problem being solved
- Approval from project maintainer(s) required
- Migration plan for existing code violating new rules (if applicable)
- Version bump according to semantic versioning

**Compliance**:

- All code reviews MUST verify compliance with constitutional principles
- Violations require fix or explicit justification and tracking for future remediation
- Complexity that violates simplicity principles must be justified with performance data or architectural necessity

**Semantic Versioning for Constitution**:

- MAJOR: Backward-incompatible principle changes or removals
- MINOR: New principles added or expanded guidance
- PATCH: Clarifications, wording improvements, typo fixes

**Version**: 1.0.0 | **Ratified**: 2026-11-24 | **Last Amended**: 2026-11-24
