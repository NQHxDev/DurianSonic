# Checklist Review Report: Configuration & Physical/Mathematical Contracts

**Reviewer**: Claude Code
**Date**: 2026-11-24
**Spec**: 001-phase1-data-config

## Summary

| Category                        | Total Items | Passed | Failed/Unclear | Pass Rate |
| ------------------------------- | ----------- | ------ | -------------- | --------- |
| Requirement Completeness        | 6           | 6      | 0              | 100%      |
| Requirement Clarity             | 7           | 6      | 1              | 86%       |
| Requirement Consistency         | 4           | 3      | 1              | 75%       |
| Acceptance Criteria Quality     | 3           | 2      | 1              | 67%       |
| Scenario & Edge Case Coverage   | 8           | 4      | 4              | 50%       |
| Non-Functional & Constitutional | 4           | 4      | 0              | 100%      |
| Dependencies & Assumptions      | 4           | 2      | 2              | 50%       |
| **TOTAL**                       | **36**      | **27** | **9**          | **75%**   |

## Items Requiring Clarification (Unchecked)

### CHK013 - Activation Function Case Sensitivity [Clarity]

**Issue**: The enumeration `["relu", "sigmoid", "tanh", "leaky_relu"]` is specified, but case-sensitivity rules are not defined.
**Question**: Should "ReLU", "RELU", "Relu" be accepted and normalized, or strictly rejected?
**Location**: Data-Model §2.5

### CHK015 - Mel-Spectrogram to Model Dimension Alignment [Consistency]

**Issue**: Mathematical relationship between DSP parameters and model input dimensions is not explicitly verified.
**Analysis**:

- `n_mels = 128` ↔ `input_height = 128` ✓ (aligned)
- Time dimension: `window_sec = 0.8s` at `sample_rate = 22050` → `17640` samples → `17640 / hop_length(128)` ≈ `137.8` frames, but `input_width = 128`
  **Question**: Is there a resampling/padding/truncation step? Should this mathematical relationship be explicitly stated?
  **Location**: Data-Model §2.2, §2.5

### CHK020 - Round-Trip Serialization Preservation [Measurability]

**Issue**: No explicit requirement that `save_to_file()` followed by `from_file()` preserves all fields exactly.
**Gap**: Implied by FR-001 but not stated as a testable criterion.
**Recommendation**: Add acceptance criterion: "Config round-trip serialization (save → load) must preserve 100% of field values."

### CHK022 - Whitespace-Only String Validation [Edge Case]

**Issue**: "Non-empty string" validation is specified, but whitespace-only strings (`"   "`) are not explicitly addressed.
**Question**: Should `data_dir = "   "` be rejected? Should strings be trimmed before validation?
**Location**: Data-Model §2.3, §2.5

### CHK024 - Floating-Point Special Values [Edge Case]

**Issue**: No explicit handling for `NaN`, `INFINITY`, `-INFINITY` in f32/f64 fields.
**Gap**: Validation constraints like `> 0.0` don't explicitly state whether `NaN` or `INFINITY` should be rejected.
**Recommendation**: Add validation requirement: "All floating-point fields must reject NaN and ±INFINITY with ValidationError."

### CHK025 - Config::from_file Validation Timing [Exception Flow]

**Issue**: Unclear whether `from_file()` automatically calls `validate()` or if callers must invoke it separately.
**Ambiguity**: Contract §1.1 shows them as separate methods; FR-003 says "check before using" but doesn't specify the entry point.
**Question**: Does `from_file()` return validated Config or raw parsed Config?

### CHK028 - Non-Existent Config File Fallback [Exception Flow]

**Issue**: Fallback behavior is specified for default path, but not for custom `--config` pointing to a missing file.
**Gap**: Contract §2.3 mentions "falling back to Config::default()" but conditions are unclear.
**Question**: Should missing custom config file cause error or fall back to defaults?

### CHK034 - Metadata Path Resolution Rules [Ambiguity]

**Issue**: Relationship between `iphone_metadata`/`android_metadata` paths and `data_dir` is ambiguous.
**Analysis**:

- Defaults: `iphone_metadata = "data/metadata_iphone.csv"`, `data_dir = "data"`
- PlatformTarget::metadata_csv_path expects `<data_dir>/metadata_iphone.csv`
  **Question**: Are metadata fields absolute paths or relative to data_dir? What happens when --data-dir overrides data_dir?

### CHK035 - Unknown TOML Keys Handling [Ambiguity]

**Issue**: No requirement specifying whether unknown keys in TOML should be ignored (Serde default) or rejected (via `#[serde(deny_unknown_fields)]`).
**Question**: Is strict schema enforcement required, or should extra keys be silently ignored for forward compatibility?

## Detailed Analysis by Category

### Requirement Completeness (6/6 PASS)

All structural requirements are explicitly defined:

- **CHK001**: All 4 config domains with complete field lists specified
- **CHK002**: DspConfig (11) and DatasetConfig (7) defaults documented
- **CHK003**: TrainingConfig new fields with defaults and `#[serde(default)]`
- **CHK004**: ModelConfig new fields with defaults and `#[serde(default)]`
- **CHK005**: Config lifecycle methods (from_file, load_default, validate, save_to_file) in contract
- **CHK006**: ConfigError variants (IoError, ParseError, ValidationError) with payload types

### Requirement Clarity (6/7 - 86%)

Mathematical and physical constraints are mostly unambiguous:

- **CHK007**: PASS - All DspConfig inequalities explicitly specified
- **CHK008**: PASS - Nyquist constraint mathematically precise (`<= sample_rate/2.0`)
- **CHK009**: PASS - Power-of-two constraint explicit (`n_fft.is_power_of_two()`)
- **CHK010**: PASS - Onset window relationship clear (`pre_onset_sec < window_sec`)
- **CHK011**: PASS - Split ratio tolerance quantified (`< 1e-4`)
- **CHK012**: PASS - Type-specific bounds clear (f32 vs f64 distinguished)
- **CHK013**: FAIL - Activation function case-sensitivity NOT specified

### Requirement Consistency (3/4 - 75%)

- **CHK014**: PASS - Legacy default value conflicts reconciled explicitly
- **CHK015**: FAIL - Mel-spectrogram to model dimension math not verified
- **CHK016**: PASS - num_classes=3 aligns with 3-class RipenessLabel taxonomy
- **CHK017**: PASS - Default metadata paths consistent with PlatformTarget resolution

### Acceptance Criteria Quality (2/3 - 67%)

- **CHK018**: PASS - US-3 scenarios objectively testable across 3 config states
- **CHK019**: PASS - SC-004 defines measurable 100% rejection for invalid configs
- **CHK020**: FAIL - Round-trip serialization not explicitly required

### Scenario & Edge Case Coverage (4/8 - 50%)

- **CHK021**: PASS - Partial TOML requirements defined (US-3, FR-002, Edge Cases)
- **CHK022**: FAIL - Whitespace-only strings not addressed
- **CHK023**: PASS - Empty and zero-element hidden_layers both covered
- **CHK024**: FAIL - NaN/INFINITY handling not specified
- **CHK025**: FAIL - validate() call timing ambiguous
- **CHK026**: PASS - ValidationError message requirements minimally specified
- **CHK027**: PASS - CLI --data-dir override precedence clear ("Override")
- **CHK028**: FAIL - Non-existent custom config file behavior unclear

### Non-Functional & Constitutional (4/4 - 100%)

- **CHK029**: PASS - No unwrap()/expect() in library code mandated
- **CHK030**: PASS - TDD Red-Green-Refactor order explicit in plan
- **CHK031**: PASS - Emoji-free and no decorative separators required
- **CHK032**: PASS - Rustdoc requirements specified

### Dependencies & Assumptions (2/4 - 50%)

- **CHK033**: PASS - Serde/toml/thiserror versions validated
- **CHK034**: FAIL - Metadata path resolution ambiguous
- **CHK035**: FAIL - Unknown TOML keys handling not specified
- **CHK036**: PASS - Phase 1 scope boundary (validation only, no audio decoding) clear

## Recommendations for Spec Improvement

### High Priority (Blocking Implementation)

1. **CHK025**: Define Config::from_file validation semantics
    - Recommendation: `from_file()` should return `Result<Config, ConfigError>` where the Config is already validated

2. **CHK034**: Clarify metadata path resolution
    - Recommendation: Specify that `iphone_metadata` and `android_metadata` are either:
        - Absolute paths (resolved from working directory), OR
        - Relative to `data_dir`

### Medium Priority (Edge Case Hardening)

1. **CHK024**: Add floating-point special value rejection
    - Add to FR-003: "All f32/f64 parameters must reject NaN, INFINITY, and -INFINITY"

2. **CHK028**: Define missing config file behavior
    - Specify: "If custom --config path does not exist, return IoError; only default path falls back to Config::default()"

3. **CHK022**: Add whitespace validation
    - Specify: "Non-empty string fields must be trimmed and reject whitespace-only values"

### Low Priority (Nice to Have)

1. **CHK013**: Define activation string case-sensitivity
    - Recommendation: Accept case-insensitive, normalize to lowercase

2. **CHK015**: Document DSP-to-model dimension relationship
    - Add note explaining time dimension calculation or pad/truncate policy

3. **CHK020**: Add round-trip serialization test requirement
    - Add to acceptance criteria

4. **CHK035**: Define unknown key policy
    - Recommendation: Use `#[serde(deny_unknown_fields)]` for strict validation

## Conclusion

**Overall Assessment**: The specification is **75% complete** for requirements quality. The core functional requirements are well-defined, but several edge cases and behavioral ambiguities remain.

**Gate Status**: **CONDITIONAL PASS** - Implementation can proceed with the following caveats:

- Implementer must make reasonable assumptions for the 9 unclear items
- Document assumptions as inline comments and unit test cases
- Flag ambiguities for reviewer confirmation during code review

**Next Steps**:

1. Review this report with spec author
2. Clarify the 9 unchecked items via `/speckit-clarify`
3. Update checklist with [x] markers after clarifications
4. Proceed to `/speckit-implement` once all critical items (CHK025, CHK034) are resolved
