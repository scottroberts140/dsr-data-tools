# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.1] - 2026-05-05

### Added

* **Staged Pipeline Execution**: `RecommendationManager.apply()` now executes recommendations in automatically derived stages rather than a single pass. Recommendations that produce new columns (feature extraction, aggregation, interaction, datetime duration) are placed in earlier stages; recommendations that consume those columns are placed in later stages automatically—no manual ordering required.
* **New Recommendation Placeholders**: `RecommendationManager.load_from_yaml()` now treats keys like `new_rec_*` and `__new__` as placeholders for user-authored recommendations and generates a fresh `rec_*` ID automatically.
* **Unassigned Recommendation Bucket**: `recommendations.yaml` now supports an optional top-level `unassigned` block for recommendations without an explicit stage. These recommendations are assigned to the default stage for their `rec_type` via `EXECUTION_PRIORITY`.
* **`explicit_stage` Field**: All `Recommendation` subclasses now accept an optional `explicit_stage: int | None` field (editable in YAML). Setting this forces a recommendation into at least that stage number, enabling manual ordering for cases where column-type sequencing matters within the same column (e.g., impute → cast to int → convert to category).
* **`required_columns()` Method**: Base `Recommendation` class exposes `required_columns() -> set[str]`, returning all columns needed as input. Overridden by `FeatureInteractionRecommendation`, `DatetimeDurationRecommendation`, and `AggregationRecommendation` to include their secondary columns.
* **`produced_columns()` Method**: Base `Recommendation` class exposes `produced_columns() -> set[str]`, returning all columns the recommendation will create. Implemented by `FeatureInteractionRecommendation` (`derived_name`), `FeatureExtractionRecommendation` (all configured output columns), `DatetimeDurationRecommendation` (`output_column`), and `AggregationRecommendation` (`output_column`).
* **`_get_staged_pipeline()` Method**: New internal `RecommendationManager` method that builds the ordered list of stages from `initial_columns`, using `required_columns()` and `produced_columns()` to determine placement. Raises `ValueError` for unsatisfiable dependencies.

### Changed

* **`_validate_pipeline()` Redesign**: Validation is now stage-aware. The initial flat "all required columns must exist in df" check is replaced by staged dependency validation. Drop conflicts and overwrite conflicts are evaluated per-stage.
* **Stage-Only YAML Contract**: Flat ID-keyed recommendation YAML loading has been removed. `load_from_yaml()` now requires stage-grouped YAML (`stage_N` / `step_N`) with optional `unassigned`.
* **`FeatureExtractionRecommendation` — `_PROPERTY_KEY_MAP`**: Added a `ClassVar` mapping from `DatetimeProperty` to feature key string. Used by `produced_columns()` to compute expected output column names from `properties` and `output_columns` configuration.
* **Stable ID Computation**: `explicit_stage` is excluded from the stable ID payload so that changing the stage hint for an existing recommendation does not alter its tracked identity.
* **Expanded ColumnHint Coverage**: Added `ColumnHint` support for integer conversion, float conversion, boolean conversion, binning, value replacement, encoding, feature interaction, outlier detection, and explicit missing-value handling.
* **Hint-to-Recommendation Translation**: Extended `RecommendationManager._apply_column_hint()` so the new `ColumnHintType` values generate locked recommendation objects with the metadata needed for downstream execution.
* **ColumnHint Documentation**: Expanded README guidance to clarify which recommendation types are representable as `ColumnHint` values and explicitly documented that `ClassImbalanceRecommendation` remains advisory-only.
* **Bucketing Message Formatting**: Normalized `_apply_bucketing` status message formatting to a single stable line for clearer logging output and snapshot diff readability.

### Fixed

* **`Flag` Enum Deserialization**: `_deserialize_value` now correctly parses pipe-delimited `Flag` enum values (e.g., `DAYOFWEEK|HOUR|SIN_HOUR|COS_HOUR` for `DatetimeProperty`) loaded from YAML, restoring combined flag members instead of raising a type error.
* **Cyclic Feature `produced_columns()` Inference**: `FeatureExtractionRecommendation.produced_columns()` now mirrors the sin/cos mate-name inference performed by `apply()`. When `output_columns` renames one side of a cyclic pair, the complementary output name is derived using the same substitution logic, so staged pipeline validation correctly resolves the inferred column names.
* **Preprocessing Regression Coverage**: Strengthened preprocessing unit tests around bucketing outputs and message assertions to reduce formatting-related false negatives in CI.

## [2.1.0] - 2026-04-22

### Added

* **Recommendation YAML Rehydration**: Added `RecommendationManager.load_from_yaml()` to reconstruct recommendation pipelines from persisted YAML files.

### Changed

* **YAML Class Resolution**: Improved class resolution for recommendation types that share enum values by persisting and consuming class identity metadata.
* **Enum-Safe Deserialization**: Added field-aware enum deserialization so editable YAML values are restored to typed enum members.
* **Documentation Refresh**: Updated recommendation manager, overwrite behavior, info output, and interaction recommendation guides to match current implementation behavior.

### Fixed

* **Round-Trip Reliability**: Ensured recommendation IDs, editable fields, and class-specific attributes survive save/load cycles used by orchestration workflows.

## [2.0.0] - 2026-04-22

### Breaking

* **Return-Only Output APIs**: `DataframeInfo.info()` no longer prints to stdout and now returns a formatted `str` only.

* **Column Analysis Output APIs**: `analyze_column_data()` no longer prints to stdout and now returns a formatted `str` only.

* **Dataset Analysis Return Shape**: `analyze_dataset()` now always returns a three-item tuple:
  * `df_info`
  * `manager`
  * `column_analysis_output: dict[str, str]`

### Changed

* **Structured Column Output**: Per-column analysis strings are now captured and exposed as structured data for downstream orchestration/reporting workflows.

* **Documentation and Tests**: Updated examples and unit tests to align with return-only behavior and the new `analyze_dataset()` tuple shape.

## [1.4.2] - 2026-04-21

### Fixed

* **YAML Persistence Typing**: Updated `RecommendationManager.save_to_yaml` to use the shared path-compatible typing expected by `dsr-files`, avoiding `str(...)` workarounds while preserving local and cloud path support.

* **Regression Coverage**: Added focused test coverage to verify `save_to_yaml` returns the concrete saved path and rejected-kwargs payload from the YAML handler.

### Changed

* **Dependency Floor**: Raised the minimum `dsr-files` requirement to `3.1.1` so installations receive the path typing and package version behavior that `save_to_yaml` now expects.

## [1.4.1] - 2026-04-14

### Removed

* **Hashing Documentation**: Removed references to "Deterministic Object Hashing" and "Audit-Safe File Hashing" as these responsibilities have been moved to the dsr-orchestrator level.

* **Hashing Examples**: Deleted the "Data Integrity & Hashing" section from the README to reflect the library's focus on recommendation logic.

### Fixed

* **Feature Accuracy**: Corrected the feature list to ensure all listed capabilities are natively provided by dsr-data-tools.

## [1.4.0] - 2026-04-14

### Added

* **Metadata-Driven Field Discovery**: Implemented dataclasses.field metadata to explicitly whitelist "editable" attributes within Recommendation subclasses.

* **Deterministic Object Hashing**: Integrated joblib.hash logic into the new hashing.py module to generate unique fingerprints for complex Python objects like pandas DataFrames and NumPy arrays.

* **Audit-Safe File Hashing**: Added chunked SHA-256 hashing for raw data files to verify integrity before memory ingestion, optimized for memory-constrained environments.

* **NumPy-Style Documentation**: Fully updated all core class and method docstrings—including Recommendation and hashing utilities—to the NumPy standard for professional MLE clarity.

### Changed

* **Recommendation Base Class**: Refactored the Recommendation class to include system-managed id generation and _locked state flags to preserve audit history.

* **Expanded Attribute Suite**: Added notes, alias, and enabled fields to the base class, tagged specifically for manual user justification and persistence.

* **Refactored Identification**: Modified internal ID generation to ensure Recommendation objects remain deterministic across multiple analysis sessions.

### Fixed

* **Type Hinting Strictness**: Resolved Pylance type-checking issues by implementing explicit None checks in hashing and serialization routines.

* **Output Visibility**: Transitioned internal status reporting from side-effect print() statements to functional list returns, supporting "Quiet Mode" CLI execution.

## [1.3.0] - 2026-04-13

### Added

* **YAML Serialization**: Integrated `RecommendationManager.save_to_yaml` to support persistent auditing workflows.
* **Serialization Logic**: Added `to_dict` to the `Recommendation` base class to safely handle Enum-to-string conversion for external exports.

### Changed

* **Dependencies**: Upgraded minimum requirement to `dsr-files>=2.1.0` to leverage the new standardized YAML handler.

## [1.2.0] - 2026-04-09

### Added

* Support for FLOAT_CONVERSION in the RecommendationManager execution pipeline.

### Fixed

* Resolved an issue where BooleanClassificationRecommendation mapped invalid/mismatched strings to True.
* Corrected unit scaling in DatetimeDurationRecommendation (defaulting to days vs minutes).

### Performance

* Full NumPy-style docstring coverage for the RecommendationManager.
* Added default_date_format to RecommendationManager for more robust datetime inference.

## [1.1.0] - 2026-02-10

### Added

* `RecommendationManager.enable_by_id()`
* `RecommendationManager.disable_by_id()`
* `RecommendationManager.toggle_enabled_by_id()`

### Fixed

* Column-drop tracking now treats missing-value recommendations as dropping only when the strategy is `DROP_COLUMN`.

## [1.0.0] - 2026-02-08

### Breaking

* Version reset to 1.0.0 to reflect non-backward-compatible changes across the library.

### Fixed

* Test fixtures updated to align with current recommendation APIs and binning behavior.

## [0.0.6] - 2025-12-20

### Fixed

* Fixed runtime error in `generate_interaction_recommendations` when `target_column` is None
  * Priority score for Status-Impact interactions now correctly uses computed variable instead of undefined mi_series reference
  * Ensures graceful fallback to 0.0 priority when no target column provided

## [0.0.5] - 2025-12-20

### Added

* **Statistically Guided Feature Interactions**: Refactored `generate_interaction_recommendations()` to use statistical measures:
  * **Rule 1 (Status-Impact)**: Uses Mutual Information to identify binary columns most predictive of target, paired with high-variance continuous columns
  * **Rule 2 (Resource Density)**: Uses Pearson Correlation (>0.7) to identify complementary continuous columns for ratio features
  * **Rule 3 (Product Utilization)**: Distribution-based logic combining discrete counts (2-20 unique) with continuous duration (>20 unique) for rate features
  * New parameters: `target_column` for statistical guidance, `random_state` for reproducibility
* Configurable binning thresholds via `min_binning_unique_values` and `max_binning_unique_values` parameters
  * Supports both global `int` values and per-column `dict[str, int]` configurations
  * Added `default_min_binning_unique_values` (default: 10) and `default_max_binning_unique_values` (default: 1000)
  * Parameters available in both `generate_recommendations()` and `analyze_dataset()`
* Benchmark script (`scripts/benchmark_integer_checks.py`) to measure performance improvements
  * Demonstrates ~5-6× speedup for vectorized integer checks vs `.apply(is_integer())`
* Makefile with `benchmark` target for easy performance testing
* GitHub Actions CI workflow for automated testing
  * Tests run on Python 3.10, 3.11, and 3.12
  * Includes coverage reporting with pytest-cov
* Comprehensive test suite covering new functionality:
  * Binning threshold configurations (int and dict forms)
  * Integer conversion detection
  * Decimal precision optimization
  * Value replacement detection
* `scikit-learn` added to dependencies for statistical feature selection

### Changed

* **Breaking**: Columns flagged as `non_informative` now short-circuit all other recommendation checks
  * Ensures no conflicting recommendations for ID columns or high-cardinality object columns
  * Improves performance by skipping unnecessary analysis
* Vectorized integer detection replacing expensive `.apply(lambda x: x.is_integer())` calls
  * Uses `(series % 1 == 0)` for ~5-6× performance improvement
  * Applied in int64 conversion, decimal precision, and column analysis functions
* Enhanced documentation with Benchmarks section in README
* Updated docstrings for `generate_recommendations()` and `analyze_dataset()` with detailed parameter descriptions
* `generate_interaction_recommendations()` now accepts `target_column` and `random_state` parameters for statistically-guided recommendations

### Fixed

* Variance comparison type errors when detecting high-variance columns
  * Added proper type guards and numeric checks before comparisons
  * Replaced list comprehension with explicit loop for better type safety
* Removed unused `min_value` variable in outlier detection logic
* Type-checker warnings for optional float values and numpy bool conversions

### Performance

* Cached `series.dropna()` as `non_null_series` to avoid repeated computation
* Cached `non_null_series.unique()` as `non_null_unique` for reuse across checks
* Lazy computation and caching of `non_null_min` and `non_null_max` values
* Cached `value_counts()` results and reused in:
  * Non-numeric value detection (`_detect_non_numeric_values`)
  * Class imbalance analysis
* Refactored `_detect_non_numeric_values()` to accept cached `value_counts` parameter
* Overall reduction in redundant pandas operations across all recommendation checks

## [0.0.4] - 2025-12-XX

### Initial

* Core recommendation engine for data preprocessing
* Support for missing values, encoding, outliers, class imbalance, and binning recommendations
* Feature interaction recommendations based on statistical patterns
* Dataset analysis with comprehensive column statistics

[0.0.5]: https://github.com/scottroberts140/dsr-data-tools/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/scottroberts140/dsr-data-tools/releases/tag/v0.0.4
