# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-08

### Breaking
- Version reset to 1.0.0 to reflect non-backward-compatible changes across the library.

### Fixed
- Test fixtures updated to align with current recommendation APIs and binning behavior.

## [0.0.6] - 2025-12-20

### Fixed
- Fixed runtime error in `generate_interaction_recommendations` when `target_column` is None
  - Priority score for Status-Impact interactions now correctly uses computed variable instead of undefined mi_series reference
  - Ensures graceful fallback to 0.0 priority when no target column provided

## [0.0.5] - 2025-12-20

### Added
- **Statistically Guided Feature Interactions**: Refactored `generate_interaction_recommendations()` to use statistical measures:
  - **Rule 1 (Status-Impact)**: Uses Mutual Information to identify binary columns most predictive of target, paired with high-variance continuous columns
  - **Rule 2 (Resource Density)**: Uses Pearson Correlation (>0.7) to identify complementary continuous columns for ratio features
  - **Rule 3 (Product Utilization)**: Distribution-based logic combining discrete counts (2-20 unique) with continuous duration (>20 unique) for rate features
  - New parameters: `target_column` for statistical guidance, `random_state` for reproducibility
- Configurable binning thresholds via `min_binning_unique_values` and `max_binning_unique_values` parameters
  - Supports both global `int` values and per-column `dict[str, int]` configurations
  - Added `default_min_binning_unique_values` (default: 10) and `default_max_binning_unique_values` (default: 1000)
  - Parameters available in both `generate_recommendations()` and `analyze_dataset()`
- Benchmark script (`scripts/benchmark_integer_checks.py`) to measure performance improvements
  - Demonstrates ~5-6× speedup for vectorized integer checks vs `.apply(is_integer())`
- Makefile with `benchmark` target for easy performance testing
- GitHub Actions CI workflow for automated testing
  - Tests run on Python 3.10, 3.11, and 3.12
  - Includes coverage reporting with pytest-cov
- Comprehensive test suite covering new functionality:
  - Binning threshold configurations (int and dict forms)
  - Integer conversion detection
  - Decimal precision optimization
  - Value replacement detection
- `scikit-learn` added to dependencies for statistical feature selection

### Changed
- **Breaking**: Columns flagged as `non_informative` now short-circuit all other recommendation checks
  - Ensures no conflicting recommendations for ID columns or high-cardinality object columns
  - Improves performance by skipping unnecessary analysis
- Vectorized integer detection replacing expensive `.apply(lambda x: x.is_integer())` calls
  - Uses `(series % 1 == 0)` for ~5-6× performance improvement
  - Applied in int64 conversion, decimal precision, and column analysis functions
- Enhanced documentation with Benchmarks section in README
- Updated docstrings for `generate_recommendations()` and `analyze_dataset()` with detailed parameter descriptions
- `generate_interaction_recommendations()` now accepts `target_column` and `random_state` parameters for statistically-guided recommendations

### Fixed
- Variance comparison type errors when detecting high-variance columns
  - Added proper type guards and numeric checks before comparisons
  - Replaced list comprehension with explicit loop for better type safety
- Removed unused `min_value` variable in outlier detection logic
- Type-checker warnings for optional float values and numpy bool conversions

### Performance
- Cached `series.dropna()` as `non_null_series` to avoid repeated computation
- Cached `non_null_series.unique()` as `non_null_unique` for reuse across checks
- Lazy computation and caching of `non_null_min` and `non_null_max` values
- Cached `value_counts()` results and reused in:
  - Non-numeric value detection (`_detect_non_numeric_values`)
  - Class imbalance analysis
- Refactored `_detect_non_numeric_values()` to accept cached `value_counts` parameter
- Overall reduction in redundant pandas operations across all recommendation checks

## [0.0.4] - 2025-12-XX

### Initial
- Core recommendation engine for data preprocessing
- Support for missing values, encoding, outliers, class imbalance, and binning recommendations
- Feature interaction recommendations based on statistical patterns
- Dataset analysis with comprehensive column statistics

[0.0.5]: https://github.com/scottroberts140/dsr-data-tools/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/scottroberts140/dsr-data-tools/releases/tag/v0.0.4
