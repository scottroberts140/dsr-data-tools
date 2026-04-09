# dsr-data-tools

[![PyPI version](https://img.shields.io/pypi/v/dsr-data-tools.svg?cacheSeconds=300)](https://pypi.org/project/dsr-data-tools/)
[![Python versions](https://img.shields.io/pypi/pyversions/dsr-data-tools.svg?cacheSeconds=300)](https://pypi.org/project/dsr-data-tools/)
[![License](https://img.shields.io/pypi/l/dsr-data-tools.svg?cacheSeconds=300)](https://pypi.org/project/dsr-data-tools/)
[![Changelog](https://img.shields.io/badge/changelog-available-blue.svg)](https://github.com/scottroberts140/dsr-data-tools/releases)

Data analysis and exploration tools for exploratory data analysis (EDA).

**Version 1.2.0**: This release introduces a more robust Recommendation Engine, including semantic boolean mapping, optimized datetime duration calculations, and enhanced execution priority management.

## Features

- **Dataset Analysis**: Comprehensive statistical summaries and data quality assessment
- **Data Exploration**: Tools for understanding data distributions, correlations, and patterns
- **Quality Metrics**: Missing value detection, data type analysis, and anomaly identification
- **Statistically Guided Feature Interactions**: Automatic discovery of meaningful feature interactions using Mutual Information and Pearson Correlation
- **Recommendation Engine**: Intelligent pipeline for Boolean mapping, Numerical casting, and Datetime standardization with customizable execution priority
- **Intelligent Boolean Mapping**: Detects and standardizes diverse truthiness indicators (e.g., "Y/N", "Active/Inactive", "1/0") into proper boolean types.
- **Explicit Numerical Casting**: Dedicated workers for `Float` and `Integer` conversions that handle "Float-as-String" and dirty data safely.

## Installation

```bash
pip install dsr-data-tools
```

## Usage

```python
import pandas as pd
from dsr_data_tools import analyze_dataset

# Load your data
df = pd.read_csv('data.csv')

# Perform comprehensive analysis
analyze_dataset(df)
```

### Datetime Conversion Recommendation

`generate_recommendations()` detects object/string columns that are likely datetimes and recommends converting them to a proper datetime dtype.

```python
import pandas as pd
from dsr_data_tools.analysis import generate_recommendations
from dsr_data_tools.recommendations import apply_recommendations

# Example column with mostly valid date strings
df = pd.DataFrame({
 'date_str': [
  '2025-01-01', '2025-01-02', '2025-01-03',
  '2025-01-04', 'invalid',  # one invalid value
 ] * 10  # scale up rows
})

recs = generate_recommendations(df)

# If detected, apply the datetime conversion recommendation
if 'date_str' in recs and 'datetime_conversion' in recs['date_str']:
 df_converted = apply_recommendations(df, {
  'date_str': recs['date_str']['datetime_conversion']
 })
 # Column is now datetime64; invalid entries coerced to NaT
 print(df_converted['date_str'].dtype)  # datetime64[ns]
```

### Boolean Classification

```python
# The engine now handles semantic mapping, recognizing 'Y' as True
# based on common indicators rather than just alphabetical order
from dsr_data_tools.recommendations import BooleanClassificationRecommendation

df = pd.DataFrame({"active": ["Y", "N", "Y"]})
rec = BooleanClassificationRecommendation(
    column_name="active",
    description="Convert to bool",
    values=["Y", "N"]
)

# Returns [True, False, True]
df_bool = rec.apply(df)
```

### Date Durations

```python
# Calculate the difference between two datetime columns in a specific unit.
from dsr_data_tools.recommendations import DatetimeDurationRecommendation

rec = DatetimeDurationRecommendation(
    column_name="start_date",
    start_column="start_date",
    end_column="end_date",
    output_column="days_to_complete",
    unit="days"  # Supports 'seconds', 'minutes', 'hours', 'days'
)

df_duration = rec.apply(df)
```

## Performance

This library is optimized for large-scale data processing using vectorized operations.

- Vectorized Integer Checks: Optimized from $O(N)$ Python-level application to vectorized modulo operations, resulting in a **5-6× speed increase**.

- Cached Data Scans: Implemented caching for common operations like dropna() and unique() to ensure each data column is scanned as few times as possible, maintaining high efficiency for wide datasets.

## Benchmarks

A benchmark script compares per-element `apply(is_integer)` against a vectorized modulo check for detecting integer-like floats. On large series, the vectorized approach is typically 5–6× faster.

- Script: [scripts/benchmark_integer_checks.py](scripts/benchmark_integer_checks.py)

Run via Python:

```bash
python scripts/benchmark_integer_checks.py           # default size (2,000,000)
python scripts/benchmark_integer_checks.py 5000000  # custom size
```

Or via Makefile target:

```bash
make benchmark                # default N=2,000,000
make benchmark N=5000000      # custom size
```

## Requirements

- Python >= 3.10
- dsr-utils >= 1.3.0
- numpy >= 2.4.4
- pandas >= 3.0.2
- scikit-learn >= 1.8.0

## License

MIT License - see LICENSE file for details
