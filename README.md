# dsr-data-tools

[![PyPI version](https://img.shields.io/pypi/v/dsr-data-tools.svg?cacheSeconds=300)](https://pypi.org/project/dsr-data-tools/)
[![Python versions](https://img.shields.io/pypi/pyversions/dsr-data-tools.svg?cacheSeconds=300)](https://pypi.org/project/dsr-data-tools/)
[![License](https://img.shields.io/pypi/l/dsr-data-tools.svg?cacheSeconds=300)](https://pypi.org/project/dsr-data-tools/)
[![Changelog](https://img.shields.io/badge/changelog-available-blue.svg)](https://github.com/scottroberts140/dsr-data-tools/releases)

Data analysis and exploration tools for exploratory data analysis (EDA).

**Version 2.1.0**: Added robust YAML round-trip support via `RecommendationManager.load_from_yaml()`, including enum-safe deserialization and concrete class resolution for persisted recommendations.

## Features

- **Dataset Analysis**: Comprehensive statistical summaries and data quality assessment.
- **Data Exploration**: Tools for understanding data distributions, correlations, and patterns.
- **Quality Metrics**: Missing value detection, data type analysis, and anomaly identification.
- **Statistically Guided Feature Interactions**: Automatic discovery of meaningful feature interactions using Mutual Information and Pearson Correlation.
- **Recommendation Engine**: Intelligent pipeline for Boolean mapping, Numerical casting, and Datetime standardization with customizable execution priority.
- **User-Guided ColumnHints**: Explicitly guide the engine with metadata for financial, geospatial, or temporal data to override automated heuristics.
- **Intelligent Boolean Mapping**: Detects and standardizes diverse truthiness indicators (e.g., "Y/N", "Active/Inactive", "1/0") into proper boolean types.
- **Cyclic Feature Extraction**: Decomposes datetimes into periodic Sine/Cosine features to preserve temporal relationships for machine learning.
- **Numerical Precision Optimization**: Standardize decimal depth using configurable rounding modes (Nearest, Bankers, Up, Down).
- **Metadata-Driven Customization**: Use class-level metadata to define "editable" fields, enabling seamless integration with YAML-based orchestration.

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
# Returns DataframeInfo, RecommendationManager (or None), and per-column text output
df_info, manager, column_analysis_output = analyze_dataset(df)

# info() returns a formatted summary string
summary = df_info.info()
print(summary)

# Access per-column analysis output (example: first available column)
first_col = next(iter(column_analysis_output))
print(column_analysis_output[first_col])
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

Calculate the numeric duration between two datetime columns in specific units such as 'seconds', 'minutes', 'hours', or 'days'.

```python
from dsr_data_tools.recommendations import DatetimeDurationRecommendation

rec = DatetimeDurationRecommendation(
    start_column="order_date",
    end_column="delivery_date",
    output_column="days_to_deliver",
    unit="days"
)

df = rec.apply(df)
```

### Interactive Missing Value handling

The engine allows choosing between statistical imputation (mean/median/mode), constant filling, or row/column removal.

```python
from dataclasses import fields

# Discover which fields are whitelisted for user edits in your pipeline
editable_fields = [
    f.name for f in fields(rec) 
    if f.metadata.get("editable", False)
]
# Returns: ['strategy', 'fill_value', 'notes', 'enabled', 'alias']
```

### Guided Recommendations with ColumnHints

Users can provide a ColumnHint to specify the 'logical type' of a column and set constraints like rounding, bounds, or specific feature extraction needs.

Supported `ColumnHint` coverage includes dataset-transforming and feature-engineering recommendation types such as datetime extraction, aggregation, categorical/encoding guidance, numeric and dtype conversions, binning, value replacement, outlier detection, boolean conversion, geospatial and distance handling, plus explicit drop/ignore behavior.

`ClassImbalanceRecommendation` is intentionally not represented as a `ColumnHint`. It is an advisory, training-time recommendation about the target distribution, and its `apply()` method does not mutate the dataset.

```python
import pandas as pd
from dsr_data_tools.analysis import RecommendationManager
from dsr_data_tools.recommendations import ColumnHint, RoundingMode

# Load data
df = pd.read_csv('data.csv')

# Define explicit hints to override or guide the engine
hints = {
    "unit_price": ColumnHint.financial(decimal_places=2, rounding_mode=RoundingMode.BANKERS),
    "user_id": ColumnHint.numeric(convert_to_int=True),
    "internal_notes": ColumnHint.ignore()
}

manager = RecommendationManager()
manager.generate_recommendations(df, hints=hints)

# Display the recommended pipeline
for rec in manager._pipeline:
    rec.info()
```

## Performance

This library is optimized for large-scale data processing using vectorized operations.

- **Vectorized Integer Checks**: Optimized from $O(N)$ Python-level application to vectorized modulo operations, resulting in a 5-6× speed increase.
- **Cached Data Scans**: Common operations like dropna() and unique() are cached to minimize redundant data scans across wide datasets.
- **Efficient Scaling**: Outlier handling and scaling utilize NumPy vectorized operations and Scikit-Learn transformers for high throughput.

## Benchmarks

A benchmark script compares per-element apply(is_integer) against a vectorized modulo check. On large series, the vectorized approach is typically 5–6× faster.

- Script: [scripts/benchmark_integer_checks.py](scripts/benchmark_integer_checks.py)

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
- dsr-utils >= 1.4.0
- numpy >= 2.4.4
- pandas >= 3.0.2
- scikit-learn >= 1.8.0

## License

MIT License - see LICENSE file for details
