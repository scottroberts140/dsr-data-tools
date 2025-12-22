# dsr-data-tools

Data analysis and exploration tools for exploratory data analysis (EDA).

## Features

- **Dataset Analysis**: Comprehensive statistical summaries and data quality assessment
- **Data Exploration**: Tools for understanding data distributions, correlations, and patterns
- **Quality Metrics**: Missing value detection, data type analysis, and anomaly identification
- **Statistically Guided Feature Interactions**: Automatic discovery of meaningful feature interactions using Mutual Information and Pearson Correlation

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

## Performance

This library is optimized for large-scale data processing using vectorized operations.

* Vectorized Integer Checks: Optimized from $O(N)$ Python-level application to vectorized modulo operations, resulting in a __5-6× speed increase__.

* Cached Data Scans: Implemented caching for common operations like dropna() and unique() to ensure each data column is scanned as few times as possible, maintaining high efficiency for wide datasets.

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
- pandas
- numpy
- scikit-learn
- dsr-utils

## License

MIT License - see LICENSE file for details
