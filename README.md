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
