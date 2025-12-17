# dsr-data-tools

Data analysis and exploration tools for exploratory data analysis (EDA).

## Features

- **Dataset Analysis**: Comprehensive statistical summaries and data quality assessment
- **Data Exploration**: Tools for understanding data distributions, correlations, and patterns
- **Quality Metrics**: Missing value detection, data type analysis, and anomaly identification

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

## Requirements

- Python >= 3.9
- pandas
- numpy
- dsr-utils

## License

MIT License - see LICENSE file for details
