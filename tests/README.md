# Testing Guide - dsr-data-tools

## Running Tests

### Install test dependencies

Ensure the library is installed in editable mode within your virtual environment to ensure tests run against your local code:

```bash
pip install -e ".[test]"
```

### Run all tests

```bash
pytest
```

### Run tests with coverage report

```bash
pytest --cov=src/dsr_data_tools --cov-report=html
```

### Run specific test files

## General analysis and metadata tests

```bash
pytest tests/test_analysis.py
```

## Recommendation engine orchestration

```bash
pytest tests/test_manager.py
```

## Specific recommendation logic

```bash
pytest tests/recommendations/test_conversions.py
```

## Debugging & Output

To disable output capture and see print() statements (essential for debugging mapping or environment issues):

```bash
pytest -s
```

To run with verbose output and match a specific test pattern:

```bash
pytest -v -k "boolean_classification"
```

## Test Structure

Tests are organized by functional area:

- tests/test_analysis.py: Tests for automated data analysis and metadata heuristics.
- tests/test_manager.py: Tests for the RecommendationManager orchestration and execution priority.
- tests/recommendations/: Tests for specific transformation logic:
  - test_conversions.py: Datetime, Boolean, and Numeric casting logic.
  - test_features.py: Feature interaction, extraction, and duration logic.

## Writing Tests

All test files should follow these standards:

1. Naming: Files must start with the test_ prefix.
2. Conventions: Use standard pytest assertions.
3. Documentation: Include NumPy-style docstrings explaining what is being tested.
4. Fixtures: Utilize shared fixtures from conftest.py where appropriate.

Example:

```python
import pytest
import pandas as pd
from dsr_data_tools.recommendations import BooleanClassificationRecommendation

def test_boolean_mapping():
    """Verify that 'Y'/'N' strings are correctly mapped to booleans."""
    df = pd.DataFrame({"active": ["Y", "N"]})
    rec = BooleanClassificationRecommendation(
        column_name="active", 
        values=["Y", "N"]
    )
    result = rec.apply(df)
    assert result["active"].tolist() == [True, False]```

## Coverage Reports

After running tests with coverage, view the HTML report to identify untested logic branches:

```bash
open htmlcov/index.html
```
