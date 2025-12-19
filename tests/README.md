# Testing Guide - dsr-data-tools

## Running Tests

### Install test dependencies
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

### Run specific test file
```bash
pytest tests/test_analysis.py
```

### Run tests matching a pattern
```bash
pytest -k "test_analysis"
```

### Run tests with verbose output
```bash
pytest -v
```

## Test Structure

Tests are organized by module:
- `tests/test_analysis.py` - Tests for data analysis functions

## Writing Tests

All test files should:
1. Start with `test_` prefix
2. Use pytest conventions
3. Include docstrings explaining what is being tested
4. Use fixtures from `conftest.py` when needed

Example:
```python
import pytest
import pandas as pd

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({'col': [1, 2, 3]})

def test_analysis_function(sample_data):
    """Test the analysis function with sample data."""
    assert len(sample_data) > 0
```

## Coverage Reports

After running tests with coverage, view the HTML report:
```bash
open htmlcov/index.html
```
