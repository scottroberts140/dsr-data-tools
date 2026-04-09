# tests/recommendations/conftest.py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def messy_numeric_df():
    """Focuses on the edge cases for numeric conversions."""
    return pd.DataFrame(
        {
            "all_nans": [np.nan, np.nan, np.nan],
            "mixed_floats": [1.0, 2.5, 3.0],  # Should NOT be converted to int
            "zero_variance": [1, 1, 1],  # Non-informative candidate
            "extreme_outliers": [1, 2, 1000000],
        }
    )


@pytest.fixture
def messy_datetime_df():
    """Focuses on the edge cases for date math."""
    return pd.DataFrame(
        {
            "start": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "end": pd.to_datetime(["2023-01-02", "2023-02-01"]),
            "bad_dates": ["not-a-date", "2023-01-01"],
        }
    )
