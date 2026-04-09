import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """A messy DataFrame to exercise all parts of the engine."""
    return pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_as_int": [1.0, 2.0, 3.0, 4.0, 5.0],
            "binary_col": [0, 1, 0, 1, 0],
            "date_str": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
            ],
            "sparse_col": [1, None, None, None, None],  # 80% null
            "cat_str": ["A", "A", "B", "B", "A"],  # Low cardinality
        }
    )
