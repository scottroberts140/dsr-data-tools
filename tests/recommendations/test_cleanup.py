import numpy as np
import pandas as pd

from dsr_data_tools.recommendations import (
    MissingValuesRecommendation,
    MissingValueStrategy,
    NonInformativeRecommendation,
)


def test_missing_values_impute_median():
    df = pd.DataFrame({"vals": [10, 20, np.nan, 40]})
    rec = MissingValuesRecommendation(
        column_name="vals",
        description="Fill nulls",
        strategy=MissingValueStrategy.IMPUTE_MEDIAN,
    )
    df = rec.apply(df)
    assert df["vals"].isna().sum() == 0
    assert df["vals"].iloc[2] == 20.0  # Median of 10, 20, 40


def test_non_informative_drop():
    df = pd.DataFrame({"junk": [None, None], "keep": [1, 2]})
    rec = NonInformativeRecommendation(
        column_name="junk", description="Drop it", reason="All null"
    )
    df = rec.apply(df)
    assert "junk" not in df.columns
