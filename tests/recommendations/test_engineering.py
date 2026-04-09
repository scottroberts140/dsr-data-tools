import pandas as pd

from dsr_data_tools.recommendations import (
    DatetimeDurationRecommendation,
    DatetimeProperty,
    FeatureExtractionRecommendation,
    FeatureInteractionRecommendation,
    InteractionType,
)


def test_datetime_duration_apply():
    df = pd.DataFrame(
        {"start": pd.to_datetime(["2023-01-01"]), "end": pd.to_datetime(["2023-01-05"])}
    )
    rec = DatetimeDurationRecommendation(
        column_name="start",
        description="Calc duration",
        start_column="start",
        end_column="end",
        unit="days",
        output_column="days_diff",
    )
    df = rec.apply(df)
    assert "days_diff" in df.columns
    assert df["days_diff"].iloc[0] == 4


def test_feature_interaction_apply():
    df = pd.DataFrame({"A": [10, 20], "B": [2, 5]})
    rec = FeatureInteractionRecommendation(
        column_name="A",
        column_name_2="B",
        interaction_type=InteractionType.RESOURCE_DENSITY,
        operation="/",
        description="Ratio",
        priority_score=0.9,
    )
    # The rec creates a column named "A_vs_B" or similar based on operation
    df = rec.apply(df)
    assert "A_vs_B" in df.columns
    assert df["A_vs_B"].iloc[0] == 5.0


def test_datetime_feature_extraction():
    df = pd.DataFrame({"date": pd.to_datetime(["2023-05-20"])})
    # Requesting Year and Day
    props = DatetimeProperty.YEAR | DatetimeProperty.DAY
    rec = FeatureExtractionRecommendation("date", "desc", properties=props)

    df = rec.apply(df)
    assert "date_year" in df.columns
    assert "date_day" in df.columns
    assert "date_month" not in df.columns  # Ensure bitmask was respected
    assert df["date_year"].iloc[0] == 2023
