import pandas as pd
import pytest
from dsr_data_tools.enums import CalculationOperation, TransformFunction
from dsr_data_tools.recommendations import (
    ColumnCalculationRecommendation,
    DatetimeDurationRecommendation,
    DatetimeProperty,
    FeatureExtractionRecommendation,
    FeatureInteractionRecommendation,
    FunctionApplyRecommendation,
    InteractionType,
    set_derived_name_policy,
)


def test_datetime_duration_apply():
    df = pd.DataFrame({
        "start": pd.to_datetime(["2023-01-01"]),
        "end": pd.to_datetime(["2023-01-05"]),
    })
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


def test_column_calculation_apply_subtraction():
    df = pd.DataFrame({"capital_gain": [1000.0, 500.0], "capital_loss": [100.0, 50.0]})
    rec = ColumnCalculationRecommendation(
        column_name="capital_gain",
        description="Net capital",
        operation=CalculationOperation.SUBTRACT,
        right_column="capital_loss",
        output_column="net_capital",
    )

    out = rec.apply(df)
    assert "net_capital" in out.columns
    assert out["net_capital"].tolist() == [900.0, 450.0]


def test_function_apply_log1p():
    df = pd.DataFrame({"capital_gain": [0.0, 9.0, 99.0]})
    rec = FunctionApplyRecommendation(
        column_name="capital_gain",
        description="log1p transform",
        function_name=TransformFunction.LOG1P,
        output_column="log_capital_gain",
    )

    out = rec.apply(df)
    assert "log_capital_gain" in out.columns
    assert out["log_capital_gain"].iloc[0] == 0.0
    assert out["log_capital_gain"].iloc[1] == pytest.approx(2.30258509299)


def test_feature_interaction_warn_mode_normalizes_invalid_derived_name():
    set_derived_name_policy("warn")
    try:
        with pytest.warns(UserWarning, match="normalized"):
            rec = FeatureInteractionRecommendation(
                column_name="A",
                column_name_2="B",
                interaction_type=InteractionType.RESOURCE_DENSITY,
                operation="/",
                description="Ratio",
                derived_name="A_<B",
            )

        assert rec.derived_name == "A_B"
    finally:
        set_derived_name_policy("warn")


def test_feature_interaction_strict_mode_rejects_invalid_derived_name():
    set_derived_name_policy("strict")
    try:
        with pytest.raises(ValueError, match="normalized"):
            FeatureInteractionRecommendation(
                column_name="A",
                column_name_2="B",
                interaction_type=InteractionType.RESOURCE_DENSITY,
                operation="/",
                description="Ratio",
                derived_name="A_<B",
            )
    finally:
        set_derived_name_policy("warn")


def test_feature_interaction_off_mode_preserves_invalid_derived_name():
    set_derived_name_policy("off")
    try:
        rec = FeatureInteractionRecommendation(
            column_name="A",
            column_name_2="B",
            interaction_type=InteractionType.RESOURCE_DENSITY,
            operation="/",
            description="Ratio",
            derived_name="A_<B",
        )

        assert rec.derived_name == "A_<B"
    finally:
        set_derived_name_policy("warn")
