import numpy as np
import pandas as pd

from dsr_data_tools.recommendations import (
    BooleanClassificationRecommendation,
    CategoricalConversionRecommendation,
    DatetimeConversionRecommendation,
    IntegerConversionRecommendation,
)


def test_boolean_classification_apply():
    df = pd.DataFrame({"active": ["Y", "N", "Y"]})

    rec = BooleanClassificationRecommendation(
        column_name="active", description="Fix bool", values=["Y", "N"]
    )

    print(f"DEBUG: 'N' from DF hex: {df['active'].iloc[1].encode().hex()}")
    print(f"DEBUG: 'N' from Rec hex: {rec.values[1].encode().hex()}")

    transformed_df = rec.apply(df)

    assert transformed_df["active"].tolist() == [True, False, True]


def test_integer_conversion_apply():
    df = pd.DataFrame({"count": [1.0, 2.0, 5.0]})
    rec = IntegerConversionRecommendation(
        column_name="count", description="Float to Int", integer_count=3
    )
    df = rec.apply(df)
    assert pd.api.types.is_integer_dtype(df["count"])
    assert df["count"].iloc[0] == 1


def test_datetime_conversion_apply():
    df = pd.DataFrame({"timestamp": ["2023-01-01", "2023-06-01"]})
    rec = DatetimeConversionRecommendation(
        column_name="timestamp", description="Fix date", detected_format="ISO8601"
    )
    df = rec.apply(df)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_boolean_classification_logic():
    df = pd.DataFrame({"col": ["Yes", "No", "Yes"]})
    rec = BooleanClassificationRecommendation(
        "col", values=["Yes", "No"], description="..."
    )
    df = rec.apply(df)
    assert df["col"].dtype == bool
    # Sort check: 'No' (smaller string) -> False, 'Yes' -> True
    assert df["col"].tolist() == [True, False, True]


def test_integer_conversion_logic():
    df = pd.DataFrame({"col": [1.0, 2.0, np.nan]})
    rec = IntegerConversionRecommendation("col", "desc", integer_count=2)
    df = rec.apply(df)
    # Using 'Int64' (capital I) allows for nullable integers in modern pandas
    assert str(df["col"].dtype).lower().startswith("int")


def test_categorical_conversion_apply():
    """
    Verify that strings are converted to the categorical dtype.
    """
    df = pd.DataFrame({"fruit": ["apple", "banana", "apple", "cherry"]})

    # In the original structure, this class usually takes:
    # column_name, description, and unique_count
    rec = CategoricalConversionRecommendation(
        column_name="fruit",
        description="Optimize string column to categorical",
        unique_values=3,
    )

    transformed_df = rec.apply(df)

    # Assertions
    assert isinstance(transformed_df["fruit"].dtype, pd.CategoricalDtype)
    assert transformed_df["fruit"].iloc[0] == "apple"
    assert transformed_df["fruit"].nunique() == 3
