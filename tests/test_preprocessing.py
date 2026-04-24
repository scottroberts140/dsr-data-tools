import pandas as pd
import pytest
from dsr_data_tools.preprocessing import apply_preprocessing


def test_apply_preprocessing_groups_sparse_categories():
    df = pd.DataFrame(
        {
            "native_country": [
                "United-States",
                "United-States",
                "Mexico",
                "India",
                "Canada",
                "France",
            ]
        }
    )

    transformed, msgs = apply_preprocessing(
        df,
        {
            "native_country": {
                "group_by_frequency": {"top_n": 2, "other_label": "Other"}
            }
        },
    )

    assert transformed["native_country"].tolist() == [
        "United-States",
        "United-States",
        "Mexico",
        "Other",
        "Other",
        "Other",
    ]
    assert any("grouped 3 categories" in msg for msg in msgs)


def test_apply_preprocessing_skips_missing_columns():
    df = pd.DataFrame({"age": [1, 2, 3]})

    transformed, msgs = apply_preprocessing(
        df,
        {"native_country": {"group_by_frequency": {"top_n": 2}}},
    )

    assert transformed.equals(df)
    assert any("missing column 'native_country'" in msg for msg in msgs)


def test_apply_preprocessing_requires_positive_top_n():
    df = pd.DataFrame({"native_country": ["A", "B", "C"]})

    with pytest.raises(ValueError, match="expected int >= 1"):
        apply_preprocessing(
            df,
            {"native_country": {"group_by_frequency": {"top_n": 0}}},
        )
