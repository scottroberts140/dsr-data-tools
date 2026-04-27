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


# ---------------------------------------------------------------------------
# min_frequency tests
# ---------------------------------------------------------------------------

def test_min_frequency_int_threshold_collapses_rare():
    # A=4, B=3, C=1, D=1, E=1  => C/D/E each appear once; threshold=2 collapses them
    df = pd.DataFrame(
        {"country": ["A", "A", "A", "A", "B", "B", "B", "C", "D", "E"]}
    )
    transformed, msgs = apply_preprocessing(
        df,
        {"country": {"min_frequency": {"threshold": 2, "other_label": "Other"}}},
    )
    result = transformed["country"].tolist()
    assert result == ["A", "A", "A", "A", "B", "B", "B", "Other", "Other", "Other"]
    assert any("collapsed 3 rare categories" in msg for msg in msgs)


def test_min_frequency_float_threshold_collapses_rare():
    # 10 rows; threshold=0.2 means min_count=2; C/D/E appear once => collapsed
    df = pd.DataFrame(
        {"country": ["A", "A", "A", "A", "B", "B", "B", "C", "D", "E"]}
    )
    transformed, msgs = apply_preprocessing(
        df,
        {"country": {"min_frequency": {"threshold": 0.2}}},
    )
    result = transformed["country"].tolist()
    assert result == ["A", "A", "A", "A", "B", "B", "B", "Other", "Other", "Other"]


def test_min_frequency_no_collapse_when_all_above_threshold():
    df = pd.DataFrame({"col": ["X", "X", "Y", "Y"]})
    transformed, msgs = apply_preprocessing(
        df,
        {"col": {"min_frequency": {"threshold": 2}}},
    )
    assert transformed["col"].tolist() == ["X", "X", "Y", "Y"]
    assert any("no categories below" in msg for msg in msgs)


def test_min_frequency_custom_other_label():
    df = pd.DataFrame({"col": ["A", "A", "B"]})
    transformed, msgs = apply_preprocessing(
        df,
        {"col": {"min_frequency": {"threshold": 2, "other_label": "Rare"}}},
    )
    assert "Rare" in transformed["col"].tolist()
    assert "Other" not in transformed["col"].tolist()


def test_min_frequency_requires_threshold_key():
    df = pd.DataFrame({"col": ["A", "B"]})
    with pytest.raises(ValueError, match="requires a 'threshold' key"):
        apply_preprocessing(df, {"col": {"min_frequency": {}}})


def test_min_frequency_rejects_invalid_threshold():
    df = pd.DataFrame({"col": ["A", "B"]})
    with pytest.raises(ValueError, match="Invalid threshold"):
        apply_preprocessing(df, {"col": {"min_frequency": {"threshold": -1}}})

    with pytest.raises(ValueError, match="Invalid threshold"):
        apply_preprocessing(df, {"col": {"min_frequency": {"threshold": 1.5}}})


def test_min_frequency_and_group_by_frequency_compose():
    # Both operations can be specified together; they apply sequentially
    df = pd.DataFrame(
        {"col": ["A", "A", "A", "B", "B", "C", "D", "E"]}
    )
    # min_frequency collapses C/D/E (count=1 < 2) into "Other"
    # group_by_frequency then keeps top_n=2 (A, B) and collapses "Other" too
    # Net result: A, A, A, B, B, Other, Other, Other
    transformed, msgs = apply_preprocessing(
        df,
        {
            "col": {
                "min_frequency": {"threshold": 2},
                "group_by_frequency": {"top_n": 2},
            }
        },
    )
    assert set(transformed["col"].unique()) == {"A", "B", "Other"}
