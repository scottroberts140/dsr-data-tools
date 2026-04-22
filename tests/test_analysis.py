import numpy as np
import pandas as pd
import pytest
from dsr_data_tools.analysis import (
    DataframeColumn,
    DataframeInfo,
    analyze_column_data,
    analyze_dataset,
)


@pytest.fixture
def analysis_df():
    """Provides a DataFrame with specific metadata targets."""
    return pd.DataFrame(
        {
            "int_col": [10, 20, 30],
            "float_col": [1.1, 2.2, np.nan],
            "str_col": ["alpha", "beta", "gamma"],
            "bool_col": [True, False, True],
        }
    )


### --- 1. DataframeColumn Tests ---


def test_dataframe_column_initialization():
    """Test that the DTO stores values correctly via properties."""
    # Option A: Pass the Python type class
    col = DataframeColumn("test_col", 10, int)

    # Option B: If you must use np.dtype, cast it to Any to bypass the check
    # col = DataframeColumn("test_col", 10, cast(Any, np.dtype("int64")))

    assert col.name == "test_col"
    assert col.non_null_count == 10
    assert col.data_type is int


def test_dfc_list_from_df(analysis_df):
    """Verify the static factory method creates the correct list of DTOs."""
    columns = DataframeColumn.dfc_list_from_df(analysis_df)

    assert len(columns) == 4
    # Check specific metadata for float_col (which has 1 NaN)
    float_meta = next(c for c in columns if c.name == "float_col")
    assert float_meta.non_null_count == 2
    assert pd.api.types.is_float_dtype(float_meta.data_type)


### --- 2. DataframeInfo Tests ---


def test_dataframe_info_stats(analysis_df):
    """Verify global row and duplicate counts."""
    # analysis_df has 3 rows.
    # Appending 1 row makes it 4 rows total.
    df_with_dup = pd.concat([analysis_df, analysis_df.iloc[[0]]], ignore_index=True)
    info = DataframeInfo(df_with_dup)

    assert info.row_count == 4
    assert info.duplicate_row_count == 1
    assert len(info.columns) == 4


def test_dataframe_info_output(analysis_df, capsys):
    """Test info() returned string output for dynamic alignment."""
    info = DataframeInfo(analysis_df)
    output = info.info()

    captured = capsys.readouterr().out
    assert isinstance(output, str)
    assert "Rows: 3" in output
    assert "Column" in output
    assert captured == ""
    assert "Non-null" in output


def test_analyze_column_data_returns_string(analysis_df, capsys):
    """Verify analyze_column_data returns formatted output without printing."""
    col = DataframeColumn(name="float_col", non_null_count=2, data_type=float)

    output = analyze_column_data(analysis_df["float_col"], col)
    captured = capsys.readouterr().out

    assert isinstance(output, str)
    assert "Column:             float_col" in output
    assert "N/A count:" in output
    assert captured == ""


def test_analyze_column_data_string_column_output(analysis_df):
    """Verify analyze_column_data output includes object-specific statistics."""
    col = DataframeColumn(name="str_col", non_null_count=3, data_type=str)

    output = analyze_column_data(analysis_df["str_col"], col)

    assert "Column:             str_col" in output
    assert "Numeric strings:" in output


### --- 3. analyze_dataset Tests ---


def test_analyze_dataset_basic(analysis_df, capsys):
    """Ensure analyze_dataset returns metadata, manager, and column outputs."""
    info, manager, column_output = analyze_dataset(analysis_df, generate_recs=False)

    captured = capsys.readouterr().out
    assert isinstance(info, DataframeInfo)
    assert manager is None
    assert set(column_output.keys()) == {"int_col", "float_col", "str_col", "bool_col"}
    assert "Column:             int_col" in column_output["int_col"]
    assert captured == ""


def test_analyze_dataset_with_normalization():
    """Verify that normalize_column_names works end-to-end."""
    df = pd.DataFrame({"Dirty Column": [1, 2], "Target Col": [0, 1]})
    info, _, _ = analyze_dataset(
        df, target_column="Target Col", normalize_column_names=True
    )

    # Check that metadata reflects normalized names
    col_names = [c.name for c in info.columns]
    assert "dirty_column" in col_names
    assert "target_col" in col_names
    assert "Dirty Column" not in col_names


def test_analyze_dataset_generates_recs(analysis_df):
    """Ensure RecommendationManager is integrated correctly."""
    # float_col has a NaN, should trigger a recommendation
    info, manager, _ = analyze_dataset(analysis_df, generate_recs=True)

    assert manager is not None
    assert len(manager._pipeline) > 0
    # Check if the missing value recommendation for float_col exists
    assert any(r.column_name == "float_col" for r in manager._pipeline)


def test_analyze_dataset_returns_column_analysis_output(analysis_df, capsys):
    """Verify per-column analysis output is always included as third item."""
    info, manager, column_output = analyze_dataset(analysis_df, generate_recs=False)

    captured = capsys.readouterr().out
    assert isinstance(info, DataframeInfo)
    assert manager is None
    assert isinstance(column_output, dict)
    assert set(column_output.keys()) == {"int_col", "float_col", "str_col", "bool_col"}
    assert "Column:             int_col" in column_output["int_col"]
    assert captured == ""
