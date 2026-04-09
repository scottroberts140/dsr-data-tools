"""Dataset analysis helpers and interaction recommendation utilities."""

from __future__ import annotations

from typing import Any, Type, cast

import numpy as np
import pandas as pd
from dsr_utils.strings import is_float_string, to_snake_case
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from dsr_data_tools.enums import InteractionType
from dsr_data_tools.recommendations import (
    ColumnHint,
    FeatureInteractionRecommendation,
    RecommendationManager,
)


class DataframeColumn:
    """
    Represents metadata for a single DataFrame column.

    Stores column name, non-null count, and data type information for analysis
    and display purposes.

    Attributes
    ----------
    name : str
        The name of the column.
    non_null_count : int
        Number of non-null values in the column.
    data_type : Type
        The pandas data type of the column.
    """

    def __init__(self, name: str, non_null_count: int, data_type: Any):
        self._name = name
        self._non_null_count = non_null_count
        self._data_type = data_type

    @property
    def name(self) -> str:
        """Return the column name."""
        return self._name

    @property
    def non_null_count(self) -> int:
        """Return the number of non-null values."""
        return self._non_null_count

    @property
    def data_type(self) -> Any:
        """Return the pandas data type."""
        return self._data_type

    @staticmethod
    def dfc_list_from_df(df: pd.DataFrame) -> list[DataframeColumn]:
        """
        Create a list of DataframeColumn objects from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to extract column information from.

        Returns
        -------
        list[DataframeColumn]
            List of DataframeColumn objects, one per column.
        """
        # counts is a Series: index=column_name, value=non_null_count
        counts = df.count()

        # dtypes is a Series: index=column_name, value=dtype
        dtypes = df.dtypes

        # Using zip avoids indexing into the Series with a variable,
        # which satisfies the type checker's overload requirements.
        return [
            DataframeColumn(name=str(name), non_null_count=int(count), data_type=dtype)
            for name, count, dtype in zip(counts.index, counts.values, dtypes.values)
        ]


class DataframeInfo:
    """
    Stores comprehensive information about a DataFrame's structure and content.

    Provides a summary of DataFrame characteristics including row counts,
    duplicate detection, and detailed column information.
    """

    def __init__(self, df: pd.DataFrame):
        self._row_count = len(df)
        # Casting to int as .sum() on boolean series can sometimes return np.int64
        self._duplicate_row_count = int(df.duplicated().sum())
        self._columns = DataframeColumn.dfc_list_from_df(df)

    @property
    def row_count(self) -> int:
        """Return the total number of rows."""
        return self._row_count

    @property
    def duplicate_row_count(self) -> int:
        """Return the count of duplicate rows."""
        return self._duplicate_row_count

    @property
    def columns(self) -> list[DataframeColumn]:
        """Return the list of DataframeColumn metadata objects."""
        return self._columns

    def info(self) -> None:
        """
        Display a formatted summary of the DataFrame information.

        Calculates dynamic padding to ensure the table remains aligned
        regardless of column name lengths.
        """
        print(f"Rows: {self.row_count}")
        print(f"Duplicate rows: {self.duplicate_row_count}")
        print()

        if not self.columns:
            print("No columns found.")
            return

        # --- Dynamic Alignment ---
        # Calculate the width needed for the 'Column' name column
        max_col_name = max(len(c.name) for c in self.columns)
        name_width = max(max_col_name, 15)  # Minimum width of 15
        count_width = 12
        type_width = 15

        # Header
        header = (
            f"{'Column':<{name_width}} "
            f"{'Non-null':>{count_width}}   "
            f"{'Data type':<{type_width}}"
        )
        print(header)
        print("-" * len(header))

        # Rows
        for c in self.columns:
            # Note: c.data_type might be a dtype object,
            # so we access .name for a clean string like 'int64'
            dtype_name = getattr(c.data_type, "name", str(c.data_type))

            print(
                f"{c.name:<{name_width}} "
                f"{c.non_null_count:>{count_width}}   "
                f"{dtype_name:<{type_width}}"
            )


def analyze_column_data(series: pd.Series, dataframe_column: DataframeColumn) -> None:
    """
    Analyze and print detailed statistics for a single DataFrame column.

    Displays type information, null counts, and logical composition (e.g.,
    detecting 'integers in disguise' within float columns).

    Parameters
    ----------
    series : pd.Series
        The data series to analyze.
    dataframe_column : DataframeColumn
        Metadata about the column.
    """
    total_len = len(series)
    non_null = series.dropna()
    unique_count = int(series.nunique())
    na_count = int(series.isna().sum())

    # Get a clean string for the data type
    dtype_name = getattr(
        dataframe_column.data_type, "name", str(dataframe_column.data_type)
    )

    # --- 1. Base Metadata ---
    lines = [
        f"Column:             {dataframe_column.name}",
        f"Data type:          {dtype_name}",
        f"Non-null:           {dataframe_column.non_null_count}",
        f"N/A count:          {na_count}",
        f"Unique values:      {unique_count}",
    ]

    # --- 2. Numeric Statistics ---
    # Use pandas utility for broader compatibility (covers float64, float32, etc.)
    if pd.api.types.is_numeric_dtype(series):
        if not non_null.empty:
            lines.append(f"Min value:          {series.min()}")
            lines.append(f"Max value:          {series.max()}")

    # --- 3. Float-Specific: Integer Discovery ---
    if pd.api.types.is_float_dtype(series) and not non_null.empty:
        integer_count = int((non_null % 1 == 0).sum())
        non_integer_count = len(non_null) - integer_count
        lines.append(f"Integer values:     {integer_count}")
        lines.append(f"Non-integer values: {non_integer_count}")

    # --- 4. Object-Specific: Numeric String Discovery ---
    if pd.api.types.is_object_dtype(series) and not non_null.empty:
        # isnumeric() only catches whole numbers; we check for digit strings
        numeric_count = series.astype(str).str.isnumeric().sum()
        # Note: 'strings.is_float_string' is an external helper we assume is imported
        try:
            float_str_count = series.apply(is_float_string).sum()
        except (ImportError, AttributeError):
            float_str_count = 0  # Fallback if helper is missing

        lines.append(f"Numeric strings:    {numeric_count}")
        if float_str_count > 0:
            lines.append(f"Float strings:      {float_str_count}")

    # Output formatting
    print("\n" + "\n".join(lines))


def analyze_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
    hints: dict[str, ColumnHint] | None = None,
    hints_only: bool = False,
    generate_recs: bool = False,
    max_decimal_places: int | dict[str, int] | None = None,
    default_max_decimal_places: int | None = None,
    normalize_column_names: bool = False,
    min_binning_unique_values: int | dict[str, int] | None = None,
    default_min_binning_unique_values: int = 10,
    max_binning_unique_values: int | dict[str, int] | None = None,
    default_max_binning_unique_values: int = 1000,
) -> tuple[DataframeInfo, RecommendationManager | None]:
    """
    Perform comprehensive analysis of all columns in a DataFrame.

    Displays overall DataFrame characteristics followed by detailed per-column
    statistics. Optionally generates an optimization pipeline via RecommendationManager.
    """
    from dsr_data_tools.recommendations import RecommendationManager

    # 1. Column Normalization
    # If we normalize, we do it at the very start so all subsequent objects
    # (Info, Hints, Recommendations) refer to the same snake_case names.
    if normalize_column_names:
        df = df.copy()
        df.columns = [to_snake_case(str(col)) for col in df.columns]
        if target_column:
            target_column = to_snake_case(target_column)

        # Note: If hints are passed, their keys might also need normalization
        # depending on user expectation. For now, we assume the df modification is enough.

    # 2. Global Metadata Analysis
    df_info = DataframeInfo(df)
    df_info.info()

    # 3. Recommendation Generation
    manager = None
    if generate_recs:
        manager = RecommendationManager()
        manager.generate_recommendations(
            df=df,
            target_column=target_column,
            hints=hints,
            hints_only=hints_only,
            max_decimal_places=max_decimal_places,
            default_max_decimal_places=default_max_decimal_places,
            min_binning_unique_values=min_binning_unique_values,
            default_min_binning_unique_values=default_min_binning_unique_values,
            max_binning_unique_values=max_binning_unique_values,
            default_max_binning_unique_values=default_max_binning_unique_values,
        )

    # 4. Detailed Per-Column Analysis
    # Iterating directly over df_info.columns is cleaner than range(n)
    for col_metadata in df_info.columns:
        analyze_column_data(df[col_metadata.name], col_metadata)

    # 5. Recommendation Summary
    if manager:
        print("\n" + "=" * 40)
        print("RECOMMENDATION SUMMARY")
        print("=" * 40)
        manager.execution_summary()

    return df_info, manager


def generate_interaction_recommendations(
    df: pd.DataFrame,
    target_column: str | None = None,
    top_n: int | None = 20,
    exclude_columns: list[str] | None = None,
    random_state: int | None = 42,
) -> list[FeatureInteractionRecommendation]:
    """
    Generate recommended feature interactions using statistical guidance.

    Analyzes numeric columns to suggest meaningful interactions based on
    Status-Impact (MI), Resource Density (Correlation), and Product Utilization.
    """
    interactions: list[FeatureInteractionRecommendation] = []

    # 1. Identify usable numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return []

    # Filter out IDs and user-excluded columns
    id_keywords = ["id", "row", "index", "number", "code"]
    exclude_set = set(exclude_columns or [])
    exclude_set.update(
        col for col in numeric_cols if any(kw in col.lower() for kw in id_keywords)
    )

    usable_cols = [col for col in numeric_cols if col not in exclude_set]
    if len(usable_cols) < 2:
        return []

    # 2. Extract target metadata if provided
    y = (
        df[target_column].dropna()
        if target_column and target_column in df.columns
        else None
    )
    is_classification = y.nunique() < 10 if y is not None else False

    # 3. Apply Statistical Rules
    # Rule 1: Status-Impact
    interactions.extend(
        _find_status_impact_interactions(
            df, usable_cols, y, is_classification, random_state
        )
    )

    # Rule 2: Resource Density
    interactions.extend(_find_resource_density_interactions(df, usable_cols))

    # Rule 3: Product Utilization
    interactions.extend(
        _find_utilization_interactions(
            df, usable_cols, y, is_classification, random_state
        )
    )

    # 4. Final Ranking and Filtering
    interactions.sort(key=lambda x: x.priority_score, reverse=True)

    return interactions[:top_n] if top_n else interactions


# --- Private Helper Workers ---


def _find_status_impact_interactions(
    df: pd.DataFrame,
    cols: list[str],
    y: pd.Series | None,
    is_clf: bool,
    seed: int | None,
) -> list[FeatureInteractionRecommendation]:
    recs = []
    binary_cols = [c for c in cols if df[c].nunique() == 2]
    if not binary_cols:
        return []

    # Determine 'Strong' binaries via Mutual Information
    mi_series: pd.Series | None = None
    if y is not None:
        X = df[binary_cols].loc[y.index]
        scores = (
            mutual_info_classif(X, y, random_state=seed)
            if is_clf
            else mutual_info_regression(X, y, random_state=seed)
        )
        mi_series = pd.Series(scores, index=binary_cols)
        strong_binaries = mi_series[mi_series > mi_series.median()].index.tolist()
    else:
        strong_binaries = binary_cols

    # Find high-variance continuous candidates
    # We define 'high variance' as the top 40% of variances in the usable set
    variances = df[cols].var()
    var_threshold = variances.quantile(0.6)
    high_var_cols = [
        c for c in cols if df[c].nunique() > 10 and variances[c] > var_threshold
    ]

    for b_col in strong_binaries:
        for v_col in high_var_cols:
            if b_col == v_col:
                continue
            score = float(mi_series[b_col]) if mi_series is not None else 0.0
            recs.append(
                FeatureInteractionRecommendation(
                    column_name=v_col,
                    column_name_2=b_col,
                    interaction_type=InteractionType.STATUS_IMPACT,
                    operation="*",
                    description=f"Status-Impact interaction: {v_col} × {b_col}",
                    rationale=f"Multiply '{v_col}' by binary status '{b_col}' to segment behavior.",
                    priority_score=score,
                )
            )
    return recs


def _find_resource_density_interactions(
    df: pd.DataFrame, cols: list[str]
) -> list[FeatureInteractionRecommendation]:
    recs = []
    cont_cols = [c for c in cols if df[c].nunique() > 20]
    if len(cont_cols) < 2:
        return []

    corr_matrix = df[cont_cols].corr().abs()
    for i, col1 in enumerate(cont_cols):
        for col2 in cont_cols[i + 1 :]:
            val: Any = corr_matrix.at[col1, col2]
            corr = float(val)
            if corr > 0.7:
                # Sparsity check to avoid zero-division issues
                if (df[col2] != 0).mean() > 0.9:
                    recs.append(
                        FeatureInteractionRecommendation(
                            column_name=col1,
                            column_name_2=col2,
                            interaction_type=InteractionType.RESOURCE_DENSITY,
                            operation="/",
                            description=f"Resource Density ratio: {col1} / {col2}",
                            rationale=f"High correlation ({corr:.2f}) suggests a meaningful ratio relationship.",
                            priority_score=corr,
                        )
                    )
    return recs


def _find_utilization_interactions(
    df: pd.DataFrame,
    cols: list[str],
    y: pd.Series | None,
    is_clf: bool,
    seed: int | None,
) -> list[FeatureInteractionRecommendation]:
    recs = []
    discrete_cols = [c for c in cols if 2 < df[c].nunique() <= 20]
    duration_cols = [c for c in cols if df[c].nunique() > 20]

    for count_col in discrete_cols:
        for dur_col in duration_cols:
            if (df[dur_col] != 0).mean() < 0.9:
                continue

            score = 0.0
            if y is not None:
                # Calculate MI for the interaction specifically
                rate = (
                    (df[count_col] / df[dur_col])
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                common_idx = rate.index.intersection(y.index)
                if not common_idx.empty:
                    X_tmp = rate.loc[common_idx].to_numpy().reshape(-1, 1)
                    y_tmp = y.loc[common_idx]
                    mi = (
                        mutual_info_classif(X_tmp, y_tmp, random_state=seed)
                        if is_clf
                        else mutual_info_regression(X_tmp, y_tmp, random_state=seed)
                    )
                    score = float(mi[0])

            recs.append(
                FeatureInteractionRecommendation(
                    column_name=count_col,
                    column_name_2=dur_col,
                    interaction_type=InteractionType.PRODUCT_UTILIZATION,
                    operation="/",
                    description=f"Product Utilization rate: {count_col} / {dur_col}",
                    rationale=f"Measure intensity of '{count_col}' relative to '{dur_col}'.",
                    priority_score=score,
                )
            )
    return recs
