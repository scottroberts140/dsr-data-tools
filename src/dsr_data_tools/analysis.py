"""Dataset analysis helpers and interaction recommendation utilities."""

from __future__ import annotations

from typing import Any

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
    data_type : Any
        The pandas data type of the column.
    """

    def __init__(self, name: str, non_null_count: int, data_type: Any):
        """
        Initialize a DataframeColumn instance.

        Parameters
        ----------
        name : str
            The name of the column.
        non_null_count : int
            Number of non-null values in the column.
        data_type : Any
            The pandas data type of the column.
        """
        self._name = name
        self._non_null_count = non_null_count
        self._data_type = data_type

    @property
    def name(self) -> str:
        """
        Return the column name.

        Returns
        -------
        str
            The name of the column.
        """
        return self._name

    @property
    def non_null_count(self) -> int:
        """
        Return the number of non-null values.

        Returns
        -------
        int
            The count of non-null entries in the column.
        """
        return self._non_null_count

    @property
    def data_type(self) -> Any:
        """
        Return the pandas data type.

        Returns
        -------
        Any
            The data type (dtype) of the column.
        """
        return self._data_type

    @staticmethod
    def dfc_list_from_df(df: pd.DataFrame) -> list["DataframeColumn"]:
        """
        Create a list of DataframeColumn objects from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to extract column information from.

        Returns
        -------
        list[DataframeColumn]
            List of DataframeColumn objects, one per column, containing name,
            non-null count, and dtype information.
        """
        counts = df.count()
        dtypes = df.dtypes

        return [
            DataframeColumn(name=str(name), non_null_count=int(count), data_type=dtype)
            for name, count, dtype in zip(counts.index, counts.values, dtypes.values)
        ]


class DataframeInfo:
    """
    Stores comprehensive information about a DataFrame's structure and content.

    Provides a summary of DataFrame characteristics including row counts,
    duplicate detection, and detailed column information.

    Attributes
    ----------
    row_count : int
        The total number of rows in the DataFrame.
    duplicate_row_count : int
        The number of rows identified as duplicates.
    columns : list[DataframeColumn]
        A list of DataframeColumn objects containing metadata for each column.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataframeInfo by analyzing the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to analyze for structural and content information.
        """
        self._row_count = len(df)
        # Casting to int as .sum() on boolean series can sometimes return np.int64
        self._duplicate_row_count = int(df.duplicated().sum())
        self._columns = DataframeColumn.dfc_list_from_df(df)

    @property
    def row_count(self) -> int:
        """
        Return the total number of rows.

        Returns
        -------
        int
            The row count of the DataFrame.
        """
        return self._row_count

    @property
    def duplicate_row_count(self) -> int:
        """
        Return the count of duplicate rows.

        Returns
        -------
        int
            The number of duplicate rows found.
        """
        return self._duplicate_row_count

    @property
    def columns(self) -> list[DataframeColumn]:
        """
        Return the list of DataframeColumn metadata objects.

        Returns
        -------
        list[DataframeColumn]
            A list containing metadata for each column in the DataFrame.
        """
        return self._columns

    def info(self) -> str:
        """
        Build a formatted summary of DataFrame information.

        Calculates dynamic padding to ensure the table remains aligned
        regardless of column name lengths.

        Returns
        -------
        str
            The formatted summary text.
        """
        lines: list[str] = [
            f"Rows: {self.row_count}",
            f"Duplicate rows: {self.duplicate_row_count}",
            "",
        ]

        if not self.columns:
            lines.append("No columns found.")
            return "\n".join(lines)

        # --- Dynamic Alignment ---
        max_col_name = max(len(c.name) for c in self.columns)
        name_width = max(max_col_name, 15)  # Minimum width of 15
        count_width = 12
        type_width = 15

        header = (
            f"{'Column':<{name_width}} "
            f"{'Non-null':>{count_width}}   "
            f"{'Data type':<{type_width}}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for c in self.columns:
            dtype_name = getattr(c.data_type, "name", str(c.data_type))

            lines.append(
                f"{c.name:<{name_width}} "
                f"{c.non_null_count:>{count_width}}   "
                f"{dtype_name:<{type_width}}"
            )

        return "\n".join(lines)


def analyze_column_data(series: pd.Series, dataframe_column: DataframeColumn) -> str:
    """
    Analyze a single DataFrame column and return detailed statistics.

    Displays type information, null counts, and logical composition, such as
    detecting 'integers in disguise' within float columns or numeric strings
    within object columns.

    Parameters
    ----------
    series : pd.Series
        The data series to analyze.
    dataframe_column : DataframeColumn
        Metadata object containing the name and data type of the column.

    Returns
    -------
    str
        The formatted analysis output.

    Raises
    ------
    AttributeError
        If the dataframe_column object is missing required attributes.
    """
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
        numeric_count = series.astype(str).str.isnumeric().sum()
        try:
            # Note: is_float_string is an external helper dependency
            float_str_count = series.apply(is_float_string).sum()
        except (NameError, ImportError, AttributeError):
            float_str_count = 0

        lines.append(f"Numeric strings:    {numeric_count}")
        if float_str_count > 0:
            lines.append(f"Float strings:      {float_str_count}")

    # Output formatting
    return "\n" + "\n".join(lines)


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
) -> tuple[DataframeInfo, RecommendationManager | None, dict[str, str]]:
    """
    Perform comprehensive analysis of all columns in a DataFrame.

    Displays overall DataFrame characteristics followed by detailed per-column
    statistics. Optionally generates an optimization pipeline via RecommendationManager.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to be analyzed.
    target_column : str | None, optional
        The name of the label or target column for supervised learning context.
    hints : dict[str, ColumnHint] | None, optional
        Manual overrides for column types to guide recommendation logic.
    hints_only : bool, default False
        If True, only generates recommendations for columns provided in `hints`.
    generate_recs : bool, default False
        Whether to instantiate a RecommendationManager and generate cleaning suggestions.
    max_decimal_places : int | dict[str, int] | None, optional
        Constraint for rounding float columns. Can be a global int or per-column dict.
    default_max_decimal_places : int | None, optional
        The default rounding precision if not specified in `max_decimal_places`.
    normalize_column_names : bool, default False
        If True, converts all column headers to snake_case before analysis.
    min_binning_unique_values : int | dict[str, int] | None, optional
        Minimum unique values required to consider a column for binning.
    default_min_binning_unique_values : int, default 10
        Default threshold for minimum binning unique values.
    max_binning_unique_values : int | dict[str, int] | None, optional
        Maximum unique values allowed to consider a column for binning.
    default_max_binning_unique_values : int, default 1000
        Default threshold for maximum binning unique values.
    Returns
    -------
    tuple[DataframeInfo, RecommendationManager | None, dict[str, str]]
        A tuple containing the structural metadata (DataframeInfo), the
        recommendation engine (RecommendationManager), if generated, and a
        mapping of column names to formatted analysis text.
    """
    from dsr_data_tools.recommendations import RecommendationManager

    # 1. Column Normalization
    if normalize_column_names:
        df = df.copy()
        df.columns = [to_snake_case(str(col)) for col in df.columns]
        if target_column:
            target_column = to_snake_case(target_column)

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
    column_analysis_output: dict[str, str] = {}
    for col_metadata in df_info.columns:
        col_output = analyze_column_data(df[col_metadata.name], col_metadata)
        column_analysis_output[col_metadata.name] = col_output

    return df_info, manager, column_analysis_output


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

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze for potential feature interactions.
    target_column : str | None, optional
        The name of the target variable, used to calculate Mutual Information
        and impact scores.
    top_n : int | None, default 20
        The maximum number of high-priority interactions to return. If None,
        all identified interactions are returned.
    exclude_columns : list[str] | None, optional
        List of specific column names to ignore during interaction analysis.
    random_state : int | None, default 42
        Seed for reproducibility in statistical sampling and MI calculations.

    Returns
    -------
    list[FeatureInteractionRecommendation]
        A sorted list of recommended interactions, ranked by their
        calculated priority score.
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
    """
    Identify potential Status-Impact interactions between binary and continuous columns.

    Uses Mutual Information (MI) to find binary columns that strongly relate to the target,
    then pairs them with high-variance continuous columns to suggest segmentation features.

    Parameters
    ----------
    df : pd.DataFrame
        The source dataset for calculating variance and distributions.
    cols : list[str]
        The list of numeric columns available for analysis.
    y : pd.Series | None
        The target variable series used for Mutual Information scoring.
    is_clf : bool
        Whether the analysis context is a classification task (True) or regression (False).
    seed : int | None
        Random seed for reproducibility in MI scoring calculations.

    Returns
    -------
    list[FeatureInteractionRecommendation]
        A list of suggested interactions where a continuous feature is multiplied
        by a binary status feature.
    """
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
    """
    Identify potential Resource Density interactions using correlation analysis.

    Finds highly correlated continuous columns and suggests ratio-based
    interactions (division), provided the denominator meets sparsity
    requirements to avoid mathematical instability.

    Parameters
    ----------
    df : pd.DataFrame
        The source dataset for correlation and sparsity calculations.
    cols : list[str]
        The list of numeric columns available for analysis.

    Returns
    -------
    list[FeatureInteractionRecommendation]
        A list of suggested ratio interactions where one continuous feature
        is divided by another.
    """
    recs = []
    # Identify continuous columns with sufficient cardinality
    cont_cols = [c for c in cols if df[c].nunique() > 20]
    if len(cont_cols) < 2:
        return []

    corr_matrix = df[cont_cols].corr().abs()
    for i, col1 in enumerate(cont_cols):
        for col2 in cont_cols[i + 1 :]:
            val: Any = corr_matrix.at[col1, col2]
            corr = float(val)

            # High correlation suggests redundancy or a strong ratio relationship
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
    """
    Identify potential Product Utilization interactions (rates) between discrete and continuous columns.

    Suggests division interactions where a discrete 'count' column is normalized
    by a continuous 'duration' or 'resource' column. If a target is provided,
    the interaction is scored based on its Mutual Information (MI) with the target.

    Parameters
    ----------
    df : pd.DataFrame
        The source dataset for calculating interactions and distributions.
    cols : list[str]
        The list of numeric columns available for analysis.
    y : pd.Series | None
        The target variable series used to score the predictive power of the suggested interaction.
    is_clf : bool
        Whether the analysis context is a classification task (True) or regression (False).
    seed : int | None
        Random seed for reproducibility in MI scoring calculations.

    Returns
    -------
    list[FeatureInteractionRecommendation]
        A list of suggested rate-based interactions representing product or resource utilization.
    """
    recs = []
    # Discrete columns are defined by cardinality between 3 and 20
    discrete_cols = [c for c in cols if 2 < df[c].nunique() <= 20]
    # Duration columns are higher-cardinality continuous features
    duration_cols = [c for c in cols if df[c].nunique() > 20]

    for count_col in discrete_cols:
        for dur_col in duration_cols:
            # Sparsity check to ensure mathematical stability for division
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
