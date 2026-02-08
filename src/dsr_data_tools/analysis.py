from __future__ import annotations
from typing import cast
from dsr_data_tools.recommendations import (
    RecommendationManager,
    FeatureInteractionRecommendation,
    ColumnHint,
)
import pandas as pd
from typing import Type
from dsr_utils import strings
from dsr_utils.strings import to_snake_case
from dsr_data_tools.enums import (
    InteractionType,
)


class DataframeColumn:
    """Represents metadata for a single DataFrame column.

    Stores column name, non-null count, and data type information for analysis
    and display purposes.

    Attributes:
        name (str): The column name.
        non_null_count (int): Number of non-null values in the column.
        data_type (Type): The pandas data type of the column.

    Example:
        >>> df = pd.DataFrame({'age': [25, 30, None, 35]})
        >>> col = DataframeColumn('age', 3, float)
        >>> col.name
        'age'
        >>> col.non_null_count
        3
    """
    @staticmethod
    def dfc_list_from_df(df: pd.DataFrame) -> list[DataframeColumn]:
        """Create a list of DataframeColumn objects from a DataFrame.

        Extracts column names, non-null counts, and data types from the DataFrame
        and creates a DataframeColumn object for each column.

        Args:
            df (pd.DataFrame): The DataFrame to extract column information from.

        Returns:
            list[DataframeColumn]: List of DataframeColumn objects, one per column.

        Example:
            >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
            >>> columns = DataframeColumn.dfc_list_from_df(df)
            >>> len(columns)
            2
        """
        df_columns = df.columns.tolist()
        df_non_null_count = df.count().tolist()
        df_data_types = df.dtypes.tolist()
        n = len(df_columns)
        dfc_list = []

        for c in range(n):
            dfc = DataframeColumn(df_columns[c],
                                  df_non_null_count[c],
                                  df_data_types[c])
            dfc_list.append(dfc)

        return dfc_list

    def __init__(
            self,
            name: str,
            non_null_count: int,
            data_type: Type
    ):
        self.__name = name
        self.__non_null_count = non_null_count
        self.__data_type = data_type

    @property
    def name(self) -> str:
        return self.__name

    @property
    def non_null_count(self) -> int:
        return self.__non_null_count

    @property
    def data_type(self) -> Type:
        return self.__data_type


class DataframeInfo:
    """Stores comprehensive information about a DataFrame's structure and content.

    Provides a summary of DataFrame characteristics including row counts, duplicate
    detection, and detailed column information. Used for data exploration and
    quality assessment.

    Attributes:
        row_count (int): Total number of rows in the DataFrame.
        duplicate_row_count (int): Number of duplicate rows detected.
        columns (list[DataframeColumn]): List of column metadata objects.

    Example:
        >>> df = pd.DataFrame({
        ...     'name': ['Alice', 'Bob', 'Alice'],
        ...     'age': [25, 30, 25]
        ... })
        >>> df_info = DataframeInfo(df)
        >>> df_info.row_count
        3
        >>> df_info.duplicate_row_count
        1
        >>> len(df_info.columns)
        2
    """

    def __init__(
            self,
            df: pd.DataFrame
    ):
        self.__row_count = len(df)
        self.__duplicate_row_count = df.duplicated().sum()
        self.__columns = DataframeColumn.dfc_list_from_df(df)

    @property
    def row_count(self) -> int:
        return self.__row_count

    @property
    def duplicate_row_count(self) -> int:
        return self.__duplicate_row_count

    @property
    def columns(self) -> list[DataframeColumn]:
        return self.__columns

    def info(self):
        """Display formatted summary of DataFrame information.

        Prints row count, duplicate count, and a table showing column names,
        non-null counts, and data types for all columns.

        Example:
            >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
            >>> df_info = DataframeInfo(df)
            >>> df_info.info()
            Rows: 2
            Duplicate rows: 0

            Column         Non-null   Data type
            name                 2   object
            age                  2   int64
        """
        print(f'Rows: {self.row_count}')
        print(f'Duplicate rows: {self.duplicate_row_count}')
        print()
        col_headers = ['Column', 'Non-null', 'Data type']
        col_width = [15, 10, 12]
        print(
            f'{col_headers[0]:<{col_width[0]}}{col_headers[1]:>{col_width[1]}}   {col_headers[2]:<{col_width[2]}}')

        for c in self.columns:
            print(
                f'{c.name:<{col_width[0]}}{c.non_null_count:>{col_width[1]}}   {c.data_type.name:<{col_width[2]}}')


def analyze_column_data(
    series: pd.Series,
    dataframe_column: DataframeColumn
):
    """Analyze and print detailed statistics for a single DataFrame column.

    Displays column name, data type, null counts, unique values, min/max values.
    For float columns, shows integer vs non-integer value counts. For object
    columns, shows numeric vs non-numeric value counts.

    Args:
        series (pd.Series): The data series to analyze.
        dataframe_column (DataframeColumn): Metadata about the column.

    Example:
        >>> df = pd.DataFrame({'price': [10.5, 20.0, 30.99]})
        >>> col = DataframeColumn('price', 3, float)
        >>> analyze_column_data(df['price'], col)
        # Prints detailed statistics
    """
    series_length = len(series)
    is_float_type = (dataframe_column.data_type.name == 'float64')
    integer_analysis = ''

    if is_float_type:
        # Vectorized count of integer-like float values
        integer_value_count = int((series % 1 == 0).sum())
        non_integer_value_count = series_length - integer_value_count
        integer_analysis = f"""Integer values:     {integer_value_count}
Non-integer values: {non_integer_value_count}"""

    is_object_data_type = (dataframe_column.data_type.name == 'object')
    object_analysis = ''

    if is_object_data_type:
        numeric_value_count = series.str.isnumeric().sum()
        non_numeric_value_count = series_length - \
            series.apply(strings.is_float_string).sum()

        object_analysis = f"""Numeric values:     {numeric_value_count}
Non-numeric values: {non_numeric_value_count}"""

    # Build analysis string, only including min/max for numeric types
    min_max_str = ""
    is_numeric_type = is_float_type or (dataframe_column.data_type.name in [
                                        'int64', 'int32', 'int16', 'int8'])
    if is_numeric_type:
        min_max_str = f"""Min value:          {series.min()}
Max value:          {series.max()}"""

    analysis = f"""
Column:             {dataframe_column.name}
Data type:          {dataframe_column.data_type.name}
Non-null:           {dataframe_column.non_null_count}
N/A count:          {series.isna().sum()}
Unique values:      {series.nunique()}
{min_max_str}"""

    print(analysis)

    if is_float_type:
        print(integer_analysis)

    if is_object_data_type:
        print(object_analysis)


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
    default_max_binning_unique_values: int = 1000
) -> tuple[DataframeInfo, RecommendationManager | None]:
    """Perform comprehensive analysis of all columns in a DataFrame.

    Displays overall DataFrame information (row count, duplicates) followed by
    detailed analysis of each column including data types, null counts, unique
    values, and type-specific statistics. Optionally generates data preparation
    recommendations.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        target_column (str | None): Name of the target column (for recommendation generation).
        generate_recs (bool): Whether to generate recommendations via RecommendationManager.
        hints (dict[str, ColumnHint] | None): Optional column-specific guidance forwarded to
            RecommendationManager.generate_recommendations().
        hints_only (bool): If True, only hint-driven recommendations are created. Forwarded to
            RecommendationManager.generate_recommendations(). Default False.
        max_decimal_places (int | dict | None): Maximum decimal places for precision optimization.
            Can be an int (applies to all float columns) or dict mapping column names to their
            specific max decimal places. If provided, float columns will be checked for
            decimal precision optimization.
        default_max_decimal_places (int | None): Default max decimal places to use for columns
            not in the max_decimal_places dict. Only used if max_decimal_places is a dict.
        normalize_column_names (bool): Whether to convert column names to snake_case. Default is False.
            If True, column names are converted at the start (before analysis and recommendations).
        min_binning_unique_values (int | dict | None): Minimum unique values for binning consideration.
            Can be an int (applies to all numeric columns) or dict mapping column names to their
            specific minimum values. If None, default_min_binning_unique_values is used for all columns.
        default_min_binning_unique_values (int): Default minimum unique values for columns not in
            the min_binning_unique_values dict. Default is 10.
        max_binning_unique_values (int | dict | None): Maximum unique values for binning consideration.
            Can be an int (applies to all numeric columns) or dict mapping column names to their
            specific maximum values. If None, default_max_binning_unique_values is used for all columns.
        default_max_binning_unique_values (int): Default maximum unique values for columns not in
            the max_binning_unique_values dict. Default is 1000.

    Returns:
        tuple[DataframeInfo, list[Recommendation] | None]: A tuple containing:
            - DataframeInfo object with structured DataFrame information
            - List of Recommendation objects (or None if generate_recs is False)

    Example:
        >>> df = pd.DataFrame({
        ...     'FirstName': ['Alice', 'Bob', 'Charlie'],
        ...     'Age': [25, 30, 35],
        ...     'Salary': [50000.0, 60000.5, 75000.0]
        ... })
        >>> info, recs = analyze_dataset(df, generate_recs=True, normalize_column_names=True)
        # Converts FirstName -> first_name, Age -> age, Salary -> salary
        # Prints comprehensive analysis of all columns and returns recommendations
    """
    from dsr_data_tools.recommendations import RecommendationManager

    # Normalize column names if requested
    if normalize_column_names:
        df = df.copy()  # Avoid modifying original DataFrame
        df.columns = [to_snake_case(col) for col in df.columns]
        # Update target_column name if it exists
        if target_column is not None:
            target_column = to_snake_case(target_column)

    df_info = DataframeInfo(df)
    df_info.info()

    n = len(df_info.columns)

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

    for c in range(n):
        col = df_info.columns[c]
        analyze_column_data(df[col.name], df_info.columns[c])

    # Display recommendation execution summary, if recommendations were created
    if manager:
        manager.execution_summary()

    return df_info, manager


def generate_interaction_recommendations(
    df: pd.DataFrame,
    target_column: str | None = None,
    top_n: int | None = 20,
    exclude_columns: list[str] | None = None,
    random_state: int | None = 42
) -> list[FeatureInteractionRecommendation]:
    """Generate recommended feature interactions using statistical guidance.

    Analyzes numeric columns to suggest meaningful interactions based on three
    statistically-guided rules:

    1. **Status-Impact (Mutual Information)**: Binary x High-variance continuous
       - Uses Mutual Information (MI) to identify which binary/status columns
         are most informative about the target outcome.
       - Only pairs high-information binary columns with high-variance
         continuous columns, ensuring interactions align with target prediction.

    2. **Resource Density (Pearson Correlation)**: Continuous / Continuous
       - Computes absolute Pearson correlation between continuous columns.
       - Creates ratio features only for highly correlated pairs (r > 0.7),
         indicating complementary financial or resource metrics.

    3. **Product Utilization (Distribution-Based)**: Discrete / Continuous
       - Identifies discrete columns (2-20 unique values) representing counts
         and continuous columns (>20 unique values) representing duration/time.
       - Creates rate features to measure utilization intensity over time.

    Args:
        df (pd.DataFrame): DataFrame to analyze for interaction opportunities.
        target_column (str | None): Target column for statistical guidance.
            If provided, MI scores are computed relative to the target,
            and Rule 1 uses only statistically significant binary columns.
            If None, Rule 1 falls back to finding high-variance columns.
        top_n (int | None): Maximum number of interactions to return, sorted by
            priority_score in descending order. If None, returns all interactions.
            Default is 20.
        exclude_columns (list[str] | None): Columns to exclude from interactions
            (e.g., ID columns). Default is None.
        random_state (int | None): Random state for Mutual Information calculation.
            Default is 42 for reproducibility.

    Returns:
        list[FeatureInteractionRecommendation]: List of recommended interactions
            sorted by priority_score in descending order (highest priority first).
            Returns empty list if no suitable candidates found.

    Example:
        >>> df = pd.DataFrame({
        ...     'Balance': [1000, 5000, 10000],
        ...     'IsActiveMember': [0, 1, 1],
        ...     'EstimatedSalary': [50000, 75000, 100000],
        ...     'Target': [0, 1, 0]
        ... })
        >>> interactions = generate_interaction_recommendations(
        ...     df, target_column='Target', exclude_columns=['Target']
        ... )
        >>> for rec in interactions:
        ...     rec.info()
    """
    import numpy as np
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    interactions: list[FeatureInteractionRecommendation] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return interactions

    # Initialize exclude list
    if exclude_columns is None:
        exclude_columns = []

    # Exclude ID-like columns and specified columns
    id_keywords = ["id", "row", "index", "number", "code"]
    exclude_cols = set(exclude_columns)
    exclude_cols.update(col for col in numeric_cols if any(
        kw in col.lower() for kw in id_keywords))

    usable_cols = [col for col in numeric_cols if col not in exclude_cols]

    if len(usable_cols) < 2:
        return interactions

    # Rule 1: Status-Impact (Categorical × High-variance continuous)
    # Instead of hardcoded keywords, we find the categorical column most
    # statistically related to our target outcome.
    binary_cols = [col for col in usable_cols if df[col].nunique() == 2]

    strong_binary_cols = []
    mi_series: pd.Series | None = None
    is_classification: bool = False
    y: pd.Series = pd.Series(dtype=object)  # Initialize y for use in Rule 3

    if target_column and target_column in df.columns and binary_cols:
        # Calculate Mutual Information between binary cols and the target
        # This identifies which status actually 'matters' for the outcome
        y = df[target_column].dropna()
        X = df[binary_cols].loc[y.index]

        # Determine if target is classification or regression
        is_classification = df[target_column].nunique() < 10
        mi_scores = mutual_info_classif(X, y, random_state=random_state) if is_classification else mutual_info_regression(
            X, y, random_state=random_state)

        # Keep columns with a MI score above a threshold (e.g., top 50th percentile)
        mi_series = pd.Series(mi_scores, index=binary_cols)
        strong_binary_cols = mi_series[mi_series >
                                       mi_series.median()].index.tolist()
    else:
        # Fallback to current behavior if no target is provided
        strong_binary_cols = binary_cols

    # Filter high-variance continuous columns (must have reasonable cardinality and variance)
    high_variance_cols = []
    for col in usable_cols:
        if df[col].nunique() > 10 and col not in binary_cols:
            col_var = df[col].var()
            quantile_var = df[usable_cols].var().quantile(0.6)
            if isinstance(col_var, (int, float)) and isinstance(quantile_var, (int, float)):
                if col_var > quantile_var:
                    high_variance_cols.append(col)

    for binary_col in strong_binary_cols:
        for cont_col in high_variance_cols:
            if binary_col != cont_col:
                priority_score = float(
                    mi_series[binary_col]) if mi_series is not None else 0.0

                interactions.append(
                    FeatureInteractionRecommendation(
                        column_name=cont_col,
                        column_name_2=binary_col,
                        interaction_type=InteractionType.STATUS_IMPACT,
                        operation="*",
                        description=f"Status-Impact interaction: {cont_col} × {binary_col}",
                        rationale=f"Multiply high-variance '{cont_col}' by binary status '{binary_col}' "
                        f"to distinguish behavior based on membership status",
                        priority_score=priority_score
                    )
                )

    # Rule 2: Resource Density (Continuous / Continuous)
    continuous_cols = [col for col in usable_cols if df[col].nunique() > 20]

    if len(continuous_cols) >= 2:
        # Calculate correlation matrix for continuous columns
        corr_matrix: pd.DataFrame = df[continuous_cols].corr().abs()

        for i, col1 in enumerate(continuous_cols):
            for col2 in continuous_cols[i + 1:]:
                corr: float = cast(float, corr_matrix.loc[col1, col2])
                # If columns are highly correlated (> 0.7), they are good ratio candidates
                if corr > 0.7:
                    # Avoid division by zero
                    non_zero_count: float = float((df[col2] != 0).sum())
                    non_zero_ratio: float = non_zero_count / len(df)
                    if non_zero_ratio > 0.9:
                        interactions.append(
                            FeatureInteractionRecommendation(
                                column_name=col1,
                                column_name_2=col2,
                                interaction_type=InteractionType.RESOURCE_DENSITY,
                                operation="/",
                                description=f"Resource Density ratio: {col1} / {col2}",
                                rationale=f"High correlation ({corr:.2f}) detected between "
                                f"'{col1}' and '{col2}', suggesting a meaningful ratio relationship.",
                                priority_score=corr
                            )
                        )

    # Rule 3: Product Utilization (Discrete / Continuous)
    discrete_cols = [col for col in usable_cols if 2 < df[col].nunique() <= 20]
    duration_like_cols = [col for col in usable_cols if df[col].nunique() > 20]

    for count_col in discrete_cols:
        for dur_col in duration_like_cols:
            if (df[dur_col] != 0).sum() / len(df) > 0.9:
                priority_score = 0.0

                if target_column:
                    # Create the temporary rate feature
                    temp_rate = df[count_col] / df[dur_col]

                    # Alignment and Cleaning
                    # MI cannot handle NANs or Infinite values
                    valid_idx = temp_rate.replace(
                        [np.inf, -np.inf], np.nan).dropna().index
                    intersect_idx = valid_idx.intersection(y.index)

                    if len(intersect_idx) > 0:
                        X_temp = temp_rate.loc[intersect_idx].to_numpy().reshape(
                            -1, 1)
                        y_temp = y.loc[intersect_idx]

                        # Calculate MI for this specific interaction
                        if is_classification:
                            mi_val = mutual_info_classif(
                                X_temp, y_temp, random_state=random_state)
                        else:
                            mi_val = mutual_info_regression(
                                X_temp, y_temp, random_state=random_state)

                        priority_score = float(mi_val[0])

                interactions.append(
                    FeatureInteractionRecommendation(
                        column_name=count_col,
                        column_name_2=dur_col,
                        interaction_type=InteractionType.PRODUCT_UTILIZATION,
                        operation="/",
                        description=f"Product Utilization rate: {count_col} / {dur_col}",
                        rationale=f"Combining discrete count '{count_col}' with continuous duration "
                        f"'{dur_col}' to measure utilization intensity over time.",
                        priority_score=priority_score
                    )
                )

    # Sort interactions from most informative to least informative
    interactions.sort(key=lambda x: x.priority_score, reverse=True)

    if top_n and len(interactions) > top_n:
        return interactions[:top_n]

    return interactions
