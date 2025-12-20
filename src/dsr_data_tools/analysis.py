from __future__ import annotations
from dsr_data_tools.recommendations import (
    Recommendation,
    NonInformativeRecommendation,
    MissingValuesRecommendation,
    EncodingRecommendation,
    ClassImbalanceRecommendation,
    OutlierDetectionRecommendation,
    BooleanClassificationRecommendation,
    BinningRecommendation,
    IntegerConversionRecommendation,
    DecimalPrecisionRecommendation,
    ValueReplacementRecommendation,
    FeatureInteractionRecommendation,
)
import pandas as pd
from typing import Type
from dsr_utils import strings
from dsr_utils.strings import to_snake_case
from dsr_data_tools.enums import (
    RecommendationType,
    EncodingStrategy,
    MissingValueStrategy,
    OutlierStrategy,
    ImbalanceStrategy,
    InteractionType,
)


def _detect_non_numeric_values(non_null_unique, value_counts) -> tuple[list[str], int]:
    """Detect non-numeric placeholder values in a series.

    Identifies string values that cannot be converted to float in a column
    that should be numeric (has some numeric values).

    Args:
        non_null_unique: Array of unique non-null values from the series.
        value_counts: Pre-computed value counts for the series.

    Returns:
        tuple[list[str], int]: A tuple of (list of non-numeric values, count of non-numeric occurrences).
                              Empty list if all values are numeric or null.

    Example:
        >>> s = pd.Series([1.0, 2.0, 'tbd', 'N/A', 3.0])
        >>> unique_vals = s.dropna().unique()
        >>> val_counts = s.value_counts()
        >>> _detect_non_numeric_values(unique_vals, val_counts)
        (['tbd', 'N/A'], 2)
    """
    non_numeric_values = []
    non_numeric_count = 0

    for val in non_null_unique:
        try:
            float(val)
        except (ValueError, TypeError):
            non_numeric_values.append(str(val))
            non_numeric_count += value_counts.get(val, 0)  # O(1) lookup

    return non_numeric_values, non_numeric_count


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


def generate_recommendations(
    df: pd.DataFrame,
    target_column: str | None = None,
    max_decimal_places: int | dict[str, int] | None = None,
    default_max_decimal_places: int | None = None,
    min_binning_unique_values: int | dict[str, int] | None = None,
    default_min_binning_unique_values: int = 10,
    max_binning_unique_values: int | dict[str, int] | None = None,
    default_max_binning_unique_values: int = 1000
) -> dict[str, dict[str, Recommendation]]:
    """Generate data preparation recommendations for each column in a DataFrame.

    Analyzes each column and generates appropriate recommendations based on
    data characteristics (missing values, cardinality, data type, etc.).

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        target_column (str | None): Name of the target column (for imbalance detection).
            If provided, class imbalance will be analyzed for this column.
        max_decimal_places (int | dict | None): Maximum decimal places for precision optimization.
            Can be an int (applies to all float columns) or dict mapping column names to their
            specific max decimal places. If provided, float columns will be checked for
            decimal precision optimization.
        default_max_decimal_places (int | None): Default max decimal places to use for columns
            not in the max_decimal_places dict. Only used if max_decimal_places is a dict.
        min_binning_unique_values (int | dict | None): Minimum unique values for binning consideration.
            Can be an int (applies to all numeric columns) or dict mapping column names to their
            specific minimum values. If None, default_min_binning_unique_values is used for all columns.
        default_min_binning_unique_values (int): Default minimum unique values for columns not in
            the min_binning_unique_values dict. Only used if min_binning_unique_values is a dict.
            Default is 10.
        max_binning_unique_values (int | dict | None): Maximum unique values for binning consideration.
            Can be an int (applies to all numeric columns) or dict mapping column names to their
            specific maximum values. If None, default_max_binning_unique_values is used for all columns.
        default_max_binning_unique_values (int): Default maximum unique values for columns not in
            the max_binning_unique_values dict. Only used if max_binning_unique_values is a dict.
            Default is 1000. Adjust higher for large datasets where 1000 unique values might still
            represent continuous data (e.g., millions of rows).

    Returns:
        dict[str, dict[str, Recommendation]]: Nested dictionary mapping column names
            to recommendation types to Recommendation instances.

    Example:
        >>> df = pd.DataFrame({
        ...     'id': range(100),
        ...     'name': ['Alice'] * 100,
        ...     'age': [25] * 100 + [30] * 50,
        ...     'salary': [50000] * 50 + [100000] * 50
        ... })
        >>> recs = generate_recommendations(df, target_column='name')
        >>> recs['id']  # Non-informative (unique count == row count)
        >>> recs['age']['encoding']  # Binary encoding recommendation
    """
    recommendations: dict[str, dict[str, Recommendation]] = {}

    for col_name in df.columns:
        col_recommendations: dict[str, Recommendation] = {}
        series = df[col_name]

        # Cache commonly used series transformations for performance
        non_null_series = series.dropna()
        non_null_unique = non_null_series.unique()

        # Cache min/max for numeric columns (computed lazily when needed)
        non_null_min = None
        non_null_max = None

        # Cache value_counts (computed lazily when needed)
        value_counts_cache = None

        # 1. Check for non-informative columns
        unique_count = series.nunique()
        total_rows = len(df)
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # Non-informative: unique count equals total rows (e.g., ID column)
        if unique_count == total_rows:
            rec = NonInformativeRecommendation(
                type=RecommendationType.NON_INFORMATIVE,
                column_name=col_name,
                description=f"Column '{col_name}' has unique value for each row.",
                reason="Unique count equals row count"
            )
            col_recommendations['non_informative'] = rec
            recommendations[col_name] = col_recommendations
            continue

        # Non-informative: high cardinality object type (> 25% unique values)
        if not is_numeric and unique_count > total_rows * 0.25:
            rec = NonInformativeRecommendation(
                type=RecommendationType.NON_INFORMATIVE,
                column_name=col_name,
                description=f"Column '{col_name}' has high cardinality ({unique_count} unique values).",
                reason="High cardinality object type"
            )
            col_recommendations['non_informative'] = rec
            recommendations[col_name] = col_recommendations
            continue

        # 2. Check for missing values
        missing_count = series.isna().sum()
        if missing_count > 0:
            missing_percentage = (missing_count / total_rows) * 100

            # Determine strategy based on percentage
            if missing_percentage < 10:
                strategy = MissingValueStrategy.DROP_ROWS
            elif missing_percentage > 50:
                strategy = MissingValueStrategy.DROP_COLUMN
            else:
                strategy = MissingValueStrategy.IMPUTE

            rec = MissingValuesRecommendation(
                type=RecommendationType.MISSING_VALUES,
                column_name=col_name,
                description=f"Column '{col_name}' has {missing_count} missing values ({missing_percentage:.1f}%).",
                missing_count=missing_count,
                missing_percentage=missing_percentage,
                strategy=strategy
            )
            col_recommendations['missing_values'] = rec

        # 3. Check for boolean classification (exactly 2 unique numeric values)
        # Skip target column as it should remain numeric for classifiers
        if is_numeric and unique_count == 2 and col_name != target_column:
            values = sorted(non_null_unique.tolist())
            if values == [0.0, 1.0] or values == [0, 1]:
                rec = BooleanClassificationRecommendation(
                    type=RecommendationType.BOOLEAN_CLASSIFICATION,
                    column_name=col_name,
                    description=f"Column '{col_name}' should be treated as boolean.",
                    values=values
                )
                col_recommendations['boolean_classification'] = rec

        # 3.5. Check for int64 conversion (float64 with all integer values)
        # Only check if column is float64 and has no non-integer values
        if series.dtype == 'float64':
            if len(non_null_series) > 0:
                # Vectorized check: all values have zero fractional part
                integer_mask = (non_null_series % 1 == 0)
                integer_count = int(integer_mask.sum())
                if integer_mask.all():
                    rec = IntegerConversionRecommendation(
                        type=RecommendationType.INT64_CONVERSION,
                        column_name=col_name,
                        description=f"Column '{col_name}' is float64 with only integer values; should be int64.",
                        integer_count=integer_count
                    )
                    col_recommendations['int64_conversion'] = rec

        # 3.6. Check for decimal precision optimization (float columns with user-specified max precision)
        # Only if max_decimal_places is provided and column is numeric float type
        if max_decimal_places is not None and series.dtype == 'float64':
            # Determine the max_decimal_places value for this column
            col_max_decimal_places: int | None = None

            if isinstance(max_decimal_places, dict):
                # If dict, check if column has specific value; otherwise use default
                col_max_decimal_places = max_decimal_places.get(
                    col_name, default_max_decimal_places)
            else:
                # If int, use it directly
                col_max_decimal_places = max_decimal_places

            # Skip if already recommended for int64 conversion or no valid precision specified
            if col_max_decimal_places is not None and 'int64_conversion' not in col_recommendations:
                if len(non_null_series) > 0:
                    # Compute min/max if not already cached
                    if non_null_min is None:
                        non_null_min = non_null_series.min()
                        non_null_max = non_null_series.max()

                    # Check if rounding to col_max_decimal_places would lose significant data
                    rounded_series = non_null_series.round(
                        col_max_decimal_places)

                    # Determine if conversion to int64 is possible (all values are integers after rounding)
                    can_convert_to_int = (
                        col_max_decimal_places == 0 and (rounded_series % 1 == 0).all()
                    )

                    # Ensure typed float values for min/max to satisfy type checkers
                    min_val = float(non_null_min) if non_null_min is not None else float('nan')
                    max_val = float(non_null_max) if non_null_max is not None else float('nan')

                    rec = DecimalPrecisionRecommendation(
                        type=RecommendationType.DECIMAL_PRECISION_OPTIMIZATION,
                        column_name=col_name,
                        description=f"Column '{col_name}' can have decimal precision optimized to {col_max_decimal_places} places.",
                        max_decimal_places=col_max_decimal_places,
                        min_value=min_val,
                        max_value=max_val,
                        convert_to_int=bool(can_convert_to_int)
                    )
                    col_recommendations['decimal_precision_optimization'] = rec

        # 3.7. Check for non-numeric placeholder values (object columns with some numeric values)
        if series.dtype == 'object':
            # Compute value_counts if not already cached
            if value_counts_cache is None:
                value_counts_cache = series.value_counts()

            non_numeric_vals, non_numeric_cnt = _detect_non_numeric_values(
                non_null_unique, value_counts_cache)

            # Only recommend if there are non-numeric values and some numeric values exist
            if non_numeric_vals and len(non_numeric_vals) > 0:
                numeric_count = 0
                for val in non_null_unique:
                    try:
                        float(val)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass

                # If there are both numeric and non-numeric values, recommend replacement
                if numeric_count > 0 and non_numeric_cnt > 0:
                    rec = ValueReplacementRecommendation(
                        type=RecommendationType.VALUE_REPLACEMENT,
                        column_name=col_name,
                        description=f"Column '{col_name}' has non-numeric placeholder values that should be replaced.",
                        non_numeric_values=non_numeric_vals,
                        non_numeric_count=non_numeric_cnt
                    )
                    col_recommendations['value_replacement'] = rec

        # 4. Check for encoding recommendations (categorical columns)

        if not is_numeric and col_name != target_column:
            # Binary categorical: 2 unique values
            if unique_count == 2:
                rec = EncodingRecommendation(
                    type=RecommendationType.ENCODING,
                    column_name=col_name,
                    description=f"Column '{col_name}' is binary categorical; recommend LabelEncoder.",
                    encoder_type=EncodingStrategy.LABEL,
                    unique_values=unique_count
                )
                col_recommendations['encoding'] = rec

            # Multi-class categorical: 3-10 unique values
            elif 3 <= unique_count <= 10:
                rec = EncodingRecommendation(
                    type=RecommendationType.ENCODING,
                    column_name=col_name,
                    description=f"Column '{col_name}' is multi-class categorical; recommend OneHotEncoder.",
                    encoder_type=EncodingStrategy.ONEHOT,
                    unique_values=unique_count
                )
                col_recommendations['encoding'] = rec

        # 5. Check for outliers (numeric columns)
        if is_numeric:
            mean_value = series.mean()
            max_value = series.max()

            # Check if max value significantly exceeds mean (potential outliers)
            if max_value > mean_value * 2:  # Max is more than 2x the mean
                rec = OutlierDetectionRecommendation(
                    type=RecommendationType.OUTLIER_DETECTION,
                    column_name=col_name,
                    description=f"Column '{col_name}' has potential outliers (max={max_value:.2f}, mean={mean_value:.2f}).",
                    strategy=OutlierStrategy.SCALING,
                    max_value=max_value,
                    mean_value=mean_value
                )
                col_recommendations['outlier_detection'] = rec

        # 6. Check for class imbalance (target column)
        if col_name == target_column and unique_count <= 2:
            # Compute value_counts if not already cached
            if value_counts_cache is None:
                value_counts_cache = series.value_counts()

            max_class_percentage = (
                value_counts_cache.max() / total_rows) * 100

            if max_class_percentage > 70:
                rec = ClassImbalanceRecommendation(
                    type=RecommendationType.CLASS_IMBALANCE,
                    column_name=col_name,
                    description=f"Target variable '{col_name}' shows class imbalance ({max_class_percentage:.1f}% majority class).",
                    majority_percentage=max_class_percentage,
                    strategy=ImbalanceStrategy.CLASS_WEIGHT
                )
                col_recommendations['class_imbalance'] = rec

        # 7. Suggest binning for continuous numeric columns with moderate cardinality
        # Candidates for binning: numeric columns with more unique values than expected categories
        # but not so many that binning loses information

        # Determine the min_binning_unique_values value for this column
        col_min_binning: int
        if min_binning_unique_values is None:
            col_min_binning = default_min_binning_unique_values
        elif isinstance(min_binning_unique_values, dict):
            col_min_binning = min_binning_unique_values.get(
                col_name, default_min_binning_unique_values)
        else:
            col_min_binning = min_binning_unique_values

        # Determine the max_binning_unique_values value for this column
        col_max_binning: int
        if max_binning_unique_values is None:
            col_max_binning = default_max_binning_unique_values
        elif isinstance(max_binning_unique_values, dict):
            col_max_binning = max_binning_unique_values.get(
                col_name, default_max_binning_unique_values)
        else:
            col_max_binning = max_binning_unique_values

        if is_numeric and col_min_binning <= unique_count <= col_max_binning:
            non_null_series = series.dropna()

            # Check if column has reasonable variance and distribution for binning
            if len(non_null_series) > 0:
                # Compute min/max if not already cached
                if non_null_min is None:
                    non_null_min = non_null_series.min()
                    non_null_max = non_null_series.max()

                # Suggest binning for columns with meaningful range (not single value)
                col_min = non_null_min
                col_max = non_null_max

                if col_min < col_max:
                    # Use describe() percentiles to suggest bins
                    desc = series.describe()
                    bins = [col_min - 0.1 * abs(col_max - col_min),
                            desc['25%'], desc['50%'],
                            desc['75%'], col_max + 0.1 * abs(col_max - col_min)]
                    labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']

                    rec = BinningRecommendation(
                        type=RecommendationType.BINNING,
                        column_name=col_name,
                        description=f"Column '{col_name}' ({unique_count} unique values) could be binned into {len(labels)} categories for better feature representation.",
                        bins=bins,
                        labels=labels
                    )
                    col_recommendations['binning'] = rec

        if col_recommendations:
            recommendations[col_name] = col_recommendations

    return recommendations


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
    generate_recs: bool = False,
    max_decimal_places: int | dict[str, int] | None = None,
    default_max_decimal_places: int | None = None,
    normalize_column_names: bool = False,
    min_binning_unique_values: int | dict[str, int] | None = None,
    default_min_binning_unique_values: int = 10,
    max_binning_unique_values: int | dict[str, int] | None = None,
    default_max_binning_unique_values: int = 1000
) -> tuple[DataframeInfo, dict[str, dict[str, Recommendation]] | None]:
    """Perform comprehensive analysis of all columns in a DataFrame.

    Displays overall DataFrame information (row count, duplicates) followed by
    detailed analysis of each column including data types, null counts, unique
    values, and type-specific statistics. Optionally generates data preparation
    recommendations.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        target_column (str | None): Name of the target column (for recommendation generation).
        generate_recs (bool): Whether to generate recommendations. Default is False.
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
        tuple[DataframeInfo, dict | None]: A tuple containing:
            - DataframeInfo object with structured DataFrame information
            - Recommendations dict (or None if generate_recs is False)

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

    recommendations = None
    if generate_recs:
        recommendations = generate_recommendations(
            df, target_column, max_decimal_places, default_max_decimal_places,
            min_binning_unique_values, default_min_binning_unique_values,
            max_binning_unique_values, default_max_binning_unique_values)

    for c in range(n):
        col = df_info.columns[c]
        analyze_column_data(df[col.name], df_info.columns[c])

        # Display recommendations for this column if available
        if recommendations and col.name in recommendations:
            col_recs = recommendations[col.name]
            if col_recs:
                print("\n  Recommendations:")
                for rec_type, recommendation in col_recs.items():
                    recommendation.info()
                print()

    return df_info, recommendations


def generate_interaction_recommendations(
    df: pd.DataFrame, exclude_columns: list[str] | None = None
) -> list[FeatureInteractionRecommendation]:
    """Generate recommended feature interactions based on statistical patterns.

    Analyzes numeric and categorical columns to suggest meaningful interactions
    based on three rules:
    1. Status-Impact: Binary × High-variance continuous
    2. Resource Density: Continuous / Continuous (financial ratios)
    3. Product Utilization: Count × Duration (frequency/rate metrics)

    Args:
        df (pd.DataFrame): DataFrame to analyze for interaction opportunities.
        exclude_columns (list[str] | None): Columns to exclude from interactions
            (e.g., target column, ID columns). Default is None.

    Returns:
        list[FeatureInteractionRecommendation]: List of recommended interactions
            (may be empty if no suitable candidates found).

    Example:
        >>> df = pd.DataFrame({
        ...     'Balance': [1000, 5000, 10000],
        ...     'IsActiveMember': [0, 1, 1],
        ...     'EstimatedSalary': [50000, 75000, 100000],
        ...     'Target': [0, 1, 0]
        ... })
        >>> interactions = generate_interaction_recommendations(df, exclude_columns=['Target'])
        >>> for rec in interactions:
        ...     rec.info()
    """
    import numpy as np

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

    # Rule 1: Status-Impact (Binary × High-variance continuous)
    binary_cols = [col for col in usable_cols if df[col].nunique() == 2]

    # Filter high-variance continuous columns (must have reasonable cardinality and variance)
    high_variance_cols = []
    for col in usable_cols:
        if df[col].nunique() > 10 and col not in binary_cols:
            col_var = df[col].var()
            quantile_var = df[usable_cols].var().quantile(0.6)
            if isinstance(col_var, (int, float)) and isinstance(quantile_var, (int, float)):
                if col_var > quantile_var:
                    high_variance_cols.append(col)

    for binary_col in binary_cols:
        for cont_col in high_variance_cols:
            if binary_col != cont_col:
                interactions.append(
                    FeatureInteractionRecommendation(
                        column_name=cont_col,
                        column_name_2=binary_col,
                        interaction_type=InteractionType.STATUS_IMPACT,
                        operation="*",
                        description=f"Status-Impact interaction: {cont_col} × {binary_col}",
                        rationale=f"Multiply high-variance '{cont_col}' by binary status '{binary_col}' "
                        f"to distinguish behavior based on membership status",
                    )
                )

    # Rule 2: Resource Density (Continuous / Continuous)
    # Look for financial or resource-like columns
    financial_keywords = ["balance", "salary", "income", "revenue", "credit"]
    financial_cols = [
        col for col in usable_cols
        if any(kw in col.lower() for kw in financial_keywords)
    ]

    # Create ratios between financial columns
    for i, col1 in enumerate(financial_cols):
        for col2 in financial_cols[i + 1:]:
            # Avoid division by columns with zeros or very small values
            if (df[col2] != 0).sum() / len(df) > 0.9:  # At least 90% non-zero
                interactions.append(
                    FeatureInteractionRecommendation(
                        column_name=col1,
                        column_name_2=col2,
                        interaction_type=InteractionType.RESOURCE_DENSITY,
                        operation="/",
                        description=f"Resource Density ratio: {col1} / {col2}",
                        rationale=f"Create a ratio of '{col1}' to '{col2}' to normalize and capture "
                        f"relative financial metrics",
                    )
                )

    # Rule 3: Product Utilization (Discrete / Continuous)
    # Look for count and duration columns
    count_keywords = ["num", "count", "product"]
    duration_keywords = ["tenure", "age", "year", "month", "day"]

    count_cols = [
        col for col in usable_cols
        if any(kw in col.lower() for kw in count_keywords) and df[col].nunique() <= 20
    ]

    duration_cols = [
        col for col in usable_cols
        if any(kw in col.lower() for kw in duration_keywords)
    ]

    for count_col in count_cols:
        for dur_col in duration_cols:
            # Avoid division by zero
            if (df[dur_col] != 0).sum() / len(df) > 0.9:
                interactions.append(
                    FeatureInteractionRecommendation(
                        column_name=count_col,
                        column_name_2=dur_col,
                        interaction_type=InteractionType.PRODUCT_UTILIZATION,
                        operation="/",
                        description=f"Product Utilization rate: {count_col} / {dur_col}",
                        rationale=f"Create a rate metric '{count_col}' per '{dur_col}' to measure "
                        f"adoption velocity and utilization intensity",
                    )
                )

    return interactions
    return df_info, recommendations
