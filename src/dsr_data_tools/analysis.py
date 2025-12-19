from __future__ import annotations
import pandas as pd
from typing import Type
from dsr_utils import strings
from dsr_data_tools.enums import (
    RecommendationType,
    EncodingStrategy,
    MissingValueStrategy,
    OutlierStrategy,
    ImbalanceStrategy,
)
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


def generate_recommendations(
    df: pd.DataFrame,
    target_column: str | None = None,
    max_decimal_places: int | dict[str, int] | None = None,
    default_max_decimal_places: int | None = None
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
            values = sorted(series.dropna().unique().tolist())
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
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                integer_count = non_null_series.apply(
                    lambda x: x.is_integer()).sum()
                if integer_count == len(non_null_series):
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
                col_max_decimal_places = max_decimal_places.get(col_name, default_max_decimal_places)
            else:
                # If int, use it directly
                col_max_decimal_places = max_decimal_places
            
            # Skip if already recommended for int64 conversion or no valid precision specified
            if col_max_decimal_places is not None and 'int64_conversion' not in col_recommendations:
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    # Check if rounding to col_max_decimal_places would lose significant data
                    rounded_series = non_null_series.round(col_max_decimal_places)

                    # Determine if conversion to int64 is possible (all values are integers after rounding)
                    can_convert_to_int = (
                        col_max_decimal_places == 0 and
                        rounded_series.apply(
                            lambda x: x.is_integer()).sum() == len(rounded_series)
                    )

                    rec = DecimalPrecisionRecommendation(
                        type=RecommendationType.DECIMAL_PRECISION_OPTIMIZATION,
                        column_name=col_name,
                        description=f"Column '{col_name}' can have decimal precision optimized to {col_max_decimal_places} places.",
                        max_decimal_places=col_max_decimal_places,
                        min_value=float(non_null_series.min()),
                        max_value=float(non_null_series.max()),
                        convert_to_int=can_convert_to_int
                    )
                    col_recommendations['decimal_precision_optimization'] = rec

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
            min_value = series.min()

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
            class_counts = series.value_counts()
            max_class_percentage = (class_counts.max() / total_rows) * 100

            if max_class_percentage > 70:
                rec = ClassImbalanceRecommendation(
                    type=RecommendationType.CLASS_IMBALANCE,
                    column_name=col_name,
                    description=f"Target variable '{col_name}' shows class imbalance ({max_class_percentage:.1f}% majority class).",
                    majority_percentage=max_class_percentage,
                    strategy=ImbalanceStrategy.CLASS_WEIGHT
                )
                col_recommendations['class_imbalance'] = rec

        # 7. Suggest binning for numeric columns (e.g., Age)
        if is_numeric and col_name.lower() in ['age', 'years']:
            # Use describe() percentiles to suggest bins
            desc = series.describe()
            bins = [series.min() - 1, desc['25%'], desc['50%'],
                    desc['75%'], series.max()]
            labels = ['Low', 'Medium_Low', 'Medium_High', 'High']

            rec = BinningRecommendation(
                type=RecommendationType.BINNING,
                column_name=col_name,
                description=f"Column '{col_name}' could be binned into {len(labels)} categories.",
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
        integer_value_count = series.apply(lambda x: x.is_integer()).sum()
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
    default_max_decimal_places: int | None = None
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

    Returns:
        tuple[DataframeInfo, dict | None]: A tuple containing:
            - DataframeInfo object with structured DataFrame information
            - Recommendations dict (or None if generate_recs is False)

    Example:
        >>> df = pd.DataFrame({
        ...     'name': ['Alice', 'Bob', 'Charlie'],
        ...     'age': [25, 30, 35],
        ...     'salary': [50000.0, 60000.5, 75000.0]
        ... })
        >>> info, recs = analyze_dataset(df, generate_recs=True)
        # Prints comprehensive analysis of all columns and returns recommendations
    """
    df_info = DataframeInfo(df)
    df_info.info()

    n = len(df_info.columns)

    recommendations = None
    if generate_recs:
        recommendations = generate_recommendations(
            df, target_column, max_decimal_places, default_max_decimal_places)

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
