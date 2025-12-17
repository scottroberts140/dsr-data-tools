from __future__ import annotations
import pandas as pd
from typing import Type
from dsr_utils import strings


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

    analysis = f"""
Column:             {dataframe_column.name}
Data type:          {dataframe_column.data_type.name}
Non-null:           {dataframe_column.non_null_count}
N/A count:          {series.isna().sum()}
Unique values:      {series.nunique()}
Min value:          {series.min()}
Max value:          {series.max()}"""

    print(analysis)

    if is_float_type:
        print(integer_analysis)

    if is_object_data_type:
        print(object_analysis)


def analyze_dataset(df: pd.DataFrame):
    """Perform comprehensive analysis of all columns in a DataFrame.

    Displays overall DataFrame information (row count, duplicates) followed by
    detailed analysis of each column including data types, null counts, unique
    values, and type-specific statistics.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        DataframeInfo: Object containing structured information about the DataFrame.

    Example:
        >>> df = pd.DataFrame({
        ...     'name': ['Alice', 'Bob', 'Charlie'],
        ...     'age': [25, 30, 35],
        ...     'salary': [50000.0, 60000.5, 75000.0]
        ... })
        >>> info = analyze_dataset(df)
        # Prints comprehensive analysis of all columns
    """
    df_info = DataframeInfo(df)
    df_info.info()

    n = len(df_info.columns)

    for c in range(n):
        col = df_info.columns[c]
        analyze_column_data(df[col.name], df_info.columns[c])

    return df_info
