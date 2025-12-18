from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

import pandas as pd

from dsr_data_tools.enums import (
    RecommendationType,
    EncodingStrategy,
    MissingValueStrategy,
    OutlierStrategy,
    ImbalanceStrategy,
)


@dataclass
class Recommendation(ABC):
    """
    Abstract base class for dataset preparation recommendations.

    Each recommendation represents a suggested action to improve data quality
    or prepare the dataset for machine learning.
    """

    type: RecommendationType
    column_name: str
    description: str

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply this recommendation to a dataset.

        Args:
            df: Input DataFrame

        Returns:
            Modified DataFrame with recommendation applied
        """
        pass

    @abstractmethod
    def info(self) -> None:
        """Display formatted information about this recommendation."""
        pass


@dataclass
class NonInformativeRecommendation(Recommendation):
    """Recommendation to remove non-informative columns."""

    reason: str
    """Explanation for why column is non-informative (e.g., 'High cardinality', 'Unique count == row count')"""

    def __post_init__(self):
        """Set type to NON_INFORMATIVE."""
        self.type = RecommendationType.NON_INFORMATIVE

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the non-informative column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with column removed
        """
        return df.drop(columns=[self.column_name])

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: NON_INFORMATIVE")
        print(f"    Reason: {self.reason}")
        print(f"    Action: Drop column '{self.column_name}'")


@dataclass
class MissingValuesRecommendation(Recommendation):
    """Recommendation for handling missing values."""

    missing_count: int
    """Number of missing values in the column"""

    missing_percentage: float
    """Percentage of missing values (0-100)"""

    strategy: MissingValueStrategy
    """Recommended strategy for handling missing values"""

    def __post_init__(self):
        """Set type to MISSING_VALUES."""
        self.type = RecommendationType.MISSING_VALUES

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing value strategy to the column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing value strategy applied
        """
        result = df.copy()

        if self.strategy == MissingValueStrategy.DROP_ROWS:
            result = result.dropna(subset=[self.column_name])

        elif self.strategy == MissingValueStrategy.DROP_COLUMN:
            result = result.drop(columns=[self.column_name])

        elif self.strategy == MissingValueStrategy.IMPUTE:
            # Default to median for numeric, mode for categorical
            if pd.api.types.is_numeric_dtype(result[self.column_name]):
                fill_value = result[self.column_name].median()
                result[self.column_name] = result[self.column_name].fillna(
                    fill_value)
            else:
                mode_value = result[self.column_name].mode()
                if len(mode_value) > 0:
                    result[self.column_name] = result[self.column_name].fillna(
                        mode_value[0])
                else:
                    result[self.column_name] = result[self.column_name].fillna(
                        'Unknown')

        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: MISSING_VALUES")
        print(
            f"    Missing count: {self.missing_count} ({self.missing_percentage:.2f}%)")
        print(f"    Strategy: {self.strategy.value}")
        print(f"    Action: {self._get_strategy_description()}")

    def _get_strategy_description(self) -> str:
        """Get human-readable description of the strategy."""
        if self.strategy == MissingValueStrategy.DROP_ROWS:
            return f"Remove {self.missing_count} rows with missing values"
        elif self.strategy == MissingValueStrategy.DROP_COLUMN:
            return f"Drop column '{self.column_name}' entirely"
        elif self.strategy == MissingValueStrategy.IMPUTE:
            return "Impute missing values using mean/median/mode"
        return "Unknown strategy"


@dataclass
class EncodingRecommendation(Recommendation):
    """Recommendation for encoding categorical columns."""

    encoder_type: EncodingStrategy
    """Recommended encoding strategy"""

    unique_values: int
    """Number of unique values in the column"""

    def __post_init__(self):
        """Set type to ENCODING."""
        self.type = RecommendationType.ENCODING

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply encoding strategy to the column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with column encoded
        """
        result = df.copy()

        if self.encoder_type == EncodingStrategy.ONEHOT:
            # One-hot encode - creates binary columns for each category
            result = pd.get_dummies(
                result, columns=[self.column_name], drop_first=False)

        elif self.encoder_type == EncodingStrategy.LABEL:
            # Label encode - assigns integer to each category
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            # Handle potential NaN values
            mask = result[self.column_name].notna()
            encoded_values = le.fit_transform(
                result.loc[mask, self.column_name].astype(str))
            # Create new series with encoded values and NaN for masked rows
            new_values = pd.Series(
                index=result.index, dtype='Int64')  # nullable int
            new_values[mask] = encoded_values
            result[self.column_name] = new_values

        elif self.encoder_type == EncodingStrategy.ORDINAL:
            # Ordinal encode - preserves order
            from sklearn.preprocessing import OrdinalEncoder
            oe = OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1)
            result[self.column_name] = oe.fit_transform(
                result[[self.column_name]])

        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: ENCODING")
        print(f"    Unique values: {self.unique_values}")
        print(f"    Encoder type: {self.encoder_type.value}")
        print(
            f"    Action: Apply {self.encoder_type.value} encoding to '{self.column_name}'")


@dataclass
class ClassImbalanceRecommendation(Recommendation):
    """Recommendation for handling class imbalance in target variable."""

    majority_percentage: float
    """Percentage of majority class"""

    strategy: ImbalanceStrategy
    """Recommended strategy for handling imbalance"""

    def __post_init__(self):
        """Set type to CLASS_IMBALANCE."""
        self.type = RecommendationType.CLASS_IMBALANCE

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply class imbalance strategy.

        Note: This is a simplified implementation. SMOTE and upsampling/downsampling
        should be applied during cross-validation to prevent data leakage.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame (or recommendation for model configuration)
        """
        # This recommendation is typically handled at model training time,
        # not during data preparation. Return df unchanged as a placeholder.
        return df

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: CLASS_IMBALANCE")
        print(f"    Majority class: {self.majority_percentage:.2f}%")
        print(f"    Strategy: {self.strategy.value}")
        print(f"    Action: Apply {self.strategy.value} during model training")


@dataclass
class OutlierDetectionRecommendation(Recommendation):
    """Recommendation for handling outliers."""

    strategy: OutlierStrategy
    """Recommended strategy for handling outliers"""

    max_value: float
    """Maximum value in the column"""

    mean_value: float
    """Mean value of the column"""

    def __post_init__(self):
        """Set type to OUTLIER_DETECTION."""
        self.type = RecommendationType.OUTLIER_DETECTION

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier handling strategy to the column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with outlier strategy applied
        """
        result = df.copy()

        if self.strategy == OutlierStrategy.SCALING:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            # Handle NaN values
            mask = result[self.column_name].notna()
            scaled_values = scaler.fit_transform(
                result.loc[mask, [self.column_name]])
            # Convert to float64 to avoid dtype incompatibility
            result[self.column_name] = result[self.column_name].astype(
                'float64')
            result.loc[mask, self.column_name] = scaled_values.flatten()

        elif self.strategy == OutlierStrategy.ROBUST_SCALER:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            # Handle NaN values
            mask = result[self.column_name].notna()
            scaled_values = scaler.fit_transform(
                result.loc[mask, [self.column_name]])
            # Convert to float64 to avoid dtype incompatibility
            result[self.column_name] = result[self.column_name].astype(
                'float64')
            result.loc[mask, self.column_name] = scaled_values.flatten()

        elif self.strategy == OutlierStrategy.REMOVE:
            # Remove rows where value exceeds 1.5 * IQR beyond quartiles
            Q1 = result[self.column_name].quantile(0.25)
            Q3 = result[self.column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            result = result[(result[self.column_name] >= lower_bound) &
                            (result[self.column_name] <= upper_bound)].reset_index(drop=True)

        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: OUTLIER_DETECTION")
        print(
            f"    Max value: {self.max_value:.2f}, Mean: {self.mean_value:.2f}")
        print(f"    Strategy: {self.strategy.value}")
        print(
            f"    Action: Apply {self.strategy.value} to handle outliers in '{self.column_name}'")


@dataclass
class BooleanClassificationRecommendation(Recommendation):
    """Recommendation to treat numeric column as boolean."""

    values: list[Any]
    """The two unique values in the column"""

    def __post_init__(self):
        """Set type to BOOLEAN_CLASSIFICATION."""
        self.type = RecommendationType.BOOLEAN_CLASSIFICATION

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert numeric column to boolean type.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with column converted to boolean
        """
        result = df.copy()
        result[self.column_name] = result[self.column_name].astype(bool)
        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: BOOLEAN_CLASSIFICATION")
        print(f"    Values: {self.values}")
        print(f"    Action: Convert '{self.column_name}' to boolean type")


@dataclass
class BinningRecommendation(Recommendation):
    """Recommendation to bin numeric column into categorical ranges."""

    bins: list[float]
    """Bin edges for pd.cut()"""

    labels: list[str]
    """Labels for each bin"""

    def __post_init__(self):
        """Set type to BINNING."""
        self.type = RecommendationType.BINNING

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bin the numeric column into categorical ranges.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with column binned and one-hot encoded
        """
        result = df.copy()
        try:
            result[self.column_name] = pd.cut(
                result[self.column_name],
                bins=self.bins,
                labels=self.labels,
                right=True,
                include_lowest=True
            )
        except Exception as e:
            print(f"Warning: Could not bin column '{self.column_name}': {e}")
            return result

        # One-hot encode the binned column
        result = pd.get_dummies(
            result, columns=[self.column_name], drop_first=False)
        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: BINNING")
        print(f"    Bins: {self.bins}")
        print(f"    Labels: {self.labels}")
        print(
            f"    Action: Bin '{self.column_name}' into {len(self.labels)} categories and encode")


def create_recommendation(
    rec_type: RecommendationType,
    column_name: str,
    description: str,
    **kwargs
) -> Recommendation:
    """
    Factory function to create appropriate Recommendation subclass.

    Args:
        rec_type: Type of recommendation to create
        column_name: Name of the column this recommendation applies to
        description: Human-readable description of the recommendation
        **kwargs: Additional type-specific keyword arguments

    Returns:
        Recommendation instance of the appropriate subclass

    Raises:
        ValueError: If rec_type is not recognized
    """
    recommendation_classes = {
        RecommendationType.NON_INFORMATIVE: NonInformativeRecommendation,
        RecommendationType.MISSING_VALUES: MissingValuesRecommendation,
        RecommendationType.ENCODING: EncodingRecommendation,
        RecommendationType.CLASS_IMBALANCE: ClassImbalanceRecommendation,
        RecommendationType.OUTLIER_DETECTION: OutlierDetectionRecommendation,
        RecommendationType.BOOLEAN_CLASSIFICATION: BooleanClassificationRecommendation,
        RecommendationType.BINNING: BinningRecommendation,
    }

    if rec_type not in recommendation_classes:
        raise ValueError(f"Unknown recommendation type: {rec_type}")

    recommendation_class = recommendation_classes[rec_type]
    return recommendation_class(
        type=rec_type,
        column_name=column_name,
        description=description,
        **kwargs
    )


def apply_recommendations(
    df: pd.DataFrame,
    recommendations: Mapping[str, Mapping[str, Recommendation]
                             | Recommendation] | None,
    exclude_types: list[RecommendationType] | None = None,
    target_column: str | None = None
) -> pd.DataFrame:
    """
    Apply all recommendations from a dictionary to a DataFrame.

    Handles both flat dictionaries (column_name -> Recommendation) and nested
    dictionaries (column_name -> recommendation_type -> Recommendation).
    If recommendations is None, returns the DataFrame unchanged.

    Applies each recommendation sequentially, with each subsequent 
    recommendation working on the output of the previous one.

    Args:
        df: Input DataFrame. Should be the ORIGINAL, untransformed DataFrame.
        recommendations: Mapping of column names to Recommendation objects.
            Can be flat (str -> Recommendation) or nested (str -> Mapping -> Recommendation),
            or None to skip applying recommendations.
        exclude_types: Optional list of RecommendationType values to exclude from being applied.
            For example, [RecommendationType.BINNING] would skip binning recommendations.
        target_column: Optional name of the target column. If provided, no recommendations
            will be applied to this column to preserve its discrete class values for
            classification tasks.

    Returns:
        DataFrame with all recommendations applied (except excluded types)

    Warning:
        Only apply this function to the original, untransformed DataFrame. Applying
        recommendations to an already-transformed DataFrame can cause errors or unexpected
        behavior, as some transformations (e.g., dropping columns, encoding) cannot be
        safely applied multiple times.

        For multiple analysis phases with different recommendation sets, always apply
        recommendations to the original DataFrame:

        Example:
            >>> # CORRECT: Each phase starts from the original DataFrame
            >>> df_baseline = ddt.apply_recommendations(df_original, recs,
            ...     exclude_types=[ddt.RecommendationType.BINNING], target_column='Exited')
            >>> df_with_binning = ddt.apply_recommendations(df_original, recs, target_column='Exited')
            >>>
            >>> # INCORRECT: Applying recommendations to already-transformed data
            >>> df_v1 = ddt.apply_recommendations(df_original, recs, target_column='Exited')
            >>> df_v2 = ddt.apply_recommendations(df_v1, recs, target_column='Exited')  # Don't do this!

    Example:
        >>> # Apply all recommendations except binning, preserving target column
        >>> result_df = apply_recommendations(df, recs, exclude_types=[RecommendationType.BINNING],
        ...     target_column='target')
    """
    result_df = df.copy()

    # Handle None case
    if recommendations is None:
        return result_df

    # Default to empty list if not provided
    if exclude_types is None:
        exclude_types = []

    for column_name, value in recommendations.items():
        # Skip target column to preserve discrete class values
        if target_column is not None and column_name == target_column:
            continue
            
        # Handle nested dictionary structure
        if isinstance(value, dict):
            for rec_type, recommendation in value.items():
                # Skip excluded recommendation types
                if recommendation.type in exclude_types:
                    continue
                try:
                    result_df = recommendation.apply(result_df)
                except Exception as e:
                    print(f"Warning: Failed to apply {recommendation.type.value} "
                          f"recommendation for '{column_name}': {str(e)}")
        # Handle flat dictionary structure
        else:
            recommendation = value
            # Skip excluded recommendation types
            if recommendation.type in exclude_types:
                continue
            try:
                result_df = recommendation.apply(result_df)
            except Exception as e:
                print(f"Warning: Failed to apply {recommendation.type.value} "
                      f"recommendation for '{column_name}': {str(e)}")
    return result_df
