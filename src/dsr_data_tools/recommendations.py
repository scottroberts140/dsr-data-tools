from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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
                result[self.column_name].fillna(
                    result[self.column_name].median(), inplace=True)
            else:
                result[self.column_name].fillna(
                    result[self.column_name].mode()[0], inplace=True)

        return result


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
            result = pd.get_dummies(result, columns=[self.column_name])

        elif self.encoder_type == EncodingStrategy.LABEL:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            result[self.column_name] = le.fit_transform(
                result[self.column_name])

        elif self.encoder_type == EncodingStrategy.ORDINAL:
            from sklearn.preprocessing import OrdinalEncoder
            oe = OrdinalEncoder()
            result[self.column_name] = oe.fit_transform(
                result[[self.column_name]])

        return result


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
            result[self.column_name] = scaler.fit_transform(
                result[[self.column_name]])

        elif self.strategy == OutlierStrategy.ROBUST_SCALER:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            result[self.column_name] = scaler.fit_transform(
                result[[self.column_name]])

        elif self.strategy == OutlierStrategy.REMOVE:
            # Remove rows where value exceeds 1.5 * IQR beyond quartiles
            Q1 = result[self.column_name].quantile(0.25)
            Q3 = result[self.column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            result = result[(result[self.column_name] >= lower_bound) & (
                result[self.column_name] <= upper_bound)]

        return result


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
        result[self.column_name] = pd.cut(
            result[self.column_name],
            bins=self.bins,
            labels=self.labels,
            right=True,
            include_lowest=True
        )
        # One-hot encode the binned column
        result = pd.get_dummies(result, columns=[self.column_name])
        return result


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
