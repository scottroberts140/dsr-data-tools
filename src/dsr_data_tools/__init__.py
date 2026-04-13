"""
dsr_data_tools: Generic data handling utilities for data splitting and analysis.
"""

from importlib.metadata import PackageNotFoundError, version

from dsr_data_tools.analysis import (
    DataframeColumn,
    DataframeInfo,
    analyze_column_data,
    analyze_dataset,
    generate_interaction_recommendations,
)
from dsr_data_tools.enums import (
    BitDepth,
    ColumnHintType,
    EncodingStrategy,
    ImbalanceStrategy,
    InteractionType,
    MissingValueStrategy,
    OutlierHandlingStrategy,
    OutlierStrategy,
    RecommendationType,
    RoundingMode,
)
from dsr_data_tools.recommendations import (
    AggregationRecommendation,
    BinningRecommendation,
    BooleanClassificationRecommendation,
    CategoricalConversionRecommendation,
    ClassImbalanceRecommendation,
    ColumnHint,
    DatetimeConversionRecommendation,
    DatetimeDurationRecommendation,
    DecimalPrecisionRecommendation,
    EncodingRecommendation,
    FeatureExtractionRecommendation,
    FeatureInteractionRecommendation,
    FloatConversionRecommendation,
    IntegerConversionRecommendation,
    MissingValuesRecommendation,
    NonInformativeRecommendation,
    OutlierDetectionRecommendation,
    OutlierHandlingRecommendation,
    Recommendation,
    RecommendationManager,
    ValueReplacementRecommendation,
)

__all__ = [
    "DataframeColumn",
    "DataframeInfo",
    "analyze_column_data",
    "analyze_dataset",
    "generate_interaction_recommendations",
    "RecommendationManager",
    "Recommendation",
    "NonInformativeRecommendation",
    "MissingValuesRecommendation",
    "EncodingRecommendation",
    "ClassImbalanceRecommendation",
    "OutlierDetectionRecommendation",
    "OutlierHandlingRecommendation",
    "CategoricalConversionRecommendation",
    "BooleanClassificationRecommendation",
    "BinningRecommendation",
    "IntegerConversionRecommendation",
    "FloatConversionRecommendation",
    "DecimalPrecisionRecommendation",
    "ValueReplacementRecommendation",
    "FeatureInteractionRecommendation",
    "DatetimeConversionRecommendation",
    "FeatureExtractionRecommendation",
    "DatetimeDurationRecommendation",
    "ColumnHint",
    "AggregationRecommendation",
    "RecommendationType",
    "EncodingStrategy",
    "MissingValueStrategy",
    "OutlierStrategy",
    "OutlierHandlingStrategy",
    "ImbalanceStrategy",
    "InteractionType",
    "ColumnHintType",
    "RoundingMode",
    "BitDepth",
    "__version__",
]

try:
    __version__ = version("dsr-data-tools")
except PackageNotFoundError:
    __version__ = "unknown"
