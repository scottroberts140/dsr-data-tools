"""
dsr_data_tools: Generic data handling utilities for data splitting and analysis.
"""

from dsr_data_tools.analysis import (
    DataframeColumn,
    DataframeInfo,
    analyze_column_data,
    analyze_dataset,
    generate_interaction_recommendations,
)
from dsr_data_tools.recommendations import apply_recommendations
from dsr_data_tools.enums import (
    RecommendationType,
    EncodingStrategy,
    MissingValueStrategy,
    OutlierStrategy,
    ImbalanceStrategy,
    InteractionType,
)

__all__ = [
    "DataframeColumn",
    "DataframeInfo",
    "analyze_column_data",
    "analyze_dataset",
    "generate_interaction_recommendations",
    "apply_recommendations",
    "RecommendationType",
    "EncodingStrategy",
    "MissingValueStrategy",
    "OutlierStrategy",
    "ImbalanceStrategy",
    "InteractionType",
]

__version__ = "0.0.1"
