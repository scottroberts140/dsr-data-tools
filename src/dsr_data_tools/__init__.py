"""
dsr_data_tools: Generic data handling utilities for data splitting and analysis.
"""

from dsr_data_tools.analysis import (
    DataframeColumn,
    DataframeInfo,
    analyze_column_data,
    analyze_dataset,
)
from dsr_data_tools.recommendations import apply_recommendations
from dsr_data_tools.enums import (
    RecommendationType,
    EncodingStrategy,
    MissingValueStrategy,
    OutlierStrategy,
    ImbalanceStrategy,
)

__all__ = [
    "DataframeColumn",
    "DataframeInfo",
    "analyze_column_data",
    "analyze_dataset",
    "apply_recommendations",
    "RecommendationType",
    "EncodingStrategy",
    "MissingValueStrategy",
    "OutlierStrategy",
    "ImbalanceStrategy",
]

__version__ = "0.0.1"
