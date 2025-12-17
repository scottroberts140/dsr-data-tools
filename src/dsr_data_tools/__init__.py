"""
dsr_data_tools: Generic data handling utilities for data splitting and analysis.
"""

from dsr_data_tools.analysis import (
    DataframeColumn,
    DataframeInfo,
    analyze_column_data,
    analyze_dataset,
)

__all__ = [
    "DataframeColumn",
    "DataframeInfo",
    "analyze_column_data",
    "analyze_dataset",
]

__version__ = "0.0.1"
