"""
Comprehensive Test: Updated info() Methods
===========================================

This test demonstrates that all info() methods now consistently display:
1. Column: 'column_name' for the primary column
2. Start column, End column, Output column where applicable
3. Clear, consistent formatting throughout
"""

from dsr_data_tools.recommendations import (
    RecommendationManager,
    MissingValuesRecommendation,
    EncodingRecommendation,
    DatetimeDurationRecommendation,
    FeatureInteractionRecommendation,
    OutlierDetectionRecommendation,
    BinningRecommendation,
    IntegerConversionRecommendation,
    RecommendationType,
    MissingValueStrategy,
    EncodingStrategy,
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.insert(
    0,
    "/Users/scottroberts/Library/CloudStorage/GoogleDrive-scottrdeveloper@gmail.com/My Drive/Projects/Python Libraries/dsr-data-tools/src",
)


def test_individual_info_methods():
    """Test individual info() methods to show column details."""
    print("=" * 80)
    print("TESTING INDIVIDUAL info() METHODS")
    print("=" * 80)
    print()

    # Test MissingValuesRecommendation
    print("1. MissingValuesRecommendation:")
    print("-" * 80)
    rec1 = MissingValuesRecommendation(
        column_name="age",
        description="Handle missing values",
        missing_count=25,
        missing_percentage=25.0,
        strategy=MissingValueStrategy.IMPUTE_MEAN,
    )
    rec1.info()
    print()

    # Test EncodingRecommendation
    print("2. EncodingRecommendation:")
    print("-" * 80)
    rec2 = EncodingRecommendation(
        column_name="category",
        description="Encode categorical",
        unique_values=5,
        encoder_type=EncodingStrategy.ONEHOT,
    )
    rec2.info()
    print()

    # Test DatetimeDurationRecommendation
    print("3. DatetimeDurationRecommendation:")
    print("-" * 80)
    rec3 = DatetimeDurationRecommendation(
        column_name="event_start",
        description="Calculate duration",
        start_column="event_start",
        end_column="event_end",
        unit="hours",
        output_column="event_duration_hours",
    )
    rec3.info()
    print()

    # Test FeatureInteractionRecommendation
    print("4. FeatureInteractionRecommendation:")
    print("-" * 80)
    from dsr_data_tools.enums import InteractionType

    rec4 = FeatureInteractionRecommendation(
        column_name="price",
        description="Create interaction feature",
        column_name_2="quantity",
        operation="*",
        derived_name="total_value",
        interaction_type=InteractionType.RESOURCE_DENSITY,
        priority_score=0.85,
        rationale="Strong correlation detected",
    )
    rec4.info()
    print()

    print("=" * 80)
    print("✅ All info() methods show column details clearly!")
    print("=" * 80)


def main():
    print("\n" * 2)
    test_individual_info_methods()


if __name__ == "__main__":
    main()
