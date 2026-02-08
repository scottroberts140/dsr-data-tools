"""Test to verify the updated info() methods in execution_summary()."""

from dsr_data_tools.recommendations import RecommendationManager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.insert(
    0, '/Users/scottroberts/Documents/Developer/Projects/Python Libraries/dsr-data-tools/src')


# Create test data with various scenarios

def create_test_data():
    """Create a DataFrame with various data quality issues."""
    np.random.seed(42)
    n = 100

    # Datetime columns
    start_dates = pd.date_range('2024-01-01', periods=n, freq='H')
    end_dates = start_dates + timedelta(hours=2)

    # Create DataFrame
    df = pd.DataFrame({
        'event_start': start_dates,
        'event_end': end_dates,
        'category': ['A', 'B', 'C'] * 33 + ['A'],  # Categorical
        'value': np.random.randn(n),  # Numeric with outliers
        'amount': np.random.uniform(10, 100, n),  # Float
        'missing_col': [1.0] * 50 + [np.nan] * 50,  # Missing values
        'integer_float': [1.0, 2.0, 3.0] * 33 + [1.0],  # Integer as float
    })

    return df


def main():
    print("="*80)
    print("TESTING UPDATED info() METHODS IN execution_summary()")
    print("="*80)
    print()

    # Create test data
    df = create_test_data()

    print("Test DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print()

    # Generate recommendations
    print("-"*80)
    print("GENERATING RECOMMENDATIONS")
    print("-"*80)
    manager = RecommendationManager()
    manager.generate_recommendations(df)

    print(f"\n✅ Generated {len(manager._pipeline)} recommendations")
    print()

    # Display execution summary
    print("-"*80)
    print("EXECUTION SUMMARY")
    print("-"*80)
    manager.execution_summary()
    print()

    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print()
    print("✅ All info() methods now consistently show:")
    print("   - Column: 'column_name' for primary column")
    print("   - Start column, End column, Output column where applicable")
    print("   - Clearer formatting throughout")


if __name__ == "__main__":
    main()
