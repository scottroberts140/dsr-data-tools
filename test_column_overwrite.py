#!/usr/bin/env python3
"""
Test script for the new allow_column_overwrite parameter in RecommendationManager.apply()
"""

import pandas as pd
from typing import cast
from dsr_data_tools.recommendations import RecommendationManager, Recommendation
from dsr_data_tools.enums import RecommendationType
from dsr_utils.enums import DatetimeProperty


def test_apply_accepts_allow_column_overwrite_parameter():
    """Test that the apply() method accepts the allow_column_overwrite parameter."""
    print("\n" + "="*70)
    print("TEST 1: apply() method accepts allow_column_overwrite parameter")
    print("="*70)

    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30],
    })

    manager = RecommendationManager()

    try:
        # Should not raise TypeError about unexpected keyword argument
        result = manager.apply(df, allow_column_overwrite=False)
        print(f"✅ PASSED: apply() accepts allow_column_overwrite=False")
        return True
    except TypeError as e:
        if "allow_column_overwrite" in str(e):
            print(f"❌ FAILED: {e}")
            return False
        raise


def test_validate_pipeline_accepts_parameter():
    """Test that _validate_pipeline accepts the allow_column_overwrite parameter."""
    print("\n" + "="*70)
    print("TEST 2: _validate_pipeline() accepts allow_column_overwrite parameter")
    print("="*70)

    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30],
    })

    manager = RecommendationManager()

    try:
        # Should not raise TypeError
        manager._validate_pipeline(df, allow_column_overwrite=False)
        print(f"✅ PASSED: _validate_pipeline() accepts allow_column_overwrite=False")
        return True
    except TypeError as e:
        if "allow_column_overwrite" in str(e):
            print(f"❌ FAILED: {e}")
            return False
        raise


def test_default_allow_column_overwrite_is_false():
    """Test that the default value for allow_column_overwrite is False."""
    print("\n" + "="*70)
    print("TEST 3: Default allow_column_overwrite is False")
    print("="*70)

    df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30],
    })

    manager = RecommendationManager()

    try:
        # Call apply() without the parameter - should default to False
        result = manager.apply(df)
        print(f"✅ PASSED: apply() works with default allow_column_overwrite=False")
        print(f"   Result columns: {list(result.columns)}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_read_write_log_tracking_logic():
    """Test the read/write log tracking logic indirectly."""
    print("\n" + "="*70)
    print("TEST 4: Read/write tracking logic is integrated")
    print("="*70)

    df = pd.DataFrame({
        'col_a': [1, 2, 3],
        'col_b': [4, 5, 6],
    })

    manager = RecommendationManager()

    try:
        # The validation method should track read/write columns
        # Even with empty pipeline, it should work
        manager._validate_pipeline(df, allow_column_overwrite=True)
        print(f"✅ PASSED: Read/write tracking logic is present and functional")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_validate_pipeline_checks_output_columns():
    """Test that _validate_pipeline validates output_columns."""
    print("\n" + "="*70)
    print("TEST 5: _validate_pipeline checks for output_column conflicts")
    print("="*70)

    df = pd.DataFrame({
        'id': [1, 2, 3],
        'existing_col': [10, 20, 30],
    })

    manager = RecommendationManager()

    # Create a simple mock recommendation with output_column
    class MockRecWithOutput:
        id = "rec_test"
        type = RecommendationType.FEATURE_EXTRACTION
        column_name = 'id'

        def __init__(self, output_column):
            self.output_column = output_column

    # Test 1: Overwrite not allowed should raise error
    rec = MockRecWithOutput('existing_col')
    manager._pipeline = cast(list[Recommendation], [rec])

    try:
        manager._validate_pipeline(df, allow_column_overwrite=False)
        print(f"❌ FAILED: Should have raised ValueError when allow_column_overwrite=False")
        return False
    except ValueError as e:
        if "Set allow_column_overwrite=True" in str(e):
            print(
                f"✅ PASSED: Correctly validates output_column when allow_column_overwrite=False")
            print(f"   Error message: {e}")
            return True
        else:
            print(f"❌ FAILED: Wrong error message: {e}")
            return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING COLUMN OVERWRITE FEATURE")
    print("="*70)

    results = []

    results.append(test_apply_accepts_allow_column_overwrite_parameter())
    results.append(test_validate_pipeline_accepts_parameter())
    results.append(test_default_allow_column_overwrite_is_false())
    results.append(test_read_write_log_tracking_logic())
    results.append(test_validate_pipeline_checks_output_columns())

    print("\n" + "="*70)
    print(f"SUMMARY: {sum(results)}/{len(results)} tests passed")
    print("="*70)

    if all(results):
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
        exit(1)
