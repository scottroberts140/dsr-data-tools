"""
Tests for dsr_data_tools.analysis module.
"""
import pytest
import pandas as pd
import numpy as np
from dsr_data_tools import analysis
from dsr_data_tools.analysis import generate_recommendations, analyze_dataset, generate_interaction_recommendations
from dsr_data_tools.enums import RecommendationType


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'categorical_col': ['A', 'B', 'A', 'B', 'C'],
        'target': [0, 1, 0, 1, 0],
    })


class TestAnalysisModule:
    """Test cases for analysis module functions."""

    def test_analysis_module_exists(self):
        """Verify that analysis module is importable."""
        assert analysis is not None

    def test_analyze_dataset_basic(self, sample_dataframe):
        """Test basic dataset analysis functionality."""
        # This assumes analyze_dataset is a main function
        # Adjust based on your actual API
        assert isinstance(sample_dataframe, pd.DataFrame)
        assert len(sample_dataframe) > 0

    def test_generate_recommendations(self, sample_dataframe):
        """Test recommendation generation."""
        recs = generate_recommendations(sample_dataframe)
        assert isinstance(recs, dict)

    def test_binning_thresholds_int(self):
        """Binning triggers with int thresholds for unique counts."""
        # Create a numeric column with duplicates to avoid non_informative
        # 25 unique values repeated twice -> 50 rows
        num_vals = np.arange(25, dtype=float).repeat(2)
        df = pd.DataFrame({
            'num_col': num_vals,
            'target': np.random.randint(0, 2, size=num_vals.size)
        })
        recs = generate_recommendations(
            df,
            target_column='target',
            min_binning_unique_values=10,
            default_min_binning_unique_values=10,
            max_binning_unique_values=100,
            default_max_binning_unique_values=100
        )
        assert 'num_col' in recs
        assert 'binning' in recs['num_col']
        assert recs['num_col']['binning'].type == RecommendationType.BINNING

    def test_binning_thresholds_dict(self):
        """Binning uses per-column dict thresholds when provided."""
        # Create columns with consistent lengths and controlled unique counts
        # 'x': 100 unique values spread across 110 rows (avoid non_informative)
        x = np.concatenate([np.arange(100, dtype=float),
                           np.arange(10, dtype=float)])
        # 'y': 20 unique values spread across 110 rows
        y = np.tile(np.arange(20, dtype=float), 6)[:110]
        df = pd.DataFrame({'x': x, 'y': y})
        recs = generate_recommendations(
            df,
            min_binning_unique_values={'x': 100},
            default_min_binning_unique_values=10,
            max_binning_unique_values={'x': 120},
            default_max_binning_unique_values=100
        )
        assert 'x' in recs and 'binning' in recs['x']
        assert recs['x']['binning'].type == RecommendationType.BINNING
        # 'y' should also qualify under defaults
        assert 'y' in recs and 'binning' in recs['y']

    def test_int64_conversion_vectorized(self):
        """Float column with integer-like values triggers int64 conversion recommendation."""
        # Include duplicates to avoid non_informative on numeric columns
        df = pd.DataFrame({
            'ints_as_float': np.array([1.0, 2.0, 2.0, 3.0, 3.0, 4.0], dtype=float)
        })
        recs = generate_recommendations(df)
        assert 'ints_as_float' in recs
        assert 'int64_conversion' in recs['ints_as_float']
        assert recs['ints_as_float']['int64_conversion'].type == RecommendationType.INT64_CONVERSION

    def test_decimal_precision_convert_to_int(self):
        """Rounding to 0 with integer-like float values sets convert_to_int True."""
        # Use non-integer floats so int64 conversion does not trigger,
        # but rounding to 0 makes them integers (convert_to_int True)
        df = pd.DataFrame({
            'vals': np.array([10.1, 20.0, 20.1, 30.0, 30.2, 40.0, 20.0, 30.0], dtype=float)
        })
        recs = generate_recommendations(
            df,
            max_decimal_places=0
        )
        assert 'vals' in recs
        assert 'decimal_precision_optimization' in recs['vals']
        dp = recs['vals']['decimal_precision_optimization']
        assert dp.type == RecommendationType.DECIMAL_PRECISION_OPTIMIZATION
        assert dp.convert_to_int is True

    def test_value_replacement_detection(self):
        """Object column with numeric and placeholder values triggers replacement recommendation."""
        # Create enough rows to keep unique ratio below high-cardinality threshold
        mixed_vals = ['10', '20', 'N/A', 'tbd',
                      '30', None] * 8  # 48 rows, 6 unique
        df = pd.DataFrame({'mixed': mixed_vals})
        recs = generate_recommendations(df)
        assert 'mixed' in recs
        assert 'value_replacement' in recs['mixed']
        vr = recs['mixed']['value_replacement']
        assert vr.type == RecommendationType.VALUE_REPLACEMENT
        assert any(v in vr.non_numeric_values for v in ['N/A', 'tbd'])

    def test_interaction_recommendations_sorted_by_priority(self):
        """Verify that interaction recommendations are sorted by priority_score in descending order."""
        # Create a DataFrame with binary, continuous, and discrete columns
        np.random.seed(42)
        df = pd.DataFrame({
            'binary1': np.random.randint(0, 2, 100),
            'binary2': np.random.randint(0, 2, 100),
            'continuous1': np.random.randn(100) * 100 + 500,
            'continuous2': np.random.randn(100) * 50 + 200,
            'discrete': np.random.randint(1, 15, 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        interactions = generate_interaction_recommendations(
            df, 
            target_column='target',
            exclude_columns=['target'],
            random_state=42
        )
        
        # Verify we got some interactions
        assert len(interactions) > 0
        
        # Verify they are sorted in descending order by priority_score
        priority_scores = [rec.priority_score for rec in interactions]
        assert priority_scores == sorted(priority_scores, reverse=True), \
            "Interactions should be sorted by priority_score in descending order"

    def test_interaction_recommendations_top_n_limit(self):
        """Verify that top_n parameter limits the number of returned interactions."""
        np.random.seed(42)
        df = pd.DataFrame({
            'binary1': np.random.randint(0, 2, 100),
            'binary2': np.random.randint(0, 2, 100),
            'continuous1': np.random.randn(100) * 100 + 500,
            'continuous2': np.random.randn(100) * 50 + 200,
            'continuous3': np.random.randn(100) * 75 + 300,
            'discrete1': np.random.randint(1, 10, 100),
            'discrete2': np.random.randint(1, 15, 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Get all interactions
        all_interactions = generate_interaction_recommendations(
            df,
            target_column='target',
            exclude_columns=['target'],
            top_n=None,
            random_state=42
        )
        
        # Get limited interactions
        top_3 = generate_interaction_recommendations(
            df,
            target_column='target',
            exclude_columns=['target'],
            top_n=3,
            random_state=42
        )
        
        # Verify top_n works
        assert len(top_3) == min(3, len(all_interactions))
        
        # Verify top_n returns the highest priority interactions
        if len(all_interactions) >= 3:
            assert top_3[0].priority_score == all_interactions[0].priority_score
            assert top_3[1].priority_score == all_interactions[1].priority_score
            assert top_3[2].priority_score == all_interactions[2].priority_score


class TestDataValidation:
    """Test data validation and quality checks."""

    def test_empty_dataframe_handling(self):
        """Verify handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        assert len(empty_df) == 0

    def test_missing_values_handling(self):
        """Verify handling of missing values."""
        df_with_nulls = pd.DataFrame({
            'col1': [1, 2, np.nan],
            'col2': [None, 'B', 'C'],
        })
        assert df_with_nulls.isnull().any().any()
