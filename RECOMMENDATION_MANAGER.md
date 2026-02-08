# RecommendationManager Class

## Overview

The `RecommendationManager` class provides a sophisticated pipeline management system for dataset preparation recommendations. It handles logical insertion, validation, and coordinated application of recommendations with automatic cleanup.

## Key Features

### 1. **Unique Identification**
- Each recommendation receives an auto-generated unique ID (e.g., `rec_8aa52948`)
- IDs enable logical insertion without index knowledge
- IDs persist across pipeline operations

### 2. **Logical Insertion (`add_after`)**
- Insert recommendations after a target by ID rather than by index
- Eliminates index management burden
- Raises `ValueError` if target ID not found

```python
manager.add_after(target_id='rec_8aa52948', new_rec=my_recommendation)
```

### 3. **Validation (`_validate_pipeline`)**
- Checks that no column is dropped before being used by later recommendations
- Prevents invalid pipelines before execution
- Provides clear error messages about conflicts

### 4. **Coordinated Application (`apply`)**
The `apply()` method executes recommendations in three phases:

1. **Validation**: Ensures pipeline integrity
2. **Execution**: Iteratively applies each recommendation
3. **Cleanup**: Automatically drops processed columns

```python
result = manager.apply(df)
```

### 5. **Query Operations**
- `len(manager)`: Get pipeline length
- `manager[i]`: Access by index
- `manager.get_by_id(rec_id)`: Retrieve by ID
- `for rec in manager`: Iterate over recommendations

## Usage Example

```python
from dsr_data_tools import RecommendationManager, NonInformativeRecommendation
from dsr_data_tools.enums import RecommendationType
import pandas as pd

# Create recommendations
rec1 = NonInformativeRecommendation(
    type=RecommendationType.NON_INFORMATIVE,
    column_name='col1',
    description='Drop low-variance column',
    reason='Unique count == row count'
)

rec2 = NonInformativeRecommendation(
    type=RecommendationType.NON_INFORMATIVE,
    column_name='col2',
    description='Drop high-cardinality column',
    reason='Cardinality > 0.95 * row count'
)

# Initialize manager
manager = RecommendationManager([rec1, rec2])

# Insert additional recommendation after rec1
rec3 = NonInformativeRecommendation(...)
manager.add_after(rec1.id, rec3)

# Apply all recommendations
df = pd.DataFrame({...})
result = manager.apply(df)
```

## Implementation Details

### Automatic Column Cleanup

The `apply()` method automatically tracks and drops processed columns:

- **DATETIME_CONVERSION**: Original datetime string column is dropped after conversion
- **FEATURE_EXTRACTION**: Original datetime column is dropped after feature extraction
- **ENCODING**: Original categorical column is dropped after encoding

### Error Handling

The class provides comprehensive error handling:

1. **Target Not Found**: `ValueError` when `add_after()` references non-existent ID
2. **Pipeline Validation Failure**: `ValueError` when column is dropped before being used
3. **Application Failure**: `RuntimeError` wrapping any exception during `apply()`

### Performance Considerations

- Linear iteration over pipeline (O(n) where n = number of recommendations)
- Validation is O(n²) in worst case but typically much faster
- Cleanup deduplicates column drops to avoid redundant operations

## Integration with Dataset Analysis Pipeline

Future integration points:

1. `generate_recommendations()` will return `RecommendationManager` instead of list
2. Users can insert custom recommendations into the pipeline
3. Manager coordinates all transformations automatically

## Type Hints

All methods include full type hints for IDE support:

```python
def add_after(self, target_id: str, new_rec: Recommendation) -> None:
    """Insert a recommendation after a target by ID."""
    
def apply(self, df: pd.DataFrame) -> pd.DataFrame:
    """Apply all recommendations with validation and cleanup."""
    
def get_by_id(self, rec_id: str) -> Recommendation | None:
    """Retrieve a recommendation by its ID."""
```

## Future Enhancements

1. **Priority-Based Execution**: Execute high-priority recommendations first
2. **Conditional Recommendations**: Skip recommendations based on column properties
3. **Rollback Support**: Checkpoint before each recommendation for rollback capability
4. **Performance Metrics**: Track execution time and data shape changes per recommendation
5. **Parallel Execution**: Execute independent recommendations concurrently
