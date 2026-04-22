# RecommendationManager

## Overview

`RecommendationManager` is the orchestration layer for data preparation recommendations in `dsr_data_tools`.

It provides:

1. Pipeline storage and ordering
2. Validation before execution
3. Controlled application to a DataFrame
4. YAML persistence for human-in-the-loop editing

## Core Behaviors

### Stable recommendation IDs

Recommendation IDs are deterministic for the same class and identity-defining fields. This makes IDs stable across runs and useful for YAML-based review/edit workflows.

### Pipeline ordering

Recommendations are executed by `EXECUTION_PRIORITY` (lower number executes first), then deterministically by `column_name` within a priority group.

### Validation before execution

Before `apply()`, `_validate_pipeline()` checks:

1. Referenced source columns exist
2. A recommendation does not depend on columns dropped earlier
3. Output column overwrite rules are respected

### Execution and cleanup

`apply()` can run in-place or on a copy. After transforms, source columns used by these recommendation types are collected for cleanup:

1. `DATETIME_CONVERSION`
2. `FEATURE_EXTRACTION`
3. `ENCODING`

## API Surface

### Build and organize

1. `add(recommendation | iterable)`
2. `add_after(target_id, new_rec)`
3. `clear()`

### Inspect and retrieve

1. `len(manager)`
2. `for rec in manager`
3. `manager[index]`
4. `get_by_id(rec_id)`
5. `get_by_alias(alias)`

### Toggle recommendation state

1. `enable_by_id(rec_id, ok_if_none=False)`
2. `disable_by_id(rec_id, ok_if_none=False)`
3. `toggle_enabled_by_id(rec_id, ok_if_none=False)`

### Generate and execute

1. `generate_recommendations(...)`
2. `apply(df, allow_column_overwrite=False, inplace=False, drop_duplicates=False)`
3. `execution_summary()`

## Apply Parameters

### `allow_column_overwrite`

Default is `False`.

1. If `False`, writing to an existing output column raises `ValueError`
2. If `True`, overwrite is allowed only when pipeline dependency checks pass
3. If dtype changes during overwrite, `TypeError` is raised

### `inplace`

1. `False` (default): apply on a copy
2. `True`: mutate the input DataFrame directly

### `drop_duplicates`

1. `False` (default): preserve row duplicates
2. `True`: drop duplicates before applying the pipeline

## YAML Round-Trip

`RecommendationManager` supports review/edit workflows with:

1. `save_to_yaml(output_dir, filename)`
2. `load_from_yaml(filepath)`

Current YAML behavior:

1. Top-level keys are recommendation IDs
2. Read-only fields are written with ` [RO]` suffix
3. `class_name [RO]` is included to disambiguate classes that share a `rec_type`
4. Enum fields are deserialized from YAML strings back to enum members

Only editable fields are intended for manual changes (for example: `enabled`, `notes`, `alias`, and other class-specific editable fields).

## Minimal Example

```python
import pandas as pd
from dsr_data_tools import RecommendationManager

df = pd.DataFrame({...})

manager = RecommendationManager()
manager.generate_recommendations(df)

# Optional human-in-the-loop cycle
path, _ = manager.save_to_yaml(".", "recommendations")
manager = RecommendationManager.load_from_yaml(path)

result = manager.apply(
    df,
    allow_column_overwrite=False,
    inplace=False,
    drop_duplicates=False,
)
```

## Notes

1. If any recommendation fails during execution, `apply()` raises a manager-level `RuntimeError` with recommendation context.
2. If recommendations are generated and validation detects structural issues, warnings are attached to `execution_summary()` output.
