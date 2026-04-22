# Column Overwrite Behavior

## Overview

`RecommendationManager.apply()` supports controlled output-column overwriting through:

```python
apply(df, allow_column_overwrite=False, inplace=False, drop_duplicates=False)
```

The default behavior is safe: existing columns cannot be overwritten unless explicitly allowed.

## Default Mode: Overwrite Blocked

When `allow_column_overwrite=False`, any recommendation that tries to write to an existing output column fails validation with `ValueError`.

This prevents accidental corruption of existing features.

## Overwrite Mode: Allowed with Guards

When `allow_column_overwrite=True`, overwrite can proceed only if pipeline dependency rules still hold.

Validation checks include:

1. Source dependencies for every enabled recommendation must exist
2. A recommendation cannot depend on data dropped earlier in priority order
3. Overwriting a column is blocked if a later enabled recommendation depends on the original column

The dependency scan includes standard and multi-column inputs such as:

1. `column_name`
2. `start_column`
3. `end_column`
4. `agg_columns`

## Dtype Compatibility Guard

During execution, if overwrite is enabled and a recommendation updates an existing `output_column`, `apply()` compares dtype before and after the recommendation.

If dtype changes, `TypeError` is raised. This is a strict safety rule intended to prevent silent schema drift.

## Related Flags

1. `inplace=True` applies transformations directly to the input DataFrame
2. `drop_duplicates=True` removes duplicate rows before recommendation execution

## Practical Guidance

Use `allow_column_overwrite=False` unless you intentionally designed recommendations to update existing columns and verified downstream dependencies.

Typical workflow:

1. Run with default overwrite protection
2. If an overwrite conflict is intentional, enable overwrite explicitly
3. Re-run and address any dependency or dtype errors

## Example

```python
result = manager.apply(
    df,
    allow_column_overwrite=True,
    inplace=False,
    drop_duplicates=False,
)
```

## Failure Modes to Expect

1. `ValueError` for overwrite conflict when overwrite is disabled
2. `ValueError` for pipeline dependency conflicts
3. `TypeError` for incompatible dtype change during overwrite
4. `RuntimeError` wrapping recommendation-specific failures during execution
