# info() Output Summary

## Purpose

This file summarizes the current `info()` display conventions used by recommendation classes and how they affect pipeline readability.

## Current Display Conventions

Across recommendation classes, `info()` is designed to show:

1. Recommendation type
2. Recommendation ID
3. Enabled state
4. Primary column context
5. Type-specific configuration fields

For user-hinted recommendations, `Source: User Hint` appears where supported.

## Multi-Column Recommendations

Recommendations that depend on multiple columns expose that explicitly in their `info()` output.

Examples:

1. `DatetimeDurationRecommendation`: `start_column`, `end_column`, `output_column`, `unit`
2. `FeatureInteractionRecommendation`: both input columns, operation, derived column, rationale, priority score
3. `AggregationRecommendation`: group and aggregation fields

## Why This Matters

These conventions make `RecommendationManager.execution_summary()` and manual review easier by making column impact obvious before running `apply()`.

## Related Recommendation Families

Current recommendation classes include:

1. Structural cleanup: non-informative
2. Missing/outlier handling
3. Type and numeric conversion
4. Categorical and encoding preparation
5. Datetime conversion and extraction
6. Cross-column engineering (interaction, duration, aggregation)

## Quick Validation Pattern

Use this pattern when auditing recommendation readability:

```python
manager = RecommendationManager()
manager.generate_recommendations(df)

for rec in manager:
    rec.info()

manager.execution_summary()
```

## Notes

1. `info()` is presentation-focused and does not change recommendation behavior.
2. Editable vs read-only behavior is enforced by dataclass field metadata and YAML load/save logic, not by `info()`.
