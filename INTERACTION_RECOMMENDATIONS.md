# Interaction Recommendations

## Overview

`generate_interaction_recommendations()` suggests `FeatureInteractionRecommendation` objects from numeric columns using three rule families:

1. `STATUS_IMPACT`
2. `RESOURCE_DENSITY`
3. `PRODUCT_UTILIZATION`

The function returns recommendations sorted by `priority_score` descending.

## Function Signature

```python
generate_interaction_recommendations(
    df,
    target_column=None,
    top_n=20,
    exclude_columns=None,
    random_state=42,
)
```

## Selection and Exclusions

The algorithm starts from numeric columns only and excludes likely identifier fields by keyword:

1. `id`
2. `row`
3. `index`
4. `number`
5. `code`

It also excludes any columns passed via `exclude_columns`.

If fewer than two usable numeric columns remain, it returns an empty list.

## Rule Families

### 1. Status-Impact

Pattern:

1. Binary column (`nunique() == 2`)
2. High-variance numeric column (`nunique() > 10`, variance above 60th percentile)
3. Operation: multiplication (`*`)

If `target_column` is provided, binary columns are filtered by mutual information strength (above median MI).

### 2. Resource-Density

Pattern:

1. Two continuous columns (`nunique() > 20`)
2. Absolute correlation > 0.7
3. Denominator non-zero ratio > 0.9
4. Operation: division (`/`)

Priority score is correlation magnitude.

### 3. Product-Utilization

Pattern:

1. Discrete count-like column (`2 < nunique() <= 20`)
2. Continuous duration/resource-like column (`nunique() > 20`)
3. Denominator non-zero ratio >= 0.9
4. Operation: division (`/`)

If `target_column` is provided, interaction score is mutual information between the computed rate feature and the target.

## Usage

```python
from dsr_data_tools import generate_interaction_recommendations

recs = generate_interaction_recommendations(
    df,
    target_column="target",
    top_n=20,
    exclude_columns=["target"],
    random_state=42,
)

for rec in recs:
    rec.info()
```

## Applying Suggested Interactions

Recommendations are editable before apply.

```python
df_out = df.copy()
for rec in recs:
    # Optional override
    rec.derived_name = f"feat_{rec.derived_name}"
    df_out = rec.apply(df_out)
```

## Return Type

Each returned item is a `FeatureInteractionRecommendation` containing:

1. `column_name`
2. `column_name_2`
3. `interaction_type`
4. `operation`
5. `description`
6. `rationale`
7. `priority_score`

## Notes

1. `top_n=None` returns all generated interactions.
2. Division interactions replace denominator zeros with `NaN` at apply time.
3. Interaction recommendation generation is independent from `RecommendationManager.generate_recommendations()`.

