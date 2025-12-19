# Automated Feature Interaction Recommendations

## Overview

The `generate_interaction_recommendations()` function automatically identifies meaningful feature combinations based on statistical patterns in your dataset. This helps discover interactions that may improve predictive model performance.

## Three Core Rules

### 1. Status-Impact (Binary × Continuous)
**Purpose:** Identify how continuous features differ based on binary status variables.

**Example:** `Balance × IsActiveMember`
- Shows how account balance varies between active and inactive members
- Captures the interaction between wealth and engagement

**Criteria:**
- Binary column: exactly 2 unique values
- Continuous column: high variance, cardinality > 10

### 2. Resource Density (Continuous / Continuous)
**Purpose:** Create normalized ratios for financial or resource metrics.

**Example:** `Balance / EstimatedSalary`
- Represents wealth relative to income
- More robust than raw balance alone
- Normalizes for income level differences

**Criteria:**
- Both columns match financial keywords (balance, salary, income, credit, revenue)
- Denominator has <10% zero values

### 3. Product Utilization (Count / Duration)
**Purpose:** Measure adoption velocity and intensity rates.

**Example:** `NumOfProducts / Tenure`
- Shows how many products per month/year of tenure
- Distinguishes fast adopters from slow adopters

**Criteria:**
- Numerator: count-like columns (num*, count*, product*) with ≤20 unique values
- Denominator: duration-like columns (tenure, age, year, month, day)

## Usage

### Basic Usage
```python
from dsr_data_tools import generate_interaction_recommendations

# Generate all interactions
interactions = generate_interaction_recommendations(df)

# Display recommendations
for rec in interactions:
    rec.info()
```

### Exclude Target Column
```python
# Prevent target variable from being used in interactions
interactions = generate_interaction_recommendations(
    df=df,
    exclude_columns=['Exited', 'Target']
)
```

### Apply Interactions to Dataset
```python
# Recommendations are editable before applying
df_with_interactions = df.copy()

for rec in interactions:
    # Optionally customize the derived feature name
    rec.derived_name = f"custom_{rec.derived_name}"
    
    # Apply to dataset
    df_with_interactions = rec.apply(df_with_interactions)

# New columns are now available for model training
print(df_with_interactions.columns)
```

### Selective Application
```python
# Apply only interactions of a specific type
from dsr_data_tools import InteractionType

resource_density_only = [
    rec for rec in interactions 
    if rec.interaction_type == InteractionType.RESOURCE_DENSITY
]

for rec in resource_density_only:
    df = rec.apply(df)
```

## Output Format

Each recommendation includes:
- **Type**: Interaction category (STATUS_IMPACT, RESOURCE_DENSITY, PRODUCT_UTILIZATION)
- **Operation**: multiply (*) or divide (/)
- **New Feature**: Generated column name (EDITABLE)
- **Rationale**: Explanation for why this interaction was recommended

## Example Output (BetaBank Churn Dataset)

```
Total interactions identified: 10

STATUS IMPACT (6 interactions)
- Balance × IsActiveMember: How balance differs by member activity
- CreditScore × IsActiveMember: How credit differs by member activity
- EstimatedSalary × IsActiveMember: Income differences by activity status

RESOURCE DENSITY (2 interactions)
- Balance / EstimatedSalary: Wealth-to-income ratio
- CreditScore / EstimatedSalary: Credit-to-income ratio

PRODUCT UTILIZATION (2 interactions)
- NumOfProducts / Tenure: Products per year of tenure
- NumOfProducts / Age: Products as percentage of customer age
```

## Implementation Notes

### Automatic Exclusions
- ID columns (RowNumber, CustomerId, etc.)
- Non-numeric columns
- Specified exclusion list

### Division Safety
- Columns with >10% zeros are skipped for denominator
- Results with NaN are preserved for later handling

### Editable Design
All recommendations are fully editable before applying:
- Change derived feature names
- Customize operations
- Adjust rationale

## Integration with Data Preparation

Interaction recommendations are separate from basic data preparation to allow:
1. First apply standard preparation (missing values, encoding, etc.)
2. Then analyze the prepared dataset for interactions
3. Apply selected interactions for model training

```python
# Workflow
df_prepared = ddt.apply_recommendations(df, recommendations)
interactions = ddt.generate_interaction_recommendations(df_prepared, exclude_columns=['Target'])
df_final = apply_interactions(df_prepared, selected_interactions)
```

