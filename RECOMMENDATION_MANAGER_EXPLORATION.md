# RecommendationManager Exploration Report

## Overview
The `RecommendationManager` is the orchestration layer for data preparation recommendations in `dsr_data_tools`. It manages a pipeline of transformation suggestions, handles their execution order, and supports human-in-the-loop review via YAML persistence.

---

## 1. Recommendation Data Structure & Schema

### Base Recommendation Class (`@dataclass`)

All recommendations inherit from the abstract `Recommendation` base class:

```python
@dataclass
class Recommendation(ABC):
    """Abstract base class for all dataset transformation suggestions."""
    
    # Read-Only Fields (System-Managed)
    column_name: str                                    # Target column [RO]
    description: str                                    # System-generated reasoning [RO]
    id: str = field(
        default_factory=_generate_recommendation_id,
        init=False,
        metadata={"editable": False}
    )                                                   # Deterministic 8-char hex ID [RO]
    is_locked: bool = False                            # True if from User Hint [RO]
    _locked: bool = field(
        default=False, init=False, repr=False,
        metadata={"editable": False}
    )                                                   # Internal identity lock flag [RO]
    
    # Editable Fields (User-Modifiable in YAML)
    notes: str = field(default="", metadata={"editable": True})
                                                        # User commentary/justification
    enabled: bool = field(default=True, metadata={"editable": True})
                                                        # Controls execution during apply()
    alias: str | None = field(
        default=None, metadata={"editable": True}
    )                                                   # User-defined display label
```

### Concrete Recommendation Examples

**NonInformativeRecommendation** (Drop columns):
```python
@dataclass
class NonInformativeRecommendation(Recommendation):
    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.NON_INFORMATIVE
    
    # Editable Field
    reason: str = field(default="", metadata={"editable": True})
                                    # e.g., "Constant value", "Unique IDs"
```

**MissingValuesRecommendation** (Imputation/removal):
```python
@dataclass
class MissingValuesRecommendation(Recommendation):
    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.MISSING_VALUES
    
    # Read-Only Fields
    missing_count: int = field(default=0, metadata={"editable": False})
    missing_percentage: float = field(default=0.0, metadata={"editable": False})
    
    # Editable Fields
    strategy: MissingValueStrategy = field(
        default=MissingValueStrategy.IMPUTE_MEAN,
        metadata={"editable": True}
    )
    fill_value: str | int | float | None = field(
        default=None, metadata={"editable": True}
    )
```

**EncodingRecommendation** (Categorical encoding):
```python
@dataclass
class EncodingRecommendation(Recommendation):
    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.ENCODING
    
    # Editable Field
    encoder_type: EncodingStrategy = field(
        default=EncodingStrategy.ONEHOT,
        metadata={"editable": True}
    )
    
    # Read-Only Field
    unique_values: int = field(default=0, metadata={"editable": False})
```

---

## 2. Read-Only vs Editable Fields

### Read-Only Fields (Cannot be modified after creation)
These fields are locked after `__post_init__()` and marked with `[RO]` suffix in YAML:

- **`column_name`** — Identity-defining, immutable
- **`description`** — System-generated reasoning, audit trail
- **`id`** — Deterministic 8-char hex hash based on class + core attributes
- **`is_locked`** — Indicates if generated from a User Hint
- **`_locked`** — Internal flag protecting identity after initialization
- **Type-specific diagnostic fields** — e.g., `missing_count`, `missing_percentage`, `unique_values`

**Protection Mechanism:**
```python
def __setattr__(self, name: str, value: Any) -> None:
    """Enforces read-only constraints after initialization."""
    if getattr(self, "_locked", False) and name in {"column_name", "id"}:
        raise AttributeError(
            f"Modification Error: '{name}' is part of the recommendation's "
            f"identity and cannot be changed after creation."
        )
    super().__setattr__(name, value)
```

### Editable Fields (Always modifiable)
These fields are whitelist-marked with `metadata={"editable": True}`:

- **`notes`** — User-provided commentary or justification
- **`enabled`** — Controls if recommendation executes (default: `True`)
- **`alias`** — User-defined label for display purposes
- **Strategy/configuration fields** — e.g., `strategy`, `encoder_type`, `reason`, `fill_value`

---

## 3. RecommendationManager Access Methods

### Retrieve Recommendations

#### By Index (positional)
```python
def __getitem__(self, index: int) -> Recommendation:
    """Retrieves a recommendation by its positional index in the pipeline."""
    return self._pipeline[index]
```

#### By ID (deterministic hash)
```python
def get_by_id(self, rec_id: str) -> Recommendation | None:
    """
    Retrieves a recommendation from the pipeline by its unique ID.
    
    Returns:
        The matching Recommendation object, or None if not found.
    """
    return next((r for r in self._pipeline if r.id == rec_id), None)
```

#### By Alias (user-defined label)
```python
def get_by_alias(self, alias: str) -> Recommendation | None:
    """
    Retrieves a recommendation using its user-defined alias.
    
    Returns:
        The matching Recommendation object, or None if not found.
    """
    if alias is None:
        return None
    return next((r for r in self._pipeline if r.alias == alias), None)
```

#### By Column Name (check existence)
```python
def _has_recommendation_for_column(self, column_name: str) -> bool:
    """Returns True if one or more recommendations exist for the column."""
    return any(rec.column_name == column_name for rec in self._pipeline)
```

### Iterate & Count

```python
def __len__(self) -> int:
    """Returns the total count of recommendations in the pipeline."""
    return len(self._pipeline)

def __iter__(self):
    """Allows for direct iteration over the pipeline (e.g., in a for-loop)."""
    return iter(self._pipeline)
```

### Manage Recommendations

#### Add recommendations
```python
def add(self, recommendation: Recommendation | Iterable[Recommendation]) -> None:
    """Adds one or more recommendations to the end of the pipeline."""
    if isinstance(recommendation, Recommendation):
        self._pipeline.append(recommendation)
    elif isinstance(recommendation, Iterable) and not isinstance(recommendation, (str, bytes)):
        self._pipeline.extend(recommendation)
    else:
        raise TypeError("Expected a Recommendation or an Iterable of Recommendations.")

def add_after(self, target_id: str, new_rec: Recommendation) -> None:
    """Inserts a recommendation immediately following a specific recommendation ID."""
    # Finds target_id, inserts new_rec after it
    # Raises ValueError if target_id not found
```

#### Clear pipeline
```python
def clear(self) -> None:
    """Removes all recommendations, resetting the pipeline to an empty state."""
    self._pipeline.clear()
```

---

## 4. Toggle Recommendation State

These methods modify the **editable `enabled` field**:

```python
def enable_by_id(self, rec_id: str, ok_if_none: bool = False) -> None:
    """
    Activates a recommendation to ensure its execution during apply().
    
    Parameters:
        rec_id: The unique identifier of the recommendation to activate
        ok_if_none: If True, returns silently if ID not found; 
                   if False, raises ValueError
    """
    rec = self.get_by_id(rec_id)
    if rec:
        rec.enabled = True
    elif not ok_if_none:
        raise ValueError(f"Enable failed: Recommendation ID '{rec_id}' not found.")

def disable_by_id(self, rec_id: str, ok_if_none: bool = False) -> None:
    """Deactivates a recommendation, skipping it during apply()."""
    rec = self.get_by_id(rec_id)
    if rec:
        rec.enabled = False
    elif not ok_if_none:
        raise ValueError(f"Disable failed: Recommendation ID '{rec_id}' not found.")

def toggle_enabled_by_id(self, rec_id: str, ok_if_none: bool = False) -> None:
    """Flips the 'enabled' state of a specific recommendation."""
    rec = self.get_by_id(rec_id)
    if rec:
        rec.enabled = not rec.enabled
    elif not ok_if_none:
        raise ValueError(f"Toggle failed: Recommendation ID '{rec_id}' not found.")
```

---

## 5. YAML Persistence: save_to_yaml & load_from_yaml

### save_to_yaml()

```python
def save_to_yaml(
    self, output_dir: PathLike, filename: str
) -> tuple[Path | CloudPath, dict[str, Any]]:
    """
    Serializes the internal pipeline to a YAML file as a dictionary keyed by ID.
    
    Parameters:
        output_dir: Destination directory (supports local paths and cloud URIs)
        filename: Base name for the YAML file
    
    Returns:
        tuple of (output_path, rejected_kwargs_dict)
    
    Process:
        1. Top-level keys are recommendation IDs
        2. Read-only fields are written with '[RO]' suffix
        3. 'class_name [RO]' added to disambiguate classes sharing rec_type
        4. Only editable fields omit the [RO] suffix
        5. Header warns users not to modify [RO] or ID keys
    """
```

**YAML Structure Example:**
```yaml
# CAUTION: Do not modify the top-level keys (IDs) or keys marked [RO].
# Only fields without [RO] (e.g., 'enabled', 'notes') are intended for manual edits.
# Modifying [RO] fields will result in those changes being ignored during 'clean'.

rec_abc12345:
  column_name [RO]: age
  description [RO]: Convert to integer values
  rec_type [RO]: INT_CONVERSION
  integer_count [RO]: 150
  class_name [RO]: IntegerConversionRecommendation
  enabled: true                    # EDITABLE
  notes: "Manual override"         # EDITABLE
  alias: "age_cast"               # EDITABLE

rec_xyz67890:
  column_name [RO]: income
  description [RO]: Handle missing values
  rec_type [RO]: MISSING_VALUES
  missing_count [RO]: 1800
  missing_percentage [RO]: 5.5
  class_name [RO]: MissingValuesRecommendation
  strategy: DROP_ROWS              # EDITABLE (was IMPUTE_MEAN)
  fill_value: null                 # EDITABLE
  enabled: true                    # EDITABLE
  notes: "Too many nulls"          # EDITABLE
  alias: null                      # EDITABLE
```

### load_from_yaml()

```python
@classmethod
def load_from_yaml(cls, filepath: PathLike) -> "RecommendationManager":
    """
    Load recommendations from YAML into a new RecommendationManager.
    
    Parameters:
        filepath: Path to a recommendations.yaml file created by save_to_yaml
    
    Returns:
        RecommendationManager instance populated with deserialized objects
    
    Process:
        1. Parses YAML into dict (IDs as keys)
        2. Strips '[RO]' suffix from read-only field names
        3. Resolves rec_type enum from string name
        4. Uses class_name [RO] to instantiate correct concrete class
        5. Deserializes enum fields (e.g., MissingValueStrategy) from strings
        6. Creates recommendation instances with deserialized values
        7. Locks ID to prevent tampering during load
    
    Raises:
        ValueError: If YAML format invalid or rec_type unrecognized
        KeyError: If required fields missing
    """
```

**Deserialization Logic:**
- Enum fields automatically converted from YAML strings back to enum members
- Read-only fields ignored during loading (preserves audit trail)
- Only editable fields can be modified and reloaded
- IDs remain locked (`_locked=True`) after deserialization

---

## 6. Test Examples: Recommendation Modifications

### Test: YAML Round-Trip (Edit during load)

```python
def test_manager_load_from_yaml_round_trip(tmp_path):
    """Verifies recommendations can be loaded back from YAML for clean step."""
    # CREATE: Initial recommendation
    rec = IntegerConversionRecommendation(
        column_name="age", description="Convert to int", integer_count=10
    )
    
    # MODIFY (EDITABLE FIELDS)
    rec.enabled = False
    rec.notes = "manual override"
    rec.alias = "age_cast"
    
    # SAVE to YAML
    manager = RecommendationManager(recommendations=[rec])
    filepath, _ = manager.save_to_yaml(tmp_path, "recommendations")
    
    # LOAD from YAML
    loaded = RecommendationManager.load_from_yaml(filepath)
    loaded_rec = loaded.get_by_id(rec.id)
    
    # VERIFY: Editable changes persisted
    assert isinstance(loaded_rec, IntegerConversionRecommendation)
    assert loaded_rec.id == rec.id                      # Same deterministic ID
    assert loaded_rec.column_name == "age"              # Identity preserved
    assert loaded_rec.enabled is False                  # EDITABLE: loaded
    assert loaded_rec.notes == "manual override"        # EDITABLE: loaded
    assert loaded_rec.alias == "age_cast"               # EDITABLE: loaded
```

### Test: Enum Field Deserialization

```python
def test_manager_load_from_yaml_parses_enum_fields(tmp_path):
    """Verifies enum-valued editable fields are restored from YAML strings."""
    yaml_text = (
        "rec_custom_001:\n"
        "  column_name [RO]: workclass\n"
        "  description [RO]: Handle missing values\n"
        "  rec_type [RO]: MISSING_VALUES\n"
        "  missing_count [RO]: 1800\n"
        "  missing_percentage [RO]: 5.5\n"
        "  strategy: DROP_ROWS\n"  # EDITABLE ENUM
        "  enabled: true\n"
    )
    filepath = tmp_path / "recommendations.yaml"
    filepath.write_text(yaml_text)
    
    # LOAD: Deserializes "DROP_ROWS" string to MissingValueStrategy enum
    loaded = RecommendationManager.load_from_yaml(filepath)
    rec = loaded.get_by_id("rec_custom_001")
    
    # VERIFY: Enum correctly deserialized
    assert isinstance(rec, MissingValuesRecommendation)
    assert rec.strategy == MissingValueStrategy.DROP_ROWS  # String → Enum
    assert rec.id == "rec_custom_001"
```

### Test: Toggling Recommendations

```python
# Enable a recommendation by ID
manager = RecommendationManager(recommendations=[rec])
manager.disable_by_id(rec.id)
assert rec.enabled is False

# Toggle enables the recommendation
manager.toggle_enabled_by_id(rec.id)
assert rec.enabled is True

# Can use ok_if_none to suppress errors
manager.enable_by_id("nonexistent_id", ok_if_none=True)  # No error
manager.enable_by_id("nonexistent_id", ok_if_none=False)  # Raises ValueError
```

### Test: Stable ID Generation

```python
def test_recommendation_ids_are_stable():
    """IDs are deterministic based on class + core attributes."""
    rec1 = IntegerConversionRecommendation(
        column_name="col_a", description="desc", integer_count=5
    )
    rec2 = IntegerConversionRecommendation(
        column_name="col_a", description="desc", integer_count=5
    )
    
    # Same attributes = same deterministic ID (SHA1 hash)
    assert rec1.id == rec2.id
    
    # ID excludes volatile fields (enabled, alias, notes)
    rec1.enabled = False
    rec1.notes = "different"
    assert rec1.id == rec2.id  # ID unchanged
```

---

## 7. Execution Priority & Pipeline Ordering

The manager executes recommendations by **priority**, then **alphabetically by column_name**:

```python
EXECUTION_PRIORITY: dict[RecommendationType, int] = {
    1: NON_INFORMATIVE,                         # Structural cleanup (drops)
    2: OUTLIER_HANDLING,                        # Value-level cleaning
    3: DATETIME_CONVERSION, INT_CONVERSION,     # Foundational casting
       FLOAT_CONVERSION,
    4: MISSING_VALUES, VALUE_REPLACEMENT,       # Data completion
    5: CATEGORICAL_CONVERSION,                  # Optimization
    6: FEATURE_EXTRACTION, FEATURE_INTERACTION, # Engineering
       FEATURE_AGGREGATION,
    7: ENCODING, BINNING, CLASS_IMBALANCE,     # ML readiness & final refinements
       OUTLIER_DETECTION, BOOLEAN_CLASSIFICATION,
       DECIMAL_PRECISION_OPTIMIZATION,
}

def _get_sorted_pipeline(self) -> list[Recommendation]:
    """Returns sorted pipeline by (EXECUTION_PRIORITY, column_name)."""
    def sort_key(rec: Recommendation) -> tuple[int, str]:
        priority = self.EXECUTION_PRIORITY.get(rec.rec_type, 999)
        return (priority, rec.column_name)
    
    return sorted(self._pipeline, key=sort_key)
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **ID Generation** | Deterministic SHA1 hash based on class + core attributes (stable across runs) |
| **Read-Only Fields** | `column_name`, `description`, `id`, `is_locked`, `_locked`, type-specific diagnostics |
| **Editable Fields** | `notes`, `enabled`, `alias`, strategy fields (e.g., `strategy`, `fill_value`) |
| **Access by Index** | `manager[0]` returns first recommendation |
| **Access by ID** | `manager.get_by_id(rec_id)` returns recommendation or None |
| **Access by Alias** | `manager.get_by_alias(alias)` returns recommendation or None |
| **YAML Persistence** | `[RO]` suffix marks read-only fields; only editable fields can be changed in YAML |
| **Enum Deserialization** | YAML strings automatically converted to enum members on load |
| **State Toggle** | `enable_by_id()`, `disable_by_id()`, `toggle_enabled_by_id()` modify the `enabled` field |
| **Execution Order** | Sorted by EXECUTION_PRIORITY (1-7), then alphabetically by column_name |
| **Lock Mechanism** | Identity fields locked after `__post_init__()` to prevent modification |
