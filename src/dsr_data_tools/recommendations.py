"""Recommendation models and orchestration for dataset preparation."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, cast, TYPE_CHECKING, Union, Iterable
from collections.abc import Mapping
import hashlib
import json
import uuid

if TYPE_CHECKING:
    from dsr_data_tools.recommendations import RecommendationManager

import numpy as np
import pandas as pd

from dsr_utils.enums import DatetimeProperty
from dsr_data_tools.enums import (
    RecommendationType,
    EncodingStrategy,
    MissingValueStrategy,
    OutlierStrategy,
    OutlierHandlingStrategy,
    ImbalanceStrategy,
    InteractionType,
    ColumnHintType,
    RoundingMode,
    BitDepth,
)


def _generate_recommendation_id() -> str:
    """Generate a unique ID for a recommendation."""
    return f"rec_{uuid.uuid4().hex[:8]}"


def _detect_non_numeric_values(non_null_unique, value_counts) -> tuple[list[str], int]:
    """Detect non-numeric placeholder values in a series.

    Identifies string values that cannot be converted to float in a column
    that should be numeric (has some numeric values).

    Args:
        non_null_unique: Array of unique non-null values from the series.
        value_counts: Pre-computed value counts for the series.

    Returns:
        tuple[list[str], int]: A tuple of (list of non-numeric values, count of non-numeric occurrences).
                              Empty list if all values are numeric or null.

    Example:
        >>> s = pd.Series([1.0, 2.0, 'tbd', 'N/A', 3.0])
        >>> unique_vals = s.dropna().unique()
        >>> val_counts = s.value_counts()
        >>> _detect_non_numeric_values(unique_vals, val_counts)
        (['tbd', 'N/A'], 2)
    """
    non_numeric_values = []
    non_numeric_count = 0

    for val in non_null_unique:
        try:
            float(val)
        except (ValueError, TypeError):
            non_numeric_values.append(str(val))
            non_numeric_count += value_counts.get(val, 0)  # O(1) lookup

    return non_numeric_values, non_numeric_count


@dataclass
class Recommendation(ABC):
    """
    Abstract base class for dataset preparation recommendations.

    Each recommendation represents a suggested action to improve data quality
    or prepare the dataset for machine learning.

    Read-only after creation: type, column_name, id.
    Editable: description and any subclass-specific editable fields.

    **Important: Stable IDs and Specification Changes**

    Recommendation IDs are deterministically computed from the recommendation's class
    name and ALL of its attributes. This ensures that identical recommendations always
    get the same ID across different calls to analyze_dataset().

    However, if the Recommendation class specification changes (e.g., new fields are
    added, existing fields are modified), the ID WILL change for existing recommendations.
    This is a breaking change for any hardcoded references to recommendation IDs in
    user code.

    Fields that WILL trigger ID changes if modified:
    - Any editable field (e.g., strategy, properties, output_columns)
    - Any field added/removed from the dataclass
    - The recommendation class definition itself

    Fields that will NOT trigger ID changes:
    - enabled (editable for filtering, excluded from hash)
    - description (for user notes, excluded from hash)
    - alias (optional user-friendly name, excluded from hash)
    - Pylance type checking won't affect IDs

    Mitigation strategies for users:
    1. Use version control to track ID changes when upgrading libraries
    2. Use RecommendationManager.execution_summary() to see current IDs after analyze_dataset()
    3. Store recommendation IDs in comments with the date/version for reference
    4. Use manager.get_by_id() or manager.get_by_alias() to retrieve and verify recommendations exist before
       operating on them (catches ID changes gracefully)
    """

    @property
    @abstractmethod
    def rec_type(self) -> RecommendationType:
        pass

    column_name: str
    description: str
    id: str = field(default_factory=_generate_recommendation_id, init=False)
    enabled: bool = True
    """Whether this recommendation should be applied (editable)"""
    alias: str | None = None
    """Optional user-friendly name for this recommendation (editable, doesn't affect ID)"""
    is_locked: bool = False
    """If True, indicates the recommendation was created from an explicit user hint and should not be auto-modified."""
    _locked: bool = field(default=False, init=False, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False) and name in {"type", "column_name", "id"}:
            raise AttributeError(
                f"{name} is read-only once the recommendation is created"
            )
        super().__setattr__(name, value)

    def _lock_fields(self) -> None:
        """Prevent further mutation of non-editable fields."""
        object.__setattr__(self, "_locked", True)

    def _stable_id_payload(self) -> dict[str, Any]:
        """Build a deterministic payload from recommendation fields for hashing.

        Excludes fields that should not affect the ID:
        - id: The field being computed
        - _locked: Internal state flag
        - enabled: User-editable flag for filtering (doesn't change the recommendation itself)
        - description: User-editable notes (not part of the recommendation's core identity)
        - alias: Optional user-friendly name (doesn't change the recommendation's core identity)

        All other fields are included in the hash, ensuring that any change to
        editable parameters (strategy, properties, output_columns, etc.) will
        produce a different ID.

        Returns:
            dict: Payload with id, _locked, enabled, description, and alias removed, ready for JSON serialization
        """
        data = asdict(self)
        data.pop("id", None)
        data.pop("_locked", None)
        data.pop("enabled", None)
        data.pop("description", None)
        data.pop("alias", None)
        data.pop("is_locked", None)
        return data

    def compute_stable_id(self) -> str:
        """Compute a stable ID based on class name and recommendation attributes.

        The ID is a SHA1 hash of the recommendation's class name and all attributes
        (excluding id, _locked, enabled, and description fields). This ensures that
        identical recommendations always produce the same ID.

        **ID Stability Guarantees:**
        - Same recommendation content → Same ID across runs
        - Different editable values (strategy, properties, etc.) → Different ID
        - Class definition change → Different ID for all instances of that class

        **Important:** If the Recommendation class specification changes (fields added,
        removed, or modified), the ID will change. See the Recommendation class docstring
        for mitigation strategies.

        Returns:
            str: Recommendation ID in format "rec_XXXXXXXX" (SHA1 first 8 hex chars)
        """
        payload = {"class": self.__class__.__name__, "data": self._stable_id_payload()}
        json_str = json.dumps(payload, sort_keys=True, default=str)
        return f"rec_{hashlib.sha1(json_str.encode('utf-8')).hexdigest()[:8]}"

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply this recommendation to a dataset.

        Args:
            df: Input DataFrame

        Returns:
            Modified DataFrame with recommendation applied
        """
        pass

    @abstractmethod
    def info(self) -> None:
        """Display formatted information about this recommendation."""
        pass


@dataclass
class NonInformativeRecommendation(Recommendation):
    """Recommendation to remove non-informative columns."""

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.NON_INFORMATIVE

    reason: str = ""
    """Explanation for why column is non-informative (e.g., 'High cardinality', 'Unique count == row count')"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "NonInformativeRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the non-informative column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with column removed
        """
        return df.drop(columns=[self.column_name])

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: NON_INFORMATIVE")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Reason: {self.reason}")
        print(f"    Action: Drop column '{self.column_name}'")


@dataclass
class MissingValuesRecommendation(Recommendation):
    """Recommendation for handling missing values with editable strategy.

    The strategy and fill_value are editable before applying the recommendation.
    """

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.MISSING_VALUES

    missing_count: int = 0
    """Number of missing values in the column"""

    missing_percentage: float = 0.0
    """Percentage of missing values (0-100)"""

    strategy: MissingValueStrategy = MissingValueStrategy.IMPUTE_MEAN
    """Strategy for handling missing values (EDITABLE before apply)"""

    fill_value: str | int | float | None = None
    """Fill value for FILL_VALUE strategy. Only used when strategy=FILL_VALUE (EDITABLE)"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "MissingValuesRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing value strategy to the column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing value strategy applied
        """
        result = df

        if self.strategy == MissingValueStrategy.DROP_ROWS:
            result = result.dropna(subset=[self.column_name])

        elif self.strategy == MissingValueStrategy.DROP_COLUMN:
            result = result.drop(columns=[self.column_name])

        elif self.strategy == MissingValueStrategy.IMPUTE_MEAN:
            # Impute numeric with mean; for non-numeric, fallback to mode
            if pd.api.types.is_numeric_dtype(result[self.column_name]):
                fill_value = result[self.column_name].mean()
                result[self.column_name] = result[self.column_name].fillna(fill_value)
            else:
                mode_value = result[self.column_name].mode()
                if len(mode_value) > 0:
                    result[self.column_name] = result[self.column_name].fillna(
                        mode_value[0]
                    )
                else:
                    result[self.column_name] = result[self.column_name].fillna(
                        "Unknown"
                    )

        elif self.strategy == MissingValueStrategy.IMPUTE_MEDIAN:
            # Impute numeric with median; for non-numeric, fallback to mode
            if pd.api.types.is_numeric_dtype(result[self.column_name]):
                fill_value = result[self.column_name].median()
                result[self.column_name] = result[self.column_name].fillna(fill_value)
            else:
                mode_value = result[self.column_name].mode()
                if len(mode_value) > 0:
                    result[self.column_name] = result[self.column_name].fillna(
                        mode_value[0]
                    )
                else:
                    result[self.column_name] = result[self.column_name].fillna(
                        "Unknown"
                    )

        elif self.strategy == MissingValueStrategy.IMPUTE_MODE:
            # Impute with mode for categorical/low-cardinality; numeric fallback to median
            mode_value = result[self.column_name].mode()
            if len(mode_value) > 0:
                result[self.column_name] = result[self.column_name].fillna(
                    mode_value[0]
                )
            else:
                # If no mode (all NaN), choose median for numeric, else 'Unknown'
                if pd.api.types.is_numeric_dtype(result[self.column_name]):
                    fill_value = result[self.column_name].median()
                    result[self.column_name] = result[self.column_name].fillna(
                        fill_value
                    )
                else:
                    result[self.column_name] = result[self.column_name].fillna(
                        "Unknown"
                    )

        elif self.strategy == MissingValueStrategy.FILL_VALUE:
            if self.fill_value is not None:
                result[self.column_name] = result[self.column_name].fillna(
                    self.fill_value
                )

        # MissingValueStrategy.LEAVE_AS_NA: Do nothing

        return result

    def info(self) -> None:
        """Display recommendation information including editable parameters."""
        print(f"  Recommendation: {self.rec_type.name}")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(
            f"    Missing count: {self.missing_count} ({self.missing_percentage:.1f}%)"
        )
        print(f"    Strategy: {self.strategy.value} (EDITABLE)")
        if self.strategy == MissingValueStrategy.FILL_VALUE:
            print(f"    Fill value: {self.fill_value} (EDITABLE)")
        print(f"    Action: {self._get_action_description()}")

    def _get_action_description(self) -> str:
        """Get a human-readable description of the action."""
        if self.strategy == MissingValueStrategy.DROP_ROWS:
            return f"Remove {self.missing_count} rows with missing values"
        elif self.strategy == MissingValueStrategy.DROP_COLUMN:
            return f"Drop column '{self.column_name}' entirely"
        elif self.strategy == MissingValueStrategy.IMPUTE_MEAN:
            return f"Impute missing values using mean"
        elif self.strategy == MissingValueStrategy.IMPUTE_MEDIAN:
            return f"Impute missing values using median"
        elif self.strategy == MissingValueStrategy.IMPUTE_MODE:
            return f"Impute missing values using mode"
        elif self.strategy == MissingValueStrategy.FILL_VALUE:
            return f"Fill missing values with: {self.fill_value}"
        elif self.strategy == MissingValueStrategy.LEAVE_AS_NA:
            return f"Leave {self.missing_count} missing values as-is"
        return ""


@dataclass
class EncodingRecommendation(Recommendation):
    """Recommendation for encoding categorical columns with editable strategy.

    The encoder_type is editable before applying the recommendation.
    """

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.ENCODING

    encoder_type: EncodingStrategy = EncodingStrategy.ONEHOT
    """Encoding strategy (EDITABLE before apply)"""

    unique_values: int = 0
    """Number of unique values in the column"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "EncodingRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply encoding strategy to the column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with column encoded or converted to categorical
        """
        result = df

        if self.encoder_type == EncodingStrategy.CATEGORICAL:
            # Convert to categorical dtype (memory optimization)
            result[self.column_name] = result[self.column_name].astype("category")

        elif self.encoder_type == EncodingStrategy.ONEHOT:
            # One-hot encode - creates binary columns for each category
            result = pd.get_dummies(
                result, columns=[self.column_name], drop_first=False
            )

            # Normalize one-hot encoded column names to lowercase snake_case
            # (e.g., "region_East" -> "region_east")
            onehot_cols = [
                col for col in result.columns if col.startswith(self.column_name + "_")
            ]
            rename_map = {col: col.lower() for col in onehot_cols}
            result = result.rename(columns=rename_map)

        elif self.encoder_type == EncodingStrategy.LABEL:
            # Label encode - assigns integer to each category
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            # Handle potential NaN values
            mask = result[self.column_name].notna()
            encoded_values = le.fit_transform(
                result.loc[mask, self.column_name].astype(str)
            )
            # Create new series with encoded values and NaN for masked rows
            new_values = pd.Series(index=result.index, dtype="Int64")  # nullable int
            new_values[mask] = encoded_values
            result[self.column_name] = new_values

        elif self.encoder_type == EncodingStrategy.ORDINAL:
            # Ordinal encode - preserves order
            from sklearn.preprocessing import OrdinalEncoder

            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            result[self.column_name] = oe.fit_transform(result[[self.column_name]])

        return result

    def info(self) -> None:
        """Display recommendation information including editable parameters."""
        print(f"  Recommendation: {self.rec_type.name}")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Unique values: {self.unique_values}")
        print(f"    Encoder type: {self.encoder_type.value} (EDITABLE)")
        print(f"    Description: {self.description}")
        print(f"    Action: {self._get_action_description()}")

    def _get_action_description(self) -> str:
        """Get a human-readable description of the encoding action."""
        if self.encoder_type == EncodingStrategy.CATEGORICAL:
            return f"Convert '{self.column_name}' to categorical dtype (memory optimization)"
        elif self.encoder_type == EncodingStrategy.ONEHOT:
            return f"Apply one-hot encoding to '{self.column_name}' ({self.unique_values} categories)"
        elif self.encoder_type == EncodingStrategy.LABEL:
            return f"Apply label encoding to '{self.column_name}'"
        elif self.encoder_type == EncodingStrategy.ORDINAL:
            return f"Apply ordinal encoding to '{self.column_name}'"
        return ""


@dataclass
class ClassImbalanceRecommendation(Recommendation):
    """Recommendation for handling class imbalance in target variable."""

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.CLASS_IMBALANCE

    majority_percentage: float = 0.0
    """Percentage of majority class"""

    strategy: ImbalanceStrategy = ImbalanceStrategy.SMOTE
    """Recommended strategy for handling imbalance"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "ClassImbalanceRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply class imbalance strategy.

        Note: This is a simplified implementation. SMOTE and upsampling/downsampling
        should be applied during cross-validation to prevent data leakage.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame (or recommendation for model configuration)
        """
        # This recommendation is typically handled at model training time,
        # not during data preparation. Return df unchanged as a placeholder.
        return df

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: CLASS_IMBALANCE")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Majority class: {self.majority_percentage:.2f}%")
        print(f"    Strategy: {self.strategy.value}")
        print(f"    Action: Apply {self.strategy.value} during model training")


@dataclass
class OutlierDetectionRecommendation(Recommendation):
    """Recommendation for handling outliers."""

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.OUTLIER_DETECTION

    strategy: OutlierStrategy = OutlierStrategy.SCALING
    """Recommended strategy for handling outliers"""

    max_value: float = 0.0
    """Maximum value in the column"""

    mean_value: float = 0.0
    """Mean value of the column"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "OutlierDetectionRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier handling strategy to the column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with outlier strategy applied
        """
        result = df

        if self.strategy == OutlierStrategy.SCALING:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            # Handle NaN values
            mask = result[self.column_name].notna()
            scaled_values = scaler.fit_transform(result.loc[mask, [self.column_name]])
            # Convert to float64 to avoid dtype incompatibility
            result[self.column_name] = result[self.column_name].astype("float64")
            result.loc[mask, self.column_name] = scaled_values.flatten()

        elif self.strategy == OutlierStrategy.ROBUST_SCALER:
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
            # Handle NaN values
            mask = result[self.column_name].notna()
            scaled_values = scaler.fit_transform(result.loc[mask, [self.column_name]])
            # Convert to float64 to avoid dtype incompatibility
            result[self.column_name] = result[self.column_name].astype("float64")
            result.loc[mask, self.column_name] = scaled_values.flatten()

        elif self.strategy == OutlierStrategy.REMOVE:
            # Remove rows where value exceeds 1.5 * IQR beyond quartiles
            Q1 = result[self.column_name].quantile(0.25)
            Q3 = result[self.column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            result = result[
                (result[self.column_name] >= lower_bound)
                & (result[self.column_name] <= upper_bound)
            ].reset_index(drop=True)

        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: OUTLIER_DETECTION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Max value: {self.max_value:.2f}, Mean: {self.mean_value:.2f}")
        print(f"    Strategy: {self.strategy.value}")
        print(f"    Action: Apply {self.strategy.value} to handle outliers")


@dataclass
class OutlierHandlingRecommendation(Recommendation):
    """Recommendation for cleaning outliers by nullifying or clipping values beyond bounds.

    This is a data cleaning recommendation distinct from scaling transformations.
    Allows setting explicit bounds and choosing between nullification or clipping strategies.
    """

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.OUTLIER_HANDLING

    strategy: OutlierHandlingStrategy = OutlierHandlingStrategy.CLIP
    """Strategy for handling outliers: NULLIFY (set to NaN) or CLIP (cap at bounds) (EDITABLE)"""

    lower_bound: float = 0.0
    """Lower threshold - values below this are considered outliers (EDITABLE)"""

    upper_bound: float = 0.0
    """Upper threshold - values above this are considered outliers (EDITABLE)"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "OutlierHandlingRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier handling strategy using vectorized operations.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with outliers handled according to strategy
        """
        result = df
        column = result[self.column_name]

        if self.strategy == OutlierHandlingStrategy.NULLIFY:
            # Use boolean mask for vectorized nullification
            mask = (column < self.lower_bound) | (column > self.upper_bound)
            result.loc[mask, self.column_name] = np.nan

        elif self.strategy == OutlierHandlingStrategy.CLIP:
            # Use pandas clip() for efficient capping
            result[self.column_name] = column.clip(
                lower=self.lower_bound, upper=self.upper_bound
            )

        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: OUTLIER_HANDLING")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Lower bound: {self.lower_bound:.2f} (EDITABLE)")
        print(f"    Upper bound: {self.upper_bound:.2f} (EDITABLE)")
        print(f"    Strategy: {self.strategy.value} (EDITABLE)")
        if self.strategy == OutlierHandlingStrategy.NULLIFY:
            print(
                f"    Action: Set values outside [{self.lower_bound:.2f}, {self.upper_bound:.2f}] to NaN"
            )
        else:
            print(
                f"    Action: Clip values to range [{self.lower_bound:.2f}, {self.upper_bound:.2f}]"
            )


@dataclass
class CategoricalConversionRecommendation(Recommendation):
    """Recommendation to convert object column to pandas categorical dtype."""

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.CATEGORICAL_CONVERSION

    unique_values: int = 0
    """Number of unique values in the column"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "CategoricalConversionRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert object column to pandas categorical dtype.

        Categorical dtype reduces memory usage for columns with repetitive string values.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with column converted to categorical type
        """
        result = df
        result[self.column_name] = result[self.column_name].astype("category")
        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: CATEGORICAL_CONVERSION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(f"    Column: '{self.column_name}'")
        print(f"    Unique values: {self.unique_values}")
        if self.alias:
            print(f"    Alias: {self.alias}")
        print(f"    Action: Convert to categorical dtype (memory optimization)")


@dataclass
class BooleanClassificationRecommendation(Recommendation):
    """Recommendation to treat numeric column as boolean."""

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.BOOLEAN_CLASSIFICATION

    values: list[Any] = field(default_factory=list)
    """The two unique values in the column"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "BooleanClassificationRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert numeric column to boolean type.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with column converted to boolean
        """
        result = df
        result[self.column_name] = result[self.column_name].astype(bool)
        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: BOOLEAN_CLASSIFICATION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Values: {self.values}")
        print(f"    Action: Convert to boolean type")


@dataclass
class BinningRecommendation(Recommendation):
    """Recommendation to bin numeric column into categorical ranges."""

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.BINNING

    bins: list[float] = field(default_factory=list)
    """Bin edges for pd.cut()"""

    labels: list[str] = field(default_factory=list)
    """Labels for each bin"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "BinningRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bin the numeric column into categorical ranges.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with column binned and one-hot encoded
        """
        result = df
        try:
            result[self.column_name] = pd.cut(
                result[self.column_name],
                bins=self.bins,
                labels=self.labels,
                right=True,
                include_lowest=True,
            )
        except Exception as e:
            print(f"Warning: Could not bin column '{self.column_name}': {e}")
            return result

        # One-hot encode the binned column
        result = pd.get_dummies(result, columns=[self.column_name], drop_first=False)
        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: BINNING")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Bins: {self.bins}")
        print(f"    Labels: {self.labels}")
        print(f"    Action: Bin into {len(self.labels)} categories and encode")


@dataclass
class IntegerConversionRecommendation(Recommendation):
    """Recommendation to convert float64 to integer types."""

    target_depth: BitDepth = BitDepth.INT32

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.INT_CONVERSION

    integer_count: int = 0
    """Number of integer values in the column"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "IntegerConversionRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check if the column is nullable (contains NaNs)
        has_nans = df[self.column_name].isna().any()

        # Pandas uses capitalized 'Int32'/'Int64' for nullable integers
        dtype_str: Any = self.target_depth.value
        if has_nans:
            dtype_str = dtype_str.capitalize()  # 'int32' -> 'Int32'

        df[self.column_name] = df[self.column_name].astype(dtype_str)
        return df

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: INT_CONVERSION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Integer values: {self.integer_count}")
        print(f"    Action: Convert to {self.target_depth.value}")


@dataclass
class FloatConversionRecommendation(Recommendation):
    """Recommendation to adjust float precision for memory or accuracy."""

    target_depth: BitDepth = BitDepth.FLOAT32

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.FLOAT_CONVERSION

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "FloatConversionRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the precision change to the specified column."""
        # Use .astype() with the string value of the Enum
        try:
            # We use copy=False where possible to save memory,
            # though astype usually creates a copy if the type changes.
            df[self.column_name] = df[self.column_name].astype(self.target_depth.value)
        except Exception as e:
            # Replace with your library's logging/warning system
            print(f"Failed to convert {self.column_name}: {e}")
        return df

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: FLOAT_CONVERSION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Action: Convert to {self.target_depth.value}")


@dataclass
class DecimalPrecisionRecommendation(Recommendation):
    """Recommendation to optimize decimal precision in float columns.

    This recommendation identifies float columns where decimal precision can be
    reduced. The max_decimal_places parameter is editable, allowing the user to
    adjust precision before applying the recommendation. If max_decimal_places is 0
    and all values are integers after rounding, the column can be converted to int64.
    """

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.DECIMAL_PRECISION_OPTIMIZATION

    max_decimal_places: int = 0
    """Maximum number of decimal places to retain (user-editable)"""

    min_value: float = 0.0
    """Minimum value in the column (for reference)"""

    max_value: float = 0.0
    """Maximum value in the column (for reference)"""

    convert_to_int: bool = False
    """Whether to convert to int64 if max_decimal_places is 0 and all values are integers"""

    rounding_mode: RoundingMode = RoundingMode.NEAREST
    """Rounding mode for decimal operations (default: NEAREST)"""

    scale_factor: float | None = None
    """Scale factor to apply before rounding (e.g., 1/1024 to convert MB to GB)"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "DecimalPrecisionRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df
        try:
            series = result[self.column_name]

            # 1. Scale and Round
            factor = 10**self.max_decimal_places
            if self.scale_factor is not None:
                series = series * self.scale_factor

            if self.rounding_mode == RoundingMode.BANKERS:
                series = np.round(series * factor) / factor
            elif self.rounding_mode == RoundingMode.UP:
                series = np.ceil(series * factor) / factor
            elif self.rounding_mode == RoundingMode.DOWN:
                series = np.floor(series * factor) / factor
            else:  # NEAREST
                series = np.floor(series * factor + 0.5) / factor

            # 2. Integer Check
            non_null = series.dropna()
            if len(non_null) > 0 and ((non_null % 1) == 0).all():
                dtype = "Int64" if series.isna().any() else "int64"
                result[self.column_name] = series.astype(dtype)
            else:
                # 3. If it's still a float, downcast to save 50% RAM
                # float32 is safe for up to 6-7 decimal places
                if self.max_decimal_places <= 6:
                    result[self.column_name] = series.astype("float32")
                else:
                    result[self.column_name] = series

        except Exception as e:
            print(
                f"Warning: Could not optimize decimal precision for '{self.column_name}': {e}"
            )
            return result
        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: DECIMAL_PRECISION_OPTIMIZATION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Range: {self.min_value} to {self.max_value}")
        print(f"    Max decimal places: {self.max_decimal_places} (EDITABLE)")
        if self.max_decimal_places == 0:
            print(f"    Convert to int64: {self.convert_to_int}")
        print(f"    Action: Round to {self.max_decimal_places} decimal places")
        if self.max_decimal_places == 0 and self.convert_to_int:
            print(f"           Then convert to int64")


@dataclass
class ValueReplacementRecommendation(Recommendation):
    """Recommendation for replacing non-numeric placeholder values with NaN.

    Detects columns with non-numeric string placeholders (like 'tbd', 'N/A')
    that should be numeric. The values to replace and replacement value are
    editable before applying.
    """

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.VALUE_REPLACEMENT

    non_numeric_values: list[str] = field(default_factory=list)
    """List of non-numeric placeholder values found in column (e.g., ['tbd'])"""

    non_numeric_count: int = 0
    """Total count of non-numeric values"""

    replacement_value: float | str = np.nan
    """Value to replace non-numeric placeholders with (EDITABLE, default: np.nan)"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "ValueReplacementRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace non-numeric placeholder values with specified replacement value.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with non-numeric values replaced
        """
        result = df

        # Replace each non-numeric value with the replacement value
        for val in self.non_numeric_values:
            result[self.column_name] = result[self.column_name].replace(
                val, self.replacement_value
            )

        return result

    def info(self) -> None:
        """Display recommendation information including editable parameters."""
        print(f"  Recommendation: {self.rec_type.name}")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Non-numeric values: {self.non_numeric_values}")
        print(f"    Count: {self.non_numeric_count}")
        print(f"    Replacement value: {self.replacement_value} (EDITABLE)")
        print(f"    Action: {self._get_action_description()}")

    def _get_action_description(self) -> str:
        """Get a human-readable description of the replacement action."""
        values_str = "', '".join(self.non_numeric_values)
        if pd.isna(self.replacement_value):
            return f"Replace '{values_str}' with NaN in '{self.column_name}'"
        else:
            return f"Replace '{values_str}' with '{self.replacement_value}' in '{self.column_name}'"


@dataclass
class FeatureInteractionRecommendation(Recommendation):
    """Recommendation to create a feature interaction between two columns.

    Suggests creating derived features by combining two columns based on
    statistical patterns (e.g., binary × continuous, continuous / continuous).
    """

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.FEATURE_INTERACTION

    column_name_2: str = ""
    """Second column name for the interaction"""

    interaction_type: InteractionType = InteractionType.STATUS_IMPACT
    """Type of interaction (STATUS_IMPACT, RESOURCE_DENSITY, PRODUCT_UTILIZATION)"""

    operation: str = "*"
    """Operation to perform: '*' (multiply), '/' (divide)"""

    rationale: str = ""
    """Explanation for why this interaction is recommended"""

    derived_name: str = ""
    """Name for the derived feature (EDITABLE)"""

    priority_score: float = 0.0
    """Priority score for the interaction (0.0-1.0 scale, higher = more important)"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "FeatureInteractionRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        """Auto-generate derived_name if not set."""
        if not self.derived_name:
            if self.operation == "*":
                self.derived_name = f"{self.column_name}_{self.column_name_2}"
            else:
                self.derived_name = f"{self.column_name}_vs_{self.column_name_2}"
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the interaction feature in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new interaction column added
        """
        if self.column_name not in df.columns or self.column_name_2 not in df.columns:
            raise ValueError(
                f"Column '{self.column_name}' or '{self.column_name_2}' not found in DataFrame"
            )

        if self.operation == "*":
            df[self.derived_name] = df[self.column_name] * df[self.column_name_2]
        elif self.operation == "/":
            # Avoid division by zero
            df[self.derived_name] = df[self.column_name] / df[
                self.column_name_2
            ].replace(0, np.nan)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        return df

    def info(self) -> None:
        """Display recommendation information."""
        operation_name = (
            "multiply"
            if self.operation == "*"
            else "divide" if self.operation == "/" else self.operation
        )
        print(f"  Recommendation: FEATURE_INTERACTION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(
            f"    Input columns: '{self.column_name}' {self.operation} '{self.column_name_2}'"
        )
        print(f"    Output column: '{self.derived_name}' (EDITABLE)")
        print(f"    Type: {self.interaction_type.value}")
        print(f"    Priority Score: {self.priority_score:.2f}")
        print(f"    Rationale: {self.rationale}")


@dataclass
class DatetimeConversionRecommendation(Recommendation):
    """Recommendation to convert object/string column to datetime dtype."""

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.DATETIME_CONVERSION

    detected_format: str | None = None
    """The detected strptime format string, or None if no consistent format found"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "DatetimeConversionRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df
        # Use the pre-detected format for performance; fallback to generic parsing
        series = result[self.column_name]
        if self.detected_format:
            result[self.column_name] = pd.to_datetime(
                series, format=self.detected_format, errors="coerce"
            )
        else:
            result[self.column_name] = pd.to_datetime(series, errors="coerce")
        return result

    def info(self) -> None:
        print(f"  Recommendation: {self.rec_type.name}")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(f"    Column: '{self.column_name}'")
        if self.detected_format:
            print(
                f"    Action: Convert to datetime using format {self.detected_format}"
            )
        else:
            print(f"    Action: Convert to datetime dtype (coerce invalid to NaT)")


@dataclass
class FeatureExtractionRecommendation(Recommendation):
    """Recommendation to extract derived features from a column.

    Currently supports datetime feature extraction (Year, Month, DayOfWeek, Hour, etc.).
    Can be extended for other feature extraction types in the future.
    All attributes are editable by the user before applying the recommendation.

    If only one member of a SIN/COS pair is given a custom output column in
    ``output_columns``, the missing mate is inferred by swapping the prefix
    (``sin`` -> ``cos`` or vice versa). This keeps paired cyclic features
    aligned without requiring duplicate entries.
    """

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.FEATURE_EXTRACTION

    properties: DatetimeProperty = DatetimeProperty(0)
    """Flags indicating which datetime properties to extract"""

    output_prefix: str = ""
    """Optional prefix for generated feature column names (EDITABLE). If empty, uses '{column_name}_'"""

    output_columns: dict[str, str] | None = None
    """Optional mapping of feature names to custom output column names (EDITABLE).
    Only features in this mapping receive custom names; unmapped features use auto-generated names
    from output_prefix. E.g., {'year': 'birth_year', 'month': 'birth_month'} will custom-name only
    'year' and 'month', while 'day' might become 'column_day' if not in mapping. Partial mappings
    are fully supported—users need not specify all features."""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "FeatureExtractionRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df

        # Check if column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(result[self.column_name]):
            # Try to convert if it's not already datetime
            try:
                result[self.column_name] = pd.to_datetime(
                    result[self.column_name], errors="coerce"
                )
            except Exception:
                return result  # Can't extract features if conversion fails

        # Determine the prefix for output columns
        prefix = self.output_prefix if self.output_prefix else f"{self.column_name}_"

        # Vectorized extraction using pandas .dt and numpy (avoid per-row loops)
        from typing import Any

        dt_accessor: Any = result[self.column_name].dt
        feature_series: dict[str, pd.Series] = {}

        if DatetimeProperty.YEAR in self.properties:
            feature_series["year"] = dt_accessor.year
        if DatetimeProperty.MONTH in self.properties:
            feature_series["month"] = dt_accessor.month
        if DatetimeProperty.DAY in self.properties:
            feature_series["day"] = dt_accessor.day
        if DatetimeProperty.DAYOFWEEK in self.properties:
            feature_series["dayofweek"] = dt_accessor.dayofweek
        if DatetimeProperty.DAYOFYEAR in self.properties:
            feature_series["dayofyear"] = dt_accessor.dayofyear
        if DatetimeProperty.QUARTER in self.properties:
            feature_series["quarter"] = dt_accessor.quarter
        if DatetimeProperty.WEEK in self.properties:
            feature_series["week"] = dt_accessor.isocalendar().week
        if DatetimeProperty.IS_MONTH_END in self.properties:
            feature_series["is_month_end"] = dt_accessor.is_month_end
        if DatetimeProperty.IS_MONTH_START in self.properties:
            feature_series["is_month_start"] = dt_accessor.is_month_start
        if DatetimeProperty.HOUR in self.properties:
            feature_series["hour"] = dt_accessor.hour
        if DatetimeProperty.MINUTE in self.properties:
            feature_series["minute"] = dt_accessor.minute
        if DatetimeProperty.SECOND in self.properties:
            feature_series["second"] = dt_accessor.second
        if DatetimeProperty.SIN_HOUR in self.properties:
            radians = dt_accessor.hour * (2 * np.pi / 24)
            feature_series["sin_hour"] = np.sin(radians)
        if DatetimeProperty.COS_HOUR in self.properties:
            radians = dt_accessor.hour * (2 * np.pi / 24)
            feature_series["cos_hour"] = np.cos(radians)

        # Cyclic encoding for Day of Week (0-6, divisor=7)
        if DatetimeProperty.SIN_DAYOFWEEK in self.properties:
            radians = dt_accessor.dayofweek * (2 * np.pi / 7)
            feature_series["sin_dayofweek"] = np.sin(radians)
        if DatetimeProperty.COS_DAYOFWEEK in self.properties:
            radians = dt_accessor.dayofweek * (2 * np.pi / 7)
            feature_series["cos_dayofweek"] = np.cos(radians)

        # Cyclic encoding for Month (1-12, divisor=12 with 1-based adjustment)
        if DatetimeProperty.SIN_MONTH in self.properties:
            radians = (dt_accessor.month - 1) * (2 * np.pi / 12)
            feature_series["sin_month"] = np.sin(radians)
        if DatetimeProperty.COS_MONTH in self.properties:
            radians = (dt_accessor.month - 1) * (2 * np.pi / 12)
            feature_series["cos_month"] = np.cos(radians)

        # Prepare output column mapping with inferred SIN/COS mates when only one side is provided
        effective_output_columns: dict[str, str] = {}
        if self.output_columns:
            effective_output_columns.update(self.output_columns)

            def _infer_pair(missing: str, existing: str) -> None:
                """Infer the missing SIN/COS mate name by swapping prefix."""
                existing_out = effective_output_columns.get(existing)
                if existing_out and missing not in effective_output_columns:
                    if "sin" in existing_out:
                        effective_output_columns[missing] = existing_out.replace(
                            "sin", "cos", 1
                        )
                    elif "cos" in existing_out:
                        effective_output_columns[missing] = existing_out.replace(
                            "cos", "sin", 1
                        )

            # Iterate pairs to infer missing output names
            for feat in list(feature_series.keys()):
                if feat.startswith("sin_"):
                    mate = "cos_" + feat[4:]
                    _infer_pair(mate, feat)
                elif feat.startswith("cos_"):
                    mate = "sin_" + feat[4:]
                    _infer_pair(mate, feat)

        # Assign extracted features to the DataFrame
        for feature_name, series_values in feature_series.items():
            if feature_name in effective_output_columns:
                new_col_name = effective_output_columns[feature_name]
            else:
                new_col_name = f"{prefix}{feature_name}"
            result[new_col_name] = series_values

        return result

    def info(self) -> None:
        print(f"  Recommendation: {self.rec_type.name}")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(f"    Column: '{self.column_name}'")

        # List the properties to extract
        property_names = []
        for prop in DatetimeProperty:
            if prop in self.properties:
                property_names.append(prop.name)

        if property_names:
            print(f"    Features: {', '.join(property_names)}")

        # Show the output prefix
        prefix = self.output_prefix if self.output_prefix else f"{self.column_name}_"
        print(f"    Output prefix: '{prefix}'")

        # Show output columns mapping if present
        if self.output_columns:
            print(f"    Output columns: {self.output_columns}")


@dataclass
class DatetimeDurationRecommendation(Recommendation):
    """Recommendation to calculate duration between two datetime columns.

    Creates a new column representing the time duration between two datetime columns,
    with configurable time units (seconds, minutes, hours, days).
    All attributes are editable by the user before applying the recommendation.
    """

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.FEATURE_EXTRACTION

    start_column: str = ""
    """Name of the column representing the start time"""

    end_column: str = ""
    """Name of the column representing the end time"""

    unit: str = "minutes"
    """Time unit for the duration: 'seconds', 'minutes', 'hours', or 'days'"""

    output_column: str | None = None
    """Name of the output column. If None, auto-generated from component columns"""

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "DatetimeDurationRecommendation | None":
        """Get a recommendation by ID from the manager, ensuring it's of this type."""
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        if not self.output_column:
            self.output_column = f"{self.start_column}_{self.end_column}_{self.unit}"
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the duration between two datetime columns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new duration column added
        """
        result = df
        delta: pd.Series = cast(
            pd.Series, result[self.end_column] - result[self.start_column]
        )

        # Convert timedelta to total seconds using numpy conversion
        total_sec = cast(
            pd.Series, pd.Series(delta.values.astype("timedelta64[s]").astype(float))
        )

        # Scale the duration based on user preference
        if self.unit == "seconds":
            result[self.output_column] = total_sec
        elif self.unit == "hours":
            result[self.output_column] = total_sec / 3600
        elif self.unit == "days":
            result[self.output_column] = total_sec / 86400
        else:  # Default to minutes
            result[self.output_column] = total_sec / 60

        return result

    def info(self) -> None:
        """Display recommendation information."""
        print(f"  Recommendation: DATETIME_DURATION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Start column: '{self.start_column}'")
        print(f"    End column: '{self.end_column}'")
        print(f"    Output column: '{self.output_column}'")
        print(f"    Unit: {self.unit}")


@dataclass
class ColumnHint:
    """User-provided hint to guide recommendation generation for a column.

    Factory methods:
    - datetime(): logical_type=ColumnHintType.DATETIME, optional datetime_format, datetime_features, output_names mapping
    - financial(): logical_type=ColumnHintType.FINANCIAL, optional floor/ceiling numeric bounds
    - categorical(): logical_type=ColumnHintType.CATEGORICAL
    - numeric(): logical_type=ColumnHintType.NUMERIC, optional floor/ceiling numeric bounds
            plus numeric-specific controls: `convert_to_int` and `decimal_places`
    - aggregate(): logical_type=ColumnHintType.AGGREGATE, agg_columns list and agg_op string, optional output_names
    - ignore(): marks a column to be left as-is, silencing warnings
    - drop(): generates a recommendation to remove this column entirely

    Attributes:
    - logical_type: ColumnHintType | None
    - floor: Optional[float]
    - ceiling: Optional[float]
    - datetime_format: Optional[str]
    - datetime_features: Optional[list[DatetimeProperty]]
    - output_names: Optional[dict[str, str]]
    - agg_columns: Optional[list[str]]
    - agg_op: Optional[str]  # e.g., 'sum', 'mean', 'min', 'max'
    - convert_to_int: Optional[bool] (for numeric hints; force int conversion)
    - decimal_places: Optional[int] (for numeric/financial/aggregate hints; float precision, can be negative)
    - is_ignored: bool (internal flag for ignore() hint)
    - should_drop: bool (internal flag for drop() hint)
    """

    logical_type: ColumnHintType | None = None
    floor: float | None = None
    ceiling: float | None = None
    datetime_format: str | None = None
    datetime_features: list[DatetimeProperty] | None = None
    output_names: dict[str, str] | None = None
    agg_columns: list[str] | None = None
    agg_op: str | None = None
    convert_to_int: bool | None = None
    decimal_places: int | None = None
    rounding_mode: RoundingMode | None = None
    """Rounding mode for decimal operations (NEAREST, BANKERS, UP, DOWN)"""
    scale_factor: float | None = None
    """Scale factor to apply before rounding (e.g., 1/1024 to convert MB to GB)"""
    lat_bounds: tuple[float, float] | None = None
    """Latitude bounds (min, max) for geospatial columns"""
    lon_bounds: tuple[float, float] | None = None
    """Longitude bounds (min, max) for geospatial columns"""
    unit: str | None = None
    """Unit of measurement for distance columns (e.g., 'miles', 'kilometers')"""
    is_ignored: bool = False
    should_drop: bool = False

    @classmethod
    def datetime(
        cls,
        datetime_format: str | None = None,
        datetime_features: list[DatetimeProperty] | None = None,
        output_names: dict[str, str] | None = None,
    ) -> "ColumnHint":
        return cls(
            logical_type=ColumnHintType.DATETIME,
            datetime_format=datetime_format,
            datetime_features=datetime_features,
            output_names=output_names,
        )

    @classmethod
    def financial(
        cls,
        floor: float | None = None,
        ceiling: float | None = None,
        decimal_places: int = 2,
        rounding_mode: RoundingMode = RoundingMode.NEAREST,
        scale_factor: float | None = None,
    ) -> "ColumnHint":
        return cls(
            logical_type=ColumnHintType.FINANCIAL,
            floor=floor,
            ceiling=ceiling,
            decimal_places=decimal_places,
            rounding_mode=rounding_mode,
            scale_factor=scale_factor,
        )

    @classmethod
    def categorical(cls, output_names: dict[str, str] | None = None) -> "ColumnHint":
        return cls(logical_type=ColumnHintType.CATEGORICAL, output_names=output_names)

    @classmethod
    def numeric(
        cls,
        floor: float | None = None,
        ceiling: float | None = None,
        *,
        convert_to_int: bool | None = None,
        decimal_places: int | None = None,
        rounding_mode: RoundingMode = RoundingMode.NEAREST,
        scale_factor: float | None = None,
    ) -> "ColumnHint":
        return cls(
            logical_type=ColumnHintType.NUMERIC,
            floor=floor,
            ceiling=ceiling,
            convert_to_int=convert_to_int,
            decimal_places=decimal_places,
            rounding_mode=rounding_mode,
            scale_factor=scale_factor,
        )

    @classmethod
    def aggregate(
        cls,
        agg_columns: list[str],
        agg_op: str,
        output_names: dict[str, str] | None = None,
        decimal_places: int | None = None,
        rounding_mode: RoundingMode = RoundingMode.NEAREST,
        scale_factor: float | None = None,
    ) -> "ColumnHint":
        # Remove duplicates while preserving order
        seen = set()
        unique_columns = []
        for col in agg_columns:
            if col not in seen:
                seen.add(col)
                unique_columns.append(col)

        # Warn if duplicates were found
        if len(unique_columns) != len(agg_columns):
            duplicates = [col for col in agg_columns if agg_columns.count(col) > 1]
            import warnings

            warnings.warn(
                f"Duplicate columns found in aggregate hint: {set(duplicates)}. "
                f"Duplicates have been removed.",
                UserWarning,
            )

        return cls(
            logical_type=ColumnHintType.AGGREGATE,
            agg_columns=unique_columns,
            agg_op=agg_op,
            output_names=output_names,
            decimal_places=decimal_places,
            rounding_mode=rounding_mode,
            scale_factor=scale_factor,
        )

    @classmethod
    def ignore(cls) -> "ColumnHint":
        """Marks a column to be left as-is, silencing warnings."""
        hint = cls()
        hint.is_ignored = True
        return hint

    @classmethod
    def drop(cls) -> "ColumnHint":
        """Generates a recommendation to remove this column entirely."""
        hint = cls()
        hint.should_drop = True
        return hint

    @classmethod
    def geospatial(
        cls,
        latitude_bounds: tuple[float, float] | None = None,
        longitude_bounds: tuple[float, float] | None = None,
    ) -> "ColumnHint":
        """Hint for latitude/longitude columns.

        Args:
            latitude_bounds: Optional tuple of (min_lat, max_lat) for constraint validation.
            longitude_bounds: Optional tuple of (min_lon, max_lon) for constraint validation.

        Returns:
            ColumnHint configured for geospatial data.
        """
        hint = cls(logical_type=ColumnHintType.GEOSPATIAL)
        hint.lat_bounds = latitude_bounds
        hint.lon_bounds = longitude_bounds
        return hint

    @classmethod
    def distance(
        cls,
        unit: str = "miles",
        floor: float | None = None,
        ceiling: float | None = None,
    ) -> "ColumnHint":
        """Hint for distance-based columns.

        Args:
            unit: Unit of measurement (e.g., 'miles', 'kilometers'). Default 'miles'.
            floor: Optional minimum distance bound.
            ceiling: Optional maximum distance bound.

        Returns:
            ColumnHint configured for distance data.
        """
        hint = cls(logical_type=ColumnHintType.DISTANCE)
        hint.unit = unit
        hint.floor = floor
        hint.ceiling = ceiling
        return hint


@dataclass
class AggregationRecommendation(Recommendation):
    """Aggregate multiple columns into a new feature or validate an existing total.

    Supported operations: sum, mean, min, max.
    If the output column already exists, validation is performed and mismatches are counted.
    """

    @property
    def rec_type(self) -> RecommendationType:
        return RecommendationType.FEATURE_AGGREGATION

    agg_columns: list[str] = field(default_factory=list)
    agg_op: str = "sum"
    output_column: str = ""
    validation_mismatch_count: int = 0
    decimal_places: int | None = None
    """Optional decimal places for rounding after aggregation (can be negative)"""
    rounding_mode: RoundingMode = RoundingMode.NEAREST
    """Rounding mode for decimal operations (default: NEAREST)"""
    scale_factor: float | None = None
    """Scale factor to apply before rounding (e.g., 1/1024 to convert MB to GB)"""

    def __post_init__(self):
        self.id = self.compute_stable_id()
        self._lock_fields()

    def _aggregate(self, df: pd.DataFrame) -> pd.Series:
        if self.agg_op == "sum":
            return df[self.agg_columns].sum(axis=1)
        elif self.agg_op == "mean":
            return df[self.agg_columns].mean(axis=1)
        elif self.agg_op == "min":
            return df[self.agg_columns].min(axis=1)
        elif self.agg_op == "max":
            return df[self.agg_columns].max(axis=1)
        else:
            raise ValueError(f"Unsupported agg_op: {self.agg_op}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df
        computed = self._aggregate(result)

        # Apply scale factor if specified
        if self.scale_factor is not None:
            computed = computed * self.scale_factor

        # Apply rounding if decimal_places is specified (after aggregation and scaling)
        if self.decimal_places is not None:
            factor = 10**self.decimal_places

            if self.rounding_mode == RoundingMode.BANKERS:
                # NumPy's round is "round to nearest even"
                computed = np.round(computed * factor) / factor
            elif self.rounding_mode == RoundingMode.UP:
                computed = np.ceil(computed * factor) / factor
            elif self.rounding_mode == RoundingMode.DOWN:
                computed = np.floor(computed * factor) / factor
            else:  # NEAREST (Standard half-up)
                # Use a small epsilon to ensure .5 rounds up correctly
                computed = np.floor(computed * factor + 0.5) / factor

        if self.output_column in result.columns:
            # Validate existing total
            diff = result[self.output_column] - computed
            self.validation_mismatch_count = int(diff.fillna(0).ne(0).sum())
            # Do not overwrite existing column by default
        else:
            result[self.output_column] = computed

        return result

    def info(self) -> None:
        print(f"  Recommendation: {self.rec_type.name}")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print(f"    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Agg op: {self.agg_op}")
        print(f"    Source columns: {', '.join(self.agg_columns)}")
        print(f"    Output column: {self.output_column}")
        if self.decimal_places is not None:
            print(f"    Decimal places: {self.decimal_places}")
        if self.validation_mismatch_count:
            print(f"    Validation mismatches: {self.validation_mismatch_count}")


class RecommendationManager:
    """
    Manages a pipeline of recommendations with logical insertion and coordinated application.

    This class serves as a repository for recommendations, enabling:
    - Logical insertion of recommendations after specific targets (by ID)
    - Validation before applying to ensure no column is dropped before it's used
    - Coordinated execution with automated cleanup of original columns
    """

    # Execution priority map for recommendation types
    # Lower numbers execute first; defines the logical order of data preparation
    EXECUTION_PRIORITY: dict[RecommendationType, int] = {
        # Priority 1: Remove non-informative data that shouldn't be processed
        RecommendationType.NON_INFORMATIVE: 1,
        # Priority 2: Clean outliers before imputing missing values
        # This ensures component columns are cleaned before aggregation
        RecommendationType.OUTLIER_HANDLING: 2,
        # Priority 3: Convert numeric types for vectorized operations
        RecommendationType.DATETIME_CONVERSION: 3,
        RecommendationType.INT_CONVERSION: 3,
        # Priority 4: Impute missing values so downstream operations don't break
        RecommendationType.MISSING_VALUES: 4,
        RecommendationType.VALUE_REPLACEMENT: 4,
        # Priority 5: Convert to categorical dtype after numeric conversions
        RecommendationType.CATEGORICAL_CONVERSION: 5,
        # Priority 6: Extract new features from converted types
        # Aggregation runs here to ensure component columns are cleaned first
        RecommendationType.FEATURE_EXTRACTION: 6,
        RecommendationType.FEATURE_INTERACTION: 6,
        RecommendationType.FEATURE_AGGREGATION: 6,
        # Priority 7: Optimize data types and handle edge cases
        RecommendationType.ENCODING: 7,
        RecommendationType.DECIMAL_PRECISION_OPTIMIZATION: 7,
        RecommendationType.BOOLEAN_CLASSIFICATION: 7,
        RecommendationType.BINNING: 7,
        RecommendationType.OUTLIER_DETECTION: 7,
        RecommendationType.CLASS_IMBALANCE: 7,
    }

    def __init__(self, recommendations: list[Recommendation] | None = None):
        """
        Initialize the RecommendationManager.

        Args:
            recommendations: Optional list of Recommendation objects to initialize with.
                           If None, starts with empty pipeline.
        """
        self._pipeline: list[Recommendation] = recommendations or []
        # Execution summary warnings collected during generation
        self._summary_warnings: list[str] = []

    def add(
        self, recommendation: Union[Recommendation, Iterable[Recommendation]]
    ) -> None:
        # 1. ALWAYS check for the list/container first.
        # This prevents a list of recommendations from being treated as one object.
        if isinstance(recommendation, (list, tuple)):
            self._pipeline.extend(recommendation)

        # 2. Then check for the single object
        elif isinstance(recommendation, Recommendation):
            self._pipeline.append(recommendation)

        # 3. Last resort: other iterables (but NOT the object itself)
        elif isinstance(recommendation, Iterable) and not isinstance(
            recommendation, (str, bytes)
        ):
            self._pipeline.extend(recommendation)

    def add_after(self, target_id: str, new_rec: Recommendation) -> None:
        """
        Logically insert a recommendation after a target recommendation by ID.

        This enables users to insert recommendations without needing to know
        the exact index in the pipeline.

        Args:
            target_id: The ID of the recommendation after which to insert
            new_rec: The Recommendation object to insert

        Raises:
            ValueError: If target_id is not found in the pipeline
        """
        # Find the index of the target recommendation
        target_index = None
        for i, rec in enumerate(self._pipeline):
            if rec.id == target_id:
                target_index = i
                break

        if target_index is None:
            raise ValueError(
                f"Target recommendation with ID '{target_id}' not found in pipeline"
            )

        # Insert after the target
        self._pipeline.insert(target_index + 1, new_rec)

    def _get_sorted_pipeline(self) -> list[Recommendation]:
        """
        Sort recommendations by execution priority, then by column name.

        Returns:
            List of recommendations sorted by:
            1. Execution priority (lower numbers execute first)
            2. Column name (alphabetically, for consistent ordering within priority level)

        Example:
            All Priority 1 recommendations sorted by column_name, then all Priority 2, etc.
        """
        # Group by priority
        priority_groups: dict[int, list[Recommendation]] = {}
        for rec in self._pipeline:
            priority = self.EXECUTION_PRIORITY.get(rec.rec_type, 999)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(rec)

        # Build sorted list: iterate through priorities in order, sort each group by column_name
        sorted_recs: list[Recommendation] = []
        for priority in sorted(priority_groups.keys()):
            sorted_recs.extend(
                sorted(priority_groups[priority], key=lambda x: x.column_name)
            )

        return sorted_recs

    def apply(
        self,
        df: pd.DataFrame,
        allow_column_overwrite: bool = False,
        inplace: bool = False,
        drop_duplicates: bool = False,
    ) -> pd.DataFrame:
        """
        Apply all recommendations in sequence with validation and cleanup.

        The process:
        1. Validation: Ensures no column is dropped before it's used by later recommendations
        2. Execution: Iteratively applies each recommendation in order
        3. Cleanup: Automatically drops original columns that were converted/extracted

        Args:
            df: Input DataFrame to process
            allow_column_overwrite: If False (default), raises an error if any recommendation's
                output_column already exists in the DataFrame. If True, allows overwriting
                existing columns, but validates that overwritten columns are not needed by
                higher-priority recommendations and that dtypes are compatible.
            inplace: If True, mutate the provided DataFrame directly when possible.
                Default False keeps previous behavior (works on a copy).
            drop_duplicates: If True, drop duplicate rows before applying recommendations.
                Default False leaves the data unchanged.

        Returns:
            DataFrame with all recommendations applied and cleaned up

        Raises:
            ValueError: If validation fails (column dropped before being used, or column
                       overwrite conflicts)
        """
        # Step 1: Validation - Check for column usage before dropping
        self._validate_pipeline(df, allow_column_overwrite=allow_column_overwrite)

        # Step 2: Execution - Apply recommendations iteratively in priority order
        result = df if inplace else df.copy()

        # Pre-execution
        if drop_duplicates:
            result = result.drop_duplicates()

        columns_to_drop = set()

        # Sort recommendations by priority, then by column_name
        sorted_recs = self._get_sorted_pipeline()

        for rec in sorted_recs:
            # Skip disabled recommendations
            if not rec.enabled:
                continue

            try:
                # Capture original dtype if we are about to overwrite
                output_col: str | None = cast(
                    str | None, getattr(rec, "output_column", None)
                )
                original_dtype = (
                    result[output_col].dtype
                    if (output_col and output_col in result.columns)
                    else None
                )

                result = rec.apply(result)

                # Post-Apply Dtype Enforcement: Validate dtype was not changed during overwrite
                if original_dtype is not None and allow_column_overwrite and output_col:
                    if (
                        output_col in result.columns
                        and result[output_col].dtype != original_dtype
                    ):
                        raise TypeError(
                            f"Overwrite error: Recommendation '{rec.id}' changed '{output_col}' "
                            f"from {original_dtype} to {result[output_col].dtype}. "
                            f"Dtypes must remain compatible when overwriting existing columns."
                        )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to apply recommendation '{rec.id}' "
                    f"(type={rec.rec_type.name}, column='{rec.column_name}'): {str(e)}"
                ) from e

            # Track columns that should be cleaned up
            # (columns that were converted or are no longer needed)
            if rec.rec_type in (
                RecommendationType.DATETIME_CONVERSION,
                RecommendationType.FEATURE_EXTRACTION,
            ):
                columns_to_drop.add(rec.column_name)
            elif rec.rec_type == RecommendationType.ENCODING:
                # Original encoded column can be dropped
                if rec.column_name in result.columns:
                    columns_to_drop.add(rec.column_name)

        # Step 3: Cleanup - Drop original columns that were processed
        columns_to_drop = {col for col in columns_to_drop if col in result.columns}
        if columns_to_drop:
            result = result.drop(columns=list(columns_to_drop))

        return result

    def execution_summary(self) -> None:
        """
        Display a summary of recommendations in execution order.

        Groups recommendations by priority level and displays them in the order
        they would be executed, with section headings for each priority level.
        """
        if not self._pipeline:
            print("No recommendations in pipeline.")
            return

        # Get sorted recommendations
        sorted_recs = self._get_sorted_pipeline()

        # Group sorted recommendations by priority for display
        priority_groups: dict[int, list[Recommendation]] = {}
        for rec in sorted_recs:
            priority = self.EXECUTION_PRIORITY.get(rec.rec_type, 999)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(rec)

        # Priority descriptions
        priority_names = {
            1: "Priority 1: Remove Non-Informative Data",
            2: "Priority 2: Clean Outliers",
            3: "Priority 3: Type Conversions",
            4: "Priority 4: Handle Missing Values & Placeholders",
            5: "Priority 5: Convert to Categorical",
            6: "Priority 6: Feature Extraction",
            7: "Priority 7: Optimization & Edge Cases",
        }

        # Display recommendations by priority
        print("\n" + "=" * 70)
        print("RECOMMENDATION EXECUTION SUMMARY")
        print("=" * 70)

        for priority in sorted(priority_groups.keys()):
            print(f"\n{priority_names.get(priority, f'Priority {priority}')}")
            print("-" * 70)

            for rec in priority_groups[priority]:
                rec.info()
                if rec.alias:
                    print(f"    Alias: {rec.alias}")
                print()

        # Show which columns will be cleaned up during apply()
        cleanup_cols = {
            r.column_name
            for r in self._pipeline
            if r.rec_type
            in (
                RecommendationType.DATETIME_CONVERSION,
                RecommendationType.FEATURE_EXTRACTION,
                RecommendationType.ENCODING,
            )
        }

        if cleanup_cols:
            print("\nPost-Execution Cleanup")
            print("-" * 70)
            print(
                f"Action: Automated drop of original source columns: {', '.join(sorted(cleanup_cols))}"
            )

        # Display any recorded warnings
        if self._summary_warnings:
            print("\nWarnings")
            print("-" * 70)
            for w in self._summary_warnings:
                print(f"! {w}")

        print("=" * 70)

    def _validate_pipeline(
        self, df: pd.DataFrame, allow_column_overwrite: bool = False
    ) -> None:
        """
        Validate the recommendation pipeline to ensure no column is dropped
        before it's used by later recommendations, and to handle column overwriting.

        Only enabled recommendations are considered during validation.

        Args:
            df: Original DataFrame for context
            allow_column_overwrite: If False, raises error if output_column exists in df.
                If True, allows overwriting but validates compatibility with pipeline.

        Raises:
            ValueError: If a column is dropped before being used, if a column in
                       a recommendation doesn't exist, if column overwrite is not allowed,
                       or if overwritten columns conflict with higher-priority recommendations.
        """
        # First, check that all columns referenced in enabled recommendations exist in the DataFrame
        available_columns = set(df.columns)
        for rec in self._pipeline:
            # Validate only enabled recommendations
            if not rec.enabled:
                continue

            if rec.column_name not in available_columns:
                raise ValueError(
                    f"Column '{rec.column_name}' referenced in recommendation "
                    f"(ID: {rec.id}, type={rec.rec_type.name}) does not exist in the DataFrame. "
                    f"Available columns: {', '.join(sorted(available_columns))}"
                )

        # Track which columns are read and written during pipeline execution
        # Maps column name to priority level that wrote it
        written_columns: dict[str, int] = {}
        read_columns: set[str] = set()  # Columns needed by any recommendation

        # First pass: identify all read columns and check for overwrite conflicts
        for rec in self._pipeline:
            # Process only enabled recommendations
            if not rec.enabled:
                continue

            rec_priority = self.EXECUTION_PRIORITY.get(rec.rec_type, 999)

            # Track columns this recommendation reads
            read_columns.add(rec.column_name)

            # Check for output_column conflicts (for recommendations that produce new columns)
            output_col: str | None = cast(
                str | None, getattr(rec, "output_column", None)
            )
            if output_col:
                # Rule 1: If allow_column_overwrite=False and column exists, raise error
                if not allow_column_overwrite and output_col in available_columns:
                    raise ValueError(
                        f"Conflict: Recommendation '{rec.id}' attempts to write to column "
                        f"'{output_col}' which already exists in the DataFrame. "
                        f"Set allow_column_overwrite=True to permit overwriting, or rename the output column."
                    )

                # Rule 2: If allow_column_overwrite=True, check for pipeline conflicts
                if allow_column_overwrite and output_col in available_columns:
                    # Check if this column is needed by higher-priority recommendations
                    for other_rec in self._pipeline:
                        if other_rec is rec:
                            continue

                        other_priority = self.EXECUTION_PRIORITY.get(
                            other_rec.rec_type, 999
                        )

                        # Higher priority = lower priority number
                        if other_priority > rec_priority:
                            # Check if other_rec reads from this output_column
                            columns_used_by_other = {other_rec.column_name}

                            # Add start_column and end_column if they exist (for DatetimeDurationRecommendation)
                            start_col: str | None = cast(
                                str | None, getattr(other_rec, "start_column", None)
                            )
                            end_col: str | None = cast(
                                str | None, getattr(other_rec, "end_column", None)
                            )

                            if start_col:
                                columns_used_by_other.add(start_col)
                            if end_col:
                                columns_used_by_other.add(end_col)

                            if output_col in columns_used_by_other:
                                raise ValueError(
                                    f"Cannot overwrite '{output_col}' because it is needed for "
                                    f"recommendation '{other_rec.id}' (type={other_rec.rec_type.name}) "
                                    f"which has higher priority (priority {other_priority} > {rec_priority})."
                                )

                # Rule 3: Check if overwritten column is used as input by any later-executing recommendations
                if output_col in available_columns:
                    # Get sorted pipeline to check execution order
                    sorted_recs = self._get_sorted_pipeline()
                    rec_index = None
                    for i, r in enumerate(sorted_recs):
                        if r is rec:
                            rec_index = i
                            break

                    if rec_index is not None:
                        # Check all recommendations that execute after this one
                        for later_rec in sorted_recs[rec_index + 1 :]:
                            # Skip disabled recommendations
                            if not later_rec.enabled:
                                continue

                            columns_used_by_later = {later_rec.column_name}

                            # Add start_column and end_column if they exist
                            later_start_col: str | None = cast(
                                str | None, getattr(later_rec, "start_column", None)
                            )
                            later_end_col: str | None = cast(
                                str | None, getattr(later_rec, "end_column", None)
                            )

                            if later_start_col:
                                columns_used_by_later.add(later_start_col)
                            if later_end_col:
                                columns_used_by_later.add(later_end_col)

                            if output_col in columns_used_by_later:
                                raise ValueError(
                                    f"Cannot overwrite '{output_col}': Recommendation '{rec.id}' would change "
                                    f"the column that recommendation '{later_rec.id}' (type={later_rec.rec_type.name}) "
                                    f"depends on as input. This breaks the pipeline logic because '{later_rec.id}' "
                                    f"would operate on stale/modified data."
                                )

                    # Rule 4: Check dtype compatibility if column is being overwritten
                    if output_col in available_columns:
                        original_dtype = df[output_col].dtype
                        # This will be validated after apply(), but we should flag it as a consideration
                        # Store the mapping for later dtype compatibility check if needed
                        written_columns[output_col] = rec_priority

        # Track which columns are dropped during pipeline execution
        dropped_columns = set()

        # Second pass: track column lifecycle through the pipeline
        for i, rec in enumerate(self._pipeline):
            # Skip disabled recommendations
            if not rec.enabled:
                continue

            # Check if this recommendation drops a column
            if rec.rec_type == RecommendationType.NON_INFORMATIVE:
                dropped_columns.add(rec.column_name)
            elif rec.rec_type == RecommendationType.MISSING_VALUES:
                # Some strategies drop the column
                if isinstance(
                    rec, MissingValuesRecommendation
                ) and rec.strategy.name in ("DROP_COLUMN",):
                    dropped_columns.add(rec.column_name)

            # Check if any later recommendation uses a dropped column
            for later_rec in self._pipeline[i + 1 :]:
                # Skip disabled recommendations
                if not later_rec.enabled:
                    continue

                if later_rec.column_name in dropped_columns:
                    # Find the recommendation that dropped this column
                    dropping_rec_id = None
                    for r in self._pipeline[: i + 1]:
                        if (
                            r.rec_type == RecommendationType.NON_INFORMATIVE
                            and r.column_name == later_rec.column_name
                        ):
                            dropping_rec_id = r.id
                            break

                    error_msg = (
                        f"Column '{later_rec.column_name}' is dropped by a previous "
                        f"recommendation"
                    )
                    if dropping_rec_id:
                        error_msg += f" (ID: {dropping_rec_id})"
                    error_msg += (
                        f", but is needed by recommendation '{later_rec.id}' "
                        f"(type={later_rec.rec_type.name})"
                    )
                    raise ValueError(error_msg)

    def _semantic_similarity(self, col1: str, col2: str) -> float:
        """
        Calculate semantic similarity between two column names (0 to 1).

        Uses shared prefixes/suffixes and temporal anchor keywords to score similarity.
        Higher score = more likely to be related datetime columns for duration calculation.

        Args:
            col1: First column name
            col2: Second column name

        Returns:
            Similarity score between 0 and 1
        """
        col1_lower = col1.lower()
        col2_lower = col2.lower()

        # Start indicators: patterns suggesting start of an interval
        start_indicators = {
            "start",
            "open",
            "begin",
            "entry",
            "from",
            "in",
            "departure",
            "arrival",
            "created",
            "opened",
        }
        # End indicators: patterns suggesting end of an interval
        end_indicators = {
            "end",
            "close",
            "finish",
            "exit",
            "to",
            "out",
            "departure",
            "arrival",
            "completed",
            "closed",
        }

        # Check for temporal anchor keywords
        col1_has_start = any(indicator in col1_lower for indicator in start_indicators)
        col1_has_end = any(indicator in col1_lower for indicator in end_indicators)
        col2_has_start = any(indicator in col2_lower for indicator in start_indicators)
        col2_has_end = any(indicator in col2_lower for indicator in end_indicators)

        # Perfect pairing: one is start, other is end
        if (col1_has_start and col2_has_end) or (col1_has_end and col2_has_start):
            return 0.9

        # Both have temporal indicators
        if (col1_has_start or col1_has_end) and (col2_has_start or col2_has_end):
            return 0.7

        # Check for shared prefixes (e.g., "event_start" and "event_end")
        parts1 = col1_lower.replace("_", " ").split()
        parts2 = col2_lower.replace("_", " ").split()
        shared_parts = len(set(parts1) & set(parts2))

        if shared_parts > 0:
            max_parts = max(len(parts1), len(parts2))
            prefix_similarity = shared_parts / max_parts if max_parts > 0 else 0
            return min(0.6, prefix_similarity)

        return 0.0

    def _check_positive_delta_ratio(
        self, df: pd.DataFrame, col_a: str, col_b: str, sample_size: int = 500
    ) -> float:
        """
        Calculate ratio of positive deltas when subtracting col_a from col_b.

        Args:
            df: DataFrame to analyze
            col_a: Column to subtract from (potential start)
            col_b: Column to subtract (potential end)
            sample_size: Number of rows to sample

        Returns:
            Ratio of positive deltas (0 to 1). Values > 0.95 indicate strong "arrow of time"
        """
        try:
            sample = df[[col_a, col_b]].dropna().head(sample_size)
            if len(sample) < 2:
                return 0.0

            # Calculate deltas
            deltas = sample[col_b] - sample[col_a]
            positive_count = (deltas > pd.Timedelta(0)).sum()
            total_count = len(deltas)

            return positive_count / total_count if total_count > 0 else 0.0
        except Exception:
            return 0.0

    def _check_reasonable_duration_magnitude(
        self, df: pd.DataFrame, col_a: str, col_b: str, sample_size: int = 500
    ) -> bool:
        """
        Check if duration between two datetime columns is within reasonable magnitude.

        Avoids pairing unrelated dates (e.g., birth_date and transaction_date).
        Considers durations within days/weeks/months reasonable, not years apart.

        Args:
            df: DataFrame to analyze
            col_a: Start column
            col_b: End column
            sample_size: Number of rows to sample

        Returns:
            True if durations are within reasonable range, False otherwise
        """
        try:
            sample = df[[col_a, col_b]].dropna().head(sample_size)
            if len(sample) < 2:
                return False

            deltas = sample[col_b] - sample[col_a]

            # Filter for positive deltas only
            positive_deltas = deltas[deltas > pd.Timedelta(0)]
            if len(positive_deltas) == 0:
                return False

            # Check if most durations are within reasonable range (< 10 years)
            max_duration = pd.Timedelta(days=3650)  # ~10 years
            reasonable_count = (positive_deltas <= max_duration).sum()

            # At least 50% of durations should be reasonable
            return reasonable_count / len(positive_deltas) > 0.5
        except Exception:
            return False

    def _generate_duration_column_name(self, col_a: str, col_b: str) -> str:
        """
        Generate a descriptive name for a duration column.

        Tries to use semantic anchors first, falls back to generic pattern.
        Examples: event_start + event_end -> event_duration

        Args:
            col_a: Start column name
            col_b: End column name

        Returns:
            Suggested duration column name
        """
        col_a_lower = col_a.lower()
        col_b_lower = col_b.lower()

        # Extract common prefix
        parts_a = col_a_lower.replace("_", " ").split()
        parts_b = col_b_lower.replace("_", " ").split()
        shared_parts = list(set(parts_a) & set(parts_b))

        # If there's a shared prefix, use it with "_duration" suffix
        if shared_parts:
            # Use up to 2 shared parts
            common_prefix = "_".join(sorted(shared_parts)[:2])
            return f"{common_prefix}_duration"

        # Check for temporal indicators and build descriptive name
        temporal_keywords = {"start", "open", "begin", "end", "close", "finish"}
        for keyword in temporal_keywords:
            if keyword in col_a_lower or keyword in col_b_lower:
                base = col_a_lower.replace(keyword, "").replace("_", " ").strip()
                if base:
                    return f"{base}_duration"

        # Fallback: combine both column names
        return f"{col_a_lower}_{col_b_lower}_duration"

    def _identify_datetime_columns(
        self, df: pd.DataFrame, existing_datetime_cols: set[str] | None = None
    ) -> set[str]:
        """
        Identify all columns that are or will be datetime type.

        Includes columns with existing datetime dtype and those with
        DATETIME_CONVERSION recommendations in the pipeline.

        Args:
            df: DataFrame to analyze
            existing_datetime_cols: Optional set of columns already known to be datetime.
                If provided, skips checking these columns to avoid redundant type checks.

        Returns:
            Set of column names that are/will be datetime
        """
        datetime_cols = set()

        # 1. Already datetime dtype
        if existing_datetime_cols is not None:
            # Use pre-identified datetime columns to avoid redundant type checks
            datetime_cols.update(existing_datetime_cols)
        else:
            # Check all columns (used when called outside generate_recommendations)
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.add(col)

        # 2. Will be datetime after DATETIME_CONVERSION recommendations
        for rec in self._pipeline:
            if rec.rec_type == RecommendationType.DATETIME_CONVERSION:
                datetime_cols.add(rec.column_name)

        return datetime_cols

    def _decide_int_depth(self, series: pd.Series) -> BitDepth:
        """Helper to pick between 32 and 64 bit based on max value."""
        # max value for int32 is 2,147,483,647
        if series.max() < 2e9 and series.min() > -2e9:
            return BitDepth.INT32
        return BitDepth.INT64

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        hints: dict[str, ColumnHint] | None = None,
        max_decimal_places: int | dict[str, int] | None = None,
        default_max_decimal_places: int | None = None,
        min_binning_unique_values: int | dict[str, int] | None = None,
        default_min_binning_unique_values: int = 10,
        max_binning_unique_values: int | dict[str, int] | None = None,
        default_max_binning_unique_values: int = 1000,
        allow_categorical_encoding: bool = True,
        hints_only: bool = False,
        overwrite: bool = True,
    ) -> None:
        """Generate and add data preparation recommendations to the pipeline.

        Analyzes each column in the DataFrame and adds appropriate recommendations
        based on data characteristics (missing values, cardinality, data type, etc.).

        If ``hints`` are provided, those columns are handled using explicit user guidance.
        Hint-based recommendations are marked with ``is_locked=True`` and have their
        descriptions appended with "[User Hint Applied]". Standard heuristics for the hinted
        concern are bypassed.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.
            target_column (str | None): Name of target column (excluded from certain recommendations).
            hints (dict[str, ColumnHint] | None): Optional column-specific guidance. When provided,
                hint-driven recommendations are added with is_locked=True and heuristic generation
                for the hinted concern is bypassed.
            max_decimal_places (int | dict[str, int] | None): Max decimal precision.
            default_max_decimal_places (int | None): Default max decimal places.
            min_binning_unique_values (int | dict[str, int] | None): Min unique values for binning.
            default_min_binning_unique_values (int): Default minimum binning cardinality.
            max_binning_unique_values (int | dict[str, int] | None): Max unique values for binning.
            default_max_binning_unique_values (int): Default maximum binning cardinality.
            allow_categorical_encoding (bool): If True (default), allow encoding recommendations
                for columns marked for categorical conversion. If False, suppress encoding
                for categorical-bound columns.
            target_column (str | None): Name of the target column (for imbalance detection).
                If provided, class imbalance will be analyzed for this column.
            max_decimal_places (int | dict | None): Maximum decimal places for precision optimization.
                Can be an int (applies to all float columns) or dict mapping column names to their
                specific max decimal places. If provided, float columns will be checked for
                decimal precision optimization.
            default_max_decimal_places (int | None): Default max decimal places to use for columns
                not in the max_decimal_places dict. Only used if max_decimal_places is a dict.
            min_binning_unique_values (int | dict | None): Minimum unique values for binning consideration.
                Can be an int (applies to all numeric columns) or dict mapping column names to their
                specific minimum values. If None, default_min_binning_unique_values is used for all columns.
            default_min_binning_unique_values (int): Default minimum unique values for columns not in
                the min_binning_unique_values dict. Only used if min_binning_unique_values is a dict.
                Default is 10.
            max_binning_unique_values (int | dict | None): Maximum unique values for binning consideration.
                Can be an int (applies to all numeric columns) or dict mapping column names to their
                specific maximum values. If None, default_max_binning_unique_values is used for all columns.
            default_max_binning_unique_values (int): Default maximum unique values for columns not in
                the max_binning_unique_values dict. Only used if max_binning_unique_values is a dict.
                Default is 1000. Adjust higher for large datasets where 1000 unique values might still
                represent continuous data (e.g., millions of rows).
            hints_only (bool): If True, only create recommendations from provided hints and skip
                heuristic generation. Also records a warning for any DataFrame columns with neither
                a hint nor a recommendation; these warnings appear in execution_summary(). Default False.
            overwrite (bool): If True, clear the pipeline before adding recommendations.
                If False, add recommendations to the existing pipeline. Default is True.

        Example:
            >>> manager = RecommendationManager()
            >>> manager.generate_recommendations(df, target_column='target')
            >>> result = manager.apply(df)

            With hints and hints_only mode:
            >>> hints = {
            ...     'date_col': ColumnHint.datetime(datetime_features=[DatetimeProperty.DAYOFWEEK, DatetimeProperty.SIN_DAYOFWEEK]),
            ...     'total': ColumnHint.aggregate(agg_columns=['a', 'b'], agg_op='sum', output_names={'aggregate': 'total_sum'})
            ... }
            >>> manager.generate_recommendations(df, hints=hints, hints_only=True)
            >>> manager.execution_summary()  # Includes warnings for unhandled columns
            >>> result = manager.apply(df)
        """
        # Clear existing pipeline if overwrite is True
        if overwrite:
            self.clear()

        # Import needed for analysis
        from dsr_utils.datetime import (
            is_string_datetime,
            infer_string_datetime_format,
            resolve_date_ambiguity,
        )
        from dsr_utils.enums import DatetimeFormat

        # Track datetime columns to avoid redundant type checks later
        existing_datetime_cols: set[str] = set()
        # Track columns marked for categorical conversion for encoding suppression
        categorical_conversion_cols: set[str] = set()

        for col_name in df.columns:
            series = df[col_name]
            # 0.a. User hints override: create recommendations directly and bypass heuristics
            hint = hints.get(col_name) if hints else None
            if hints_only and hint is None:
                # Skip heuristics when only hints should be used
                continue
            if hint is not None:
                user_note = " [User Hint Applied]"

                # Handle ignore hint: skip all processing
                if hint.is_ignored:
                    # Do nothing; column is left as-is, and warnings will be suppressed in hints_only
                    continue

                # Handle drop hint: create a NonInformativeRecommendation
                if hint.should_drop:
                    rec_drop = NonInformativeRecommendation(
                        column_name=col_name,
                        description=f"Drop column '{col_name}'" + user_note,
                        reason="User hint: drop",
                    )
                    rec_drop.is_locked = True
                    self._pipeline.append(rec_drop)
                    continue

                # Datetime hints: conversion and/or feature extraction
                if hint.logical_type == ColumnHintType.DATETIME:
                    # Datetime conversion if not already datetime
                    if not pd.api.types.is_datetime64_any_dtype(series):
                        rec_dt = DatetimeConversionRecommendation(
                            column_name=col_name,
                            description=(
                                f"Convert '{col_name}' to datetime"
                                + (
                                    f" using format {hint.datetime_format}"
                                    if hint.datetime_format
                                    else ""
                                )
                                + user_note
                            ),
                            detected_format=hint.datetime_format,
                        )
                        rec_dt.is_locked = True
                        self._pipeline.append(rec_dt)

                    # Feature extraction as per hint
                    props = DatetimeProperty(0)
                    if hint.datetime_features:
                        for p in hint.datetime_features:
                            props |= p
                    extraction_rec = FeatureExtractionRecommendation(
                        column_name=col_name,
                        description=f"Extract datetime features from '{col_name}'"
                        + user_note,
                        properties=props,
                        output_columns=hint.output_names,
                    )
                    extraction_rec.is_locked = True
                    self._pipeline.append(extraction_rec)
                    # Skip remaining heuristics for this column
                    continue

                # Categorical hint: prefer categorical conversion
                if hint.logical_type == ColumnHintType.CATEGORICAL:
                    unique_count_hint = int(series.nunique())
                    rec_cat = CategoricalConversionRecommendation(
                        column_name=col_name,
                        description=f"Convert '{col_name}' to categorical dtype"
                        + user_note,
                        unique_values=unique_count_hint,
                    )
                    rec_cat.is_locked = True
                    self._pipeline.append(rec_cat)
                    continue

                # Numeric/Financial hints: apply floor/ceiling via clipping; support int/decimal controls for numeric
                if hint.logical_type in (
                    ColumnHintType.NUMERIC,
                    ColumnHintType.FINANCIAL,
                ):
                    # Bounds clipping if specified
                    if hint.floor is not None or hint.ceiling is not None:
                        lower = (
                            float(hint.floor)
                            if hint.floor is not None
                            else float(series.min())
                        )
                        upper = (
                            float(hint.ceiling)
                            if hint.ceiling is not None
                            else float(series.max())
                        )
                        rec_clip = OutlierHandlingRecommendation(
                            column_name=col_name,
                            description=f"Clip '{col_name}' to bounds [{lower}, {upper}]"
                            + user_note,
                            strategy=OutlierHandlingStrategy.CLIP,
                            lower_bound=lower,
                            upper_bound=upper,
                        )
                        rec_clip.is_locked = True
                        self._pipeline.append(rec_clip)

                    # Numeric-specific controls
                    if hint.logical_type == ColumnHintType.NUMERIC:
                        # Prefer explicit int conversion if requested
                        if hint.convert_to_int is True:
                            # Compute integer_count for info (counts integer-like values)
                            non_null = series.dropna()
                            integer_count = (
                                int(((non_null % 1) == 0).sum())
                                if pd.api.types.is_numeric_dtype(series)
                                else 0
                            )
                            rec_int = IntegerConversionRecommendation(
                                column_name=col_name,
                                description=f"Convert '{col_name}' to int64"
                                + user_note,
                                integer_count=integer_count,
                            )
                            rec_int.is_locked = True
                            self._pipeline.append(rec_int)
                        elif hint.decimal_places is not None:
                            non_null = series.dropna()
                            min_val = (
                                float(non_null.min())
                                if len(non_null) > 0
                                else float("nan")
                            )
                            max_val = (
                                float(non_null.max())
                                if len(non_null) > 0
                                else float("nan")
                            )
                            rec_dec = DecimalPrecisionRecommendation(
                                column_name=col_name,
                                description=f"Optimize decimal precision of '{col_name}' to {hint.decimal_places} places"
                                + user_note,
                                max_decimal_places=int(hint.decimal_places),
                                min_value=min_val,
                                max_value=max_val,
                                convert_to_int=False,
                                rounding_mode=(
                                    hint.rounding_mode
                                    if hint.rounding_mode
                                    else RoundingMode.NEAREST
                                ),
                                scale_factor=hint.scale_factor,
                            )
                            rec_dec.is_locked = True
                            self._pipeline.append(rec_dec)

                    # Financial-specific controls: apply decimal_places if specified
                    if (
                        hint.logical_type == ColumnHintType.FINANCIAL
                        and hint.decimal_places is not None
                    ):
                        non_null = series.dropna()
                        min_val = (
                            float(non_null.min()) if len(non_null) > 0 else float("nan")
                        )
                        max_val = (
                            float(non_null.max()) if len(non_null) > 0 else float("nan")
                        )
                        rec_dec = DecimalPrecisionRecommendation(
                            column_name=col_name,
                            description=f"Optimize decimal precision of '{col_name}' to {hint.decimal_places} places"
                            + user_note,
                            max_decimal_places=int(hint.decimal_places),
                            min_value=min_val,
                            max_value=max_val,
                            convert_to_int=False,
                            rounding_mode=(
                                hint.rounding_mode
                                if hint.rounding_mode
                                else RoundingMode.NEAREST
                            ),
                            scale_factor=hint.scale_factor,
                        )
                        rec_dec.is_locked = True
                        self._pipeline.append(rec_dec)

                    # Skip remaining heuristics for this column after applying numeric/financial hints
                    continue

                # Distance hint: use physical bounds with nullify strategy
                if hint.logical_type == ColumnHintType.DISTANCE:
                    # Set lower bound from hint.floor, default to 0.0
                    lower_bound = float(hint.floor) if hint.floor is not None else 0.0
                    # Set upper bound from hint.ceiling, default to 500 miles (domain-safe)
                    upper_bound = (
                        float(hint.ceiling) if hint.ceiling is not None else 500.0
                    )

                    rec_distance = OutlierHandlingRecommendation(
                        column_name=col_name,
                        description=f"Distance column '{col_name}' ({hint.unit}): nullify values outside [{lower_bound}, {upper_bound}] to catch sensor errors"
                        + user_note,
                        strategy=OutlierHandlingStrategy.NULLIFY,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )
                    rec_distance.is_locked = True
                    self._pipeline.append(rec_distance)
                    # Skip remaining heuristics for this column
                    continue

                # Geospatial hint: create bounds checks for latitude and longitude
                if hint.logical_type == ColumnHintType.GEOSPATIAL:
                    # For latitude column
                    if hint.lat_bounds is not None:
                        lat_lower, lat_upper = hint.lat_bounds
                        rec_lat = OutlierHandlingRecommendation(
                            column_name=col_name,
                            description=f"Latitude column '{col_name}': nullify values outside bounding box [{lat_lower}, {lat_upper}]"
                            + user_note,
                            strategy=OutlierHandlingStrategy.NULLIFY,
                            lower_bound=float(lat_lower),
                            upper_bound=float(lat_upper),
                        )
                        rec_lat.is_locked = True
                        self._pipeline.append(rec_lat)

                    # For longitude column
                    if hint.lon_bounds is not None:
                        lon_lower, lon_upper = hint.lon_bounds
                        rec_lon = OutlierHandlingRecommendation(
                            column_name=col_name,
                            description=f"Longitude column '{col_name}': nullify values outside bounding box [{lon_lower}, {lon_upper}]"
                            + user_note,
                            strategy=OutlierHandlingStrategy.NULLIFY,
                            lower_bound=float(lon_lower),
                            upper_bound=float(lon_upper),
                        )
                        rec_lon.is_locked = True
                        self._pipeline.append(rec_lon)

                    # Skip remaining heuristics for this column
                    continue

                # Aggregation hint: create/validate aggregate feature
                if (
                    hint.logical_type == ColumnHintType.AGGREGATE
                    and hint.agg_columns
                    and hint.agg_op
                ):
                    # Determine output column name
                    out_name = None
                    if hint.output_names:
                        out_name = hint.output_names.get(
                            "aggregate"
                        ) or hint.output_names.get("output")
                    if not out_name:
                        out_name = f"{col_name}_{hint.agg_op}"

                    rec_agg = AggregationRecommendation(
                        column_name=(
                            hint.agg_columns[0] if hint.agg_columns else col_name
                        ),
                        description=f"Aggregate columns {hint.agg_columns} with '{hint.agg_op}' into '{out_name}'"
                        + user_note,
                        agg_columns=hint.agg_columns,
                        agg_op=hint.agg_op,
                        output_column=out_name,
                        decimal_places=hint.decimal_places,
                        rounding_mode=(
                            hint.rounding_mode
                            if hint.rounding_mode
                            else RoundingMode.NEAREST
                        ),
                        scale_factor=hint.scale_factor,
                    )
                    rec_agg.is_locked = True
                    self._pipeline.append(rec_agg)
                    continue

            # Cache commonly used series transformations for performance
            non_null_series = series.dropna()
            non_null_unique = non_null_series.unique()

            # Cache min/max for numeric columns (computed lazily when needed)
            non_null_min = None
            non_null_max = None

            # Cache value_counts (computed lazily when needed)
            value_counts_cache = None

            # 0. Early detection: Check if column is datetime or datetime-like string
            is_datetime = pd.api.types.is_datetime64_any_dtype(series)
            if is_datetime:
                existing_datetime_cols.add(col_name)
            is_datetime_string = False
            detected_format = None

            if not is_datetime and series.dtype == "object":
                try:
                    if is_string_datetime(series):
                        is_datetime_string = True
                        detected_format = infer_string_datetime_format(series)
                except Exception:
                    pass

            unique_count = series.nunique()
            total_rows = len(df)
            is_numeric = pd.api.types.is_numeric_dtype(series)
            suggested_categorical_conversion = False

            # 2. Check for missing values
            missing_count = series.isna().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / total_rows) * 100

                if missing_percentage < 10:
                    strategy = MissingValueStrategy.DROP_ROWS
                elif missing_percentage > 50:
                    strategy = MissingValueStrategy.DROP_COLUMN
                else:
                    # Default to mean; refined later based on categorical qualification or skewness
                    strategy = MissingValueStrategy.IMPUTE_MEAN

                rec = MissingValuesRecommendation(
                    column_name=col_name,
                    description=f"Column '{col_name}' has {missing_count} missing values ({missing_percentage:.1f}%).",
                    missing_count=missing_count,
                    missing_percentage=missing_percentage,
                    strategy=strategy,
                )
                self._pipeline.append(rec)

            # 2.5. Check for existing datetime columns
            if is_datetime:
                has_time_component = False
                if not series.dropna().empty:
                    sample_dt = series.dropna().iloc[0]
                    if hasattr(sample_dt, "hour"):
                        time_check = series.dropna().head(100)
                        has_time_component = any(
                            dt.hour != 0 or dt.minute != 0 or dt.second != 0
                            for dt in time_check
                            if pd.notna(dt)
                        )

                properties = (
                    DatetimeProperty.YEAR
                    | DatetimeProperty.MONTH
                    | DatetimeProperty.DAYOFWEEK
                )
                # Add cyclic features in pairs for better seasonal/cyclical representations
                properties |= (
                    DatetimeProperty.SIN_DAYOFWEEK | DatetimeProperty.COS_DAYOFWEEK
                )
                properties |= DatetimeProperty.SIN_MONTH | DatetimeProperty.COS_MONTH
                if has_time_component:
                    properties |= (
                        DatetimeProperty.HOUR
                        | DatetimeProperty.SIN_HOUR
                        | DatetimeProperty.COS_HOUR
                    )

                extraction_rec = FeatureExtractionRecommendation(
                    column_name=col_name,
                    description=f"Extract datetime features from '{col_name}' (Year, Month, DayOfWeek with cyclic encoding{', Hour with cyclic encoding' if has_time_component else ''}).",
                    properties=properties,
                )
                self._pipeline.append(extraction_rec)

                # Remove conflicting NonInformativeRecommendation if present
                self._remove_conflicting_non_informative(col_name)

            # 2.6. Handle datetime string columns
            if is_datetime_string:
                if detected_format:
                    format_name = None
                    for fmt_enum in DatetimeFormat:
                        if fmt_enum.value == detected_format:
                            format_name = fmt_enum.name
                            break

                    ambiguity_note = ""
                    if format_name and (
                        format_name == "US_DATE" or format_name == "EU_DATE"
                    ):
                        ambiguity_result = resolve_date_ambiguity(series)
                        if ambiguity_result == "AMBIGUOUS":
                            ambiguity_note = " [WARNING: Format is ambiguous; verify with domain knowledge]"
                        elif ambiguity_result in ["US", "EU"]:
                            if ambiguity_result == "US" and format_name == "EU_DATE":
                                format_name = "US_DATE"
                            elif ambiguity_result == "EU" and format_name == "US_DATE":
                                format_name = "EU_DATE"

                    if format_name:
                        desc = f"Column '{col_name}' appears to contain datetimes; convert using DatetimeFormat.{format_name} ({detected_format}).{ambiguity_note}"
                    else:
                        desc = f"Column '{col_name}' appears to contain datetimes; convert using format {detected_format}."
                else:
                    desc = f"Column '{col_name}' appears to contain datetimes; convert to datetime dtype."

                rec = DatetimeConversionRecommendation(
                    column_name=col_name,
                    description=desc,
                    detected_format=detected_format,
                )
                self._pipeline.append(rec)

                properties = (
                    DatetimeProperty.YEAR
                    | DatetimeProperty.MONTH
                    | DatetimeProperty.DAYOFWEEK
                )
                # Add cyclic features in pairs for better seasonal/cyclical representations
                properties |= (
                    DatetimeProperty.SIN_DAYOFWEEK | DatetimeProperty.COS_DAYOFWEEK
                )
                properties |= DatetimeProperty.SIN_MONTH | DatetimeProperty.COS_MONTH
                if detected_format and any(
                    time_indicator in detected_format
                    for time_indicator in ["%H", "%M", "%S"]
                ):
                    properties |= (
                        DatetimeProperty.HOUR
                        | DatetimeProperty.SIN_HOUR
                        | DatetimeProperty.COS_HOUR
                    )

                extraction_rec = FeatureExtractionRecommendation(
                    column_name=col_name,
                    description=f"Extract datetime features from '{col_name}' (Year, Month, DayOfWeek with cyclic encoding{', Hour with cyclic encoding' if DatetimeProperty.HOUR in properties else ''}).",
                    properties=properties,
                )
                self._pipeline.append(extraction_rec)

                # Remove conflicting NonInformativeRecommendation if present
                self._remove_conflicting_non_informative(col_name)
                continue

            if is_datetime or is_datetime_string:
                continue

            # 3. Check for boolean classification
            if is_numeric and unique_count == 2 and col_name != target_column:
                values = sorted(non_null_unique.tolist())
                if values == [0.0, 1.0] or values == [0, 1]:
                    rec = BooleanClassificationRecommendation(
                        column_name=col_name,
                        description=f"Column '{col_name}' should be treated as boolean.",
                        values=values,
                    )
                    self._pipeline.append(rec)

            # 3.5. Check for integer conversion (Float64 -> Int32/64)
            if series.dtype == "float64":
                if len(non_null_series) > 0:
                    integer_mask = non_null_series % 1 == 0
                    if integer_mask.all():
                        # Use a helper to decide between INT32 or INT64
                        target_depth = (
                            BitDepth.INT32
                            if (
                                non_null_series.max() < 2e9
                                and non_null_series.min() > -2e9
                            )
                            else BitDepth.INT64
                        )

                        rec = IntegerConversionRecommendation(
                            column_name=col_name,
                            description=f"Column '{col_name}' is float64 with only integer values; should be {target_depth.value}.",
                            integer_count=int(integer_mask.sum()),
                            target_depth=target_depth,
                        )
                        self._pipeline.append(rec)
                        # If we found an integer conversion, skip the float check for this column
                        continue

            # 3.6. Check for decimal precision optimization
            if max_decimal_places is not None and series.dtype == "float64":
                col_max_decimal_places: int | None = None

                if isinstance(max_decimal_places, dict):
                    col_max_decimal_places = max_decimal_places.get(
                        col_name, default_max_decimal_places
                    )
                else:
                    col_max_decimal_places = max_decimal_places

                if col_max_decimal_places is not None and "int64_conversion" not in [
                    r.rec_type.name for r in self._pipeline if r.column_name == col_name
                ]:
                    if len(non_null_series) > 0:
                        if non_null_min is None:
                            non_null_min = non_null_series.min()
                            non_null_max = non_null_series.max()

                        rounded_series = non_null_series.round(col_max_decimal_places)
                        can_convert_to_int = (
                            col_max_decimal_places == 0
                            and (rounded_series % 1 == 0).all()
                        )

                        min_val = (
                            float(non_null_min)
                            if non_null_min is not None
                            else float("nan")
                        )
                        max_val = (
                            float(non_null_max)
                            if non_null_max is not None
                            else float("nan")
                        )

                        rec = DecimalPrecisionRecommendation(
                            column_name=col_name,
                            description=f"Column '{col_name}' can have decimal precision optimized to {col_max_decimal_places} places.",
                            max_decimal_places=col_max_decimal_places,
                            min_value=min_val,
                            max_value=max_val,
                            convert_to_int=bool(can_convert_to_int),
                        )
                        self._pipeline.append(rec)

            # 3.7.1. New: Check for Float32 downcasting
            if series.dtype == "float64":
                # If it's a true float and not already flagged for integer conversion
                # We recommend Float32 for memory efficiency unless it's an excluded high-precision column
                rec = FloatConversionRecommendation(
                    column_name=col_name,
                    description=f"Column '{col_name}' is float64; reducing to float32 can save 50% memory with negligible precision loss.",
                    target_depth=BitDepth.FLOAT32,
                )
                self._pipeline.append(rec)

            # 3.7.2 Check for non-numeric placeholder values
            if series.dtype == "object":
                if value_counts_cache is None:
                    value_counts_cache = series.value_counts()

                non_numeric_vals, non_numeric_cnt = _detect_non_numeric_values(
                    non_null_unique, value_counts_cache
                )

                if non_numeric_vals and len(non_numeric_vals) > 0:
                    numeric_count = 0
                    for val in non_null_unique:
                        try:
                            float(val)
                            numeric_count += 1
                        except (ValueError, TypeError):
                            pass

                    if numeric_count > 0 and non_numeric_cnt > 0:
                        rec = ValueReplacementRecommendation(
                            column_name=col_name,
                            description=f"Column '{col_name}' has non-numeric placeholder values that should be replaced.",
                            non_numeric_values=non_numeric_vals,
                            non_numeric_count=non_numeric_cnt,
                        )
                        self._pipeline.append(rec)

            # 3.8. Check for categorical conversion (memory optimization)
            # Exclusion keywords: monetary/count metrics that should remain numeric
            excluded_keywords = (
                "fee",
                "surcharge",
                "tax",
                "amount",
                "count",
                "dist",
                "price",
                "total",
                "sum",
            )
            col_name_lower = col_name.lower()
            should_exclude_categorical = any(
                keyword in col_name_lower for keyword in excluded_keywords
            )

            # For object columns: unique_count < len(series) / 2
            if (
                series.dtype == "object"
                and col_name != target_column
                and not should_exclude_categorical
            ):
                if unique_count < len(series) / 2 and unique_count >= 2:
                    rec = CategoricalConversionRecommendation(
                        column_name=col_name,
                        description=f"Column '{col_name}' has {unique_count} unique categorical values; convert to categorical dtype for memory optimization.",
                        unique_values=unique_count,
                    )
                    self._pipeline.append(rec)
                    suggested_categorical_conversion = True
                    categorical_conversion_cols.add(col_name)
            if (
                is_numeric
                and col_name != target_column
                and not should_exclude_categorical
            ):
                cardinality_ratio = unique_count / len(df)
                if (
                    unique_count < 500
                    and cardinality_ratio < 0.05
                    and unique_count >= 2
                ):
                    rec = CategoricalConversionRecommendation(
                        column_name=col_name,
                        description=f"Column '{col_name}' has {unique_count} unique values ({cardinality_ratio*100:.2f}% cardinality); convert to categorical dtype for memory optimization.",
                        unique_values=unique_count,
                    )
                    self._pipeline.append(rec)
                    suggested_categorical_conversion = True
                    categorical_conversion_cols.add(col_name)
            if not is_numeric and col_name != target_column:
                # Skip encoding if categorical conversion suggested and allow_categorical_encoding is False
                if (
                    col_name in categorical_conversion_cols
                    and not allow_categorical_encoding
                ):
                    pass  # Suppress encoding
                elif unique_count == 2:
                    description = f"Column '{col_name}' is binary categorical; recommend LabelEncoder."
                    # Add note if also marked for categorical conversion
                    if col_name in categorical_conversion_cols:
                        description += (
                            " [Note: Also suitable for categorical conversion]."
                        )
                    rec = EncodingRecommendation(
                        column_name=col_name,
                        description=description,
                        encoder_type=EncodingStrategy.LABEL,
                        unique_values=unique_count,
                    )
                    self._pipeline.append(rec)

                elif 3 <= unique_count <= 10:
                    description = f"Column '{col_name}' is multi-class categorical; recommend OneHotEncoder."
                    # Add note if also marked for categorical conversion
                    if col_name in categorical_conversion_cols:
                        description += (
                            " [Note: Also suitable for categorical conversion]."
                        )
                    rec = EncodingRecommendation(
                        column_name=col_name,
                        description=description,
                        encoder_type=EncodingStrategy.ONEHOT,
                        unique_values=unique_count,
                    )
                    self._pipeline.append(rec)

            # 5. Check for outliers (skip if too few unique values)
            if (
                is_numeric
                and unique_count >= 15
                and not suggested_categorical_conversion
            ):
                mean_value = series.mean()
                max_value = series.max()
                min_value = series.min()

                if max_value > mean_value * 2:
                    # Calculate IQR for outlier detection
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1

                    # Define bounds for different outlier intensities
                    # 1.5x IQR: mild outliers (traditional boxplot whiskers)
                    clip_lower = q1 - 1.5 * iqr
                    clip_upper = q3 + 1.5 * iqr

                    # 3x IQR: extreme outliers
                    nullify_lower = q1 - 3 * iqr
                    nullify_upper = q3 + 3 * iqr

                    # Check for extreme outliers (beyond 3x IQR) - suggest NULLIFY
                    if min_value < nullify_lower or max_value > nullify_upper:
                        rec_nullify = OutlierHandlingRecommendation(
                            column_name=col_name,
                            description=f"Column '{col_name}' has extreme outliers (beyond 3x IQR); recommend nullifying.",
                            strategy=OutlierHandlingStrategy.NULLIFY,
                            lower_bound=float(nullify_lower),
                            upper_bound=float(nullify_upper),
                        )
                        self._pipeline.append(rec_nullify)

                    # Check for moderate outliers (between 1.5x and 3x IQR) - suggest CLIP
                    if min_value < clip_lower or max_value > clip_upper:
                        rec_clip = OutlierHandlingRecommendation(
                            column_name=col_name,
                            description=f"Column '{col_name}' has moderate outliers (1.5x-3x IQR); recommend clipping.",
                            strategy=OutlierHandlingStrategy.CLIP,
                            lower_bound=float(clip_lower),
                            upper_bound=float(clip_upper),
                        )
                        self._pipeline.append(rec_clip)

                    # Also suggest outlier detection (for awareness if not cleaned)
                    rec_detection = OutlierDetectionRecommendation(
                        column_name=col_name,
                        description=f"Column '{col_name}' has potential outliers (max={max_value:.2f}, mean={mean_value:.2f}).",
                        strategy=OutlierStrategy.SCALING,
                        max_value=max_value,
                        mean_value=mean_value,
                    )
                    self._pipeline.append(rec_detection)

            # 6. Check for class imbalance
            if col_name == target_column and unique_count <= 2:
                if value_counts_cache is None:
                    value_counts_cache = series.value_counts()

                max_class_percentage = (value_counts_cache.max() / total_rows) * 100

                if max_class_percentage > 70:
                    rec = ClassImbalanceRecommendation(
                        column_name=col_name,
                        description=f"Target variable '{col_name}' shows class imbalance ({max_class_percentage:.1f}% majority class).",
                        majority_percentage=max_class_percentage,
                        strategy=ImbalanceStrategy.CLASS_WEIGHT,
                    )
                    self._pipeline.append(rec)

            # 7. Suggest binning
            col_min_binning: int
            if min_binning_unique_values is None:
                col_min_binning = default_min_binning_unique_values
            elif isinstance(min_binning_unique_values, dict):
                col_min_binning = min_binning_unique_values.get(
                    col_name, default_min_binning_unique_values
                )
            else:
                col_min_binning = min_binning_unique_values

            col_max_binning: int
            if max_binning_unique_values is None:
                col_max_binning = default_max_binning_unique_values
            elif isinstance(max_binning_unique_values, dict):
                col_max_binning = max_binning_unique_values.get(
                    col_name, default_max_binning_unique_values
                )
            else:
                col_max_binning = max_binning_unique_values

            if (
                is_numeric
                and not suggested_categorical_conversion
                and col_min_binning <= unique_count <= col_max_binning
            ):
                non_null_series = series.dropna()

                if len(non_null_series) > 0:
                    if non_null_min is None:
                        non_null_min = non_null_series.min()
                        non_null_max = non_null_series.max()

                    col_min = non_null_min
                    col_max = non_null_max

                    if col_min < col_max:
                        desc = series.describe()
                        bins = [
                            col_min - 0.1 * abs(col_max - col_min),
                            desc["25%"],
                            desc["50%"],
                            desc["75%"],
                            col_max + 0.1 * abs(col_max - col_min),
                        ]
                        labels = ["Very_Low", "Low", "Medium", "High", "Very_High"]

                        rec = BinningRecommendation(
                            column_name=col_name,
                            description=f"Column '{col_name}' ({unique_count} unique values) could be binned into {len(labels)} categories for better feature representation.",
                            bins=bins,
                            labels=labels,
                        )
                        self._pipeline.append(rec)

        # Post-processing: Refine missing value strategies based on categorical conversion and skewness
        # - Use IMPUTE_MODE for columns qualifying for categorical conversion
        # - Use IMPUTE_MEDIAN for skewed numeric columns
        # - Otherwise, keep IMPUTE_MEAN for numeric
        for rec in self._pipeline:
            if rec.rec_type == RecommendationType.MISSING_VALUES and isinstance(
                rec, MissingValuesRecommendation
            ):
                # Skip explicit actions
                if rec.strategy in (
                    MissingValueStrategy.DROP_ROWS,
                    MissingValueStrategy.DROP_COLUMN,
                    MissingValueStrategy.FILL_VALUE,
                    MissingValueStrategy.LEAVE_AS_NA,
                ):
                    continue

                col = rec.column_name
                series = df[col]

                if col in categorical_conversion_cols:
                    rec.strategy = MissingValueStrategy.IMPUTE_MODE
                    rec.description = (
                        rec.description
                        + " Using mode imputation due to categorical conversion."
                    )
                elif pd.api.types.is_numeric_dtype(series):
                    skew = series.skew()
                    skew_val = cast(float, skew)
                    if not np.isnan(skew_val) and abs(skew_val) >= 0.75:
                        rec.strategy = MissingValueStrategy.IMPUTE_MEDIAN
                        rec.description = (
                            rec.description
                            + " Using median imputation due to skewed distribution."
                        )
                    else:
                        rec.strategy = MissingValueStrategy.IMPUTE_MEAN

        # Check for non-informative columns last, only if no other recommendations exist
        # This allows columns to be considered useful before marking them as non-informative
        for col_name in df.columns:
            if self._has_recommendations_for_column(col_name):
                continue

            series = df[col_name]
            unique_count = series.nunique()
            total_rows = len(df)
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_datetime = pd.api.types.is_datetime64_any_dtype(series)

            # Skip datetime columns (already handled above)
            if is_datetime:
                continue

            # Check for string datetime columns
            is_datetime_string = False
            if series.dtype == "object":
                try:
                    if is_string_datetime(series):
                        is_datetime_string = True
                except Exception:
                    pass

            if is_datetime_string:
                continue

            # Mark column as non-informative if it meets criteria
            if unique_count == total_rows:
                rec = NonInformativeRecommendation(
                    column_name=col_name,
                    description=f"Column '{col_name}' has unique value for each row.",
                    reason="Unique count equals row count",
                )
                self._pipeline.append(rec)
            elif not is_numeric and unique_count > total_rows * 0.25:
                rec = NonInformativeRecommendation(
                    column_name=col_name,
                    description=f"Column '{col_name}' has high cardinality ({unique_count} unique values).",
                    reason="High cardinality object type",
                )
                self._pipeline.append(rec)

        # Process hints for columns not present in the DataFrame (e.g., aggregate outputs)
        if hints:
            for hint_col, hint in hints.items():
                if hint_col in df.columns:
                    continue  # already handled in main loop
                if (
                    hint.logical_type == ColumnHintType.AGGREGATE
                    and hint.agg_columns
                    and hint.agg_op
                ):
                    out_name = None
                    if hint.output_names:
                        out_name = hint.output_names.get(
                            "aggregate"
                        ) or hint.output_names.get("output")
                    if not out_name:
                        out_name = f"{hint_col}_{hint.agg_op}"

                    rec_agg = AggregationRecommendation(
                        column_name=(
                            hint.agg_columns[0] if hint.agg_columns else hint_col
                        ),
                        description=f"Aggregate columns {hint.agg_columns} with '{hint.agg_op}' into '{out_name}' [User Hint Applied]",
                        agg_columns=hint.agg_columns,
                        agg_op=hint.agg_op,
                        output_column=out_name,
                        decimal_places=hint.decimal_places,
                        rounding_mode=(
                            hint.rounding_mode
                            if hint.rounding_mode
                            else RoundingMode.NEAREST
                        ),
                        scale_factor=hint.scale_factor,
                    )
                    rec_agg.is_locked = True
                    self._pipeline.append(rec_agg)

        # If only hints should be used, keep only hint-driven recs and record warnings
        if hints_only:
            # Retain only locked recommendations created from hints
            self._pipeline = [
                rec for rec in self._pipeline if getattr(rec, "is_locked", False)
            ]

            hinted_cols = set(hints.keys()) if hints else set()
            pipeline_cols = {rec.column_name for rec in self._pipeline}
            df_cols = set(df.columns)
            # Exclude ignored columns from unhandled warnings
            ignored_cols = (
                {col for col, hint in hints.items() if hint.is_ignored}
                if hints
                else set()
            )
            unhandled = sorted(df_cols - hinted_cols - pipeline_cols - ignored_cols)
            if unhandled:
                self._summary_warnings.append(
                    f"Columns without hint or recommendation: {', '.join(unhandled)}"
                )
            return

        # Stage 1: Identify all datetime columns (existing + post-conversion)
        # Pass existing_datetime_cols to avoid redundant type checks
        datetime_columns = self._identify_datetime_columns(df, existing_datetime_cols)

        if len(datetime_columns) >= 2:
            # Stage 2: Pair datetime columns with statistical and semantic checks
            datetime_cols_list = list(datetime_columns)

            for i, col_a in enumerate(datetime_cols_list):
                for col_b in datetime_cols_list[i + 1 :]:
                    # Skip same column
                    if col_a == col_b:
                        continue

                    # Get positive delta ratio (arrow of time)
                    positive_ratio_ab = self._check_positive_delta_ratio(
                        df, col_a, col_b
                    )
                    positive_ratio_ba = self._check_positive_delta_ratio(
                        df, col_b, col_a
                    )

                    # Determine direction: which has higher positive delta ratio
                    if positive_ratio_ab > positive_ratio_ba:
                        start_col, end_col = col_a, col_b
                        positive_ratio = positive_ratio_ab
                    elif positive_ratio_ba > positive_ratio_ab:
                        start_col, end_col = col_b, col_a
                        positive_ratio = positive_ratio_ba
                    else:
                        # No clear direction or both very low
                        continue

                    # Stage 3: Apply confidence threshold
                    # Statistical: Check for strong arrow of time (95%+ positive deltas)
                    statistical_confidence = positive_ratio >= 0.95

                    # Semantic: Check for temporal anchor keywords
                    semantic_score = self._semantic_similarity(start_col, end_col)
                    semantic_confidence = semantic_score >= 0.6

                    # Magnitude: Check if durations are within reasonable range
                    reasonable_magnitude = self._check_reasonable_duration_magnitude(
                        df, start_col, end_col
                    )

                    # Create recommendation if confidence thresholds are met
                    if (
                        statistical_confidence or semantic_confidence
                    ) and reasonable_magnitude:
                        duration_col_name = self._generate_duration_column_name(
                            start_col, end_col
                        )

                        rec = DatetimeDurationRecommendation(
                            column_name=start_col,  # Use start column as primary reference
                            description=(
                                f"Calculate duration from '{start_col}' to '{end_col}'. "
                                f"Statistical confidence: {positive_ratio:.0%} positive deltas. "
                                f"Semantic match score: {semantic_score:.1f}."
                            ),
                            start_column=start_col,
                            end_column=end_col,
                            unit="seconds",
                            output_column=duration_col_name,
                        )
                        self._pipeline.append(rec)

    def clear(self) -> None:
        """Clear all recommendations from the pipeline."""
        self._pipeline.clear()

    def _remove_conflicting_non_informative(self, column_name: str) -> None:
        """
        Remove any NonInformativeRecommendation for the given column.

        Used to resolve conflicts when a column scheduled for feature extraction
        also has a NonInformativeRecommendation. Feature extraction takes precedence
        since it enables value creation from the column.

        Args:
            column_name: The column name to check for conflicting recommendations
        """
        self._pipeline = [
            rec
            for rec in self._pipeline
            if not (
                rec.rec_type == RecommendationType.NON_INFORMATIVE
                and rec.column_name == column_name
            )
        ]

    def _has_recommendations_for_column(self, column_name: str) -> bool:
        """
        Check if the pipeline has any recommendations for a given column.

        Args:
            column_name: The column name to check

        Returns:
            True if any recommendations exist for the column, False otherwise
        """
        return any(rec.column_name == column_name for rec in self._pipeline)

    def __len__(self) -> int:
        """Return the number of recommendations in the pipeline."""
        return len(self._pipeline)

    def __iter__(self):
        """Iterate over recommendations in the pipeline."""
        return iter(self._pipeline)

    def __getitem__(self, index: int) -> Recommendation:
        """Get a recommendation by index."""
        return self._pipeline[index]

    def get_by_id(self, rec_id: str) -> Recommendation | None:
        """
        Retrieve a recommendation by its ID.

        Args:
            rec_id: The ID of the recommendation to retrieve

        Returns:
            The Recommendation object if found, None otherwise
        """
        for rec in self._pipeline:
            if rec.id == rec_id:
                return rec
        return None

    def get_by_alias(self, alias: str) -> Recommendation | None:
        """
        Retrieve a recommendation by its alias.

        Args:
            alias: The alias of the recommendation to retrieve

        Returns:
            The Recommendation object if found, None otherwise
        """
        if alias is None:
            return None
        for rec in self._pipeline:
            if rec.alias == alias:
                return rec
        return None

    def enable_by_id(self, rec_id: str, ok_if_none: bool = False) -> None:
        """
        Enable a recommendation by its ID.

        Args:
            rec_id: The ID of the recommendation to enable
            ok_if_none: If False, raise when the ID is not found

        Raises:
            ValueError: If the recommendation is not found and ok_if_none is False
        """
        rec = self.get_by_id(rec_id)
        if rec is not None:
            rec.enabled = True
        else:
            if not ok_if_none:
                raise ValueError(f"Rec ID not found: {rec_id}")

    def disable_by_id(self, rec_id: str, ok_if_none: bool = False) -> None:
        """
        Disable a recommendation by its ID.

        Args:
            rec_id: The ID of the recommendation to disable
            ok_if_none: If False, raise when the ID is not found

        Raises:
            ValueError: If the recommendation is not found and ok_if_none is False
        """
        rec = self.get_by_id(rec_id)
        if rec is not None:
            rec.enabled = False
        else:
            if not ok_if_none:
                raise ValueError(f"Rec ID not found: {rec_id}")

    def toggle_enabled_by_id(self, rec_id: str, ok_if_none: bool = False) -> None:
        """
        Toggle enabled state for a recommendation by its ID.

        Args:
            rec_id: The ID of the recommendation to toggle
            ok_if_none: If False, raise when the ID is not found

        Raises:
            ValueError: If the recommendation is not found and ok_if_none is False
        """
        rec = self.get_by_id(rec_id)
        if rec is not None:
            rec.enabled = not rec.enabled
        else:
            if not ok_if_none:
                raise ValueError(f"Rec ID not found: {rec_id}")
