"""Recommendation models and orchestration for dataset preparation."""

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, cast

from cloudpathlib import CloudPath
from dsr_files.utils import PathLike
from dsr_files.yaml_handler import save_yaml

if TYPE_CHECKING:
    from dsr_data_tools.recommendations import RecommendationManager

import numpy as np
import pandas as pd
from dsr_utils.enums import DatetimeProperty

from dsr_data_tools.enums import (
    BitDepth,
    ColumnHintType,
    EncodingStrategy,
    ImbalanceStrategy,
    InteractionType,
    MissingValueStrategy,
    OutlierHandlingStrategy,
    OutlierStrategy,
    RecommendationType,
    RoundingMode,
)


def _generate_recommendation_id() -> str:
    """
    Generate a unique, short identifier for a recommendation instance.

    Returns
    -------
    str
        A string ID prefixed with 'rec_' followed by 8 random hex characters.
    """
    return f"rec_{uuid.uuid4().hex[:8]}"


def _detect_non_numeric_values(
    non_null_unique: np.ndarray, value_counts: pd.Series
) -> tuple[list[str], int]:
    """
    Identify non-numeric string placeholders in a candidate numeric series.

    Iterates through unique values to find strings that cannot be cast to float.
    Used to find sentinel values (e.g., 'N/A', 'tbd') in otherwise numeric
    columns.

    Parameters
    ----------
    non_null_unique : np.ndarray
        Array of unique non-null values from the series.
    value_counts : pd.Series
        Frequency map of values in the series.

    Returns
    -------
    list of str
        Identified non-numeric placeholder strings.
    int
        The total sum of occurrences (frequency) for these placeholders.
    """
    non_numeric_values: list[str] = []
    non_numeric_count: int = 0

    for val in non_null_unique:
        # Optimization: skip actual numbers immediately before trying string cast
        if isinstance(val, (int, float, np.number)):
            continue

        try:
            # We use float() as the litmus test for 'numeric-ness'
            float(val)
        except (ValueError, TypeError):
            # Check for strings that aren't actually placeholder text (like empty strings)
            val_str = str(val)
            non_numeric_values.append(val_str)

            # Use .get(val, 0) to safely access frequencies
            non_numeric_count += int(value_counts.get(val, 0))

    return non_numeric_values, non_numeric_count


@dataclass
class Recommendation(ABC):
    """
    Abstract base class for all dataset transformation suggestions.

    A Recommendation defines a specific operation to be performed on a DataFrame.
    It supports deterministic ID generation and provides a "Human-in-the-Loop"
    interface for manual auditing and justification.

    Parameters
    ----------
    column_name : str
        The target column for the transformation (Read-Only).
    description : str
        A human-readable summary of the system-generated reasoning (Read-Only).
    notes : str, default ""
        User-provided commentary or justification for manual overrides.
        Whitelisted for user edits in YAML persistence.
    enabled : bool, default True
        Controls if the RecommendationManager executes this during `apply()`.
        Whitelisted for user edits in YAML persistence.
    alias : str, optional
        An optional user-defined label for display purposes.
        Whitelisted for user edits in YAML persistence.
    is_locked : bool, default False
        True if this was generated from a User Hint, protecting it from
        automated deletion during re-analysis (Read-Only).

    Attributes
    ----------
    id : str
        A deterministic 8-character hex ID used to track and persist the
        recommendation across audit sessions (System-Managed).
    _locked : bool
        Internal flag indicating if identity-defining fields are read-only to
        preserve audit integrity (System-Managed).
    """

    @property
    @abstractmethod
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation (e.g., TYPE_CONVERSION).

        Returns
        -------
        RecommendationType
            The specific enum category of this recommendation.
        """
        pass

    column_name: str
    description: str
    notes: str = field(default="", metadata={"editable": True})
    id: str = field(
        default_factory=_generate_recommendation_id,
        init=False,
        metadata={"editable": False},
    )
    enabled: bool = field(default=True, metadata={"editable": True})
    alias: str | None = field(default=None, metadata={"editable": True})
    is_locked: bool = False
    _locked: bool = field(
        default=False, init=False, repr=False, metadata={"editable": False}
    )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Enforces read-only constraints on core identity fields after initialization.

        Parameters
        ----------
        name : str
            The name of the attribute to modify.
        value : Any
            The new value to assign to the attribute.

        Raises
        ------
        AttributeError
            If attempting to modify 'column_name' or 'id' on a locked instance.
        """
        if getattr(self, "_locked", False) and name in {"column_name", "id"}:
            raise AttributeError(
                f"Modification Error: '{name}' is part of the recommendation's "
                f"identity and cannot be changed after creation."
            )
        super().__setattr__(name, value)

    def _lock_fields(self) -> None:
        """Transitions the instance to a read-only state for identity-defining fields."""
        object.__setattr__(self, "_locked", True)

    def _stable_id_payload(self) -> dict[str, Any]:
        """
        Serializes the core attributes that define the uniqueness of this recommendation.

        Excludes volatile state like 'enabled' or UI-only fields like 'alias'.

        Returns
        -------
        dict[str, Any]
            A dictionary of fields used for stable ID generation.
        """
        data = asdict(self)
        ignored_fields = {
            "id",
            "_locked",
            "enabled",
            "description",
            "alias",
            "is_locked",
        }
        for field_name in ignored_fields:
            data.pop(field_name, None)
        return data

    def compute_stable_id(self) -> str:
        """
        Generates a deterministic SHA1 hash based on the class and its attributes.

        Returns
        -------
        str
            A string in the format 'rec_XXXXXXXX'.
        """
        payload = {"class": self.__class__.__name__, "data": self._stable_id_payload()}
        json_str = json.dumps(payload, sort_keys=True, default=str)
        hash_val = hashlib.sha1(json_str.encode("utf-8")).hexdigest()
        return f"rec_{hash_val[:8]}"

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the transformation on the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.
        """
        pass

    @abstractmethod
    def info(self) -> None:
        """Prints a developer/user-friendly summary of the recommendation."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the recommendation to a dictionary for external persistence.

        Captures all dataclass fields, including subclass-specific attributes,
        ensuring they are converted to YAML-safe primitives when processed
        by the library handler.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the recommendation, including the
            string-name of its `rec_type`.
        """
        data = asdict(self)
        data["rec_type"] = self.rec_type.name
        data.pop("_locked", None)
        return data

    def get_editable_attribute_names(self) -> list[str]:
        """
        Returns a list of attribute names tagged as editable in the metadata.
        """
        return [f.name for f in fields(self) if f.metadata.get("editable", False)]


@dataclass
class NonInformativeRecommendation(Recommendation):
    """
    Recommendation to remove columns that provide no predictive or analytical value.

    Commonly suggested for columns with zero variance (constant values),
    100% missing data, or high-cardinality identifiers (like UUIDs) that
    don't contribute to machine learning patterns.

    Parameters
    ----------
    reason : str, default ""
        The specific diagnostic finding (e.g., 'Constant value', 'Unique IDs').
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.NON_INFORMATIVE.
        """
        return RecommendationType.NON_INFORMATIVE

    reason: str = field(default="", metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "NonInformativeRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        NonInformativeRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a NonInformativeRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes the target column from the DataFrame.

        Uses 'errors=ignore' to ensure idempotency if the column was already removed.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame from which the column should be removed.

        Returns
        -------
        pd.DataFrame
            The DataFrame without the target column.
        """
        return df.drop(columns=[self.column_name], errors="ignore")

    def info(self) -> None:
        """Prints the rationale for removal and the target column name."""
        print("  Recommendation: NON_INFORMATIVE")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print("    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Reason: {self.reason}")
        print("    Action: Drop non-informative column")


@dataclass
class MissingValuesRecommendation(Recommendation):
    """
    Recommendation to handle null values using various imputation or removal strategies.

    This is a highly interactive recommendation where the user can choose the best
    fit for their domain—whether that's statistical imputation (mean/median/mode),
    constant filling, or row/column removal.

    Parameters
    ----------
    missing_count : int, default 0
        Absolute number of nulls detected in the column.
    missing_percentage : float, default 0.0
        Null density represented as a percentage (0-100%).
    strategy : MissingValueStrategy, default MissingValueStrategy.IMPUTE_MEAN
        The chosen imputation or removal strategy (EDITABLE).
    fill_value : str, int, float, or None, default None
        The specific value used if the strategy is set to 'FILL_VALUE' (EDITABLE).
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.MISSING_VALUES.
        """
        return RecommendationType.MISSING_VALUES

    missing_count: int = field(default=0, metadata={"editable": True})
    missing_percentage: float = field(default=0.0, metadata={"editable": True})
    strategy: MissingValueStrategy = field(
        default=MissingValueStrategy.IMPUTE_MEAN, metadata={"editable": True}
    )
    fill_value: str | int | float | None = field(
        default=None, metadata={"editable": True}
    )

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "MissingValuesRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        MissingValuesRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a MissingValuesRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the selected strategy to the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.

        Notes
        -----
        If a numeric strategy (mean/median) is selected for a non-numeric
        column, the logic automatically falls back to mode-based imputation.
        """
        if self.column_name not in df.columns:
            return df

        series = df[self.column_name]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if self.strategy == MissingValueStrategy.DROP_ROWS:
            return df.dropna(subset=[self.column_name])

        if self.strategy == MissingValueStrategy.DROP_COLUMN:
            return df.drop(columns=[self.column_name], errors="ignore")

        # Handle Imputation Strategies
        target_fill = None

        if self.strategy == MissingValueStrategy.IMPUTE_MEAN:
            target_fill = series.mean() if is_numeric else self._get_mode(series)

        elif self.strategy == MissingValueStrategy.IMPUTE_MEDIAN:
            target_fill = series.median() if is_numeric else self._get_mode(series)

        elif self.strategy == MissingValueStrategy.IMPUTE_MODE:
            target_fill = self._get_mode(series)

        elif self.strategy == MissingValueStrategy.FILL_VALUE:
            target_fill = self.fill_value

        if target_fill is not None:
            df[self.column_name] = series.fillna(target_fill)

        return df

    def _get_mode(self, series: pd.Series) -> Any:
        """
        Helper to get the first mode value or a fallback value.

        Parameters
        ----------
        series : pd.Series
            The data series to analyze.

        Returns
        -------
        Any
            The first mode found, "Unknown" for object dtypes if empty, or None.
        """
        modes = series.mode()
        if not modes.empty:
            return modes[0]
        return "Unknown" if pd.api.types.is_object_dtype(series) else None

    def info(self) -> None:
        """Displays null statistics and the currently selected strategy."""
        print("  Recommendation: MISSING_VALUES")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(f"    Column: '{self.column_name}'")
        print(
            f"    Detected: {self.missing_count} nulls ({self.missing_percentage:.1f}%)"
        )
        print(f"    Current Strategy: {self.strategy.name} (EDITABLE)")
        if self.strategy == MissingValueStrategy.FILL_VALUE:
            print(f"    Custom Fill Value: {self.fill_value}")
        print(f"    Action: {self._get_action_description()}")

    def _get_action_description(self) -> str:
        """
        Generates a human-readable summary of the chosen strategy.

        Returns
        -------
        str
            A descriptive string mapping the current strategy to a natural language action.
        """
        descriptions = {
            MissingValueStrategy.DROP_ROWS: f"Drop {self.missing_count} rows with missing values",
            MissingValueStrategy.DROP_COLUMN: f"Remove column '{self.column_name}'",
            MissingValueStrategy.IMPUTE_MEAN: "Impute missing values using the mean (fallback to mode)",
            MissingValueStrategy.IMPUTE_MEDIAN: "Impute missing values using the median (fallback to mode)",
            MissingValueStrategy.IMPUTE_MODE: "Impute missing values using the mode",
            MissingValueStrategy.FILL_VALUE: f"Fill missing values with '{self.fill_value}'",
            MissingValueStrategy.LEAVE_AS_NA: "No action (leave values as NaN)",
        }
        return descriptions.get(self.strategy, "")


@dataclass
class EncodingRecommendation(Recommendation):
    """
    Recommendation to transform categorical data into formats suitable for ML models.

    Categorical columns require transformation before they can be used in most
    mathematical models. This class supports One-Hot Encoding (binary columns),
    Label Encoding (integers), or Categorical Dtype (memory optimization).

    Parameters
    ----------
    encoder_type : EncodingStrategy, default EncodingStrategy.ONEHOT
        The chosen EncodingStrategy (EDITABLE).
    unique_values : int, default 0
        Cardinality of the column, used to estimate the impact of One-Hot encoding.
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.ENCODING.
        """
        return RecommendationType.ENCODING

    encoder_type: EncodingStrategy = field(
        default=EncodingStrategy.ONEHOT, metadata={"editable": True}
    )
    unique_values: int = field(default=0, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "EncodingRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        EncodingRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not an EncodingRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the encoding transformation.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.

        Notes
        -----
        - Label and Ordinal strategies preserve null values as NaN/None.
        - One-Hot encoding generates new columns and removes the original.
        """
        if self.column_name not in df.columns:
            return df

        if self.encoder_type == EncodingStrategy.CATEGORICAL:
            df[self.column_name] = df[self.column_name].astype("category")

        elif self.encoder_type == EncodingStrategy.ONEHOT:
            # get_dummies removes the original column by default when columns=[...] is used
            df = pd.get_dummies(df, columns=[self.column_name], drop_first=False)

            # Normalize new column names to lowercase for consistency
            prefix = f"{self.column_name}_".lower()
            rename_map = {
                col: col.lower() for col in df.columns if col.lower().startswith(prefix)
            }
            df = df.rename(columns=rename_map)

        elif self.encoder_type == EncodingStrategy.LABEL:
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            # Masking prevents NaNs from being treated as a distinct category 'nan'
            mask = df[self.column_name].notna()
            encoded = le.fit_transform(df.loc[mask, self.column_name].astype(str))

            # Use nullable Int64 to maintain the NaN state
            res_series = pd.Series(index=df.index, dtype="Int64")
            res_series[mask] = encoded
            df[self.column_name] = res_series

        elif self.encoder_type == EncodingStrategy.ORDINAL:
            from sklearn.preprocessing import OrdinalEncoder

            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            # OrdinalEncoder expects a 2D input (DataFrame)
            df[self.column_name] = oe.fit_transform(df[[self.column_name]])

        return df

    def info(self) -> None:
        """Displays cardinality and the selected encoding strategy."""
        print("  Recommendation: ENCODING")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(f"    Column: '{self.column_name}'")
        print(f"    Cardinality: {self.unique_values} unique values")
        print(f"    Strategy: {self.encoder_type.name} (EDITABLE)")
        print(f"    Action: {self._get_action_description()}")

    def _get_action_description(self) -> str:
        """
        Generates a human-readable summary of the encoding action.

        Returns
        -------
        str
            A descriptive string mapping the current strategy to a natural language action.
        """
        desc_map = {
            EncodingStrategy.CATEGORICAL: "Convert to Categorical dtype (reduces memory usage)",
            EncodingStrategy.ONEHOT: f"Expand into {self.unique_values} binary features",
            EncodingStrategy.LABEL: "Map categories to unique integers (preserves nulls)",
            EncodingStrategy.ORDINAL: "Map categories to sequential integers (unknowns = -1)",
        }
        return desc_map.get(self.encoder_type, "Apply encoding")


@dataclass
class ClassImbalanceRecommendation(Recommendation):
    """
    Recommendation to address skewed class distributions in target variables.

    Severe class imbalance can cause models to ignore minority classes. This
    recommendation suggests resampling techniques (oversampling, undersampling,
    or synthetic generation like SMOTE) to be implemented during the model
    training phase.

    Parameters
    ----------
    majority_percentage : float, default 0.0
        The proportion of the dataset held by the dominant class (0-100).
    strategy : ImbalanceStrategy, default ImbalanceStrategy.SMOTE
        The resampling strategy suggested for the training pipeline (EDITABLE).
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.CLASS_IMBALANCE.
        """
        return RecommendationType.CLASS_IMBALANCE

    majority_percentage: float = field(default=0.0, metadata={"editable": True})
    strategy: ImbalanceStrategy = field(
        default=ImbalanceStrategy.SMOTE, metadata={"editable": True}
    )

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "ClassImbalanceRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        ClassImbalanceRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a ClassImbalanceRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the DataFrame unchanged.

        Resampling should be performed within cross-validation loops to prevent
        data leakage. This recommendation serves as a configuration flag for
        the model training pipeline rather than a pre-processing step.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            The original DataFrame, unmodified.
        """
        return df

    def info(self) -> None:
        """Displays imbalance statistics and training-time instructions."""
        print("  Recommendation: CLASS_IMBALANCE")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(f"    Target Column: '{self.column_name}'")
        print(f"    Skew: {self.majority_percentage:.1f}% Majority Class")
        print(f"    Suggested Strategy: {self.strategy.name} (EDITABLE)")
        print("    Note: This action is applied during training, not pre-processing.")

    def _get_action_description(self) -> str:
        """
        Describes the training-time configuration.

        Returns
        -------
        str
            A string describing the suggested resampling technique for the target column.
        """
        return f"Configure training pipeline to use {self.strategy.value} on '{self.column_name}'"


@dataclass
class OutlierDetectionRecommendation(Recommendation):
    """
    Recommendation to mitigate the impact of extreme numeric values.

    Outliers can skew statistical models and lead to poor generalization.
    This recommendation offers strategies to either squish extreme values
    through robust scaling or remove them using the Interquartile Range (IQR) method.

    Parameters
    ----------
    strategy : OutlierStrategy, default OutlierStrategy.SCALING
        The chosen OutlierStrategy (EDITABLE).
    max_value : float, default 0.0
        The highest value detected during analysis.
    mean_value : float, default 0.0
        The average value detected during analysis.
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.OUTLIER_DETECTION.
        """
        return RecommendationType.OUTLIER_DETECTION

    strategy: OutlierStrategy = field(
        default=OutlierStrategy.SCALING, metadata={"editable": True}
    )
    max_value: float = field(default=0.0, metadata={"editable": True})
    mean_value: float = field(default=0.0, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "OutlierDetectionRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        OutlierDetectionRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not an OutlierDetectionRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies scaling or row-removal based on the selected strategy.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.

        Warnings
        --------
        The 'REMOVE' strategy will filter the DataFrame and reset the index.
        """
        if self.column_name not in df.columns:
            return df

        if self.strategy in {OutlierStrategy.SCALING, OutlierStrategy.ROBUST_SCALER}:
            from sklearn.preprocessing import RobustScaler, StandardScaler

            scaler = (
                RobustScaler()
                if self.strategy == OutlierStrategy.ROBUST_SCALER
                else StandardScaler()
            )

            mask = df[self.column_name].notna()
            if mask.any():
                scaled_values = scaler.fit_transform(df.loc[mask, [self.column_name]])
                df[self.column_name] = df[self.column_name].astype("float64")
                df.loc[mask, self.column_name] = scaled_values.flatten()

        elif self.strategy == OutlierStrategy.REMOVE:
            q1 = df[self.column_name].quantile(0.25)
            q3 = df[self.column_name].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            df = df[(df[self.column_name] >= lower) & (df[self.column_name] <= upper)]
            df = df.reset_index(drop=True)

        return df

    def info(self) -> None:
        """Displays outlier statistics and the mitigation strategy."""
        print("  Recommendation: OUTLIER_DETECTION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(f"    Column: '{self.column_name}'")
        print(f"    Stats: Max={self.max_value:.2f}, Mean={self.mean_value:.2f}")
        print(f"    Strategy: {self.strategy.name} (EDITABLE)")
        print(f"    Action: {self._get_action_description()}")

    def _get_action_description(self) -> str:
        """
        Generates a summary of the handling method.

        Returns
        -------
        str
            A descriptive string mapping the current strategy to a natural language action.
        """
        if self.strategy == OutlierStrategy.REMOVE:
            return "Remove rows outside 1.5x IQR (filtering)"
        if self.strategy == OutlierStrategy.ROBUST_SCALER:
            return "Scale values using median and quantiles (RobustScaler)"
        return "Standardize values to zero mean and unit variance"


@dataclass
class OutlierHandlingRecommendation(Recommendation):
    """
    Recommendation to clean extreme values by capping them or setting them to NaN.

    Unlike scaling (which transforms all values), this recommendation specifically
    targets values outside of user-defined or detected bounds. This is useful for
    removing sensor errors, unrealistic financial entries, or extreme noise.

    Parameters
    ----------
    strategy : OutlierHandlingStrategy, default OutlierHandlingStrategy.CLIP
        The chosen handling method: NULLIFY (set to NaN) or CLIP (cap at the bound) (EDITABLE).
    lower_bound : float, default 0.0
        Values below this threshold are treated as outliers (EDITABLE).
    upper_bound : float, default 0.0
        Values above this threshold are treated as outliers (EDITABLE).
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.OUTLIER_HANDLING.
        """
        return RecommendationType.OUTLIER_HANDLING

    strategy: OutlierHandlingStrategy = field(
        default=OutlierHandlingStrategy.CLIP, metadata={"editable": True}
    )
    lower_bound: float = field(default=0.0, metadata={"editable": True})
    upper_bound: float = field(default=0.0, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "OutlierHandlingRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        OutlierHandlingRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not an OutlierHandlingRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes outlier cleaning using vectorized operations.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be cleaned.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame.

        Notes
        -----
        The NULLIFY strategy will cast integer columns to float64 to accommodate
        NaN values.
        """
        if self.column_name not in df.columns:
            return df

        if self.strategy == OutlierHandlingStrategy.NULLIFY:
            # Vectorized mask for values outside [lower, upper]
            mask = (df[self.column_name] < self.lower_bound) | (
                df[self.column_name] > self.upper_bound
            )
            df.loc[mask, self.column_name] = np.nan

        elif self.strategy == OutlierHandlingStrategy.CLIP:
            # Highly optimized capping operation
            df[self.column_name] = df[self.column_name].clip(
                lower=self.lower_bound, upper=self.upper_bound
            )

        return df

    def info(self) -> None:
        """Displays bound thresholds and the chosen cleaning action."""
        print("  Recommendation: OUTLIER_HANDLING")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(f"    Column: '{self.column_name}'")
        print(
            f"    Bounds: [{self.lower_bound:.2f}, {self.upper_bound:.2f}] (EDITABLE)"
        )
        print(f"    Strategy: {self.strategy.name} (EDITABLE)")
        print(f"    Action: {self._get_action_description()}")

    def _get_action_description(self) -> str:
        """
        Generates a summary of how extreme values will be treated.

        Returns
        -------
        str
            A descriptive string mapping the chosen strategy and bounds to a
            natural language action.
        """
        bounds_str = f"[{self.lower_bound:.2f}, {self.upper_bound:.2f}]"
        if self.strategy == OutlierHandlingStrategy.NULLIFY:
            return f"Set values outside {bounds_str} to NaN"
        return f"Cap (clip) values to the range {bounds_str}"


@dataclass
class CategoricalConversionRecommendation(Recommendation):
    """
    Recommendation to convert a string/object column to the pandas 'category' dtype.

    Categorical dtypes are highly efficient for columns with low-to-medium
    cardinality. They reduce memory footprint by storing data as integer codes
    pointing to a mapping table, which also speeds up operations like sorting
    and grouping.

    Parameters
    ----------
    unique_values : int, default 0
        The number of distinct categories found in the column.
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.CATEGORICAL_CONVERSION.
        """
        return RecommendationType.CATEGORICAL_CONVERSION

    unique_values: int = field(default=0, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "CategoricalConversionRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        CategoricalConversionRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a CategoricalConversionRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the target column to 'category' dtype.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to be converted.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the column converted to categorical dtype, or the
            original DataFrame if the column is missing.
        """
        if self.column_name not in df.columns:
            return df

        df[self.column_name] = df[self.column_name].astype("category")
        return df

    def info(self) -> None:
        """Displays memory optimization benefits and the target column."""
        print("  Recommendation: CATEGORICAL_CONVERSION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print("    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Cardinality: {self.unique_values} unique values")
        if self.alias:
            print(f"    Alias: {self.alias}")
        print("    Action: Convert to categorical dtype for memory optimization")


@dataclass
class BooleanClassificationRecommendation(Recommendation):
    """
    Recommendation to convert a binary-value column to a boolean dtype.

    Identifies columns that contain exactly two unique values (e.g., 0/1, Y/N)
    and suggests converting them to a proper boolean type for better
    semantic clarity and memory efficiency.

    Parameters
    ----------
    values : list of Any, optional
        The two unique values identified in the column that will be mapped to
        True and False.
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.BOOLEAN_CLASSIFICATION.
        """
        return RecommendationType.BOOLEAN_CLASSIFICATION

    values: list[Any] = field(default_factory=list, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "BooleanClassificationRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        BooleanClassificationRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a BooleanClassificationRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes indicators and converts the column to boolean.

        Uses a predefined set of 'True' indicators (e.g., 'Y', 'YES', '1') to
        determine mapping direction. Vectorized replacement ensures that
        unmatched values do not introduce NaNs.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to be transformed.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the target column converted to boolean dtype.
        """
        if self.column_name not in df.columns or len(self.values) != 2:
            return df

        # 1. Standardize indicators
        TRUE_INDICATORS = {"Y", "YES", "1", "TRUE", "ON", "T", "ACTIVE"}

        val_a_str = str(self.values[0]).strip().upper()
        _ = str(self.values[1]).strip().upper()

        # 2. Determine Mapping: If the first value is a 'True' indicator,
        # it gets True and the second gets False. Otherwise, reverse it.
        if val_a_str in TRUE_INDICATORS:
            mapping = {self.values[0]: True, self.values[1]: False}
        else:
            # Default or specific 'val_b' as True
            mapping = {self.values[0]: False, self.values[1]: True}

        # 3. Vectorized replacement
        # We explicitly call infer_objects(copy=False) to resolve the silent downcasting
        # deprecation warning in Pandas 3.0+.
        df[self.column_name] = (
            df[self.column_name].replace(mapping).infer_objects().astype(bool)
        )

        return df

    def info(self) -> None:
        """Displays the detected binary values and the conversion intent."""
        print("  Recommendation: BOOLEAN_CLASSIFICATION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print("    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Detected Binary Values: {self.values}")
        print("    Action: Map to True/False and convert to boolean dtype")


@dataclass
class BinningRecommendation(Recommendation):
    """
    Recommendation to discretize a continuous numeric column into range-based bins.

    This transformation is useful for non-linear numeric data. It segments
    values into intervals (e.g., [0-10, 11-20]) and automatically expands
    the result into multiple one-hot encoded binary features.

    Parameters
    ----------
    bins : list of float, optional
        List of numeric edges for the bins (e.g., [0, 10, 20, 100]).
    labels : list of str, optional
        Descriptive names for the resulting categories (e.g., ['Low', 'Med', 'High']).
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.BINNING.
        """
        return RecommendationType.BINNING

    bins: list[float] = field(default_factory=list, metadata={"editable": True})
    labels: list[str] = field(default_factory=list, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "BinningRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        BinningRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a BinningRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies range-based binning followed by one-hot encoding.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the numeric column to be binned.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with new binary features, or the original
            DataFrame if the operation fails or the column is missing.

        Notes
        -----
        This process involves two steps:
        1. Discretization of data into categories using `pd.cut`.
        2. Expansion of categories into binary features using `pd.get_dummies`.
        """
        if self.column_name not in df.columns:
            return df

        try:
            # Step 1: Discretize the data
            df[self.column_name] = pd.cut(
                df[self.column_name],
                bins=self.bins,
                labels=self.labels,
                right=True,
                include_lowest=True,
            )

            # Step 2: Expand into binary features (One-Hot)
            df = pd.get_dummies(df, columns=[self.column_name], drop_first=False)

            # Normalize column names to lowercase to match our Encoding style
            prefix = f"{self.column_name}_".lower()
            df = df.rename(
                columns={
                    col: col.lower()
                    for col in df.columns
                    if col.lower().startswith(prefix)
                }
            )

        except Exception as e:
            import warnings

            warnings.warn(f"Binning failed for '{self.column_name}': {e}")
            return df

        return df

    def info(self) -> None:
        """Displays binning edges and resulting categorical labels."""
        print("  Recommendation: BINNING")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print("    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Range Edges: {self.bins}")
        print(f"    Category Labels: {self.labels}")
        print(
            f"    Action: Segment into {len(self.labels)} bins and expand via One-Hot encoding"
        )


@dataclass
class IntegerConversionRecommendation(Recommendation):
    """
    Recommendation to convert floating-point or object columns to integer types.

    This is suggested when a column contains whole numbers. By converting
    to a specific bit-depth (e.g., int16, int32), memory usage is reduced
    and data semantics are clarified.

    Parameters
    ----------
    target_depth : BitDepth, default BitDepth.INT32
        The specific bit-depth (BitDepth enum) for the conversion.
    integer_count : int, default 0
        The number of rows currently containing whole numbers.
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.INT_CONVERSION.
        """
        return RecommendationType.INT_CONVERSION

    target_depth: BitDepth = field(default=BitDepth.INT32, metadata={"editable": True})
    integer_count: int = field(default=0, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "IntegerConversionRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        IntegerConversionRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not an IntegerConversionRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the column to the target bit-depth.

        Automatically detects NaNs and upgrades to Pandas' 'Nullable' integer
        types (e.g., 'Int32' instead of 'int32') to prevent casting back to float.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to be converted.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the column converted to the target integer type.
        """
        if self.column_name not in df.columns:
            return df

        # Detect nulls to determine if we need a Nullable Extension Type
        has_nans = df[self.column_name].isna().any()

        # Determine the string representation (e.g., "int32")
        dtype_str: str = str(self.target_depth.value)

        if has_nans:
            # Shift from standard numpy 'int' to Pandas nullable 'Int'
            dtype_str = dtype_str.capitalize()

        try:
            df[self.column_name] = df[self.column_name].astype(cast(Any, dtype_str))
        except (ValueError, TypeError) as e:
            import warnings

            warnings.warn(f"Integer conversion failed for '{self.column_name}': {e}")

        return df

    def info(self) -> None:
        """Displays target depth and the number of valid integers detected."""
        print("  Recommendation: INT_CONVERSION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print("    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Found: {self.integer_count} whole numbers")

        # Indicate if we are using Nullable types in the action description
        is_nullable = " (Nullable)" if "(None)" in str(self.target_depth) else ""
        print(f"    Action: Convert to {self.target_depth.value}{is_nullable}")


@dataclass
class FloatConversionRecommendation(Recommendation):
    """
    Recommendation to optimize floating-point precision for memory or speed.

    Downcasting (e.g., from float64 to float32) can drastically reduce memory
    usage in large datasets. While float64 offers higher precision, float32 is
    typically sufficient for the majority of machine learning use cases.

    Parameters
    ----------
    target_depth : BitDepth, default BitDepth.FLOAT32
        The specific bit-depth (BitDepth enum) for the conversion.
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.FLOAT_CONVERSION.
        """
        return RecommendationType.FLOAT_CONVERSION

    target_depth: BitDepth = field(
        default=BitDepth.FLOAT32, metadata={"editable": True}
    )

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "FloatConversionRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        FloatConversionRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a FloatConversionRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the column to the target floating-point bit-depth.

        Standard floating-point types (float32, float64) natively support NaNs.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to be converted.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the column converted to the target floating-point type.
        """
        if self.column_name not in df.columns:
            return df

        try:
            from typing import Any, cast

            df[self.column_name] = df[self.column_name].astype(
                cast(Any, self.target_depth.value)
            )
        except (ValueError, TypeError) as e:
            import warnings

            warnings.warn(f"Float conversion failed for '{self.column_name}': {e}")

        return df

    def info(self) -> None:
        """Displays target precision and the target column."""
        print("  Recommendation: FLOAT_CONVERSION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print("    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Action: Convert to {self.target_depth.value}")


@dataclass
class DecimalPrecisionRecommendation(Recommendation):
    """
    Recommendation to standardize decimal precision and optimize numeric storage.

    Standardizes floating-point columns by rounding to a specific decimal depth.
    If rounding results in whole numbers, it can automatically convert the column
    to an integer type. For remaining floats, it suggests downcasting to float32
    to reduce memory usage.

    Parameters
    ----------
    max_decimal_places : int, default 0
        Maximum decimal digits to retain (EDITABLE).
    min_value : float, default 0.0
        Reference lower bound for the column's data.
    max_value : float, default 0.0
        Reference upper bound for the column's data.
    convert_to_int : bool, default False
        If True, attempts integer conversion after rounding if no decimals remain.
    rounding_mode : RoundingMode, default RoundingMode.NEAREST
        The algorithm used for rounding (e.g., BANKERS, UP, DOWN) (EDITABLE).
    scale_factor : float, optional
        Multiplier applied to the data before the rounding step.
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.DECIMAL_PRECISION_OPTIMIZATION.
        """
        return RecommendationType.DECIMAL_PRECISION_OPTIMIZATION

    max_decimal_places: int = field(default=0, metadata={"editable": True})
    min_value: float = field(default=0.0, metadata={"editable": True})
    max_value: float = field(default=0.0, metadata={"editable": True})
    convert_to_int: bool = field(default=False, metadata={"editable": True})
    rounding_mode: RoundingMode = field(
        default=RoundingMode.NEAREST, metadata={"editable": True}
    )
    scale_factor: float | None = field(default=None, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "DecimalPrecisionRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        DecimalPrecisionRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a DecimalPrecisionRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self) -> None:
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes scaling, rounding, and conditional type-downcasting.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to be optimized.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with updated precision and optimized storage types.

        Notes
        -----
        The method follows a three-step pipeline:
        1. Multiplies by `scale_factor` if provided.
        2. Applies rounding based on the selected `rounding_mode`.
        3. Attempts to downcast to 'Int64' (if `convert_to_int` is True) or 'float32'
           to reduce memory footprint.
        """
        if self.column_name not in df.columns:
            return df

        try:
            from typing import Any, cast

            series = df[self.column_name]

            # 1. Scaling
            if self.scale_factor is not None:
                series = series * self.scale_factor

            # 2. Rounding
            factor = 10**self.max_decimal_places
            if self.rounding_mode == RoundingMode.BANKERS:
                series = np.round(series * factor) / factor
            elif self.rounding_mode == RoundingMode.UP:
                series = np.ceil(series * factor) / factor
            elif self.rounding_mode == RoundingMode.DOWN:
                series = np.floor(series * factor) / factor
            else:  # NEAREST / Half-Up
                series = np.floor(series * factor + 0.5) / factor

            # 3. Type Optimization
            non_null = series.dropna()

            if (
                self.convert_to_int
                and not non_null.empty
                and ((non_null % 1) == 0).all()
            ):
                dtype = "Int64" if series.isna().any() else "int64"
                df[self.column_name] = series.astype(cast(Any, dtype))
            elif self.max_decimal_places <= 6:
                df[self.column_name] = series.astype(cast(Any, "float32"))
            else:
                df[self.column_name] = series

        except Exception as e:
            import warnings

            warnings.warn(
                f"Precision optimization failed for '{self.column_name}': {e}"
            )

        return df

    def info(self) -> None:
        """Displays rounding configuration and memory optimization status."""
        print("  Recommendation: DECIMAL_PRECISION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print("    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Data Range: [{self.min_value}, {self.max_value}]")
        print(
            f"    Strategy: Round to {self.max_decimal_places} places ({self.rounding_mode.name})"
        )

        if self.max_decimal_places == 0 and self.convert_to_int:
            print("    Optimization: Will attempt conversion to Integer")
        elif self.max_decimal_places <= 6:
            print("    Optimization: Will downcast to float32 (memory savings)")


@dataclass
class ValueReplacementRecommendation(Recommendation):
    """
    Recommendation to replace non-numeric placeholder strings with a numeric-compatible value.

    This detector identifies columns that are primarily numeric but contain specific
    string-based placeholders (e.g., 'n/a', 'tbd', 'unknown'). It allows for
    batch replacement to prepare the column for numeric casting.

    Parameters
    ----------
    non_numeric_values : list of str, optional
        A list of specific strings identified as placeholders.
    non_numeric_count : int, default 0
        The total occurrences of these placeholders in the column.
    replacement_value : float or str, default np.nan
        The value to insert in place of the strings (EDITABLE).
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.VALUE_REPLACEMENT.
        """
        return RecommendationType.VALUE_REPLACEMENT

    non_numeric_values: list[str] = field(
        default_factory=list, metadata={"editable": True}
    )
    non_numeric_count: int = field(default=0, metadata={"editable": True})
    replacement_value: float | str = field(default=np.nan, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "ValueReplacementRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        ValueReplacementRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a ValueReplacementRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        """Computes the stable identity and freezes core fields."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the replacement on the target column.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to be cleaned.

        Returns
        -------
        pd.DataFrame
            The DataFrame with placeholders replaced by the target value.

        Notes
        -----
        This method uses a single-pass replacement for optimization. It does not
        automatically cast the column to a numeric type, as other recommendations
        in the pipeline typically handle type conversion.
        """
        if not self.non_numeric_values:
            return df

        # Optimization: Pass the whole list to .replace() for a single-pass operation
        df[self.column_name] = df[self.column_name].replace(
            self.non_numeric_values, self.replacement_value
        )
        return df

    def info(self) -> None:
        """Prints a summary of the placeholders found and the intended replacement."""
        print(f"  Recommendation: {self.rec_type.name}")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print("    Source: User Hint")
        print(f"    Column: '{self.column_name}'")
        print(f"    Non-numeric values: {self.non_numeric_values}")
        print(f"    Count: {self.non_numeric_count}")
        print(f"    Replacement value: {self.replacement_value} (EDITABLE)")
        print(f"    Action: {self._get_action_description()}")

    def _get_action_description(self) -> str:
        """
        Generates a human-readable summary of the replacement action.

        Returns
        -------
        str
            A descriptive string mapping the placeholders to the replacement target.
        """
        values_str = "', '".join(map(str, self.non_numeric_values))
        target = (
            "NaN" if pd.isna(self.replacement_value) else f"'{self.replacement_value}'"
        )
        return f"Replace '{values_str}' with {target} in '{self.column_name}'"


@dataclass
class FeatureInteractionRecommendation(Recommendation):
    """
    Recommendation to engineer a new feature by combining two existing columns.

    This identifies statistically significant interactions—such as multiplying
    unit price by quantity or dividing revenue by headcount—to create
    higher-order features for modeling.

    Parameters
    ----------
    column_name_2 : str, default ""
        The secondary column used in the interaction.
    interaction_type : InteractionType, default InteractionType.STATUS_IMPACT
        The domain-specific category of the interaction.
    operation : str, default "*"
        The mathematical operator ('*' or '/').
    rationale : str, default ""
        A human-readable explanation of the feature's potential value.
    derived_name : str, default ""
        The name of the new column. If empty, it is auto-generated during
        initialization.
    priority_score : float, default 0.0
        Magnitude of the statistical signal (0.0 to 1.0).
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.FEATURE_INTERACTION.
        """
        return RecommendationType.FEATURE_INTERACTION

    column_name_2: str = field(default="", metadata={"editable": True})
    interaction_type: InteractionType = field(
        default=InteractionType.STATUS_IMPACT, metadata={"editable": True}
    )
    operation: str = field(default="*", metadata={"editable": True})
    rationale: str = field(default="", metadata={"editable": True})
    derived_name: str = field(default="", metadata={"editable": True})
    priority_score: float = field(default=0.0, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "FeatureInteractionRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        FeatureInteractionRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a FeatureInteractionRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        """Initializes identifiers and handles default naming logic."""
        if not self.derived_name:
            suffix = self.column_name_2
            sep = "_" if self.operation == "*" else "_vs_"
            self.derived_name = f"{self.column_name}{sep}{suffix}"

        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the interaction and appends the new column to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the columns to be combined.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the newly engineered feature column appended.

        Raises
        ------
        KeyError
            If one or both columns are missing from the input DataFrame.
        ValueError
            If an unsupported mathematical operator is provided.

        Notes
        -----
        The method handles division by zero by converting denominators of 0
        to NaN to prevent calculation errors.
        """
        if self.column_name not in df.columns or self.column_name_2 not in df.columns:
            raise KeyError(
                f"Interaction failed: One or both columns ('{self.column_name}', "
                f"'{self.column_name_2}') missing from DataFrame."
            )

        if self.operation == "*":
            df[self.derived_name] = df[self.column_name] * df[self.column_name_2]
        elif self.operation == "/":
            denominator = df[self.column_name_2].replace(0, np.nan)
            df[self.derived_name] = df[self.column_name] / denominator
        else:
            raise ValueError(f"Unsupported interaction operation: {self.operation}")

        return df

    def info(self) -> None:
        """Prints a detailed summary of the feature engineering step."""
        op_map = {"*": "multiplication", "/": "division"}
        op_desc = op_map.get(self.operation, self.operation)

        print(f"  Recommendation: FEATURE_INTERACTION ({op_desc})")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(
            f"    Interaction: '{self.column_name}' {self.operation} '{self.column_name_2}'"
        )
        print(f"    Resulting Column: '{self.derived_name}'")
        print(f"    Rationale: {self.rationale}")
        print(f"    Priority Score: {self.priority_score:.2f}")


@dataclass
class DatetimeConversionRecommendation(Recommendation):
    """
    Recommendation to convert a string or object column into a datetime object.

    Proper datetime typing enables time-series analysis, period extraction,
    and optimized storage. If a specific format string is provided,
    conversion is significantly faster and more reliable.

    Parameters
    ----------
    detected_format : str, optional
        The strptime format string (e.g., '%Y-%m-%d %H:%M:%S').
        If None, the parser will attempt to infer the format for each row.
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.DATETIME_CONVERSION.
        """
        return RecommendationType.DATETIME_CONVERSION

    detected_format: str | None = field(default=None, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "DatetimeConversionRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        DatetimeConversionRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a DatetimeConversionRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        """Finalizes the unique identifier and locks the recommendation state."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the target column to datetime64[ns] dtype.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to be converted.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the target column converted to datetime objects.

        Notes
        -----
        Uses high-performance parsing if `detected_format` is available.
        Falls back to 'mixed' parsing for flexible inference when the format is unknown,
        ensuring compatibility with varied string representations.
        """
        if self.detected_format:
            df[self.column_name] = pd.to_datetime(
                df[self.column_name], format=self.detected_format, errors="coerce"
            )
        else:
            df[self.column_name] = pd.to_datetime(
                df[self.column_name], format="mixed", errors="coerce"
            )
        return df

    def info(self) -> None:
        """Displays the conversion details and the format string being used."""
        print("  Recommendation: DATETIME_CONVERSION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(f"    Column: '{self.column_name}'")

        if self.detected_format:
            print(f"    Format Found: '{self.detected_format}'")
            print("    Action: Convert to datetime using specified format.")
        else:
            print("    Format Found: None (Auto-detect)")
            print("    Action: Convert to datetime (invalid values coerced to NaT).")


@dataclass
class FeatureExtractionRecommendation(Recommendation):
    """
    Recommendation to derive granular features from complex types, primarily datetimes.

    This class decomposes datetime columns into constituent parts (Year, Month, etc.)
    and supports cyclic encoding (Sine/Cosine transforms). Cyclic encoding is
    essential for models to understand that time is periodic (e.g., December and
    January are 'close').

    Parameters
    ----------
    properties : DatetimeProperty, default DatetimeProperty(0)
        A bitmask defining which specific features to extract from the datetime
        column.
    output_prefix : str, default ""
        A string prepended to auto-generated column names. If empty, the
        `column_name` followed by an underscore is used.
    output_columns : dict of str to str, optional
        A dictionary for manual column renaming. If a 'sin' column is renamed,
        the 'cos' counterpart is automatically renamed to match.
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.FEATURE_EXTRACTION.
        """
        return RecommendationType.FEATURE_EXTRACTION

    properties: DatetimeProperty = field(
        default=DatetimeProperty(0), metadata={"editable": True}
    )
    output_prefix: str = field(default="", metadata={"editable": True})
    output_columns: dict[str, str] | None = field(
        default=None, metadata={"editable": True}
    )

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "FeatureExtractionRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        FeatureExtractionRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a FeatureExtractionRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        """Initializes identifiers and handles default naming logic."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts selected properties and appends them as new columns to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to be decomposed.

        Returns
        -------
        pd.DataFrame
            The DataFrame with extracted features appended as new columns.

        Notes
        -----
        Attempts a 'mixed' datetime conversion if the target column is not already
        a datetime type. If the conversion fails, the original DataFrame is
        returned.
        """
        # Ensure we are working with datetimes
        if not pd.api.types.is_datetime64_any_dtype(df[self.column_name]):
            try:
                # Use "mixed" to match our utility's detection logic
                df[self.column_name] = pd.to_datetime(
                    df[self.column_name], errors="coerce", format="mixed"
                )
            except Exception:
                return df

        prefix = self.output_prefix if self.output_prefix else f"{self.column_name}_"
        dt = df[self.column_name].dt
        feature_series: dict[str, pd.Series] = {}

        # Standard Extractions
        if DatetimeProperty.YEAR in self.properties:
            feature_series["year"] = dt.year
        if DatetimeProperty.MONTH in self.properties:
            feature_series["month"] = dt.month
        if DatetimeProperty.DAY in self.properties:
            feature_series["day"] = dt.day
        if DatetimeProperty.DAYOFWEEK in self.properties:
            feature_series["dayofweek"] = dt.dayofweek
        if DatetimeProperty.DAYOFYEAR in self.properties:
            feature_series["dayofyear"] = dt.dayofyear
        if DatetimeProperty.QUARTER in self.properties:
            feature_series["quarter"] = dt.quarter
        if DatetimeProperty.WEEK in self.properties:
            feature_series["week"] = dt.isocalendar().week
        if DatetimeProperty.HOUR in self.properties:
            feature_series["hour"] = dt.hour
        if DatetimeProperty.MINUTE in self.properties:
            feature_series["minute"] = dt.minute
        if DatetimeProperty.SECOND in self.properties:
            feature_series["second"] = dt.second
        if DatetimeProperty.IS_MONTH_END in self.properties:
            feature_series["is_month_end"] = dt.is_month_end
        if DatetimeProperty.IS_MONTH_START in self.properties:
            feature_series["is_month_start"] = dt.is_month_start

        # Cyclic encoding helpers
        def _to_rad_series(series: pd.Series, max_val: float) -> pd.Series:
            return series * (2 * np.pi / max_val)

        if DatetimeProperty.SIN_HOUR in self.properties:
            feature_series["sin_hour"] = pd.Series(np.sin(_to_rad_series(dt.hour, 24)))
        if DatetimeProperty.COS_HOUR in self.properties:
            feature_series["cos_hour"] = pd.Series(np.cos(_to_rad_series(dt.hour, 24)))
        if DatetimeProperty.SIN_DAYOFWEEK in self.properties:
            feature_series["sin_dayofweek"] = pd.Series(
                np.sin(_to_rad_series(dt.dayofweek, 7))
            )
        if DatetimeProperty.COS_DAYOFWEEK in self.properties:
            feature_series["cos_dayofweek"] = pd.Series(
                np.cos(_to_rad_series(dt.dayofweek, 7))
            )
        if DatetimeProperty.SIN_MONTH in self.properties:
            feature_series["sin_month"] = pd.Series(
                np.sin(_to_rad_series(dt.month - 1, 12))
            )
        if DatetimeProperty.COS_MONTH in self.properties:
            feature_series["cos_month"] = pd.Series(
                np.cos(_to_rad_series(dt.month - 1, 12))
            )

        # Build final mapping including inferred pairs
        mapping = (self.output_columns or {}).copy()
        for feat in list(feature_series.keys()):
            if feat.startswith(("sin_", "cos_")):
                mate = ("cos_" if "sin_" in feat else "sin_") + feat[4:]
                if feat in mapping and mate not in mapping:
                    mapping[mate] = (
                        mapping[feat].replace("sin", "cos", 1)
                        if "sin" in mapping[feat]
                        else mapping[feat].replace("cos", "sin", 1)
                    )

        # Final column assignment
        for name, data in feature_series.items():
            final_name = mapping.get(name, f"{prefix}{name}")
            df[final_name] = data

        return df

    def info(self) -> None:
        """Displays a summary of which features will be extracted and their naming scheme."""
        active_props: list[str] = [
            p.name
            for p in DatetimeProperty
            if p in self.properties and p.name is not None
        ]

        print("  Recommendation: FEATURE_EXTRACTION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        print(f"    Column: '{self.column_name}'")

        features_str = ", ".join(active_props) if active_props else "None"
        print(f"    Features: {features_str}")

        prefix = self.output_prefix if self.output_prefix else f"{self.column_name}_"
        print(f"    Output prefix: '{prefix}'")

        if self.output_columns:
            print(f"    Output columns: {self.output_columns}")


@dataclass
class DatetimeDurationRecommendation(Recommendation):
    """
    Recommendation to calculate the elapsed time between two datetime columns.

    This creates a numeric feature representing the duration (delta) between
    a start and end point. This is useful for calculating "Time to Resolution,"
    "Shipping Duration," or "Age".

    Parameters
    ----------
    start_column : str, default ""
        The 'from' datetime column.
    end_column : str, default ""
        The 'to' datetime column.
    unit : str, default "minutes"
        The scale of the resulting number ('seconds', 'minutes', 'hours', 'days').
    output_column : str, optional
        The name of the new feature. Auto-generated if not provided.
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.FEATURE_EXTRACTION.
        """
        return RecommendationType.FEATURE_EXTRACTION

    start_column: str = field(default="", metadata={"editable": True})
    end_column: str = field(default="", metadata={"editable": True})
    unit: str = field(default="minutes", metadata={"editable": True})
    output_column: str | None = field(default=None, metadata={"editable": True})

    @classmethod
    def get_by_id(
        cls, manager: "RecommendationManager", rec_id: str
    ) -> "DatetimeDurationRecommendation | None":
        """
        Retrieve and validate a recommendation of this type from the manager.

        Parameters
        ----------
        manager : RecommendationManager
            The manager instance containing the pipeline.
        rec_id : str
            The unique identifier for the recommendation.

        Returns
        -------
        DatetimeDurationRecommendation or None
            The matched recommendation instance if found and of the correct type.

        Raises
        ------
        TypeError
            If the recommendation exists but is not a DatetimeDurationRecommendation.
        """
        rec = manager.get_by_id(rec_id)
        if rec is None:
            return None
        if not isinstance(rec, cls):
            raise TypeError(
                f"Recommendation {rec_id} is a {type(rec).__name__}, not a {cls.__name__}"
            )
        return rec

    def __post_init__(self):
        """Generates a default output name and locks the recommendation state."""
        if not self.output_column:
            self.output_column = f"{self.start_column}_{self.end_column}_{self.unit}"
        self.id = self.compute_stable_id()
        self._lock_fields()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subtracts the start column from the end column and converts to the target unit.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the datetime columns for calculation.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the newly calculated duration feature column.

        Notes
        -----
        Missing values in either source column will result in NaN in the output.
        Uses the pandas `.dt` accessor for vectorized `total_seconds` calculation.
        """
        if self.start_column not in df.columns or self.end_column not in df.columns:
            return df

        # Calculate the raw timedelta
        delta = df[self.end_column] - df[self.start_column]
        total_sec = delta.dt.total_seconds()

        # Scale based on unit
        if self.unit == "seconds":
            df[self.output_column] = total_sec
        elif self.unit == "hours":
            df[self.output_column] = total_sec / 3600
        elif self.unit == "days":
            df[self.output_column] = total_sec / 86400
        else:  # minutes
            df[self.output_column] = total_sec / 60

        return df

    def info(self) -> None:
        """Prints the duration calculation details."""
        print("  Recommendation: DATETIME_DURATION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print("    Source: User Hint")
        print(f"    Calculation: '{self.end_column}' - '{self.start_column}'")
        print(f"    Output Column: '{self.output_column}'")
        print(f"    Result Unit: {self.unit}")


@dataclass
class AggregationRecommendation(Recommendation):
    """
    Recommendation to aggregate multiple source columns or validate an existing total.

    This class supports horizontal (row-wise) operations like sum, mean, min, and max.
    If the `output_column` already exists in the DataFrame, this recommendation
    acts as a validator, identifying rows where the stored total doesn't match
    the computed total.

    Parameters
    ----------
    agg_columns : list of str, optional
        The list of columns to aggregate.
    agg_op : str, default "sum"
        The operation to perform ('sum', 'mean', 'min', 'max').
    output_column : str, default ""
        The target column name for the result or validation check.
    validation_mismatch_count : int, default 0
        Count of rows where computed values do not match existing values.
    decimal_places : int, optional
        Precision for rounding (supports negative for rounding to tens/hundreds).
    rounding_mode : RoundingMode, default RoundingMode.NEAREST
        The strategy used for decimals (e.g., BANKERS, UP, DOWN).
    scale_factor : float, optional
        A multiplier applied post-aggregation but pre-rounding.
    **kwargs : dict
        Additional keyword arguments passed to the Recommendation base class.
    """

    @property
    def rec_type(self) -> RecommendationType:
        """
        The categorical type of the recommendation.

        Returns
        -------
        RecommendationType
            Always returns RecommendationType.FEATURE_AGGREGATION.
        """
        return RecommendationType.FEATURE_AGGREGATION

    agg_columns: list[str] = field(default_factory=list, metadata={"editable": True})
    agg_op: str = field(default="sum", metadata={"editable": True})
    output_column: str = field(default="", metadata={"editable": True})
    validation_mismatch_count: int = field(default=0, metadata={"editable": True})
    decimal_places: int | None = field(default=None, metadata={"editable": True})
    rounding_mode: RoundingMode = field(
        default=RoundingMode.NEAREST, metadata={"editable": True}
    )
    scale_factor: float | None = field(default=None, metadata={"editable": True})

    def __post_init__(self):
        """Initializes identifiers and locks the recommendation state."""
        self.id = self.compute_stable_id()
        self._lock_fields()

    def _aggregate(self, df: pd.DataFrame) -> pd.Series:
        """
        Performs the row-wise math operation across the specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the source columns.

        Returns
        -------
        pd.Series
            A series containing the results of the row-wise operation.

        Raises
        ------
        ValueError
            If an unsupported aggregation operation is specified.
        """
        op_map = {
            "sum": df[self.agg_columns].sum,
            "mean": df[self.agg_columns].mean,
            "min": df[self.agg_columns].min,
            "max": df[self.agg_columns].max,
        }

        if self.agg_op not in op_map:
            raise ValueError(f"Unsupported aggregation operation: {self.agg_op}")

        return op_map[self.agg_op](axis=1)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes aggregation, scaling, and rounding.

        Updates the mismatch count if the output column already exists;
        otherwise, creates the new column.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be transformed or validated.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with updated features or validation counts.
        """
        computed = self._aggregate(df)

        if self.scale_factor is not None:
            computed *= self.scale_factor

        if self.decimal_places is not None:
            factor = 10**self.decimal_places
            if self.rounding_mode == RoundingMode.BANKERS:
                computed = np.round(computed * factor) / factor
            elif self.rounding_mode == RoundingMode.UP:
                computed = np.ceil(computed * factor) / factor
            elif self.rounding_mode == RoundingMode.DOWN:
                computed = np.floor(computed * factor) / factor
            else:  # NEAREST (Half-up)
                computed = np.floor(computed * factor + 0.5) / factor

        if self.output_column in df.columns:
            diff = df[self.output_column] - computed
            self.validation_mismatch_count = int(diff.fillna(0).ne(0).sum())
        else:
            df[self.output_column] = computed

        return df

    def info(self) -> None:
        """Prints aggregation summary, including any validation discrepancies found."""
        print("  Recommendation: FEATURE_AGGREGATION")
        print(f"    ID: {self.id}")
        print(f"    Enabled: {self.enabled}")
        if self.is_locked:
            print("    Source: User Hint")
        print(
            f"    Operation: {self.agg_op.upper()} of {len(self.agg_columns)} columns"
        )
        print(f"    Target Column: '{self.output_column}'")

        if self.decimal_places is not None:
            print(
                f"    Rounding: {self.decimal_places} places ({self.rounding_mode.name})"
            )

        if self.validation_mismatch_count > 0:
            print(
                f"    [!] Validation Warning: {self.validation_mismatch_count} mismatches detected"
            )


@dataclass
class ColumnHint:
    """
    User-provided metadata to override or guide the recommendation engine.

    Instead of relying solely on automated inference, users can provide a `ColumnHint`
    to specify the 'logical type' of a column and set constraints like rounding,
    bounds, or specific feature extraction needs.

    Parameters
    ----------
    logical_type : ColumnHintType, optional
        The inferred or user-specified type of the column.
    floor : float, optional
        The minimum valid value for numeric data.
    ceiling : float, optional
        The maximum valid value for numeric data.
    datetime_format : str, optional
        The strptime format string for datetime parsing.
    datetime_features : list of DatetimeProperty, optional
        Specific properties to extract from a datetime column.
    output_names : dict of str to str, optional
        Mapping for custom column naming in transformations.
    agg_columns : list of str, optional
        Source columns for aggregate feature engineering.
    agg_op : str, optional
        The mathematical operator for aggregation (e.g., 'sum', 'mean').
    convert_to_int : bool, optional
        If True, forces conversion to an integer type.
    decimal_places : int, optional
        Number of decimal digits to retain during rounding.
    rounding_mode : RoundingMode, optional
        The algorithm used for rounding operations.
    scale_factor : float, optional
        Multiplier applied to the data before processing.
    lat_bounds : tuple of (float, float), optional
        Valid latitude range for geospatial data.
    lon_bounds : tuple of (float, float), optional
        Valid longitude range for geospatial data.
    unit : str, optional
        Measurement unit for distance-based data.
    is_ignored : bool, default False
        If True, the column is completely bypassed by the engine.
    should_drop : bool, default False
        If True, forces a recommendation to drop the column.

    Notes
    -----
    It is highly recommended to use the provided factory methods (e.g., `.financial()`,
    `.datetime()`) rather than instantiating this class directly.
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
    scale_factor: float | None = None
    lat_bounds: tuple[float, float] | None = None
    lon_bounds: tuple[float, float] | None = None
    unit: str | None = None
    is_ignored: bool = False
    should_drop: bool = False

    @classmethod
    def datetime(
        cls,
        datetime_format: str | None = None,
        datetime_features: list[DatetimeProperty] | None = None,
        output_names: dict[str, str] | None = None,
    ) -> "ColumnHint":
        """
        Hint that a column should be treated as a Datetime.

        Parameters
        ----------
        datetime_format : str, optional
            Explicit format string for parsing.
        datetime_features : list of DatetimeProperty, optional
            Features to extract (e.g., Year, Month).
        output_names : dict of str to str, optional
            Renaming mapping for extracted features.

        Returns
        -------
        ColumnHint
            A hint configured for datetime processing.
        """
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
        """
        Hint for currency or financial data, defaulting to 2 decimal places.

        Parameters
        ----------
        floor : float, optional
            Minimum expected value.
        ceiling : float, optional
            Maximum expected value.
        decimal_places : int, default 2
            Rounding precision.
        rounding_mode : RoundingMode, default RoundingMode.NEAREST
            Algorithm for rounding.
        scale_factor : float, optional
            Pre-processing multiplier.

        Returns
        -------
        ColumnHint
            A hint configured for financial numeric data.
        """
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
        """
        Hint that a column is categorical (e.g., for encoding purposes).

        Parameters
        ----------
        output_names : dict of str to str, optional
            A mapping for custom column naming during encoding.

        Returns
        -------
        ColumnHint
            A hint configured for categorical processing.
        """
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
        """
        Hint for general numeric data with optional bounds and precision controls.

        Parameters
        ----------
        floor : float, optional
            The minimum valid value for the numeric data.
        ceiling : float, optional
            The maximum valid value for the numeric data.
        convert_to_int : bool, optional
            If True, forces conversion to an integer type.
        decimal_places : int, optional
            Number of decimal digits to retain during rounding.
        rounding_mode : RoundingMode, default RoundingMode.NEAREST
            The algorithm used for rounding operations.
        scale_factor : float, optional
            Multiplier applied to the data before processing.

        Returns
        -------
        ColumnHint
            A hint configured for numeric data processing.
        """
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
        """
        Hint to generate an aggregate feature (e.g., sum of multiple columns).

        Parameters
        ----------
        agg_columns : list of str
            The source columns for the operation.
        agg_op : str
            The operation to perform ('sum', 'mean', etc.).
        output_names : dict, optional
            Target name for the new feature.
        decimal_places : int, optional
            Rounding precision for the result.
        rounding_mode : RoundingMode, default RoundingMode.NEAREST
            Algorithm for rounding.
        scale_factor : float, optional
            Pre-processing multiplier.

        Returns
        -------
        ColumnHint
            A hint configured for feature aggregation.

        Notes
        -----
        Automatically removes duplicate source columns while preserving order.
        """
        seen = set()
        unique_columns = [
            col for col in agg_columns if not (col in seen or seen.add(col))
        ]

        if len(unique_columns) != len(agg_columns):
            import warnings

            warnings.warn(
                f"Duplicate columns removed from aggregate hint: "
                f"{[c for c in agg_columns if agg_columns.count(c) > 1]}",
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
        """
        Marks a column to be completely bypassed by the recommendation engine.

        Returns
        -------
        ColumnHint
            A hint that prevents any recommendations for the column.
        """
        return cls(is_ignored=True)

    @classmethod
    def drop(cls) -> "ColumnHint":
        """
        Forces the generation of a 'DropColumn' recommendation.

        Returns
        -------
        ColumnHint
            A hint that results in a recommendation to remove the column.
        """
        return cls(should_drop=True)

    @classmethod
    def geospatial(
        cls,
        latitude_bounds: tuple[float, float] | None = None,
        longitude_bounds: tuple[float, float] | None = None,
    ) -> "ColumnHint":
        """
        Hint for GPS/Geospatial coordinates.

        Parameters
        ----------
        latitude_bounds : tuple of (float, float), optional
            Valid latitude range (e.g., (-90, 90)).
        longitude_bounds : tuple of (float, float), optional
            Valid longitude range (e.g., (-180, 180)).

        Returns
        -------
        ColumnHint
            A hint configured for geospatial validation.
        """
        return cls(
            logical_type=ColumnHintType.GEOSPATIAL,
            lat_bounds=latitude_bounds,
            lon_bounds=longitude_bounds,
        )

    @classmethod
    def distance(
        cls,
        unit: str = "miles",
        floor: float | None = None,
        ceiling: float | None = None,
    ) -> "ColumnHint":
        """
        Hint for distance measurements (e.g., 'radius' or 'length').

        Parameters
        ----------
        unit : str, default "miles"
            The unit of measurement for the distance.
        floor : float, optional
            The minimum valid distance value.
        ceiling : float, optional
            The maximum valid distance value.

        Returns
        -------
        ColumnHint
            A hint configured for distance-based data.
        """
        return cls(
            logical_type=ColumnHintType.DISTANCE,
            unit=unit,
            floor=floor,
            ceiling=ceiling,
        )


class RecommendationManager:
    """
    Orchestrates a pipeline of data transformation recommendations.

    The Manager acts as a central repository and execution engine. It ensures
    that data preparation steps occur in a logically sound order (e.g.,
    imputing missing values before performing feature interaction) and
    provides global configuration for automated detection heuristics.

    Attributes
    ----------
    EXECUTION_PRIORITY : dict[RecommendationType, int]
        A mapping of RecommendationTypes to their relative execution order.
        Lower values indicate earlier execution (e.g., structural cleanup
        occurs before encoding).
    DEFAULT_CONFIG : dict[str, Any]
        Baseline thresholds for automated heuristics, including categorical
        cardinality and null-ratio limits for column drops.
    default_date_format : str
        The preferred format string used for datetime inference and
        conversion (defaults to 'ISO8601').
    """

    # Execution priority map: Defines the "Gravity" of the pipeline
    EXECUTION_PRIORITY: dict[RecommendationType, int] = {
        # 1: Structural cleanup
        RecommendationType.NON_INFORMATIVE: 1,
        # 2: Value-level cleaning
        RecommendationType.OUTLIER_HANDLING: 2,
        # 3: Foundational casting
        RecommendationType.DATETIME_CONVERSION: 3,
        RecommendationType.INT_CONVERSION: 3,
        RecommendationType.FLOAT_CONVERSION: 3,
        # 4: Data completion
        RecommendationType.MISSING_VALUES: 4,
        RecommendationType.VALUE_REPLACEMENT: 4,
        # 5: Optimization
        RecommendationType.CATEGORICAL_CONVERSION: 5,
        # 6: Engineering / Extraction
        RecommendationType.FEATURE_EXTRACTION: 6,
        RecommendationType.FEATURE_INTERACTION: 6,
        RecommendationType.FEATURE_AGGREGATION: 6,
        # 7: ML readiness & final refinements
        RecommendationType.ENCODING: 7,
        RecommendationType.DECIMAL_PRECISION_OPTIMIZATION: 7,
        RecommendationType.BOOLEAN_CLASSIFICATION: 7,
        RecommendationType.BINNING: 7,
        RecommendationType.OUTLIER_DETECTION: 7,
        RecommendationType.CLASS_IMBALANCE: 7,
    }

    # Baseline thresholds for automated heuristics
    DEFAULT_CONFIG: dict[str, Any] = {
        "categorical_threshold": 0.05,  # Unique values / Total rows
        "max_drop_threshold": 0.90,  # Null ratio to trigger a drop
        "outlier_threshold": 0.10,  # Max % of outliers to allow clipping
        "verbose": True,
    }

    def __init__(
        self,
        recommendations: list[Recommendation] | None = None,
        default_date_format: str = "ISO8601",
    ):
        """
        Initializes the RecommendationManager.

        Parameters
        ----------
        recommendations : list of Recommendation, optional
            A starting set of Recommendation objects to populate the pipeline.
            If None, initializes with an empty list.
        default_date_format : str, default "ISO8601"
            The format string or parsing strategy used by datetime workers
            when inferring or converting timestamp columns.
        """
        self._pipeline: list[Recommendation] = recommendations or []
        self._summary_warnings: list[str] = []
        self._column_hints: dict[str, ColumnHint] = {}
        self.default_date_format = default_date_format

    def add(self, recommendation: Recommendation | Iterable[Recommendation]) -> None:
        """
        Adds one or more recommendations to the end of the pipeline.

        Parameters
        ----------
        recommendation : Recommendation or iterable of Recommendation
            A single Recommendation instance or a collection of them to be
            appended to the existing pipeline.

        Raises
        ------
        TypeError
            If the provided input is not a Recommendation instance or a
            valid iterable of Recommendations.
        """
        if isinstance(recommendation, Recommendation):
            self._pipeline.append(recommendation)
        elif isinstance(recommendation, Iterable) and not isinstance(
            recommendation, (str, bytes)
        ):
            self._pipeline.extend(recommendation)
        else:
            raise TypeError(
                "Expected a Recommendation or an Iterable of Recommendations."
            )

    def add_after(self, target_id: str, new_rec: Recommendation) -> None:
        """
        Inserts a recommendation immediately following a specific recommendation ID.

        Parameters
        ----------
        target_id : str
            The unique ID of the existing recommendation to use as an anchor.
        new_rec : Recommendation
            The new recommendation instance to be inserted into the pipeline.

        Raises
        ------
        ValueError
            If the target_id does not exist within the current pipeline.
        """
        # Find index using a generator expression for efficiency
        try:
            target_index = next(
                i for i, rec in enumerate(self._pipeline) if rec.id == target_id
            )
        except StopIteration:
            raise ValueError(f"Target recommendation ID '{target_id}' not found.")

        self._pipeline.insert(target_index + 1, new_rec)

    def _get_sorted_pipeline(self) -> list[Recommendation]:
        """
        Retrieves the pipeline sorted by execution priority and column identity.

        The sorting logic follows a two-tier hierarchy:
        1. Priority: Dictated by `EXECUTION_PRIORITY` (lower values execute first).
        2. Alphabetical: Within a priority level, columns are sorted by name to
           ensure deterministic execution.

        Returns
        -------
        list of Recommendation
            A new list containing the Recommendations in their optimal
            execution order.

        Notes
        -----
        Unknown recommendation types are assigned a fallback priority of 999,
        placing them at the very end of the execution sequence.
        """

        def sort_key(rec: Recommendation) -> tuple[int, str]:
            # Priority first, then column_name for deterministic results
            priority = self.EXECUTION_PRIORITY.get(rec.rec_type, 999)
            return (priority, rec.column_name)

        return sorted(self._pipeline, key=sort_key)

    def apply(
        self,
        df: pd.DataFrame,
        allow_column_overwrite: bool = False,
        inplace: bool = False,
        drop_duplicates: bool = False,
    ) -> pd.DataFrame:
        """
        Executes the recommendation pipeline on the provided DataFrame.

        The application process follows a strict lifecycle:
        1. Validation: Verifies column existence and dependency safety.
        2. Preparation: Handles duplicates and memory allocation (copy vs. inplace).
        3. Execution: Applies recommendations in priority order with dtype safety.
        4. Cleanup: Removes redundant source columns after successful transformation.

        Parameters
        ----------
        df : pd.DataFrame
            The source DataFrame to be transformed.
        allow_column_overwrite : bool, default False
            If True, permits recommendations to update existing columns,
            provided the resulting data types remain compatible with the
            original structure.
        inplace : bool, default False
            If True, operates directly on the input DataFrame. If False,
            operates on a copy.
        drop_duplicates : bool, default False
            If True, removes identical rows prior to processing.

        Returns
        -------
        pd.DataFrame
            A transformed DataFrame containing all applied features and
            optimizations.

        Raises
        ------
        ValueError
            If the pipeline fails structural validation prior to execution.
        TypeError
            If an overwrite results in an incompatible data type.
        RuntimeError
            If a specific recommendation fails during the execution phase.
        """
        # 1. Structural Guard: Ensure the pipeline is logically sound
        self._validate_pipeline(df, allow_column_overwrite=allow_column_overwrite)

        # 2. Preparation
        result = df if inplace else df.copy()
        if drop_duplicates:
            result = result.drop_duplicates()

        # Track columns slated for removal (e.g., raw strings after conversion)
        garbage_collector: set[str] = set()

        # 3. Execution: Priority-ordered application
        for rec in self._get_sorted_pipeline():
            if not rec.enabled:
                continue

            try:
                # Capture state for overwrite validation
                out_col = getattr(rec, "output_column", None)
                pre_apply_dtype = (
                    result[out_col].dtype
                    if (out_col and out_col in result.columns)
                    else None
                )

                # Perform the transformation
                result = rec.apply(result)

                # Dtype Safety Check: Prevent 'Object' columns from becoming 'Int'
                # via overwrite unexpectedly
                if pre_apply_dtype is not None and allow_column_overwrite and out_col:
                    current_dtype = result[out_col].dtype
                    if current_dtype != pre_apply_dtype:
                        raise TypeError(
                            f"Incompatible Overwrite: {rec.id} changed '{out_col}' "
                            f"from {pre_apply_dtype} to {current_dtype}."
                        )

            except Exception as e:
                # Wrap internal errors with manager-level context
                raise RuntimeError(
                    f"Pipeline Failure: {rec.id} ({rec.rec_type.name}) failed on "
                    f"column '{rec.column_name}'. Error: {e}"
                ) from e

            # 4. Cleanup Registration: Flag source columns that are now redundant
            is_transformative = rec.rec_type in (
                RecommendationType.DATETIME_CONVERSION,
                RecommendationType.FEATURE_EXTRACTION,
                RecommendationType.ENCODING,
            )
            if is_transformative and rec.column_name in result.columns:
                garbage_collector.add(rec.column_name)

        # 5. Final Cleanup: Drop source columns if they weren't overwritten
        if garbage_collector:
            # We only drop columns that aren't also output columns
            # (to avoid dropping a column we just updated via overwrite)
            final_drops = [c for c in garbage_collector if c in result.columns]
            result.drop(columns=final_drops, inplace=True)

        return result

    def execution_summary(self) -> None:
        """
        Prints a structured roadmap of the recommendation pipeline.

        Recommendations are grouped by their logical execution priority to
        visualize the transformation lifecycle from data cleaning to feature
        engineering.
        """
        if not self._pipeline:
            print("\n[!] Pipeline is empty. No recommendations to display.")
            return

        # 1. Grouping Logic
        sorted_recs = self._get_sorted_pipeline()
        priority_groups: dict[int, list[Recommendation]] = {}

        for rec in sorted_recs:
            p = self.EXECUTION_PRIORITY.get(rec.rec_type, 999)
            priority_groups.setdefault(p, []).append(rec)

        # 2. Section Headers
        priority_labels = {
            1: "STAGE 1: Structural Cleanup (Dropping Columns)",
            2: "STAGE 2: Outlier Mitigation & Robustness",
            3: "STAGE 3: Type Casting & Schema Refinement",
            4: "STAGE 4: Imputation & Placeholder Handling",
            5: "STAGE 5: Categorical Optimizations",
            6: "STAGE 6: Feature Extraction & Engineering",
            7: "STAGE 7: Final Polishing & Memory Optimization",
        }

        # 3. Render Summary
        print("\n" + "═" * 80)
        print(f"{'DATA PREPARATION STRATEGY':^80}")
        print("═" * 80)

        for p_level in sorted(priority_groups.keys()):
            label = priority_labels.get(p_level, f"STAGE {p_level}: Additional Tasks")
            print(f"\n● {label}")
            print("─" * 80)

            for rec in priority_groups[p_level]:
                status = "[ACTIVE]" if rec.enabled else "[DISABLED]"
                print(f"  {status} {rec.column_name:<20} | {rec.description}")
                if rec.alias:
                    print(f"    └─ Alias: {rec.alias}")

        # 4. Cleanup & Resource Management
        cleanup_targets = {
            r.column_name
            for r in self._pipeline
            if r.rec_type
            in (
                RecommendationType.DATETIME_CONVERSION,
                RecommendationType.FEATURE_EXTRACTION,
                RecommendationType.ENCODING,
            )
        }

        if cleanup_targets:
            print("\n" + "─" * 80)
            print("RESOURCE MANAGEMENT: Redundant source columns to be purged:")
            print(f"» {', '.join(sorted(cleanup_targets))}")

        # 5. Alerts & Warnings
        if self._summary_warnings:
            print("\n" + "!" * 80)
            print(f"{'CONFIGURATION ALERTS':^80}")
            print("!" * 80)
            for warning in self._summary_warnings:
                print(f" • {warning}")

        print("\n" + "═" * 80)

    def _validate_pipeline(
        self, df: pd.DataFrame, allow_column_overwrite: bool = False
    ) -> None:
        """
        Validates the execution integrity and dependency graph of the current pipeline.

        Performs a dry-run of the pipeline logic to ensure that all source columns
        exist, no required data is dropped prematurely, and column overwrites do
        not create logical "stale data" conflicts for downstream transformations.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to validate against.
        allow_column_overwrite : bool, default False
            If False, raises an error if any recommendation attempts to write to
            an existing column name.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a required source column is missing.
        ValueError
            If a recommendation depends on a column dropped by a higher-priority task.
        ValueError
            If an unauthorized or unsafe column overwrite is detected.
        """
        enabled_recs = [r for r in self._pipeline if r.enabled]
        available_columns = set(df.columns)

        # 1. Verification of Initial Source Existence
        for rec in enabled_recs:
            if rec.column_name not in available_columns:
                raise ValueError(
                    f"Missing Source: Column '{rec.column_name}' referenced in {rec.id} "
                    f"does not exist in the DataFrame."
                )

        # 2. Sequential Dependency & Overwrite Analysis
        sorted_recs = self._get_sorted_pipeline()
        dropped_columns: set[str] = set()

        for i, rec in enumerate(sorted_recs):
            if not rec.enabled:
                continue

            # Resolve all input requirements for this recommendation
            dependencies = {rec.column_name}
            for attr in ("start_column", "end_column", "agg_columns"):
                val = getattr(rec, attr, None)
                if isinstance(val, str):
                    dependencies.add(val)
                elif isinstance(val, list):
                    dependencies.update(val)

            # Safety Check: Is the input actually available at this stage?
            conflict = dependencies.intersection(dropped_columns)
            if conflict:
                raise ValueError(
                    f"Logic Conflict: Recommendation {rec.id} depends on {conflict}, "
                    f"which was dropped in an earlier priority stage."
                )

            # 3. Overwrite Safety Analysis
            output_col = getattr(rec, "output_column", None)
            if output_col and output_col in available_columns:
                if not allow_column_overwrite:
                    raise ValueError(
                        f"Overwrite Conflict: {rec.id} attempts to update existing "
                        f"column '{output_col}'. Set allow_column_overwrite=True to permit."
                    )

                # Lookahead Check: Ensure no later step expects the 'original' version of this column
                for later_rec in sorted_recs[i + 1 :]:
                    if not later_rec.enabled:
                        continue

                    later_deps = {later_rec.column_name}
                    for attr in ("start_column", "end_column", "agg_columns"):
                        l_val = getattr(later_rec, attr, None)
                        if isinstance(l_val, str):
                            later_deps.add(l_val)
                        elif isinstance(l_val, list):
                            later_deps.update(l_val)

                    if output_col in later_deps:
                        raise ValueError(
                            f"Pipeline Break: Cannot overwrite '{output_col}' in {rec.id} "
                            f"because a later step ({later_rec.id}) requires the original "
                            f"data for that column."
                        )

            # 4. State Tracking: Update dropped columns for the next iteration
            is_dropped = rec.rec_type == RecommendationType.NON_INFORMATIVE or (
                rec.rec_type == RecommendationType.MISSING_VALUES
                and getattr(rec, "strategy", None) == MissingValueStrategy.DROP_COLUMN
            )
            if is_dropped:
                dropped_columns.add(rec.column_name)

    def _semantic_similarity(self, col1: str, col2: str) -> float:
        """
        Calculates a heuristic similarity score between two column names.

        This utility identifies potential temporal pairs (e.g., 'start_date' and
        'end_date') by evaluating linguistic "polarity" and shared naming
        tokens. It is primarily used by the duration discovery engine to
        assign confidence scores to identified intervals.

        Parameters
        ----------
        col1 : str
            The name of the first column to compare.
        col2 : str
            The name of the second column to compare.

        Returns
        -------
        float
            A similarity score ranging from 0.0 (no relation) to 1.0 (perfect match).
            - 0.9: Opposing temporal anchors (Start/End pairs).
            - 0.7: Shared temporal context (Both indicate time).
            - 0.1 - 0.6: Structural token overlap.
        """
        c1, c2 = col1.lower(), col2.lower()

        # Group indicators by their temporal 'polarity'
        start_terms = {
            "start",
            "open",
            "begin",
            "entry",
            "from",
            "created",
            "opened",
            "departure",
            "inception",
        }
        end_terms = {
            "end",
            "close",
            "finish",
            "exit",
            "to",
            "completed",
            "closed",
            "arrival",
            "termination",
        }

        # Determine if columns contain start or end indicators
        c1_start = any(t in c1 for t in start_terms)
        c1_end = any(t in c1 for t in end_terms)
        c2_start = any(t in c2 for t in start_terms)
        c2_end = any(t in c2 for t in end_terms)

        # 1. Perfect Pairing: Opposing polarities (e.g., 'start' and 'end')
        if (c1_start and c2_end) or (c1_end and c2_start):
            return 0.9

        # 2. Strong Temporal Context: Both are time-related but polarity is unclear
        if (c1_start or c1_end) and (c2_start or c2_end):
            return 0.7

        # 3. Structural Similarity: Shared word components
        # Normalize delimiters to spaces then split into tokens
        import re

        tokens1 = set(re.split(r"[_ \-]", c1))
        tokens2 = set(re.split(r"[_ \-]", c2))

        # Remove empty strings from potential trailing delimiters
        tokens1.discard("")
        tokens2.discard("")

        shared_tokens = tokens1.intersection(tokens2)
        if shared_tokens:
            # Calculate Jaccard-like overlap relative to the longest name
            max_token_count = max(len(tokens1), len(tokens2))
            token_score = (
                len(shared_tokens) / max_token_count if max_token_count > 0 else 0
            )

            # Cap structural similarity at 0.6 to ensure temporal
            # anchors always take precedence in discovery.
            return min(0.6, token_score)

        return 0.0

    def _check_positive_delta_ratio(
        self, df: pd.DataFrame, col_a: str, col_b: str, sample_size: int = 500
    ) -> float:
        """
        Heuristically determines the temporal "arrow" between two columns.

        Calculates the statistical ratio of rows where `col_b` occurs at or after
        `col_a`. A high ratio (typically > 0.95) provides strong evidence that
        `col_a` is the temporal origin and `col_b` is the terminus.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the temporal data.
        col_a : str
            The name of the potential "start" column.
        col_b : str
            The name of the potential "end" column.
        sample_size : int, default 500
            The maximum number of non-null rows to evaluate to ensure
            performance on large datasets.

        Returns
        -------
        float
            The ratio of positive time deltas ranging from 0.0 to 1.0.
            Returns 0.0 if comparison is impossible or data is empty.
        """
        try:
            # Efficiency: Sample only necessary columns and drop missing data
            sample = df[[col_a, col_b]].dropna()

            if sample.empty:
                return 0.0

            if len(sample) > sample_size:
                sample = sample.sample(n=sample_size, random_state=42)

            # Ensure we are working with datetime objects for subtraction
            s_a = pd.to_datetime(sample[col_a], errors="coerce")
            s_b = pd.to_datetime(sample[col_b], errors="coerce")

            # Re-drop any coercion failures (e.g. invalid date strings)
            valid_mask = s_a.notna() & s_b.notna()
            if not valid_mask.any():
                return 0.0

            # Vectorized subtraction: col_b - col_a >= 0
            is_positive = (s_b[valid_mask] - s_a[valid_mask]) >= pd.Timedelta(0)

            return float(is_positive.mean())

        except (ValueError, TypeError, OverflowError, AttributeError):
            # Fallback for incompatible types or data overflow
            return 0.0

    def _check_reasonable_duration_magnitude(
        self, df: pd.DataFrame, col_a: str, col_b: str, sample_size: int = 500
    ) -> bool:
        """
        Validates if the temporal interval between columns is analytically plausible.

        This heuristic filters out pairs that are mathematically forward-moving but
        semantically unrelated (e.g., 'DateOfBirth' vs 'TransactionDate') by
        verifying that the majority of durations fall within a 10-year window.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the potential datetime columns.
        col_a : str
            The name of the potential "start" column.
        col_b : str
            The name of the potential "end" column.
        sample_size : int, default 500
            The number of rows to sample to verify magnitude without
            performance degradation.

        Returns
        -------
        bool
            True if the majority of sampled forward-moving durations are
            under 10 years; False otherwise.
        """
        try:
            # Efficiency: Sample only necessary columns and drop missing data
            sample = df[[col_a, col_b]].dropna()

            if sample.empty:
                return False

            if len(sample) > sample_size:
                sample = sample.sample(n=sample_size, random_state=42)

            # Ensure we have proper datetime objects for subtraction
            s_a = pd.to_datetime(sample[col_a], errors="coerce")
            s_b = pd.to_datetime(sample[col_b], errors="coerce")

            # Align timezones if one is aware and the other is naive
            if s_a.dt.tz is not None and s_b.dt.tz is None:
                s_a = s_a.dt.tz_localize(None)
            elif s_b.dt.tz is not None and s_a.dt.tz is None:
                s_b = s_b.dt.tz_localize(None)

            deltas = s_b - s_a

            # We only care about the magnitude of forward-moving events
            forward_deltas = deltas[deltas >= pd.Timedelta(0)]
            if forward_deltas.empty:
                return False

            # 10-year window: 365 days * 10 + 2 leap days
            max_duration = pd.Timedelta(days=3652)

            # Check if the majority (>50%) of data falls in this window
            is_within_limit = forward_deltas <= max_duration

            return bool(is_within_limit.mean() > 0.5)

        except (ValueError, TypeError, OverflowError, AttributeError):
            # If the calculation is impossible, we assume no reasonable relationship
            return False

    def _generate_duration_column_name(self, col_a: str, col_b: str) -> str:
        """
        Derives a semantic and sanitized name for a new duration feature.

        Uses a tiered naming strategy:
        1. Shared Context: Extracts common prefixes/tokens between columns.
        2. Anchor Stripping: Removes temporal keywords to find the subject.
        3. Concatenation: Joins cleaned names as a descriptive fallback.

        Parameters
        ----------
        col_a : str
            The name of the start column.
        col_b : str
            The name of the end column.

        Returns
        -------
        str
            A lower-case, snake_case string representing the duration column.

        Examples
        --------
        >>> manager._generate_duration_column_name('session_start', 'session_end')
        'session_duration'
        >>> manager._generate_duration_column_name('created_at', 'resolved_at')
        'created_at_duration'
        """
        a_low, b_low = col_a.lower(), col_b.lower()

        # 1. Shared Sequence Extraction
        # Split by underscore or space to find linguistic components
        import re

        parts_a = re.split(r"[_ ]", a_low)
        parts_b = re.split(r"[_ ]", b_low)

        # Look for tokens that appear in both, maintaining relative order
        shared_tokens = [p for p in parts_a if p in parts_b and p.strip()]

        if shared_tokens:
            # Use up to the first two shared tokens to keep names concise
            base = "_".join(shared_tokens[:2])
            return f"{base}_duration"

        # 2. Temporal Anchor Stripping
        # If no shared tokens, try to strip the "time" indicator from the first match
        anchors = {
            "start",
            "begin",
            "open",
            "from",
            "end",
            "finish",
            "close",
            "to",
            "at",
        }
        for col in [a_low, b_low]:
            # Use regex to replace anchors as whole words or delimited parts
            # This avoids mangling words like 'startle' or 'attend'
            for anchor in anchors:
                pattern = rf"\b{anchor}\b|(?<=[_ ]){anchor}(?=[_ ])"
                base = re.sub(pattern, "", col).strip("_ ").replace(" ", "_")
                if base:
                    return f"{base}_duration"

        # 3. Fallback: Structural Concatenation
        # Final safety net: join both names cleanly
        name_a = a_low.replace(" ", "_").strip("_")
        name_b = b_low.replace(" ", "_").strip("_")
        return f"{name_a}_{name_b}_duration"

    def _identify_datetime_columns(
        self, df: pd.DataFrame, existing_datetime_cols: set[str] | None = None
    ) -> set[str]:
        """
        Aggregates all current and pending datetime columns within the pipeline.

        This utility provides a unified view of temporal data by combining columns
        that are natively datetime-typed with those slated for conversion via
        pending 'DATETIME_CONVERSION' recommendations. This look-ahead capability
        is essential for cross-column discovery (e.g., duration calculation)
        on raw string data.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame being analyzed.
        existing_datetime_cols : set of str, optional
            A pre-calculated set of native datetime columns to avoid
            redundant type-checking.

        Returns
        -------
        set of str
            A set containing the names of all columns that are, or will
            become, datetime objects.
        """
        # 1. Gather native datetime columns
        if existing_datetime_cols is not None:
            datetime_cols = set(existing_datetime_cols)
        else:
            # Fallback to vectorized check if no cache is provided
            datetime_cols = {
                col
                for col in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[col])
            }

        # 2. Gather "Future" datetimes from the pipeline
        # We only consider enabled recommendations to ensure the
        # look-ahead represents the actual intended output.
        future_datetimes = {
            rec.column_name
            for rec in self._pipeline
            if (rec.rec_type == RecommendationType.DATETIME_CONVERSION and rec.enabled)
        }

        # Combine both sets
        datetime_cols.update(future_datetimes)

        return datetime_cols

    def _decide_int_depth(self, series: pd.Series) -> BitDepth:
        """
        Determines the optimal integer bit-depth based on the observed data range.

        Evaluates the minimum and maximum values of a series to suggest the most
        memory-efficient integer type. Downcasting from 64-bit to 32-bit integers
        can reduce memory consumption by 50% without data loss.

        Parameters
        ----------
        series : pd.Series
            The numeric pandas Series to evaluate for bit-depth optimization.

        Returns
        -------
        BitDepth
            Returns BitDepth.INT32 if the data range fits within a conservative
            ±2 billion threshold; otherwise returns BitDepth.INT64.

        Notes
        -----
        The threshold is set at 2,000,000,000 (2.0e9) to provide a safety buffer
        below the theoretical 32-bit signed integer limit of 2,147,483,647.
        """
        try:
            # Drop nulls to evaluate actual data range
            s_min = series.min()
            s_max = series.max()

            # Guard against empty or all-null series
            if pd.isna(s_min) or pd.isna(s_max):
                return BitDepth.INT64

            # Conservative threshold to prevent overflow in future data appends
            limit = 2_000_000_000

            if s_min > -limit and s_max < limit:
                return BitDepth.INT32

        except (TypeError, ValueError):
            # Fallback to standard 64-bit if data is non-numeric or incompatible
            return BitDepth.INT64

        return BitDepth.INT64

    def _apply_column_hint(
        self, col_name: str, series: pd.Series, hint: ColumnHint
    ) -> None:
        """
        Translates a user-provided ColumnHint into specific pipeline recommendations.

        Parameters
        ----------
        col_name : str
            The name of the column to which the hint applies.
        series : pd.Series
            The data series associated with the column for validation.
        hint : ColumnHint
            The user-defined hint object containing logical types and strategies.
        """
        user_note = " [User Hint Applied]"

        # 1. Immediate Exit: User wants to bypass analysis
        if hint.is_ignored:
            return

        # 2. Forced Drop
        if hint.should_drop:
            rec_drop = NonInformativeRecommendation(
                column_name=col_name,
                description=f"Drop column '{col_name}' [User Hint: Forced Drop]",
                reason="Forced drop via user hint",
            )
            rec_drop.is_locked = True
            self._pipeline.append(rec_drop)
            return

        # 3. Datetime Logic
        if hint.logical_type == ColumnHintType.DATETIME:
            if not pd.api.types.is_datetime64_any_dtype(series):
                fmt_desc = (
                    f" using format {hint.datetime_format}"
                    if hint.datetime_format
                    else ""
                )
                rec_dt = DatetimeConversionRecommendation(
                    column_name=col_name,
                    description=f"Convert '{col_name}' to datetime{fmt_desc}{user_note}",
                    detected_format=hint.datetime_format,
                )
                rec_dt.is_locked = True
                self._pipeline.append(rec_dt)

            props = DatetimeProperty(0)
            if hint.datetime_features:
                for p in hint.datetime_features:
                    props |= p

            extraction_rec = FeatureExtractionRecommendation(
                column_name=col_name,
                description=f"Extract datetime features from '{col_name}'{user_note}",
                properties=props,
                output_columns=hint.output_names,
            )
            extraction_rec.is_locked = True
            self._pipeline.append(extraction_rec)
            return

        # 4. Distance Logic
        elif hint.logical_type == ColumnHintType.DISTANCE:
            lower_bound = float(hint.floor) if hint.floor is not None else 0.0
            upper_bound = float(hint.ceiling) if hint.ceiling is not None else 500.0

            rec_distance = OutlierHandlingRecommendation(
                column_name=col_name,
                description=f"Distance column '{col_name}' ({hint.unit}): nullify values outside [{lower_bound}, {upper_bound}] to catch sensor errors{user_note}",
                strategy=OutlierHandlingStrategy.NULLIFY,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            rec_distance.is_locked = True
            self._pipeline.append(rec_distance)
            return

        # 5. Geospatial Logic
        elif hint.logical_type == ColumnHintType.GEOSPATIAL:
            if hint.lat_bounds is not None:
                lat_lower, lat_upper = hint.lat_bounds
                rec_lat = OutlierHandlingRecommendation(
                    column_name=col_name,
                    description=f"Latitude column '{col_name}': nullify values outside bounding box [{lat_lower}, {lat_upper}]{user_note}",
                    strategy=OutlierHandlingStrategy.NULLIFY,
                    lower_bound=float(lat_lower),
                    upper_bound=float(lat_upper),
                )
                rec_lat.is_locked = True
                self._pipeline.append(rec_lat)

            if hint.lon_bounds is not None:
                lon_lower, lon_upper = hint.lon_bounds
                rec_lon = OutlierHandlingRecommendation(
                    column_name=col_name,
                    description=f"Longitude column '{col_name}': nullify values outside bounding box [{lon_lower}, {lon_upper}]{user_note}",
                    strategy=OutlierHandlingStrategy.NULLIFY,
                    lower_bound=float(lon_lower),
                    upper_bound=float(lon_upper),
                )
                rec_lon.is_locked = True
                self._pipeline.append(rec_lon)
            return

        # 6. Aggregation Logic
        elif (
            hint.logical_type == ColumnHintType.AGGREGATE
            and hint.agg_columns
            and hint.agg_op
        ):
            out_name = None
            if hint.output_names:
                out_name = hint.output_names.get("aggregate") or hint.output_names.get(
                    "output"
                )
            if not out_name:
                out_name = f"{col_name}_{hint.agg_op}"

            rec_agg = AggregationRecommendation(
                column_name=(hint.agg_columns[0] if hint.agg_columns else col_name),
                description=f"Aggregate columns {hint.agg_columns} with '{hint.agg_op}' into '{out_name}'{user_note}",
                agg_columns=hint.agg_columns,
                agg_op=hint.agg_op,
                output_column=out_name,
                decimal_places=hint.decimal_places,
                rounding_mode=(
                    hint.rounding_mode if hint.rounding_mode else RoundingMode.NEAREST
                ),
                scale_factor=hint.scale_factor,
            )
            rec_agg.is_locked = True
            self._pipeline.append(rec_agg)
            return

        # 7. Numeric & Financial Logic
        elif hint.logical_type in (ColumnHintType.NUMERIC, ColumnHintType.FINANCIAL):
            if hint.floor is not None or hint.ceiling is not None:
                lower = (
                    float(hint.floor) if hint.floor is not None else float(series.min())
                )
                upper = (
                    float(hint.ceiling)
                    if hint.ceiling is not None
                    else float(series.max())
                )

                rec_clip = OutlierHandlingRecommendation(
                    column_name=col_name,
                    description=f"Clip '{col_name}' to bounds [{lower}, {upper}]{user_note}",
                    strategy=OutlierHandlingStrategy.CLIP,
                    lower_bound=lower,
                    upper_bound=upper,
                )
                rec_clip.is_locked = True
                self._pipeline.append(rec_clip)

            non_null = series.dropna()
            if (
                hint.logical_type == ColumnHintType.NUMERIC
                and hint.convert_to_int is True
            ):
                integer_count = (
                    int(((non_null % 1) == 0).sum())
                    if pd.api.types.is_numeric_dtype(series)
                    else 0
                )
                rec_int = IntegerConversionRecommendation(
                    column_name=col_name,
                    description=f"Convert '{col_name}' to int64{user_note}",
                    integer_count=integer_count,
                )
                rec_int.is_locked = True
                self._pipeline.append(rec_int)
            elif hint.decimal_places is not None:
                min_val = float(non_null.min()) if len(non_null) > 0 else float("nan")
                max_val = float(non_null.max()) if len(non_null) > 0 else float("nan")
                rec_dec = DecimalPrecisionRecommendation(
                    column_name=col_name,
                    description=f"Optimize decimal precision of '{col_name}' to {hint.decimal_places} places{user_note}",
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
            return

        # 8. Categorical Logic
        elif hint.logical_type == ColumnHintType.CATEGORICAL:
            rec_cat = CategoricalConversionRecommendation(
                column_name=col_name,
                description=f"Convert '{col_name}' to categorical{user_note}",
                unique_values=int(series.nunique()),
            )
            rec_cat.is_locked = True
            self._pipeline.append(rec_cat)
            return

    def _is_string_datetime(self, sample: pd.Series) -> tuple[bool, str | None]:
        """
        Heuristically determines if a string sample represents datetime data.

        Parameters
        ----------
        sample : pd.Series
            A small sample of non-null string values from a column.

        Returns
        -------
        is_datetime : bool
            True if the sample can be reliably parsed as datetime.
        inferred_format : str or None
            The format string if one could be inferred, otherwise None.
        """
        if sample.empty:
            return False, None

        try:
            # We use errors='coerce' so that it returns NaT for unparseable strings
            # and doesn't raise a hard exception.
            converted = pd.to_datetime(
                sample,
                errors="coerce",
                format=(
                    "mixed"
                    if self.default_date_format == "ISO8601"
                    else self.default_date_format
                ),
            )

            # If at least 80% of the sample was successfully converted,
            # we consider it a datetime column.
            success_ratio = converted.notna().mean()

            if success_ratio > 0.8:
                # We attempt to guess the format from the first valid value
                # for the recommendation's description.
                # In modern Pandas (2.0+), we can use pd.to_datetime with
                # dayfirst/yearfirst logic if needed.
                return True, "ISO8601 or similar"

            return False, None

        except (ValueError, TypeError, OverflowError):
            return False, None

    def _analyze_string_heuristics(
        self,
        col_name: str,
        series: pd.Series,
        unique_count: int,
        total_rows: int,
        config: dict,
    ) -> None:
        """
        Heuristic brain for object/string columns to find dates or categories.

        Analyzes string-based columns to determine if they represent temporal data
        or high-redundancy categories. If a date pattern is detected, it suggests
        both a conversion and a standard feature extraction. Otherwise, it
        evaluates the cardinality ratio to suggest categorical conversion.

        Parameters
        ----------
        col_name : str
            The name of the column currently being analyzed.
        series : pd.Series
            The actual data values used for pattern sampling and inference.
        unique_count : int
            The number of distinct non-null values in the column.
        total_rows : int
            The total number of rows in the dataset, used for cardinality ratios.
        config : dict
            Configuration thresholds, specifically 'categorical_threshold'.

        Returns
        -------
        None
            Appends identified recommendations directly to the internal pipeline.
        """
        # A. Automated Datetime Inference
        # Sample the column to check for date patterns
        sample = series.dropna().head(100).astype(str)
        is_dt, fmt = self._is_string_datetime(sample)

        if is_dt:
            rec_dt = DatetimeConversionRecommendation(
                column_name=col_name,
                description=f"Convert '{col_name}' to datetime (Detected format: {fmt})",
                detected_format=fmt,
            )
            self._pipeline.append(rec_dt)

            # Auto-suggest extraction for new dates (Year, Month, Day by default)
            rec_ext = FeatureExtractionRecommendation(
                column_name=col_name,
                description=f"Extract standard temporal features from '{col_name}'",
                properties=(
                    DatetimeProperty.YEAR
                    | DatetimeProperty.MONTH
                    | DatetimeProperty.DAY
                ),
            )
            self._pipeline.append(rec_ext)
            return  # Datetimes are prioritized over categorical conversion

        # B. Categorical Conversion Heuristic
        # If low cardinality relative to total rows, suggest 'category' dtype
        cardinality_ratio = unique_count / total_rows if total_rows > 0 else 1.0
        if unique_count > 1 and cardinality_ratio < config.get(
            "categorical_threshold", 0.05
        ):
            rec_cat = CategoricalConversionRecommendation(
                column_name=col_name,
                description=f"Convert high-redundancy column '{col_name}' to categorical",
                unique_values=unique_count,
            )
            self._pipeline.append(rec_cat)

    def _analyze_missing_values(
        self,
        col_name: str,
        series: pd.Series,
        null_count: int,
        null_ratio: float,
        config: dict,
    ) -> None:
        """
        Analyzes missing data and recommends an appropriate imputation or drop strategy.

        Parameters
        ----------
        col_name : str
            The name of the column.
        series : pd.Series
            The data series to analyze.
        null_count : int
            Number of missing values.
        null_ratio : float
            Ratio of missing values to total rows.
        config : dict
            Configuration thresholds, specifically 'impute_threshold'
            and 'max_drop_threshold'.
        """
        # Strategy A: Column is too sparse to be useful
        if null_ratio > config.get("max_drop_threshold", 0.9):
            rec_drop = NonInformativeRecommendation(
                column_name=col_name,
                description=f"Drop '{col_name}': {null_ratio:.1%} missing values exceeds threshold.",
                reason="High null density",
            )
            self._pipeline.append(rec_drop)
            return

        # Strategy B: Recommend Imputation
        if null_ratio > 0:
            # Heuristic: Use Median for numeric with potential outliers,
            # Mean for normal-ish distributions, or Mode for strings.
            if pd.api.types.is_numeric_dtype(series):
                strategy = MissingValueStrategy.IMPUTE_MEDIAN
            else:
                strategy = MissingValueStrategy.IMPUTE_MODE

            rec_mv = MissingValuesRecommendation(
                column_name=col_name,
                description=f"Handle {null_count} missing values in '{col_name}' using {strategy.name} strategy.",
                strategy=strategy,
                missing_count=null_count,
                missing_percentage=null_ratio,
            )
            self._pipeline.append(rec_mv)

    def _analyze_numeric_heuristics(
        self,
        col_name: str,
        series: pd.Series,
        unique_count: int,
        total_rows: int,
        config: dict,
    ) -> None:
        """
        Automated discovery of boolean flags and integer bit-depth optimizations.

        Evaluates numeric columns to identify binary states that should be
        represented as booleans. Additionally, it checks if floating-point columns
        consist entirely of whole numbers and suggests the most efficient integer
        bit-depth for conversion.

        Parameters
        ----------
        col_name : str
            The name of the column currently being analyzed.
        series : pd.Series
            The numeric data series used for distribution and type analysis.
        unique_count : int
            The number of distinct non-null values in the column.
        total_rows : int
            The total number of rows in the dataset.
        config : dict
            Configuration thresholds for automated detection.

        Returns
        -------
        None
            Appends identified recommendations directly to the internal pipeline.
        """
        non_null = series.dropna()
        if non_null.empty:
            return

        # A. Boolean Detection
        # Identify binary columns (exactly 2 unique values)
        if unique_count == 2:
            unique_vals = list(non_null.unique())

            # Numeric-specific logic for 0/1
            if pd.api.types.is_numeric_dtype(series):
                if set(unique_vals) == {0, 1}:
                    rec_bool = BooleanClassificationRecommendation(
                        column_name=col_name,
                        description=f"Convert binary column '{col_name}' (0, 1) to boolean.",
                        values=unique_vals,
                    )
                    self._pipeline.append(rec_bool)
                    return

        # B. Integer Bit-Depth Optimization
        is_float = pd.api.types.is_float_dtype(series)
        if is_float:
            # Check if all values are mathematically integers
            integer_count = int(((non_null % 1) == 0).sum())
            if integer_count == len(non_null) and len(non_null) > 0:
                depth = self._decide_int_depth(series)
                rec_int = IntegerConversionRecommendation(
                    column_name=col_name,
                    description=f"Cast float '{col_name}' to {depth.name} (all values are integers).",
                    integer_count=integer_count,
                )
                self._pipeline.append(rec_int)

    def _analyze_outliers_and_distribution(
        self, col_name: str, series: pd.Series, config: dict
    ) -> None:
        """
        Detects statistical outliers using the Interquartile Range (IQR) method.

        Analyzes numeric columns for extreme values that fall outside the standard
        1.5x IQR bounds. If outliers are detected within a manageable range
        (between 0% and the configured threshold), it suggests a clipping strategy
        to mitigate their impact on statistical models.

        Parameters
        ----------
        col_name : str
            The name of the column currently being analyzed.
        series : pd.Series
            The numeric data series to check for statistical anomalies.
        config : dict
            Configuration settings, specifically 'outlier_threshold' to control
            when an automated recommendation is generated.

        Returns
        -------
        None
            Appends an OutlierHandlingRecommendation to the pipeline if detection
            criteria are met.

        Notes
        -----
        This method includes a significance guard that skips analysis if the
        number of non-null values is less than 20.
        """
        non_null = series.dropna()
        if len(non_null) < 20:  # Statistical significance guard
            return

        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
        outlier_ratio = len(outliers) / len(non_null)

        # If outliers exist but aren't overwhelming (>0% but <10%)
        if 0 < outlier_ratio < config.get("outlier_threshold", 0.1):
            rec_out = OutlierHandlingRecommendation(
                column_name=col_name,
                description=f"Handle outliers in '{col_name}' ({outlier_ratio:.1%}) using IQR clipping.",
                strategy=OutlierHandlingStrategy.CLIP,
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
            )
            self._pipeline.append(rec_out)

    def _analyze_column_heuristics(
        self,
        col_name: str,
        series: pd.Series,
        unique_count: int,
        total_rows: int,
        config: dict,
    ) -> None:
        """
        Executes automated heuristic analysis for a single column.

        This worker method identifies data quality issues (missing values) and
        logical data types (datetimes, booleans, integers, categories) through
        statistical analysis and pattern matching.

        Parameters
        ----------
        col_name : str
            The name of the column to analyze.
        series : pd.Series
            The data series to evaluate.
        unique_count : int
            The number of unique values in the series (pre-calculated).
        total_rows : int
            Total row count of the DataFrame for ratio calculations.
        config : dict
            Configuration dictionary containing thresholds (e.g.,
            'categorical_threshold', 'impute_threshold').

        Returns
        -------
        None
        """
        non_null_count = int(series.count())
        null_count = total_rows - non_null_count
        null_ratio = null_count / total_rows if total_rows > 0 else 0

        # 1. Automated Missing Value Detection
        if null_count > 0:
            self._analyze_missing_values(
                col_name, series, null_count, null_ratio, config
            )

        # 2. Type-Specific Logic Dispatcher
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric:
            self._analyze_numeric_heuristics(
                col_name, series, unique_count, total_rows, config
            )
        else:
            # Analyze strings/objects for Datetime potential or Categorical conversion
            self._analyze_string_heuristics(
                col_name, series, unique_count, total_rows, config
            )

        # 3. Outlier and Distribution Analysis
        if is_numeric and not pd.api.types.is_bool_dtype(series):
            # Only run for continuous-like numeric data
            self._analyze_outliers_and_distribution(col_name, series, config)

    def _discover_cross_column_features(
        self, df: pd.DataFrame, datetime_cols: set[str]
    ) -> None:
        """
        Scans pairs of datetime columns to identify and suggest interval features.

        Uses linguistic similarity and statistical "arrow of time" checks to
        discover logical durations (e.g., 'start_date' to 'end_date').

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame being analyzed.
        datetime_cols : set of str
            The set of columns identified as current or future datetimes.

        Returns
        -------
        None
        """
        # We need at least two columns to find an interval
        cols = sorted(list(datetime_cols))
        if len(cols) < 2:
            return

        import itertools

        # Evaluate all unique pairs (A, B)
        for col_a, col_b in itertools.permutations(cols, 2):
            # 1. Linguistic Check: Do the names sound like a pair?
            # (e.g., 'start' vs 'end')
            sim_score = self._semantic_similarity(col_a, col_b)
            if sim_score < 0.4:
                continue

            # 2. Statistical Check: Does B usually happen after A?
            # We use a sample to keep this fast.
            arrow_ratio = self._check_positive_delta_ratio(df, col_a, col_b)
            if arrow_ratio < 0.9:
                continue

            # 3. Magnitude Check: Is the duration "human-scale" (< 10 years)?
            # Prevents pairing unrelated dates like BirthDate and OrderDate.
            if not self._check_reasonable_duration_magnitude(df, col_a, col_b):
                continue

            # 4. Generate Recommendation
            # Derive a clean name (e.g., 'order_processing_duration')
            out_name = self._generate_duration_column_name(col_a, col_b)

            # Check if we already have a recommendation for this pair to avoid duplicates
            exists = any(
                isinstance(r, DatetimeDurationRecommendation)
                and r.start_column == col_a
                and r.end_column == col_b
                for r in self._pipeline
            )

            if not exists:
                rec_dur = DatetimeDurationRecommendation(
                    # Use col_a as the primary anchor for the recommendation
                    column_name=col_a,
                    description=(
                        f"Calculate duration between '{col_a}' and '{col_b}' "
                        f"as new feature '{out_name}'."
                    ),
                    start_column=col_a,
                    end_column=col_b,
                    output_column=out_name,
                    unit="days",
                )
                self._pipeline.append(rec_dur)

    def _refine_pipeline_strategies(self) -> None:
        """
        Performs a final refinement pass on the generated recommendations.

        This method handles three critical post-generation tasks:
        1. Conflict Resolution: Ensures no two recommendations attempt to create
           the same output column.
        2. Priority Alignment: Sorts the entire pipeline based on the
           EXECUTION_PRIORITY mapping.
        3. Metadata Enrichment: Updates descriptions or adds warnings for
           inter-dependent recommendations.

        Returns
        -------
        None
        """
        if not self._pipeline:
            return

        # 1. Conflict Resolution: Check for duplicate output column names
        # This prevents two different heuristics from naming a feature
        # 'user_id_duration', which would cause a collision during apply().
        seen_outputs: dict[str, Recommendation] = {}
        duplicates_to_remove = []

        for rec in self._pipeline:
            output_col = getattr(rec, "output_column", None)
            if output_col:
                if output_col in seen_outputs:
                    # Logic: Locked (User-hinted) recs always win over automated ones.
                    existing_rec = seen_outputs[output_col]
                    if rec.is_locked and not existing_rec.is_locked:
                        duplicates_to_remove.append(existing_rec)
                        seen_outputs[output_col] = rec
                    else:
                        duplicates_to_remove.append(rec)
                else:
                    seen_outputs[output_col] = rec

        # Remove identifies duplicates
        if duplicates_to_remove:
            self._pipeline = [
                r for r in self._pipeline if r not in duplicates_to_remove
            ]

        # 2. Priority Alignment
        # We sort by the predefined EXECUTION_PRIORITY to ensure that 'Drops'
        # happen before 'Conversions', which happen before 'Imputations'.
        self._pipeline.sort(key=lambda r: self.EXECUTION_PRIORITY.get(r.rec_type, 999))

        # 3. Global Integrity Check
        # If a column is slated to be dropped, ensure no 'Conversion' or
        # 'Optimization' recommendations remain for it (Cleanup).
        dropped_cols = {
            r.column_name
            for r in self._pipeline
            if r.rec_type == RecommendationType.NON_INFORMATIVE
        }

        if dropped_cols:
            self._pipeline = [
                r
                for r in self._pipeline
                if not (
                    r.column_name in dropped_cols
                    and r.rec_type != RecommendationType.NON_INFORMATIVE
                )
            ]

        # 4. Final Description Polishing
        # Optional: Add " (Chained)" suffix to recommendations that depend on
        # the output of a previous recommendation.

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
        **kwargs: Any,
    ) -> None:
        """
        Orchestrates the analysis pipeline to generate data improvement suggestions.

        This method executes a three-stage process: individual column heuristics
        (or hint applications), cross-column relationship discovery (e.g., date
        durations), and pipeline refinement to ensure logical consistency.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataset to analyze.
        target_column : str, optional
            The name of the label or target variable to protect from certain
            transformations.
        hints : dict of str to ColumnHint, optional
            User-provided metadata to override automated inference for specific
            columns.
        max_decimal_places : int or dict of str to int, optional
            Constraint on floating-point precision, globally or per column.
        default_max_decimal_places : int, optional
            The fallback rounding precision if not specified elsewhere.
        min_binning_unique_values : int or dict of str to int, optional
            Lower threshold of cardinality required to trigger binning.
        default_min_binning_unique_values : int, default 10
            The default minimum cardinality for binning.
        max_binning_unique_values : int or dict of str to int, optional
            Upper threshold of cardinality to prevent excessive feature expansion.
        default_max_binning_unique_values : int, default 1000
            The default maximum cardinality for binning.
        allow_categorical_encoding : bool, default True
            Whether to suggest One-Hot or Label encoding for categories.
        hints_only : bool, default False
            If True, skips all automated heuristics and only processes provided hints.
        overwrite : bool, default True
            If True, clears the existing pipeline before starting the analysis.
        **kwargs : Any
            Additional parameters passed to heuristic workers (e.g., custom thresholds).

        Returns
        -------
        None
            Populates the internal `_pipeline` attribute with Recommendation objects.
        """
        # 1. Initialize/Reset State
        if overwrite:
            self._pipeline = []
            self._summary_warnings = []

        # Sync passed hints to instance state for worker access
        self._column_hints = hints or {}

        # Package thresholds into a single config object for workers
        config = {
            **self.DEFAULT_CONFIG,
            "target_column": target_column,
            "max_decimal_places": max_decimal_places,
            "default_max_decimal_places": default_max_decimal_places,
            "min_binning_unique_values": min_binning_unique_values,
            "default_min_binning_unique_values": default_min_binning_unique_values,
            "max_binning_unique_values": max_binning_unique_values,
            "default_max_binning_unique_values": default_max_binning_unique_values,
            "allow_categorical_encoding": allow_categorical_encoding,
            **kwargs,
        }

        total_rows = len(df)

        # --- STAGE 1: Individual Column Analysis ---
        for col in df.columns:
            series = df[col]
            unique_count = int(series.nunique())

            # Priority A: Check for explicit User Instructions (Hints)
            hint = self._column_hints.get(col)
            if hint:
                self._apply_column_hint(col, series, hint)
                continue

            # Priority B: Automated Analysis (Skip if hints_only is True)
            if not hints_only:
                self._analyze_column_heuristics(
                    col, series, unique_count, total_rows, config
                )

        # --- STAGE 2: Cross-Column Relationship Discovery ---
        if not hints_only:
            datetime_cols = self._identify_datetime_columns(df)
            self._discover_cross_column_features(df, datetime_cols)

        # --- STAGE 3: Strategy Refinement ---
        self._refine_pipeline_strategies()

        try:
            self._validate_pipeline(df)
        except ValueError as e:
            self._summary_warnings.append(f"Pipeline Integrity Warning: {str(e)}")

    def clear(self) -> None:
        """
        Removes all recommendations, resetting the pipeline to an empty state.

        This is useful when you want to re-run an analysis on the same
        DataFrame with entirely new parameters without instantiating a
        new Manager.
        """
        self._pipeline.clear()

    def _remove_conflicting_non_informative(self, column_name: str) -> None:
        """
        Prunes 'NON_INFORMATIVE' recommendations for a specific column.

        This utility resolves conflicts when the engine identifies a useful
        transformation (such as datetime feature extraction) for a column
        previously flagged for removal. It enforces a "value creation"
        precedence, ensuring that potential features are not lost to
        earlier "drop" heuristics.

        Parameters
        ----------
        column_name : str
            The name of the column to reconcile.

        Returns
        -------
        None

        Notes
        -----
        This method rebuilds the internal pipeline in-place, filtering out
        any recommendations of type `RecommendationType.NON_INFORMATIVE`
        that target the specified column.
        """
        # Rebuild the pipeline using a list comprehension to filter out
        # the specific 'drop' recommendation for this column.
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
        Checks if any recommendations are currently targeted at a specific column.

        Parameters
        ----------
        column_name : str
            The name of the column to look up.

        Returns:
        -------
        True if one or more recommendations exist for the column.
        """
        return any(rec.column_name == column_name for rec in self._pipeline)

    def __len__(self) -> int:
        """Returns the total count of recommendations in the pipeline."""
        return len(self._pipeline)

    def __iter__(self):
        """Allows for direct iteration over the pipeline (e.g., in a for-loop)."""
        return iter(self._pipeline)

    def __getitem__(self, index: int) -> Recommendation:
        """
        Retrieves a recommendation by its positional index in the pipeline.

        Parameters
        ----------
        index : int
            The integer index of the recommendation.

        Returns:
        -------
        The Recommendation object at that position.
        """
        return self._pipeline[index]

    def get_by_id(self, rec_id: str) -> Recommendation | None:
        """
        Retrieves a recommendation from the pipeline by its unique ID.

        Parameters
        ----------
        rec_id : str
            The unique identifier to search for.

        Returns:
        -------
        The matching Recommendation object, or None if not found.
        """
        return next((r for r in self._pipeline if r.id == rec_id), None)

    def get_by_alias(self, alias: str) -> Recommendation | None:
        """
        Retrieves a recommendation using its user-defined alias.

        Parameters
        ----------
        alias : str
            The string alias to search for.

        Returns:
        -------
        The matching Recommendation object, or None if not found.
        """
        if alias is None:
            return None
        return next((r for r in self._pipeline if r.alias == alias), None)

    def enable_by_id(self, rec_id: str, ok_if_none: bool = False) -> None:
        """
        Activates a recommendation to ensure its execution during apply().

        Sets the `enabled` attribute of a specific recommendation to True.
        Once enabled, the recommendation will be processed according to its
        assigned priority level when the pipeline is executed.

        Parameters
        ----------
        rec_id : str
            The unique identifier of the recommendation to activate.
        ok_if_none : bool, default False
            If True, the method returns silently if the ID is not found.
            If False, a ValueError is raised.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `rec_id` is not found in the pipeline and `ok_if_none` is False.
        """
        rec = self.get_by_id(rec_id)
        if rec:
            rec.enabled = True
        elif not ok_if_none:
            raise ValueError(f"Enable failed: Recommendation ID '{rec_id}' not found.")

    def disable_by_id(self, rec_id: str, ok_if_none: bool = False) -> None:
        """
        Deactivates a recommendation, skipping it during apply().

        Parameters
        ----------
        rec_id : str
            The unique identifier of the recommendation to activate.
        ok_if_none : bool, default False
            If True, the method returns silently if the ID is not found.
            If False, a ValueError is raised.

        Raises
        ------
        ValueError
            If `rec_id` is not found in the pipeline and `ok_if_none` is False.
        """
        rec = self.get_by_id(rec_id)
        if rec:
            rec.enabled = False
        elif not ok_if_none:
            raise ValueError(f"Disable failed: Recommendation ID '{rec_id}' not found.")

    def toggle_enabled_by_id(self, rec_id: str, ok_if_none: bool = False) -> None:
        """
        Flips the 'enabled' state of a specific recommendation.

        Parameters
        ----------
        rec_id : str
            The unique identifier of the recommendation to activate.
        ok_if_none : bool, default False
            If True, the method returns silently if the ID is not found.
            If False, a ValueError is raised.

        Raises
        ------
        ValueError
            If `rec_id` is not found in the pipeline and `ok_if_none` is False.
        """
        rec = self.get_by_id(rec_id)
        if rec:
            rec.enabled = not rec.enabled
        elif not ok_if_none:
            raise ValueError(f"Toggle failed: Recommendation ID '{rec_id}' not found.")

    from dataclasses import fields

    def save_to_yaml(
        self, output_dir: PathLike, filename: str
    ) -> tuple[Path | CloudPath, dict[str, Any]]:
        """
        Serializes the internal pipeline to a YAML file as a dictionary keyed by ID.

        Parameters
        ----------
        output_dir : str | Path | CloudPath
            The destination directory for the recommendation file. Supports
            local filesystem paths and cloud URIs.
        filename : str
            The base name for the YAML recommendation file.

        Notes
        -----
        Each recommendation is converted to a dictionary keyed by ID. Attributes
        not whitelisted as 'editable' in the class metadata receive a [RO] suffix
        to prevent accidental modification of the audit trail.

        Returns
        -------
        tuple[Path | CloudPath, dict[str, Any]]
            The full output path and rejected keyword arguments returned by
            `dsr_files.yaml_handler.save_yaml`.
        """
        data = {}

        for rec in self._pipeline:
            # Get all fields for this specific recommendation instance
            editable_fields = [
                f.name for f in fields(rec) if f.metadata.get("editable", False)
            ]

            # Convert the recommendation to its dictionary representation
            rec_dict = rec.to_dict()
            rec_dict.pop("id", None)  # Remove redundant ID as it is the top-level key

            # Apply [RO] indicators to non-editable keys
            processed_rec = {}
            for key, value in rec_dict.items():
                if key in editable_fields:
                    processed_rec[key] = value
                else:
                    processed_rec[f"{key} [RO]"] = value

            data[rec.id] = processed_rec

        header = (
            "CAUTION: Do not modify the top-level keys (IDs) or keys marked [RO].\n"
            "Only fields without [RO] (e.g., 'enabled', 'notes') are intended for manual edits.\n"
            "Modifying [RO] fields will result in those changes being ignored during 'clean'."
        )

        return save_yaml(data, output_dir, filename, header=header)
