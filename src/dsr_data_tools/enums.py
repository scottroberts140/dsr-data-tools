"""Enumerations for recommendation and preprocessing strategies."""

from enum import Enum


class RecommendationType(Enum):
    """
    Enumeration of recommendation types for dataset preparation.

    Attributes
    ----------
    NON_INFORMATIVE : str
        Column with no predictive value (e.g., unique count equals row count).
    MISSING_VALUES : str
        Column has null/missing values; recommend removal or imputation.
    ENCODING : str
        Categorical column requires encoding (OneHot, Label, etc.).
    CLASS_IMBALANCE : str
        Target variable shows class imbalance; recommend resampling or weighting.
    OUTLIER_DETECTION : str
        Numeric column has outliers; recommend scaling or robust methods.
    OUTLIER_HANDLING : str
        Clean outliers by nullifying or clipping values beyond bounds.
    BOOLEAN_CLASSIFICATION : str
        Numeric column with exactly two unique values (0, 1); treat as boolean.
    BINNING : str
        Numeric column should be binned into categorical ranges.
    INT_CONVERSION : str
        Convert a column to integer dtype.
    FLOAT_CONVERSION : str
        Convert a column to float dtype.
    DECIMAL_PRECISION_OPTIMIZATION : str
        Reduce float precision or convert to int64 for efficiency.
    VALUE_REPLACEMENT : str
        Replace non-numeric placeholder values (e.g., '?', 'NA').
    FEATURE_INTERACTION : str
        Recommended interaction feature combining two columns.
    DATETIME_CONVERSION : str
        Convert object/string column to datetime dtype.
    FEATURE_EXTRACTION : str
        Extract derived features (e.g., hour, month) from a column.
    CATEGORICAL_CONVERSION : str
        Convert to pandas categorical dtype for memory optimization.
    FEATURE_AGGREGATION : str
        Aggregate multiple columns into a new feature (sum, mean, etc.).
    """

    NON_INFORMATIVE = "non_informative"
    MISSING_VALUES = "missing_values"
    ENCODING = "encoding"
    CLASS_IMBALANCE = "class_imbalance"
    OUTLIER_DETECTION = "outlier_detection"
    OUTLIER_HANDLING = "outlier_handling"
    BOOLEAN_CLASSIFICATION = "boolean_classification"
    BINNING = "binning"
    INT_CONVERSION = "int_conversion"
    FLOAT_CONVERSION = "float_conversion"
    DECIMAL_PRECISION_OPTIMIZATION = "decimal_precision_optimization"
    VALUE_REPLACEMENT = "value_replacement"
    FEATURE_INTERACTION = "feature_interaction"
    DATETIME_CONVERSION = "datetime_conversion"
    FEATURE_EXTRACTION = "feature_extraction"
    CATEGORICAL_CONVERSION = "categorical_conversion"
    FEATURE_AGGREGATION = "feature_aggregation"


class InteractionType(Enum):
    """
    Types of feature interactions.

    Attributes
    ----------
    STATUS_IMPACT : str
        Binary × Continuous: Status column paired with continuous column.
    RESOURCE_DENSITY : str
        Continuous / Continuous: Resource ratio (e.g., Balance / Salary).
    PRODUCT_UTILIZATION : str
        Discrete / Continuous: Utilization rate (e.g., Products / Tenure).
    """

    STATUS_IMPACT = "status_impact"
    RESOURCE_DENSITY = "resource_density"
    PRODUCT_UTILIZATION = "product_utilization"


class EncodingStrategy(Enum):
    """
    Strategies for encoding categorical columns.

    Attributes
    ----------
    ONEHOT : str
        One-hot encoding for multi-class categorical variables.
    LABEL : str
        Label encoding for ordinal or binary variables.
    ORDINAL : str
        Ordinal encoding for ordered categorical variables.
    CATEGORICAL : str
        Convert to pandas categorical dtype for memory optimization.
    """

    ONEHOT = "onehot"
    LABEL = "label"
    ORDINAL = "ordinal"
    CATEGORICAL = "categorical"


class MissingValueStrategy(Enum):
    """
    Strategies for handling missing values.

    Attributes
    ----------
    DROP_ROWS : str
        Remove rows with missing values.
    IMPUTE_MEAN : str
        Impute missing values with column mean.
    IMPUTE_MEDIAN : str
        Impute missing values with column median.
    IMPUTE_MODE : str
        Impute missing values with column mode.
    DROP_COLUMN : str
        Remove the column entirely.
    FILL_VALUE : str
        Fill missing values with a specified constant.
    LEAVE_AS_NA : str
        Leave missing values as-is (no action taken).
    """

    DROP_ROWS = "drop_rows"
    IMPUTE_MEAN = "impute_mean"
    IMPUTE_MEDIAN = "impute_median"
    IMPUTE_MODE = "impute_mode"
    DROP_COLUMN = "drop_column"
    FILL_VALUE = "fill_value"
    LEAVE_AS_NA = "leave_as_na"


class OutlierStrategy(Enum):
    """
    Strategies for initial outlier detection/scaling.

    Attributes
    ----------
    SCALING : str
        Use StandardScaler to normalize values.
    ROBUST_SCALER : str
        Use RobustScaler to handle outliers.
    REMOVE : str
        Remove rows containing outliers.
    """

    SCALING = "scaling"
    ROBUST_SCALER = "robust_scaler"
    REMOVE = "remove"


class OutlierHandlingStrategy(Enum):
    """
    Strategies for cleaning outliers by modification.

    Attributes
    ----------
    NULLIFY : str
        Set outliers beyond specified bounds to NaN.
    CLIP : str
        Cap outliers at the lower and upper bounds.
    """

    NULLIFY = "nullify"
    CLIP = "clip"


class ImbalanceStrategy(Enum):
    """
    Strategies for handling class imbalance.

    Attributes
    ----------
    SMOTE : str
        Synthetic Minority Over-sampling Technique.
    CLASS_WEIGHT : str
        Adjust class weights within the model algorithm.
    UPSAMPLING : str
        Over-sample the minority class.
    DOWNSAMPLING : str
        Under-sample the majority class.
    """

    SMOTE = "smote"
    CLASS_WEIGHT = "class_weight"
    UPSAMPLING = "upsampling"
    DOWNSAMPLING = "downsampling"


class ColumnHintType(Enum):
    """
    Logical types for ColumnHint to guide recommendations.

    Attributes
    ----------
    DATETIME : str
        Datetime column; extract temporal features or convert.
    FINANCIAL : str
        Financial value; apply floor/ceiling bounds.
    CATEGORICAL : str
        Convert to categorical dtype for memory efficiency.
    NUMERIC : str
        Standard numeric column; apply bounds if specified.
    AGGREGATE : str
        Aggregate multiple columns into a new feature.
    GEOSPATIAL : str
        Latitude/longitude columns; apply bounding box constraints.
    DISTANCE : str
        Distance-based column; specify unit and optional bounds.
    """

    DATETIME = "datetime"
    FINANCIAL = "financial"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    AGGREGATE = "aggregate"
    GEOSPATIAL = "geospatial"
    DISTANCE = "distance"


class RoundingMode(Enum):
    """
    Rounding modes for decimal precision operations.

    Attributes
    ----------
    NEAREST : str
        Standard 'Round Half Up' (biased).
    BANKERS : str
        Round to nearest even (unbiased).
    UP : str
        Ceiling; always away from zero.
    DOWN : str
        Floor; always toward zero.
    """

    NEAREST = "nearest"
    BANKERS = "bankers"
    UP = "up"
    DOWN = "down"


class BitDepth(Enum):
    """
    Supported bit depths for numeric conversions.

    Attributes
    ----------
    INT32 : str
        32-bit integer.
    INT64 : str
        64-bit integer.
    FLOAT32 : str
        32-bit floating point.
    FLOAT64 : str
        64-bit floating point.
    """

    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    @property
    def is_float(self) -> bool:
        """
        Return True if the bit depth is a floating-point type.

        Returns
        -------
        bool
            True if float, False otherwise.
        """
        return self in (BitDepth.FLOAT32, BitDepth.FLOAT64)
