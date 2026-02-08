from enum import Enum


class RecommendationType(Enum):
    """Enumeration of recommendation types for dataset preparation."""

    NON_INFORMATIVE = "non_informative"
    """Column with no predictive value (unique count equals row count, high cardinality, etc.)"""

    MISSING_VALUES = "missing_values"
    """Column has null/missing values; recommend removal or imputation strategy"""

    ENCODING = "encoding"
    """Categorical column requires encoding (OneHotEncoder, LabelEncoder)"""

    CLASS_IMBALANCE = "class_imbalance"
    """Target variable shows class imbalance; recommend SMOTE or class weighting"""

    OUTLIER_DETECTION = "outlier_detection"
    """Numeric column has outliers; recommend scaling or RobustScaler"""

    OUTLIER_HANDLING = "outlier_handling"
    """Numeric column has outliers; clean by nullifying or clipping values beyond bounds"""

    BOOLEAN_CLASSIFICATION = "boolean_classification"
    """Numeric column with exactly two unique values (0, 1); treat as boolean"""

    BINNING = "binning"
    """Numeric column should be binned into categorical ranges"""

    INT_CONVERSION = "int_conversion"
    """Convert a column to int"""

    FLOAT_CONVERSION = "float_conversion"
    """Convert a column to float"""

    DECIMAL_PRECISION_OPTIMIZATION = "decimal_precision_optimization"
    """Float column can have decimal precision reduced or be converted to int64"""

    VALUE_REPLACEMENT = "value_replacement"
    """Column contains non-numeric placeholder values that should be replaced"""

    FEATURE_INTERACTION = "feature_interaction"
    """Recommended interaction feature combining two columns"""

    DATETIME_CONVERSION = "datetime_conversion"
    """Object/string column likely contains datetimes; convert to datetime dtype"""

    FEATURE_EXTRACTION = "feature_extraction"
    """Extract derived features from a column (e.g., datetime components)"""

    CATEGORICAL_CONVERSION = "categorical_conversion"
    """Convert object column to pandas categorical dtype for memory optimization"""

    FEATURE_AGGREGATION = "feature_aggregation"
    """Aggregate multiple columns into a new feature (e.g., sum, mean)."""


class InteractionType(Enum):
    """Types of feature interactions."""

    STATUS_IMPACT = "status_impact"
    """Binary × Continuous: Status column paired with high-variance continuous column"""

    RESOURCE_DENSITY = "resource_density"
    """Continuous / Continuous: Resource ratio (e.g., Balance / Salary)"""

    PRODUCT_UTILIZATION = "product_utilization"
    """Discrete / Continuous: Utilization rate (e.g., Products / Tenure)"""


class EncodingStrategy(Enum):
    """Strategies for encoding categorical columns."""

    ONEHOT = "onehot"
    """One-hot encoding for multi-class categorical variables"""

    LABEL = "label"
    """Label encoding for ordinal or binary categorical variables"""

    ORDINAL = "ordinal"
    """Ordinal encoding for ordered categorical variables"""
    CATEGORICAL = "categorical"
    """Convert to pandas categorical dtype (memory optimization, no transformation)"""


class MissingValueStrategy(Enum):
    """Strategies for handling missing values."""

    DROP_ROWS = "drop_rows"
    """Remove rows with missing values"""

    IMPUTE_MEAN = "impute_mean"
    """Impute missing values with column mean"""

    IMPUTE_MEDIAN = "impute_median"
    """Impute missing values with column median"""

    IMPUTE_MODE = "impute_mode"
    """Impute missing values with column mode"""

    DROP_COLUMN = "drop_column"
    """Remove the column entirely"""

    FILL_VALUE = "fill_value"
    """Fill missing values with a specified value"""

    LEAVE_AS_NA = "leave_as_na"
    """Leave missing values as-is (no action taken)"""


class OutlierStrategy(Enum):
    """Strategies for handling outliers."""

    SCALING = "scaling"
    """Use StandardScaler to normalize values"""

    ROBUST_SCALER = "robust_scaler"
    """Use RobustScaler to handle outliers"""

    REMOVE = "remove"
    """Remove rows with outliers"""


class OutlierHandlingStrategy(Enum):
    """Strategies for cleaning outliers by nullifying or clipping."""

    NULLIFY = "nullify"
    """Set outliers beyond bounds to NaN"""

    CLIP = "clip"
    """Cap outliers at the lower and upper bounds"""


class ImbalanceStrategy(Enum):
    """Strategies for handling class imbalance."""

    SMOTE = "smote"
    """Synthetic Minority Over-sampling Technique"""

    CLASS_WEIGHT = "class_weight"
    """Adjust class weights in the model"""

    UPSAMPLING = "upsampling"
    """Over-sample the minority class"""

    DOWNSAMPLING = "downsampling"
    """Under-sample the majority class"""


class ColumnHintType(Enum):
    """Logical types for ColumnHint to guide recommendation generation."""

    DATETIME = "datetime"
    """Datetime column; extract temporal features or convert from string"""

    FINANCIAL = "financial"
    """Financial/monetary value; apply floor/ceiling bounds"""

    CATEGORICAL = "categorical"
    """Convert to categorical dtype for memory optimization"""

    NUMERIC = "numeric"
    """Numeric column; apply floor/ceiling bounds if specified"""

    AGGREGATE = "aggregate"
    """Aggregate multiple columns into a new feature"""
    GEOSPATIAL = "geospatial"
    """Latitude/longitude columns; apply bounding box constraints"""

    DISTANCE = "distance"
    """Distance-based column; specify unit and optional bounds"""


class RoundingMode(Enum):
    """Rounding modes for decimal precision operations."""

    NEAREST = "nearest"
    """Standard 'Round Half Up' (biased) - 1.5→2.0, 2.5→3.0"""

    BANKERS = "bankers"
    """Round to nearest even (unbiased) - 1.5→2.0, 2.5→2.0"""

    UP = "up"
    """Ceiling - always away from zero - 1.5→2.0, 2.5→3.0"""

    DOWN = "down"
    """Floor - always toward zero - 1.5→1.0, 2.5→2.0"""


class BitDepth(Enum):
    """Supported bit depths for numeric conversions."""

    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    @property
    def is_float(self) -> bool:
        return self in (BitDepth.FLOAT32, BitDepth.FLOAT64)
