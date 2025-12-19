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

    BOOLEAN_CLASSIFICATION = "boolean_classification"
    """Numeric column with exactly two unique values (0, 1); treat as boolean"""

    BINNING = "binning"
    """Numeric column should be binned into categorical ranges"""

    INT64_CONVERSION = "int64_conversion"
    """Float column with only integer values should be converted to int64"""

    DECIMAL_PRECISION_OPTIMIZATION = "decimal_precision_optimization"
    """Float column can have decimal precision reduced or be converted to int64"""


class EncodingStrategy(Enum):
    """Strategies for encoding categorical columns."""

    ONEHOT = "onehot"
    """One-hot encoding for multi-class categorical variables"""

    LABEL = "label"
    """Label encoding for ordinal or binary categorical variables"""

    ORDINAL = "ordinal"
    """Ordinal encoding for ordered categorical variables"""


class MissingValueStrategy(Enum):
    """Strategies for handling missing values."""

    DROP_ROWS = "drop_rows"
    """Remove rows with missing values"""

    IMPUTE = "impute"
    """Impute missing values (mean, median, mode, etc.)"""

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
