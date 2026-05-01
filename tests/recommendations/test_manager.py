import pandas as pd
from dsr_data_tools.enums import (
    BitDepth,
    EncodingStrategy,
    InteractionType,
    MissingValueStrategy,
    OutlierStrategy,
    RecommendationType,
)
from dsr_data_tools.recommendations import (
    BinningRecommendation,
    BooleanClassificationRecommendation,
    ColumnHint,
    EncodingRecommendation,
    FeatureInteractionRecommendation,
    FloatConversionRecommendation,
    IntegerConversionRecommendation,
    MissingValuesRecommendation,
    OutlierDetectionRecommendation,
    RecommendationManager,
    ValueReplacementRecommendation,
)


def test_recommendation_ids_are_stable():
    rec1 = IntegerConversionRecommendation(
        column_name="col_a", description="desc", integer_count=5
    )
    rec2 = IntegerConversionRecommendation(
        column_name="col_a", description="desc", integer_count=5
    )
    assert rec1.id == rec2.id  # Deterministic hashing


def test_priority_mapping_exists():
    from dsr_data_tools.recommendations import RecommendationManager

    manager = RecommendationManager()

    # Every RecommendationType should have a priority defined
    for rec_type in RecommendationType:
        assert rec_type in manager.EXECUTION_PRIORITY


def test_manager_priority_sorting():
    from dsr_data_tools.recommendations import (
        BooleanClassificationRecommendation,
        NonInformativeRecommendation,
        RecommendationManager,
        RecommendationType,
    )

    manager = RecommendationManager()

    # Use keyword arguments to ensure values are mapped correctly
    rec_bool = BooleanClassificationRecommendation(
        column_name="a", description="bool desc", values=["1", "0"]
    )

    rec_drop = NonInformativeRecommendation(
        column_name="b", description="drop desc", reason="redundant"
    )

    manager._pipeline = [rec_bool, rec_drop]
    manager._refine_pipeline_strategies()

    # Verify priority sorting (Drops should be higher priority than Bool Classification)
    assert manager._pipeline[0].rec_type == RecommendationType.NON_INFORMATIVE
    assert manager._pipeline[1].rec_type == RecommendationType.BOOLEAN_CLASSIFICATION


def test_recommendation_to_dict_serialization():
    """Verifies that Recommendation objects serialize Enums to strings."""
    rec = IntegerConversionRecommendation(
        column_name="col_a", description="desc", integer_count=5
    )

    data = rec.to_dict()

    # Assert base attributes and Enum names
    assert data["column_name"] == "col_a"
    assert data["rec_type"] == "INT_CONVERSION"

    # Assert internal-only fields are excluded
    assert "_locked" not in data
    assert "integer_count" in data


def test_manager_save_to_yaml(tmp_path):
    """Verifies the manager correctly writes the pipeline to a YAML file."""
    filepath = tmp_path / "recommendations.yaml"
    rec = IntegerConversionRecommendation(
        column_name="age", description="Convert to int", integer_count=10
    )

    manager = RecommendationManager(recommendations=[rec])
    output_path, rejected = manager.save_to_yaml(tmp_path, "recommendations")

    # Check file exists and has content
    assert filepath.exists()
    assert output_path == filepath
    assert rejected == {}
    content = filepath.read_text()
    assert "age" in content
    assert "INT_CONVERSION" in content
    assert "stage_3" in content
    assert "explicit_stage" in content


def test_manager_load_from_yaml_round_trip(tmp_path):
    """Verifies recommendations can be loaded back from YAML for clean step."""
    rec = IntegerConversionRecommendation(
        column_name="age", description="Convert to int", integer_count=10
    )
    rec.enabled = False
    rec.notes = "manual override"
    rec.alias = "age_cast"

    manager = RecommendationManager(recommendations=[rec])
    filepath, _ = manager.save_to_yaml(tmp_path, "recommendations")

    loaded = RecommendationManager.load_from_yaml(filepath)
    loaded_rec = loaded.get_by_id(rec.id)

    assert isinstance(loaded_rec, IntegerConversionRecommendation)
    assert loaded_rec is not None
    assert loaded_rec.id == rec.id
    assert loaded_rec.column_name == "age"
    assert loaded_rec.enabled is False
    assert loaded_rec.notes == "manual override"
    assert loaded_rec.alias == "age_cast"


def test_manager_load_from_yaml_parses_enum_fields(tmp_path):
    """Verifies enum-valued editable fields are restored from YAML strings."""
    yaml_text = (
        "rec_custom_001:\n"
        "  column_name [RO]: workclass\n"
        "  description [RO]: Handle missing values\n"
        "  rec_type [RO]: MISSING_VALUES\n"
        "  missing_count [RO]: 1800\n"
        "  missing_percentage [RO]: 5.5\n"
        "  strategy: DROP_ROWS\n"
        "  enabled: true\n"
    )
    filepath = tmp_path / "recommendations.yaml"
    filepath.write_text(yaml_text)

    loaded = RecommendationManager.load_from_yaml(filepath)
    rec = loaded.get_by_id("rec_custom_001")

    assert isinstance(rec, MissingValuesRecommendation)
    assert rec is not None
    assert rec.strategy == MissingValueStrategy.DROP_ROWS
    assert rec.id == "rec_custom_001"


def test_manager_load_from_staged_yaml_syncs_explicit_stage(tmp_path):
    """Verifies stage placement in YAML overrides stale explicit_stage values."""
    yaml_text = (
        "stage_2:\n"
        "  rec_custom_002:\n"
        "    column_name [RO]: workclass\n"
        "    description [RO]: Handle missing values\n"
        "    rec_type [RO]: MISSING_VALUES\n"
        "    missing_count [RO]: 10\n"
        "    missing_percentage [RO]: 1.0\n"
        "    strategy: DROP_ROWS\n"
        "    explicit_stage: 99\n"
        "    enabled: true\n"
    )
    filepath = tmp_path / "recommendations.yaml"
    filepath.write_text(yaml_text)

    loaded = RecommendationManager.load_from_yaml(filepath)
    rec = loaded.get_by_id("rec_custom_002")

    assert isinstance(rec, MissingValuesRecommendation)
    assert rec is not None
    assert rec.explicit_stage == 2


def test_manager_generates_integer_and_float_recommendations_from_hints():
    df = pd.DataFrame({"count": [1.0, 2.0, 3.0], "ratio": [0.1, 0.2, 0.3]})

    manager = RecommendationManager()
    manager.generate_recommendations(
        df,
        hints={
            "count": ColumnHint.integer(target_depth=BitDepth.INT64),
            "ratio": ColumnHint.floating(target_depth=BitDepth.FLOAT32),
        },
        hints_only=True,
    )

    rec_count = next(rec for rec in manager._pipeline if rec.column_name == "count")
    rec_ratio = next(rec for rec in manager._pipeline if rec.column_name == "ratio")

    assert isinstance(rec_count, IntegerConversionRecommendation)
    assert rec_count.target_depth == BitDepth.INT64
    assert rec_count.is_locked is True

    assert isinstance(rec_ratio, FloatConversionRecommendation)
    assert rec_ratio.target_depth == BitDepth.FLOAT32
    assert rec_ratio.is_locked is True


def test_manager_generates_boolean_binning_and_replacement_from_hints():
    df = pd.DataFrame(
        {
            "flag": ["Y", "N", "Y"],
            "age": [10, 25, 40],
            "score": ["10", "n/a", "25"],
        }
    )

    manager = RecommendationManager()
    manager.generate_recommendations(
        df,
        hints={
            "flag": ColumnHint.boolean(values=["Y", "N"]),
            "age": ColumnHint.binning(
                bins=[0, 18, 65],
                labels=["minor", "adult"],
            ),
            "score": ColumnHint.value_replacement(
                values=["n/a"],
                replacement_value="0",
            ),
        },
        hints_only=True,
    )

    rec_flag = next(rec for rec in manager._pipeline if rec.column_name == "flag")
    rec_age = next(rec for rec in manager._pipeline if rec.column_name == "age")
    rec_score = next(rec for rec in manager._pipeline if rec.column_name == "score")

    assert isinstance(rec_flag, BooleanClassificationRecommendation)
    assert rec_flag.values == ["Y", "N"]
    assert rec_flag.is_locked is True

    assert isinstance(rec_age, BinningRecommendation)
    assert rec_age.bins == [0, 18, 65]
    assert rec_age.labels == ["minor", "adult"]
    assert rec_age.is_locked is True

    assert isinstance(rec_score, ValueReplacementRecommendation)
    assert rec_score.non_numeric_values == ["n/a"]
    assert rec_score.non_numeric_count == 1
    assert rec_score.replacement_value == "0"
    assert rec_score.is_locked is True


def test_manager_generates_encoding_and_interaction_from_hints():
    df = pd.DataFrame(
        {
            "category": ["a", "b", "a"],
            "fare_amount": [10.0, 20.0, 30.0],
            "trip_distance": [2.0, 4.0, 6.0],
        }
    )

    manager = RecommendationManager()
    manager.generate_recommendations(
        df,
        hints={
            "category": ColumnHint.encoding(strategy=EncodingStrategy.LABEL),
            "fare_amount": ColumnHint.feature_interaction(
                interaction_column="trip_distance",
                interaction_type=InteractionType.RESOURCE_DENSITY,
                operation="/",
                rationale="fare per mile",
                derived_name="fare_per_mile",
            ),
        },
        hints_only=True,
    )

    rec_category = next(
        rec for rec in manager._pipeline if rec.column_name == "category"
    )
    rec_fare = next(
        rec for rec in manager._pipeline if rec.column_name == "fare_amount"
    )

    assert isinstance(rec_category, EncodingRecommendation)
    assert rec_category.encoder_type == EncodingStrategy.LABEL
    assert rec_category.unique_values == 2
    assert rec_category.is_locked is True

    assert isinstance(rec_fare, FeatureInteractionRecommendation)
    assert rec_fare.column_name_2 == "trip_distance"
    assert rec_fare.interaction_type == InteractionType.RESOURCE_DENSITY
    assert rec_fare.operation == "/"
    assert rec_fare.rationale == "fare per mile"
    assert rec_fare.derived_name == "fare_per_mile"
    assert rec_fare.is_locked is True


def test_manager_generates_outlier_detection_from_hints():
    df = pd.DataFrame({"fare_amount": [10.0, 12.0, 500.0]})

    manager = RecommendationManager()
    manager.generate_recommendations(
        df,
        hints={
            "fare_amount": ColumnHint.outlier_detection(
                strategy=OutlierStrategy.ROBUST_SCALER
            )
        },
        hints_only=True,
    )

    rec = next(rec for rec in manager._pipeline if rec.column_name == "fare_amount")

    assert isinstance(rec, OutlierDetectionRecommendation)
    assert rec.strategy == OutlierStrategy.ROBUST_SCALER
    assert rec.max_value == 500.0
    assert rec.mean_value == df["fare_amount"].mean()
    assert rec.is_locked is True
