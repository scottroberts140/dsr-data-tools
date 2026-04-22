from dsr_data_tools.enums import MissingValueStrategy, RecommendationType
from dsr_data_tools.recommendations import (
    IntegerConversionRecommendation,
    MissingValuesRecommendation,
    RecommendationManager,
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
