from dsr_data_tools.enums import RecommendationType
from dsr_data_tools.recommendations import IntegerConversionRecommendation


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
