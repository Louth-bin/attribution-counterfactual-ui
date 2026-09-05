import numpy as np

from src.cognitive_models.minimal_mental_model import (
    CounterfactualStrategies,
    FeatureSpace,
    FeatureSpec,
    MentalModel,
    PerfectMemoryCognitiveModel,
)


def make_space():
    return FeatureSpace(
        [
            FeatureSpec.numerical("x", 0, 10),
            FeatureSpec.numerical("y", 0, 10),
        ]
    )


def test_attribution_explanation_sets_shared_relevance():
    model = PerfectMemoryCognitiveModel(make_space(), ("left", "right"))
    model.observe(
        {"x": 2, "y": 2},
        "left",
        explanation="attribution",
        attributions={"x": -0.75, "y": 0.25},
    )

    np.testing.assert_allclose(model.relevance(), [0.75, 0.25])
    assert model.global_importance() == {"x": 0.75, "y": 0.25}


def test_counterfactual_is_stored_as_target_instance_and_change():
    model = PerfectMemoryCognitiveModel(make_space(), ("left", "right"))
    model.observe(
        {"x": 2, "y": 2},
        "left",
        explanation="counterfactual",
        counterfactual_profile={"x": 8, "y": 2},
        counterfactual_target_label="right",
    )

    assert len(model.instances) == 2
    assert len(model.changes) == 1
    assert model.instances[-1].label == "right"
    np.testing.assert_allclose(model.relevance(), [1.0, 0.0])


def test_all_forward_models_return_probabilities_without_decay():
    model = PerfectMemoryCognitiveModel(make_space(), ("left", "right"))
    model.observe({"x": 1, "y": 1}, "left")
    model.observe({"x": 9, "y": 9}, "right")

    for mental_model in MentalModel:
        prediction = model.predict({"x": 8, "y": 8}, mental_model)
        assert set(prediction.probabilities) == {"left", "right"}
        assert np.isclose(sum(prediction.probabilities.values()), 1.0)

    # Perfect memory means repeated prediction does not alter or decay memory.
    before = len(model.instances)
    model.predict({"x": 5, "y": 5}, MentalModel.EXEMPLAR)
    assert len(model.instances) == before


def test_reviewed_strategy_names_work_with_perfect_memory():
    model = PerfectMemoryCognitiveModel(make_space(), ("left", "right"))
    model.observe(
        {"x": 1, "y": 1},
        "left",
        explanation="counterfactual",
        counterfactual_profile={"x": 9, "y": 1},
        counterfactual_target_label="right",
    )
    strategies = CounterfactualStrategies(model)

    exemplar_edit = strategies.change_toward_remembered_exemplar(
        {"x": 2, "y": 1}, "right"
    )
    copied_edit = strategies.copy_changes_from_remembered_example(
        {"x": 2, "y": 1}, "right"
    )

    assert exemplar_edit.strategy == "change_toward_remembered_exemplar"
    assert copied_edit.strategy == "copy_changes_from_remembered_example"
    assert exemplar_edit.edited["x"] == 9
    assert copied_edit.edited["x"] > 2


def test_encoded_edits_preserve_unselected_raw_values_outside_model_range():
    model = PerfectMemoryCognitiveModel(make_space(), ("left", "right"))
    model.observe({"x": 1, "y": 1}, "left")
    model.observe({"x": 9, "y": 9}, "right")
    strategies = CounterfactualStrategies(model)

    proposal = strategies.change_most_influential_attributes_by_fixed_amount(
        {"x": -2, "y": 5},
        "right",
        max_changes=1,
        fixed_amount=0.25,
    )

    assert len(proposal.changed_features) <= 1
    if "x" not in proposal.changed_features:
        assert proposal.edited["x"] == -2
    if "y" not in proposal.changed_features:
        assert proposal.edited["y"] == 5
