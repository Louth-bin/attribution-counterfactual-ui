from __future__ import annotations

import numpy as np

from src.cognitive_models.minimal_mental_model import (
    CounterfactualStrategies,
    DeclarativeMemory,
    FeatureSpace,
    FeatureSpec,
    MentalModel,
    MinimalCognitiveModel,
)


def build_model() -> MinimalCognitiveModel:
    space = FeatureSpace(
        [
            FeatureSpec.numerical("x1", 0, 10),
            FeatureSpec.numerical("x2", 0, 10),
            FeatureSpec.categorical("group", ["a", "b"]),
        ]
    )
    model = MinimalCognitiveModel(space, class_labels=(0, 1), learning_rate=0.5)
    model.observe({"x1": 1, "x2": 9, "group": "a"}, 0)
    model.observe({"x1": 2, "x2": 8, "group": "a"}, 0)
    model.observe(
        {"x1": 8, "x2": 2, "group": "b"},
        1,
        explanation="attribution",
        attributions={"x1": 0.8, "x2": -0.6, "group": 0.2},
    )
    model.observe({"x1": 9, "x2": 1, "group": "b"}, 1)
    return model


def test_memory_activation_rewards_reuse() -> None:
    memory = DeclarativeMemory(retrieval_threshold=-10)
    first = memory.store("fact", {"name": "a"}, {"value": 1})
    memory.advance(10)
    second = memory.store("fact", {"name": "b"}, {"value": 2})
    assert memory.activation(second) > memory.activation(first)
    memory.touch(first)
    assert memory.activation(first) > memory.activation(second)


def test_three_forward_models_learn_class_structure() -> None:
    model = build_model()
    low = {"x1": 1.5, "x2": 8.5, "group": "a"}
    high = {"x1": 8.5, "x2": 1.5, "group": "b"}
    for mental_model in MentalModel:
        assert model.predict(low, mental_model).label == 0
        assert model.predict(high, mental_model).label == 1


def test_example_explanation_rehearses_its_chunk() -> None:
    model = build_model()
    model.observe({"x1": 5, "x2": 5, "group": "a"}, 0, explanation="example")
    assert len(model.memory.chunks[-1].use_times) == 2


def test_feature_space_builds_from_dataset_bundle_fields() -> None:
    space = FeatureSpace.from_dataset_fields(
        ["amount", "kind"],
        ["numerical", "categorical"],
        [[0, 10], ["a", "b"]],
    )
    assert space.encoded_size == 3
    assert space.decode(space.encode({"amount": 4, "kind": "b"}))["kind"] == "b"


def test_counterfactual_learning_changes_weights_and_stores_change() -> None:
    model = build_model()
    before = model.attribution_weights().copy()
    model.observe_counterfactual(
        {"x1": 3, "x2": 7, "group": "a"},
        {"x1": 7, "x2": 3, "group": "b"},
        0,
        1,
    )
    after = model.attribution_weights()
    assert not np.allclose(before, after)
    assert len(model.memory.retrieve_many(kind="change")) == 1


def test_strategies_produce_bounded_edits() -> None:
    model = build_model()
    model.observe_counterfactual(
        {"x1": 3, "x2": 7, "group": "a"},
        {"x1": 7, "x2": 4, "group": "b"},
        0,
        1,
    )
    query = {"x1": 2, "x2": 8, "group": "a"}
    strategies = CounterfactualStrategies(model)
    proposals = [
        strategies.most_contributing(query, 1, max_changes=2),
        strategies.global_influence(query, 1, mode="fixed_value", max_changes=2),
        strategies.global_influence(query, 1, mode="fixed_delta", max_changes=2),
        strategies.prototype_mismatch(query, 1, max_changes=2),
        strategies.remembered_exemplar(query, 1, max_changes=2),
        strategies.copy_remembered_change(query, 1, max_changes=2),
    ]
    assert all(proposal.changed_features for proposal in proposals)
    for proposal in proposals:
        assert 0 <= proposal.edited["x1"] <= 10
        assert 0 <= proposal.edited["x2"] <= 10
        assert proposal.edited["group"] in {"a", "b"}
