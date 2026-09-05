"""Run with: python -m src.cognitive_models.minimal_mental_model.example"""

from . import CounterfactualStrategies, FeatureSpace, FeatureSpec, MentalModel, MinimalCognitiveModel


def main() -> None:
    space = FeatureSpace(
        [
            FeatureSpec.numerical("income", 0, 100),
            FeatureSpec.numerical("debt", 0, 100),
            FeatureSpec.categorical("history", ["poor", "good"]),
        ]
    )
    model = MinimalCognitiveModel(space, class_labels=("reject", "approve"))
    model.observe(
        {"income": 25, "debt": 80, "history": "poor"}, "reject"
    )
    model.observe(
        {"income": 75, "debt": 20, "history": "good"},
        "approve",
        explanation="attribution",
        attributions={"income": 0.7, "debt": -0.5, "history": 0.3},
    )
    model.observe_counterfactual(
        {"income": 35, "debt": 70, "history": "poor"},
        {"income": 65, "debt": 40, "history": "good"},
        "reject",
        "approve",
    )

    query = {"income": 40, "debt": 65, "history": "poor"}
    for mental_model in MentalModel:
        print(mental_model.value, model.predict(query, mental_model))

    strategies = CounterfactualStrategies(model)
    print(strategies.most_contributing(query, "approve", max_changes=2))
    print(strategies.prototype_mismatch(query, "approve", max_changes=2))
    print(strategies.remembered_exemplar(query, "approve", max_changes=2))
    print(strategies.copy_remembered_change(query, "approve", max_changes=2))


if __name__ == "__main__":
    main()

