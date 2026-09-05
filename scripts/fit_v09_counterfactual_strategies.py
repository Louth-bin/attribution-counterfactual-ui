"""Fit counterfactual strategies to participants' testing-phase edits.

Training trials supply only the information participants were exposed to. They
do not determine the participant's mental model. The mental representation and
counterfactual strategy are inferred jointly from testing edits by comparing
the observed signed normalized edit vector with each strategy's prediction.

Discrete k and fixed-amount settings are selected with leave-one-instance-out
cross-validation. This prevents flexible strategy families from being scored
on the same responses used to select their settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_models.minimal_mental_model import (  # noqa: E402
    CounterfactualStrategies,
    FeatureSpace,
    PerfectMemoryCognitiveModel,
)


DEFAULT_INPUT = ROOT / "qualtrics" / "qualtrics_results_v0.9.for-plausibility.csv"
DEFAULT_BUNDLE = ROOT / "static" / "experiment-data.json"
DEFAULT_PARTICIPANTS = (
    ROOT / "qualtrics" / "v0.9_participant_counterfactual_strategy_fits.csv"
)
DEFAULT_TRIALS = ROOT / "qualtrics" / "v0.9_counterfactual_strategy_trial_predictions.csv"
DEFAULT_SUMMARY = ROOT / "qualtrics" / "v0.9_counterfactual_strategy_summary.csv"
DEFAULT_REPORT = ROOT / "qualtrics" / "v0.9_counterfactual_strategy_fit_report.md"

EPSILON = 1e-12
FIXED_AMOUNTS = tuple(float(value) for value in np.linspace(0.025, 1.0, 40))


STRATEGY_TO_MODEL = {
    "change_most_contributing_attributes": "attribution",
    "change_most_influential_attributes_to_flip": "attribution",
    "change_most_influential_attributes_by_fixed_amount": "attribution",
    "set_attributes_to_target_profile_values": "prototype",
    "change_attributes_not_matching_target_profile": "prototype",
    "change_largest_target_profile_mismatches": "prototype",
    "change_toward_remembered_exemplar": "exemplar",
    "copy_changes_from_remembered_example": "shared change memory",
}

# Participant-fitted action settings. The all-mismatch rule is fully specified;
# most strategy families fit only k; the fixed-amount rule fits both k and one
# common normalized amount.
STRATEGY_PARAMETER_COUNT = {
    "change_most_contributing_attributes": 1,
    "change_most_influential_attributes_to_flip": 1,
    "change_most_influential_attributes_by_fixed_amount": 2,
    "set_attributes_to_target_profile_values": 1,
    "change_attributes_not_matching_target_profile": 0,
    "change_largest_target_profile_mismatches": 1,
    "change_toward_remembered_exemplar": 1,
    "copy_changes_from_remembered_example": 1,
}

MODEL_ORDER = ("exemplar", "attribution", "prototype")
STRATEGY_ORDER = tuple(STRATEGY_TO_MODEL)


@dataclass(frozen=True)
class Setting:
    k: int | None = None
    fixed_amount: float | None = None

    @property
    def label(self) -> str:
        pieces = []
        if self.k is not None:
            pieces.append(f"k={self.k}")
        if self.fixed_amount is not None:
            pieces.append(f"amount={self.fixed_amount:.3f}")
        return ";".join(pieces) if pieces else "none"


@dataclass
class ActionPrediction:
    setting: Setting
    predicted_delta: np.ndarray
    predicted_changed: np.ndarray
    feature_loss: np.ndarray
    normalized_l1_loss: float
    feature_f1: float
    direction_accuracy: float | None
    sparsity_difference: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def profile_from_case(case: dict[str, Any], values_key: str = "raw_feature_values") -> dict[str, Any]:
    return dict(zip(case["raw_feature_names"], case[values_key]))


def explanation_payload(case: dict[str, Any], xai: str) -> dict[str, Any]:
    if xai == "none":
        return {"explanation": "none"}
    if xai == "attribution":
        return {
            "explanation": "attribution",
            "attributions": dict(
                zip(case["raw_feature_names"], case["attribution"]["values"])
            ),
        }
    if xai == "counterfactual":
        counterfactual = case["counterfactual"]
        return {
            "explanation": "counterfactual",
            "counterfactual_profile": dict(
                zip(case["raw_feature_names"], counterfactual["raw_feature_values"])
            ),
            "counterfactual_target_label": counterfactual["prediction"]["label"],
        }
    raise ValueError(f"Unknown explanation condition {xai!r}")


def load_bundle(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["datasets"]


def case_maps(dataset: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {str(case["instance_id"]): case for case in dataset["training_pool"]},
        {str(case["instance_id"]): case for case in dataset["test_pool"]},
    )


def build_participant_model(
    training_rows: list[dict[str, str]],
    training_cases: dict[str, dict[str, Any]],
) -> PerfectMemoryCognitiveModel:
    ordered = sorted(
        training_rows,
        key=lambda row: int(row.get("case") or row["trial number"]),
    )
    first = training_cases[ordered[0]["instance id"]]
    space = FeatureSpace.from_dataset_fields(
        first["raw_feature_names"], first["feature_types"], first["raw_feature_ranges"]
    )
    model = PerfectMemoryCognitiveModel(space, first["prediction_labels"])
    xai = ordered[0]["xai"]
    for row in ordered:
        case = training_cases[row["instance id"]]
        if case["prediction"]["label"] != row["original label"]:
            raise ValueError(
                f"Training label mismatch for {row['participant']}/{row['instance id']}"
            )
        # The participant's training response is deliberately not used here.
        model.observe(
            profile_from_case(case),
            case["prediction"]["label"],
            **explanation_payload(case, xai),
        )
    return model


def observed_action(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray([float(row[f"x_{index}_change"]) for index in range(1, 6)])
    changed = np.asarray(
        [bool(int(float(row[f"x_{index}_changed"]))) for index in range(1, 6)],
        dtype=bool,
    )
    return delta, changed


def normalized_feature_value(space: FeatureSpace, name: str, value: Any) -> float:
    spec = space.spec(name)
    if spec.kind == "numerical":
        assert spec.low is not None and spec.high is not None
        return float(np.clip((float(value) - spec.low) / (spec.high - spec.low), 0.0, 1.0))
    normalized = str(value).casefold()
    matches = [
        index
        for index, category in enumerate(spec.categories)
        if str(category).casefold() == normalized
    ]
    if not matches:
        raise ValueError(f"Unknown category {value!r} for {name!r}")
    return matches[0] / (len(spec.categories) - 1)


def predicted_action(
    space: FeatureSpace,
    original: dict[str, Any],
    edited: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(
        [
            normalized_feature_value(space, name, edited[name])
            - normalized_feature_value(space, name, original[name])
            for name in space.names
        ],
        dtype=float,
    )
    changed = np.asarray(
        [
            (
                str(original[name]).casefold() != str(edited[name]).casefold()
                if space.spec(name).kind == "categorical"
                else abs(float(original[name]) - float(edited[name])) > 1e-9
            )
            for name in space.names
        ],
        dtype=bool,
    )
    return delta, changed


def score_action(
    setting: Setting,
    observed_delta: np.ndarray,
    observed_changed: np.ndarray,
    predicted_delta: np.ndarray,
    predicted_changed: np.ndarray,
) -> ActionPrediction:
    feature_loss = np.abs(observed_delta - predicted_delta)
    true_positive = int(np.sum(observed_changed & predicted_changed))
    denominator = int(np.sum(observed_changed)) + int(np.sum(predicted_changed))
    feature_f1 = 1.0 if denominator == 0 else 2.0 * true_positive / denominator
    comparable = (
        observed_changed
        & predicted_changed
        & (np.abs(observed_delta) > EPSILON)
        & (np.abs(predicted_delta) > EPSILON)
    )
    direction_accuracy = None
    if np.any(comparable):
        direction_accuracy = float(
            np.mean(np.sign(observed_delta[comparable]) == np.sign(predicted_delta[comparable]))
        )
    return ActionPrediction(
        setting=setting,
        predicted_delta=predicted_delta,
        predicted_changed=predicted_changed,
        feature_loss=feature_loss,
        normalized_l1_loss=float(np.mean(feature_loss)),
        feature_f1=float(feature_f1),
        direction_accuracy=direction_accuracy,
        sparsity_difference=float(abs(np.sum(observed_changed) - np.sum(predicted_changed))),
    )


def settings_for(strategy: str, feature_count: int) -> list[Setting]:
    if strategy == "change_attributes_not_matching_target_profile":
        return [Setting()]
    if strategy == "change_largest_target_profile_mismatches":
        return [Setting(k=value) for value in range(1, feature_count)]
    if strategy == "change_most_influential_attributes_by_fixed_amount":
        return [
            Setting(k=k, fixed_amount=amount)
            for k in range(1, feature_count + 1)
            for amount in FIXED_AMOUNTS
        ]
    return [Setting(k=value) for value in range(1, feature_count + 1)]


def call_strategy(
    strategies: CounterfactualStrategies,
    strategy: str,
    profile: dict[str, Any],
    target_label: Any,
    setting: Setting,
):
    if strategy == "change_most_contributing_attributes":
        return strategies.change_most_contributing_attributes(
            profile, target_label, max_changes=setting.k
        )
    if strategy == "change_most_influential_attributes_to_flip":
        return strategies.change_most_influential_attributes_to_flip(
            profile, target_label, max_changes=setting.k
        )
    if strategy == "change_most_influential_attributes_by_fixed_amount":
        return strategies.change_most_influential_attributes_by_fixed_amount(
            profile,
            target_label,
            max_changes=setting.k,
            fixed_amount=setting.fixed_amount,
        )
    if strategy == "set_attributes_to_target_profile_values":
        return strategies.set_attributes_to_target_profile_values(
            profile, target_label, max_changes=setting.k
        )
    if strategy == "change_attributes_not_matching_target_profile":
        return strategies.change_attributes_not_matching_target_profile(
            profile, target_label
        )
    if strategy == "change_largest_target_profile_mismatches":
        return strategies.change_largest_target_profile_mismatches(
            profile, target_label, max_changes=setting.k
        )
    if strategy == "change_toward_remembered_exemplar":
        return strategies.change_toward_remembered_exemplar(
            profile, target_label, max_changes=setting.k
        )
    if strategy == "copy_changes_from_remembered_example":
        return strategies.copy_changes_from_remembered_example(
            profile, target_label, max_changes=setting.k
        )
    raise ValueError(f"Unknown strategy {strategy!r}")


def setting_sort_key(setting: Setting) -> tuple[float, float]:
    return (
        float(setting.k if setting.k is not None else 0),
        float(setting.fixed_amount if setting.fixed_amount is not None else 0),
    )


def fit_strategy_family(
    strategy: str,
    strategies: CounterfactualStrategies,
    model: PerfectMemoryCognitiveModel,
    trials: list[tuple[dict[str, str], dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    settings = settings_for(strategy, len(model.feature_space.features))
    predictions: dict[Setting, list[ActionPrediction]] = {}

    for setting in settings:
        setting_predictions: list[ActionPrediction] = []
        available = True
        for row, case in trials:
            profile = profile_from_case(case)
            target_label = row["target label"]
            if target_label not in model.class_labels:
                raise ValueError(f"Unknown target label {target_label!r}")
            try:
                proposal = call_strategy(
                    strategies, strategy, profile, target_label, setting
                )
            except ValueError:
                available = False
                break
            observed_delta, observed_changed = observed_action(row)
            predicted_delta, predicted_changed = predicted_action(
                model.feature_space, profile, proposal.edited
            )
            setting_predictions.append(
                score_action(
                    setting,
                    observed_delta,
                    observed_changed,
                    predicted_delta,
                    predicted_changed,
                )
            )
        if available:
            predictions[setting] = setting_predictions

    if not predictions:
        return None

    ordered_settings = sorted(predictions, key=setting_sort_key)
    n_trials = len(trials)
    loss_matrix = np.asarray(
        [
            [prediction.normalized_l1_loss for prediction in predictions[setting]]
            for setting in ordered_settings
        ],
        dtype=float,
    )
    full_means = np.mean(loss_matrix, axis=1)
    full_best_index = min(
        range(len(ordered_settings)),
        key=lambda index: (full_means[index], setting_sort_key(ordered_settings[index])),
    )
    full_best = ordered_settings[full_best_index]

    cv_selected: list[ActionPrediction] = []
    cv_settings: list[Setting] = []
    row_sums = np.sum(loss_matrix, axis=1)
    for held_out in range(n_trials):
        train_means = (row_sums - loss_matrix[:, held_out]) / (n_trials - 1)
        selected_index = min(
            range(len(ordered_settings)),
            key=lambda index: (
                train_means[index],
                setting_sort_key(ordered_settings[index]),
            ),
        )
        selected_setting = ordered_settings[selected_index]
        cv_settings.append(selected_setting)
        cv_selected.append(predictions[selected_setting][held_out])

    cv_losses = np.asarray(
        [prediction.normalized_l1_loss for prediction in cv_selected], dtype=float
    )
    directions = [
        prediction.direction_accuracy
        for prediction in cv_selected
        if prediction.direction_accuracy is not None
    ]
    result = {
        "strategy": strategy,
        "implied_mental_model": STRATEGY_TO_MODEL[strategy],
        "cv_mean_normalized_l1": float(np.mean(cv_losses)),
        "cv_total_normalized_l1": float(np.sum(cv_losses)),
        "cv_sd_normalized_l1": float(np.std(cv_losses, ddof=1)),
        "cv_mean_feature_f1": float(
            np.mean([prediction.feature_f1 for prediction in cv_selected])
        ),
        "cv_mean_direction_accuracy": (
            float(np.mean(directions)) if directions else math.nan
        ),
        "cv_mean_sparsity_difference": float(
            np.mean([prediction.sparsity_difference for prediction in cv_selected])
        ),
        "full_data_best_k": full_best.k,
        "full_data_best_fixed_amount": full_best.fixed_amount,
        "full_data_mean_normalized_l1": float(full_means[full_best_index]),
        "cv_loss_vector": cv_losses,
    }
    trial_output: list[dict[str, Any]] = []
    for trial_index, ((row, case), prediction, selected_setting) in enumerate(
        zip(trials, cv_selected, cv_settings), start=1
    ):
        observed_delta, observed_changed = observed_action(row)
        trial_output.append(
            {
                "trial_order": trial_index,
                "instance_id": row["instance id"],
                "target_label": row["target label"],
                "strategy": strategy,
                "implied_mental_model": STRATEGY_TO_MODEL[strategy],
                "cv_selected_k": selected_setting.k,
                "cv_selected_fixed_amount": selected_setting.fixed_amount,
                "normalized_l1_edit_loss": prediction.normalized_l1_loss,
                "feature_selection_f1": prediction.feature_f1,
                "direction_accuracy": prediction.direction_accuracy,
                "sparsity_difference": prediction.sparsity_difference,
                "observed_delta": compact_json(
                    dict(zip(case["raw_feature_names"], observed_delta.tolist()))
                ),
                "predicted_delta": compact_json(
                    dict(zip(case["raw_feature_names"], prediction.predicted_delta.tolist()))
                ),
                "observed_changed_features": " / ".join(
                    name
                    for name, changed in zip(case["raw_feature_names"], observed_changed)
                    if changed
                ),
                "predicted_changed_features": " / ".join(
                    name
                    for name, changed in zip(
                        case["raw_feature_names"], prediction.predicted_changed
                    )
                    if changed
                ),
            }
        )
    return result, trial_output


def mental_models_for_strategies(strategy_names: Iterable[str]) -> list[str]:
    models: set[str] = set()
    for strategy in strategy_names:
        implied = STRATEGY_TO_MODEL[strategy]
        if implied == "shared change memory":
            models.update(MODEL_ORDER)
        else:
            models.add(implied)
    return [model for model in MODEL_ORDER if model in models]


def fit_participant(
    participant: str,
    rows: list[dict[str, str]],
    dataset: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    training_rows = [row for row in rows if row["phase"] == "training"]
    testing_rows = sorted(
        [row for row in rows if row["phase"] == "testing"],
        key=lambda row: int(row.get("case") or row["trial number"]),
    )
    training_cases, testing_cases = case_maps(dataset)
    model = build_participant_model(training_rows, training_cases)
    strategies = CounterfactualStrategies(model, evaluate_predictions=False)
    trials = [(row, testing_cases[row["instance id"]]) for row in testing_rows]

    family_results: list[dict[str, Any]] = []
    family_trials: list[dict[str, Any]] = []
    for strategy in STRATEGY_ORDER:
        fitted = fit_strategy_family(strategy, strategies, model, trials)
        if fitted is None:
            continue
        result, trial_predictions = fitted
        family_results.append(result)
        for trial_prediction in trial_predictions:
            trial_prediction.update(
                {
                    "participant": participant,
                    "domain": testing_rows[0]["domain"],
                    "xai": testing_rows[0]["xai"],
                }
            )
        family_trials.extend(trial_predictions)

    family_results.sort(
        key=lambda result: (
            result["cv_mean_normalized_l1"],
            STRATEGY_ORDER.index(result["strategy"]),
        )
    )
    minimum_loss = family_results[0]
    best_losses = minimum_loss["cv_loss_vector"]
    supported: list[str] = []
    support_margins: dict[str, float] = {}
    for result in family_results:
        differences = result["cv_loss_vector"] - best_losses
        standard_error = (
            float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
            if len(differences) > 1
            else 0.0
        )
        support_margins[result["strategy"]] = standard_error
        if result["cv_mean_normalized_l1"] <= (
            minimum_loss["cv_mean_normalized_l1"] + standard_error + EPSILON
        ):
            supported.append(result["strategy"])

    result_by_strategy = {result["strategy"]: result for result in family_results}
    selected_name = min(
        supported,
        key=lambda name: (
            STRATEGY_PARAMETER_COUNT[name],
            result_by_strategy[name]["cv_mean_normalized_l1"],
            STRATEGY_ORDER.index(name),
        ),
    )
    selected = result_by_strategy[selected_name]
    supported_models = mental_models_for_strategies(supported)
    selected_implied = selected["implied_mental_model"]
    if selected_implied == "shared change memory":
        selected_implied = "undetermined: shared change memory"
    second_gap = (
        family_results[1]["cv_mean_normalized_l1"]
        - minimum_loss["cv_mean_normalized_l1"]
    )
    if len(supported) == 1:
        conclusion = f"one-SE simplest: {selected_name} (only supported strategy)"
    else:
        conclusion = (
            f"one-SE simplest: {selected_name}; supported: "
            + " / ".join(supported)
        )

    participant_result: dict[str, Any] = {
        "participant": participant,
        "domain": testing_rows[0]["domain"],
        "xai": testing_rows[0]["xai"],
        "n_counterfactual_responses": len(testing_rows),
        "best_counterfactual_strategy": selected_name,
        "minimum_loss_counterfactual_strategy": minimum_loss["strategy"],
        "strategy_selection_rule": "fewest fitted parameters within paired one-SE support set; CV loss breaks complexity ties",
        "selected_strategy_fitted_parameter_count": STRATEGY_PARAMETER_COUNT[
            selected_name
        ],
        "supported_counterfactual_strategies_one_se": " / ".join(supported),
        "counterfactual_strategy_fit_conclusion": conclusion,
        "counterfactual_implied_mental_model": selected_implied,
        "supported_counterfactual_implied_mental_models": " / ".join(
            supported_models
        ),
        "best_strategy_cv_mean_normalized_l1": selected["cv_mean_normalized_l1"],
        "best_strategy_cv_feature_selection_f1": selected["cv_mean_feature_f1"],
        "best_strategy_cv_direction_accuracy": selected["cv_mean_direction_accuracy"],
        "best_strategy_cv_mean_sparsity_difference": selected[
            "cv_mean_sparsity_difference"
        ],
        "minimum_loss_strategy_cv_mean_normalized_l1": minimum_loss[
            "cv_mean_normalized_l1"
        ],
        "counterfactual_strategy_gap_to_second_best": second_gap,
        "best_strategy_full_data_k": selected["full_data_best_k"],
        "best_strategy_full_data_fixed_amount": selected[
            "full_data_best_fixed_amount"
        ],
        "available_strategy_count": len(family_results),
        "training_responses_used_for_assignment": 0,
    }
    for result in family_results:
        name = result["strategy"]
        participant_result[f"{name}__cv_mean_normalized_l1"] = result[
            "cv_mean_normalized_l1"
        ]
        participant_result[f"{name}__cv_feature_f1"] = result["cv_mean_feature_f1"]
        participant_result[f"{name}__cv_direction_accuracy"] = result[
            "cv_mean_direction_accuracy"
        ]
        participant_result[f"{name}__full_data_k"] = result["full_data_best_k"]
        participant_result[f"{name}__full_data_fixed_amount"] = result[
            "full_data_best_fixed_amount"
        ]
        participant_result[f"{name}__one_se_margin_from_best"] = support_margins[name]
        result.pop("cv_loss_vector")
        result.update(
            {
                "participant": participant,
                "domain": testing_rows[0]["domain"],
                "xai": testing_rows[0]["xai"],
                "strict_best": int(name == minimum_loss["strategy"]),
                "supported_one_se": int(name in supported),
                "selected_one_se_simplest": int(name == selected_name),
                "fitted_parameter_count": STRATEGY_PARAMETER_COUNT[name],
            }
        )
    return participant_result, family_trials, family_results


def make_summary(family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_specs: list[tuple[str, str]] = [("all", "all")]
    domains = sorted({row["domain"] for row in family_rows})
    conditions = sorted({row["xai"] for row in family_rows})
    group_specs.extend((domain, "all") for domain in domains)
    group_specs.extend(("all", xai) for xai in conditions)
    group_specs.extend(
        (domain, xai)
        for domain in domains
        for xai in conditions
        if any(row["domain"] == domain and row["xai"] == xai for row in family_rows)
    )
    summary: list[dict[str, Any]] = []
    for domain, xai in group_specs:
        group = [
            row
            for row in family_rows
            if (domain == "all" or row["domain"] == domain)
            and (xai == "all" or row["xai"] == xai)
        ]
        participants = len({row["participant"] for row in group})
        for strategy in STRATEGY_ORDER:
            rows = [row for row in group if row["strategy"] == strategy]
            if not rows:
                continue
            summary.append(
                {
                    "domain": domain,
                    "xai": xai,
                    "strategy": strategy,
                    "implied_mental_model": STRATEGY_TO_MODEL[strategy],
                    "n_participants_with_strategy_available": len(rows),
                    "n_participants_in_group": participants,
                    "strict_best_count": sum(row["strict_best"] for row in rows),
                    "selected_one_se_simplest_count": sum(
                        row["selected_one_se_simplest"] for row in rows
                    ),
                    "supported_one_se_count": sum(
                        row["supported_one_se"] for row in rows
                    ),
                    "mean_cv_normalized_l1": float(
                        np.mean([row["cv_mean_normalized_l1"] for row in rows])
                    ),
                    "mean_cv_feature_f1": float(
                        np.mean([row["cv_mean_feature_f1"] for row in rows])
                    ),
                    "mean_cv_direction_accuracy": float(
                        np.nanmean(
                            [row["cv_mean_direction_accuracy"] for row in rows]
                        )
                    ),
                }
            )
    return summary


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def make_report(
    participants: list[dict[str, Any]], summary: list[dict[str, Any]]
) -> str:
    strict_counts = Counter(
        row["minimum_loss_counterfactual_strategy"] for row in participants
    )
    selected_counts = Counter(
        row["best_counterfactual_strategy"] for row in participants
    )
    mental_counts = Counter(
        row["counterfactual_implied_mental_model"] for row in participants
    )
    support_sizes = Counter(
        len(row["supported_counterfactual_strategies_one_se"].split(" / "))
        for row in participants
    )
    overall = {
        row["strategy"]: row
        for row in summary
        if row["domain"] == "all" and row["xai"] == "all"
    }
    strategy_rows = []
    for strategy in STRATEGY_ORDER:
        if strategy not in overall:
            continue
        row = overall[strategy]
        strategy_rows.append(
            [
                strategy,
                STRATEGY_TO_MODEL[strategy],
                row["n_participants_with_strategy_available"],
                strict_counts[strategy],
                selected_counts[strategy],
                row["supported_one_se_count"],
                f"{row['mean_cv_normalized_l1']:.3f}",
                f"{row['mean_cv_feature_f1']:.3f}",
            ]
        )
    group_rows = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for participant in participants:
        groups[(participant["domain"], participant["xai"])].append(participant)
    for (domain, xai), rows in sorted(groups.items()):
        counts = Counter(row["counterfactual_implied_mental_model"] for row in rows)
        unique_strategy = sum(
            len(row["supported_counterfactual_strategies_one_se"].split(" / ")) == 1
            for row in rows
        )
        group_rows.append(
            [
                domain,
                xai,
                len(rows),
                counts["exemplar"],
                counts["attribution"],
                counts["prototype"],
                counts["undetermined: shared change memory"],
                unique_strategy,
            ]
        )
    return "\n".join(
        [
            "# v0.9 counterfactual-strategy fit",
            "",
            "## Method",
            "",
            (
                "The training phase is used only to construct the information "
                "available to each participant under their assigned explanation. "
                "Training responses are not used to select a mental model. The "
                "mental model and strategy are inferred jointly from testing-phase "
                "counterfactual edits."
            ),
            "",
            (
                "For each testing instance, the observed and predicted actions are "
                "five-dimensional signed edit vectors in the experiment's normalized "
                "feature ranges. Primary loss is their mean absolute difference "
                "(normalized L1 divided by five), which jointly scores feature "
                "selection, direction, and magnitude. Feature-selection F1 and "
                "direction accuracy are auxiliary diagnostics."
            ),
            "",
            (
                "The number of selected features k and the fixed normalized amount "
                "are chosen by leave-one-instance-out cross-validation. The minimum-"
                "loss strategy is retained, then the reported assignment is the "
                "strategy with the fewest participant-fitted parameters inside the "
                "paired one-standard-error support set. CV loss breaks ties between "
                "strategies with equal parameter count."
            ),
            "",
            "## Overall assignments",
            "",
            markdown_table(
                [
                    "Strategy",
                    "Implied representation",
                    "Available N",
                    "Minimum loss",
                    "Selected: simplest within 1 SE",
                    "Supported (1 SE)",
                    "Mean CV loss",
                    "Mean feature F1",
                ],
                strategy_rows,
            ),
            "",
            "Selected implied mental-model counts: "
            + ", ".join(f"{name}={count}" for name, count in sorted(mental_counts.items())),
            "",
            "Support-set sizes: "
            + ", ".join(
                f"{size} strategies={count} participants"
                for size, count in sorted(support_sizes.items())
            ),
            "",
            "## Selected implied mental model by domain and explanation",
            "",
            markdown_table(
                [
                    "Domain",
                    "XAI",
                    "N",
                    "Exemplar",
                    "Attribution",
                    "Prototype",
                    "Shared/undetermined",
                    "One supported strategy",
                ],
                group_rows,
            ),
            "",
            "## Interpretation limits",
            "",
            (
                "A strategy assignment is a conservative computational description, "
                "not proof of a participant's verbal reasoning. The copy-change "
                "strategy uses shared episodic change memory and therefore does not "
                "uniquely identify exemplar, attribution, or prototype forward "
                "representation. The one-SE rule resolves supported ties by a "
                "predeclared fitted-parameter count rather than by the earlier "
                "forward-simulation fit."
            ),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--participants", type=Path, default=DEFAULT_PARTICIPANTS)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    datasets = load_bundle(args.bundle)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["participant"]].append(row)

    participant_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for participant in sorted(grouped):
        participant_data = grouped[participant]
        domains = {row["domain"] for row in participant_data}
        if len(domains) != 1:
            raise ValueError(f"Participant {participant} occurs in multiple domains")
        domain = next(iter(domains))
        fitted, trials, families = fit_participant(
            participant, participant_data, datasets[domain]
        )
        participant_rows.append(fitted)
        trial_rows.extend(trials)
        family_rows.extend(families)

    participant_rows.sort(key=lambda row: (row["domain"], row["xai"], row["participant"]))
    trial_rows.sort(
        key=lambda row: (
            row["domain"],
            row["xai"],
            row["participant"],
            int(row["trial_order"]),
            STRATEGY_ORDER.index(row["strategy"]),
        )
    )
    family_rows.sort(
        key=lambda row: (
            row["domain"],
            row["xai"],
            row["participant"],
            STRATEGY_ORDER.index(row["strategy"]),
        )
    )
    summary = make_summary(family_rows)

    all_participant_fields: list[str] = []
    for row in participant_rows:
        for field in row:
            if field not in all_participant_fields:
                all_participant_fields.append(field)
    write_csv(args.participants, participant_rows, all_participant_fields)
    write_csv(args.trials, trial_rows, trial_rows[0].keys())
    write_csv(args.summary, summary, summary[0].keys())
    args.report.write_text(make_report(participant_rows, summary), encoding="utf-8")
    print(
        f"participants={len(participant_rows)} strategy_trials={len(trial_rows)} "
        f"family_fits={len(family_rows)}"
    )
    print(f"participant_fits={args.participants}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
