"""Fit counterfactual attribute selection and edit amount independently.

Selection methods are compared with leave-one-testing-case-out feature-set F1.
Amount methods are compared with normalized absolute-error only on attributes
the participant actually changed, so choosing the wrong attribute cannot be
compensated for by predicting a convenient change magnitude.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fit_v09_counterfactual_strategies import (  # noqa: E402
    CounterfactualStrategies,
    build_participant_model,
    case_maps,
    load_bundle,
    normalized_feature_value,
    observed_action,
    predicted_action,
    profile_from_case,
)


SELECTION_METHODS = (
    "Current-case contribution",
    "Global importance",
    "Target-profile mismatch",
    "Nearest target example",
    "Similar remembered edit",
)
AMOUNT_METHODS = (
    "One fixed normalized amount",
    "Feature-specific fixed amount",
    "Target-specific endpoint",
    "Estimated amount needed to flip",
    "Target-profile centre",
    "Nearest target-profile range edge",
    "Nearest target-example value",
    "Remembered edit amount",
)
AMOUNT_FITTED_PARAMETER_COUNT = {
    "One fixed normalized amount": 1,
    "Feature-specific fixed amount": 5,
    "Target-specific endpoint": 10,
    "Estimated amount needed to flip": 0,
    "Target-profile centre": 0,
    "Nearest target-profile range edge": 0,
    "Nearest target-example value": 0,
    "Remembered edit amount": 0,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def set_f1(observed: set[str], predicted: set[str]) -> float:
    denominator = len(observed) + len(predicted)
    return 1.0 if denominator == 0 else 2.0 * len(observed & predicted) / denominator


def proposal_set(proposal) -> set[str]:
    return set(proposal.changed_features)


def selection_ranking(
    model,
    profile: dict[str, Any],
    target: str,
    method: str,
) -> list[str] | None:
    if method == "Current-case contribution":
        contributions = model.attribution_contributions(profile)
        target_sign = model.class_sign(target)
        return sorted(
            contributions,
            key=lambda name: -target_sign * contributions[name],
            reverse=True,
        )
    if method == "Global importance":
        importance = model.global_importance()
        return sorted(importance, key=importance.get, reverse=True)
    if method == "Target-profile mismatch":
        mismatch = model.prototype_mismatches(profile, target)
        return sorted(mismatch, key=mismatch.get, reverse=True)
    if method == "Nearest target example":
        encoded = model.feature_space.encode(profile)
        candidates = [instance for instance in model.instances if instance.label == target]
        if not candidates:
            return None
        closest = min(
            candidates,
            key=lambda instance: model.feature_space.distance(encoded, instance.encoded),
        )
        mismatch = model.feature_space.per_feature_distance(encoded, closest.encoded)
        return sorted(mismatch, key=mismatch.get, reverse=True)
    if method == "Similar remembered edit":
        candidates = [change for change in model.changes if change.target_label == target]
        if not candidates:
            return None
        encoded = model.feature_space.encode(profile)
        closest = min(
            candidates,
            key=lambda change: model.feature_space.distance(
                encoded, change.original_encoded
            ),
        )
        amount = model.feature_space.per_feature_distance(
            closest.original_encoded, closest.edited_encoded
        )
        return sorted(amount, key=amount.get, reverse=True)
    raise ValueError(method)


def average_precision(ranking: list[str], observed: set[str]) -> float:
    if not observed:
        return 1.0
    hits = 0
    precision_sum = 0.0
    for rank, name in enumerate(ranking, start=1):
        if name in observed:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(observed)


def cv_selection_fit(
    observed: list[set[str]],
    predictions: dict[str, dict[int, list[set[str]]]],
) -> dict[str, dict[str, float | int]]:
    fitted: dict[str, dict[str, float | int]] = {}
    for method, by_k in predictions.items():
        if not by_k:
            continue
        cv_scores: list[float] = []
        selected_k: list[int] = []
        trial_count = len(observed)
        for held in range(trial_count):
            training = [index for index in range(trial_count) if index != held]
            k_scores = {
                k: float(np.mean([set_f1(observed[index], values[index]) for index in training]))
                for k, values in by_k.items()
            }
            best_k = min(k_scores, key=lambda k: (-k_scores[k], k))
            selected_k.append(best_k)
            cv_scores.append(set_f1(observed[held], by_k[best_k][held]))
        full_scores = {
            k: float(np.mean([set_f1(observed[index], values[index]) for index in range(trial_count)]))
            for k, values in by_k.items()
        }
        full_k = min(full_scores, key=lambda k: (-full_scores[k], k))
        fitted[method] = {
            "cv_f1": float(np.mean(cv_scores)),
            "full_k": full_k,
            "median_cv_k": float(median(selected_k)),
        }
    return fitted


def raw_amount_vector(model, original: dict[str, Any], edited: dict[str, Any]) -> dict[str, float]:
    delta, _ = predicted_action(model.feature_space, original, edited)
    return {
        name: abs(float(delta[index]))
        for index, name in enumerate(model.feature_space.names)
    }


def fixed_amount_predictions(
    trials: list[dict[str, Any]],
    held: int,
    *,
    by_feature: bool,
) -> dict[str, float]:
    training_values: list[float] = []
    feature_values: dict[str, list[float]] = defaultdict(list)
    for index, trial in enumerate(trials):
        if index == held:
            continue
        for name, amount in trial["observed_amounts"].items():
            training_values.append(amount)
            feature_values[name].append(amount)
    fallback = float(median(training_values))
    return {
        name: (
            float(median(feature_values[name]))
            if by_feature and feature_values[name]
            else fallback
        )
        for name in trials[held]["observed_amounts"]
    }


def endpoint_predictions(trials: list[dict[str, Any]], held: int) -> dict[str, float]:
    target = trials[held]["target"]
    training_endpoints: dict[tuple[str, str], list[float]] = defaultdict(list)
    feature_endpoints: dict[str, list[float]] = defaultdict(list)
    all_endpoints: list[float] = []
    for index, trial in enumerate(trials):
        if index == held:
            continue
        for name, endpoint in trial["observed_endpoints"].items():
            training_endpoints[(trial["target"], name)].append(endpoint)
            feature_endpoints[name].append(endpoint)
            all_endpoints.append(endpoint)
    result = {}
    for name in trials[held]["observed_amounts"]:
        values = training_endpoints[(target, name)] or feature_endpoints[name] or all_endpoints
        endpoint = float(median(values))
        result[name] = abs(endpoint - trials[held]["original_normalized"][name])
    return result


def cognitive_amount_predictions(
    trial: dict[str, Any],
    method: str,
) -> dict[str, float] | None:
    model = trial["model"]
    strategies = trial["strategies"]
    profile = trial["profile"]
    target = trial["target"]
    observed_names = list(trial["observed_amounts"])
    if method == "Estimated amount needed to flip":
        importance = model.global_importance()
        order = sorted(observed_names, key=lambda name: importance[name], reverse=True)
        edited = strategies._attribution_flip(
            profile, target, order, max_changes=len(order)
        )
    elif method == "Target-profile centre":
        edited = dict(profile)
        centre = model.prototype_profile(target)
        for name in observed_names:
            edited[name] = centre[name]
    elif method == "Nearest target-profile range edge":
        edited = dict(profile)
        ranges = model.prototype_ranges(target)
        centre = model.prototype_profile(target)
        for name in observed_names:
            spec = model.feature_space.spec(name)
            if spec.kind == "numerical":
                low, high = ranges[name]
                edited[name] = float(np.clip(float(profile[name]), low, high))
            else:
                edited[name] = centre[name]
    elif method == "Nearest target-example value":
        try:
            proposal = strategies.change_toward_remembered_exemplar(
                profile, target, max_changes=len(model.feature_space.names)
            )
        except ValueError:
            return None
        edited = dict(profile)
        for name in observed_names:
            edited[name] = proposal.edited.get(name, profile[name])
    elif method == "Remembered edit amount":
        try:
            proposal = strategies.copy_changes_from_remembered_example(
                profile, target, max_changes=len(model.feature_space.names)
            )
        except ValueError:
            return None
        edited = dict(profile)
        for name in observed_names:
            edited[name] = proposal.edited.get(name, profile[name])
    else:
        raise ValueError(method)
    amounts = raw_amount_vector(model, profile, edited)
    return {name: amounts[name] for name in observed_names}


def amount_mae(observed: dict[str, float], predicted: dict[str, float]) -> float:
    return float(np.mean([abs(observed[name] - predicted[name]) for name in observed]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "static" / "experiment-data.json")
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.input)
    datasets = load_bundle(args.bundle)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["participant"]].append(row)

    output_rows: list[dict[str, Any]] = []
    for participant in sorted(grouped):
        participant_rows = grouped[participant]
        domain = participant_rows[0]["domain"]
        xai = participant_rows[0]["xai"]
        training_cases, testing_cases = case_maps(datasets[domain])
        training_rows = [row for row in participant_rows if row["phase"] == "training"]
        testing_rows = sorted(
            [row for row in participant_rows if row["phase"] == "testing"],
            key=lambda row: int(row["case"]),
        )
        model = build_participant_model(training_rows, training_cases)
        strategies = CounterfactualStrategies(model, evaluate_predictions=False)
        names = list(model.feature_space.names)

        trial_data: list[dict[str, Any]] = []
        selection_scores: dict[str, list[float]] = defaultdict(list)
        selection_ap: dict[str, list[float]] = defaultdict(list)
        unavailable_selection: set[str] = set()

        for row in testing_rows:
            case = testing_cases[row["instance id"]]
            profile = profile_from_case(case)
            target = row["target label"]
            observed_delta, observed_changed = observed_action(row)
            observed = {name for name, changed in zip(names, observed_changed) if changed}
            original_normalized = {
                name: normalized_feature_value(model.feature_space, name, profile[name])
                for name in names
            }
            observed_amounts = {
                name: abs(float(observed_delta[names.index(name)])) for name in observed
            }
            observed_endpoints = {
                name: original_normalized[name] + float(observed_delta[names.index(name)])
                for name in observed
            }
            trial_data.append(
                {
                    "model": model,
                    "strategies": strategies,
                    "profile": profile,
                    "target": target,
                    "observed_amounts": observed_amounts,
                    "observed_endpoints": observed_endpoints,
                    "original_normalized": original_normalized,
                }
            )
            for method in SELECTION_METHODS:
                ranking = selection_ranking(model, profile, target, method)
                if ranking is None:
                    unavailable_selection.add(method)
                else:
                    predicted = set(ranking[: len(observed)])
                    selection_scores[method].append(set_f1(observed, predicted))
                    selection_ap[method].append(average_precision(ranking, observed))

        for method in unavailable_selection:
            selection_scores.pop(method, None)
            selection_ap.pop(method, None)
        selection_fit = {
            method: {
                "ranking_score": float(np.mean(scores)),
                "average_precision": float(np.mean(selection_ap[method])),
            }
            for method, scores in selection_scores.items()
            if len(scores) == len(testing_rows)
        }
        selection_order = sorted(
            selection_fit,
            key=lambda method: (
                -float(selection_fit[method]["ranking_score"]),
                -float(selection_fit[method]["average_precision"]),
                method,
            ),
        )
        raw_best_selection = selection_order[0]
        selection_gap = (
            float(selection_fit[selection_order[0]]["ranking_score"])
            - float(selection_fit[selection_order[1]]["ranking_score"])
        )
        selection_score = float(selection_fit[raw_best_selection]["ranking_score"])
        identified_selection = (
            raw_best_selection
            if selection_score >= 0.50 and selection_gap >= 0.05
            else "Unclear or overlapping"
        )
        best_supported_selection = (
            raw_best_selection if selection_score >= 0.50 else "Unclear"
        )

        amount_scores: dict[str, list[float]] = defaultdict(list)
        for held, trial in enumerate(trial_data):
            fixed = fixed_amount_predictions(trial_data, held, by_feature=False)
            amount_scores["One fixed normalized amount"].append(
                amount_mae(trial["observed_amounts"], fixed)
            )
            feature_fixed = fixed_amount_predictions(trial_data, held, by_feature=True)
            amount_scores["Feature-specific fixed amount"].append(
                amount_mae(trial["observed_amounts"], feature_fixed)
            )
            endpoint = endpoint_predictions(trial_data, held)
            amount_scores["Target-specific endpoint"].append(
                amount_mae(trial["observed_amounts"], endpoint)
            )
            for method in AMOUNT_METHODS[3:]:
                predicted = cognitive_amount_predictions(trial, method)
                if predicted is not None:
                    amount_scores[method].append(
                        amount_mae(trial["observed_amounts"], predicted)
                    )

        complete_amount_scores = {
            method: float(np.mean(scores))
            for method, scores in amount_scores.items()
            if len(scores) == len(trial_data)
        }
        amount_order = sorted(
            complete_amount_scores,
            key=lambda method: (complete_amount_scores[method], method),
        )
        best_amount = amount_order[0]
        amount_gap = (
            complete_amount_scores[amount_order[1]]
            - complete_amount_scores[amount_order[0]]
        )
        one_se_supported: list[str] = []
        minimum_losses = np.asarray(amount_scores[best_amount], dtype=float)
        for method in amount_order:
            candidate_losses = np.asarray(amount_scores[method], dtype=float)
            paired_difference = candidate_losses - minimum_losses
            paired_se = (
                float(np.std(paired_difference, ddof=1) / math.sqrt(len(paired_difference)))
                if len(paired_difference) > 1
                else 0.0
            )
            if complete_amount_scores[method] <= complete_amount_scores[best_amount] + paired_se:
                one_se_supported.append(method)
        one_se_amount = min(
            one_se_supported,
            key=lambda method: (
                AMOUNT_FITTED_PARAMETER_COUNT[method],
                complete_amount_scores[method],
                method,
            ),
        )

        output = {
            "participant": participant,
            "domain": domain,
            "xai": xai,
            "testing_cases": len(testing_rows),
            "attribute selection method": identified_selection,
            "best-supported attribute selection method": best_supported_selection,
            "raw best attribute selection method": raw_best_selection,
            "attribute selection ranking score": selection_score,
            "attribute selection average precision": selection_fit[raw_best_selection]["average_precision"],
            "attribute selection score gap": selection_gap,
            "best independent amount method": best_amount,
            "one-SE independent amount method": one_se_amount,
            "amount normalized MAE": complete_amount_scores[best_amount],
            "change amount method": one_se_amount,
            "change amount normalized MAE": complete_amount_scores[one_se_amount],
            "amount MAE gap": amount_gap,
        }
        for method in SELECTION_METHODS:
            output[f"selection ranking score: {method}"] = (
                selection_fit[method]["ranking_score"] if method in selection_fit else ""
            )
        for method in AMOUNT_METHODS:
            output[f"amount MAE: {method}"] = complete_amount_scores.get(method, "")
        output_rows.append(output)

    write_csv(args.participants, output_rows)
    summary = {
        "participants": len(output_rows),
        "selection_measure": "mean top-m overlap/F1 where m is the participant's observed number changed; higher is better",
        "amount_measure": "mean absolute error in absolute normalized magnitude on observed changed attributes; lower is better",
        "identified_selection_counts": dict(
            sorted(Counter(row["attribute selection method"] for row in output_rows).items())
        ),
        "raw_best_selection_counts": dict(
            sorted(Counter(row["raw best attribute selection method"] for row in output_rows).items())
        ),
        "best_amount_counts": dict(
            sorted(Counter(row["best independent amount method"] for row in output_rows).items())
        ),
        "one_se_amount_counts": dict(
            sorted(Counter(row["one-SE independent amount method"] for row in output_rows).items())
        ),
        "selection_ambiguous": sum(row["attribute selection method"] == "Unclear or overlapping" for row in output_rows),
        "amount_clear_gap_at_least_0.01": dict(
            sorted(
                Counter(
                    row["best independent amount method"]
                    for row in output_rows
                    if float(row["amount MAE gap"]) >= 0.01
                ).items()
            )
        ),
        "amount_ambiguous_gap_below_0.01": sum(
            float(row["amount MAE gap"]) < 0.01 for row in output_rows
        ),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
