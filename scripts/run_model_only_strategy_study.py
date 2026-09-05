"""Run a participant-free counterfactual strategy effectiveness study.

For each experiment domain and explanation condition, a parameter-free perfect-
memory cognitive model observes only the training profiles and explanations in
the requested experiment bundle. Implemented strategy rules then edit its fixed
test profiles with maximum feature counts k=1,2,3.

No survey response, participant edit, participant strategy assignment, or
participant-fitted amount is read or used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calculate_v09_boundary_distance import (  # noqa: E402
    WachterBoundarySearch,
    model_logit,
    stable_sigmoid,
)
from scripts.fit_v09_counterfactual_strategies import (  # noqa: E402
    explanation_payload,
    profile_from_case,
)
from src.cognitive_models.minimal_mental_model import (  # noqa: E402
    CounterfactualStrategies,
    FeatureSpace,
    PerfectMemoryCognitiveModel,
)


XAI_CONDITIONS = ("none", "attribution", "counterfactual")
MAX_CHANGES = (1, 2, 3)


def most_contributing(strategies, profile, target, k):
    return strategies.change_most_contributing_attributes(
        profile, target, max_changes=k
    )


def global_flip(strategies, profile, target, k):
    return strategies.change_most_influential_attributes_to_flip(
        profile, target, max_changes=k
    )


def profile_centre(strategies, profile, target, k):
    return strategies.set_attributes_to_target_profile_values(
        profile, target, max_changes=k
    )


def profile_range(strategies, profile, target, k):
    return strategies.change_largest_target_profile_mismatches(
        profile, target, max_changes=k
    )


def nearest_example(strategies, profile, target, k):
    return strategies.change_toward_remembered_exemplar(
        profile, target, max_changes=k
    )


def remembered_edit(strategies, profile, target, k):
    return strategies.copy_changes_from_remembered_example(
        profile, target, max_changes=k
    )


STRATEGIES: tuple[dict[str, Any], ...] = (
    {
        "strategy": "Current-case contribution + estimated flip amount",
        "selection_method": "Current-case contribution",
        "amount_method": "Estimated amount needed to flip the cognitive model",
        "call": most_contributing,
    },
    {
        "strategy": "Global importance + estimated flip amount",
        "selection_method": "Global importance",
        "amount_method": "Estimated amount needed to flip the cognitive model",
        "call": global_flip,
    },
    {
        "strategy": "Target-profile mismatch + target centre",
        "selection_method": "Target-profile mismatch",
        "amount_method": "Target-profile centre",
        "call": profile_centre,
    },
    {
        "strategy": "Target-profile mismatch + nearest range edge",
        "selection_method": "Target-profile mismatch",
        "amount_method": "Nearest target-profile range edge",
        "call": profile_range,
    },
    {
        "strategy": "Nearest target example + example values",
        "selection_method": "Nearest target example",
        "amount_method": "Nearest target-example value",
        "call": nearest_example,
    },
    {
        "strategy": "Similar remembered edit + copied amounts",
        "selection_method": "Similar remembered edit",
        "amount_method": "Remembered edit amount",
        "call": remembered_edit,
    },
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalized_feature(
    value: Any, kind: str, feature_range: list[Any], *, clip: bool
) -> float:
    if kind == "categorical":
        normalized = str(value).casefold()
        categories = [str(item).casefold() for item in feature_range]
        index = categories.index(normalized)
        return index / (len(categories) - 1) if len(categories) > 1 else 0.0
    low, high = float(feature_range[0]), float(feature_range[1])
    result = 0.0 if math.isclose(low, high) else (float(value) - low) / (high - low)
    return float(np.clip(result, 0.0, 1.0)) if clip else float(result)


def normalized_profile(
    case: dict[str, Any], values: list[Any], *, clip: bool = True
) -> np.ndarray:
    return np.asarray(
        [
            normalized_feature(value, kind, feature_range, clip=clip)
            for value, kind, feature_range in zip(
                values, case["feature_types"], case["raw_feature_ranges"]
            )
        ],
        dtype=float,
    )


def build_cognitive_model(
    dataset: dict[str, Any], xai: str
) -> PerfectMemoryCognitiveModel:
    training = dataset["training_pool"]
    first = training[0]
    space = FeatureSpace.from_dataset_fields(
        first["raw_feature_names"],
        first["feature_types"],
        first["raw_feature_ranges"],
    )
    model = PerfectMemoryCognitiveModel(space, first["prediction_labels"])
    for case in training:
        model.observe(
            profile_from_case(case),
            case["prediction"]["label"],
            **explanation_payload(case, xai),
        )
    return model


def reference_matrices(
    domain: str, dataset: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    representative = dataset["test_pool"][0]
    feature_names = list(representative["raw_feature_names"])

    full_rows = []
    for record in read_csv(ROOT / "src" / "data" / domain / "train.csv"):
        full_rows.append(
            normalized_profile(
                representative, [record[name] for name in feature_names]
            )
        )
    subset_rows = [
        normalized_profile(case, list(case["raw_feature_values"]))
        for case in dataset["training_pool"]
    ]
    if not subset_rows:
        raise ValueError(f"Training pool is empty for {domain}")
    return np.asarray(full_rows, dtype=float), np.asarray(subset_rows, dtype=float)


def similarity(profile: np.ndarray, reference: np.ndarray) -> float:
    nearest_gower = float(np.abs(reference - profile).mean(axis=1).min())
    return float(np.clip(1.0 - nearest_gower, 0.0, 1.0))


def changed_names(
    names: list[str], kinds: list[str], original: list[Any], edited: list[Any]
) -> list[str]:
    changed: list[str] = []
    for name, kind, before, after in zip(names, kinds, original, edited):
        if kind == "categorical":
            differs = str(before).casefold() != str(after).casefold()
        else:
            differs = abs(float(after) - float(before)) > 1e-9
        if differs:
            changed.append(name)
    return changed


def mean_value(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return float(np.mean(values)) if values else ""


def median_value(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return float(np.median(values)) if values else ""


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return centre - half, centre + half


def summarize_group(
    rows: list[dict[str, Any]], aggregation: str, domain: str, xai: str
) -> dict[str, Any]:
    valid = [row for row in rows if int(row["valid counterfactual (0/1)"]) == 1]
    invalid = [row for row in rows if int(row["valid counterfactual (0/1)"]) == 0]
    successes = len(valid)
    ci_low, ci_high = wilson_interval(successes, len(rows))
    first = rows[0]
    return {
        "aggregation": aggregation,
        "domain": domain,
        "training explanation condition": xai,
        "strategy": first["strategy"],
        "attribute selection method": first["attribute selection method"],
        "change amount method": first["change amount method"],
        "maximum attributes allowed": first["maximum attributes allowed"],
        "trials": len(rows),
        "valid counterfactual rate": successes / len(rows),
        "valid rate Wilson 95% lower": ci_low,
        "valid rate Wilson 95% upper": ci_high,
        "valid and actionable rate": mean_value(
            rows, "valid and actionable (0/1)"
        ),
        "actionable rate": mean_value(rows, "actionable (0/1)"),
        "mean actual attributes changed": mean_value(
            rows, "actual attributes changed"
        ),
        "mean sparsity": mean_value(rows, "sparsity"),
        "mean target confidence": mean_value(rows, "target confidence edited"),
        "mean target confidence gain": mean_value(
            rows, "target confidence gain"
        ),
        "mean proximity": mean_value(rows, "proximity"),
        "median proximity": median_value(rows, "proximity"),
        "mean model-consistent proximity": mean_value(
            rows, "model-consistent proximity"
        ),
        "mean plausibility": mean_value(rows, "plausibility"),
        "mean plausibility subset": mean_value(rows, "plausibility (subset)"),
        "mean boundary distance": mean_value(rows, "boundary distance edited"),
        "mean boundary distance change": mean_value(
            rows, "boundary distance change (edited - original)"
        ),
        "valid trials": len(valid),
        "mean proximity valid": mean_value(valid, "proximity"),
        "mean model-consistent proximity valid": mean_value(
            valid, "model-consistent proximity"
        ),
        "mean plausibility valid": mean_value(valid, "plausibility"),
        "mean plausibility subset valid": mean_value(
            valid, "plausibility (subset)"
        ),
        "mean boundary distance valid": mean_value(
            valid, "boundary distance edited"
        ),
        "median boundary distance valid": median_value(
            valid, "boundary distance edited"
        ),
        "invalid trials": len(invalid),
        "mean remaining boundary distance invalid": mean_value(
            invalid, "boundary distance edited"
        ),
    }


def make_summary(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []

    def add_groups(
        aggregation: str,
        key_function: Callable[[dict[str, Any]], tuple[Any, ...]],
    ) -> None:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in trials:
            groups[key_function(row)].append(row)
        for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
            if aggregation == "domain_xai":
                domain, xai, _, _ = key
            elif aggregation == "domain":
                domain, _, _ = key
                xai = "all"
            elif aggregation == "xai":
                xai, _, _ = key
                domain = "all"
            else:
                _, _ = key
                domain = "all"
                xai = "all"
            summary.append(summarize_group(group, aggregation, domain, xai))

    add_groups(
        "domain_xai",
        lambda row: (
            row["domain"],
            row["training explanation condition"],
            row["strategy"],
            row["maximum attributes allowed"],
        ),
    )
    add_groups(
        "domain",
        lambda row: (
            row["domain"],
            row["strategy"],
            row["maximum attributes allowed"],
        ),
    )
    add_groups(
        "xai",
        lambda row: (
            row["training explanation condition"],
            row["strategy"],
            row["maximum attributes allowed"],
        ),
    )
    add_groups(
        "overall",
        lambda row: (row["strategy"], row["maximum attributes allowed"]),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=ROOT / "static" / "experiment-data.json"
    )
    parser.add_argument(
        "--trials",
        type=Path,
        default=ROOT / "qualtrics" / "model_only_strategy_trials_v1.0.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "qualtrics" / "model_only_strategy_summary_v1.0.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "qualtrics" / "model_only_strategy_study_v1.0.json",
    )
    parser.add_argument("--limit-cases", type=int)
    args = parser.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    trial_rows: list[dict[str, Any]] = []
    boundary_cache: dict[tuple[str, tuple[float, ...]], float] = {}
    unavailable: list[dict[str, str]] = []
    original_prediction_mismatches = 0

    for domain, dataset in bundle["datasets"].items():
        test_cases = dataset["test_pool"]
        if args.limit_cases is not None:
            test_cases = test_cases[: args.limit_cases]
        representative = dataset["test_pool"][0]
        names = list(representative["raw_feature_names"])
        full_reference, subset_reference = reference_matrices(domain, dataset)
        solver = WachterBoundarySearch(
            dataset["browser_model"],
            representative,
            allow_reference_outside_bounds=True,
        )
        controllable = set(
            representative["counterfactual_settings"][
                "raw_controllable_feature_names"
            ]
        )

        def boundary_distance(profile: np.ndarray) -> float:
            key = (domain, tuple(round(float(value), 12) for value in profile))
            if key not in boundary_cache:
                boundary_cache[key] = solver.solve(profile).distance
                if len(boundary_cache) % 100 == 0:
                    print(
                        f"boundary_profiles={len(boundary_cache)} trials={len(trial_rows)}",
                        flush=True,
                    )
            return boundary_cache[key]

        for xai in XAI_CONDITIONS:
            cognitive_model = build_cognitive_model(dataset, xai)
            strategies = CounterfactualStrategies(
                cognitive_model, evaluate_predictions=False
            )
            for case_index, case in enumerate(test_cases, start=1):
                original_raw = list(case["raw_feature_values"])
                # Model/boundary profiles retain values outside the configured
                # display range so predictions remain identical to the task model.
                # Distance/plausibility profiles use the study's bounded [0,1]
                # normalization, matching the participant-result metrics.
                original_model = normalized_profile(case, original_raw, clip=False)
                original_metric = normalized_profile(case, original_raw, clip=True)
                original_logit = model_logit(
                    dataset["browser_model"], original_model, representative
                )
                original_class = int(original_logit > 0.0)
                if original_class != int(case["prediction"]["value"]):
                    original_prediction_mismatches += 1
                target_class = 1 - original_class
                target_label = case["prediction_labels"][target_class]
                original_target_confidence = (
                    stable_sigmoid(original_logit)
                    if target_class == 1
                    else 1.0 - stable_sigmoid(original_logit)
                )
                original_boundary = boundary_distance(original_model)
                profile = profile_from_case(case)

                for strategy_spec in STRATEGIES:
                    for maximum in MAX_CHANGES:
                        try:
                            proposal = strategy_spec["call"](
                                strategies, profile, target_label, maximum
                            )
                        except ValueError as error:
                            unavailable.append(
                                {
                                    "domain": domain,
                                    "training explanation condition": xai,
                                    "strategy": strategy_spec["strategy"],
                                    "reason": str(error),
                                }
                            )
                            continue

                        edited_raw = [proposal.edited[name] for name in names]
                        edited_model = normalized_profile(case, edited_raw, clip=False)
                        edited_metric = normalized_profile(case, edited_raw, clip=True)
                        changed = changed_names(
                            names,
                            list(case["feature_types"]),
                            original_raw,
                            edited_raw,
                        )
                        edited_logit = model_logit(
                            dataset["browser_model"], edited_model, representative
                        )
                        edited_class = int(edited_logit > 0.0)
                        valid = int(edited_class == target_class)
                        edited_target_confidence = (
                            stable_sigmoid(edited_logit)
                            if target_class == 1
                            else 1.0 - stable_sigmoid(edited_logit)
                        )
                        proximity = float(
                            np.abs(edited_metric - original_metric).sum()
                        )
                        model_proximity = float(
                            np.abs(edited_model - original_model).sum()
                        )
                        edited_boundary = boundary_distance(edited_model)
                        actionable = int(set(changed).issubset(controllable))
                        trial_rows.append(
                            {
                                "domain": domain,
                                "training explanation condition": xai,
                                "test case": case_index,
                                "instance id": case["instance_id"],
                                "original label": case["prediction_labels"][
                                    original_class
                                ],
                                "target label": target_label,
                                "strategy": strategy_spec["strategy"],
                                "attribute selection method": strategy_spec[
                                    "selection_method"
                                ],
                                "change amount method": strategy_spec[
                                    "amount_method"
                                ],
                                "maximum attributes allowed": maximum,
                                "actual attributes changed": len(changed),
                                "changed attributes": " / ".join(changed),
                                "sparsity": 1.0 - len(changed) / 5.0,
                                "proximity": proximity,
                                "model-consistent proximity": model_proximity,
                                "plausibility": similarity(
                                    edited_metric, full_reference
                                ),
                                "plausibility (subset)": similarity(
                                    edited_metric, subset_reference
                                ),
                                "original target confidence": original_target_confidence,
                                "target confidence edited": edited_target_confidence,
                                "target confidence gain": edited_target_confidence
                                - original_target_confidence,
                                "valid counterfactual (0/1)": valid,
                                "actionable (0/1)": actionable,
                                "valid and actionable (0/1)": valid * actionable,
                                "boundary distance original": original_boundary,
                                "boundary distance edited": edited_boundary,
                                "boundary distance change (edited - original)": edited_boundary
                                - original_boundary,
                            }
                        )

    if original_prediction_mismatches:
        raise ValueError(
            f"Browser model disagreed with {original_prediction_mismatches} bundled predictions"
        )

    summary_rows = make_summary(trial_rows)
    write_csv(args.trials, trial_rows)
    write_csv(args.summary, summary_rows)

    unique_unavailable = sorted(
        {
            (
                row["domain"],
                row["training explanation condition"],
                row["strategy"],
                row["reason"],
            )
            for row in unavailable
        }
    )
    metadata = {
        "participant_data_used": False,
        "training_profiles_per_domain": {
            domain: len(dataset["training_pool"])
            for domain, dataset in bundle["datasets"].items()
        },
        "test_profiles_per_domain": {
            domain: (
                len(dataset["test_pool"])
                if args.limit_cases is None
                else min(args.limit_cases, len(dataset["test_pool"]))
            )
            for domain, dataset in bundle["datasets"].items()
        },
        "domains": list(bundle["datasets"]),
        "training_explanation_conditions": list(XAI_CONDITIONS),
        "maximum_attributes_allowed": list(MAX_CHANGES),
        "strategy_count": len(STRATEGIES),
        "trial_rows": len(trial_rows),
        "summary_rows": len(summary_rows),
        "unique_boundary_profiles": len(boundary_cache),
        "unavailable_strategy_condition_combinations": [
            {
                "domain": domain,
                "training explanation condition": xai,
                "strategy": strategy,
                "reason": reason,
            }
            for domain, xai, strategy, reason in unique_unavailable
        ],
        "metric_definitions": {
            "valid counterfactual": "browser-model prediction equals the opposite of the original prediction",
            "proximity": "summed range-normalized L1 edit distance across five features; lower is better",
            "model-consistent proximity": "summed unbounded range-normalized L1 edit distance retaining raw out-of-range values; used only with the model-consistent boundary distance",
            "boundary distance": "nearest summed range-normalized L1 distance found from the edited profile to an opposite-prediction profile at the browser-model decision boundary; lower is closer. A deterministic multi-start search is followed by exact linear programming within each discovered ReLU region. Raw out-of-range test values are retained for model consistency, while candidate edits cannot move farther outside configured ranges.",
            "plausibility": "1 - nearest five-feature Gower distance to the complete domain model-training split; higher is better",
            "plausibility subset": "1 - nearest five-feature Gower distance to the displayed domain training profiles; higher is better",
            "sparsity": "1 - actual changed feature count / 5; higher is sparser",
            "actionable": "all changed attributes belong to the domain's configured controllable feature set",
        },
        "important_scope_note": "The study evaluates deterministic strategy implementations learned from displayed training information. It does not estimate how frequently people use them.",
        "normalization_note": "Prediction and boundary calculations retain raw numeric values outside display ranges so all bundled original predictions are reproduced. Proximity and plausibility use the bounded [0,1] normalization used in the participant results.",
        "excluded_rule": "A participant-fitted fixed-amount strategy is excluded because model-only analysis supplies no defensible participant-specific amount parameter.",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
