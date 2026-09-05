"""Simulate latest fitted cognitive models and compare participant performance."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calculate_v09_boundary_distance import (  # noqa: E402
    WachterBoundarySearch,
    model_logit,
)
from scripts.fit_parsimonious_weighted_models import (  # noqa: E402
    case_metrics,
    label_sign,
    observed_trials,
    training_representation,
)
from scripts.fit_probabilistic_attribute_selection import (  # noqa: E402
    contribution_distribution,
    memory_distribution,
)
from scripts.run_model_only_strategy_study import reference_matrices, similarity  # noqa: E402


RESULTS = ROOT / "qualtrics" / "qualtrics_results_v1.9.csv"
FITS = ROOT / "qualtrics" / "v1.9_latest_probabilistic_model_fits.csv"
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
TRIAL_OUTPUT = ROOT / "qualtrics" / "v1.9_latest_model_simulation_trials.csv"
EDIT_COMPARISON_OUTPUT = ROOT / "qualtrics" / "v1.9_latest_user_model_edit_comparison.csv"
PARTICIPANT_OUTPUT = ROOT / "qualtrics" / "v1.9_latest_model_simulation_participants.csv"
SUMMARY_OUTPUT = ROOT / "qualtrics" / "v1.9_latest_model_simulation_summary.json"


METRICS = {
    "success": "Counterfactual success rate",
    "target confidence gain": "Target-confidence gain (probability)",
    "boundary improvement": "Boundary-distance improvement",
    "plausibility": "Plausibility",
    "actionability": "Actionability rate",
    "edit L1": "Normalized edit distance (L1)",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prediction_label(profile: np.ndarray, case: dict, browser_model: dict) -> str:
    logit = model_logit(browser_model, profile, case)
    labels = case["prediction_labels"]
    return labels[1] if logit >= 0 else labels[0]


def target_probability(profile: np.ndarray, target_label: str, case: dict, browser_model: dict) -> float:
    logit = model_logit(browser_model, profile, case)
    positive_probability = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -40.0, 40.0))))
    return positive_probability if target_label == case["prediction_labels"][1] else 1.0 - positive_probability


def raw_profile(normalized: np.ndarray, case: dict) -> list[float | str]:
    raw: list[float | str] = []
    for value, feature_type, feature_range in zip(
        normalized, case["feature_types"], case["raw_feature_ranges"]
    ):
        if feature_type == "categorical":
            position = int(round(float(value) * (len(feature_range) - 1)))
            raw.append(str(feature_range[position]))
        else:
            low, high = float(feature_range[0]), float(feature_range[1])
            raw.append(low + float(value) * (high - low))
    return raw


def edit_description(
    names: list[str], original: list[float | str], edited: list[float | str]
) -> str:
    changes = []
    for name, before, after in zip(names, original, edited):
        if isinstance(before, str) or isinstance(after, str):
            if str(before) != str(after):
                changes.append(f"{name}: {before} -> {after}")
        elif abs(float(after) - float(before)) > 1e-9:
            changes.append(
                f"{name}: {float(before):.2f} -> {float(after):.2f} "
                f"(delta {float(after) - float(before):+.2f})"
            )
    return "; ".join(changes) if changes else "No change"


def boundary_chunk(args):
    browser_model, representative_case, items = args
    solver = WachterBoundarySearch(
        browser_model,
        representative_case,
        allow_reference_outside_bounds=True,
    )
    return [(key, solver.solve(np.asarray(profile, dtype=float)).distance) for key, profile in items]


def exact_boundaries(browser_model: dict, representative_case: dict, profiles: dict[str, np.ndarray]) -> dict[str, float]:
    unique: dict[tuple[float, ...], list[str]] = defaultdict(list)
    for key, profile in profiles.items():
        unique[tuple(np.round(profile, 12))].append(key)
    items = [(profile, list(profile)) for profile in unique]
    worker_count = min(int(os.environ.get("V19_BOUNDARY_WORKERS", "8")), len(items))
    chunks = [items[index::worker_count] for index in range(worker_count)]
    solved = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for result in executor.map(
            boundary_chunk,
            [(browser_model, representative_case, chunk) for chunk in chunks],
        ):
            solved.update(result)
    return {
        key: solved[tuple(np.round(profile, 12))]
        for key, profile in profiles.items()
    }


def main() -> None:
    experiment = json.loads(BUNDLE.read_text(encoding="utf-8"))
    dataset = experiment["datasets"]["diabetes"]
    browser_model = dataset["browser_model"]
    cases = dataset["training_pool"] + dataset["test_pool"]
    case_map = {int(case["instance_id"]): case for case in cases}
    training = dataset["training_pool"]
    reference = reference_matrices("diabetes", dataset)[0]

    rows = read_csv(RESULTS)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["participant"]].append(row)
    selected_fits = [
        row for row in read_csv(FITS) if row["selected family by CV"] == "1"
    ]
    fit_map = {row["participant"]: row for row in selected_fits}
    if len(fit_map) != 36:
        raise RuntimeError(f"Expected 36 selected fits, found {len(fit_map)}")

    trial_rows: list[dict[str, object]] = []
    profiles: dict[str, np.ndarray] = {}
    for participant, participant_rows in sorted(grouped.items()):
        fit = fit_map[participant]
        condition = fit["xai"]
        params = {
            name: float(fit[name]) if fit[name] not in ("", "nan") else math.nan
            for name in ("eta", "alpha", "rho", "lambda", "beta")
        }
        params["age actionable"] = int(float(fit["age actionable"]))
        params["immutable index"] = int(float(fit["immutable index"]))
        representation = training_representation(training, condition, params["eta"])
        trials = observed_trials(participant_rows, case_map)
        source_testing = sorted(
            (row for row in participant_rows if row["phase"] == "testing"),
            key=lambda row: int(float(row["case"])),
        )
        for trial, source in zip(trials, source_testing):
            case_id = int(float(source["instance id"]))
            case = case_map[case_id]
            if fit["model family"] == "feature contribution":
                distribution = contribution_distribution(
                    trial["x"], trial["target"], representation, params, trial["observed_k"]
                )
            else:
                distribution = memory_distribution(
                    trial["x"], trial["target"], representation, condition, params, trial["observed_k"]
                )
            _, subset_probability, model_delta = max(distribution, key=lambda item: item[1])
            model_profile = np.clip(trial["x"] + model_delta, 0.0, 1.0)
            agreement = case_metrics(model_delta, trial["observed"])
            user_profile = np.clip(trial["x"] + trial["observed"], 0.0, 1.0)
            feature_names = list(case["raw_feature_names"])
            original_raw = raw_profile(trial["x"], case)
            user_raw = raw_profile(user_profile, case)
            model_raw = raw_profile(model_profile, case)
            edit_vector_mae = float(np.mean(np.abs(model_delta - trial["observed"])))
            edit_vector_l1 = float(np.sum(np.abs(model_delta - trial["observed"])))
            edit_size_total = float(
                np.sum(np.abs(model_delta)) + np.sum(np.abs(trial["observed"]))
            )
            edit_closeness = (
                1.0
                if edit_size_total <= 1e-12
                else float(np.clip(1.0 - edit_vector_l1 / edit_size_total, 0.0, 1.0))
            )
            target_label = source["target label"]
            original_target_probability = target_probability(trial["x"], target_label, case, browser_model)
            model_target_probability = target_probability(model_profile, target_label, case, browser_model)
            profile_key = f"{participant}|{case_id}"
            profiles[profile_key] = model_profile
            trial_rows.append(
                {
                    "participant": participant,
                    "xai": condition,
                    "case": source["case"],
                    "instance id": case_id,
                    "selected model family": fit["model family"],
                    "model MAP subset probability": subset_probability,
                    "user suggested edits": edit_description(
                        feature_names, original_raw, user_raw
                    ),
                    "model suggested edits": edit_description(
                        feature_names, original_raw, model_raw
                    ),
                    "edit vector MAE (lower is closer)": edit_vector_mae,
                    "edit closeness 0-1 (higher is closer)": edit_closeness,
                    "edit closeness definition": "1 - sum(abs(model delta - user delta)) / (sum(abs(model delta)) + sum(abs(user delta)))",
                    "selection F1": agreement[1],
                    "amount MAE": agreement[2],
                    "direction accuracy": agreement[3],
                    "observed success": float(source["successful counterfactual (0/1)"]),
                    "simulated success": float(
                        prediction_label(model_profile, case, browser_model) == target_label
                    ),
                    "observed target confidence gain": float(source["delta confidence of target label"]),
                    "simulated target confidence gain": (
                        model_target_probability - original_target_probability
                    ),
                    "observed boundary improvement": -float(
                        source["boundary distance change (new - original)"]
                    ),
                    "observed boundary distance original": float(source["boundary distance original"]),
                    "observed plausibility": float(source["plausibility"]),
                    "simulated plausibility": similarity(model_profile, reference),
                    "observed actionability": float(source["actionability (0/1)"]),
                    "simulated actionability": float(abs(model_delta[4]) <= 1e-9),
                    "observed edit L1": float(
                        sum(abs(float(source[f"x_{index}_change"])) for index in range(1, 6))
                    ),
                    "simulated edit L1": float(np.abs(model_delta).sum()),
                    "_profile_key": profile_key,
                }
            )
            for index, name in enumerate(feature_names):
                key = name.strip().lower().replace(" ", "_")
                trial_rows[-1][f"original value: {key}"] = original_raw[index]
                trial_rows[-1][f"user after: {key}"] = user_raw[index]
                trial_rows[-1][f"model after: {key}"] = model_raw[index]
                trial_rows[-1][f"user normalized change: {key}"] = float(
                    trial["observed"][index]
                )
                trial_rows[-1][f"model normalized change: {key}"] = float(
                    model_delta[index]
                )
                trial_rows[-1][f"absolute change difference: {key}"] = float(
                    abs(model_delta[index] - trial["observed"][index])
                )

    print(f"boundary_profiles={len(profiles)}", flush=True)
    boundaries = exact_boundaries(browser_model, dataset["test_pool"][0], profiles)
    for row in trial_rows:
        row["simulated boundary distance new"] = boundaries[row.pop("_profile_key")]
        row["simulated boundary improvement"] = (
            float(row["observed boundary distance original"])
            - float(row["simulated boundary distance new"])
        )
    write_csv(TRIAL_OUTPUT, trial_rows)
    write_csv(EDIT_COMPARISON_OUTPUT, trial_rows)

    grouped_trials: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        grouped_trials[str(row["participant"])].append(row)
    participant_output = []
    for participant, values in sorted(grouped_trials.items()):
        result: dict[str, object] = {
            "participant": participant,
            "xai": values[0]["xai"],
            "selected model family": values[0]["selected model family"],
            "testing cases": len(values),
            "mean selection F1": float(np.mean([row["selection F1"] for row in values])),
            "mean amount MAE": float(np.mean([row["amount MAE"] for row in values])),
            "mean direction accuracy": float(
                np.nanmean([row["direction accuracy"] for row in values])
            ),
        }
        for metric in METRICS:
            for source in ("observed", "simulated"):
                result[f"{source} {metric}"] = float(
                    np.mean([row[f"{source} {metric}"] for row in values])
                )
        participant_output.append(result)
    write_csv(PARTICIPANT_OUTPUT, participant_output)

    metric_results = {}
    for metric, label in METRICS.items():
        observed = np.asarray([row[f"observed {metric}"] for row in participant_output])
        simulated = np.asarray([row[f"simulated {metric}"] for row in participant_output])
        pearson = stats.pearsonr(observed, simulated)
        spearman = stats.spearmanr(observed, simulated)
        metric_results[metric] = {
            "label": label,
            "n": len(observed),
            "observed mean": float(observed.mean()),
            "simulated mean": float(simulated.mean()),
            "pearson r": float(pearson.statistic),
            "pearson p": float(pearson.pvalue),
            "spearman rho": float(spearman.statistic),
            "spearman p": float(spearman.pvalue),
            "MAE": float(np.mean(np.abs(observed - simulated))),
        }
    summary = {
        "model_version": "latest additive-rho probabilistic attribute-selection model",
        "participants": len(participant_output),
        "testing_trials": len(trial_rows),
        "simulation": "maximum-probability feature subset at each participant's fitted parameters and observed K",
        "aggregation": "performance averaged over 20 testing trials per participant before correlation",
        "metrics": metric_results,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
