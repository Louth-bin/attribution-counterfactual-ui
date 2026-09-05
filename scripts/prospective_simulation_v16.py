"""Prospective condition-effect simulation for the unused v1.6 diabetes cases."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import f_oneway, ttest_ind

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_v10_model_downstream_metrics import exact_boundaries, prediction_label
from scripts.fit_parsimonious_weighted_models import label_sign, normalize_case, training_representation
from scripts.fit_probabilistic_attribute_selection import contribution_distribution, memory_distribution
from scripts.run_model_only_strategy_study import reference_matrices, similarity

BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
FITS = ROOT / "qualtrics" / "parsimonious_probabilistic_model_fits_v1.4_all_old_domains.csv"
HISTORICAL = ROOT / "qualtrics" / "qualtrics_results_v1.1.schema-export.csv"
CONDITIONS = ("none", "attribution", "counterfactual")
SAMPLE_SIZES = (10, 20, 30, 40)
REPLICATIONS = 2000
SEED = 20260828


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def target_label(case: dict) -> str:
    current = case["prediction"]["label"]
    return next(label for label in case["prediction_labels"] if label != current)


def main() -> None:
    experiment = json.loads(BUNDLE.read_text(encoding="utf-8"))
    dataset = experiment["datasets"]["diabetes"]
    training = dataset["training_pool"]
    cases = dataset["test_pool"]
    browser_model = dataset["browser_model"]
    reference = reference_matrices("diabetes", dataset)[0]

    fits = {
        row["participant"]: row
        for row in read_csv(FITS)
        if row["domain"] == "diabetes" and row["selected family by CV"] == "1"
    }
    historical_rows = [
        row for row in read_csv(HISTORICAL)
        if row["domain"] == "diabetes"
        and row["phase"] == "testing"
        and row["participant"] in fits
    ]
    grouped_history = defaultdict(list)
    for row in historical_rows:
        k = sum(int(float(row[f"x_{index}_changed"])) for index in range(1, 6))
        grouped_history[row["participant"]].append(k)
    if len(fits) != 58 or any(len(grouped_history[p]) != 20 for p in fits):
        raise RuntimeError("Expected 58 fitted diabetes participants with 20 historical K values each")

    profile_keys = {}
    raw_metrics = {}
    for participant, fit in fits.items():
        condition = fit["xai"]
        params = {
            name: float(fit[name]) if fit[name] not in ("", "nan") else math.nan
            for name in ("eta", "alpha", "rho", "lambda", "beta")
        }
        params["age actionable"] = int(float(fit["age actionable"]))
        params["immutable index"] = int(float(fit["immutable index"]))
        representation = training_representation(training, condition, params["eta"])
        for case_index, case in enumerate(cases):
            x = normalize_case(case)
            target = -label_sign(case["prediction"]["label"])
            for k in sorted(set(grouped_history[participant])):
                if fit["model family"] == "feature contribution":
                    proposals = contribution_distribution(x, target, representation, params, k)
                else:
                    proposals = memory_distribution(x, target, representation, condition, params, k)
                if not proposals:
                    continue
                _, _, delta = max(proposals, key=lambda item: item[1])
                profile = np.clip(x + delta, 0.0, 1.0)
                key = f"{participant}|{case_index}|{k}"
                profile_keys[key] = profile
                raw_metrics[key] = {
                    "validity": float(prediction_label(profile, case, browser_model) == target_label(case)),
                    "plausibility": similarity(profile, reference),
                    "complexity": float(k),
                }

    print(f"participants={len(fits)} candidate_profiles={len(profile_keys)}", flush=True)
    boundaries = exact_boundaries(browser_model, cases[0], profile_keys)
    for key, distance in boundaries.items():
        raw_metrics[key]["boundary distance"] = float(distance)

    rng = np.random.default_rng(SEED)
    participants_by_condition = {
        condition: [participant for participant, fit in fits.items() if fit["xai"] == condition]
        for condition in CONDITIONS
    }
    metrics = ("validity", "boundary distance", "plausibility", "complexity")
    results = {
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "training cases": len(training),
        "testing cases": len(cases),
        "historical fitted participants": {condition: len(participants_by_condition[condition]) for condition in CONDITIONS},
        "K": "For every synthetic participant, their 20 observed historical K values are randomly permuted across the 20 new cases.",
        "selection": "Maximum-probability K-feature subset under the participant's fitted probabilistic model.",
        "replications": REPLICATIONS,
        "sample-size results": {},
    }

    for sample_size in SAMPLE_SIZES:
        replication_records = {metric: [] for metric in metrics}
        for _ in range(REPLICATIONS):
            condition_values = {condition: {metric: [] for metric in metrics} for condition in CONDITIONS}
            for condition in CONDITIONS:
                sampled = rng.choice(participants_by_condition[condition], size=sample_size, replace=True)
                for participant in sampled:
                    assigned_k = rng.permutation(grouped_history[participant])
                    values = {metric: [] for metric in metrics}
                    for case_index, k in enumerate(assigned_k):
                        row = raw_metrics[f"{participant}|{case_index}|{int(k)}"]
                        for metric in metrics:
                            values[metric].append(row[metric])
                    for metric in metrics:
                        condition_values[condition][metric].append(float(np.mean(values[metric])))

            for metric in metrics:
                none = np.asarray(condition_values["none"][metric])
                attribution = np.asarray(condition_values["attribution"][metric])
                counterfactual = np.asarray(condition_values["counterfactual"][metric])
                omnibus_p = float(f_oneway(none, attribution, counterfactual).pvalue)
                cf_none_p = float(ttest_ind(counterfactual, none, equal_var=False).pvalue)
                replication_records[metric].append({
                    "means": {
                        "none": float(none.mean()),
                        "attribution": float(attribution.mean()),
                        "counterfactual": float(counterfactual.mean()),
                    },
                    "counterfactual - none": float(counterfactual.mean() - none.mean()),
                    "omnibus significant": omnibus_p < 0.05,
                    "counterfactual vs none significant": cf_none_p < 0.05,
                })

        sample_result = {}
        for metric, records in replication_records.items():
            sample_result[metric] = {
                "expected means": {
                    condition: float(np.mean([row["means"][condition] for row in records]))
                    for condition in CONDITIONS
                },
                "expected counterfactual - none": float(np.mean([row["counterfactual - none"] for row in records])),
                "omnibus detection probability": float(np.mean([row["omnibus significant"] for row in records])),
                "counterfactual vs none detection probability": float(np.mean([row["counterfactual vs none significant"] for row in records])),
            }
        results["sample-size results"][str(sample_size)] = sample_result

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
