"""Fit the latest additive-rho cognitive models to the v1.9 latest cohort."""

from __future__ import annotations

import csv
import concurrent.futures
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fit_parsimonious_weighted_models import observed_trials  # noqa: E402
from scripts.fit_probabilistic_attribute_selection import fit_family  # noqa: E402


INPUT = ROOT / "qualtrics" / "qualtrics_results_v1.9.csv"
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
OUTPUT = ROOT / "qualtrics" / "v1.9_latest_probabilistic_model_fits.csv"
SUMMARY = ROOT / "qualtrics" / "v1.9_latest_probabilistic_model_summary.json"
STRICT_OUTPUT = ROOT / "qualtrics" / "v1.9_latest_additive_rho_fits.csv"
STRICT_SUMMARY = ROOT / "qualtrics" / "v1.9_latest_additive_rho_summary.json"
MODEL_FAMILIES = ("feature contribution", "weighted examples")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_parameter_counts(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    result = {}
    for name in ("eta", "alpha", "rho", "lambda", "beta", "age actionable"):
        values = [str(row[name]) for row in rows if not (isinstance(row[name], float) and math.isnan(row[name]))]
        result[name] = dict(sorted(Counter(values).items()))
    return result


def fit_participant(args):
    participant, rows, training, case_map = args
    condition = rows[0]["xai"]
    trials = observed_trials(rows, case_map)
    fitted = []
    for family in MODEL_FAMILIES:
        params, insample, cv = fit_family(
            family,
            condition,
            training,
            trials,
            actionability_options=(0, 1),
            immutable_index=4,
        )
        result: dict[str, object] = {
            "participant": participant,
            "domain": "diabetes",
            "xai": condition,
            "testing cases": len(trials),
            "model family": family,
            "k": "observed per instance",
            **params,
        }
        for name, value in insample.items():
            result[f"in-sample {name}"] = value
        for name, value in cv.items():
            result[f"5-fold CV {name}"] = value
        fitted.append(result)
    winner = min(
        fitted,
        key=lambda row: (
            float(row["5-fold CV selection_nll"]),
            float(row["5-fold CV amount_mae"]),
        ),
    )["model family"]
    for result in fitted:
        result["selected family by CV"] = int(result["model family"] == winner)
    return fitted


def main() -> None:
    source_rows = read_csv(INPUT)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        grouped[row["participant"]].append(row)
    participants = {
        participant: rows
        for participant, rows in grouped.items()
        if len(rows) == 32
        and sum(row["phase"] == "training" for row in rows) == 12
        and sum(row["phase"] == "testing" for row in rows) == 20
    }
    if len(participants) != 36:
        raise RuntimeError(f"Expected 36 complete latest-instance participants, found {len(participants)}")

    dataset = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    cases = dataset["training_pool"] + dataset["test_pool"]
    case_map = {int(case["instance_id"]): case for case in cases}
    displayed_training_ids = sorted(
        {
            int(float(row["instance id"]))
            for rows in participants.values()
            for row in rows
            if row["phase"] == "training"
        }
    )
    training = [case_map[case_id] for case_id in displayed_training_ids]
    if len(training) != 12:
        raise RuntimeError(f"Expected 12 displayed training cases, found {len(training)}")

    tasks = [
        (participant, rows, training, case_map)
        for participant, rows in sorted(participants.items())
    ]
    worker_count = min(int(os.environ.get("V19_FIT_WORKERS", "8")), len(tasks))
    output_rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(fit_participant, task): task[0]
            for task in tasks
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            participant = futures[future]
            output_rows.extend(future.result())
            print(f"fitted {completed}/{len(tasks)} {participant}", flush=True)
    output_rows.sort(key=lambda row: (str(row["participant"]), str(row["model family"])))

    # Always retain an unlocked, explicit audit output before attempting to
    # replace the conventional v1.9 file, which may be open in Excel or JMP.
    write_csv(STRICT_OUTPUT, output_rows)
    try:
        write_csv(OUTPUT, output_rows)
    except PermissionError:
        print(f"warning: could not replace locked {OUTPUT}; using {STRICT_OUTPUT}", flush=True)
    winners = [row for row in output_rows if row["selected family by CV"] == 1]
    groups = {}
    for condition in ("none", "attribution", "counterfactual"):
        subset = [row for row in winners if row["xai"] == condition]
        groups[condition] = {
            "participants": len(subset),
            "families": dict(Counter(str(row["model family"]) for row in subset)),
            "mean winner 5-fold CV selection NLL": float(
                np.mean([float(row["5-fold CV selection_nll"]) for row in subset])
            ),
            "mean winner 5-fold CV selection F1": float(
                np.mean([float(row["5-fold CV selection_f1"]) for row in subset])
            ),
            "mean winner 5-fold CV conditional amount MAE": float(
                np.mean([float(row["5-fold CV amount_mae"]) for row in subset])
            ),
            "mean winner 5-fold CV direction accuracy": float(
                np.mean([float(row["5-fold CV direction_accuracy"]) for row in subset])
            ),
        }

    summary = {
        "model_version": "latest additive-rho probabilistic attribute-selection model",
        "participants": len(participants),
        "training_cases": 12,
        "testing_cases_per_participant": 20,
        "model_families": list(MODEL_FAMILIES),
        "selection_distribution": "P(S|x,K) proportional to exp(sum of standardized feature scores in S), over all K-subsets; fixed temperature 1",
        "amount_fitting": (
            "two-stage: select feature-choice parameters by selection NLL, then fit rho "
            "as an additive normalized margin per selected feature using amount MAE "
            "after forcing each participant's observed feature subset, independent of "
            "subset eligibility/probability"
        ),
        "fitting_algorithm": (
            "separable staged search: feature-selection parameters are evaluated once; "
            "rho is searched only after identifying selection-NLL-optimal candidates"
        ),
        "rho_grid": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "k": "observed per participant-instance",
        "age_actionable": "binary fitted parameter; 0 removes Age from eligible subsets",
        "winner_counts": dict(Counter(str(row["model family"]) for row in winners)),
        "winner_metrics": {
            "mean 5-fold CV selection NLL": float(
                np.mean([float(row["5-fold CV selection_nll"]) for row in winners])
            ),
            "mean 5-fold CV selection F1": float(
                np.mean([float(row["5-fold CV selection_f1"]) for row in winners])
            ),
            "mean 5-fold CV conditional amount MAE": float(
                np.mean([float(row["5-fold CV amount_mae"]) for row in winners])
            ),
            "mean 5-fold CV direction accuracy": float(
                np.mean([float(row["5-fold CV direction_accuracy"]) for row in winners])
            ),
        },
        "winner_parameter_counts": finite_parameter_counts(winners),
        "groups": groups,
    }
    summary_text = json.dumps(summary, indent=2) + "\n"
    STRICT_SUMMARY.write_text(summary_text, encoding="utf-8")
    try:
        SUMMARY.write_text(summary_text, encoding="utf-8")
    except PermissionError:
        print(f"warning: could not replace locked {SUMMARY}; using {STRICT_SUMMARY}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
