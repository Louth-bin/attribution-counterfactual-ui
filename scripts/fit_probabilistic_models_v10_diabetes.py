"""Apply the current probabilistic models to the older v1.0 diabetes cohort."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fit_parsimonious_weighted_models import observed_trials
from scripts.fit_probabilistic_attribute_selection import fit_family

INPUT = ROOT / "qualtrics" / "qualtrics_results_v1.1.schema-export.csv"
BUNDLE = ROOT / "static" / "experiment-data.json"
OUTPUT = ROOT / "qualtrics" / "parsimonious_probabilistic_model_fits_v1.0_older_data_relevance_weighted.csv"
SUMMARY = ROOT / "qualtrics" / "parsimonious_probabilistic_model_summary_v1.0_older_data_relevance_weighted.json"


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    grouped = defaultdict(list)
    for row in read_csv(INPUT):
        if row["domain"] == "diabetes":
            grouped[row["participant"]].append(row)
    participants = {
        participant: rows for participant, rows in grouped.items()
        if len(rows) == 30
        and sum(row["phase"] == "training" for row in rows) == 10
        and sum(row["phase"] == "testing" for row in rows) == 20
    }
    if len(participants) != 58:
        raise RuntimeError(f"Expected 58 complete diabetes participants, found {len(participants)}")

    dataset = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    all_cases = dataset["training_pool"] + dataset["test_pool"]
    case_map = {int(case["instance_id"]): case for case in all_cases}
    training_ids_by_participant = {
        participant: tuple(sorted(int(float(row["instance id"])) for row in rows if row["phase"] == "training"))
        for participant, rows in participants.items()
    }
    unique_training_sets = set(training_ids_by_participant.values())
    if len(unique_training_sets) != 1:
        raise RuntimeError(f"Expected one shared training set, found {len(unique_training_sets)}")
    training_ids = next(iter(unique_training_sets))
    training = [case_map[case_id] for case_id in training_ids]

    output_rows = []
    for participant, rows in sorted(participants.items()):
        condition = rows[0]["xai"]
        trials = observed_trials(rows, case_map)
        fits = []
        for family in ("feature contribution", "weighted examples"):
            params, insample, cv = fit_family(family, condition, training, trials)
            result = {"participant": participant, "xai": condition, "model family": family, "k": "observed per instance", **params}
            for name, value in insample.items(): result[f"in-sample {name}"] = value
            for name, value in cv.items(): result[f"5-fold CV {name}"] = value
            fits.append(result)
        winner = min(fits, key=lambda row: (row["5-fold CV selection_nll"], row["5-fold CV amount_mae"]))["model family"]
        for result in fits:
            result["selected family by CV"] = int(result["model family"] == winner)
            output_rows.append(result)
    write_csv(OUTPUT, output_rows)

    winners = [row for row in output_rows if row["selected family by CV"]]
    summary = {
        "source": "qualtrics_results_v1.0.jmp data via unchanged v1.1 schema export",
        "domain": "diabetes",
        "participants": len(participants),
        "participants_by_condition": dict(Counter(rows[0]["xai"] for rows in participants.values())),
        "training_cases": len(training),
        "testing_cases": 20,
        "training_instance_ids": list(training_ids),
        "winners": {condition: dict(Counter(row["model family"] for row in winners if row["xai"] == condition)) for condition in ("none", "attribution", "counterfactual")},
        "winner_metrics": {
            condition: {name: float(np.mean([row[name] for row in winners if row["xai"] == condition])) for name in ("5-fold CV selection_f1", "5-fold CV amount_mae", "5-fold CV direction_accuracy", "5-fold CV loss", "5-fold CV selection_nll")}
            for condition in ("none", "attribution", "counterfactual")
        },
        "age_actionable_counts": {condition: dict(Counter(str(row["age actionable"]) for row in winners if row["xai"] == condition)) for condition in ("none", "attribution", "counterfactual")},
        "parameter_counts_winners": {
            condition: {
                parameter: dict(Counter(str(row[parameter]) for row in winners if row["xai"] == condition and str(row[parameter]) != "nan"))
                for parameter in ("eta", "alpha", "rho", "lambda", "beta", "age actionable")
            }
            for condition in ("none", "attribution", "counterfactual")
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
