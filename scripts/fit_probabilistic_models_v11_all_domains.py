"""Fit the conditional-rho models to all 104 complete older participants."""

from __future__ import annotations

import csv
import json
import math
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
OUTPUT = ROOT / "qualtrics" / "parsimonious_probabilistic_model_fits_v1.4_all_old_domains.csv"
SUMMARY = ROOT / "qualtrics" / "parsimonious_probabilistic_model_summary_v1.4_all_old_domains.json"

DOMAIN_ACTIONABILITY = {
    "diabetes": ((0, 1), 4),       # Age may or may not be treated as actionable.
    "housing": ((1,), -1),         # All five displayed attributes are actionable.
    "safelimit": ((0, 1), 3),      # Gender may be excluded as immutable.
}


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source_rows = read_csv(INPUT)
    grouped = defaultdict(list)
    for row in source_rows:
        grouped[row["participant"]].append(row)
    participants = {
        participant: rows
        for participant, rows in grouped.items()
        if sum(row["phase"] == "training" for row in rows) == 10
        and sum(row["phase"] == "testing" for row in rows) in (10, 20)
    }
    if len(participants) != 104:
        raise RuntimeError(f"Expected 104 complete participants, found {len(participants)}")

    experiment = json.loads(BUNDLE.read_text(encoding="utf-8"))
    domain_cases = {}
    domain_training = {}
    for domain, dataset in experiment["datasets"].items():
        cases = dataset["training_pool"] + dataset["test_pool"]
        domain_cases[domain] = {int(case["instance_id"]): case for case in cases}
        displayed_ids = sorted({
            int(float(row["instance id"]))
            for rows in participants.values()
            for row in rows
            if row["domain"] == domain and row["phase"] == "training"
        })
        domain_training[domain] = [domain_cases[domain][case_id] for case_id in displayed_ids]

    output_rows = []
    for participant, rows in sorted(participants.items()):
        domain = rows[0]["domain"]
        condition = rows[0]["xai"]
        trials = observed_trials(rows, domain_cases[domain])
        actionability_options, immutable_index = DOMAIN_ACTIONABILITY[domain]
        fits = []
        for family in ("feature contribution", "weighted examples"):
            params, insample, cv = fit_family(
                family,
                condition,
                domain_training[domain],
                trials,
                actionability_options=actionability_options,
                immutable_index=immutable_index,
            )
            result = {
                "participant": participant,
                "domain": domain,
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
            fits.append(result)
        winner = min(
            fits,
            key=lambda row: (row["5-fold CV selection_nll"], row["5-fold CV amount_mae"]),
        )["model family"]
        for result in fits:
            result["selected family by CV"] = int(result["model family"] == winner)
            output_rows.append(result)

    write_csv(OUTPUT, output_rows)
    winners = [row for row in output_rows if row["selected family by CV"]]
    groups = {}
    for domain in ("diabetes", "housing", "safelimit"):
        groups[domain] = {}
        for condition in ("none", "attribution", "counterfactual"):
            subset = [row for row in winners if row["domain"] == domain and row["xai"] == condition]
            groups[domain][condition] = {
                "participants": len(subset),
                "families": dict(Counter(row["model family"] for row in subset)),
                "mean CV selection F1": float(np.mean([row["5-fold CV selection_f1"] for row in subset])),
                "mean CV conditional amount MAE": float(np.mean([row["5-fold CV amount_mae"] for row in subset])),
                "rho": dict(Counter(str(row["rho"]) for row in subset)),
            }
    summary = {
        "participants": len(participants),
        "participant_counts": dict(Counter(rows[0]["domain"] for rows in participants.values())),
        "amount_fitting": "rho fitted conditional on each participant's observed feature subset",
        "aggregation_note": "fit separately by domain; downstream metrics should be averaged per participant before pooling",
        "groups": groups,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
