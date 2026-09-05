"""Compare observed edit magnitudes with each selected model's rho=1 proposal."""

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

from scripts.fit_parsimonious_weighted_models import (
    BUNDLE,
    load_participants,
    observed_trials,
    training_representation,
)
from scripts.fit_probabilistic_attribute_selection import (
    contribution_distribution,
    memory_distribution,
)


FITS = ROOT / "qualtrics" / "parsimonious_probabilistic_model_fits_v1.2.csv"
TOLERANCE = 0.10
EPS = 1e-10


def classify(ratio: float) -> str:
    if ratio < 1.0 - TOLERANCE:
        return "undershoot"
    if ratio > 1.0 + TOLERANCE:
        return "overshoot"
    return "approximately matched"


def main() -> None:
    with FITS.open(encoding="utf-8-sig", newline="") as source:
        selected = {
            row["participant"]: row
            for row in csv.DictReader(source)
            if int(row["selected family by CV"]) == 1
        }

    dataset = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    training = dataset["training_pool"]
    case_map = {int(case["instance_id"]): case for case in dataset["test_pool"]}
    participants = load_participants()

    case_results = []
    participant_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for participant, rows in participants.items():
        fit = selected[participant]
        condition = fit["xai"]
        params = {
            key: float(fit[key])
            for key in ("eta", "alpha", "rho", "lambda", "beta")
        }
        params["age actionable"] = 1
        params["rho"] = 1.0
        representation = training_representation(training, condition, params["eta"])

        for trial in observed_trials(rows, case_map):
            observed_set = tuple(np.flatnonzero(np.abs(trial["observed"]) > EPS))
            if not observed_set:
                continue
            if fit["model family"] == "feature contribution":
                proposals = contribution_distribution(
                    trial["x"], trial["target"], representation, params, trial["observed_k"]
                )
            else:
                proposals = memory_distribution(
                    trial["x"], trial["target"], representation, condition, params, trial["observed_k"]
                )
            matches = [delta for subset, _, delta in proposals if subset == observed_set]
            if not matches:
                continue
            predicted = matches[0]
            observed_amount = float(np.mean(np.abs(trial["observed"][list(observed_set)])))
            base_amount = float(np.mean(np.abs(predicted[list(observed_set)])))
            if base_amount <= EPS:
                continue
            ratio = observed_amount / base_amount
            participant_values[participant].append((observed_amount, base_amount))
            direction_match = bool(np.all(
                np.sign(predicted[list(observed_set)])
                == np.sign(trial["observed"][list(observed_set)])
            ))
            case_results.append((condition, fit["model family"], classify(ratio), ratio, direction_match))

    participant_results = []
    for participant, values in participant_values.items():
        observed_mean = float(np.mean([value[0] for value in values]))
        base_mean = float(np.mean([value[1] for value in values]))
        ratio = observed_mean / base_mean
        participant_results.append((selected[participant]["xai"], selected[participant]["model family"], classify(ratio), ratio))

    def report(label: str, values) -> None:
        print(label)
        for group_name, group_filter in (
            ("overall", lambda row: True),
            ("none", lambda row: row[0] == "none"),
            ("attribution", lambda row: row[0] == "attribution"),
            ("counterfactual", lambda row: row[0] == "counterfactual"),
            ("feature contribution", lambda row: row[1] == "feature contribution"),
            ("weighted examples", lambda row: row[1] == "weighted examples"),
        ):
            subset = [row for row in values if group_filter(row)]
            counts = Counter(row[2] for row in subset)
            median_ratio = float(np.median([row[3] for row in subset])) if subset else float("nan")
            mean_ratio = float(np.mean([row[3] for row in subset])) if subset else float("nan")
            print(
                f"{group_name:20s} n={len(subset):3d} "
                f"under={counts['undershoot']:3d} match={counts['approximately matched']:3d} "
                f"over={counts['overshoot']:3d} mean_ratio={mean_ratio:.3f} median_ratio={median_ratio:.3f}"
            )

    report("CASE LEVEL", case_results)
    report("CASE LEVEL, ALL DIRECTIONS MATCH", [row for row in case_results if row[4]])
    report("PARTICIPANT LEVEL", participant_results)


if __name__ == "__main__":
    main()
