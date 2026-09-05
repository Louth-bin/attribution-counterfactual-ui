"""Fit the frozen v0.1 baseline to the latest complete Qualtrics cohort."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .model import AGE_INDEX, FEATURES, MODEL_FAMILIES, fit_family, observed_trials


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_RESPONSES = REPOSITORY / "qualtrics" / "qualtrics_results_v1.9.csv"
DEFAULT_CASES = REPOSITORY / "analysis" / "diabetes-experiment-bundle-v1.6.json"
DEFAULT_OUTPUT = Path(__file__).with_name("fits.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_data(response_path: Path, case_path: Path):
    grouped = defaultdict(list)
    for row in read_csv(response_path):
        grouped[row["participant"]].append(row)
    participants = {
        participant: rows
        for participant, rows in grouped.items()
        if len(rows) == 32
        and sum(row["phase"] == "training" for row in rows) == 12
        and sum(row["phase"] == "testing" for row in rows) == 20
    }

    with case_path.open(encoding="utf-8") as source:
        dataset = json.load(source)["datasets"]["diabetes"]
    cases = dataset["training_pool"] + dataset["test_pool"]
    case_map = {int(case["instance_id"]): case for case in cases}
    training_ids = sorted({
        int(float(row["instance id"]))
        for rows in participants.values()
        for row in rows
        if row["phase"] == "training"
    })
    training = [case_map[case_id] for case_id in training_ids]
    return participants, training, case_map


def fit_participant(
    participant: str,
    rows: list[dict],
    training,
    case_map,
    locked_feature: str | None = None,
):
    condition = rows[0]["xai"]
    trials = observed_trials(rows, case_map)
    fitted = []
    for family in MODEL_FAMILIES:
        immutable_index = (
            FEATURES.index(locked_feature) if locked_feature else AGE_INDEX
        )
        parameters, in_sample, cross_validated = fit_family(
            family,
            condition,
            training,
            trials,
            immutable_index=immutable_index,
            actionability_options=(0,) if locked_feature else (0, 1),
        )
        result = {
            "participant": participant,
            "domain": "diabetes",
            "xai": condition,
            "testing cases": len(trials),
            "model family": family,
            "k": "observed per instance",
            "locked feature": locked_feature or "",
            **parameters,
        }
        if locked_feature:
            # The legacy parameter name referred specifically to Age. Keep its
            # reported meaning accurate when a different feature is locked.
            result["age actionable"] = int(locked_feature != "age")
        for name, value in in_sample.items():
            result[f"in-sample {name}"] = value
        for name, value in cross_validated.items():
            result[f"5-fold CV {name}"] = value
        fitted.append(result)

    winner = min(
        fitted,
        key=lambda result: (
            result["5-fold CV selection_nll"],
            result["5-fold CV amount_mae"],
        ),
    )["model family"]
    for result in fitted:
        result["selected family by CV"] = int(result["model family"] == winner)
    return fitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--participant", help="Fit only one participant")
    parser.add_argument(
        "--locked-feature",
        choices=FEATURES,
        help="Exclude this interface-locked feature from every candidate subset.",
    )
    arguments = parser.parse_args()

    participants, training, case_map = load_data(
        arguments.responses, arguments.cases
    )
    if arguments.participant:
        participants = {
            arguments.participant: participants[arguments.participant]
        }

    results = []
    for participant, rows in sorted(participants.items()):
        results.extend(fit_participant(
            participant,
            rows,
            training,
            case_map,
            locked_feature=arguments.locked_feature,
        ))
        print(f"fitted {participant}", flush=True)
    results.sort(key=lambda row: (row["participant"], row["model family"]))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output, results)


if __name__ == "__main__":
    main()
