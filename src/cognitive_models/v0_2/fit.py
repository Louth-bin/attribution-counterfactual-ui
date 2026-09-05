"""Fit v0.2 perfect and variable memory models to the latest cohort."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .model import MEMORY_VARIANTS, MODEL_FAMILIES, fit_family, observed_trials


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_RESPONSES = REPOSITORY / "qualtrics" / "qualtrics_results_v1.9.csv"
DEFAULT_CASES = REPOSITORY / "analysis" / "diabetes-experiment-bundle-v1.6.json"
DEFAULT_OUTPUT = Path(__file__).with_name("fits.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
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
    return participants, {int(case["instance_id"]): case for case in cases}


def participant_training(rows: list[dict], case_map: dict[int, dict]) -> list[dict]:
    training_rows = sorted(
        (row for row in rows if row["phase"] == "training"),
        key=lambda row: int(float(row["case"])),
    )
    return [case_map[int(float(row["instance id"]))] for row in training_rows]


def fit_participant(participant: str, rows: list[dict], case_map):
    condition = rows[0]["xai"]
    training = participant_training(rows, case_map)
    trials = observed_trials(rows, case_map)
    fitted = []
    for memory_variant in MEMORY_VARIANTS:
        variant_rows = []
        for family in MODEL_FAMILIES:
            parameters, in_sample, cross_validated = fit_family(
                family, condition, memory_variant, training, trials
            )
            result = {
                "participant": participant,
                "domain": "diabetes",
                "xai": condition,
                "testing cases": len(trials),
                "memory variant": memory_variant,
                "model family": family,
                "k": "observed per instance",
                **parameters,
            }
            for name, value in in_sample.items():
                result[f"in-sample {name}"] = value
            for name, value in cross_validated.items():
                result[f"5-fold CV {name}"] = value
            variant_rows.append(result)

        winner = min(
            variant_rows,
            key=lambda result: (
                result["5-fold CV selection_nll"],
                result["5-fold CV amount_mae"],
            ),
        )["model family"]
        for result in variant_rows:
            result["selected family by CV"] = int(result["model family"] == winner)
        fitted.extend(variant_rows)
    return fitted


def comparison_rows(results: list[dict]) -> list[dict]:
    selected = {
        (row["participant"], row["memory variant"]): row
        for row in results
        if row["selected family by CV"] == 1
    }
    participants = sorted({row["participant"] for row in results})
    rows = []
    for participant in participants:
        perfect = selected[(participant, "perfect memory")]
        variable = selected[(participant, "memory variation")]
        rows.append({
            "participant": participant,
            "xai": perfect["xai"],
            "perfect family": perfect["model family"],
            "variable family": variable["model family"],
            "variable memory decay": variable["memory decay"],
            "perfect CV selection_nll": perfect["5-fold CV selection_nll"],
            "variable CV selection_nll": variable["5-fold CV selection_nll"],
            "delta selection_nll (variable - perfect)": variable["5-fold CV selection_nll"] - perfect["5-fold CV selection_nll"],
            "perfect CV selection_f1": perfect["5-fold CV selection_f1"],
            "variable CV selection_f1": variable["5-fold CV selection_f1"],
            "delta selection_f1 (variable - perfect)": variable["5-fold CV selection_f1"] - perfect["5-fold CV selection_f1"],
            "perfect CV amount_mae": perfect["5-fold CV amount_mae"],
            "variable CV amount_mae": variable["5-fold CV amount_mae"],
            "delta amount_mae (variable - perfect)": variable["5-fold CV amount_mae"] - perfect["5-fold CV amount_mae"],
            "perfect CV loss": perfect["5-fold CV loss"],
            "variable CV loss": variable["5-fold CV loss"],
            "delta loss (variable - perfect)": variable["5-fold CV loss"] - perfect["5-fold CV loss"],
        })
    return rows


def summary(results: list[dict], comparisons: list[dict]) -> dict:
    report = {"participants": len(comparisons), "variants": {}}
    for variant in MEMORY_VARIANTS:
        selected = [
            row for row in results
            if row["memory variant"] == variant and row["selected family by CV"] == 1
        ]
        report["variants"][variant] = {
            "selected families": {
                family: sum(row["model family"] == family for row in selected)
                for family in MODEL_FAMILIES
            },
            "mean selected-family CV selection_nll": float(np.mean([row["5-fold CV selection_nll"] for row in selected])),
            "mean selected-family CV selection_f1": float(np.mean([row["5-fold CV selection_f1"] for row in selected])),
            "mean selected-family CV amount_mae": float(np.mean([row["5-fold CV amount_mae"] for row in selected])),
            "mean selected-family CV loss": float(np.mean([row["5-fold CV loss"] for row in selected])),
        }
    report["paired variable minus perfect"] = {
        "mean delta selection_nll": float(np.mean([row["delta selection_nll (variable - perfect)"] for row in comparisons])),
        "mean delta selection_f1": float(np.mean([row["delta selection_f1 (variable - perfect)"] for row in comparisons])),
        "mean delta amount_mae": float(np.mean([row["delta amount_mae (variable - perfect)"] for row in comparisons])),
        "mean delta loss": float(np.mean([row["delta loss (variable - perfect)"] for row in comparisons])),
        "participants with lower variable CV selection_nll": sum(row["delta selection_nll (variable - perfect)"] < 0 for row in comparisons),
        "participants with lower variable CV loss": sum(row["delta loss (variable - perfect)"] < 0 for row in comparisons),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--participant", help="Fit only one participant")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel participant fits")
    arguments = parser.parse_args()

    participants, case_map = load_data(arguments.responses, arguments.cases)
    if arguments.participant:
        participants = {arguments.participant: participants[arguments.participant]}

    results = []
    participant_items = sorted(participants.items())
    if arguments.jobs > 1 and len(participant_items) > 1:
        with ProcessPoolExecutor(max_workers=arguments.jobs) as executor:
            futures = {
                executor.submit(fit_participant, participant, rows, case_map): participant
                for participant, rows in participant_items
            }
            for future in as_completed(futures):
                participant = futures[future]
                results.extend(future.result())
                print(f"fitted {participant}", flush=True)
    else:
        for participant, rows in participant_items:
            results.extend(fit_participant(participant, rows, case_map))
            print(f"fitted {participant}", flush=True)
    results.sort(key=lambda row: (row["participant"], row["memory variant"], row["model family"]))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output, results)

    comparisons = comparison_rows(results)
    write_csv(arguments.output.with_name("memory_comparison.csv"), comparisons)
    with arguments.output.with_name("summary.json").open("w", encoding="utf-8") as destination:
        json.dump(summary(results, comparisons), destination, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
