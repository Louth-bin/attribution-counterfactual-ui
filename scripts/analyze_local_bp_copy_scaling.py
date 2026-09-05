"""Measure whether scaled Blood Pressure copies transfer in cluster 2."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_local_copy_vs_exemplar import (
    FEATURES,
    copy_candidates,
    metrics,
    target_probability,
)


MULTIPLIERS = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
CSV_OUT = ROOT / "analysis" / "local_blood_pressure_copy_scaling.csv"
JSON_OUT = ROOT / "analysis" / "local_blood_pressure_copy_scaling_summary.json"


def minimum_directional_change(
    model: Any,
    original: np.ndarray,
    target: int,
    feature_index: int,
    direction: float,
    ranges: np.ndarray,
) -> float | None:
    endpoint = ranges[feature_index, 1] if direction > 0 else ranges[feature_index, 0]
    values = np.linspace(original[feature_index], endpoint, 10001)
    candidates = np.tile(original, (len(values), 1))
    candidates[:, feature_index] = values
    probabilities = model.predict_proba(pd.DataFrame(candidates, columns=FEATURES))[:, target]
    crossings = np.flatnonzero(probabilities >= 0.5)
    if not len(crossings):
        return None
    return float(values[crossings[0]] - original[feature_index])


def minimum_change_either_direction(
    model: Any,
    original: np.ndarray,
    target: int,
    feature_index: int,
    ranges: np.ndarray,
) -> float | None:
    values = np.linspace(ranges[feature_index, 0], ranges[feature_index, 1], 10001)
    candidates = np.tile(original, (len(values), 1))
    candidates[:, feature_index] = values
    probabilities = model.predict_proba(pd.DataFrame(candidates, columns=FEATURES))[:, target]
    crossings = np.flatnonzero(probabilities >= 0.5)
    if not len(crossings):
        return None
    nearest = crossings[np.argmin(np.abs(values[crossings] - original[feature_index]))]
    return float(values[nearest] - original[feature_index])


def main() -> None:
    bundle = json.loads(
        (ROOT / "qualtrics" / "experimental-actionable-preview-data.json").read_text(encoding="utf-8")
    )["datasets"]["diabetes"]
    training = bundle["training_pool"]
    testing = [case for case in bundle["test_pool"] if int(case["selection_cluster"]) == 2]
    model = joblib.load(ROOT / "analysis" / "diabetes_mlp_smoother_regularized.joblib")
    ranges = np.asarray(training[0]["raw_feature_ranges"], float)
    bp_index = FEATURES.index("blood_pressure")
    rows = []

    for test in testing:
        original = np.asarray(test["raw_feature_values"], float)
        target = 1 - int(test["prediction"]["value"])
        remembered, _ = copy_candidates(test, training, ranges)
        remembered_original = np.asarray(remembered["raw_feature_values"], float)
        remembered_edited = np.asarray(remembered["counterfactual"]["raw_feature_values"], float)
        copied_delta = float(remembered_edited[bp_index] - remembered_original[bp_index])
        required_delta = minimum_directional_change(
            model, original, target, bp_index, np.sign(copied_delta), ranges
        )
        any_direction_delta = minimum_change_either_direction(
            model, original, target, bp_index, ranges
        )
        base = {
            "test_instance_id": int(test["instance_id"]),
            "original_prediction": test["prediction"]["label"],
            "target_prediction": bundle["labels"][target],
            "nearest_training_change": int(remembered["instance_id"]),
            "copied_bp_delta": copied_delta,
            "minimum_bp_delta_to_flip": required_delta,
            "required_copy_multiplier": (
                abs(required_delta / copied_delta)
                if required_delta is not None and abs(copied_delta) > 1e-12 else None
            ),
            "minimum_bp_delta_either_direction": any_direction_delta,
            "either_direction_matches_copied_sign": (
                int(np.sign(any_direction_delta) == np.sign(copied_delta))
                if any_direction_delta is not None else None
            ),
        }
        for multiplier in MULTIPLIERS:
            edited = original.copy()
            edited[bp_index] = np.clip(
                original[bp_index] + multiplier * copied_delta,
                ranges[bp_index, 0],
                ranges[bp_index, 1],
            )
            result = metrics(model, original, edited, target, ranges)
            row = dict(base)
            row.update({
                "multiplier": multiplier,
                "applied_bp_delta": float(edited[bp_index] - original[bp_index]),
                "target_probability_after": result["target_probability_after"],
                "boundary_progress": result["boundary_progress"],
                "capped_boundary_progress": result["capped_boundary_progress"],
                "crossed_boundary": result["crossed_boundary"],
            })
            rows.append(row)

    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    required = [
        row for row in rows if row["multiplier"] == 1.0 and row["required_copy_multiplier"] is not None
    ]
    summary = {
        "cluster_2_cases": len(testing),
        "cases_with_directional_bp_only_solution": len(required),
        "cases_with_bp_only_solution_in_either_direction": len({
            row["test_instance_id"] for row in rows
            if row["minimum_bp_delta_either_direction"] is not None
        }),
        "cases_where_bp_solution_matches_copied_direction": len({
            row["test_instance_id"] for row in rows
            if row["either_direction_matches_copied_sign"] == 1
        }),
        "required_multiplier_median": (
            float(np.median([row["required_copy_multiplier"] for row in required])) if required else None
        ),
        "required_multiplier_range": (
            [
                float(min(row["required_copy_multiplier"] for row in required)),
                float(max(row["required_copy_multiplier"] for row in required)),
            ] if required else None
        ),
        "by_multiplier": [
            {
                "multiplier": multiplier,
                "flip_rate": float(np.mean([row["crossed_boundary"] for row in rows if row["multiplier"] == multiplier])),
                "mean_capped_boundary_progress": float(np.mean([
                    row["capped_boundary_progress"] for row in rows if row["multiplier"] == multiplier
                ])),
            }
            for multiplier in MULTIPLIERS
        ],
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
