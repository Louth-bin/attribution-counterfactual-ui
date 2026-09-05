"""Compare displayed training CFs with numeric two-feature minimal flips."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
MODEL = ROOT / "analysis" / "diabetes_mlp_regularized.joblib"
OUTPUT_DIR = ROOT / "outputs" / "v20-training-minimal-cf-check"
FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]


def probability(model, original: np.ndarray, selected: list[int], z: np.ndarray, ranges: np.ndarray, target: int) -> float:
    values = original.copy()
    for pos, index in enumerate(selected):
        values[index] = ranges[index, 0] + z[pos] * (ranges[index, 1] - ranges[index, 0])
    frame = pd.DataFrame([values], columns=FEATURES)
    return float(model.predict_proba(frame)[0, target])


def probability_batch(model, original: np.ndarray, selected: list[int], zs: np.ndarray, ranges: np.ndarray, target: int) -> np.ndarray:
    values = np.repeat(original[None, :], len(zs), axis=0)
    for pos, index in enumerate(selected):
        values[:, index] = ranges[index, 0] + zs[:, pos] * (ranges[index, 1] - ranges[index, 0])
    frame = pd.DataFrame(values, columns=FEATURES)
    return np.asarray(model.predict_proba(frame), dtype=float)[:, target]


def predict_label(model, original: np.ndarray, selected: list[int], z: np.ndarray, ranges: np.ndarray) -> int:
    values = original.copy()
    for pos, index in enumerate(selected):
        values[index] = ranges[index, 0] + z[pos] * (ranges[index, 1] - ranges[index, 0])
    frame = pd.DataFrame([values], columns=FEATURES)
    return int(model.predict(frame)[0])


def norm_values(values: np.ndarray, ranges: np.ndarray, selected: list[int]) -> np.ndarray:
    return np.asarray([
        (values[index] - ranges[index, 0]) / (ranges[index, 1] - ranges[index, 0])
        for index in selected
    ], dtype=float)


def solve_case(model, case: dict[str, object]) -> dict[str, object]:
    raw_names = list(case["raw_feature_names"])
    original = np.asarray(case["raw_feature_values"], dtype=float)
    displayed = np.asarray(case["counterfactual"]["raw_feature_values"], dtype=float)
    ranges = np.asarray(case["raw_feature_ranges"], dtype=float)
    selected_names = list(case["counterfactual"]["raw_selected_feature_names"])
    selected = [raw_names.index(name) for name in selected_names]
    target = int(case["counterfactual"]["target_prediction"]["value"])
    z0 = norm_values(original, ranges, selected)
    z_displayed = norm_values(displayed, ranges, selected)

    def l1(z: np.ndarray) -> float:
        return float(np.sum(np.abs(np.asarray(z) - z0)))

    directions: list[np.ndarray] = []
    for value in np.linspace(-1.0, 1.0, 2001):
        remainder = 1.0 - abs(float(value))
        for sign in (-1.0, 1.0):
            direction = np.asarray([float(value), sign * remainder], dtype=float)
            if np.sum(np.abs(direction)) > 0:
                directions.append(direction)
    directions.extend([z_displayed - z0, z0 - z_displayed])

    ray_data: list[tuple[np.ndarray, float]] = []
    endpoints: list[np.ndarray] = []
    for direction in directions:
        norm = float(np.sum(np.abs(direction)))
        if norm <= 0:
            continue
        direction = direction / norm
        max_radius = math.inf
        for pos, step in enumerate(direction):
            if step > 0:
                max_radius = min(max_radius, (1.0 - z0[pos]) / step)
            elif step < 0:
                max_radius = min(max_radius, (0.0 - z0[pos]) / step)
        if not math.isfinite(max_radius) or max_radius <= 0:
            continue
        endpoint = np.clip(z0 + max_radius * direction, 0.0, 1.0)
        ray_data.append((direction, max_radius))
        endpoints.append(endpoint)

    ray_candidates: list[np.ndarray] = []
    if endpoints:
        endpoint_probabilities = probability_batch(model, original, selected, np.asarray(endpoints), ranges, target)
        crossing = [
            (direction, max_radius)
            for (direction, max_radius), p in zip(ray_data, endpoint_probabilities)
            if p >= 0.5
        ]
        if crossing:
            directions_array = np.asarray([item[0] for item in crossing], dtype=float)
            low = np.zeros(len(crossing), dtype=float)
            high = np.asarray([item[1] for item in crossing], dtype=float)
            for _ in range(40):
                mid = (low + high) / 2.0
                candidates_array = np.clip(z0[None, :] + mid[:, None] * directions_array, 0.0, 1.0)
                probabilities = probability_batch(model, original, selected, candidates_array, ranges, target)
                crossed = probabilities >= 0.5
                high[crossed] = mid[crossed]
                low[~crossed] = mid[~crossed]
            ray_candidates = list(np.clip(z0[None, :] + high[:, None] * directions_array, 0.0, 1.0))

    display_direction = z_displayed - z0
    display_norm = float(np.sum(np.abs(display_direction)))
    display_line_distance = float("nan")
    display_line_probability = float("nan")
    display_line_z: np.ndarray | None = None
    if display_norm > 0:
        unit = display_direction / display_norm
        low = 0.0
        high = display_norm
        for _ in range(60):
            mid = (low + high) / 2.0
            candidate = np.clip(z0 + mid * unit, 0.0, 1.0)
            if probability(model, original, selected, candidate, ranges, target) >= 0.5:
                high = mid
            else:
                low = mid
        display_line_distance = high
        display_line_z = np.clip(z0 + high * unit, 0.0, 1.0)
        display_line_probability = probability(model, original, selected, display_line_z, ranges, target)

    candidates: list[tuple[float, np.ndarray, float, int, bool]] = []
    candidate_points = ray_candidates + [z_displayed]
    if display_line_z is not None:
        candidate_points.append(display_line_z)
    for z in candidate_points:
        p = probability(model, original, selected, np.asarray(z), ranges, target)
        pred = predict_label(model, original, selected, z, ranges)
        valid = pred == target and p >= 0.5 - 1e-8
        candidates.append((l1(z), z, p, pred, valid))

    valid_candidates = [item for item in candidates if item[4]]
    if not valid_candidates:
        # Fall back to the displayed CF and mark it invalid.
        z = z_displayed
        p = probability(model, original, selected, z, ranges, target)
        pred = predict_label(model, original, selected, z, ranges)
        best_distance = l1(z)
        valid = False
    else:
        best_distance, z, p, pred, valid = min(valid_candidates, key=lambda item: item[0])

    true_values = original.copy()
    for pos, index in enumerate(selected):
        true_values[index] = ranges[index, 0] + z[pos] * (ranges[index, 1] - ranges[index, 0])

    displayed_distance = l1(z_displayed)
    profile_gap = float(np.sum(np.abs(z_displayed - z)))
    excess = displayed_distance - best_distance

    return {
        "instance id": int(case["instance_id"]),
        "source": f"{case['source_split']}:{case['source_instance_id']}",
        "original label": case["prediction"]["label"],
        "target label": case["counterfactual"]["target_prediction"]["label"],
        "feature pair": "|".join(selected_names),
        "displayed CF target probability": float(case["counterfactual"]["target_probability"]),
        "numeric minimal target probability": p,
        "displayed CF normalized L1": displayed_distance,
        "numeric minimal normalized L1": best_distance,
        "displayed-direction boundary L1": display_line_distance,
        "displayed-direction boundary target probability": display_line_probability,
        "displayed excess L1 over numeric minimum": excess,
        "L1 distance from displayed CF to numeric minimum": profile_gap,
        "displayed / numeric distance ratio": displayed_distance / best_distance if best_distance > 0 else math.inf,
        "numeric solution valid": valid,
        "numeric prediction value": pred,
        "numeric changed feature count above 1e-4": int(np.sum(np.abs(z - z0) > 1e-4)),
        "displayed normalized changes": {
            name: float(z_displayed[pos] - z0[pos])
            for pos, name in enumerate(selected_names)
        },
        "numeric minimal normalized changes": {
            name: float(z[pos] - z0[pos])
            for pos, name in enumerate(selected_names)
        },
        "original selected raw values": {
            name: float(original[index])
            for name, index in zip(selected_names, selected)
        },
        "displayed CF selected raw values": {
            name: float(displayed[index])
            for name, index in zip(selected_names, selected)
        },
        "numeric minimal selected raw values": {
            name: float(true_values[index])
            for name, index in zip(selected_names, selected)
        },
    }


def main() -> None:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    model = joblib.load(MODEL)
    rows = [solve_case(model, case) for case in bundle["training_pool"]]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for row in rows:
        flat = dict(row)
        for key in (
            "displayed normalized changes",
            "numeric minimal normalized changes",
            "original selected raw values",
            "displayed CF selected raw values",
            "numeric minimal selected raw values",
        ):
            flat[key] = json.dumps(flat[key], sort_keys=True)
        csv_rows.append(flat)

    with (OUTPUT_DIR / "training_two_feature_minimal_cf_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=list(csv_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "scope": "v1.6/v2.0 training cases only",
        "solver": "numeric continuous two-feature constrained dense vectorized ray scan; also bisection along displayed-CF direction",
        "interpretation": "negative/near-zero excess means the displayed CF is as close as the numeric two-feature minimum; positive excess means displayed overshoots in normalized L1",
        "cases": rows,
        "mean displayed CF normalized L1": float(np.mean([row["displayed CF normalized L1"] for row in rows])),
        "mean numeric minimal normalized L1": float(np.mean([row["numeric minimal normalized L1"] for row in rows])),
        "mean displayed excess L1": float(np.mean([row["displayed excess L1 over numeric minimum"] for row in rows])),
        "max displayed excess L1": float(np.max([row["displayed excess L1 over numeric minimum"] for row in rows])),
        "cases with displayed excess > 0.01": [
            row["instance id"] for row in rows
            if row["displayed excess L1 over numeric minimum"] > 0.01
        ],
    }
    (OUTPUT_DIR / "training_two_feature_minimal_cf_comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
