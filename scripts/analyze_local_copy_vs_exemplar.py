"""Compare one-feature copy transfer with exemplar strategies in the local preview."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "qualtrics" / "experimental-actionable-preview-data.json"
MODEL_PATH = ROOT / "analysis" / "diabetes_mlp_smoother_regularized.joblib"
TRIALS_OUT = ROOT / "analysis" / "local_copy_vs_exemplar_trials.csv"
SUMMARY_OUT = ROOT / "analysis" / "local_copy_vs_exemplar_summary.json"
CANDIDATES_OUT = ROOT / "analysis" / "local_copy_vs_exemplar_candidate_pool.csv"
FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
EDITABLE = [1, 2, 3, 4]


def normalized(values: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    return np.clip((values - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]), 0.0, 1.0)


def distance(left: np.ndarray, right: np.ndarray, ranges: np.ndarray) -> float:
    return float(np.mean(np.abs(normalized(left, ranges) - normalized(right, ranges))))


def target_probability(model: Any, values: np.ndarray, target: int) -> float:
    frame = pd.DataFrame([values], columns=FEATURES)
    return float(model.predict_proba(frame)[0, target])


def copy_candidates(
    test: dict[str, Any],
    training: list[dict[str, Any]],
    ranges: np.ndarray,
) -> tuple[dict[str, Any], list[tuple[int, np.ndarray]]]:
    original = np.asarray(test["raw_feature_values"], float)
    target = 1 - int(test["prediction"]["value"])
    eligible = [
        case for case in training
        if int(case["counterfactual"]["target_prediction"]["value"]) == target
    ]
    remembered = min(
        eligible,
        key=lambda case: distance(original, np.asarray(case["raw_feature_values"], float), ranges),
    )
    remembered_original = np.asarray(remembered["raw_feature_values"], float)
    remembered_edited = np.asarray(remembered["counterfactual"]["raw_feature_values"], float)
    candidates = []
    for name in remembered["counterfactual"]["raw_selected_feature_names"]:
        index = FEATURES.index(name)
        if index not in EDITABLE:
            continue
        edited = original.copy()
        edited[index] = np.clip(
            original[index] + remembered_edited[index] - remembered_original[index],
            ranges[index, 0],
            ranges[index, 1],
        )
        candidates.append((index, edited))
    if not candidates:
        raise RuntimeError(f"No editable copied component for test case {test['instance_id']}")
    return remembered, candidates


def displayed_exemplar_edit(
    test: dict[str, Any],
    training: list[dict[str, Any]],
    ranges: np.ndarray,
) -> tuple[dict[str, Any], int, np.ndarray]:
    original = np.asarray(test["raw_feature_values"], float)
    target = 1 - int(test["prediction"]["value"])
    eligible = [case for case in training if int(case["prediction"]["value"]) == target]
    exemplar = min(
        eligible,
        key=lambda case: distance(original, np.asarray(case["raw_feature_values"], float), ranges),
    )
    exemplar_values = np.asarray(exemplar["raw_feature_values"], float)
    mismatch = np.abs(normalized(exemplar_values, ranges) - normalized(original, ranges))
    index = max(EDITABLE, key=lambda candidate: (mismatch[candidate], -candidate))
    edited = original.copy()
    edited[index] = exemplar_values[index]
    return exemplar, index, edited


def real_exemplar_edit(
    test: dict[str, Any],
    real_training: pd.DataFrame,
    ranges: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    original = np.asarray(test["raw_feature_values"], float)
    target = 1 - int(test["prediction"]["value"])
    eligible = real_training.loc[real_training["target"].eq(target)].reset_index(drop=False)
    matrix = eligible[FEATURES].to_numpy(float)
    nearest_position = int(np.argmin(np.mean(np.abs(normalized(matrix, ranges) - normalized(original, ranges)), axis=1)))
    exemplar = matrix[nearest_position]
    mismatch = np.abs(normalized(exemplar, ranges) - normalized(original, ranges))
    index = max(EDITABLE, key=lambda candidate: (mismatch[candidate], -candidate))
    edited = original.copy()
    edited[index] = exemplar[index]
    return int(eligible.iloc[nearest_position]["index"]), index, edited


def metrics(
    model: Any,
    original: np.ndarray,
    edited: np.ndarray,
    target: int,
    ranges: np.ndarray,
) -> dict[str, float | int]:
    before = target_probability(model, original, target)
    after = target_probability(model, edited, target)
    initial_gap = max(1e-12, 0.5 - before)
    crossed = int(after >= 0.5)
    return {
        "target_probability_before": before,
        "target_probability_after": after,
        "target_probability_gain": after - before,
        "boundary_progress": (after - before) / initial_gap,
        "capped_boundary_progress": min(1.0, max(0.0, (after - before) / initial_gap)),
        "crossed_boundary": crossed,
        "remaining_boundary_gap": max(0.0, 0.5 - after),
        "boundary_overshoot": max(0.0, after - 0.5),
        "normalized_edit_size": float(np.sum(np.abs(normalized(edited, ranges) - normalized(original, ranges)))),
    }


def summarize(rows: list[dict[str, Any]], grouping: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[grouping(row)].append(row)
    output = []
    for key, values in sorted(groups.items()):
        output.append({
            "group": key,
            "n": len(values),
            "flip_rate": float(np.mean([row["crossed_boundary"] for row in values])),
            "mean_target_probability_gain": float(np.mean([row["target_probability_gain"] for row in values])),
            "mean_boundary_progress": float(np.mean([row["boundary_progress"] for row in values])),
            "mean_capped_boundary_progress": float(np.mean([row["capped_boundary_progress"] for row in values])),
            "mean_remaining_boundary_gap": float(np.mean([row["remaining_boundary_gap"] for row in values])),
            "mean_normalized_edit_size": float(np.mean([row["normalized_edit_size"] for row in values])),
            "median_boundary_overshoot_if_flipped": (
                float(np.median([row["boundary_overshoot"] for row in values if row["crossed_boundary"]]))
                if any(row["crossed_boundary"] for row in values) else None
            ),
        })
    return output


def main() -> None:
    bundle = json.loads(DATA_PATH.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    training = bundle["training_pool"]
    testing = bundle["test_pool"]
    ranges = np.asarray(training[0]["raw_feature_ranges"], float)
    model = joblib.load(MODEL_PATH)
    real_training = pd.read_csv(ROOT / "src" / "data" / "diabetes" / "train.csv")
    rows: list[dict[str, Any]] = []

    for test in testing:
        original = np.asarray(test["raw_feature_values"], float)
        target = 1 - int(test["prediction"]["value"])
        remembered, copied = copy_candidates(test, training, ranges)
        remembered_original = np.asarray(remembered["raw_feature_values"], float)
        remembered_edited = np.asarray(remembered["counterfactual"]["raw_feature_values"], float)
        full_copy = original.copy()
        for name in remembered["counterfactual"]["raw_selected_feature_names"]:
            index = FEATURES.index(name)
            full_copy[index] = np.clip(
                original[index] + remembered_edited[index] - remembered_original[index],
                ranges[index, 0],
                ranges[index, 1],
            )
        copied_by_size = max(
            copied,
            key=lambda item: abs(item[1][item[0]] - original[item[0]]) / (ranges[item[0], 1] - ranges[item[0], 0]),
        )
        copied_best = max(copied, key=lambda item: target_probability(model, item[1], target))
        scaled_copies = []
        for multiplier in (1.25, 1.5):
            feature_index = copied_by_size[0]
            scaled = original.copy()
            scaled[feature_index] = np.clip(
                original[feature_index]
                + multiplier * (copied_by_size[1][feature_index] - original[feature_index]),
                ranges[feature_index, 0],
                ranges[feature_index, 1],
            )
            scaled_copies.append((multiplier, feature_index, scaled))
        displayed_exemplar, displayed_index, displayed_edit = displayed_exemplar_edit(test, training, ranges)
        real_exemplar_index, real_index, real_edit = real_exemplar_edit(test, real_training, ranges)
        optimal = np.asarray(test["counterfactual"]["raw_feature_values"], float)

        proposals = [
            ("copy nearest training: full two-feature diagnostic", -1, full_copy, remembered["instance_id"]),
            ("copy nearest training: largest editable delta", copied_by_size[0], copied_by_size[1], remembered["instance_id"]),
            *[
                (f"copy nearest training: {multiplier:.2f}x delta", feature_index, scaled, remembered["instance_id"])
                for multiplier, feature_index, scaled in scaled_copies
            ],
            ("copy nearest training: best component oracle", copied_best[0], copied_best[1], remembered["instance_id"]),
            ("nearest displayed target exemplar", displayed_index, displayed_edit, displayed_exemplar["instance_id"]),
            ("nearest real target exemplar oracle", real_index, real_edit, real_exemplar_index),
            ("optimized one-feature recourse", FEATURES.index(test["counterfactual"]["raw_selected_feature_names"][0]), optimal, "optimized"),
        ]
        for strategy, feature_index, edited, source in proposals:
            row = {
                "test_instance_id": int(test["instance_id"]),
                "cluster": int(test["selection_cluster"]),
                "original_prediction": test["prediction"]["label"],
                "target_prediction": bundle["labels"][target],
                "strategy": strategy,
                "selected_feature": " + ".join(remembered["counterfactual"]["raw_selected_feature_names"])
                if feature_index == -1 else FEATURES[feature_index],
                "source_exemplar_or_change": source,
            }
            row.update(metrics(model, original, edited, target, ranges))
            rows.append(row)

    selected_sources = {
        (str(case["source_split"]), int(case["source_instance_id"]))
        for case in training
    }
    candidate_rows: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        frame = pd.read_csv(ROOT / "src" / "data" / "diabetes" / f"{split}.csv")
        for source_index, source_row in frame.iterrows():
            if (split, int(source_index)) in selected_sources:
                continue
            original = source_row[FEATURES].to_numpy(float)
            prediction = int(model.predict(pd.DataFrame([original], columns=FEATURES))[0])
            target = 1 - prediction
            pseudo_test = {
                "instance_id": f"{split}:{source_index}",
                "raw_feature_values": original.tolist(),
                "prediction": {"value": prediction},
            }
            remembered, copied = copy_candidates(pseudo_test, training, ranges)
            copied_by_size = max(
                copied,
                key=lambda item: abs(item[1][item[0]] - original[item[0]]) / (ranges[item[0], 1] - ranges[item[0], 0]),
            )
            _, exemplar_index, exemplar_edit = displayed_exemplar_edit(pseudo_test, training, ranges)
            copy_metrics = metrics(model, original, copied_by_size[1], target, ranges)
            exemplar_metrics = metrics(model, original, exemplar_edit, target, ranges)
            candidate_rows.append({
                "source": f"{split}:{source_index}",
                "cluster": int(remembered["selection_cluster"]),
                "original_prediction": bundle["labels"][prediction],
                "target_prediction": bundle["labels"][target],
                "nearest_training_change": int(remembered["instance_id"]),
                "copy_feature": FEATURES[copied_by_size[0]],
                "copy_crossed_boundary": copy_metrics["crossed_boundary"],
                "copy_boundary_progress": copy_metrics["boundary_progress"],
                "copy_target_probability_after": copy_metrics["target_probability_after"],
                "exemplar_feature": FEATURES[exemplar_index],
                "exemplar_crossed_boundary": exemplar_metrics["crossed_boundary"],
                "exemplar_boundary_progress": exemplar_metrics["boundary_progress"],
                "exemplar_target_probability_after": exemplar_metrics["target_probability_after"],
                "copy_minus_exemplar_target_probability": (
                    copy_metrics["target_probability_after"] - exemplar_metrics["target_probability_after"]
                ),
            })

    with TRIALS_OUT.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with CANDIDATES_OUT.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)

    candidate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidate_groups[f"cluster {row['cluster']} | {row['original_prediction']} -> {row['target_prediction']}"] .append(row)
    candidate_availability = []
    for group, values in sorted(candidate_groups.items()):
        candidate_availability.append({
            "group": group,
            "n": len(values),
            "copy_flip_count": int(sum(row["copy_crossed_boundary"] for row in values)),
            "copy_progress_at_least_half_count": int(sum(row["copy_boundary_progress"] >= 0.5 for row in values)),
            "copy_beats_displayed_exemplar_count": int(sum(row["copy_minus_exemplar_target_probability"] > 0 for row in values)),
            "copy_beats_exemplar_and_progress_at_least_half_count": int(sum(
                row["copy_minus_exemplar_target_probability"] > 0 and row["copy_boundary_progress"] >= 0.5
                for row in values
            )),
        })

    summary = {
        "definitions": {
            "copy": "Retrieve the displayed training counterfactual whose original profile is nearest among changes to the target label; transfer its largest normalized non-Glucose component.",
            "copy_best_component_oracle": "Same retrieved change, but choose whichever editable component produces the larger target probability.",
            "full_copy_diagnostic": "Transfer both remembered changes; this violates the one-feature test rule and, in the Glucose cluster, the Glucose lock.",
            "displayed_exemplar": "Retrieve the nearest displayed training profile with the target label; set its largest non-Glucose mismatch to the exemplar value.",
            "real_exemplar_oracle": "Same exemplar rule using all opposing-ground-truth-class rows in the real training split; unavailable to participants.",
            "boundary_progress": "Target-probability gain divided by the original probability gap to 0.5; 1 reaches the boundary, negative moves away.",
        },
        "overall": summarize(rows, lambda row: row["strategy"]),
        "by_cluster": summarize(rows, lambda row: f"{row['strategy']} | cluster {row['cluster']}"),
        "by_direction": summarize(rows, lambda row: f"{row['strategy']} | {row['original_prediction']} -> {row['target_prediction']}"),
        "full_pool_candidate_availability": candidate_availability,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
