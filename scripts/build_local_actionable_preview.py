"""Build an isolated preview for diverse, actionable diabetes recourse cases.

This script does not modify the production static bundle or any QSF.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_recourse_v15 import browser_model_payload  # noqa: E402


FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
DISPLAY = ["Glucose", "Blood Pressure", "Insulin", "BMI", "Age"]
LABELS = ["Diabetes", "No Diabetes"]
ACTIONABLE_TRAINING_INDICES = [0, 1, 2, 3]
TEST_EDITABLE_INDICES = [1, 2, 3, 4]
ACTION_COST = np.asarray([1.00, 1.15, 1.10, 0.85, 3.00], dtype=float)
PLAUSIBILITY_WEIGHT = 0.35
MIN_NORMALIZED_CHANGE = 0.10
TRAINING_MIN_NORMALIZED_CHANGE = 0.15
MAX_TRAINING_NORMALIZED_CHANGE = 0.25
TRAINING_PER_PAIR_LABEL = 3
TESTING_PER_LABEL = 10
PAIR_COUNT = 2
PAIR_GRID_POINTS = 61
MODEL_PATH = ROOT / "analysis" / "diabetes_mlp_smoother_regularized.joblib"
FIXED_PAIR_FAMILIES = [(0, 3), (1, 2)]
MAX_TEST_PROFILE_DISTANCE = 0.20
OUT_JSON = ROOT / "qualtrics" / "experimental-actionable-preview-data.json"
OUT_JS = ROOT / "qualtrics" / "experimental-actionable-preview-data.js"
OUT_CSV = ROOT / "qualtrics" / "experimental-actionable-preview-instances.csv"
OUT_SUMMARY = ROOT / "qualtrics" / "experimental-actionable-preview-summary.json"
OUT_REMINDER = ROOT / "qualtrics" / "NEXT_ITERATION_REMINDER.md"


def normalized(values: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    return np.clip((values - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]), 0.0, 1.0)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for split in ("train", "dev", "test"):
        frame = pd.read_csv(ROOT / f"src/data/diabetes/{split}.csv")
        frame["source_split"] = split
        frame["source_instance_id"] = np.arange(len(frame), dtype=int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), frames[0]


def target_reference(train: pd.DataFrame, target: int) -> np.ndarray:
    """Return real training profiles with the opposing ground-truth class."""
    labels = train["target"].to_numpy(int)
    return train.loc[labels == target, FEATURES].to_numpy(float)


def plausibility_distance(
    values: np.ndarray,
    target_values: np.ndarray,
    ranges: np.ndarray,
) -> float:
    point = normalized(values, ranges)
    candidates = normalized(target_values, ranges)
    distances = np.mean(np.abs(candidates - point), axis=1)
    return float(np.mean(np.partition(distances, min(2, len(distances) - 1))[:3]))


def feature_grid(index: int, ranges: np.ndarray) -> np.ndarray:
    low, high = ranges[index]
    values = np.linspace(low, high, PAIR_GRID_POINTS)
    if index in (0, 1, 2, 4):
        values = np.round(values)
    else:
        values = np.round(values, 1)
    return np.unique(values)


def two_feature_solution(
    model: Any,
    original: np.ndarray,
    prediction: int,
    pair: tuple[int, int],
    ranges: np.ndarray,
    target_values: np.ndarray,
    target_neighbors: NearestNeighbors,
    target_direction: np.ndarray,
) -> dict[str, Any] | None:
    first_values, second_values = feature_grid(pair[0], ranges), feature_grid(pair[1], ranges)
    first_mesh, second_mesh = np.meshgrid(first_values, second_values, indexing="ij")
    candidates = np.tile(original, (first_mesh.size, 1))
    candidates[:, pair[0]] = first_mesh.ravel()
    candidates[:, pair[1]] = second_mesh.ravel()
    deltas = np.abs((candidates - original) / (ranges[:, 1] - ranges[:, 0]))
    signed_changes = candidates - original
    feasible = (
        np.all(deltas[:, list(pair)] >= TRAINING_MIN_NORMALIZED_CHANGE - 1e-9, axis=1)
        & np.all(deltas[:, list(pair)] <= MAX_TRAINING_NORMALIZED_CHANGE + 1e-9, axis=1)
        & np.all(signed_changes[:, list(pair)] * target_direction[list(pair)] > 0, axis=1)
    )
    candidates = candidates[feasible]
    deltas = deltas[feasible]
    if not len(candidates):
        return None
    target_mask = np.asarray(model.predict(pd.DataFrame(candidates, columns=FEATURES)), dtype=int) == 1 - prediction
    candidates = candidates[target_mask]
    deltas = deltas[target_mask]
    if not len(candidates):
        return None
    neighbour_distances, _ = target_neighbors.kneighbors(normalized(candidates, ranges), n_neighbors=3)
    plausibilities = neighbour_distances.mean(axis=1) / len(FEATURES)
    proximities = deltas.sum(axis=1)
    actionable_l1s = np.sum(deltas * ACTION_COST, axis=1)
    objectives = actionable_l1s + PLAUSIBILITY_WEIGHT * plausibilities
    best = int(np.lexsort((proximities, objectives))[0])
    edited = candidates[best]
    return {
        "edited": edited,
        "pair": pair,
        "proximity": float(proximities[best]),
        "actionable_l1": float(actionable_l1s[best]),
        "plausibility_distance": float(plausibilities[best]),
        "objective": float(objectives[best]),
        "target_probability": float(model.predict_proba(pd.DataFrame([edited], columns=FEATURES))[0, 1 - prediction]),
    }


def one_feature_solution(
    model: Any,
    original: np.ndarray,
    prediction: int,
    ranges: np.ndarray,
    target_values: np.ndarray,
    target_neighbors: NearestNeighbors,
    target_direction: np.ndarray,
) -> dict[str, Any] | None:
    best = None
    for index in TEST_EDITABLE_INDICES:
        low, high = ranges[index]
        values = np.linspace(low, high, 401)
        candidates = np.tile(original, (len(values), 1))
        candidates[:, index] = values
        predictions = np.asarray(model.predict(pd.DataFrame(candidates, columns=FEATURES)), dtype=int)
        deltas = np.abs((candidates[:, index] - original[index]) / (high - low))
        signed_changes = candidates[:, index] - original[index]
        feasible = (
            (predictions == 1 - prediction)
            & (deltas >= MIN_NORMALIZED_CHANGE - 1e-9)
            & (signed_changes * target_direction[index] > 0)
        )
        feasible_positions = np.flatnonzero(feasible)
        if not len(feasible_positions):
            continue
        feasible_candidates = candidates[feasible_positions]
        feasible_deltas = deltas[feasible_positions]
        neighbour_distances, _ = target_neighbors.kneighbors(normalized(feasible_candidates, ranges), n_neighbors=3)
        plausibilities = neighbour_distances.mean(axis=1) / len(FEATURES)
        scores = feasible_deltas * ACTION_COST[index] + PLAUSIBILITY_WEIGHT * plausibilities
        for local_position in np.argsort(scores)[:1]:
            edited = feasible_candidates[local_position]
            delta = feasible_deltas[local_position]
            plausibility = plausibilities[local_position]
            score = scores[local_position]
            record = {
                "edited": edited.copy(),
                "feature": index,
                "proximity": float(delta),
                "actionable_l1": float(delta * ACTION_COST[index]),
                "plausibility_distance": plausibility,
                "objective": score,
                "target_probability": float(model.predict_proba(pd.DataFrame([edited], columns=FEATURES))[0, 1 - prediction]),
            }
            if best is None or (record["objective"], record["proximity"], index) < (
                best["objective"], best["proximity"], best["feature"]
            ):
                best = record
    return best


def select_pair_families(records: list[dict[str, Any]]) -> list[tuple[int, int]]:
    grouped: dict[tuple[int, int], dict[int, list[dict[str, Any]]]] = {}
    for record in records:
        grouped.setdefault(record["solution"]["pair"], {0: [], 1: []})[record["prediction"]].append(record)
    eligible = [pair for pair, sides in grouped.items() if min(len(sides[0]), len(sides[1])) >= TRAINING_PER_PAIR_LABEL]
    if all(pair in eligible for pair in FIXED_PAIR_FAMILIES):
        return list(FIXED_PAIR_FAMILIES)
    choices = []
    for pairs in itertools.combinations(eligible, PAIR_COUNT):
        feature_counts = Counter(index for pair in pairs for index in pair)
        if max(feature_counts.values()) >= PAIR_COUNT:
            continue
        median_score = np.mean([
            np.median([row["solution"]["objective"] for side in (0, 1) for row in grouped[pair][side]])
            for pair in pairs
        ])
        balanced_supply = sum(min(len(grouped[pair][0]), len(grouped[pair][1])) for pair in pairs)
        choices.append((-len(feature_counts), median_score, -balanced_supply, pairs))
    if not choices:
        raise RuntimeError(f"No diverse {PAIR_COUNT}-pair training design was available; eligible={eligible}")
    return list(min(choices)[-1])


def explain_selected(
    rows: list[dict[str, Any]], model: Any, train: pd.DataFrame
) -> dict[tuple[str, int], dict[str, Any]]:
    def predict_array(values: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(values, columns=FEATURES))

    explainer = LimeTabularExplainer(
        train[FEATURES].to_numpy(float),
        feature_names=DISPLAY,
        class_names=LABELS,
        mode="classification",
        discretize_continuous=True,
        sample_around_instance=True,
        feature_selection="lasso_path",
        random_state=42,
    )
    output = {}
    for record in rows:
        original = record["original"]
        prediction = record["prediction"]
        explainer.random_state = np.random.RandomState(42)
        explanation = explainer.explain_instance(
            original,
            predict_array,
            labels=(prediction,),
            num_features=2,
            num_samples=500,
        )
        signed = np.zeros(5, dtype=float)
        for index, weight in explanation.local_exp[prediction]:
            signed[int(index)] = float(weight if prediction == 1 else -weight)
        shown = sorted(range(5), key=lambda index: -abs(signed[index]))[:2]
        output[(record["source_split"], record["source_instance_id"])] = {
            "values": signed,
            "shown": shown,
            "conditions": list(explanation.domain_mapper.discretized_feature_names),
            "fidelity": float(explanation.score),
        }
    return output


def make_payload(
    record: dict[str, Any],
    assigned_id: int,
    role: str,
    lime: dict[str, Any],
    model: Any,
    ranges: np.ndarray,
) -> dict[str, Any]:
    original = record["original"]
    solution = record["solution"]
    edited = solution["edited"]
    prediction = record["prediction"]
    probabilities = model.predict_proba(pd.DataFrame([original], columns=FEATURES))[0]
    if role == "training":
        changed = list(solution["pair"])
        source = "actionable_plausible_weighted_l1_exact_two"
    else:
        changed = [solution["feature"]]
        source = "preview_best_non_glucose_single_feature"
    attribution_values = lime["values"].tolist()
    return {
        "dataset": "diabetes",
        "model": "mlp",
        "xai_method": "lime",
        "xai_type": "attribution",
        "split": "train" if role == "training" else "test",
        "explanation_feature_count": 2,
        "instance_id": assigned_id,
        "available_instance_count": 392,
        "feature_names": DISPLAY,
        "raw_feature_names": FEATURES,
        "feature_types": ["numerical"] * 5,
        "feature_ranges": ranges.tolist(),
        "raw_feature_ranges": ranges.tolist(),
        "feature_values": original.tolist(),
        "raw_feature_values": original.tolist(),
        "prediction": {
            "value": prediction,
            "label": LABELS[prediction],
            "probabilities": [
                {"label": LABELS[index], "value": float(probabilities[index])}
                for index in (0, 1)
            ],
        },
        "prediction_labels": LABELS,
        "feature_importance_by_name": dict(zip(FEATURES, attribution_values)),
        "counterfactual_settings": {
            "mode": "actionable_plausible_weighted_l1",
            "controllable_only": True,
            "controllable_feature_names": DISPLAY[:4],
            "raw_controllable_feature_names": FEATURES[:4],
            "preview_immutable_feature_names": ["Glucose"] if role == "testing" else [],
            "preview_max_changed_features": 1 if role == "testing" else None,
        },
        "attribution": {
            "method": "lime",
            "feature_selection": {
                "method": "discrete_lime_quartile_bins_lasso_path",
                "num_features": 2,
                "num_samples": 500,
                "discretize_continuous": True,
            },
            "values": attribution_values,
            "ranking_values": [abs(value) for value in attribution_values],
            "raw_values": attribution_values,
            "max_abs_value": max(abs(value) for value in attribution_values),
            "shown_feature_count": 2,
            "shown_feature_indices": lime["shown"],
            "direction_labels": {"left": "Diabetes", "right": "No Diabetes"},
            "feature_conditions": lime["conditions"],
            "featureConditions": lime["conditions"],
            "local_fidelity": lime["fidelity"],
        },
        "counterfactual": {
            "feature_values": edited.tolist(),
            "raw_feature_values": edited.tolist(),
            "prediction": {"value": 1 - prediction, "label": LABELS[1 - prediction]},
            "target_prediction": {"value": 1 - prediction, "label": LABELS[1 - prediction]},
            "target_probability": solution["target_probability"],
            "selected_feature_names": [DISPLAY[index] for index in changed],
            "raw_selected_feature_names": [FEATURES[index] for index in changed],
            "source": source,
            "generation_mode": "minimal",
            "optimization": {
                "objective": "actionability_weighted_normalized_L1 + 0.35 * target_class_3NN_Gower",
                "objective_value": solution["objective"],
                "normalized_l1": solution["proximity"],
                "actionability_weighted_l1": solution["actionable_l1"],
                "target_class_3nn_gower": solution["plausibility_distance"],
                "plausibility_reference": "opposing ground-truth class instances from the real training split",
                "constraints": {
                    "minimum_normalized_change_per_edited_feature": (
                        TRAINING_MIN_NORMALIZED_CHANGE if role == "training" else MIN_NORMALIZED_CHANGE
                    ),
                    "maximum_training_normalized_change_per_edited_feature": (
                        MAX_TRAINING_NORMALIZED_CHANGE if role == "training" else None
                    ),
                    "training_exact_changed_features": 2 if role == "training" else None,
                    "training_age_immutable": role == "training",
                    "testing_glucose_immutable": role == "testing",
                    "testing_max_changed_features": 1 if role == "testing" else None,
                },
            },
        },
        "feature_pair_key": "|".join(FEATURES[index] for index in changed),
        "feature_pair_names": [DISPLAY[index] for index in changed],
        "source_split": record["source_split"],
        "source_instance_id": record["source_instance_id"],
        "experimental_phase": role,
        "selection_cluster": record["cluster"],
        "selection_role": "diverse_pair_training" if role == "training" else "matched_single_feature_testing",
        "nearest_training_source": record.get("nearest_training_source"),
        "profile_distance_to_nearest_training": record.get("profile_distance_to_nearest_training"),
        "copy_transfer": record.get("copy_transfer"),
    }


def main() -> None:
    data, train = load_data()
    model = joblib.load(MODEL_PATH)
    with (ROOT / "static" / "experiment-data.json").open(encoding="utf-8") as source:
        existing = json.load(source)["datasets"]["diabetes"]
    ranges = np.asarray(existing["training_pool"][0]["raw_feature_ranges"], dtype=float)
    values = data[FEATURES].to_numpy(float)
    predictions = np.asarray(model.predict(data[FEATURES]), dtype=int)
    probabilities = np.asarray(model.predict_proba(data[FEATURES]), dtype=float)
    target_values = {target: target_reference(train, target) for target in (0, 1)}
    target_neighbors = {
        target: NearestNeighbors(n_neighbors=3, metric="manhattan").fit(normalized(target_values[target], ranges))
        for target in (0, 1)
    }
    class_medians = train.groupby("target")[FEATURES].median()
    target_directions = {
        target: np.sign(class_medians.loc[target].to_numpy(float) - class_medians.loc[1 - target].to_numpy(float))
        for target in (0, 1)
    }

    training_options = []
    pairs = list(itertools.combinations(ACTIONABLE_TRAINING_INDICES, 2))
    for prediction in (0, 1):
        positions = np.flatnonzero(predictions == prediction)
        positions = positions[np.argsort(np.abs(probabilities[positions, 1] - 0.5))][:50]
        for position in positions:
            original = values[position]
            for pair in pairs:
                solution = two_feature_solution(
                    model,
                    original,
                    prediction,
                    pair,
                    ranges,
                    target_values[1 - prediction],
                    target_neighbors[1 - prediction],
                    target_directions[1 - prediction],
                )
                if solution is not None:
                    training_options.append({
                        "position": int(position),
                        "source_split": str(data.iloc[position]["source_split"]),
                        "source_instance_id": int(data.iloc[position]["source_instance_id"]),
                        "prediction": prediction,
                        "original": original.copy(),
                        "solution": solution,
                    })
    chosen_pairs = select_pair_families(training_options)
    selected_training = []
    used_positions = set()
    for cluster, pair in enumerate(chosen_pairs, start=1):
        for prediction in (0, 1):
            candidates = sorted(
                (row for row in training_options if row["solution"]["pair"] == pair and row["prediction"] == prediction and row["position"] not in used_positions),
                key=lambda row: (row["solution"]["objective"], row["position"]),
            )
            chosen = candidates[:TRAINING_PER_PAIR_LABEL]
            if len(chosen) != TRAINING_PER_PAIR_LABEL:
                raise RuntimeError(f"Incomplete training cell for pair={pair}, prediction={prediction}")
            for row in chosen:
                row["cluster"] = cluster
                selected_training.append(row)
                used_positions.add(row["position"])

    single_options = []
    for position, (original, prediction) in enumerate(zip(values, predictions)):
        if position in used_positions:
            continue
        solution = one_feature_solution(
            model,
            original,
            int(prediction),
            ranges,
            target_values[1 - int(prediction)],
            target_neighbors[1 - int(prediction)],
            target_directions[1 - int(prediction)],
        )
        if solution is None:
            continue
        same_label_training = [row for row in selected_training if row["prediction"] == int(prediction)]
        distances = [float(np.mean(np.abs(normalized(original, ranges) - normalized(row["original"], ranges)))) for row in same_label_training]
        nearest = same_label_training[int(np.argmin(distances))]
        remembered_delta = nearest["solution"]["edited"] - nearest["original"]
        editable_pair = [index for index in nearest["solution"]["pair"] if index in TEST_EDITABLE_INDICES]
        copied_index = max(
            editable_pair,
            key=lambda index: abs(remembered_delta[index]) / (ranges[index, 1] - ranges[index, 0]),
        )
        copied = original.copy()
        copied[copied_index] = np.clip(
            original[copied_index] + remembered_delta[copied_index],
            ranges[copied_index, 0],
            ranges[copied_index, 1],
        )
        target = 1 - int(prediction)
        target_probability_before = float(probabilities[position, target])
        target_probability_after = float(
            model.predict_proba(pd.DataFrame([copied], columns=FEATURES))[0, target]
        )
        initial_gap = max(1e-12, 0.5 - target_probability_before)
        copy_progress = (target_probability_after - target_probability_before) / initial_gap
        single_options.append({
            "position": position,
            "source_split": str(data.iloc[position]["source_split"]),
            "source_instance_id": int(data.iloc[position]["source_instance_id"]),
            "prediction": int(prediction),
            "original": original.copy(),
            "solution": solution,
            "cluster": nearest["cluster"],
            "nearest_training_source": f"{nearest['source_split']}:{nearest['source_instance_id']}",
            "profile_distance_to_nearest_training": min(distances),
            "copy_transfer": {
                "training_feature": FEATURES[copied_index],
                "raw_delta": float(remembered_delta[copied_index]),
                "target_probability_before": target_probability_before,
                "target_probability_after": target_probability_after,
                "target_probability_gain": target_probability_after - target_probability_before,
                "boundary_progress": copy_progress,
                "capped_boundary_progress": float(np.clip(copy_progress, 0.0, 1.0)),
                "crossed_boundary": int(target_probability_after >= 0.5),
            },
        })

    selected_testing = []
    quotas = [5, 5]
    used_test_positions = set()
    for prediction in (0, 1):
        for cluster, quota in enumerate(quotas, start=1):
            candidates = sorted(
                (
                    row for row in single_options
                    if row["prediction"] == prediction
                    and row["cluster"] == cluster
                    and row["position"] not in used_test_positions
                    and row["profile_distance_to_nearest_training"] <= MAX_TEST_PROFILE_DISTANCE
                    and row["copy_transfer"]["target_probability_gain"] > 0
                ),
                key=lambda row: (
                    -row["copy_transfer"]["crossed_boundary"],
                    -row["copy_transfer"]["capped_boundary_progress"],
                    row["profile_distance_to_nearest_training"],
                    row["solution"]["objective"],
                    row["position"],
                ),
            )
            chosen = candidates[:quota]
            if len(chosen) != quota:
                raise RuntimeError(f"Incomplete testing cell for cluster={cluster}, prediction={prediction}")
            selected_testing.extend(chosen)
            used_test_positions.update(row["position"] for row in chosen)

    all_selected = selected_training + selected_testing
    lime_by_source = explain_selected(all_selected, model, train)
    training_payloads = []
    for assigned_id, record in zip(range(180100, 180112), sorted(selected_training, key=lambda row: (row["cluster"], row["prediction"], row["position"]))):
        lime = lime_by_source[(record["source_split"], record["source_instance_id"])]
        training_payloads.append(make_payload(record, assigned_id, "training", lime, model, ranges))
    testing_payloads = []
    for prediction, id_start in ((0, 180200), (1, 180300)):
        rows = sorted((row for row in selected_testing if row["prediction"] == prediction), key=lambda row: (row["cluster"], row["position"]))
        for assigned_id, record in zip(range(id_start, id_start + 10), rows):
            lime = lime_by_source[(record["source_split"], record["source_instance_id"])]
            testing_payloads.append(make_payload(record, assigned_id, "testing", lime, model, ranges))

    bundle = {
        "version": "local-actionable-plausible-one-feature-preview-v3-transfer-consistent",
        "generated_at": "2026-09-03",
        "default_model": "mlp",
        "datasets": {
            "diabetes": {
                "metadata": {
                    **{key: value for key, value in existing["metadata"].items() if not key.startswith("qualtrics_")},
                    "preview_training_ids": [row["instance_id"] for row in training_payloads],
                    "preview_testing_ids_by_prediction": {
                        "0": [row["instance_id"] for row in testing_payloads if row["prediction"]["value"] == 0],
                        "1": [row["instance_id"] for row in testing_payloads if row["prediction"]["value"] == 1],
                    },
                    "preview_training_pairs": ["|".join(FEATURES[index] for index in pair) for pair in chosen_pairs],
                    "preview_objective": "actionability-weighted normalized L1 + 0.35 × target-class 3-nearest-neighbour Gower distance",
                    "preview_plausibility_reference": "three nearest real training instances with the opposing ground-truth class label",
                    "preview_model": "smaller regularized MLP: hidden layers 8x4, alpha 1.0, early stopping",
                    "preview_training_constraints": "exactly two changed features; each normalized change is 0.15 to 0.25 and points toward the opposing real class median; Age excluded; valid opposite prediction",
                    "preview_testing_constraints": "Glucose disabled; at most one changed feature; normalized change >= 0.10 and points toward the opposing real class median",
                    "preview_pair_design": "two disjoint pairs: Glucose + BMI and Blood Pressure + Insulin; each retains a non-Glucose test-transfer feature",
                    "preview_testing_selection": (
                        f"same-label nearest training cluster, profile distance <= {MAX_TEST_PROFILE_DISTANCE:.2f}, "
                        "positive copied-delta target gain, then copied-delta boundary progress"
                    ),
                },
                "browser_model": browser_model_payload(model),
                "labels": LABELS,
                "training_pool": training_payloads,
                "test_pool": testing_payloads,
            }
        },
    }
    OUT_JSON.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    serialized = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    OUT_JS.write_text(
        "window.EXPERIMENTAL_ACTIONABLE_PREVIEW_DATA = " + serialized + ";\n"
        "window.EXPERIMENT_DATA = window.EXPERIMENTAL_ACTIONABLE_PREVIEW_DATA;\n",
        encoding="utf-8",
    )

    review = []
    for payload in training_payloads + testing_payloads:
        optimization = payload["counterfactual"]["optimization"]
        changed_indices = [
            payload["raw_feature_names"].index(name)
            for name in payload["counterfactual"]["raw_selected_feature_names"]
        ]
        normalized_changed_amounts = [
            abs(
                payload["counterfactual"]["raw_feature_values"][index] - payload["raw_feature_values"][index]
            ) / (payload["raw_feature_ranges"][index][1] - payload["raw_feature_ranges"][index][0])
            for index in changed_indices
        ]
        review.append({
            "phase": payload["experimental_phase"],
            "instance id": payload["instance_id"],
            "cluster": payload["selection_cluster"],
            "original label": payload["prediction"]["label"],
            "counterfactual changed features": " + ".join(payload["feature_pair_names"]),
            "normalized L1": optimization["normalized_l1"],
            "actionability-weighted L1": optimization["actionability_weighted_l1"],
            "target-class 3NN Gower": optimization["target_class_3nn_gower"],
            "smallest normalized changed amount": min(normalized_changed_amounts),
            "objective": optimization["objective_value"],
            "target probability": payload["counterfactual"]["target_probability"],
            "nearest training profile distance": payload.get("profile_distance_to_nearest_training"),
            "copied training feature": (payload.get("copy_transfer") or {}).get("training_feature"),
            "copied raw delta": (payload.get("copy_transfer") or {}).get("raw_delta"),
            "copied boundary progress": (payload.get("copy_transfer") or {}).get("boundary_progress"),
            "copied edit crosses boundary": (payload.get("copy_transfer") or {}).get("crossed_boundary"),
            "source": f"{payload['source_split']}:{payload['source_instance_id']}",
        })
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)

    summary = {
        "production_assets_modified": False,
        "training_cases": len(training_payloads),
        "testing_cases": len(testing_payloads),
        "training_pair_counts": dict(Counter(row["feature_pair_key"] for row in training_payloads)),
        "training_feature_counts": dict(Counter(name for row in training_payloads for name in row["counterfactual"]["raw_selected_feature_names"])),
        "testing_solution_feature_counts": dict(Counter(row["counterfactual"]["raw_selected_feature_names"][0] for row in testing_payloads)),
        "training_label_counts": dict(Counter(row["prediction"]["label"] for row in training_payloads)),
        "testing_label_counts": dict(Counter(row["prediction"]["label"] for row in testing_payloads)),
        "maximum_testing_profile_distance_to_training": max(row["profile_distance_to_nearest_training"] for row in testing_payloads),
        "testing_copy_transfer_flip_rate": float(np.mean([
            row["copy_transfer"]["crossed_boundary"] for row in testing_payloads
        ])),
        "testing_copy_transfer_mean_capped_boundary_progress": float(np.mean([
            row["copy_transfer"]["capped_boundary_progress"] for row in testing_payloads
        ])),
        "objective": bundle["datasets"]["diabetes"]["metadata"]["preview_objective"],
        "minimum_testing_normalized_change_per_edited_feature": MIN_NORMALIZED_CHANGE,
        "minimum_training_normalized_change_per_edited_feature": TRAINING_MIN_NORMALIZED_CHANGE,
        "maximum_training_normalized_change_per_edited_feature": MAX_TRAINING_NORMALIZED_CHANGE,
        "direction_reference": "difference between opposing ground-truth class medians in the real training split",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    OUT_REMINDER.write_text(
        "# Next iteration reminder\n\n"
        "Add a post-survey 1–10 rating for how actionable participants consider each feature: "
        "Glucose, Blood Pressure, Insulin, BMI, and Age.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
