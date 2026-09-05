"""Prospectively select a boundary-teaching diabetes experiment.

Selection is deliberately independent of participant outcomes and cognitive-
model predictions.  It uses only the classifier-generated, two-feature
minimal-counterfactual geometry in the existing candidate banks.  The output
is a provisional analysis bundle; it does not modify the live/static study.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_recourse_v12 import build_payload  # noqa: E402
from scripts.generate_static_experiment import build_metadata  # noqa: E402
from src.pipeline import ExplanationPipeline  # noqa: E402


SOURCE_BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
CANDIDATE_FILES = (
    ROOT / "analysis" / "diabetes_hierarchical_oriented_v6_reselected_candidate_shortlist.csv",
    ROOT / "analysis" / "diabetes_hierarchical_oriented_q90_anchor_search_candidate_shortlist.csv",
)
OUTPUT_DIR = ROOT / "outputs" / "v20-boundary-teaching-selection"
OUTPUT_BUNDLE = OUTPUT_DIR / "boundary_teaching_candidate_bundle.json"
OUTPUT_SELECTION = OUTPUT_DIR / "boundary_teaching_selection.json"

FEATURES = ("glucose", "blood_pressure", "insulin", "bmi", "age")
PAIRS = ("glucose | bmi", "blood_pressure | insulin")
TRAIN_COUNTS = {
    ("glucose | bmi", "No Diabetes"): 4,
    ("glucose | bmi", "Diabetes"): 4,
    ("blood_pressure | insulin", "No Diabetes"): 2,
    ("blood_pressure | insulin", "Diabetes"): 2,
}
TEST_COUNTS = {
    ("glucose | bmi", "No Diabetes"): 6,
    ("glucose | bmi", "Diabetes"): 6,
    ("blood_pressure | insulin", "No Diabetes"): 4,
    ("blood_pressure | insulin", "Diabetes"): 4,
}
RANDOM_SEED = 20260831
SEARCH_DRAWS = 50000


def read_candidates() -> list[dict[str, Any]]:
    by_source: dict[tuple[str, int], dict[str, str]] = {}
    for path in CANDIDATE_FILES:
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if row["main_pair"] in PAIRS:
                    by_source[(row["source_split"], int(row["instance_id"]))] = row

    source_dataset = json.loads(SOURCE_BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    ranges = np.asarray(source_dataset["training_pool"][0]["feature_ranges"], dtype=float)
    low, span = ranges[:, 0], ranges[:, 1] - ranges[:, 0]
    output: list[dict[str, Any]] = []
    for row in by_source.values():
        raw = json.loads(row["original_profile"])
        profile = np.clip(
            (np.asarray([float(raw[name]) for name in FEATURES]) - low) / span,
            0.0,
            1.0,
        )
        delta = np.asarray(json.loads(row["normalized_change_vector"]), dtype=float)
        cf_distance = float(row["counterfactual_distance"])
        boundary_distance = float(row["local_boundary_distance"])
        target_probability = float(row["counterfactual_target_probability"])
        output.append({
            "row": row,
            "key": (row["source_split"], int(row["instance_id"])),
            "stratum": (row["main_pair"], row["prediction_label"]),
            "profile": profile,
            "delta": delta,
            "cf_distance": cf_distance,
            "boundary_distance": boundary_distance,
            "target_probability": target_probability,
            "margin_error": abs(target_probability - 0.5),
            "minimality_gap": max(cf_distance - boundary_distance, 0.0),
        })
    return output


def training_quality(case: dict[str, Any]) -> float:
    distance = float(case["cf_distance"])
    distance_penalty = max(0.04 - distance, 0.0) + max(distance - 0.32, 0.0)
    relative_gap = float(case["minimality_gap"]) / max(distance, 0.03)
    return 7.0 * float(case["margin_error"]) + 0.30 * relative_gap + 2.0 * distance_penalty


def transfer_score(case: dict[str, Any], training: list[dict[str, Any]]) -> float:
    eligible = [item for item in training if item["stratum"] == case["stratum"]]
    distances = np.asarray([
        np.mean(np.abs(case["profile"] - item["profile"])) for item in eligible
    ])
    weights = np.exp(-8.0 * distances)
    weights /= weights.sum()
    predicted_delta = np.sum(
        np.vstack([item["delta"] for item in eligible]) * weights[:, None], axis=0
    )
    transfer_error = float(np.sum(np.abs(predicted_delta - case["delta"])))
    relative_error = transfer_error / max(float(case["cf_distance"]), 0.05)
    nearest_source = float(np.min(distances))
    margin_error = float(case["margin_error"])
    difficulty_penalty = (
        max(0.035 - float(case["cf_distance"]), 0.0)
        + max(float(case["cf_distance"]) - 0.34, 0.0)
    )
    return relative_error + 0.35 * nearest_source + 5.0 * margin_error + 2.0 * difficulty_penalty


def choose_tests(
    candidates: list[dict[str, Any]], training: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float] | None:
    training_keys = {item["key"] for item in training}
    selected: list[dict[str, Any]] = []
    for stratum, count in TEST_COUNTS.items():
        ranked = sorted(
            (
                ({**item, "transfer_score": transfer_score(item, training)})
                for item in candidates
                if item["stratum"] == stratum and item["key"] not in training_keys
            ),
            key=lambda item: float(item["transfer_score"]),
        )
        if len(ranked) < count:
            return None
        selected.extend(ranked[:count])
    distances = np.asarray([float(item["cf_distance"]) for item in selected])
    spread_penalty = max(0.08 - float(np.std(distances)), 0.0)
    objective = float(np.mean([item["transfer_score"] for item in selected])) + 2.0 * spread_penalty
    return selected, objective


def search(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    rng = random.Random(RANDOM_SEED)
    by_stratum = {
        stratum: [item for item in candidates if item["stratum"] == stratum]
        for stratum in TRAIN_COUNTS
    }
    best: tuple[float, list[dict[str, Any]], list[dict[str, Any]]] | None = None
    for _ in range(SEARCH_DRAWS):
        training = []
        for stratum, count in TRAIN_COUNTS.items():
            pool = by_stratum[stratum]
            weights = [math.exp(-3.0 * training_quality(item)) for item in pool]
            keys = list(range(len(pool)))
            chosen = []
            for _position in range(count):
                index = rng.choices(keys, weights=[weights[key] for key in keys], k=1)[0]
                chosen.append(pool[index])
                keys.remove(index)
            training.extend(chosen)
        result = choose_tests(candidates, training)
        if result is None:
            continue
        testing, test_objective = result
        train_distances = np.asarray([float(item["cf_distance"]) for item in training])
        training_objective = float(np.mean([training_quality(item) for item in training]))
        diversity_penalty = max(0.075 - float(np.std(train_distances)), 0.0)
        objective = test_objective + 0.45 * training_objective + diversity_penalty
        if best is None or objective < best[0]:
            best = objective, training, testing
    if best is None:
        raise RuntimeError("No feasible selection found")
    return best[1], best[2], best[0]


def selected_row(item: dict[str, Any], phase: str, rank: int) -> dict[str, str]:
    row = dict(item["row"])
    row["experimental_phase"] = phase
    row["cluster_rank"] = str(rank)
    row["selection_role"] = "prospective_boundary_teaching_geometry"
    return row


def make_payloads(
    training: list[dict[str, Any]], testing: list[dict[str, Any]]
) -> tuple[list[dict], list[dict]]:
    pipeline = ExplanationPipeline()
    training_payloads = [
        build_payload(pipeline, selected_row(item, "training", rank), 170100 + rank)
        for rank, item in enumerate(training)
    ]
    label_positions: Counter[str] = Counter()
    testing_payloads = []
    for rank, item in enumerate(testing):
        prediction = item["row"]["prediction_label"]
        assigned = (170200 if prediction == "Diabetes" else 170300) + label_positions[prediction]
        label_positions[prediction] += 1
        testing_payloads.append(
            build_payload(pipeline, selected_row(item, "testing", rank), assigned)
        )
    return training_payloads, testing_payloads


def serializable(item: dict[str, Any], phase: str, assigned_id: int) -> dict[str, Any]:
    return {
        "phase": phase,
        "assigned_instance_id": assigned_id,
        "source_split": item["key"][0],
        "source_instance_id": item["key"][1],
        "feature_pair": item["stratum"][0],
        "prediction_label": item["stratum"][1],
        "target_label": item["row"]["target_label"],
        "original_boundary_distance": item["boundary_distance"],
        "minimal_cf_L1": item["cf_distance"],
        "minimal_cf_target_probability": item["target_probability"],
        "boundary_margin_error": item["margin_error"],
        "unrestricted_vs_two_feature_gap": item["minimality_gap"],
        "geometry_transfer_score": item.get("transfer_score"),
        "original_profile": json.loads(item["row"]["original_profile"]),
        "minimal_cf_profile": json.loads(item["row"]["counterfactual_profile"]),
        "normalized_minimal_cf_change": item["delta"].tolist(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = read_candidates()
    training, testing, objective = search(candidates)
    training.sort(key=lambda item: (item["stratum"], item["key"]))
    testing.sort(key=lambda item: (item["stratum"], item["key"]))
    training_payloads, testing_payloads = make_payloads(training, testing)

    source = json.loads(SOURCE_BUNDLE.read_text(encoding="utf-8"))
    source_dataset = source["datasets"]["diabetes"]
    analysis_dataset = {
        "metadata": build_metadata(
            ExplanationPipeline(), "diabetes", training_payloads, testing_payloads
        ),
        "browser_model": source_dataset["browser_model"],
        "training_pool": training_payloads,
        "test_pool": testing_payloads,
    }
    OUTPUT_BUNDLE.write_text(
        json.dumps({
            "version": "prospective-boundary-teaching-v0.1",
            "generated_at": date.today().isoformat(),
            "default_model": "mlp",
            "datasets": {"diabetes": analysis_dataset},
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    training_rows = [
        serializable(item, "training", payload["instance_id"])
        for item, payload in zip(training, training_payloads)
    ]
    testing_rows = [
        serializable(item, "testing", payload["instance_id"])
        for item, payload in zip(testing, testing_payloads)
    ]
    summary = {
        "analysis": "prospective geometry-only boundary-teaching selection",
        "participant outcomes used": False,
        "cognitive model used for selection": False,
        "random seed": RANDOM_SEED,
        "search draws": SEARCH_DRAWS,
        "candidate count": len(candidates),
        "objective": objective,
        "training count": len(training_rows),
        "testing count": len(testing_rows),
        "training": training_rows,
        "testing": testing_rows,
    }
    OUTPUT_SELECTION.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "candidate_count": len(candidates),
        "objective": objective,
        "training": [(row["source_split"], row["source_instance_id"]) for row in training_rows],
        "testing": [(row["source_split"], row["source_instance_id"]) for row in testing_rows],
        "bundle": str(OUTPUT_BUNDLE.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
