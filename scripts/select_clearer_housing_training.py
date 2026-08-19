"""Select a disjoint, slightly clearer housing training set for review.

The hard design constraints match the current training pool: ten cases, five
per predicted label, one successful two-feature counterfactual for every
possible feature pair. Among feasible sets, this selects the smallest
improvement in monotonic counterfactual consistency over the current set and
then prefers the smaller total normalized counterfactual change.
"""

from __future__ import annotations

import json
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_static_experiment import (
    STATIC_JSON,
    _pair_key_from_attribution,
    actual_changed_raw_feature_names,
    feature_pair_key,
    is_successful_counterfactual,
)
from src.pipeline import (
    ATTRIBUTION_CACHE_VERSION,
    ExplanationPipeline,
    _attribution_cache_key,
    _attribution_disk_cache_path,
    _cache_safe_value,
    _split_row_ids,
)


OUTPUT = REPO_ROOT / "qualtrics" / "housing-training-alternative.json"
CANDIDATE_TARGET_PER_PAIR = 3


def cached_candidate_ids_by_pair(
    pipeline: ExplanationPipeline,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Recover candidate pair assignments from the existing SHAP disk cache."""
    assets = pipeline.prepare_assets("housing", "mlp")
    dataset = assets.dataset
    feature_frame = dataset.train_df[dataset.feature_names]
    artifact_path = getattr(assets.model_artifact, "artifact_path", None)
    artifact_signature = (
        (str(artifact_path), artifact_path.stat().st_mtime_ns)
        if artifact_path is not None and artifact_path.exists()
        else None
    )
    base_key = (
        ATTRIBUTION_CACHE_VERSION,
        "housing",
        "mlp",
        "shap",
        artifact_signature,
        tuple(dataset.feature_names),
        tuple(dataset.class_labels),
        tuple(_split_row_ids(dataset.train_df)),
    )
    # json.dumps(tuple) renders a JSON array. Pre-hashing its constant prefix
    # avoids serializing the full train-row-id list once per candidate row.
    base_json = json.dumps(base_key, sort_keys=True)
    prefix_hasher = hashlib.sha256((base_json[:-1] + ", ").encode("utf-8"))
    cache_dir = dataset.dataset_dir / "explanation_cache"
    cache_paths = {path.stem: path for path in cache_dir.glob("*.json")}
    candidates: dict[str, list[int]] = {}

    for instance_id in range(len(feature_frame)):
        values = tuple(
            _cache_safe_value(feature_frame.iloc[instance_id][name])
            for name in dataset.feature_names
        )
        digest = prefix_hasher.copy()
        digest.update((json.dumps(values, sort_keys=True) + "]").encode("utf-8"))
        cache_path = cache_paths.get(digest.hexdigest())
        if cache_path is None:
            continue
        attribution = json.loads(cache_path.read_text(encoding="utf-8"))
        pair = _pair_key_from_attribution(
            np.asarray(attribution["values"], dtype=float),
            list(dataset.feature_names),
        )
        if pair:
            candidates.setdefault(pair, []).append(instance_id)

    # Guard the optimized digest construction against cache-key drift.
    if candidates:
        sample_id = next(iter(next(iter(candidates.values()))))
        sample_frame = feature_frame.iloc[[sample_id]]
        official_path = _attribution_disk_cache_path(
            dataset,
            _attribution_cache_key(
                "housing", "mlp", "shap", assets.model_artifact, dataset, sample_frame
            ),
        )
        if not official_path.exists():
            raise RuntimeError("Optimized attribution cache lookup does not match pipeline keys")

    return candidates, {
        "cached_attributions": sum(len(ids) for ids in candidates.values()),
        **{f"cached_{pair}": len(ids) for pair, ids in candidates.items()},
    }


def annotate(payload: dict[str, Any]) -> dict[str, Any]:
    payload["feature_pair_key"] = feature_pair_key(payload)
    changed_names = actual_changed_raw_feature_names(payload)
    payload["feature_pair_names"] = [
        payload["feature_names"][payload["raw_feature_names"].index(name)]
        for name in changed_names
    ]
    return payload


def clarity(payload: dict[str, Any]) -> int:
    """Count changes aligned with larger/more -> Expensive."""
    target = int(payload["counterfactual"]["prediction"]["value"])
    expected_sign = 1 if target == 1 else -1
    return sum(
        (float(updated) - float(original)) * expected_sign > 0
        for original, updated in zip(
            payload["raw_feature_values"],
            payload["counterfactual"]["raw_feature_values"],
        )
        if float(updated) != float(original)
    )


def objective(payload: dict[str, Any]) -> float:
    return float(payload["counterfactual"]["optimization"]["objective_value"])


def changes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    target = int(payload["counterfactual"]["prediction"]["value"])
    expected_sign = 1 if target == 1 else -1
    result = []
    for name, original, updated in zip(
        payload["feature_names"],
        payload["feature_values"],
        payload["counterfactual"]["feature_values"],
    ):
        difference = float(updated) - float(original)
        if difference == 0:
            continue
        result.append({
            "feature": name,
            "from": original,
            "to": updated,
            "direction_consistent": difference * expected_sign > 0,
        })
    return result


def main() -> None:
    current_bundle = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
    current = current_bundle["datasets"]["housing"]["training_pool"]
    current_ids = {int(payload["instance_id"]) for payload in current}
    current_clarity = sum(clarity(payload) for payload in current)

    pipeline = ExplanationPipeline()
    candidate_ids_by_pair, index_stats = cached_candidate_ids_by_pair(pipeline)
    candidates_by_pair: dict[str, list[dict[str, Any]]] = {}
    generation_stats: Counter[str] = Counter(index_stats)

    for required_pair, candidate_ids in candidate_ids_by_pair.items():
        candidates = []
        for instance_id in candidate_ids:
            if instance_id in current_ids:
                generation_stats[f"excluded_current_{required_pair}"] += 1
                continue
            payload = pipeline.get_instance_payload(
                dataset_name="housing",
                model_name="mlp",
                xai_method_name="shap",
                instance_id=instance_id,
                xai_type="attribution",
                explanation_feature_count=2,
                counterfactual_mode="minimal",
                controllable_only=False,
                split="train",
            )
            generation_stats[f"examined_{required_pair}"] += 1
            if not is_successful_counterfactual(payload):
                generation_stats[f"best_effort_{required_pair}"] += 1
                continue
            annotate(payload)
            if payload["feature_pair_key"] != required_pair:
                generation_stats[f"pair_mismatch_{required_pair}"] += 1
                continue
            candidates.append(payload)
            if len(candidates) >= CANDIDATE_TARGET_PER_PAIR:
                break
        if not candidates:
            raise RuntimeError(f"No disjoint successful candidate for {required_pair}")
        candidates_by_pair[required_pair] = candidates

    # State: (cheap count, expensive count, clarity) -> (objective, cases).
    states: dict[tuple[int, int, int], tuple[float, list[dict[str, Any]]]] = {
        (0, 0, 0): (0.0, [])
    }
    for pair_key, candidates in candidates_by_pair.items():
        next_states: dict[
            tuple[int, int, int], tuple[float, list[dict[str, Any]]]
        ] = {}
        for (cheap_count, expensive_count, score), (total_objective, selected) in states.items():
            for candidate in candidates:
                label = int(candidate["prediction"]["value"])
                new_counts = [cheap_count, expensive_count]
                new_counts[label] += 1
                if new_counts[label] > 5:
                    continue
                key = (
                    new_counts[0],
                    new_counts[1],
                    score + clarity(candidate),
                )
                value = (total_objective + objective(candidate), selected + [candidate])
                if key not in next_states or value[0] < next_states[key][0]:
                    next_states[key] = value
        states = next_states

    feasible_scores = sorted(
        score for cheap, expensive, score in states if (cheap, expensive) == (5, 5)
    )
    clearer_scores = [score for score in feasible_scores if score > current_clarity]
    if not clearer_scores:
        raise RuntimeError(
            f"No disjoint set improves on current clarity {current_clarity}/20; "
            f"feasible scores={feasible_scores}"
        )
    selected_clarity = min(clearer_scores)
    selected_objective, selected = states[(5, 5, selected_clarity)]

    pair_counts = Counter(payload["feature_pair_key"] for payload in selected)
    label_counts = Counter(int(payload["prediction"]["value"]) for payload in selected)
    if set(pair_counts.values()) != {1} or label_counts != {0: 5, 1: 5}:
        raise RuntimeError(
            f"Invalid selected design: pairs={pair_counts}, labels={label_counts}"
        )

    report = {
        "selection_rule": {
            "cases": 10,
            "labels": {"Cheap": 5, "Expensive": 5},
            "feature_pairs": "Every possible pair exactly once",
            "counterfactual": "Successful and changes exactly two attributes",
            "overlap_with_current_training_ids": 0,
            "clarity_metric": (
                "Number of changed attributes whose direction agrees with "
                "larger/more -> Expensive"
            ),
        },
        "comparison": {
            "current_consistent_changes": current_clarity,
            "current_total_changes": 20,
            "alternative_consistent_changes": selected_clarity,
            "alternative_total_changes": 20,
            "feasible_alternative_scores": feasible_scores,
            "alternative_total_normalized_change": selected_objective,
        },
        "summary": [
            {
                "instance_id": int(payload["instance_id"]),
                "prediction": payload["prediction"]["label"],
                "counterfactual_prediction": payload["counterfactual"]["prediction"]["label"],
                "feature_pair": payload["feature_pair_key"],
                "consistent_changes": clarity(payload),
                "changes": changes(payload),
            }
            for payload in selected
        ],
        "training_pool": selected,
        "generation_stats": dict(generation_stats),
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selection_rule", "comparison", "summary")}, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
