"""Build the static housing, drink-driving, and diabetes experiment bundle.

Only profiles with successful two-attribute counterfactuals are retained. The
ten training profiles contain every possible pair of most influential features
exactly once and are balanced 5/5 by predicted label. The twenty test profiles
are balanced 10/10 by predicted label.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import ExplanationPipeline


STATIC_JSON = REPO_ROOT / "static" / "experiment-data.json"
STATIC_JS = REPO_ROOT / "static" / "experiment-data.js"
PAIR_SEPARATOR = "|"
DATASETS = ("housing", "safelimit", "diabetes")
TRAINING_TARGET_PER_PAIR = 1
TRAINING_CANDIDATE_RESERVE_PER_PAIR = 6
TRAINING_PAIR_SCAN_BATCH_SIZE = 500
TEST_TARGET_PER_PREDICTION = 10


def feature_pair_key(payload: dict[str, Any]) -> str:
    selected = actual_changed_raw_feature_names(payload)
    order = {name: index for index, name in enumerate(payload["raw_feature_names"])}
    return PAIR_SEPARATOR.join(sorted(selected, key=order.get))


def actual_changed_raw_feature_names(payload: dict[str, Any]) -> list[str]:
    counterfactual = payload.get("counterfactual") or {}
    original_values = payload.get("raw_feature_values", [])
    changed_values = counterfactual.get("raw_feature_values", [])
    changed: list[str] = []
    for feature_name, feature_type, original, updated in zip(
        payload.get("raw_feature_names", []),
        payload.get("feature_types", []),
        original_values,
        changed_values,
    ):
        if feature_type == "categorical":
            is_changed = str(original) != str(updated)
        else:
            is_changed = not np.isclose(
                float(original), float(updated), rtol=0.0, atol=1e-9
            )
        if is_changed:
            changed.append(feature_name)
    return changed


def is_successful_counterfactual(payload: dict[str, Any]) -> bool:
    counterfactual = payload.get("counterfactual")
    changed_features = actual_changed_raw_feature_names(payload)
    selected_features = counterfactual.get("raw_selected_feature_names", []) if counterfactual else []
    return bool(
        counterfactual
        and len(selected_features) == 2
        and len(changed_features) == 2
        and set(changed_features) == set(selected_features)
        and int(counterfactual["prediction"]["value"])
        != int(payload["prediction"]["value"])
    )


def all_feature_pair_keys(feature_names: list[str]) -> list[str]:
    return [
        PAIR_SEPARATOR.join((feature_names[left], feature_names[right]))
        for left in range(len(feature_names))
        for right in range(left + 1, len(feature_names))
    ]


def _positive_class_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[min(1, len(shap_values) - 1)], dtype=float)

    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3:
        return values[:, :, min(1, values.shape[2] - 1)]
    if values.ndim == 2:
        return values
    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


def _pair_key_from_attribution(
    values: np.ndarray,
    feature_names: list[str],
) -> str:
    ranked_indices = sorted(
        range(len(feature_names)),
        key=lambda index: (round(abs(float(values[index])), 6), -index),
        reverse=True,
    )
    selected_indices = sorted([
        index
        for index in ranked_indices
        if abs(float(values[index])) > 0
    ][:2])
    if len(selected_indices) != 2:
        return ""
    return PAIR_SEPARATOR.join(feature_names[index] for index in selected_indices)


def index_training_candidates_by_pair(
    pipeline: ExplanationPipeline,
    dataset_name: str,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    assets = pipeline.prepare_assets(dataset_name, "mlp")
    dataset = assets.dataset
    feature_names = list(dataset.feature_names)
    pair_keys = all_feature_pair_keys(feature_names)
    candidates_by_pair = {pair_key: [] for pair_key in pair_keys}
    feature_frame = dataset.train_df[feature_names]
    encoded_frame = feature_frame.copy()
    category_maps: dict[str, dict[int, str]] = {}
    for feature_name in feature_names:
        if pd.api.types.is_numeric_dtype(feature_frame[feature_name]):
            encoded_frame[feature_name] = feature_frame[feature_name].astype(float)
            continue
        categories = feature_frame[feature_name].astype(str).drop_duplicates().tolist()
        forward = {category: index for index, category in enumerate(categories)}
        category_maps[feature_name] = {
            index: category for category, index in forward.items()
        }
        encoded_frame[feature_name] = (
            feature_frame[feature_name].astype(str).map(forward).astype(float)
        )
    background = encoded_frame.sample(n=min(len(encoded_frame), 50), random_state=42)

    def wrapped_predict(encoded_rows: np.ndarray) -> np.ndarray:
        rows = pd.DataFrame(encoded_rows, columns=feature_names)
        for feature_name, inverse in category_maps.items():
            rows[feature_name] = (
                rows[feature_name]
                .round()
                .clip(0, len(inverse) - 1)
                .astype(int)
                .map(inverse)
            )
        rows = rows.astype(feature_frame.dtypes.to_dict())
        return assets.model_artifact.estimator.predict_proba(rows)

    explainer = shap.KernelExplainer(
        wrapped_predict,
        background.to_numpy(dtype=float),
        feature_names=feature_names,
    )
    candidate_ids = list(range(len(feature_frame)))
    random.Random(f"{dataset_name}:train:pair-scan-v1").shuffle(candidate_ids)
    stats: Counter[str] = Counter()

    for start in range(0, len(candidate_ids), TRAINING_PAIR_SCAN_BATCH_SIZE):
        batch_ids = candidate_ids[start:start + TRAINING_PAIR_SCAN_BATCH_SIZE]
        batch = encoded_frame.iloc[batch_ids]
        random_state = np.random.get_state()
        np.random.seed(42)
        try:
            raw_values = explainer.shap_values(
                batch.to_numpy(dtype=float),
                nsamples=2 ** len(feature_names),
                l1_reg=0.0,
                silent=True,
            )
        finally:
            np.random.set_state(random_state)

        positive_values = _positive_class_values(raw_values)
        for instance_id, values in zip(batch_ids, positive_values):
            pair_key = _pair_key_from_attribution(values, feature_names)
            if pair_key in candidates_by_pair:
                candidates_by_pair[pair_key].append(instance_id)
                stats[f"indexed_{pair_key}"] += 1
        stats["examined"] += len(batch_ids)

        if all(
            len(candidates_by_pair[pair_key]) >= TRAINING_CANDIDATE_RESERVE_PER_PAIR
            for pair_key in pair_keys
        ):
            break

    missing = [pair_key for pair_key, ids in candidates_by_pair.items() if not ids]
    if missing:
        raise RuntimeError(
            f"{dataset_name} training split has no natural candidates for pairs: {missing}"
        )
    return candidates_by_pair, dict(stats)


def _select_label_balanced_pair_cases(
    candidates_by_pair: dict[str, list[dict[str, Any]]],
    target_per_label: int,
) -> list[dict[str, Any]]:
    pair_keys = list(candidates_by_pair)

    def search(
        pair_index: int,
        label_counts: dict[int, int],
        selected: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        if pair_index == len(pair_keys):
            return list(selected) if all(
                label_counts[label] == target_per_label for label in (0, 1)
            ) else None

        remaining = len(pair_keys) - pair_index
        if any(label_counts[label] > target_per_label for label in (0, 1)):
            return None
        if any(
            label_counts[label] + remaining < target_per_label
            for label in (0, 1)
        ):
            return None

        pair_key = pair_keys[pair_index]
        for candidate in candidates_by_pair[pair_key]:
            label = int(candidate["prediction"]["value"])
            label_counts[label] += 1
            selected.append(candidate)
            result = search(pair_index + 1, label_counts, selected)
            if result is not None:
                return result
            selected.pop()
            label_counts[label] -= 1
        return None

    result = search(0, {0: 0, 1: 0}, [])
    if result is None:
        candidate_labels = {
            pair_key: [int(payload["prediction"]["value"]) for payload in candidates]
            for pair_key, candidates in candidates_by_pair.items()
        }
        raise RuntimeError(
            "Could not choose one case per feature pair with a 5/5 label balance; "
            f"candidate labels={candidate_labels}"
        )
    return result


def generate_balanced_training_pool(
    pipeline: ExplanationPipeline,
    dataset_name: str,
    target_per_pair: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates_by_pair, index_stats = index_training_candidates_by_pair(
        pipeline,
        dataset_name,
    )
    successful_by_pair: dict[str, list[dict[str, Any]]] = {
        pair_key: [] for pair_key in candidates_by_pair
    }
    stats: Counter[str] = Counter(index_stats)

    for required_pair, candidate_ids in candidates_by_pair.items():
        for instance_id in candidate_ids:
            payload = pipeline.get_instance_payload(
                dataset_name=dataset_name,
                model_name="mlp",
                xai_method_name="shap",
                instance_id=instance_id,
                xai_type="attribution",
                explanation_feature_count=2,
                counterfactual_mode="minimal",
                controllable_only=False,
                split="train",
            )
            stats[f"payload_examined_{required_pair}"] += 1
            if not is_successful_counterfactual(payload):
                stats[f"best_effort_{required_pair}"] += 1
                continue

            actual_pair = feature_pair_key(payload)
            if actual_pair != required_pair:
                stats[f"pair_mismatch_{required_pair}"] += 1
                continue

            payload["feature_pair_key"] = actual_pair
            changed_raw_names = actual_changed_raw_feature_names(payload)
            payload["feature_pair_names"] = [
                payload["feature_names"][payload["raw_feature_names"].index(name)]
                for name in changed_raw_names
            ]
            successful_by_pair[required_pair].append(payload)
            stats[f"included_{required_pair}"] += 1
            if len(successful_by_pair[required_pair]) >= TRAINING_CANDIDATE_RESERVE_PER_PAIR:
                break

        if len(successful_by_pair[required_pair]) < target_per_pair:
            raise RuntimeError(
                f"{dataset_name} training pool needs {target_per_pair} successful "
                f"profiles for {required_pair}, but found "
                f"{len(successful_by_pair[required_pair])}; "
                f"indexed candidates={len(candidate_ids)}"
            )

    generated = _select_label_balanced_pair_cases(
        successful_by_pair,
        target_per_label=len(successful_by_pair) // 2,
    )
    stats["selected_prediction_0"] = sum(
        int(payload["prediction"]["value"]) == 0 for payload in generated
    )
    stats["selected_prediction_1"] = sum(
        int(payload["prediction"]["value"]) == 1 for payload in generated
    )
    return generated, dict(stats)


def ordered_candidate_ids(
    prediction_ids: dict[str, list[int]],
    dataset_name: str,
    split: str,
) -> list[int]:
    rng = random.Random(f"{dataset_name}:{split}:static-pool-v1")
    groups = {label: list(instance_ids) for label, instance_ids in prediction_ids.items()}
    for instance_ids in groups.values():
        rng.shuffle(instance_ids)

    ordered: list[int] = []
    while any(groups.values()):
        for label in sorted(groups):
            if groups[label]:
                ordered.append(groups[label].pop())
    return ordered


def generate_pool(
    pipeline: ExplanationPipeline,
    dataset_name: str,
    split: str,
    target_per_prediction: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    metadata = pipeline.get_metadata(dataset_name, "mlp")
    prediction_ids = metadata["prediction_instance_ids_by_split"][split]
    candidate_ids = ordered_candidate_ids(prediction_ids, dataset_name, split)
    accepted_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    generated: list[dict[str, Any]] = []

    for instance_id in candidate_ids:
        if all(accepted_counts[str(label)] >= target_per_prediction for label in (0, 1)):
            break

        payload = pipeline.get_instance_payload(
            dataset_name=dataset_name,
            model_name="mlp",
            xai_method_name="shap",
            instance_id=instance_id,
            xai_type="attribution",
            explanation_feature_count=2,
            counterfactual_mode="minimal",
            controllable_only=False,
            split=split,
        )
        prediction_key = str(int(payload["prediction"]["value"]))
        stats[f"examined_{prediction_key}"] += 1
        if accepted_counts[prediction_key] >= target_per_prediction:
            continue
        if not is_successful_counterfactual(payload):
            stats[f"best_effort_{prediction_key}"] += 1
            continue

        payload["feature_pair_key"] = feature_pair_key(payload)
        changed_raw_names = actual_changed_raw_feature_names(payload)
        payload["feature_pair_names"] = [
            payload["feature_names"][payload["raw_feature_names"].index(name)]
            for name in changed_raw_names
        ]
        generated.append(payload)
        accepted_counts[prediction_key] += 1
        stats[f"included_{prediction_key}"] += 1

    missing = {
        label: target_per_prediction - accepted_counts[str(label)]
        for label in (0, 1)
        if accepted_counts[str(label)] < target_per_prediction
    }
    if missing:
        raise RuntimeError(
            f"{dataset_name} {split} pool lacks successful counterfactuals: "
            f"{missing}; stats={dict(stats)}"
        )
    if split == "test":
        generated.sort(key=lambda payload: int(payload["prediction"]["value"]))
    return generated, dict(stats)


def build_metadata(
    pipeline: ExplanationPipeline,
    dataset_name: str,
    training_pool: list[dict[str, Any]],
    test_pool: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = pipeline.get_metadata(dataset_name, "mlp")
    raw_names = training_pool[0]["raw_feature_names"]
    pair_keys = all_feature_pair_keys(raw_names)
    metadata.update({
        "raw_feature_names": raw_names,
        "all_feature_pair_keys": pair_keys,
        "training_pair_counts": dict(Counter(p["feature_pair_key"] for p in training_pool)),
        "static_training_pool_count": len(training_pool),
        "static_test_pool_count": len(test_pool),
        "supported_explanations": ["attribution", "counterfactual", "none"],
    })
    return metadata


def build_browser_model(
    pipeline: ExplanationPipeline,
    dataset_name: str,
) -> dict[str, Any]:
    assets = pipeline.prepare_assets(dataset_name, "mlp")
    estimator = assets.model_artifact.estimator
    preprocessor = estimator.named_steps["preprocessor"]
    mlp = estimator.named_steps["model"]
    feature_names = list(assets.dataset.feature_names)
    numeric_features = list(preprocessor.transformers_[0][2])
    categorical_features = list(preprocessor.transformers_[1][2])
    if sorted(numeric_features + categorical_features) != sorted(feature_names):
        raise RuntimeError(f"{dataset_name} browser export has unsupported preprocessing.")

    scaler = preprocessor.named_transformers_["numeric"]
    one_hot_encoder = preprocessor.named_transformers_["categorical"]
    if mlp.out_activation_ != "logistic" or len(mlp.classes_) != 2:
        raise RuntimeError(
            f"{dataset_name} browser export requires a binary logistic MLP."
        )

    return {
        "format": "sklearn-mlp-binary-v1",
        "feature_names": feature_names,
        "class_labels": list(assets.dataset.class_labels),
        "classes": [int(value) for value in mlp.classes_.tolist()],
        "preprocessing": {
            "type": "column-transformer-v1",
            "numeric": {
                "feature_names": numeric_features,
                "mean": [float(value) for value in scaler.mean_.tolist()],
                "scale": [float(value) for value in scaler.scale_.tolist()],
            },
            "categorical": {
                "feature_names": categorical_features,
                "categories": [
                    [_json_safe_category(value) for value in categories.tolist()]
                    for categories in getattr(one_hot_encoder, "categories_", [])
                ],
            },
        },
        "hidden_activation": str(mlp.activation),
        "output_activation": str(mlp.out_activation_),
        "layers": [
            {
                "weights": [
                    [float(value) for value in row]
                    for row in weights.tolist()
                ],
                "biases": [float(value) for value in biases.tolist()],
            }
            for weights, biases in zip(mlp.coefs_, mlp.intercepts_)
        ],
    }


def _json_safe_category(value: Any) -> Any:
    return value.item() if isinstance(value, (np.floating, np.integer)) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=STATIC_JSON)
    parser.add_argument("--output-js", type=Path, default=STATIC_JS)
    parser.add_argument(
        "--preserve-training",
        action="store_true",
        help="Keep the existing fixed training pools while regenerating testing pools.",
    )
    parser.add_argument(
        "--regenerate-training",
        action="append",
        default=[],
        choices=DATASETS,
        metavar="DATASET",
        help="Regenerate this dataset's training pool even with --preserve-training.",
    )
    args = parser.parse_args()

    pipeline = ExplanationPipeline()
    previous = (
        json.loads(args.output_json.read_text(encoding="utf-8"))
        if args.preserve_training and args.output_json.exists()
        else None
    )
    datasets: dict[str, Any] = {}
    report: dict[str, Any] = {}
    for dataset_name in DATASETS:
        if (
            previous is not None
            and dataset_name in previous.get("datasets", {})
            and dataset_name not in set(args.regenerate_training)
        ):
            training_pool = previous["datasets"][dataset_name]["training_pool"]
            training_labels = Counter(
                int(payload["prediction"]["value"]) for payload in training_pool
            )
            training_pairs = Counter(
                payload["feature_pair_key"] for payload in training_pool
            )
            if (
                len(training_pool) != 10
                or training_labels != {0: 5, 1: 5}
                or len(training_pairs) != 10
                or set(training_pairs.values()) != {1}
            ):
                raise RuntimeError(
                    f"{dataset_name} existing training pool is not the fixed "
                    "balanced ten-case design"
                )
            training_stats = {"preserved": len(training_pool)}
        else:
            training_pool, training_stats = generate_balanced_training_pool(
                pipeline, dataset_name, TRAINING_TARGET_PER_PAIR
            )
        test_pool, test_stats = generate_pool(
            pipeline, dataset_name, "test", TEST_TARGET_PER_PREDICTION
        )
        datasets[dataset_name] = {
            "metadata": build_metadata(
                pipeline, dataset_name, training_pool, test_pool
            ),
            "browser_model": build_browser_model(pipeline, dataset_name),
            "training_pool": training_pool,
            "test_pool": test_pool,
        }
        report[dataset_name] = {
            "training": training_stats,
            "test": test_stats,
            "training_pairs": datasets[dataset_name]["metadata"]["training_pair_counts"],
        }

    bundle = {
        "version": "static-experiment-v10-three-domains",
        "generated_at": date.today().isoformat(),
        "default_model": "mlp",
        "datasets": datasets,
    }
    compact = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    args.output_json.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_js.write_text(
        f"window.EXPERIMENT_DATA = {compact};\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
