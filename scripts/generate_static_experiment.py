"""Regenerate the static experiment bundle without running models in production.

The exporter uses the stored datasets, trained models, and explanation pipeline. For
the diabetes condition it exports every source profile whose displayed numerical
attributes are non-zero and whose generated counterfactual is also non-zero. The
existing SafeLimit bundle is preserved unless this script is extended deliberately.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import ExplanationPipeline


STATIC_JSON = REPO_ROOT / "static" / "experiment-data.json"
STATIC_JS = REPO_ROOT / "static" / "experiment-data.js"
PAIR_SEPARATOR = "|"


def contains_numeric_zero(values: list[Any], feature_types: list[str]) -> bool:
    return any(
        feature_type == "numerical" and float(value) == 0
        for value, feature_type in zip(values, feature_types)
    )


def feature_pair_key(payload: dict[str, Any]) -> str:
    selected = payload["counterfactual"]["raw_selected_feature_names"]
    order = {name: index for index, name in enumerate(payload["raw_feature_names"])}
    return PAIR_SEPARATOR.join(sorted(selected, key=order.get))


def generate_diabetes_pool(
    pipeline: ExplanationPipeline,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    assets = pipeline.prepare_assets("diabetes", "mlp")
    frame = assets.dataset.train_df if split == "train" else assets.dataset.test_df
    feature_names = assets.dataset.feature_names
    stats: Counter[str] = Counter()
    generated: list[dict[str, Any]] = []

    for instance_id in range(len(frame)):
        source_values = frame.iloc[instance_id][feature_names].tolist()
        if contains_numeric_zero(source_values, assets.dataset.feature_types):
            stats["source_zero"] += 1
            continue

        payload = pipeline.get_instance_payload(
            dataset_name="diabetes",
            model_name="mlp",
            xai_method_name="shap",
            instance_id=instance_id,
            xai_type="attribution",
            explanation_feature_count=2,
            counterfactual_mode="minimal",
            controllable_only=False,
            split=split,
        )
        counterfactual = payload.get("counterfactual")
        if not counterfactual or len(counterfactual.get("raw_selected_feature_names", [])) != 2:
            stats["invalid_counterfactual"] += 1
            continue
        if contains_numeric_zero(counterfactual["raw_feature_values"], payload["feature_types"]):
            stats["counterfactual_zero"] += 1
            continue

        payload["feature_pair_key"] = feature_pair_key(payload)
        payload["feature_pair_names"] = counterfactual["selected_feature_names"]
        generated.append(payload)
        stats["included"] += 1

    return generated, dict(stats)


def build_diabetes_metadata(
    pipeline: ExplanationPipeline,
    training_pool: list[dict[str, Any]],
    test_pool: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = pipeline.get_metadata("diabetes", "mlp")
    raw_names = training_pool[0]["raw_feature_names"]
    pair_keys = [
        PAIR_SEPARATOR.join((raw_names[left], raw_names[right]))
        for left in range(len(raw_names))
        for right in range(left + 1, len(raw_names))
    ]
    metadata.update({
        "raw_feature_names": raw_names,
        "all_feature_pair_keys": pair_keys,
        "training_pair_counts": dict(Counter(p["feature_pair_key"] for p in training_pool)),
        "static_training_pool_count": len(training_pool),
        "static_test_pool_count": len(test_pool),
        "supported_explanations": ["attribution", "counterfactual", "none"],
    })
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=STATIC_JSON)
    parser.add_argument("--output-js", type=Path, default=STATIC_JS)
    args = parser.parse_args()

    existing = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
    pipeline = ExplanationPipeline()
    training_pool, training_stats = generate_diabetes_pool(pipeline, "train")
    test_pool, test_stats = generate_diabetes_pool(pipeline, "test")
    if not training_pool or not test_pool:
        raise RuntimeError("Regeneration produced an empty diabetes pool.")

    diabetes = {
        "metadata": build_diabetes_metadata(pipeline, training_pool, test_pool),
        "training_pool": training_pool,
        "test_pool": test_pool,
    }
    bundle = {
        **existing,
        "version": "static-experiment-v2-zero-free-diabetes",
        "generated_at": date.today().isoformat(),
        "datasets": {**existing["datasets"], "diabetes": diabetes},
    }
    compact = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    args.output_json.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_js.write_text(f"window.EXPERIMENT_DATA = {compact};\n", encoding="utf-8")
    print(json.dumps({
        "training": training_stats,
        "test": test_stats,
        "training_pairs": diabetes["metadata"]["training_pair_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
