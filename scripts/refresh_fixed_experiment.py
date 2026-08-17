"""Recompute explanations for the already selected fixed 10+20 case IDs."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_static_experiment import (
    STATIC_JS,
    STATIC_JSON,
    actual_changed_raw_feature_names,
    build_browser_model,
    build_metadata,
    feature_pair_key,
    is_successful_counterfactual,
)
from src.pipeline import ExplanationPipeline


def main() -> None:
    previous = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
    pipeline = ExplanationPipeline()
    datasets = {}
    report = {}

    for dataset_name in ("housing", "safelimit"):
        old_bundle = previous["datasets"][dataset_name]
        pools = {}
        for pool_name, split in (("training_pool", "train"), ("test_pool", "test")):
            refreshed = []
            for old_payload in old_bundle[pool_name]:
                payload = pipeline.get_instance_payload(
                    dataset_name=dataset_name,
                    model_name="mlp",
                    xai_method_name="shap",
                    instance_id=int(old_payload["instance_id"]),
                    xai_type="attribution",
                    explanation_feature_count=2,
                    counterfactual_mode="minimal",
                    controllable_only=False,
                    split=split,
                )
                if not is_successful_counterfactual(payload):
                    raise RuntimeError(
                        f"{dataset_name} {split} instance {payload['instance_id']} "
                        "does not produce a successful two-change counterfactual"
                    )
                payload["feature_pair_key"] = feature_pair_key(payload)
                changed_names = actual_changed_raw_feature_names(payload)
                payload["feature_pair_names"] = [
                    payload["feature_names"][payload["raw_feature_names"].index(name)]
                    for name in changed_names
                ]
                refreshed.append(payload)
            if pool_name == "test_pool":
                refreshed.sort(key=lambda payload: int(payload["prediction"]["value"]))
            pools[pool_name] = refreshed

        training_labels = Counter(
            int(payload["prediction"]["value"])
            for payload in pools["training_pool"]
        )
        test_labels = Counter(
            int(payload["prediction"]["value"])
            for payload in pools["test_pool"]
        )
        training_pairs = Counter(
            payload["feature_pair_key"] for payload in pools["training_pool"]
        )
        if training_labels != {0: 5, 1: 5} or test_labels != {0: 10, 1: 10}:
            raise RuntimeError(
                f"{dataset_name} label balance changed: "
                f"training={training_labels}, test={test_labels}"
            )
        if len(training_pairs) != 10 or set(training_pairs.values()) != {1}:
            raise RuntimeError(
                f"{dataset_name} training pair balance changed: {training_pairs}"
            )

        datasets[dataset_name] = {
            "metadata": build_metadata(
                pipeline,
                dataset_name,
                pools["training_pool"],
                pools["test_pool"],
            ),
            "browser_model": build_browser_model(pipeline, dataset_name),
            **pools,
        }
        report[dataset_name] = {
            "training_ids": [p["instance_id"] for p in pools["training_pool"]],
            "test_ids": [p["instance_id"] for p in pools["test_pool"]],
            "training_labels": dict(training_labels),
            "test_labels": dict(test_labels),
            "training_pairs": dict(training_pairs),
        }

    bundle = {
        "version": "static-experiment-v7-grouped-test-directions",
        "generated_at": date.today().isoformat(),
        "default_model": "mlp",
        "datasets": datasets,
    }
    compact = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    STATIC_JSON.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    STATIC_JS.write_text(
        f"window.EXPERIMENT_DATA = {compact};\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
