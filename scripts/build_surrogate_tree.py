"""Fit and report a shallow decision-tree surrogate for a saved AI model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import ExplanationPipeline


def evaluate_split(
    surrogate: DecisionTreeClassifier,
    feature_frame: pd.DataFrame,
    ai_predictions: np.ndarray,
    true_labels: pd.Series,
) -> dict[str, Any]:
    surrogate_predictions = surrogate.predict(feature_frame)
    return {
        "count": int(len(feature_frame)),
        "fidelity": float(accuracy_score(ai_predictions, surrogate_predictions)),
        "cohen_kappa": float(cohen_kappa_score(ai_predictions, surrogate_predictions)),
        "ai_by_surrogate_confusion_matrix": confusion_matrix(
            ai_predictions,
            surrogate_predictions,
            labels=[0, 1],
        ).astype(int).tolist(),
        "surrogate_label_accuracy": float(
            accuracy_score(true_labels, surrogate_predictions)
        ),
        "ai_label_accuracy": float(accuracy_score(true_labels, ai_predictions)),
    }


def bootstrap_fidelity_interval(
    matches: np.ndarray,
    random_state: int,
    repetitions: int = 10_000,
) -> list[float]:
    rng = np.random.default_rng(random_state)
    scores = np.asarray([
        matches[rng.integers(0, len(matches), len(matches))].mean()
        for _ in range(repetitions)
    ])
    return [float(value) for value in np.quantile(scores, [0.025, 0.975])]


def evaluate_static_pool(
    surrogate: DecisionTreeClassifier,
    feature_names: list[str],
    dataset_name: str,
    pool_name: str,
) -> dict[str, Any] | None:
    bundle_path = REPO_ROOT / "static" / "experiment-data.json"
    if not bundle_path.exists():
        return None
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    pool = bundle.get("datasets", {}).get(dataset_name, {}).get(pool_name, [])
    if not pool:
        return None

    feature_frame = pd.DataFrame(
        [payload["raw_feature_values"] for payload in pool],
        columns=feature_names,
    )
    ai_predictions = np.asarray([
        int(payload["prediction"]["value"])
        for payload in pool
    ])
    surrogate_predictions = surrogate.predict(feature_frame)
    return {
        "count": int(len(pool)),
        "fidelity": float(accuracy_score(ai_predictions, surrogate_predictions)),
        "ai_by_surrogate_confusion_matrix": confusion_matrix(
            ai_predictions,
            surrogate_predictions,
            labels=[0, 1],
        ).astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="housing")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-model", type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()

    output_stem = f"{args.dataset}_{args.model}_surrogate_depth{args.max_depth}"
    output_dir = REPO_ROOT / "src" / "ai_models" / "saved_models"
    output_model = args.output_model or output_dir / f"{output_stem}.joblib"
    output_report = args.output_report or output_dir / f"{output_stem}.json"

    pipeline = ExplanationPipeline()
    assets = pipeline.prepare_assets(args.dataset, args.model)
    dataset = assets.dataset
    ai_model = assets.model_artifact.estimator
    feature_names = list(dataset.feature_names)

    train_features = dataset.train_df[feature_names]
    train_ai_predictions = ai_model.predict(train_features)
    surrogate = DecisionTreeClassifier(
        max_depth=args.max_depth,
        random_state=args.random_state,
    ).fit(train_features, train_ai_predictions)

    split_frames = {
        "train": dataset.train_df,
        "dev": dataset.dev_df,
        "test": dataset.test_df,
    }
    split_metrics: dict[str, Any] = {}
    for split_name, split_frame in split_frames.items():
        features = split_frame[feature_names]
        ai_predictions = ai_model.predict(features)
        split_metrics[split_name] = evaluate_split(
            surrogate=surrogate,
            feature_frame=features,
            ai_predictions=ai_predictions,
            true_labels=split_frame[dataset.target_column],
        )

    test_features = dataset.test_df[feature_names]
    test_matches = (
        surrogate.predict(test_features) == ai_model.predict(test_features)
    ).astype(float)
    split_metrics["test"]["fidelity_bootstrap_95_interval"] = (
        bootstrap_fidelity_interval(test_matches, args.random_state)
    )

    used_feature_indices = sorted(
        int(index)
        for index in set(surrogate.tree_.feature.tolist())
        if index >= 0
    )
    report = {
        "dataset": args.dataset,
        "surrogate_for_model": args.model,
        "surrogate_type": "DecisionTreeClassifier",
        "training_target": "AI model predictions",
        "max_depth": args.max_depth,
        "fitted_depth": int(surrogate.get_depth()),
        "leaf_count": int(surrogate.get_n_leaves()),
        "feature_names": feature_names,
        "features_used": [feature_names[index] for index in used_feature_indices],
        "class_labels": list(dataset.class_labels),
        "split_metrics": split_metrics,
        "static_training_pool": evaluate_static_pool(
            surrogate,
            feature_names,
            args.dataset,
            "training_pool",
        ),
        "static_test_pool": evaluate_static_pool(
            surrogate,
            feature_names,
            args.dataset,
            "test_pool",
        ),
        "rules": export_text(
            surrogate,
            feature_names=feature_names,
            decimals=2,
        ),
    }

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(surrogate, output_model)
    output_report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"Saved model: {output_model}")
    print(f"Saved report: {output_report}")


if __name__ == "__main__":
    main()
