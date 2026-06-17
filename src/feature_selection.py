from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance

from .ai_models.base import BaseModelTrainer
from .data_manager import DatasetBundle
from .runtime_config import FeatureSelectionConfig


FEATURE_SELECTION_CACHE_VERSION = "source_order_v1"


@dataclass
class FeatureSelectionResult:
    selected_feature_names: list[str]
    importance_by_feature: dict[str, float]


def select_top_features(
    dataset_bundle: DatasetBundle,
    model_name: str,
    model_trainer: BaseModelTrainer,
    selection_config: FeatureSelectionConfig,
    force_retrain: bool = False,
) -> FeatureSelectionResult:
    feature_names = list(dataset_bundle.feature_names)

    if (
        not selection_config.enabled
        or selection_config.top_n is None
        or selection_config.top_n >= len(feature_names)
    ):
        return FeatureSelectionResult(
            selected_feature_names=feature_names,
            importance_by_feature={feature_name: 0.0 for feature_name in feature_names},
        )

    if selection_config.method != "permutation":
        raise ValueError(
            f"Unsupported feature selection method '{selection_config.method}'."
        )
    if isinstance(selection_config.top_n, bool) or int(selection_config.top_n) <= 0:
        raise ValueError(
            "feature_selection.top_n must be a positive integer when feature selection is enabled."
        )

    cache_path = _selection_cache_path(dataset_bundle.dataset_dir, model_name)
    cache_key = {
        "dataset_name": dataset_bundle.dataset_name,
        "model_name": model_name,
        "feature_names": feature_names,
        "feature_types": dataset_bundle.feature_types,
        "class_labels": dataset_bundle.class_labels,
        "top_n": selection_config.top_n,
        "method": selection_config.method,
        "n_repeats": selection_config.n_repeats,
        "scoring": selection_config.scoring,
        "random_state": selection_config.random_state,
        "selection_order": FEATURE_SELECTION_CACHE_VERSION,
    }

    if cache_path.exists() and not (selection_config.force_recompute or force_retrain):
        cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached_payload.get("cache_key") == cache_key:
            return FeatureSelectionResult(
                selected_feature_names=list(cached_payload["selected_feature_names"]),
                importance_by_feature=dict(cached_payload["importance_by_feature"]),
            )

    estimator = model_trainer.build_estimator(
        feature_names=feature_names,
        feature_types=dataset_bundle.feature_types,
    )
    X_train = dataset_bundle.train_df[feature_names]
    y_train = dataset_bundle.train_df[dataset_bundle.target_column]
    X_dev = dataset_bundle.dev_df[feature_names]
    y_dev = dataset_bundle.dev_df[dataset_bundle.target_column]
    model_trainer.fit_estimator(estimator, X_train, y_train, X_dev, y_dev)

    importance_result = permutation_importance(
        estimator,
        X_dev,
        y_dev,
        n_repeats=selection_config.n_repeats,
        random_state=selection_config.random_state,
        scoring=selection_config.scoring,
    )
    ranking = sorted(
        zip(feature_names, importance_result.importances_mean.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    top_feature_names = [
        feature_name
        for feature_name, _ in ranking[: selection_config.top_n]
    ]
    top_feature_name_set = set(top_feature_names)
    selected_feature_names = [
        feature_name
        for feature_name in feature_names
        if feature_name in top_feature_name_set
    ]
    importance_by_feature = {
        feature_name: float(importance)
        for feature_name, importance in ranking
    }

    cache_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "selected_feature_names": selected_feature_names,
                "importance_by_feature": importance_by_feature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return FeatureSelectionResult(
        selected_feature_names=selected_feature_names,
        importance_by_feature=importance_by_feature,
    )


def _selection_cache_path(dataset_dir: Path, model_name: str) -> Path:
    return dataset_dir / f"feature_selection_{model_name}.json"
