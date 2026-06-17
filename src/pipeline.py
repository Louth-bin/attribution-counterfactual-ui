from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .ai_models import get_model_trainer, list_model_names
from .data_manager import (
    DatasetBundle,
    display_feature_values,
    ensure_dataset_bundle,
    exclude_features_from_dataset_bundle,
    subset_dataset_bundle,
)
from .feature_selection import select_top_features
from .runtime_config import get_dataset_runtime_config
from .xai_methods import generate_counterfactual, get_xai_method, list_xai_methods


ATTRIBUTION_CACHE_LIMIT = 256
ATTRIBUTION_RANKING_DECIMALS = 6
ATTRIBUTION_CACHE_VERSION = "raw_shap_v2"
LOGGER = logging.getLogger("counterfactual.pipeline")


@dataclass
class PreparedAssets:
    dataset: DatasetBundle
    model_artifact: Any
    feature_importance_by_name: dict[str, float]


class ExplanationPipeline:
    def __init__(self) -> None:
        self._raw_attribution_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def prepare_assets(
        self,
        dataset_name: str,
        model_name: str,
        force_resplit: bool = False,
        force_retrain: bool = False,
    ) -> PreparedAssets:
        dataset_bundle = ensure_dataset_bundle(
            dataset_name=dataset_name,
            force_resplit=force_resplit,
        )
        runtime_config = get_dataset_runtime_config(dataset_name)
        dataset_bundle = exclude_features_from_dataset_bundle(
            dataset_bundle=dataset_bundle,
            excluded_feature_names=runtime_config.excluded_feature_names,
        )
        model_trainer = get_model_trainer(model_name)
        feature_selection_result = select_top_features(
            dataset_bundle=dataset_bundle,
            model_name=model_name,
            model_trainer=model_trainer,
            selection_config=runtime_config.feature_selection,
            force_retrain=force_retrain,
        )
        dataset_bundle = subset_dataset_bundle(
            dataset_bundle=dataset_bundle,
            selected_feature_names=feature_selection_result.selected_feature_names,
        )
        model_artifact = model_trainer.load_or_train(
            dataset_name=dataset_bundle.dataset_name,
            train_df=dataset_bundle.train_df,
            dev_df=dataset_bundle.dev_df,
            feature_names=dataset_bundle.feature_names,
            feature_types=dataset_bundle.feature_types,
            target_column=dataset_bundle.target_column,
            class_labels=dataset_bundle.class_labels,
            force_retrain=force_retrain,
        )
        return PreparedAssets(
            dataset=dataset_bundle,
            model_artifact=model_artifact,
            feature_importance_by_name=feature_selection_result.importance_by_feature,
        )

    def get_metadata(
        self,
        dataset_name: str,
        model_name: str | None = None,
        force_resplit: bool = False,
    ) -> dict[str, Any]:
        prepared_assets: PreparedAssets | None = None
        if model_name:
            prepared_assets = self.prepare_assets(
                dataset_name=dataset_name,
                model_name=model_name,
                force_resplit=force_resplit,
            )
            dataset_bundle = prepared_assets.dataset
        else:
            dataset_bundle = ensure_dataset_bundle(
                dataset_name=dataset_name,
                force_resplit=force_resplit,
            )
            runtime_config = get_dataset_runtime_config(dataset_name)
            dataset_bundle = exclude_features_from_dataset_bundle(
                dataset_bundle=dataset_bundle,
                excluded_feature_names=runtime_config.excluded_feature_names,
            )

        payload = {
            "dataset": dataset_bundle.dataset_name,
            "available_instance_count": dataset_bundle.available_instance_count,
            "train_instance_count": len(dataset_bundle.train_df),
            "dev_instance_count": len(dataset_bundle.dev_df),
            "test_instance_count": len(dataset_bundle.test_df),
            "feature_names": dataset_bundle.feature_display_names,
            "feature_types": dataset_bundle.feature_types,
            "models": list_model_names(),
            "xai_methods": list_xai_methods(),
            "prediction_labels": dataset_bundle.class_labels,
        }

        if prepared_assets:
            prediction_instance_ids_by_split = _get_prediction_instance_ids_by_split(
                dataset_bundle=dataset_bundle,
                estimator=prepared_assets.model_artifact.estimator,
            )
            payload["model"] = model_name
            payload["prediction_instance_ids_by_split"] = prediction_instance_ids_by_split
            payload["prediction_counts_by_split"] = {
                split: {
                    prediction: len(instance_ids)
                    for prediction, instance_ids in split_groups.items()
                }
                for split, split_groups in prediction_instance_ids_by_split.items()
            }

        return payload

    def get_instance_payload(
        self,
        dataset_name: str,
        model_name: str,
        xai_method_name: str,
        instance_id: int,
        xai_type: str = "attribution",
        explanation_feature_count: int = 3,
        counterfactual_mode: str = "minimal",
        controllable_only: bool = False,
        split: str = "test",
        force_resplit: bool = False,
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        prepared_assets = self.prepare_assets(
            dataset_name=dataset_name,
            model_name=model_name,
            force_resplit=force_resplit,
            force_retrain=force_retrain,
        )
        dataset_bundle = prepared_assets.dataset
        estimator = prepared_assets.model_artifact.estimator
        runtime_config = get_dataset_runtime_config(dataset_name)
        split_frame = _get_split_frame(dataset_bundle, split)

        if instance_id < 0 or instance_id >= len(split_frame):
            raise IndexError(
                f"Instance id {instance_id} is outside the {split} split range "
                f"0-{len(split_frame) - 1}."
            )

        instance_frame = split_frame.iloc[[instance_id]].copy()
        feature_frame = instance_frame[dataset_bundle.feature_names]

        prediction_value = int(estimator.predict(feature_frame)[0])
        probabilities: list[float] | None = None
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(feature_frame)[0].tolist()

        attribution = self._get_raw_attribution(
            dataset_name=dataset_bundle.dataset_name,
            model_name=model_name,
            xai_method_name=xai_method_name,
            prepared_assets=prepared_assets,
            feature_frame=feature_frame,
        )
        raw_attribution_values = list(attribution.get("values", []))
        attribution["raw_values"] = raw_attribution_values
        eligible_feature_indices = _eligible_counterfactual_indices(
            feature_names=dataset_bundle.feature_names,
            controllable_feature_names=runtime_config.controllable_feature_names,
            controllable_only=controllable_only,
        )
        shown_feature_indices = _top_k_indices(
            values=raw_attribution_values,
            top_k=explanation_feature_count,
            eligible_indices=eligible_feature_indices,
        )
        attribution["values"] = _keep_top_k_values(
            values=raw_attribution_values,
            top_k=explanation_feature_count,
            selected_indices=shown_feature_indices,
        )
        attribution["max_abs_value"] = max(
            [abs(value) for value in attribution["values"]],
            default=1e-9,
        )
        attribution["shown_feature_count"] = explanation_feature_count
        attribution["shown_feature_indices"] = shown_feature_indices
        attribution["direction_labels"] = {
            "left": dataset_bundle.class_labels[0]
            if dataset_bundle.class_labels
            else "Class 0",
            "right": dataset_bundle.class_labels[1]
            if len(dataset_bundle.class_labels) > 1
            else "Class 1",
        }

        counterfactual = generate_counterfactual(
            estimator=estimator,
            reference_frame=feature_frame,
            target_distribution_frame=dataset_bundle.train_df[
                dataset_bundle.train_df[dataset_bundle.target_column] == (1 - prediction_value)
            ][dataset_bundle.feature_names],
            feature_names=dataset_bundle.feature_names,
            feature_types=dataset_bundle.feature_types,
            feature_ranges=dataset_bundle.feature_ranges,
            class_labels=dataset_bundle.class_labels,
            shap_values=raw_attribution_values,
            top_k=explanation_feature_count,
            selected_feature_indices=shown_feature_indices,
            generation_mode=counterfactual_mode,
        )

        prediction_probabilities = []
        if probabilities is not None:
            prediction_probabilities = [
                {
                    "label": dataset_bundle.class_labels[index]
                    if index < len(dataset_bundle.class_labels)
                    else f"Class {index}",
                    "value": float(probability),
                }
                for index, probability in enumerate(probabilities)
            ]

        raw_feature_values = [
            _json_safe_value(feature_frame.iloc[0][feature_name])
            for feature_name in dataset_bundle.feature_names
        ]
        display_feature_values_payload = display_feature_values(
            dataset_bundle=dataset_bundle,
            feature_names=dataset_bundle.feature_names,
            values=raw_feature_values,
        )
        display_counterfactual = None
        if counterfactual is not None:
            display_counterfactual = {
                **counterfactual,
                "raw_feature_values": counterfactual["feature_values"],
                "feature_values": display_feature_values(
                    dataset_bundle=dataset_bundle,
                    feature_names=dataset_bundle.feature_names,
                    values=counterfactual["feature_values"],
                ),
                "selected_feature_names": [
                    dataset_bundle.feature_display_names[
                        dataset_bundle.feature_names.index(feature_name)
                    ]
                    for feature_name in counterfactual.get("selected_feature_names", [])
                    if feature_name in dataset_bundle.feature_names
                ],
                "raw_selected_feature_names": counterfactual.get(
                    "selected_feature_names",
                    [],
                ),
            }

        return {
            "dataset": dataset_bundle.dataset_name,
            "model": model_name.lower(),
            "xai_method": xai_method_name.lower(),
            "xai_type": xai_type.lower(),
            "split": _normalize_split_name(split),
            "explanation_feature_count": explanation_feature_count,
            "instance_id": int(instance_id),
            "available_instance_count": len(split_frame),
            "feature_names": dataset_bundle.feature_display_names,
            "raw_feature_names": dataset_bundle.feature_names,
            "feature_types": dataset_bundle.feature_types,
            "feature_ranges": dataset_bundle.display_feature_ranges,
            "raw_feature_ranges": dataset_bundle.feature_ranges,
            "feature_values": display_feature_values_payload,
            "raw_feature_values": raw_feature_values,
            "prediction": {
                "value": prediction_value,
                "label": dataset_bundle.class_labels[prediction_value]
                if prediction_value < len(dataset_bundle.class_labels)
                else str(prediction_value),
                "probabilities": prediction_probabilities,
            },
            "prediction_labels": dataset_bundle.class_labels,
            "feature_importance_by_name": prepared_assets.feature_importance_by_name,
            "counterfactual_settings": {
                "mode": _normalize_counterfactual_mode(counterfactual_mode),
                "controllable_only": controllable_only,
                "controllable_feature_names": [
                    dataset_bundle.feature_display_names[
                        dataset_bundle.feature_names.index(feature_name)
                    ]
                    for feature_name in runtime_config.controllable_feature_names
                    if feature_name in dataset_bundle.feature_names
                ],
                "raw_controllable_feature_names": [
                    feature_name
                    for feature_name in runtime_config.controllable_feature_names
                    if feature_name in dataset_bundle.feature_names
                ],
            },
            "attribution": attribution,
            "counterfactual": display_counterfactual,
            "model_metrics": prepared_assets.model_artifact.metrics,
        }

    def _get_raw_attribution(
        self,
        dataset_name: str,
        model_name: str,
        xai_method_name: str,
        prepared_assets: PreparedAssets,
        feature_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        dataset_bundle = prepared_assets.dataset
        model_artifact = prepared_assets.model_artifact
        cache_key = _attribution_cache_key(
            dataset_name=dataset_name,
            model_name=model_name,
            xai_method_name=xai_method_name,
            model_artifact=model_artifact,
            dataset_bundle=dataset_bundle,
            feature_frame=feature_frame,
        )
        cached_attribution = self._raw_attribution_cache.get(cache_key)
        if cached_attribution is not None:
            return copy.deepcopy(cached_attribution)

        disk_cache_path = _attribution_disk_cache_path(
            dataset_bundle=dataset_bundle,
            cache_key=cache_key,
        )
        if disk_cache_path.exists():
            cached_attribution = json.loads(
                disk_cache_path.read_text(encoding="utf-8")
            )
            self._remember_raw_attribution(cache_key, cached_attribution)
            return copy.deepcopy(cached_attribution)

        xai_method = get_xai_method(xai_method_name)
        attribution = xai_method.explain(
            estimator=model_artifact.estimator,
            background_frame=dataset_bundle.train_df[dataset_bundle.feature_names],
            instance_frame=feature_frame,
            positive_class_index=min(1, len(dataset_bundle.class_labels) - 1),
        )
        self._remember_raw_attribution(cache_key, attribution)
        try:
            disk_cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_cache_path = disk_cache_path.with_suffix(
                f"{disk_cache_path.suffix}.tmp"
            )
            temporary_cache_path.write_text(
                json.dumps(attribution, indent=2),
                encoding="utf-8",
            )
            temporary_cache_path.replace(disk_cache_path)
        except OSError as error:
            LOGGER.warning(
                "Skipping attribution disk cache write for %s: %s",
                disk_cache_path,
                error,
            )

        return attribution

    def _remember_raw_attribution(
        self,
        cache_key: tuple[Any, ...],
        attribution: dict[str, Any],
    ) -> None:
        self._raw_attribution_cache[cache_key] = copy.deepcopy(attribution)
        if len(self._raw_attribution_cache) > ATTRIBUTION_CACHE_LIMIT:
            oldest_key = next(iter(self._raw_attribution_cache))
            self._raw_attribution_cache.pop(oldest_key, None)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _attribution_cache_key(
    dataset_name: str,
    model_name: str,
    xai_method_name: str,
    model_artifact: Any,
    dataset_bundle: DatasetBundle,
    feature_frame: pd.DataFrame,
) -> tuple[Any, ...]:
    artifact_path = getattr(model_artifact, "artifact_path", None)
    artifact_signature = None
    if artifact_path is not None and artifact_path.exists():
        artifact_signature = (
            str(artifact_path),
            artifact_path.stat().st_mtime_ns,
        )

    return (
        ATTRIBUTION_CACHE_VERSION,
        dataset_name.lower(),
        model_name.lower(),
        xai_method_name.lower(),
        artifact_signature,
        tuple(dataset_bundle.feature_names),
        tuple(dataset_bundle.class_labels),
        tuple(_split_row_ids(dataset_bundle.train_df)),
        tuple(
            _cache_safe_value(feature_frame.iloc[0][feature_name])
            for feature_name in dataset_bundle.feature_names
        ),
    )


def _attribution_disk_cache_path(
    dataset_bundle: DatasetBundle,
    cache_key: tuple[Any, ...],
) -> Any:
    cache_payload = json.dumps(cache_key, sort_keys=True)
    cache_digest = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
    return dataset_bundle.dataset_dir / "explanation_cache" / f"{cache_digest}.json"


def _split_row_ids(dataframe: pd.DataFrame) -> list[Any]:
    if "row_id" in dataframe.columns:
        return [
            _cache_safe_value(value)
            for value in dataframe["row_id"].tolist()
        ]

    return [
        _cache_safe_value(value)
        for value in dataframe.index.tolist()
    ]


def _cache_safe_value(value: Any) -> Any:
    value = _json_safe_value(value)
    if isinstance(value, float):
        return round(value, 12)
    return str(value)


def _normalize_counterfactual_mode(mode: str) -> str:
    normalized_mode = str(mode or "minimal").strip().lower()
    if normalized_mode in {"prototype", "prototypical", "distribution"}:
        return "prototypical"
    return "minimal"


def _normalize_split_name(split: str) -> str:
    normalized_split = str(split or "test").strip().lower()
    if normalized_split in {"training", "train"}:
        return "train"
    if normalized_split in {"development", "validation", "dev"}:
        return "dev"
    if normalized_split == "test":
        return "test"
    raise ValueError("Split must be one of: train, dev, test.")


def _get_split_frame(dataset_bundle: DatasetBundle, split: str) -> pd.DataFrame:
    normalized_split = _normalize_split_name(split)
    if normalized_split == "train":
        return dataset_bundle.train_df
    if normalized_split == "dev":
        return dataset_bundle.dev_df
    return dataset_bundle.test_df


def _get_prediction_instance_ids_by_split(
    dataset_bundle: DatasetBundle,
    estimator: Any,
) -> dict[str, dict[str, list[int]]]:
    prediction_groups_by_split: dict[str, dict[str, list[int]]] = {}
    for split in ("train", "dev", "test"):
        split_frame = _get_split_frame(dataset_bundle, split)
        feature_frame = split_frame[dataset_bundle.feature_names]
        predictions = estimator.predict(feature_frame)
        split_groups: dict[str, list[int]] = {}
        for instance_id, prediction in enumerate(predictions):
            split_groups.setdefault(str(int(prediction)), []).append(instance_id)
        prediction_groups_by_split[split] = split_groups
    return prediction_groups_by_split


def _eligible_counterfactual_indices(
    feature_names: list[str],
    controllable_feature_names: tuple[str, ...],
    controllable_only: bool,
) -> list[int] | None:
    if not controllable_only:
        return None

    controllable_features = set(controllable_feature_names)
    return [
        index
        for index, feature_name in enumerate(feature_names)
        if feature_name in controllable_features
    ]


def _top_k_indices(
    values: list[float],
    top_k: int,
    eligible_indices: list[int] | None = None,
) -> list[int]:
    if top_k <= 0:
        return []

    candidate_indices = eligible_indices if eligible_indices is not None else list(range(len(values)))
    ranked_indices = sorted(
        candidate_indices,
        key=lambda index: (
            round(abs(values[index]), ATTRIBUTION_RANKING_DECIMALS),
            -index,
        ),
        reverse=True,
    )
    return [
        index
        for index in ranked_indices
        if abs(values[index]) > 0
    ][:top_k]


def _keep_top_k_values(
    values: list[float],
    top_k: int,
    selected_indices: list[int] | None = None,
) -> list[float]:
    if top_k <= 0:
        return [0.0 for _ in values]

    selected_indices = set(selected_indices or _top_k_indices(values, top_k))
    return [
        float(value) if index in selected_indices and abs(value) > 0 else 0.0
        for index, value in enumerate(values)
    ]
