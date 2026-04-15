from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainedModelArtifact:
    name: str
    estimator: Any
    artifact_path: Path
    metadata_path: Path
    class_labels: list[str]
    metrics: dict[str, float]


class BaseModelTrainer(ABC):
    name: str

    def get_artifact_paths(self, dataset_name: str) -> tuple[Path, Path]:
        artifact_path = SAVED_MODELS_DIR / f"{dataset_name}_{self.name}.joblib"
        metadata_path = SAVED_MODELS_DIR / f"{dataset_name}_{self.name}.json"
        return artifact_path, metadata_path

    def load(
        self,
        dataset_name: str,
        feature_names: list[str] | None = None,
        class_labels: list[str] | None = None,
    ) -> TrainedModelArtifact | None:
        artifact_path, metadata_path = self.get_artifact_paths(dataset_name)
        if not artifact_path.exists() or not metadata_path.exists():
            return None

        estimator = joblib.load(artifact_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_feature_names = metadata.get("feature_names")
        if feature_names is not None and saved_feature_names != feature_names:
            return None
        saved_class_labels = metadata.get("class_labels")
        if class_labels is not None and saved_class_labels != class_labels:
            return None
        return TrainedModelArtifact(
            name=self.name,
            estimator=estimator,
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            class_labels=list(metadata.get("class_labels", [])),
            metrics=dict(metadata.get("metrics", {})),
        )

    def save(
        self,
        dataset_name: str,
        estimator: Any,
        feature_names: list[str],
        class_labels: list[str],
        metrics: dict[str, float],
    ) -> TrainedModelArtifact:
        artifact_path, metadata_path = self.get_artifact_paths(dataset_name)
        temporary_artifact_path = artifact_path.with_suffix(f"{artifact_path.suffix}.tmp")
        joblib.dump(estimator, temporary_artifact_path)
        temporary_artifact_path.replace(artifact_path)

        temporary_metadata_path = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")
        temporary_metadata_path.write_text(
            json.dumps(
                {
                    "feature_names": feature_names,
                    "class_labels": class_labels,
                    "metrics": metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary_metadata_path.replace(metadata_path)
        return TrainedModelArtifact(
            name=self.name,
            estimator=estimator,
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            class_labels=class_labels,
            metrics=metrics,
        )

    def load_or_train(
        self,
        dataset_name: str,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        feature_names: list[str],
        feature_types: list[str],
        target_column: str,
        class_labels: list[str],
        force_retrain: bool = False,
    ) -> TrainedModelArtifact:
        if not force_retrain:
            cached_artifact = self.load(
                dataset_name,
                feature_names=feature_names,
                class_labels=class_labels,
            )
            if cached_artifact is not None:
                return cached_artifact

        X_train = train_df[feature_names]
        y_train = train_df[target_column]
        X_dev = dev_df[feature_names]
        y_dev = dev_df[target_column]

        estimator = self.build_estimator(feature_names, feature_types)
        self.fit_estimator(estimator, X_train, y_train, X_dev, y_dev)

        metrics = self.evaluate(estimator, X_dev, y_dev)
        return self.save(dataset_name, estimator, feature_names, class_labels, metrics)

    @abstractmethod
    def build_estimator(
        self,
        feature_names: list[str],
        feature_types: list[str],
    ) -> Any:
        raise NotImplementedError

    def fit_estimator(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
    ) -> None:
        estimator.fit(X_train, y_train)

    def evaluate(
        self,
        estimator: Any,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
    ) -> dict[str, float]:
        predictions = estimator.predict(X_dev)
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_dev, predictions)),
        }

        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(X_dev)
            metrics["log_loss"] = float(log_loss(y_dev, probabilities))

        return metrics
