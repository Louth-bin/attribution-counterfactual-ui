from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import BaseModelTrainer


class MLPModelTrainer(BaseModelTrainer):
    name = "mlp"

    def build_estimator(
        self,
        feature_names: list[str],
        feature_types: list[str],
    ) -> Pipeline:
        numeric_features = [
            feature_name
            for feature_name, feature_type in zip(feature_names, feature_types)
            if feature_type == "numerical"
        ]
        categorical_features = [
            feature_name
            for feature_name, feature_type in zip(feature_names, feature_types)
            if feature_type == "categorical"
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric_features),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ]
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=0.0005,
                        batch_size=32,
                        learning_rate_init=0.001,
                        max_iter=800,
                        random_state=42,
                    ),
                ),
            ]
        )
