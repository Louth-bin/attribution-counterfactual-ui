from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from .base import BaseModelTrainer


class XGBoostModelTrainer(BaseModelTrainer):
    name = "xgboost"

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
                ("numeric", "passthrough", numeric_features),
                (
                    "categorical",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    categorical_features,
                ),
            ]
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        )
