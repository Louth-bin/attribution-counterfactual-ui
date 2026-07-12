from __future__ import annotations

from .base import BaseModelTrainer
from .mlp_model import MLPModelTrainer
from .xgboost_model import XGBoostModelTrainer


MODEL_REGISTRY: dict[str, BaseModelTrainer] = {
    "mlp": MLPModelTrainer(),
    "xgboost": XGBoostModelTrainer(),
}


def get_model_trainer(model_name: str) -> BaseModelTrainer:
    normalized_name = model_name.lower()
    if normalized_name not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model '{model_name}'. Available models: {available_models}."
        )

    return MODEL_REGISTRY[normalized_name]


def list_model_names() -> list[str]:
    return sorted(MODEL_REGISTRY)
