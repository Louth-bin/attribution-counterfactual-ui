from __future__ import annotations

import numpy as np
import pandas as pd

from src.xai_methods.counterfactual import generate_counterfactual


class LinearProbabilityEstimator:
    def __init__(self, weights: tuple[float, float], threshold: float) -> None:
        self.weights = np.asarray(weights, dtype=float)
        self.threshold = float(threshold)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        scores = frame[["x1", "x2"]].to_numpy(dtype=float) @ self.weights
        logits = 8.0 * (scores - self.threshold)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        scores = frame[["x1", "x2"]].to_numpy(dtype=float) @ self.weights
        return (scores >= self.threshold).astype(int)


def target_distribution() -> pd.DataFrame:
    values = np.linspace(0.0, 1.0, 1001)
    return pd.DataFrame({"x1": values, "x2": values})


def test_selected_features_change_without_attribution_ratio_constraint() -> None:
    estimator = LinearProbabilityEstimator(weights=(1.0, 3.0), threshold=0.8)
    reference = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
    result = generate_counterfactual(
        estimator=estimator,
        reference_frame=reference,
        target_distribution_frame=target_distribution(),
        feature_names=["x1", "x2"],
        feature_types=["numerical", "numerical"],
        feature_ranges=[[0.0, 1.0], [0.0, 1.0]],
        class_labels=["negative", "positive"],
        shap_values=[4.0, 1.0],
        top_k=2,
        selected_feature_indices=[0, 1],
    )

    assert result is not None
    assert result["prediction"]["value"] == 1
    x1, x2 = result["feature_values"]
    assert x1 > 0.0 and x2 > 0.0
    assert not np.isclose(x1 / x2, 4.0, rtol=0.05)
    assert result["optimization"]["constraints"][
        "change_amounts_proportional_to_attribution"
    ] is False


def test_instance_is_rejected_when_one_selected_feature_cannot_support_target() -> None:
    estimator = LinearProbabilityEstimator(weights=(1.0, 0.0), threshold=0.8)
    reference = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
    result = generate_counterfactual(
        estimator=estimator,
        reference_frame=reference,
        target_distribution_frame=target_distribution(),
        feature_names=["x1", "x2"],
        feature_types=["numerical", "numerical"],
        feature_ranges=[[0.0, 1.0], [0.0, 1.0]],
        class_labels=["negative", "positive"],
        shap_values=[1.0, 0.5],
        top_k=2,
        selected_feature_indices=[0, 1],
    )

    assert result is None
