from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generate_counterfactual(
    estimator: Any,
    reference_frame: pd.DataFrame,
    target_distribution_frame: pd.DataFrame,
    feature_names: list[str],
    feature_types: list[str],
    feature_ranges: list[list[Any]],
    class_labels: list[str],
    shap_values: list[float],
    top_k: int,
    selected_feature_indices: list[int] | None = None,
    generation_mode: str = "minimal",
) -> dict[str, Any] | None:
    current_prediction = int(estimator.predict(reference_frame)[0])
    target_prediction = 1 - current_prediction
    normalized_generation_mode = _normalize_generation_mode(generation_mode)
    selected_indices = _resolve_selected_indices(
        shap_values=shap_values,
        top_k=top_k,
        selected_feature_indices=selected_feature_indices,
    )
    if not selected_indices:
        return None

    reference_series = reference_frame.iloc[0].copy()
    base_target_probability = _predict_target_probability(
        estimator,
        reference_frame,
        target_prediction,
    )
    shap_magnitudes = np.asarray([abs(shap_values[index]) for index in selected_indices])
    max_shap_magnitude = float(np.max(shap_magnitudes)) if len(shap_magnitudes) else 0.0
    if max_shap_magnitude <= 0:
        return None

    best_frame = reference_frame.copy()
    best_target_probability = base_target_probability
    best_prediction = current_prediction
    found_opposing_prediction = False

    for scale in _scale_schedule(normalized_generation_mode):
        candidate_series = reference_series.copy()

        for selected_index in selected_indices:
            feature_name = feature_names[selected_index]
            feature_type = feature_types[selected_index]
            feature_range = feature_ranges[selected_index]
            normalized_shap_weight = abs(shap_values[selected_index]) / max_shap_magnitude

            if feature_type == "categorical":
                if normalized_generation_mode == "prototypical":
                    candidate_series[feature_name] = _best_prototypical_categorical_value(
                        estimator=estimator,
                        reference_series=candidate_series,
                        feature_name=feature_name,
                        categories=feature_range,
                        target_prediction=target_prediction,
                        target_distribution_frame=target_distribution_frame,
                    )
                else:
                    candidate_series[feature_name] = _best_categorical_value(
                        estimator=estimator,
                        reference_series=candidate_series,
                        feature_name=feature_name,
                        categories=feature_range,
                        target_prediction=target_prediction,
                    )
                continue

            direction = _direction_toward_target(
                estimator=estimator,
                reference_frame=reference_frame,
                feature_name=feature_name,
                feature_range=feature_range,
                target_prediction=target_prediction,
            )
            if direction == 0:
                continue

            min_value, max_value = [float(value) for value in feature_range]
            range_span = max_value - min_value
            if range_span <= 0:
                continue

            original_value = float(reference_series[feature_name])
            normalized_value = (original_value - min_value) / range_span
            if normalized_generation_mode == "prototypical":
                prototype_value = _prototypical_numeric_value(
                    reference_series=reference_series,
                    feature_name=feature_name,
                    feature_range=feature_range,
                    direction=direction,
                    target_distribution_frame=target_distribution_frame,
                )
                prototype_delta = (prototype_value - original_value) / range_span
                normalized_value += scale * normalized_shap_weight * prototype_delta
            else:
                normalized_value += direction * scale * normalized_shap_weight
            normalized_value = float(np.clip(normalized_value, 0.0, 1.0))
            updated_value = min_value + normalized_value * range_span
            if _is_integer_like(reference_series[feature_name]):
                updated_value = round(updated_value)
                if updated_value == reference_series[feature_name]:
                    stepped_value = float(reference_series[feature_name]) + direction
                    updated_value = round(float(np.clip(stepped_value, min_value, max_value)))
            candidate_series[feature_name] = updated_value

        candidate_frame = pd.DataFrame([candidate_series], columns=feature_names)
        candidate_prediction = int(estimator.predict(candidate_frame)[0])
        candidate_target_probability = _predict_target_probability(
            estimator,
            candidate_frame,
            target_prediction,
        )

        if candidate_target_probability > best_target_probability:
            best_frame = candidate_frame
            best_prediction = candidate_prediction
            best_target_probability = candidate_target_probability

        if candidate_prediction == target_prediction:
            best_frame = candidate_frame
            best_prediction = candidate_prediction
            best_target_probability = candidate_target_probability
            found_opposing_prediction = True
            break

    return {
        "feature_values": [
            _json_safe_value(best_frame.iloc[0][feature_name])
            for feature_name in feature_names
        ],
        "prediction": {
            "value": best_prediction,
            "label": class_labels[best_prediction]
            if best_prediction < len(class_labels)
            else str(best_prediction),
        },
        "source": "shap_guided_proportional_change"
        if found_opposing_prediction
        else "shap_guided_best_effort",
        "generation_mode": normalized_generation_mode,
        "target_prediction": {
            "value": target_prediction,
            "label": class_labels[target_prediction]
            if target_prediction < len(class_labels)
            else str(target_prediction),
        },
        "selected_feature_names": [
            feature_names[index]
            for index in selected_indices
        ],
        "target_probability": best_target_probability,
    }


def _top_k_indices(values: list[float], top_k: int) -> list[int]:
    if top_k <= 0:
        return []

    ranked_indices = sorted(
        range(len(values)),
        key=lambda index: abs(values[index]),
        reverse=True,
    )
    return [
        index
        for index in ranked_indices
        if abs(values[index]) > 0
    ][:top_k]


def _resolve_selected_indices(
    shap_values: list[float],
    top_k: int,
    selected_feature_indices: list[int] | None,
) -> list[int]:
    if selected_feature_indices is None:
        return _top_k_indices(shap_values, top_k)

    return [
        index
        for index in selected_feature_indices
        if 0 <= index < len(shap_values)
    ][:top_k]


def _normalize_generation_mode(mode: str) -> str:
    normalized_mode = str(mode or "minimal").strip().lower()
    if normalized_mode in {"prototype", "prototypical", "distribution"}:
        return "prototypical"
    return "minimal"


def _scale_schedule(generation_mode: str) -> np.ndarray:
    if generation_mode == "prototypical":
        return np.linspace(0.35, 1.35, 12)

    return np.linspace(0.15, 1.0, 12)


def _direction_toward_target(
    estimator: Any,
    reference_frame: pd.DataFrame,
    feature_name: str,
    feature_range: list[Any],
    target_prediction: int,
) -> int:
    min_value, max_value = [float(value) for value in feature_range]
    range_span = max_value - min_value
    if range_span <= 0:
        return 0

    original_value = float(reference_frame.iloc[0][feature_name])
    step = max(range_span * 0.01, 1e-6)
    lower_value = max(original_value - step, min_value)
    upper_value = min(original_value + step, max_value)
    if lower_value == upper_value:
        return 0

    lower_frame = reference_frame.copy()
    upper_frame = reference_frame.copy()
    lower_frame.loc[lower_frame.index[0], feature_name] = lower_value
    upper_frame.loc[upper_frame.index[0], feature_name] = upper_value

    lower_probability = _predict_target_probability(
        estimator,
        lower_frame,
        target_prediction,
    )
    upper_probability = _predict_target_probability(
        estimator,
        upper_frame,
        target_prediction,
    )
    local_gradient = (upper_probability - lower_probability) / (upper_value - lower_value)
    gradient_direction = 1 if local_gradient > 0 else -1 if local_gradient < 0 else 0

    min_frame = reference_frame.copy()
    max_frame = reference_frame.copy()
    min_frame.loc[min_frame.index[0], feature_name] = min_value
    max_frame.loc[max_frame.index[0], feature_name] = max_value
    min_probability = _predict_target_probability(
        estimator,
        min_frame,
        target_prediction,
    )
    max_probability = _predict_target_probability(
        estimator,
        max_frame,
        target_prediction,
    )
    endpoint_direction = 1 if max_probability > min_probability else -1 if min_probability > max_probability else 0

    if gradient_direction == 0:
        return endpoint_direction

    if endpoint_direction != 0 and endpoint_direction != gradient_direction:
        endpoint_probability = max(max_probability, min_probability)
        local_probability = max(upper_probability, lower_probability)
        if endpoint_probability > local_probability:
            return endpoint_direction

    return gradient_direction


def _prototypical_numeric_value(
    reference_series: pd.Series,
    feature_name: str,
    feature_range: list[Any],
    direction: int,
    target_distribution_frame: pd.DataFrame,
) -> float:
    min_value, max_value = [float(value) for value in feature_range]
    original_value = float(reference_series[feature_name])
    if feature_name not in target_distribution_frame.columns:
        return float(np.clip(original_value + direction * (max_value - min_value), min_value, max_value))

    target_values = pd.to_numeric(
        target_distribution_frame[feature_name],
        errors="coerce",
    ).dropna()
    target_values = target_values[
        (target_values >= min_value) &
        (target_values <= max_value)
    ]
    if target_values.empty:
        return float(np.clip(original_value + direction * (max_value - min_value), min_value, max_value))

    quantiles = [0.5, 0.75, 0.9] if direction > 0 else [0.5, 0.25, 0.1]
    for quantile in quantiles:
        candidate_value = float(target_values.quantile(quantile))
        if (direction > 0 and candidate_value > original_value) or (
            direction < 0 and candidate_value < original_value
        ):
            return float(np.clip(candidate_value, min_value, max_value))

    fallback_quantile = 0.95 if direction > 0 else 0.05
    fallback_value = float(target_values.quantile(fallback_quantile))
    if fallback_value == original_value:
        fallback_value = original_value + direction * (max_value - min_value)
    return float(np.clip(fallback_value, min_value, max_value))


def _best_categorical_value(
    estimator: Any,
    reference_series: pd.Series,
    feature_name: str,
    categories: list[Any],
    target_prediction: int,
) -> Any:
    original_value = reference_series[feature_name]
    best_value = reference_series[feature_name]
    best_probability = -np.inf
    best_non_original_value = None
    best_non_original_probability = -np.inf

    for category in categories:
        candidate_series = reference_series.copy()
        candidate_series[feature_name] = category
        candidate_frame = pd.DataFrame([candidate_series])
        target_probability = _predict_target_probability(
            estimator,
            candidate_frame,
            target_prediction,
        )
        if target_probability > best_probability:
            best_probability = target_probability
            best_value = category
        if category != original_value and target_probability > best_non_original_probability:
            best_non_original_probability = target_probability
            best_non_original_value = category

    if best_value == original_value and best_non_original_value is not None:
        return best_non_original_value
    return best_value


def _best_prototypical_categorical_value(
    estimator: Any,
    reference_series: pd.Series,
    feature_name: str,
    categories: list[Any],
    target_prediction: int,
    target_distribution_frame: pd.DataFrame,
) -> Any:
    original_value = reference_series[feature_name]
    target_frequencies = _category_frequencies(
        feature_name=feature_name,
        categories=categories,
        target_distribution_frame=target_distribution_frame,
    )
    best_value = original_value
    best_score = -np.inf
    best_non_original_value = None
    best_non_original_score = -np.inf

    for category in categories:
        candidate_series = reference_series.copy()
        candidate_series[feature_name] = category
        candidate_frame = pd.DataFrame([candidate_series])
        target_probability = _predict_target_probability(
            estimator,
            candidate_frame,
            target_prediction,
        )
        prototypicality = target_frequencies.get(str(category), 0.0)
        score = target_probability + 0.25 * prototypicality
        if score > best_score:
            best_score = score
            best_value = category
        if category != original_value and score > best_non_original_score:
            best_non_original_score = score
            best_non_original_value = category

    if best_value == original_value and best_non_original_value is not None:
        return best_non_original_value
    return best_value


def _category_frequencies(
    feature_name: str,
    categories: list[Any],
    target_distribution_frame: pd.DataFrame,
) -> dict[str, float]:
    if feature_name not in target_distribution_frame.columns:
        return {}

    observed_counts = (
        target_distribution_frame[feature_name]
        .astype(str)
        .value_counts(normalize=True)
        .to_dict()
    )
    return {
        str(category): float(observed_counts.get(str(category), 0.0))
        for category in categories
    }


def _predict_target_probability(
    estimator: Any,
    frame: pd.DataFrame,
    target_prediction: int,
) -> float:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(frame)[0]
        if target_prediction < len(probabilities):
            return float(probabilities[target_prediction])

    prediction = int(estimator.predict(frame)[0])
    return 1.0 if prediction == target_prediction else 0.0


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _is_integer_like(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) or (
        isinstance(value, float) and value.is_integer()
    )
