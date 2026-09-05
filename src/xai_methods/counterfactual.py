from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd


INDEPENDENT_SEARCH_GRID_SIZE = 65
INDEPENDENT_REFINEMENT_GRID_SIZE = 33
INDEPENDENT_REFINEMENT_STEPS = 3
CHANGE_TOLERANCE = 1e-9
TARGET_IMPROVEMENT_TOLERANCE = 1e-10


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
    normalized_generation_mode = _normalize_generation_mode(generation_mode)
    if normalized_generation_mode == "minimal":
        return _generate_independent_counterfactual(
            estimator=estimator,
            reference_frame=reference_frame,
            target_distribution_frame=target_distribution_frame,
            feature_names=feature_names,
            feature_types=feature_types,
            feature_ranges=feature_ranges,
            class_labels=class_labels,
            shap_values=shap_values,
            top_k=top_k,
            selected_feature_indices=selected_feature_indices,
        )

    return _generate_scaled_counterfactual(
        estimator=estimator,
        reference_frame=reference_frame,
        target_distribution_frame=target_distribution_frame,
        feature_names=feature_names,
        feature_types=feature_types,
        feature_ranges=feature_ranges,
        class_labels=class_labels,
        shap_values=shap_values,
        top_k=top_k,
        selected_feature_indices=selected_feature_indices,
        generation_mode=normalized_generation_mode,
    )


def _generate_scaled_counterfactual(
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


def _generate_independent_counterfactual(
    estimator: Any,
    reference_frame: pd.DataFrame,
    target_distribution_frame: pd.DataFrame,
    feature_names: list[str],
    feature_types: list[str],
    feature_ranges: list[list[Any]],
    class_labels: list[str],
    shap_values: list[float],
    top_k: int,
    selected_feature_indices: list[int] | None,
) -> dict[str, Any] | None:
    current_prediction = int(estimator.predict(reference_frame)[0])
    target_prediction = 1 - current_prediction
    selected_indices = _resolve_selected_indices(
        shap_values=shap_values,
        top_k=top_k,
        selected_feature_indices=selected_feature_indices,
    )
    # The experiment explanation is defined by exactly the two regularized
    # Kernel SHAP features.  We do not substitute another feature when either
    # selected feature cannot contribute toward the target prediction.
    if len(selected_indices) != 2:
        return None

    base_target_probability = _predict_target_probability(
        estimator, reference_frame, target_prediction
    )
    numeric_resolutions = {
        index: _numeric_resolution(
            reference_frame,
            target_distribution_frame,
            feature_names[index],
        )
        for index in selected_indices
        if feature_types[index] != "categorical"
    }

    candidate_axes: dict[int, list[Any]] = {}
    for index in selected_indices:
        raw_candidates = _initial_independent_values(
            reference_frame=reference_frame,
            feature_index=index,
            feature_names=feature_names,
            feature_types=feature_types,
            feature_ranges=feature_ranges,
            numeric_resolution=numeric_resolutions.get(index, 0.0),
        )
        supporting_candidates = _target_supporting_values(
            estimator=estimator,
            reference_frame=reference_frame,
            feature_index=index,
            candidate_values=raw_candidates,
            feature_names=feature_names,
            target_prediction=target_prediction,
            base_target_probability=base_target_probability,
        )
        if not supporting_candidates:
            return None
        candidate_axes[index] = supporting_candidates

    chosen = _search_independent_product(
        estimator=estimator,
        reference_frame=reference_frame,
        selected_indices=selected_indices,
        candidate_axes=candidate_axes,
        feature_names=feature_names,
        feature_types=feature_types,
        feature_ranges=feature_ranges,
        target_prediction=target_prediction,
    )
    if chosen is None:
        return None

    for _ in range(INDEPENDENT_REFINEMENT_STEPS):
        refined_axes: dict[int, list[Any]] = {}
        for index in selected_indices:
            if feature_types[index] == "categorical":
                refined_axes[index] = candidate_axes[index]
                continue
            local_values = _refined_independent_values(
                reference_frame=reference_frame,
                feature_index=index,
                chosen_value=chosen["values"][index],
                previous_values=candidate_axes[index],
                feature_names=feature_names,
                feature_ranges=feature_ranges,
                numeric_resolution=numeric_resolutions.get(index, 0.0),
            )
            refined_axes[index] = _target_supporting_values(
                estimator=estimator,
                reference_frame=reference_frame,
                feature_index=index,
                candidate_values=local_values,
                feature_names=feature_names,
                target_prediction=target_prediction,
                base_target_probability=base_target_probability,
            )
            if not refined_axes[index]:
                refined_axes[index] = candidate_axes[index]
        refined = _search_independent_product(
            estimator=estimator,
            reference_frame=reference_frame,
            selected_indices=selected_indices,
            candidate_axes=refined_axes,
            feature_names=feature_names,
            feature_types=feature_types,
            feature_ranges=feature_ranges,
            target_prediction=target_prediction,
        )
        if refined is None or refined["objective"] >= chosen["objective"] - CHANGE_TOLERANCE:
            break
        chosen = refined
        candidate_axes = refined_axes

    prediction_value = int(estimator.predict(chosen["frame"])[0])
    if prediction_value != target_prediction or not _all_selected_features_changed(
        reference_frame, chosen["frame"], selected_indices, feature_names
    ):
        return None

    return {
        "feature_values": [
            _json_safe_value(chosen["frame"].iloc[0][feature_name])
            for feature_name in feature_names
        ],
        "prediction": {
            "value": prediction_value,
            "label": class_labels[prediction_value]
            if prediction_value < len(class_labels)
            else str(prediction_value),
        },
        "source": "regularized_shap_top2_independent_change_optimization",
        "generation_mode": "minimal",
        "target_prediction": {
            "value": target_prediction,
            "label": class_labels[target_prediction]
            if target_prediction < len(class_labels)
            else str(target_prediction),
        },
        "selected_feature_names": [feature_names[index] for index in selected_indices],
        "target_probability": chosen["target_probability"],
        "optimization": {
            "objective": "minimum_total_normalized_change",
            "objective_value": chosen["objective"],
            "directions": {
                feature_names[index]: _change_direction(
                    reference_frame.iloc[0][feature_names[index]],
                    chosen["values"][index],
                )
                for index in selected_indices
            },
            "selection_attribution_values": {
                feature_names[index]: float(shap_values[index])
                for index in selected_indices
            },
            "normalized_changes": {
                feature_names[index]: chosen["normalized_changes"][index]
                for index in selected_indices
            },
            "constraints": {
                "exact_selected_feature_count": 2,
                "all_selected_features_changed": True,
                "each_change_individually_improves_target_probability": True,
                "change_amounts_proportional_to_attribution": False,
            },
            "search_grid_size": INDEPENDENT_SEARCH_GRID_SIZE,
            "refinement_steps": INDEPENDENT_REFINEMENT_STEPS,
        },
    }


def _initial_independent_values(
    reference_frame: pd.DataFrame,
    feature_index: int,
    feature_names: list[str],
    feature_types: list[str],
    feature_ranges: list[list[Any]],
    numeric_resolution: float,
) -> list[Any]:
    feature_name = feature_names[feature_index]
    original = reference_frame.iloc[0][feature_name]
    if feature_types[feature_index] == "categorical":
        return [
            value
            for value in feature_ranges[feature_index]
            if str(value) != str(original)
        ]

    original_float = float(original)
    lower, upper = _effective_numeric_bounds(
        original_float, feature_ranges[feature_index]
    )
    values = np.linspace(lower, upper, INDEPENDENT_SEARCH_GRID_SIZE).tolist()
    if numeric_resolution > CHANGE_TOLERANCE:
        values.extend([
            original_float - numeric_resolution,
            original_float + numeric_resolution,
        ])
    return _normalize_numeric_candidates(
        values=values,
        original=original_float,
        lower=lower,
        upper=upper,
        numeric_resolution=numeric_resolution,
        integer_dtype=pd.api.types.is_integer_dtype(reference_frame[feature_name]),
    )


def _refined_independent_values(
    reference_frame: pd.DataFrame,
    feature_index: int,
    chosen_value: Any,
    previous_values: list[Any],
    feature_names: list[str],
    feature_ranges: list[list[Any]],
    numeric_resolution: float,
) -> list[Any]:
    feature_name = feature_names[feature_index]
    original = float(reference_frame.iloc[0][feature_name])
    chosen = float(chosen_value)
    lower, upper = _effective_numeric_bounds(original, feature_ranges[feature_index])
    neighbouring_distances = sorted(
        abs(float(value) - chosen)
        for value in [*previous_values, original]
        if abs(float(value) - chosen) > CHANGE_TOLERANCE
    )
    nominal_step = (upper - lower) / max(INDEPENDENT_SEARCH_GRID_SIZE - 1, 1)
    radius = neighbouring_distances[0] if neighbouring_distances else nominal_step
    radius = max(radius, numeric_resolution, CHANGE_TOLERANCE)
    values = np.linspace(
        max(lower, chosen - radius),
        min(upper, chosen + radius),
        INDEPENDENT_REFINEMENT_GRID_SIZE,
    ).tolist()
    values.append(chosen)
    return _normalize_numeric_candidates(
        values=values,
        original=original,
        lower=lower,
        upper=upper,
        numeric_resolution=numeric_resolution,
        integer_dtype=pd.api.types.is_integer_dtype(reference_frame[feature_name]),
    )


def _normalize_numeric_candidates(
    values: list[float],
    original: float,
    lower: float,
    upper: float,
    numeric_resolution: float,
    integer_dtype: bool,
) -> list[Any]:
    normalized: list[Any] = []
    seen: set[float] = set()
    for value in values:
        candidate = float(np.clip(float(value), lower, upper))
        if numeric_resolution > CHANGE_TOLERANCE:
            steps = round((candidate - original) / numeric_resolution)
            candidate = float(np.clip(
                original + steps * numeric_resolution, lower, upper
            ))
        if integer_dtype:
            candidate = int(round(candidate))
        else:
            candidate = round(candidate, 12)
        if abs(float(candidate) - original) <= CHANGE_TOLERANCE:
            continue
        key = round(float(candidate), 12)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)
    return sorted(normalized, key=float)


def _target_supporting_values(
    estimator: Any,
    reference_frame: pd.DataFrame,
    feature_index: int,
    candidate_values: list[Any],
    feature_names: list[str],
    target_prediction: int,
    base_target_probability: float,
) -> list[Any]:
    if not candidate_values:
        return []
    feature_name = feature_names[feature_index]
    rows = []
    for value in candidate_values:
        row = reference_frame.iloc[0].copy()
        row[feature_name] = value
        rows.append(row)
    batch = pd.DataFrame(rows, columns=feature_names).astype(
        reference_frame.dtypes.to_dict()
    )
    probabilities = _predict_target_probabilities(
        estimator=estimator,
        frame=batch,
        target_prediction=target_prediction,
    )
    return [
        value
        for value, probability in zip(candidate_values, probabilities)
        if float(probability) > base_target_probability + TARGET_IMPROVEMENT_TOLERANCE
    ]


def _search_independent_product(
    estimator: Any,
    reference_frame: pd.DataFrame,
    selected_indices: list[int],
    candidate_axes: dict[int, list[Any]],
    feature_names: list[str],
    feature_types: list[str],
    feature_ranges: list[list[Any]],
    target_prediction: int,
) -> dict[str, Any] | None:
    combinations = list(itertools.product(
        *(candidate_axes[index] for index in selected_indices)
    ))
    if not combinations:
        return None

    rows = []
    for combination in combinations:
        row = reference_frame.iloc[0].copy()
        for index, value in zip(selected_indices, combination):
            row[feature_names[index]] = value
        rows.append(row)
    batch = pd.DataFrame(rows, columns=feature_names).astype(
        reference_frame.dtypes.to_dict()
    )
    predictions = np.asarray(estimator.predict(batch), dtype=int)
    probabilities = _predict_target_probabilities(
        estimator=estimator,
        frame=batch,
        target_prediction=target_prediction,
    )
    valid_indices = np.flatnonzero(predictions == target_prediction)
    if len(valid_indices) == 0:
        return None

    records: list[dict[str, Any]] = []
    for batch_index in valid_indices.tolist():
        values = {
            index: combinations[batch_index][position]
            for position, index in enumerate(selected_indices)
        }
        normalized_changes = {
            index: _normalized_change(
                reference_frame.iloc[0][feature_names[index]],
                values[index],
                feature_types[index],
                feature_ranges[index],
            )
            for index in selected_indices
        }
        records.append({
            "batch_index": batch_index,
            "values": values,
            "normalized_changes": normalized_changes,
            "objective": float(sum(normalized_changes.values())),
            "target_probability": float(probabilities[batch_index]),
        })
    chosen = min(
        records,
        key=lambda record: (
            record["objective"],
            -record["target_probability"],
            tuple(str(record["values"][index]) for index in selected_indices),
        ),
    )
    chosen["frame"] = batch.iloc[[chosen["batch_index"]]].copy()
    chosen["frame"].index = reference_frame.index
    return chosen


def _change_direction(original: Any, updated: Any) -> str:
    if isinstance(original, str) or isinstance(updated, str):
        return f"set:{updated}"
    return "increase" if float(updated) > float(original) else "decrease"


def _effective_numeric_bounds(original_value: float, feature_range: list[Any]) -> tuple[float, float]:
    min_value, max_value = [float(value) for value in feature_range]
    return min(min_value, original_value), max(max_value, original_value)


def _numeric_resolution(
    reference_frame: pd.DataFrame,
    target_distribution_frame: pd.DataFrame,
    feature_name: str,
) -> float:
    if pd.api.types.is_integer_dtype(reference_frame[feature_name]):
        return 1.0
    values = pd.concat([
        pd.to_numeric(reference_frame[feature_name], errors="coerce"),
        pd.to_numeric(target_distribution_frame[feature_name], errors="coerce"),
    ]).dropna().unique()
    if len(values) < 2:
        return 0.0
    differences = np.diff(np.sort(values.astype(float)))
    positive_differences = differences[differences > CHANGE_TOLERANCE]
    return float(np.min(positive_differences)) if len(positive_differences) else 0.0


def _all_selected_features_changed(
    reference_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    selected_indices: list[int],
    feature_names: list[str],
) -> bool:
    for index in selected_indices:
        feature_name = feature_names[index]
        original = reference_frame.iloc[0][feature_name]
        candidate = candidate_frame.iloc[0][feature_name]
        if isinstance(original, str) or isinstance(candidate, str):
            if str(original) == str(candidate):
                return False
        elif abs(float(candidate) - float(original)) <= CHANGE_TOLERANCE:
            return False
    return True


def _predict_target_probabilities(
    estimator: Any,
    frame: pd.DataFrame,
    target_prediction: int,
) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(frame), dtype=float)
        if probabilities.ndim == 2 and target_prediction < probabilities.shape[1]:
            return probabilities[:, target_prediction]
    predictions = np.asarray(estimator.predict(frame), dtype=int)
    return (predictions == target_prediction).astype(float)


def _normalized_change(
    original: Any,
    candidate: Any,
    feature_type: str,
    feature_range: list[Any],
) -> float:
    if feature_type == "categorical":
        categories = [str(value) for value in feature_range]
        if len(categories) <= 1:
            return 0.0
        try:
            original_index = categories.index(str(original))
            candidate_index = categories.index(str(candidate))
        except ValueError:
            return 1.0
        return abs(candidate_index - original_index) / (len(categories) - 1)
    span = max(float(feature_range[1]) - float(feature_range[0]), CHANGE_TOLERANCE)
    return abs(float(candidate) - float(original)) / span


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

    reference_value = reference_frame.iloc[0][feature_name]
    original_value = float(reference_value)
    integer_feature = _is_integer_feature(reference_frame, feature_name, reference_value)
    step = (
        max(round(range_span * 0.01), 1)
        if integer_feature
        else max(range_span * 0.01, 1e-6)
    )
    lower_value = max(original_value - step, min_value)
    upper_value = min(original_value + step, max_value)
    if integer_feature:
        lower_value = int(round(lower_value))
        upper_value = int(round(upper_value))
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
    min_frame.loc[min_frame.index[0], feature_name] = (
        int(round(min_value)) if integer_feature else min_value
    )
    max_frame.loc[max_frame.index[0], feature_name] = (
        int(round(max_value)) if integer_feature else max_value
    )
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


def _is_integer_feature(
    reference_frame: pd.DataFrame,
    feature_name: str,
    reference_value: Any,
) -> bool:
    return pd.api.types.is_integer_dtype(reference_frame[feature_name]) or _is_integer_like(
        reference_value
    )
