from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


MAX_EXACT_KERNEL_SHAP_FEATURES = 10


class ShapXAI:
    name = "shap"

    def explain(
        self,
        estimator: Any,
        background_frame: pd.DataFrame,
        instance_frame: pd.DataFrame,
        positive_class_index: int = 1,
    ) -> dict[str, Any]:
        sampled_background = background_frame.sample(
            n=min(len(background_frame), 50),
            random_state=42,
        )
        encoded_background, encoded_instance, category_maps = _encode_frames(
            sampled_background,
            instance_frame,
        )

        def wrapped_predict(encoded_rows: np.ndarray) -> np.ndarray:
            decoded_frame = _decode_rows(
                encoded_rows=encoded_rows,
                reference_columns=list(instance_frame.columns),
                reference_background=sampled_background,
                category_maps=category_maps,
            )
            return estimator.predict_proba(decoded_frame)

        explainer = shap.KernelExplainer(
            wrapped_predict,
            encoded_background.to_numpy(dtype=float),
            feature_names=list(instance_frame.columns),
        )
        random_state = np.random.get_state()
        np.random.seed(42)
        try:
            shap_values = explainer.shap_values(
                encoded_instance.to_numpy(dtype=float),
                nsamples=_kernel_shap_sample_count(encoded_background.shape[1]),
                l1_reg=0.0,
                silent=True,
            )
        finally:
            np.random.set_state(random_state)
        expected_values = explainer.expected_value

        class_index = min(positive_class_index, 1)
        class_values = _extract_class_values(shap_values)
        if isinstance(shap_values, list):
            class_index = min(class_index, len(shap_values) - 1)
            instance_values = np.asarray(shap_values[class_index][0], dtype=float)
        else:
            values = np.asarray(shap_values, dtype=float)
            if values.ndim == 3:
                class_index = min(class_index, values.shape[2] - 1)
                instance_values = values[0, :, class_index]
            elif values.ndim == 2:
                instance_values = values[0]
            else:
                raise ValueError("Unexpected SHAP output shape.")

        if isinstance(expected_values, list):
            class_index = min(class_index, len(expected_values) - 1)
            base_value = float(expected_values[class_index])
        else:
            flattened_expected = np.ravel(expected_values)
            class_index = min(class_index, len(flattened_expected) - 1)
            base_value = float(flattened_expected[class_index])

        max_abs_value = float(np.max(np.abs(instance_values))) if len(instance_values) else 0.0

        return {
            "method": self.name,
            "values": [float(value) for value in instance_values.tolist()],
            "class_values": [
                [float(value) for value in class_instance_values.tolist()]
                for class_instance_values in class_values
            ],
            "ranking_values": _class_agnostic_ranking_values(class_values),
            "base_value": base_value,
            "max_abs_value": max(max_abs_value, 1e-9),
        }


def _encode_frames(
    background_frame: pd.DataFrame,
    instance_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    encoded_background = background_frame.copy()
    encoded_instance = instance_frame.copy()
    category_maps: dict[str, dict[str, Any]] = {}

    for column in background_frame.columns:
        if pd.api.types.is_numeric_dtype(background_frame[column]):
            encoded_background[column] = background_frame[column].astype(float)
            encoded_instance[column] = instance_frame[column].astype(float)
            continue

        categories = pd.Index(
            pd.concat([background_frame[column], instance_frame[column]], axis=0)
            .astype(str)
            .unique()
            .tolist()
        )
        forward_map = {category: index for index, category in enumerate(categories)}
        inverse_map = {index: category for category, index in forward_map.items()}
        category_maps[column] = {
            "forward": forward_map,
            "inverse": inverse_map,
            "max_index": len(categories) - 1,
        }
        encoded_background[column] = background_frame[column].astype(str).map(forward_map).astype(float)
        encoded_instance[column] = instance_frame[column].astype(str).map(forward_map).astype(float)

    return encoded_background, encoded_instance, category_maps


def _kernel_shap_sample_count(feature_count: int) -> int:
    if feature_count <= MAX_EXACT_KERNEL_SHAP_FEATURES:
        return 2 ** feature_count

    return min(2000, 2 * feature_count + 2048)


def _extract_class_values(shap_values: Any) -> list[np.ndarray]:
    if isinstance(shap_values, list):
        return [
            np.asarray(class_values[0], dtype=float)
            for class_values in shap_values
        ]

    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3:
        return [
            np.asarray(values[0, :, class_index], dtype=float)
            for class_index in range(values.shape[2])
        ]
    if values.ndim == 2:
        return [np.asarray(values[0], dtype=float)]

    raise ValueError("Unexpected SHAP output shape.")


def _class_agnostic_ranking_values(class_values: list[np.ndarray]) -> list[float]:
    if not class_values:
        return []

    stacked_values = np.vstack(class_values)
    return [
        float(value)
        for value in np.max(np.abs(stacked_values), axis=0).tolist()
    ]


def _decode_rows(
    encoded_rows: np.ndarray,
    reference_columns: list[str],
    reference_background: pd.DataFrame,
    category_maps: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    decoded_frame = pd.DataFrame(encoded_rows, columns=reference_columns)

    for column in reference_columns:
        if column not in category_maps:
            decoded_frame[column] = decoded_frame[column].astype(float)
            continue

        inverse_map = category_maps[column]["inverse"]
        max_index = category_maps[column]["max_index"]
        decoded_frame[column] = decoded_frame[column].round().clip(0, max_index).astype(int).map(inverse_map)

    return decoded_frame.astype(reference_background.dtypes.to_dict())
