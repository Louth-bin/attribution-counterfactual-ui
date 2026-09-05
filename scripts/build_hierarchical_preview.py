"""Build the isolated self-test bundle for the two reciprocal LIME clusters.

This updates only the local review/self-test data. It does not modify the
deployed static survey bundle or any Qualtrics QSF.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "analysis" / "regularized_mlp_lime_endpoint_clusters.json"
STATIC_DATA = ROOT / "static" / "experiment-data.json"
MODEL_PATH = ROOT / "analysis" / "diabetes_mlp_regularized.joblib"
OUTPUT_JSON = Path(tempfile.gettempdir()) / "regularized_mlp_two_cluster_preview.json"
OUTPUT_JS = Path(tempfile.gettempdir()) / "hierarchical-preview-data.js"

RAW_FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
DISPLAY_FEATURES = ["Glucose", "Blood Pressure", "Insulin", "BMI", "Age"]
LABELS = ["Diabetes", "No Diabetes"]


def load_selection() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    rows = selection["selected_instances"]
    counts = Counter(row["role"] for row in rows)
    if counts != {"training": 12, "testing": 20}:
        raise ValueError(f"Expected 12 training and 20 testing cases, found {counts}")
    for cluster in (1, 2):
        cluster_training = [
            row for row in rows if row["role"] == "training" and row["cluster"] == cluster
        ]
        predictions = Counter(row["prediction"] for row in cluster_training)
        if predictions != {"Diabetes": 3, "No Diabetes": 3}:
            raise ValueError(f"Cluster {cluster} training balance is {predictions}")
    return selection, rows


def payload_for(
    row: dict[str, Any],
    assigned_id: int,
    model: Any,
    ranges: list[list[float]],
) -> dict[str, Any]:
    original = np.asarray([float(row["original"][name]) for name in DISPLAY_FEATURES])
    counterfactual = np.asarray([float(row["counterfactual"][name]) for name in DISPLAY_FEATURES])
    original_probs = model.predict_proba(pd.DataFrame([original], columns=RAW_FEATURES))[0]
    original_prediction = int(np.argmax(original_probs))
    expected_prediction = LABELS.index(row["prediction"])
    if original_prediction != expected_prediction:
        raise ValueError(f"Prediction mismatch for {row['source']}")
    counterfactual_probs = model.predict_proba(pd.DataFrame([counterfactual], columns=RAW_FEATURES))[0]
    target_prediction = 1 - original_prediction
    if int(np.argmax(counterfactual_probs)) != target_prediction:
        raise ValueError(f"Counterfactual does not flip for {row['source']}")

    lime_values = [float(value) for value in row["lime_values_no_diabetes_direction"]]
    shown_indices = sorted(range(len(lime_values)), key=lambda index: -abs(lime_values[index]))[:2]
    changed_names = list(row["counterfactual_changes"].keys())
    changed_indices = [DISPLAY_FEATURES.index(name) for name in changed_names]
    if set(changed_indices) != set(shown_indices):
        raise ValueError(f"LIME/counterfactual pair mismatch for {row['source']}")

    phase = row["role"]
    return {
        "dataset": "diabetes",
        "model": "mlp",
        "xai_method": "lime",
        "xai_type": "attribution",
        "split": "train" if phase == "training" else "test",
        "explanation_feature_count": 2,
        "instance_id": assigned_id,
        "available_instance_count": 392,
        "feature_names": DISPLAY_FEATURES,
        "raw_feature_names": RAW_FEATURES,
        "feature_types": ["numerical"] * 5,
        "feature_ranges": ranges,
        "raw_feature_ranges": ranges,
        "feature_values": original.tolist(),
        "raw_feature_values": original.tolist(),
        "prediction": {
            "value": original_prediction,
            "label": LABELS[original_prediction],
            "probabilities": [
                {"label": label, "value": float(original_probs[index])}
                for index, label in enumerate(LABELS)
            ],
        },
        "prediction_labels": LABELS,
        "feature_importance_by_name": {
            RAW_FEATURES[index]: lime_values[index] for index in range(5)
        },
        "counterfactual_settings": {
            "mode": "minimal",
            "controllable_only": False,
            "controllable_feature_names": DISPLAY_FEATURES,
            "raw_controllable_feature_names": RAW_FEATURES,
        },
        "attribution": {
            "method": "lime",
            "feature_selection": {
                "method": "continuous_lime_lasso_path",
                "num_features": 2,
                "num_samples": 500,
            },
            "values": lime_values,
            "ranking_values": [abs(value) for value in lime_values],
            "raw_values": lime_values,
            "max_abs_value": max(abs(value) for value in lime_values),
            "shown_feature_count": 2,
            "shown_feature_indices": shown_indices,
            "direction_labels": {"left": "Diabetes", "right": "No Diabetes"},
            "local_fidelity": float(row["lime_fidelity"]),
        },
        "counterfactual": {
            "feature_values": counterfactual.tolist(),
            "raw_feature_values": counterfactual.tolist(),
            "prediction": {"value": target_prediction, "label": LABELS[target_prediction]},
            "target_prediction": {"value": target_prediction, "label": LABELS[target_prediction]},
            "target_probability": float(counterfactual_probs[target_prediction]),
            "selected_feature_names": changed_names,
            "raw_selected_feature_names": [RAW_FEATURES[index] for index in changed_indices],
            "source": "lime_top2_independent_change_optimization",
            "generation_mode": "minimal",
        },
        "feature_pair_key": "|".join(sorted(RAW_FEATURES[index] for index in changed_indices)),
        "feature_pair_names": changed_names,
        "source_split": row["source_split"],
        "source_instance_id": int(row["source_instance_id"]),
        "source_row_id": int(row["source_row_id"]),
        "experimental_phase": phase,
        "selection_cluster": int(row["cluster"]),
        "selection_role": "shared_threshold_training" if phase == "training" else "nearest_training_profile",
        "nearest_training_source": row.get("nearest_training_source"),
        "profile_distance_to_nearest_training": row.get("profile_distance_to_nearest_training"),
    }


def build() -> dict[str, Any]:
    selection, rows = load_selection()
    static = json.loads(STATIC_DATA.read_text(encoding="utf-8"))
    static_diabetes = static["datasets"]["diabetes"]
    ranges = static_diabetes["training_pool"][0]["raw_feature_ranges"]
    model = joblib.load(MODEL_PATH)

    training_rows = sorted(
        (row for row in rows if row["role"] == "training"),
        key=lambda row: (int(row["cluster"]), LABELS.index(row["prediction"]), row["source"]),
    )
    testing_rows = sorted(
        (row for row in rows if row["role"] == "testing"),
        key=lambda row: (int(row["cluster"]), LABELS.index(row["prediction"]), row["source"]),
    )
    training = [payload_for(row, 160_100 + index, model, ranges) for index, row in enumerate(training_rows)]
    testing = [payload_for(row, 160_200 + index, model, ranges) for index, row in enumerate(testing_rows)]

    metadata = dict(static_diabetes["metadata"])
    metadata.update({
        "xai_methods": ["lime"],
        "supported_explanations": ["attribution", "counterfactual", "none"],
        "static_training_pool_count": len(training),
        "static_test_pool_count": len(testing),
        "training_pair_counts": dict(Counter(case["feature_pair_key"] for case in training)),
        "preview_clusters": selection["selected_clusters"],
    })
    preprocessor = model.named_steps["preprocessor"]
    mlp = model.named_steps["model"]
    scaler = preprocessor.named_transformers_["numeric"]
    browser_model = {
        "format": "sklearn-mlp-binary-v1",
        "feature_names": RAW_FEATURES,
        "class_labels": LABELS,
        "classes": [int(value) for value in mlp.classes_.tolist()],
        "preprocessing": {
            "type": "column-transformer-v1",
            "numeric": {
                "feature_names": list(preprocessor.transformers_[0][2]),
                "mean": [float(value) for value in scaler.mean_.tolist()],
                "scale": [float(value) for value in scaler.scale_.tolist()],
            },
            "categorical": {"feature_names": [], "categories": []},
        },
        "hidden_activation": str(mlp.activation),
        "output_activation": str(mlp.out_activation_),
        "layers": [
            {
                "weights": [[float(value) for value in row] for row in weights.tolist()],
                "biases": [float(value) for value in biases.tolist()],
            }
            for weights, biases in zip(mlp.coefs_, mlp.intercepts_)
        ],
    }
    return {
        "version": "diabetes-regularized-mlp-lime-endpoint-clusters-min10-preview-v2",
        "generated_at": date.today().isoformat(),
        "default_model": "mlp",
        "datasets": {
            "diabetes": {
                "metadata": metadata,
                "browser_model": browser_model,
                "training_pool": training,
                "test_pool": testing,
            }
        },
    }


def main() -> None:
    bundle = build()
    OUTPUT_JSON.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    OUTPUT_JS.write_text(
        "window.HIERARCHICAL_EXPERIMENT_DATA = " + compact + ";\n"
        "if (new URLSearchParams(window.location.search).get('dataPreview') === 'hierarchical') {\n"
        "  window.EXPERIMENT_DATA = window.HIERARCHICAL_EXPERIMENT_DATA;\n"
        "}\n",
        encoding="utf-8",
    )
    diabetes = bundle["datasets"]["diabetes"]
    print(json.dumps({
        "version": bundle["version"],
        "training_count": len(diabetes["training_pool"]),
        "testing_count": len(diabetes["test_pool"]),
        "training_pairs": diabetes["metadata"]["training_pair_counts"],
        "output": str(OUTPUT_JS),
    }, indent=2))


if __name__ == "__main__":
    main()
