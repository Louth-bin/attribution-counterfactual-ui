"""Materialize the suggested copy-vs-linear/prototype diabetes instance set.

This takes the quick design outputs in outputs/v20-copy-vs-linear-prototype-design
and makes them usable by the static UI/Qualtrics interface with synthetic UI IDs:

- training: 170100+
- testing, original Diabetes / label 0: 170200+
- testing, original No Diabetes / label 1: 170300+
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calculate_v09_boundary_distance import model_logit
OUTDIR = ROOT / "outputs" / "v20-copy-vs-linear-prototype-design"
CANDIDATES = ROOT / "analysis" / "diabetes_hierarchical_oriented_q90_anchor_search_candidate_shortlist.csv"
TRAINING_SELECTION = OUTDIR / "selected_training_cases.csv"
TESTING_SELECTION = OUTDIR / "selected_testing_cases.csv"
STATIC_JSON = ROOT / "static" / "experiment-data.json"
STATIC_JS = ROOT / "static" / "experiment-data.js"
SOURCE_QSF = ROOT / "qualtrics" / "Recourse_v1.4.qsf"
OUTPUT_QSF = ROOT / "qualtrics" / "Recourse_v1.7.qsf"
MANIFEST = ROOT / "qualtrics" / "case-manifest-v1.7.json"
ANALYSIS = ROOT / "analysis" / "diabetes-experiment-bundle-v1.7.json"
FRAME_JS = ROOT / "qualtrics" / "qualtrics-frame.js"

FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
DISPLAY_NAMES = {
    "glucose": "Glucose",
    "blood_pressure": "Blood Pressure",
    "insulin": "Insulin",
    "bmi": "BMI",
    "age": "Age",
}
PREDICTION_LABELS = ["Diabetes", "No Diabetes"]
TRAINING_IDS_START = 170100
TESTING_IDS_START_BY_PREDICTION = {0: 170200, 1: 170300}


def sigmoid(logit: float) -> float:
    logit = float(np.clip(logit, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-logit))


def prediction_for(raw_values: list[float], model: dict[str, Any], rep_case: dict[str, Any]) -> dict[str, Any]:
    p_no_diabetes = sigmoid(model_logit(model, np.asarray(raw_values, dtype=float), rep_case, normalized=False))
    probs = [1.0 - p_no_diabetes, p_no_diabetes]
    value = int(probs[1] >= 0.5)
    return {
        "value": value,
        "label": PREDICTION_LABELS[value],
        "probabilities": [
            {"label": PREDICTION_LABELS[index], "value": float(probability)}
            for index, probability in enumerate(probs)
        ],
    }


def selected_rows(selection_path: Path, phase: str, candidates: pd.DataFrame) -> list[dict[str, Any]]:
    selection = pd.read_csv(selection_path)
    rows: list[dict[str, Any]] = []
    for _, selected in selection.iterrows():
        match = candidates[
            candidates["source_split"].eq(str(selected["source_split"]))
            & candidates["instance_id"].eq(int(selected["source_instance_id"]))
        ]
        if len(match) != 1:
            raise RuntimeError(
                f"Expected one candidate for {selected['source_split']}:{selected['source_instance_id']}, found {len(match)}"
            )
        row = match.iloc[0].to_dict()
        row["experimental_phase"] = phase
        row["selection_rank"] = int(selected["rank"])
        rows.append(row)
    return rows


def pair_cluster(pair: str) -> int:
    return 1 if pair == "glucose | bmi" else 2


def raw_vector(raw_json: str) -> list[float]:
    profile = json.loads(raw_json)
    return [float(profile[name]) for name in FEATURES]


def shown_feature_indices(selected_features: str) -> list[int]:
    selected = [name.strip() for name in selected_features.split("|")]
    return [FEATURES.index(name) for name in selected]


def feature_pair_key(selected_features: str) -> str:
    return "|".join(sorted(name.strip() for name in selected_features.split("|")))


def payload_from_row(
    row: dict[str, Any],
    assigned_id: int,
    model: dict[str, Any],
    rep_case: dict[str, Any],
) -> dict[str, Any]:
    original = raw_vector(row["original_profile"])
    counterfactual = raw_vector(row["counterfactual_profile"])
    attribution_values = [float(value) for value in json.loads(row["attribution_vector"])]
    indices = shown_feature_indices(row["selected_features"])
    prediction = prediction_for(original, model, rep_case)
    cf_prediction = prediction_for(counterfactual, model, rep_case)
    expected_prediction = int(row["prediction"])
    if int(prediction["value"]) != expected_prediction:
        raise RuntimeError(
            f"Prediction mismatch for {row['source_split']}:{row['instance_id']} "
            f"expected {expected_prediction}, got {prediction['value']}"
        )
    if int(cf_prediction["value"]) == expected_prediction:
        raise RuntimeError(f"Counterfactual does not flip {row['source_split']}:{row['instance_id']}")

    pair_key = feature_pair_key(row["selected_features"])
    pair_names = [DISPLAY_NAMES[FEATURES[index]] for index in indices]
    return {
        "dataset": "diabetes",
        "model": "mlp",
        "xai_method": "lime",
        "xai_type": "attribution",
        "split": "train" if row["experimental_phase"] == "training" else "test",
        "explanation_feature_count": 2,
        "instance_id": assigned_id,
        "available_instance_count": rep_case.get("available_instance_count", 392),
        "feature_names": [DISPLAY_NAMES[name] for name in FEATURES],
        "raw_feature_names": FEATURES,
        "feature_types": list(rep_case["feature_types"]),
        "feature_ranges": list(rep_case["feature_ranges"]),
        "raw_feature_ranges": list(rep_case["raw_feature_ranges"]),
        "feature_values": original,
        "raw_feature_values": original,
        "prediction": prediction,
        "prediction_labels": PREDICTION_LABELS,
        "feature_importance_by_name": {
            name: float(attribution_values[index]) for index, name in enumerate(FEATURES)
        },
        "counterfactual_settings": {
            "mode": "minimal",
            "controllable_only": False,
            "controllable_feature_names": [DISPLAY_NAMES[name] for name in FEATURES],
            "raw_controllable_feature_names": FEATURES,
        },
        "attribution": {
            "method": "lime",
            "feature_selection": {
                "method": "continuous_lime_lasso_path",
                "num_features": 2,
                "num_samples": 500,
            },
            "values": attribution_values,
            "ranking_values": [abs(value) for value in attribution_values],
            "raw_values": attribution_values,
            "max_abs_value": max(max(abs(value) for value in attribution_values), 1e-12),
            "shown_feature_count": 2,
            "shown_feature_indices": indices,
            "direction_labels": {"left": "Diabetes", "right": "No Diabetes"},
            "local_fidelity": None,
        },
        "counterfactual": {
            "feature_values": counterfactual,
            "raw_feature_values": counterfactual,
            "prediction": {
                "value": int(cf_prediction["value"]),
                "label": cf_prediction["label"],
            },
            "target_prediction": {
                "value": int(cf_prediction["value"]),
                "label": cf_prediction["label"],
            },
            "target_probability": float(row["counterfactual_target_probability"]),
            "selected_feature_names": pair_names,
            "raw_selected_feature_names": [FEATURES[index] for index in indices],
            "source": "suggested_copy_vs_linear_prototype_selection",
            "generation_mode": "minimal",
        },
        "feature_pair_key": pair_key,
        "feature_pair_names": pair_names,
        "source_split": row["source_split"],
        "source_instance_id": int(row["instance_id"]),
        "source_row_id": int(row["instance_id"]),
        "experimental_phase": row["experimental_phase"],
        "selection_cluster": pair_cluster(row["main_pair"]),
        "selection_cluster_rank": int(row["selection_rank"]),
        "selection_role": "copy_vs_linear_prototype_design",
        "profile_distance_to_nearest_training": None,
    }


def build_payloads(static_bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = pd.read_csv(CANDIDATES)
    rows = selected_rows(TRAINING_SELECTION, "training", candidates) + selected_rows(
        TESTING_SELECTION, "testing", candidates
    )
    diabetes = static_bundle["datasets"]["diabetes"]
    model = diabetes["browser_model"]
    rep_case = diabetes["training_pool"][0]

    training: list[dict[str, Any]] = []
    testing: list[dict[str, Any]] = []
    testing_positions = {0: 0, 1: 0}
    for row in rows:
        if row["experimental_phase"] == "training":
            assigned_id = TRAINING_IDS_START + len(training)
            training.append(payload_from_row(row, assigned_id, model, rep_case))
            continue
        prediction = int(row["prediction"])
        assigned_id = TESTING_IDS_START_BY_PREDICTION[prediction] + testing_positions[prediction]
        testing_positions[prediction] += 1
        testing.append(payload_from_row(row, assigned_id, model, rep_case))
    testing.sort(key=lambda payload: (int(payload["prediction"]["value"]), int(payload["instance_id"])))
    return training, testing


def write_static(training: list[dict[str, Any]], testing: list[dict[str, Any]], static_bundle: dict[str, Any]) -> None:
    diabetes = static_bundle["datasets"]["diabetes"]
    new_ids = {payload["instance_id"] for payload in training + testing}
    diabetes["training_pool"] = [
        payload for payload in diabetes["training_pool"] if int(payload["instance_id"]) not in new_ids
    ] + training
    diabetes["test_pool"] = [
        payload for payload in diabetes["test_pool"] if int(payload["instance_id"]) not in new_ids
    ] + testing

    training_blocks = {
        "glucose_bmi": [
            payload["instance_id"] for payload in training if payload["feature_pair_key"] == "bmi|glucose"
        ],
        "blood_pressure_insulin": [
            payload["instance_id"] for payload in training if payload["feature_pair_key"] == "blood_pressure|insulin"
        ],
    }
    testing_ids = {
        str(prediction): [
            payload["instance_id"] for payload in testing if int(payload["prediction"]["value"]) == prediction
        ]
        for prediction in (0, 1)
    }
    metadata = diabetes["metadata"]
    metadata["active_interface_design"] = "suggested_copy_vs_linear_prototype_v1"
    metadata["active_training_ids"] = [payload["instance_id"] for payload in training]
    metadata["active_testing_ids_by_prediction"] = testing_ids
    metadata["active_training_blocks"] = training_blocks
    metadata["suggested_copy_v1_training_ids"] = metadata["active_training_ids"]
    metadata["suggested_copy_v1_testing_ids_by_prediction"] = testing_ids
    metadata["suggested_copy_v1_training_blocks"] = training_blocks
    metadata["suggested_copy_v1_selection"] = "outputs/v20-copy-vs-linear-prototype-design selected_training_cases.csv + selected_testing_cases.csv"
    metadata["static_training_pool_count"] = len(diabetes["training_pool"])
    metadata["static_test_pool_count"] = len(diabetes["test_pool"])
    static_bundle["version"] = "static-experiment-v17-suggested-copy-vs-linear-prototype"
    static_bundle["generated_at"] = datetime.now().date().isoformat()

    STATIC_JSON.write_text(json.dumps(static_bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    STATIC_JS.write_text(
        "window.EXPERIMENT_DATA = "
        + json.dumps(static_bundle, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )


def replace_case_ids(javascript: str, training_ids: list[int], testing_ids: dict[int, list[int]]) -> str:
    pattern = re.compile(
        r"(diabetes:\s*\{\s*training:\s*)\[[^\]]*\]"
        r"(,\s*test:\s*\{\s*0:\s*)\[[^\]]*\]"
        r"(,\s*1:\s*)\[[^\]]*\](\s*\}\s*\})",
        re.DOTALL,
    )
    replacement = (
        r"\g<1>" + json.dumps(training_ids)
        + r"\g<2>" + json.dumps(testing_ids[0])
        + r"\g<3>" + json.dumps(testing_ids[1])
        + r"\g<4>"
    )
    updated, count = pattern.subn(replacement, javascript, count=1)
    if count != 1:
        raise RuntimeError("Could not replace diabetes CASE_IDS in frame JavaScript")
    return updated


def loop_rows(values: list[int]) -> dict[str, dict[str, str]]:
    return {str(index + 1): {"1": str(value)} for index, value in enumerate(values)}


def write_qsf(training: list[dict[str, Any]], testing: list[dict[str, Any]], frame_js: str) -> None:
    document = json.loads(SOURCE_QSF.read_text(encoding="utf-8"))
    document["SurveyEntry"]["SurveyName"] = "Recourse v1.7"
    document["SurveyEntry"]["SurveyDescription"] = (
        "Diabetes-warning recourse study using the suggested copy-vs-linear/prototype "
        "instance set, with attention MCQs and attempt logging."
    )
    document["SurveyEntry"]["LastModified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    for question_id in ("QID12", "QID409", "QID376", "QID379"):
        questions[question_id]["QuestionJS"] = frame_js

    blocks = next(
        element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "BL"
    )
    blocks_by_id = {block["ID"]: block for block in blocks}
    training_blocks = {
        "glucose_bmi": [
            payload["instance_id"] for payload in training if payload["feature_pair_key"] == "bmi|glucose"
        ],
        "blood_pressure_insulin": [
            payload["instance_id"] for payload in training if payload["feature_pair_key"] == "blood_pressure|insulin"
        ],
    }
    blocks_by_id["BL_3fJJUKH6EXaeRIq"]["Description"] = "Training block A (hidden): Glucose + BMI"
    blocks_by_id["BL_3fJJUKH6EXaeRIq"]["Options"]["LoopingOptions"]["Static"] = loop_rows(
        list(range(0, len(training_blocks["glucose_bmi"])))
    )
    blocks_by_id["BL_6Hb2Nq8Tx4Lm7Wp"]["Description"] = "Training block B (hidden): Blood Pressure + Insulin"
    blocks_by_id["BL_6Hb2Nq8Tx4Lm7Wp"]["Options"]["LoopingOptions"]["Static"] = loop_rows(
        list(range(len(training_blocks["glucose_bmi"]), len(training)))
    )
    testing_ids = {
        prediction: [
            payload["instance_id"] for payload in testing if int(payload["prediction"]["value"]) == prediction
        ]
        for prediction in (0, 1)
    }
    blocks_by_id["BL_5aT4RYQul04UaHA"]["Description"] = "Testing: Diabetes to No Diabetes (4 Cases)"
    blocks_by_id["BL_5aT4RYQul04UaHA"]["Options"]["LoopingOptions"]["Static"] = loop_rows(
        list(range(len(testing_ids[0])))
    )
    blocks_by_id["BL_3W1PIk3tbtEX0BE"]["Description"] = "Testing: No Diabetes to Diabetes (4 Cases)"
    blocks_by_id["BL_3W1PIk3tbtEX0BE"]["Options"]["LoopingOptions"]["Static"] = loop_rows(
        list(range(len(testing_ids[1])))
    )

    OUTPUT_QSF.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def write_manifest(training: list[dict[str, Any]], testing: list[dict[str, Any]]) -> None:
    rows = []
    for payload in training + testing:
        rows.append({
            "domain": "diabetes",
            "experimental_phase": payload["experimental_phase"],
            "qualtrics_instance_id": payload["instance_id"],
            "source_split": payload["source_split"],
            "source_instance_id": payload["source_instance_id"],
            "prediction": int(payload["prediction"]["value"]),
            "prediction_label": payload["prediction"]["label"],
            "selection_cluster": payload["selection_cluster"],
            "selection_cluster_rank": payload["selection_cluster_rank"],
            "feature_pair_key": payload["feature_pair_key"],
        })
    MANIFEST.write_text(json.dumps({"version": "1.7", "cases": rows}, indent=2) + "\n", encoding="utf-8")


def write_analysis_bundle(training: list[dict[str, Any]], testing: list[dict[str, Any]], static_bundle: dict[str, Any]) -> None:
    diabetes = static_bundle["datasets"]["diabetes"]
    ANALYSIS.write_text(
        json.dumps({
            "version": "diabetes-experiment-v1.7-suggested-copy-vs-linear-prototype",
            "generated_at": datetime.now().date().isoformat(),
            "default_model": "mlp",
            "datasets": {
                "diabetes": {
                    "metadata": {
                        "selection": "suggested_copy_vs_linear_prototype_v1",
                        "training_count": len(training),
                        "testing_count": len(testing),
                    },
                    "browser_model": diabetes["browser_model"],
                    "training_pool": training,
                    "test_pool": testing,
                }
            },
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    static_bundle = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
    training, testing = build_payloads(static_bundle)
    training_ids = [payload["instance_id"] for payload in training]
    testing_ids = {
        prediction: [
            payload["instance_id"] for payload in testing if int(payload["prediction"]["value"]) == prediction
        ]
        for prediction in (0, 1)
    }
    frame_js = replace_case_ids(FRAME_JS.read_text(encoding="utf-8"), training_ids, testing_ids)
    FRAME_JS.write_text(frame_js, encoding="utf-8")
    write_static(training, testing, static_bundle)
    write_qsf(training, testing, frame_js)
    write_manifest(training, testing)
    write_analysis_bundle(training, testing, static_bundle)
    print(json.dumps({
        "training_ids": training_ids,
        "testing_ids_by_prediction": testing_ids,
        "training_count": len(training),
        "testing_count": len(testing),
        "qsf": str(OUTPUT_QSF),
        "manifest": str(MANIFEST),
        "analysis": str(ANALYSIS),
    }, indent=2))


if __name__ == "__main__":
    main()
