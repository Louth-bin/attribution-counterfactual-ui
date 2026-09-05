"""Build Recourse v1.2 from the v0.11 QSF and selected diabetes cases.

The selected model cases originate in train/dev/test source splits.  Qualtrics,
however, addresses one static training pool and one static testing pool.  This
builder therefore gives the selected payloads stable synthetic IDs while
retaining their original split and instance ID as audit metadata.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_static_experiment import (  # noqa: E402
    actual_changed_raw_feature_names,
    build_metadata,
    feature_pair_key,
    is_successful_counterfactual,
)
from src.pipeline import ExplanationPipeline  # noqa: E402


SOURCE_QSF = ROOT / "qualtrics" / "Recourse_v011 (updated clustering).qsf"
OUTPUT_QSF = ROOT / "qualtrics" / "Recourse_v1.2.qsf"
SELECTION_CSV = ROOT / "analysis" / "diabetes_explanation_pattern_natural_v3.csv"
STATIC_JSON = ROOT / "static" / "experiment-data.json"
STATIC_JS = ROOT / "static" / "experiment-data.js"
ANALYSIS_BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.2.json"
CASE_MANIFEST = ROOT / "qualtrics" / "case-manifest-v1.2.json"

SYNTHETIC_TRAINING_START = 120_100
SYNTHETIC_TEST_0_START = 120_200
SYNTHETIC_TEST_1_START = 120_300
SYNTHETIC_MINIMUM = 120_000
SYNTHETIC_MAXIMUM = 120_399


def read_selection() -> list[dict[str, str]]:
    with SELECTION_CSV.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 30:
        raise ValueError(f"Expected 30 selected cases, found {len(rows)}")
    phase_counts = Counter(row["experimental_phase"] for row in rows)
    if phase_counts != {"training": 12, "testing": 18}:
        raise ValueError(f"Unexpected phase counts: {phase_counts}")
    for phase, expected_per_prediction in (("training", 6), ("testing", 9)):
        counts = Counter(
            int(row["prediction"])
            for row in rows
            if row["experimental_phase"] == phase
        )
        if counts != {0: expected_per_prediction, 1: expected_per_prediction}:
            raise ValueError(f"Unexpected {phase} prediction counts: {counts}")
    return rows


def selected_order(rows: list[dict[str, str]], phase: str) -> list[dict[str, str]]:
    return sorted(
        (row for row in rows if row["experimental_phase"] == phase),
        key=lambda row: (
            int(row["cluster_rank"]),
            int(row["prediction"]),
            row["source_split"],
            int(row["instance_id"]),
        ),
    )


def synthetic_id(row: dict[str, str], position: int) -> int:
    if row["experimental_phase"] == "training":
        return SYNTHETIC_TRAINING_START + position
    prediction = int(row["prediction"])
    start = SYNTHETIC_TEST_0_START if prediction == 0 else SYNTHETIC_TEST_1_START
    return start + position


def profiles_match(payload: dict[str, Any], row: dict[str, str]) -> bool:
    expected = json.loads(row["original_profile"])
    actual = dict(zip(payload["raw_feature_names"], payload["raw_feature_values"]))
    return all(
        np.isclose(float(actual[name]), float(value), rtol=0.0, atol=1e-9)
        for name, value in expected.items()
    )


def build_payload(
    pipeline: ExplanationPipeline,
    row: dict[str, str],
    assigned_id: int,
) -> dict[str, Any]:
    source_split = row["source_split"]
    source_instance_id = int(row["instance_id"])
    payload = pipeline.get_instance_payload(
        dataset_name="diabetes",
        model_name="mlp",
        xai_method_name="shap",
        instance_id=source_instance_id,
        xai_type="attribution",
        explanation_feature_count=2,
        counterfactual_mode="minimal",
        controllable_only=False,
        split=source_split,
    )
    if int(payload["prediction"]["value"]) != int(row["prediction"]):
        raise ValueError(f"Prediction mismatch for {source_split}:{source_instance_id}")
    if not profiles_match(payload, row):
        raise ValueError(f"Profile mismatch for {source_split}:{source_instance_id}")
    if not is_successful_counterfactual(payload):
        raise ValueError(f"Invalid counterfactual for {source_split}:{source_instance_id}")

    selected_names = row["selected_features"].split(" | ")
    if set(selected_names) != set(actual_changed_raw_feature_names(payload)):
        raise ValueError(f"Feature-pair mismatch for {source_split}:{source_instance_id}")

    payload["feature_pair_key"] = feature_pair_key(payload)
    payload["feature_pair_names"] = [
        payload["feature_names"][payload["raw_feature_names"].index(name)]
        for name in actual_changed_raw_feature_names(payload)
    ]
    payload["source_split"] = source_split
    payload["source_instance_id"] = source_instance_id
    payload["experimental_phase"] = row["experimental_phase"]
    payload["selection_cluster"] = int(row["cluster"])
    payload["selection_cluster_rank"] = int(row["cluster_rank"])
    payload["selection_role"] = row.get("selection_role", "natural_cluster_sample")
    payload["split"] = "train" if row["experimental_phase"] == "training" else "test"
    payload["instance_id"] = assigned_id
    return payload


def build_selected_payloads(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pipeline = ExplanationPipeline()
    training_rows = selected_order(rows, "training")
    testing_rows = selected_order(rows, "testing")
    training = [
        build_payload(pipeline, row, synthetic_id(row, position))
        for position, row in enumerate(training_rows)
    ]

    prediction_positions = Counter()
    testing = []
    for row in testing_rows:
        prediction = int(row["prediction"])
        position = prediction_positions[prediction]
        prediction_positions[prediction] += 1
        testing.append(build_payload(pipeline, row, synthetic_id(row, position)))
    testing.sort(key=lambda payload: (
        int(payload["prediction"]["value"]),
        int(payload["selection_cluster_rank"]),
        int(payload["instance_id"]),
    ))
    return training, testing


def write_static_and_analysis_bundle(
    training: list[dict[str, Any]],
    testing: list[dict[str, Any]],
) -> None:
    static_bundle = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
    diabetes = static_bundle["datasets"]["diabetes"]
    diabetes["training_pool"] = [
        payload for payload in diabetes["training_pool"]
        if not SYNTHETIC_MINIMUM <= int(payload["instance_id"]) <= SYNTHETIC_MAXIMUM
    ] + training
    diabetes["test_pool"] = [
        payload for payload in diabetes["test_pool"]
        if not SYNTHETIC_MINIMUM <= int(payload["instance_id"]) <= SYNTHETIC_MAXIMUM
    ] + testing
    diabetes["metadata"]["static_training_pool_count"] = len(diabetes["training_pool"])
    diabetes["metadata"]["static_test_pool_count"] = len(diabetes["test_pool"])
    diabetes["metadata"]["qualtrics_v1_2_training_ids"] = [
        payload["instance_id"] for payload in training
    ]
    diabetes["metadata"]["qualtrics_v1_2_testing_ids_by_prediction"] = {
        str(prediction): [
            payload["instance_id"] for payload in testing
            if int(payload["prediction"]["value"]) == prediction
        ]
        for prediction in (0, 1)
    }
    static_bundle["version"] = "static-experiment-v12-diabetes-selected"
    static_bundle["generated_at"] = date.today().isoformat()
    STATIC_JSON.write_text(
        json.dumps(static_bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    STATIC_JS.write_text(
        "window.EXPERIMENT_DATA = "
        + json.dumps(static_bundle, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )

    pipeline = ExplanationPipeline()
    analysis_dataset = {
        "metadata": build_metadata(pipeline, "diabetes", training, testing),
        "browser_model": diabetes["browser_model"],
        "training_pool": training,
        "test_pool": testing,
    }
    analysis_bundle = {
        "version": "diabetes-experiment-v1.2",
        "generated_at": date.today().isoformat(),
        "default_model": "mlp",
        "datasets": {"diabetes": analysis_dataset},
    }
    ANALYSIS_BUNDLE.write_text(
        json.dumps(analysis_bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def loop_rows(count: int) -> dict[str, dict[str, str]]:
    return {str(index + 1): {"1": str(index)} for index in range(count)}


def replace_diabetes_ids(
    javascript: str,
    training_ids: list[int],
    testing_ids: dict[int, list[int]],
) -> str:
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
        raise ValueError("Could not replace diabetes CASE_IDS in shared JavaScript")
    updated = updated.replace(
        'title.textContent = "Training case " + presentationPosition + " of 10";',
        'title.textContent = "Training case " + presentationPosition + " of " + caseList.length;',
    )
    updated = updated.replace(
        'LABELS[domain][1 - testLabel] + ": case " + presentationPosition + " of 10";',
        'LABELS[domain][1 - testLabel] + ": case " + presentationPosition + " of " + caseList.length;',
    )
    updated = updated.replace(
        "Loop & Merge field 1 is 0-9 for training and each testing block.",
        "Loop & Merge field 1 indexes the fixed case list for the active block.",
    )
    return updated


def build_qsf(training: list[dict[str, Any]], testing: list[dict[str, Any]]) -> None:
    document = json.loads(SOURCE_QSF.read_text(encoding="utf-8"))
    document["SurveyEntry"]["SurveyName"] = "Recourse v1.2"
    document["SurveyEntry"]["SurveyDescription"] = (
        "Diabetes-warning recourse study with 12 training and 18 testing cases; "
        "three randomized explanation conditions and four CRT-2 questions."
    )

    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    blocks_element = next(
        element for element in document["SurveyElements"]
        if element.get("Element") == "BL"
    )
    blocks = blocks_element["Payload"]
    block_list = list(blocks.values()) if isinstance(blocks, dict) else blocks
    blocks_by_id = {block["ID"]: block for block in block_list}

    training_ids = [int(payload["instance_id"]) for payload in training]
    testing_ids = {
        prediction: [
            int(payload["instance_id"]) for payload in testing
            if int(payload["prediction"]["value"]) == prediction
        ]
        for prediction in (0, 1)
    }
    for question_id in ("QID12", "QID376", "QID379"):
        questions[question_id]["QuestionJS"] = replace_diabetes_ids(
            questions[question_id]["QuestionJS"], training_ids, testing_ids
        )

    blocks_by_id["BL_3fJJUKH6EXaeRIq"]["Description"] = "Training: 12 Fixed Cases"
    blocks_by_id["BL_3fJJUKH6EXaeRIq"]["Options"]["LoopingOptions"]["Static"] = loop_rows(12)
    blocks_by_id["BL_5aT4RYQul04UaHA"]["Description"] = (
        "Testing: Label 0 to Label 1 (9 Cases)"
    )
    blocks_by_id["BL_5aT4RYQul04UaHA"]["Options"]["LoopingOptions"]["Static"] = loop_rows(9)
    blocks_by_id["BL_3W1PIk3tbtEX0BE"]["Description"] = (
        "Testing: Label 1 to Label 0 (9 Cases)"
    )
    blocks_by_id["BL_3W1PIk3tbtEX0BE"]["Options"]["LoopingOptions"]["Static"] = loop_rows(9)

    questions["QID286"]["QuestionText"] = (
        questions["QID286"]["QuestionText"]
        .replace("A training session involving 10 trials.", "A training session involving 12 trials.")
        .replace("A testing session involving 20 trials.", "A testing session involving 18 trials.")
    )
    questions["QID286"]["QuestionJS"] = """Qualtrics.SurveyEngine.addOnload(function () {
  var domain = String(Qualtrics.SurveyEngine.getEmbeddedData('appId') || '').toLowerCase();
  var explanation = String(Qualtrics.SurveyEngine.getEmbeddedData('xaiType') || '').toLowerCase();
  if (domain === 'diabetes' && ['none', 'attribution', 'counterfactuals'].includes(explanation)) {
    Qualtrics.SurveyEngine.setEmbeddedData('random_assignment_complete', '1');
  }
});"""
    questions["QID11"]["QuestionJS"] = questions["QID11"]["QuestionJS"].replace(
        "You will see <b>10 health profiles</b>",
        "You will see <b>12 health profiles</b>",
    )
    questions["QID390"]["QuestionJS"] = questions["QID390"]["QuestionJS"].replace(
        "diabetes: 'You will complete two sessions with ten profiles each.",
        "diabetes: 'You will complete two sessions with nine profiles each.",
    )
    questions["QID19"]["QuestionText"] = questions["QID19"]["QuestionText"].replace(
        "Each of the following ten people receives a <b>Diabetes</b> warning.",
        "Each of the following nine people receives a <b>Diabetes</b> warning.",
    )
    questions["QID20"]["QuestionText"] = questions["QID20"]["QuestionText"].replace(
        "Each of the following ten people receives a <b>No Diabetes</b> warning.",
        "Each of the following nine people receives a <b>No Diabetes</b> warning.",
    )

    options = next(
        element["Payload"] for element in document["SurveyElements"]
        if element.get("Element") == "SO"
    )
    options["SurveyName"] = "Recourse v1.2"
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
    CASE_MANIFEST.write_text(
        json.dumps({"version": "1.2", "cases": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = read_selection()
    training, testing = build_selected_payloads(rows)
    write_static_and_analysis_bundle(training, testing)
    build_qsf(training, testing)
    write_manifest(training, testing)
    print(json.dumps({
        "qsf": str(OUTPUT_QSF),
        "analysis_bundle": str(ANALYSIS_BUNDLE),
        "case_manifest": str(CASE_MANIFEST),
        "training": len(training),
        "testing": len(testing),
        "testing_by_prediction": dict(Counter(
            int(payload["prediction"]["value"]) for payload in testing
        )),
    }, indent=2))


if __name__ == "__main__":
    main()
