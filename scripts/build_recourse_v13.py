"""Build Recourse v1.3 with the two-cluster diabetes experiment.

The 12 training cases are presented as two hidden six-case blocks. Rows are
randomized within each block and Qualtrics randomizes the order of the two
blocks. Testing retains the two randomized prediction-direction sessions, now
with six cases per direction.

This builder uses a new 130xxx synthetic-ID range, so the v1.2 survey's 120xxx
cases remain available in the shared static bundle.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_recourse_v12 import build_payload, replace_diabetes_ids  # noqa: E402
from scripts.generate_static_experiment import build_metadata  # noqa: E402
from src.pipeline import ExplanationPipeline  # noqa: E402


SOURCE_QSF = ROOT / "qualtrics" / "Recourse_v1.2.qsf"
OUTPUT_QSF = ROOT / "qualtrics" / "Recourse_v1.3.qsf"
SELECTION_CSV = ROOT / "analysis" / "diabetes_hierarchical_oriented_v5.csv"
STATIC_JSON = ROOT / "static" / "experiment-data.json"
STATIC_JS = ROOT / "static" / "experiment-data.js"
ANALYSIS_BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.3.json"
CASE_MANIFEST = ROOT / "qualtrics" / "case-manifest-v1.3.json"

SYNTHETIC_TRAINING_START = 130_100
SYNTHETIC_TEST_0_START = 130_200
SYNTHETIC_TEST_1_START = 130_300
SYNTHETIC_MINIMUM = 130_000
SYNTHETIC_MAXIMUM = 130_399

TRAINING_BLOCK_A = "BL_3fJJUKH6EXaeRIq"
TRAINING_BLOCK_B = "BL_6Hb2Nq8Tx4Lm7Wp"
TRAINING_QUESTION_A = "QID12"
TRAINING_QUESTION_B = "QID409"
TRAINING_RANDOMIZER_FLOW = "FL_44"


def read_selection() -> list[dict[str, str]]:
    with SELECTION_CSV.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    phase_counts = Counter(row["experimental_phase"] for row in rows)
    if phase_counts != {"training": 12, "testing": 12}:
        raise ValueError(f"Unexpected phase counts: {phase_counts}")
    for phase in ("training", "testing"):
        labels = Counter(
            int(row["prediction"])
            for row in rows
            if row["experimental_phase"] == phase
        )
        if labels != {0: 6, 1: 6}:
            raise ValueError(f"Unexpected {phase} prediction counts: {labels}")
    training_pairs = Counter(
        row["main_pair"] for row in rows if row["experimental_phase"] == "training"
    )
    if training_pairs != {"glucose | insulin": 6, "blood_pressure | age": 6}:
        raise ValueError(f"Unexpected training pairs: {training_pairs}")
    for row in rows:
        # build_payload uses the earlier selector's column names.
        row["cluster"] = row["subcluster"]
        row["cluster_rank"] = row["main_cluster_rank"]
    return rows


def selected_order(rows: list[dict[str, str]], phase: str) -> list[dict[str, str]]:
    return sorted(
        (row for row in rows if row["experimental_phase"] == phase),
        key=lambda row: (
            int(row["main_cluster_rank"]),
            int(row["prediction"]),
            float(row["distance_to_subcluster_centroid"]),
            row["source_split"],
            int(row["instance_id"]),
        ),
    )


def build_selected_payloads(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pipeline = ExplanationPipeline()
    training_rows = selected_order(rows, "training")
    testing_rows = selected_order(rows, "testing")
    training = [
        build_payload(pipeline, row, SYNTHETIC_TRAINING_START + position)
        for position, row in enumerate(training_rows)
    ]

    testing: list[dict[str, Any]] = []
    positions: Counter[int] = Counter()
    for row in testing_rows:
        prediction = int(row["prediction"])
        start = SYNTHETIC_TEST_0_START if prediction == 0 else SYNTHETIC_TEST_1_START
        assigned_id = start + positions[prediction]
        positions[prediction] += 1
        testing.append(build_payload(pipeline, row, assigned_id))
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
    diabetes["metadata"]["qualtrics_v1_3_training_ids"] = [
        payload["instance_id"] for payload in training
    ]
    diabetes["metadata"]["qualtrics_v1_3_testing_ids_by_prediction"] = {
        str(prediction): [
            payload["instance_id"] for payload in testing
            if int(payload["prediction"]["value"]) == prediction
        ]
        for prediction in (0, 1)
    }
    diabetes["metadata"]["qualtrics_v1_3_training_blocks"] = {
        "glucose_insulin": [payload["instance_id"] for payload in training[:6]],
        "blood_pressure_age": [payload["instance_id"] for payload in training[6:]],
    }
    static_bundle["version"] = "static-experiment-v13-diabetes-two-cluster"
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
        "version": "diabetes-experiment-v1.3",
        "generated_at": date.today().isoformat(),
        "default_model": "mlp",
        "datasets": {"diabetes": analysis_dataset},
    }
    ANALYSIS_BUNDLE.write_text(
        json.dumps(analysis_bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def loop_rows(indices: list[int]) -> dict[str, dict[str, str]]:
    return {
        str(position): {"1": str(index)}
        for position, index in enumerate(indices, start=1)
    }


def patch_shared_javascript(
    javascript: str,
    training_ids: list[int],
    testing_ids: dict[int, list[int]],
) -> str:
    updated = replace_diabetes_ids(javascript, training_ids, testing_ids)
    instance_needle = "    var instanceId = caseList && caseList[loopIndex];\n"
    if instance_needle not in updated:
        raise ValueError("Could not locate the shared instance-ID assignment")
    numbering = instance_needle + """

    // The training cases come from two backend blocks whose Loop & Merge
    // counters each restart at one. Continue the participant-facing numbering
    // and log the true presentation position without exposing block identity.
    if (phase === "training") {
        var priorTrainingRecords = [];
        try {
            priorTrainingRecords = JSON.parse(getEmbeddedData("training_log_json") || "[]");
            if (!Array.isArray(priorTrainingRecords)) priorTrainingRecords = [];
        } catch (_trainingLogError) {
            priorTrainingRecords = [];
        }
        var priorCurrentRecord = priorTrainingRecords.find(function (item) {
            return Number(item.instanceId) === Number(instanceId);
        });
        presentationPosition = priorCurrentRecord && Number(priorCurrentRecord.caseNumber)
            ? Number(priorCurrentRecord.caseNumber)
            : priorTrainingRecords.length + 1;
    }
"""
    updated = updated.replace(instance_needle, numbering, 1)
    pool_needle = "            casePoolPosition: loopIndex + 1,\n"
    if pool_needle not in updated:
        raise ValueError("Could not locate the training record's pool position")
    updated = updated.replace(
        pool_needle,
        pool_needle
        + "            trainingBlock: loopIndex < 6 ? \"glucose_insulin\" : \"blood_pressure_age\",\n",
        1,
    )
    return updated


def duplicate_training_question(document: dict[str, Any]) -> None:
    elements = document["SurveyElements"]
    if any(
        element.get("Element") == "SQ" and element.get("PrimaryAttribute") == TRAINING_QUESTION_B
        for element in elements
    ):
        raise ValueError(f"{TRAINING_QUESTION_B} already exists")
    original_index = next(
        index for index, element in enumerate(elements)
        if element.get("Element") == "SQ"
        and element.get("PrimaryAttribute") == TRAINING_QUESTION_A
    )
    duplicate = deepcopy(elements[original_index])
    duplicate["PrimaryAttribute"] = TRAINING_QUESTION_B
    duplicate["SecondaryAttribute"] = "Training case (hidden block 2)"
    duplicate["Payload"]["QuestionID"] = TRAINING_QUESTION_B
    duplicate["Payload"]["DataExportTag"] = "training_block_2"
    duplicate["Payload"]["QuestionDescription"] = "Training case (hidden block 2)"
    elements.insert(original_index + 1, duplicate)


def concise_task_preview() -> str:
    return """<div id="cf-task-preview-root" class="cf-task-preview">
  <h1>Preview of the final task</h1>
  <p>In the final task, make the smallest changes you think will <b>flip the AI prediction</b>.</p>
  <p>You may not yet know what to change. Keep this goal in mind during training.</p>
  <p>Try the controls below. The prediction stays fixed, and you will not receive feedback on whether your changes flip it.</p>
  <iframe id="cf-task-preview-frame" title="Final task preview" style="width:100%;height:420px;border:0;display:block"></iframe>
</div>
<style>
  .cf-task-preview{font-size:18px;line-height:1.4;max-width:1050px;margin:0 auto}
  .cf-task-preview h1{font-size:30px;margin:0 0 18px}
  .cf-task-preview p{margin:0 0 12px}
</style>"""


def build_qsf(training: list[dict[str, Any]], testing: list[dict[str, Any]]) -> None:
    document = json.loads(SOURCE_QSF.read_text(encoding="utf-8"))
    document["SurveyEntry"]["SurveyName"] = "Recourse v1.3"
    document["SurveyEntry"]["SurveyDescription"] = (
        "Diabetes-warning recourse study with two hidden randomized training "
        "blocks, 12 training cases, 12 testing cases, three explanation "
        "conditions, and four CRT-2 questions."
    )
    document["SurveyEntry"]["LastModified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    duplicate_training_question(document)
    question_elements = {
        element["PrimaryAttribute"]: element
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    questions = {
        question_id: element["Payload"]
        for question_id, element in question_elements.items()
    }

    blocks_element = next(
        element for element in document["SurveyElements"]
        if element.get("Element") == "BL"
    )
    blocks = blocks_element["Payload"]
    if not isinstance(blocks, list):
        raise ValueError("Expected the QSF block payload to be a list")
    blocks_by_id = {block["ID"]: block for block in blocks}

    training_ids = [int(payload["instance_id"]) for payload in training]
    testing_ids = {
        prediction: [
            int(payload["instance_id"]) for payload in testing
            if int(payload["prediction"]["value"]) == prediction
        ]
        for prediction in (0, 1)
    }
    for question_id in (TRAINING_QUESTION_A, TRAINING_QUESTION_B, "QID376", "QID379"):
        questions[question_id]["QuestionJS"] = patch_shared_javascript(
            questions[question_id]["QuestionJS"], training_ids, testing_ids
        )

    training_a = blocks_by_id[TRAINING_BLOCK_A]
    training_a["Description"] = "Training block A (hidden): Glucose + Insulin"
    training_a["BlockElements"] = [
        {"Type": "Question", "QuestionID": TRAINING_QUESTION_A}
    ]
    training_a["Options"]["LoopingOptions"]["Static"] = loop_rows(list(range(0, 6)))
    training_a["Options"]["LoopingOptions"]["Randomization"] = "All"

    training_b = deepcopy(training_a)
    training_b["ID"] = TRAINING_BLOCK_B
    training_b["Description"] = "Training block B (hidden): Blood Pressure + Age"
    training_b["BlockElements"] = [
        {"Type": "Question", "QuestionID": TRAINING_QUESTION_B}
    ]
    training_b["Options"]["LoopingOptions"]["Static"] = loop_rows(list(range(6, 12)))
    training_a_index = blocks.index(training_a)
    blocks.insert(training_a_index + 1, training_b)

    for block_id, description in (
        ("BL_5aT4RYQul04UaHA", "Testing: Label 0 to Label 1 (6 Cases)"),
        ("BL_3W1PIk3tbtEX0BE", "Testing: Label 1 to Label 0 (6 Cases)"),
    ):
        block = blocks_by_id[block_id]
        block["Description"] = description
        block["Options"]["LoopingOptions"]["Static"] = loop_rows(list(range(6)))
        block["Options"]["LoopingOptions"]["Randomization"] = "All"

    flow_payload = next(
        element["Payload"] for element in document["SurveyElements"]
        if element.get("Element") == "FL"
    )
    root_flow = flow_payload["Flow"]
    training_flow_index = next(
        index for index, element in enumerate(root_flow)
        if element.get("Type") == "Standard" and element.get("ID") == TRAINING_BLOCK_A
    )
    root_flow[training_flow_index] = {
        "Type": "BlockRandomizer",
        "FlowID": TRAINING_RANDOMIZER_FLOW,
        "SubSet": 2,
        "Flow": [
            {
                "Type": "Standard",
                "ID": TRAINING_BLOCK_A,
                "FlowID": "FL_45",
                "Autofill": [],
            },
            {
                "Type": "Standard",
                "ID": TRAINING_BLOCK_B,
                "FlowID": "FL_46",
                "Autofill": [],
            },
        ],
    }

    questions["QID286"]["QuestionText"] = questions["QID286"]["QuestionText"].replace(
        "A testing session involving 18 trials.",
        "A testing session involving 12 trials.",
    )
    questions["QID390"]["QuestionJS"] = questions["QID390"]["QuestionJS"].replace(
        "diabetes: 'You will complete two sessions with nine profiles each.",
        "diabetes: 'You will complete two sessions with six profiles each.",
    )
    questions["QID19"]["QuestionText"] = questions["QID19"]["QuestionText"].replace(
        "Each of the following nine people receives a <b>Diabetes</b> warning.",
        "Each of the following six people receives a <b>Diabetes</b> warning.",
    )
    questions["QID20"]["QuestionText"] = questions["QID20"]["QuestionText"].replace(
        "Each of the following nine people receives a <b>No Diabetes</b> warning.",
        "Each of the following six people receives a <b>No Diabetes</b> warning.",
    )

    questions["QID26"]["QuestionText"] = concise_task_preview()
    questions["QID26"]["QuestionDescription"] = (
        "Preview of the final task: keep the goal in mind during training"
    )
    questions["QID26"]["QuestionJS"] = questions["QID26"]["QuestionJS"].replace(
        "var previewIds = { housing: 14693, safelimit: 610, diabetes: 225 };",
        "var previewIds = { housing: 14693, safelimit: 610, diabetes: 225 };\n"
        "  if (explanation === 'counterfactuals') explanation = 'counterfactual';",
    )

    OUTPUT_QSF.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def write_manifest(training: list[dict[str, Any]], testing: list[dict[str, Any]]) -> None:
    rows = []
    for payload in training + testing:
        pair = payload["feature_pair_key"]
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
            "feature_pair_key": pair,
            "training_block": (
                "glucose_insulin" if pair == "glucose|insulin"
                else "blood_pressure_age" if pair == "blood_pressure|age"
                else ""
            ) if payload["experimental_phase"] == "training" else "",
        })
    CASE_MANIFEST.write_text(
        json.dumps({"version": "1.3", "cases": rows}, indent=2) + "\n",
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
        "training_blocks": {
            "glucose_insulin": [payload["instance_id"] for payload in training[:6]],
            "blood_pressure_age": [payload["instance_id"] for payload in training[6:]],
        },
        "testing": len(testing),
        "testing_by_prediction": dict(Counter(
            int(payload["prediction"]["value"]) for payload in testing
        )),
    }, indent=2))


if __name__ == "__main__":
    main()
