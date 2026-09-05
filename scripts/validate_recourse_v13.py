"""Validate Recourse v1.3, its hidden training blocks, and static cases."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QSF = ROOT / "qualtrics" / "Recourse_v1.3.qsf"
STATIC = ROOT / "static" / "experiment-data.json"
ANALYSIS = ROOT / "analysis" / "diabetes_experiment_bundle_v1.3.json"
MANIFEST = ROOT / "qualtrics" / "case-manifest-v1.3.json"

TRAIN_A = "BL_3fJJUKH6EXaeRIq"
TRAIN_B = "BL_6Hb2Nq8Tx4Lm7Wp"
TEST_0 = "BL_5aT4RYQul04UaHA"
TEST_1 = "BL_3W1PIk3tbtEX0BE"


def walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def check_javascript(question_id: str, source: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"-{question_id}.js", encoding="utf-8", delete=False
    ) as handle:
        handle.write(source)
        path = Path(handle.name)
    try:
        result = subprocess.run(
            ["node", "--check", str(path)], capture_output=True, text=True
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
    finally:
        path.unlink(missing_ok=True)


def changed_names(payload: dict[str, Any]) -> list[str]:
    counterfactual = payload["counterfactual"]
    return [
        name
        for name, kind, before, after in zip(
            payload["raw_feature_names"],
            payload["feature_types"],
            payload["raw_feature_values"],
            counterfactual["raw_feature_values"],
        )
        if (
            str(before) != str(after)
            if kind == "categorical"
            else abs(float(before) - float(after)) > 1e-9
        )
    ]


def loop_indices(block: dict[str, Any]) -> list[str]:
    options = block["Options"]["LoopingOptions"]
    assert options["Randomization"] == "All"
    rows = options["Static"]
    assert list(rows) == [str(index) for index in range(1, len(rows) + 1)]
    return [rows[str(index)]["1"] for index in range(1, len(rows) + 1)]


def main() -> None:
    qsf = json.loads(QSF.read_text(encoding="utf-8"))
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    assert qsf["SurveyEntry"]["SurveyName"] == "Recourse v1.3"
    survey_options = next(
        element["Payload"]
        for element in qsf["SurveyElements"]
        if element.get("Element") == "SO"
    )
    assert "SurveyName" not in survey_options

    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in qsf["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    assert len(questions) == len(set(questions))
    assert questions["QID409"]["DataExportTag"] == "training_block_2"
    scripts = 0
    for question_id, payload in questions.items():
        source = payload.get("QuestionJS") or ""
        if source:
            check_javascript(question_id, source)
            scripts += 1
    shared = questions["QID12"]["QuestionJS"]
    assert shared == questions["QID409"]["QuestionJS"]
    assert shared == questions["QID376"]["QuestionJS"]
    assert shared == questions["QID379"]["QuestionJS"]
    assert 'trainingBlock: loopIndex < 6 ? "glucose_insulin" : "blood_pressure_age"' in shared
    assert "priorTrainingRecords.length + 1" in shared

    block_payload = next(
        element["Payload"] for element in qsf["SurveyElements"]
        if element.get("Element") == "BL"
    )
    assert isinstance(block_payload, list)
    blocks = {block["ID"]: block for block in block_payload}
    assert loop_indices(blocks[TRAIN_A]) == [str(index) for index in range(6)]
    assert loop_indices(blocks[TRAIN_B]) == [str(index) for index in range(6, 12)]
    assert loop_indices(blocks[TEST_0]) == [str(index) for index in range(6)]
    assert loop_indices(blocks[TEST_1]) == [str(index) for index in range(6)]
    assert blocks[TRAIN_A]["BlockElements"] == [
        {"Type": "Question", "QuestionID": "QID12"}
    ]
    assert blocks[TRAIN_B]["BlockElements"] == [
        {"Type": "Question", "QuestionID": "QID409"}
    ]

    question_references = [
        element["QuestionID"]
        for block in block_payload
        for element in block.get("BlockElements", [])
        if element.get("Type") == "Question"
    ]
    assert len(question_references) == len(set(question_references))
    assert set(question_references) == set(questions)

    flow = next(
        element["Payload"] for element in qsf["SurveyElements"]
        if element.get("Element") == "FL"
    )
    nodes = list(walk(flow))
    flow_ids = [node["FlowID"] for node in nodes if "FlowID" in node]
    assert len(flow_ids) == len(set(flow_ids))
    training_randomizer = next(
        node for node in nodes
        if node.get("FlowID") == "FL_44"
    )
    assert training_randomizer["Type"] == "BlockRandomizer"
    assert training_randomizer["SubSet"] == 2
    assert "EvenPresentation" not in training_randomizer
    assert [child["ID"] for child in training_randomizer["Flow"]] == [TRAIN_A, TRAIN_B]

    test_randomizer = next(
        node for node in nodes
        if node.get("Type") == "BlockRandomizer"
        and {child.get("ID") for child in node.get("Flow", [])} == {TEST_0, TEST_1}
    )
    assert test_randomizer["SubSet"] == 2

    xai_randomizer = next(
        node for node in nodes
        if node.get("Type") == "BlockRandomizer"
        and str(node.get("SubSet")) == "1"
        and len(node.get("Flow", [])) == 3
    )
    assert xai_randomizer.get("EvenPresentation") is True
    xai_values = {
        field["Value"]
        for child in xai_randomizer["Flow"]
        for field in child.get("EmbeddedData", [])
        if field.get("Field") == "xaiType"
    }
    assert xai_values == {"none", "attribution", "counterfactuals"}

    preview = questions["QID26"]["QuestionText"]
    assert "You may not yet know what to change. Keep this goal in mind during training." in preview
    assert preview.count("<b>") == preview.count("</b>") == 1
    visible_preview = re.sub(r"<style\b[^>]*>.*?</style>", "", preview, flags=re.DOTALL)
    visible_preview = re.sub(r"<[^>]+>", " ", visible_preview)
    assert "block" not in visible_preview.casefold()
    assert "counterfactuals') explanation = 'counterfactual'" in questions["QID26"]["QuestionJS"]
    assert "testing session involving 12 trials" in questions["QID286"]["QuestionText"]
    assert "testing session involving 18 trials" not in questions["QID286"]["QuestionText"]
    assert "following six people" in questions["QID19"]["QuestionText"]
    assert "following six people" in questions["QID20"]["QuestionText"]
    assert "two sessions with six profiles each" in questions["QID390"]["QuestionJS"]

    diabetes = static["datasets"]["diabetes"]
    training_ids = diabetes["metadata"]["qualtrics_v1_3_training_ids"]
    testing = {
        int(key): value
        for key, value in diabetes["metadata"]["qualtrics_v1_3_testing_ids_by_prediction"].items()
    }
    assert training_ids == list(range(130_100, 130_112))
    assert testing == {
        0: list(range(130_200, 130_206)),
        1: list(range(130_300, 130_306)),
    }
    all_ids = training_ids + testing[0] + testing[1]
    qsf_ids = [int(value) for value in re.findall(r"130[123]\d{2}", shared)]
    assert Counter(qsf_ids) == Counter(all_ids)

    # The new range coexists with v1.2 rather than breaking the old survey.
    old_ids = diabetes["metadata"]["qualtrics_v1_2_training_ids"]
    assert old_ids and set(old_ids).isdisjoint(all_ids)
    all_static_ids = {
        int(payload["instance_id"])
        for payload in diabetes["training_pool"] + diabetes["test_pool"]
    }
    assert set(old_ids).issubset(all_static_ids)
    assert set(all_ids).issubset(all_static_ids)

    analysis_cases = (
        analysis["datasets"]["diabetes"]["training_pool"]
        + analysis["datasets"]["diabetes"]["test_pool"]
    )
    payloads = {int(payload["instance_id"]): payload for payload in analysis_cases}
    assert set(payloads) == set(all_ids)
    assert len(manifest) == 24
    assert {int(row["qualtrics_instance_id"]) for row in manifest} == set(all_ids)
    assert Counter(
        payload["feature_pair_key"]
        for payload in analysis["datasets"]["diabetes"]["training_pool"]
    ) == {"glucose|insulin": 6, "blood_pressure|age": 6}
    for phase in ("training_pool", "test_pool"):
        cases = analysis["datasets"]["diabetes"][phase]
        assert Counter(int(payload["prediction"]["value"]) for payload in cases) == {0: 6, 1: 6}
    for payload in payloads.values():
        changed = changed_names(payload)
        assert len(changed) == 2
        assert set(changed) == set(payload["counterfactual"]["raw_selected_feature_names"])
        assert int(payload["counterfactual"]["prediction"]["value"]) != int(
            payload["prediction"]["value"]
        )

    report = {
        "status": "pass",
        "survey": qsf["SurveyEntry"]["SurveyName"],
        "qsf_sha256": hashlib.sha256(QSF.read_bytes()).hexdigest(),
        "question_scripts_checked": scripts,
        "training_blocks": {
            "glucose_insulin": training_ids[:6],
            "blood_pressure_age": training_ids[6:],
        },
        "within_block_randomization": "All",
        "between_block_randomization": True,
        "training_cases": 12,
        "testing_cases": 12,
        "selected_counterfactuals_valid": len(payloads),
        "v1_2_cases_preserved": True,
        "final_task_preview_bold_phrases": 1,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
