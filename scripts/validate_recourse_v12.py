"""Validate the v1.2 Qualtrics survey and its selected static cases."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QSF = ROOT / "qualtrics" / "Recourse_v1.2.qsf"
STATIC = ROOT / "static" / "experiment-data.json"
MANIFEST = ROOT / "qualtrics" / "case-manifest-v1.2.json"


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
    changed = []
    for name, kind, before, after in zip(
        payload["raw_feature_names"],
        payload["feature_types"],
        payload["raw_feature_values"],
        counterfactual["raw_feature_values"],
    ):
        differs = str(before) != str(after) if kind == "categorical" else abs(
            float(before) - float(after)
        ) > 1e-9
        if differs:
            changed.append(name)
    return changed


def main() -> None:
    qsf = json.loads(QSF.read_text(encoding="utf-8"))
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    assert qsf["SurveyEntry"]["SurveyName"] == "Recourse v1.2"

    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in qsf["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    scripts = 0
    for question_id, payload in questions.items():
        source = payload.get("QuestionJS") or ""
        if source:
            check_javascript(question_id, source)
            scripts += 1
    assert questions["QID12"]["QuestionJS"] == questions["QID376"]["QuestionJS"]
    assert questions["QID12"]["QuestionJS"] == questions["QID379"]["QuestionJS"]

    blocks_payload = next(
        element["Payload"] for element in qsf["SurveyElements"]
        if element.get("Element") == "BL"
    )
    blocks = list(blocks_payload.values()) if isinstance(blocks_payload, dict) else blocks_payload
    blocks_by_id = {block["ID"]: block for block in blocks}
    expected_loops = {
        "BL_3fJJUKH6EXaeRIq": 12,
        "BL_5aT4RYQul04UaHA": 9,
        "BL_3W1PIk3tbtEX0BE": 9,
    }
    for block_id, count in expected_loops.items():
        looping = blocks_by_id[block_id]["Options"]["LoopingOptions"]
        assert looping["Randomization"] == "All"
        rows = looping["Static"]
        assert [rows[str(index + 1)]["1"] for index in range(count)] == [
            str(index) for index in range(count)
        ]

    flow = next(
        element["Payload"] for element in qsf["SurveyElements"]
        if element.get("Element") == "FL"
    )
    xai_randomizers = [
        node for node in walk(flow)
        if node.get("Type") == "BlockRandomizer"
        and node.get("SubSet") == "1"
        and len(node.get("Flow", [])) == 3
    ]
    assert len(xai_randomizers) == 1
    assert xai_randomizers[0].get("EvenPresentation") is True
    xai_values = {
        field["Value"]
        for child in xai_randomizers[0]["Flow"]
        for field in child.get("EmbeddedData", [])
        if field.get("Field") == "xaiType"
    }
    assert xai_values == {"none", "attribution", "counterfactuals"}

    diabetes = static["datasets"]["diabetes"]
    training_ids = diabetes["metadata"]["qualtrics_v1_2_training_ids"]
    testing_by_prediction = {
        int(key): value for key, value in
        diabetes["metadata"]["qualtrics_v1_2_testing_ids_by_prediction"].items()
    }
    assert len(training_ids) == 12 and len(set(training_ids)) == 12
    assert {key: len(value) for key, value in testing_by_prediction.items()} == {0: 9, 1: 9}
    all_ids = training_ids + testing_by_prediction[0] + testing_by_prediction[1]
    assert len(all_ids) == len(set(all_ids)) == 30

    qsf_ids = [int(value) for value in re.findall(r"120[123]\d{2}", questions["QID12"]["QuestionJS"])]
    assert Counter(qsf_ids) == Counter(all_ids)

    payloads = {
        int(payload["instance_id"]): payload
        for payload in diabetes["training_pool"] + diabetes["test_pool"]
        if int(payload["instance_id"]) in set(all_ids)
    }
    assert set(payloads) == set(all_ids)
    assert len(manifest) == 30
    assert {int(row["qualtrics_instance_id"]) for row in manifest} == set(all_ids)
    for instance_id, payload in payloads.items():
        assert len(changed_names(payload)) == 2
        assert set(changed_names(payload)) == set(
            payload["counterfactual"]["raw_selected_feature_names"]
        )
        assert int(payload["counterfactual"]["prediction"]["value"]) != int(
            payload["prediction"]["value"]
        )

    print(json.dumps({
        "survey_name": qsf["SurveyEntry"]["SurveyName"],
        "question_scripts_checked": scripts,
        "loop_rows": {"training": 12, "test_prediction_0": 9, "test_prediction_1": 9},
        "selected_static_cases": len(payloads),
        "selected_counterfactuals_valid": len(payloads),
        "xai_conditions": sorted(xai_values),
    }, indent=2))


if __name__ == "__main__":
    main()
