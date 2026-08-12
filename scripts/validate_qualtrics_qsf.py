"""Strict offline integrity checks for the generated Qualtrics QSF."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QSF = REPO_ROOT / "qualtrics" / "UPLOAD_THIS_Qualtrics_Starter.qsf"
STATIC_DATA = REPO_ROOT / "static" / "experiment-data.json"
FRAME_JS = REPO_ROOT / "qualtrics" / "qualtrics-frame.js"
IFRAME_JS = REPO_ROOT / "iframe.js"
EXPERIMENTAL_JS = REPO_ROOT / "experimental.js"
INSTANCE_BROWSER_JS = REPO_ROOT / "qualtrics" / "instance-browser.js"


def strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-standard JSON constant {value!r} in {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def assert_unique(values: list[str], label: str) -> None:
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    assert not duplicates, f"Duplicate {label}: {duplicates}"


def walk_flow(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_flow(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_flow(child)


def check_javascript(question_id: str, source: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"-{question_id}.js",
        encoding="utf-8",
        delete=False,
    ) as temporary:
        temporary.write(source)
        temporary_path = Path(temporary.name)
    try:
        result = subprocess.run(
            ["node", "--check", str(temporary_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"JavaScript syntax failure in {question_id}:\n"
            f"{result.stdout}{result.stderr}"
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def validate(qsf_path: Path) -> dict[str, Any]:
    document = strict_json(qsf_path)
    assert list(document) == ["SurveyEntry", "SurveyElements"]
    entry = document["SurveyEntry"]
    elements = document["SurveyElements"]
    assert isinstance(entry, dict) and isinstance(elements, list)
    for key in ("SurveyID", "SurveyName", "SurveyLanguage"):
        assert entry.get(key), f"SurveyEntry.{key} is required"
    assert entry["SurveyName"] == "Summative Study - Housing and Drink Driving"

    element_types = [element.get("Element") for element in elements]
    for required in ("BL", "FL", "SO", "PROJ", "SQ"):
        assert required in element_types, f"Missing SurveyElement type {required}"
    assert element_types.count("BL") == 1
    assert element_types.count("FL") == 1

    raw_question_ids = [
        element["PrimaryAttribute"]
        for element in elements
        if element.get("Element") == "SQ"
    ]
    assert_unique(raw_question_ids, "question IDs")
    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in elements
        if element.get("Element") == "SQ"
    }
    assert all(question_id == payload.get("QuestionID") for question_id, payload in questions.items())

    blocks = next(element["Payload"] for element in elements if element.get("Element") == "BL")
    assert isinstance(blocks, dict) and list(blocks) == [str(index) for index in range(len(blocks))]
    block_ids = [block["ID"] for block in blocks.values()]
    assert_unique(block_ids, "block IDs")
    assert sum(block.get("Type") == "Default" for block in blocks.values()) == 1
    assert sum(block.get("Type") == "Trash" for block in blocks.values()) == 1

    question_references: list[str] = []
    for block_payload in blocks.values():
        for item in block_payload.get("BlockElements", []):
            if item.get("Type") != "Question":
                continue
            question_id = item.get("QuestionID")
            assert question_id in questions, f"Block references missing question {question_id}"
            question_references.append(question_id)
    assert_unique(question_references, "question references across blocks")
    assert set(question_references) == set(questions), "Every question must be assigned to a block"

    looped = [
        block_payload for block_payload in blocks.values()
        if block_payload.get("Options", {}).get("Looping") == "Static"
    ]
    assert len(looped) == 3, "Expected one training loop and two testing loops"
    loop_row_counts = []
    for loop_block in looped:
        loop_options = loop_block["Options"]["LoopingOptions"]
        assert loop_options.get("Randomization") == "None"
        rows = loop_options.get("Static")
        row_count = len(rows)
        loop_row_counts.append(row_count)
        assert list(rows) == [str(index) for index in range(1, row_count + 1)]
        assert [rows[str(index)]["1"] for index in range(1, row_count + 1)] == [
            str(index) for index in range(row_count)
        ]
    assert sorted(loop_row_counts) == [5, 5, 10]

    flow = next(element["Payload"] for element in elements if element.get("Element") == "FL")
    assert flow.get("Type") == "Root" and isinstance(flow.get("Flow"), list)
    flow_nodes = list(walk_flow(flow))
    flow_ids = [node["FlowID"] for node in flow_nodes if "FlowID" in node]
    assert_unique(flow_ids, "flow IDs")
    flow_block_ids = [
        node["ID"] for node in flow_nodes
        if node.get("Type") in {"Block", "Standard"} and "ID" in node
    ]
    assert all(block_id in block_ids for block_id in flow_block_ids)
    trash_id = next(block["ID"] for block in blocks.values() if block.get("Type") == "Trash")
    assert trash_id not in flow_block_ids
    expected_active = set(block_ids) - {trash_id}
    assert set(flow_block_ids) == expected_active
    assert len(flow_block_ids) == len(expected_active)

    randomizers = [node for node in flow_nodes if node.get("Type") == "BlockRandomizer"]
    assert len(randomizers) == 1
    test_randomizer = randomizers[0]
    assert test_randomizer.get("SubSet") == 2
    assert len(test_randomizer.get("Flow", [])) == 2
    randomized_block_ids = {
        child.get("ID") for child in test_randomizer["Flow"]
    }
    assert randomized_block_ids == {
        "BL_5aT4RYQul04UaHA",
        "BL_3W1PIk3tbtEX0BE",
    }

    initial_node = flow["Flow"][0]
    assert initial_node.get("Type") == "EmbeddedData"
    initial_fields = {field.get("Field"): field.get("Value") for field in initial_node["EmbeddedData"]}
    assert set(initial_fields) == {
        "ui_base_url", "xaiType", "training_log_json", "testing_log_json"
    }
    assert initial_fields.get("xaiType") in {"attribution", "counterfactual", "none"}
    assert initial_fields.get("training_log_json") == "[]"
    assert initial_fields.get("testing_log_json") == "[]"
    assert initial_fields.get("ui_base_url") == (
        "https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html"
    )
    domain_nodes = [
        node for node in flow["Flow"]
        if node.get("Type") == "EmbeddedData" and
        any(field.get("Field") == "appId" for field in node.get("EmbeddedData", []))
    ]
    assert len(domain_nodes) == 1
    app_field = next(
        field for field in domain_nodes[0]["EmbeddedData"]
        if field.get("Field") == "appId"
    )
    assert app_field.get("Value") in {"housing", "safelimit"}

    scripts_checked = 0
    for question_id, payload in questions.items():
        script = payload.get("QuestionJS") or ""
        if script:
            check_javascript(question_id, script)
            scripts_checked += 1
    frame_source = FRAME_JS.read_text(encoding="utf-8")
    assert questions["QID12"]["QuestionJS"] == frame_source
    assert questions["QID376"]["QuestionJS"] == frame_source
    assert questions["QID379"]["QuestionJS"] == frame_source
    assert 'data-phase="training"' in questions["QID12"]["QuestionText"]
    assert "DisplayLogic" not in questions["QID280"]
    assert 'id="cf-scenario-root"' in questions["QID280"]["QuestionText"]
    assert 'data-domain="housing"' in questions["QID280"]["QuestionText"]
    assert 'data-domain="safelimit"' in questions["QID280"]["QuestionText"]
    assert "What you will do" in questions["QID280"]["QuestionText"]
    assert 'data-phase="test"' in questions["QID376"]["QuestionText"]
    assert 'data-test-label="0"' in questions["QID376"]["QuestionText"]
    assert 'data-test-label="1"' in questions["QID379"]["QuestionText"]
    assert 'data-domain="housing"' in questions["QID376"]["QuestionText"]
    assert 'data-domain="safelimit"' in questions["QID376"]["QuestionText"]
    assert 'data-domain="housing"' in questions["QID379"]["QuestionText"]
    assert 'data-domain="safelimit"' in questions["QID379"]["QuestionText"]
    iframe_source = IFRAME_JS.read_text(encoding="utf-8")
    assert 'const explanationView = "persona";' in iframe_source
    assert "getExplanationView(" not in iframe_source
    assert "explanationView:" not in EXPERIMENTAL_JS.read_text(encoding="utf-8")
    assert "explanationView:" not in INSTANCE_BROWSER_JS.read_text(encoding="utf-8")
    assert 'simulationQuestion.className = "counterfactual-simulation-question"' not in iframe_source
    assert "if (counterfactualSimulationEnabled)" in iframe_source
    simulation_branch = iframe_source.split(
        "if (counterfactualSimulationEnabled) {", 1
    )[1].split("showAttributeValues(noneExplanationTbody);", 1)[0]
    assert "createCounterfactualSimulation();" in simulation_branch
    assert "return;" in simulation_branch
    for removed_parameter in (
        "AIModel", "expAlgorithm", "explanationView", "faceFigures",
        "attributeOrderSeed", "simulationMode", "tutorialCallouts",
    ):
        assert removed_parameter not in frame_source
    assert 'iframe.src = makeIframeUrl("none", true, true);' in frame_source

    data = strict_json(STATIC_DATA)
    domain_report: dict[str, Any] = {}
    for domain in ("housing", "safelimit"):
        bundle = data["datasets"][domain]
        training = bundle["training_pool"]
        testing = bundle["test_pool"]
        assert len(training) == 10 and len(testing) == 10
        training_labels = Counter(int(case["prediction"]["value"]) for case in training)
        testing_labels = Counter(int(case["prediction"]["value"]) for case in testing)
        training_pairs = Counter(case["feature_pair_key"] for case in training)
        assert training_labels == {0: 5, 1: 5}
        assert testing_labels == {0: 5, 1: 5}
        assert [int(case["prediction"]["value"]) for case in testing] == [
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1
        ]
        assert len(training_pairs) == 10 and set(training_pairs.values()) == {1}
        for case in training + testing:
            counterfactual = case["counterfactual"]
            changed_names = []
            for name, feature_type, original, updated in zip(
                case["raw_feature_names"],
                case["feature_types"],
                case["raw_feature_values"],
                counterfactual["raw_feature_values"],
            ):
                changed = (
                    str(original) != str(updated)
                    if feature_type == "categorical"
                    else abs(float(updated) - float(original)) > 1e-9
                )
                if changed:
                    changed_names.append(name)
            assert len(changed_names) == 2
            assert set(changed_names) == set(counterfactual["raw_selected_feature_names"])
            assert counterfactual["prediction"]["value"] != case["prediction"]["value"]
            assert counterfactual["source"] == "shap_proportional_direction_optimization"
            assert counterfactual.get("optimization", {}).get("objective") == (
                "minimum_total_normalized_change"
            )
        training_ids = [case["instance_id"] for case in training]
        test_ids = [case["instance_id"] for case in testing]
        assert json.dumps(training_ids) in frame_source
        assert json.dumps(test_ids[:5]) in frame_source
        assert json.dumps(test_ids[5:]) in frame_source
        domain_report[domain] = {
            "training_cases": len(training),
            "test_cases": len(testing),
            "training_label_counts": dict(training_labels),
            "test_label_counts": dict(testing_labels),
            "unique_training_feature_pairs": len(training_pairs),
        }

    return {
        "status": "pass",
        "qsf": str(qsf_path),
        "bytes": qsf_path.stat().st_size,
        "sha256": hashlib.sha256(qsf_path.read_bytes()).hexdigest(),
        "survey_elements": len(elements),
        "questions": len(questions),
        "blocks": len(blocks),
        "active_flow_blocks": len(flow_block_ids),
        "looped_blocks": len(looped),
        "javascript_payloads_checked": scripts_checked,
        "domains": domain_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qsf", type=Path, default=DEFAULT_QSF)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(args.qsf)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
