"""Expand Recourse_v07 to two randomized ten-case testing blocks in place."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QSF_PATH = REPO_ROOT / "qualtrics" / "Recourse_v07.qsf"
FRAME_JS_PATH = REPO_ROOT / "qualtrics" / "qualtrics-frame.js"
STATIC_DATA_PATH = REPO_ROOT / "static" / "experiment-data.json"
TEST_BLOCK_IDS = {
    "BL_5aT4RYQul04UaHA": "Testing: Label 0 to Label 1 (10 Cases)",
    "BL_3W1PIk3tbtEX0BE": "Testing: Label 1 to Label 0 (10 Cases)",
}
FRAME_QUESTION_IDS = {"QID12", "QID376", "QID379"}


def direction_intro_js(direction: int) -> str:
    return f"""
Qualtrics.SurveyEngine.addOnload(function () {{
    "use strict";

    var questionContainer = this.getQuestionContainer();
    var seenField = "test_direction_{direction}_intro_seen";
    if (String(Qualtrics.SurveyEngine.getEmbeddedData(seenField) || "0") === "1") {{
        questionContainer.style.display = "none";
        return;
    }}
    Qualtrics.SurveyEngine.setEmbeddedData(seenField, "1");

    var domain = String(
        Qualtrics.SurveyEngine.getEmbeddedData("appId") || "housing"
    ).toLowerCase();
    if (domain !== "housing" && domain !== "safelimit") domain = "housing";

    Array.prototype.forEach.call(
        questionContainer.querySelectorAll("[data-domain]"),
        function (section) {{
            section.hidden = section.getAttribute("data-domain") !== domain;
        }}
    );
}});
Qualtrics.SurveyEngine.addOnReady(function () {{}});
Qualtrics.SurveyEngine.addOnUnload(function () {{}});
""".strip()


def main() -> None:
    document = json.loads(QSF_PATH.read_text(encoding="utf-8-sig"))
    static_data = json.loads(STATIC_DATA_PATH.read_text(encoding="utf-8"))
    frame_js = FRAME_JS_PATH.read_text(encoding="utf-8")

    for domain, bundle in static_data["datasets"].items():
        for label in (0, 1):
            ids = [
                int(case["instance_id"])
                for case in bundle["test_pool"]
                if int(case["prediction"]["value"]) == label
            ]
            if len(ids) != 10 or len(set(ids)) != 10:
                raise RuntimeError(
                    f"{domain} label {label} must have 10 unique test cases; got {ids}"
                )
            rendered_ids = ", ".join(str(instance_id) for instance_id in ids)
            if f"{label}: [{rendered_ids}]" not in frame_js:
                raise RuntimeError(
                    f"qualtrics-frame.js is not synchronized with {domain} label {label}"
                )

    block_payload = next(
        element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "BL"
    )
    blocks = block_payload.values() if isinstance(block_payload, dict) else block_payload
    updated_blocks = set()
    for block in blocks:
        block_id = block.get("ID")
        if block_id not in TEST_BLOCK_IDS:
            continue
        block["Description"] = TEST_BLOCK_IDS[block_id]
        loop_options = block["Options"]["LoopingOptions"]
        loop_options["Static"] = {
            str(row + 1): {"1": str(row)} for row in range(10)
        }
        loop_options["Randomization"] = "All"
        updated_blocks.add(block_id)
    if updated_blocks != set(TEST_BLOCK_IDS):
        raise RuntimeError(f"Missing testing blocks: {set(TEST_BLOCK_IDS) - updated_blocks}")

    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    for question_id in FRAME_QUESTION_IDS:
        questions[question_id]["QuestionJS"] = frame_js

    intro = questions["QID390"]["QuestionJS"]
    intro = intro.replace(
        "two sessions with five profiles each",
        "two sessions with ten profiles each",
    )
    intro = intro.replace("two five-profile phases", "two ten-profile phases")
    questions["QID390"]["QuestionJS"] = intro
    for question_id in ("QID19", "QID20"):
        for field in ("QuestionText", "QuestionDescription"):
            if field in questions[question_id]:
                questions[question_id][field] = questions[question_id][field].replace(
                    "following five", "following ten"
                )
    for element in document["SurveyElements"]:
        if element.get("PrimaryAttribute") in {"QID19", "QID20"}:
            element["SecondaryAttribute"] = str(
                element.get("SecondaryAttribute") or ""
            ).replace("following five", "following ten")
    questions["QID19"]["QuestionJS"] = direction_intro_js(0)
    questions["QID20"]["QuestionJS"] = direction_intro_js(1)
    questions["QID286"]["QuestionText"] = questions["QID286"]["QuestionText"].replace(
        "10 testing cases", "20 testing cases"
    )
    document["SurveyEntry"]["SurveyDescription"] = document["SurveyEntry"].get(
        "SurveyDescription", ""
    ).replace("10 fixed test", "20 fixed test")

    initial_embedded_data = next(
        node["EmbeddedData"]
        for node in document["SurveyElements"]
        if node.get("Element") == "FL"
        for node in node["Payload"]["Flow"]
        if node.get("Type") == "EmbeddedData"
        and any(
            field.get("Field") == "testing_log_json"
            for field in node.get("EmbeddedData", [])
        )
    )
    existing_fields = {field.get("Field") for field in initial_embedded_data}
    for field_name in (
        "test_direction_0_intro_seen",
        "test_direction_1_intro_seen",
    ):
        if field_name not in existing_fields:
            initial_embedded_data.append({
                "Description": field_name,
                "Type": "Custom",
                "Field": field_name,
                "VariableType": "String",
                "DataVisibility": [],
                "AnalyzeText": False,
                "Value": "0",
            })

    QSF_PATH.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(QSF_PATH)


if __name__ == "__main__":
    main()
