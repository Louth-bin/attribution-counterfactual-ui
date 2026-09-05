"""Update Recourse v1.7 with patient-change ratings and no CRT-2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUALTRICS = ROOT / "qualtrics"
SOURCE = QUALTRICS / "Recourse_v1.7.qsf"
OUTPUT = QUALTRICS / "Recourse_v1.7.qsf"

CRT_BLOCK_IDS = {"BL_10CRT2Intro", "BL_10CRT2Items"}
CRT_QUESTION_IDS = {"QID404", "QID405", "QID406", "QID407", "QID408"}
POST_TASK_BLOCK_ID = "BL_10PostTaskReflection"
ACTIONABILITY_BLOCK_ID = "BL_18FeatureActionability"
ACTIONABILITY_QUESTION_ID = "QID410"
ACTIONABILITY_FLOW_ID = "FL_180"


def remove_crt_flows(flow: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retained = []
    for item in flow:
        if item.get("ID") in CRT_BLOCK_IDS:
            continue
        if isinstance(item.get("Flow"), list):
            item["Flow"] = remove_crt_flows(item["Flow"])
        retained.append(item)
    return retained


def actionability_question(survey_id: str) -> dict[str, Any]:
    choices = {
        "1": {"Display": "Glucose"},
        "2": {"Display": "Blood Pressure"},
        "3": {"Display": "Insulin"},
        "4": {"Display": "BMI"},
        "5": {"Display": "Age"},
    }
    answers = {
        str(value): {
            "Display": (
                "1 — Strongly disagree"
                if value == 1
                else "10 — Strongly agree"
                if value == 10
                else str(value)
            )
        }
        for value in range(1, 11)
    }
    payload = {
        "QuestionText": (
            "<strong>How much do you agree that a real patient can change each of "
            "the following in practice?</strong><br>Rate each one from "
            "1 (Strongly disagree) to 10 (Strongly agree)."
        ),
        "DefaultChoices": False,
        "DataExportTag": "FeatureActionability",
        "QuestionID": ACTIONABILITY_QUESTION_ID,
        "QuestionType": "Matrix",
        "Selector": "Likert",
        "SubSelector": "SingleAnswer",
        "DataVisibility": {"Private": False, "Hidden": False},
        "Configuration": {"QuestionDescriptionOption": "UseText"},
        "QuestionDescription": "How much a real patient can change each attribute (1–10)",
        "Choices": choices,
        "ChoiceOrder": list(choices),
        "Answers": answers,
        "AnswerOrder": list(answers),
        "Validation": {
            "Settings": {
                "ForceResponse": "ON",
                "ForceResponseType": "ON",
                "Type": "None",
            }
        },
        "GradingData": [],
        "Language": [],
        "NextChoiceId": 6,
        "NextAnswerId": 11,
    }
    return {
        "SurveyID": survey_id,
        "Element": "SQ",
        "PrimaryAttribute": ACTIONABILITY_QUESTION_ID,
        "SecondaryAttribute": "Feature actionability ratings (1–10)",
        "TertiaryAttribute": None,
        "Payload": payload,
    }


def main() -> None:
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    survey_id = document["SurveyEntry"]["SurveyID"]
    document["SurveyEntry"]["SurveyName"] = "Recourse v1.7"
    document["SurveyEntry"]["SurveyDescription"] = (
        "Diabetes recourse study with actionable instance design and post-task "
        "feature-actionability ratings"
    )

    elements = document["SurveyElements"]
    block_element = next(element for element in elements if element.get("Element") == "BL")
    blocks = block_element["Payload"]
    blocks[:] = [block for block in blocks if block.get("ID") not in CRT_BLOCK_IDS]
    blocks[:] = [block for block in blocks if block.get("ID") != ACTIONABILITY_BLOCK_ID]
    post_index = next(
        index for index, block in enumerate(blocks) if block.get("ID") == POST_TASK_BLOCK_ID
    )
    blocks.insert(
        post_index + 1,
        {
            "Type": "Standard",
            "SubType": "",
            "Description": "Post-task feature actionability ratings",
            "ID": ACTIONABILITY_BLOCK_ID,
            "BlockElements": [
                {"Type": "Question", "QuestionID": ACTIONABILITY_QUESTION_ID}
            ],
            "Options": {
                "BlockLocking": "false",
                "RandomizeQuestions": "false",
                "BlockVisibility": "Expanded",
            },
        },
    )

    flow_element = next(element for element in elements if element.get("Element") == "FL")
    root_flow = flow_element["Payload"]["Flow"]
    root_flow[:] = remove_crt_flows(root_flow)
    root_flow[:] = [item for item in root_flow if item.get("FlowID") != ACTIONABILITY_FLOW_ID]
    post_flow_index = next(
        index for index, item in enumerate(root_flow) if item.get("ID") == POST_TASK_BLOCK_ID
    )
    root_flow.insert(
        post_flow_index + 1,
        {
            "Type": "Standard",
            "ID": ACTIONABILITY_BLOCK_ID,
            "FlowID": ACTIONABILITY_FLOW_ID,
            "Autofill": [],
        },
    )

    elements[:] = [
        element
        for element in elements
        if not (
            element.get("Element") == "SQ"
            and element.get("PrimaryAttribute")
            in CRT_QUESTION_IDS | {ACTIONABILITY_QUESTION_ID}
        )
    ]
    statistics_index = next(
        (index for index, element in enumerate(elements) if element.get("Element") == "STAT"),
        len(elements),
    )
    elements.insert(statistics_index, actionability_question(survey_id))

    OUTPUT.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "removed_crt_blocks": sorted(CRT_BLOCK_IDS),
                "removed_crt_questions": sorted(CRT_QUESTION_IDS),
                "added_question": ACTIONABILITY_QUESTION_ID,
                "rated_features": ["Glucose", "Blood Pressure", "Insulin", "BMI", "Age"],
                "scale": "1 (Strongly disagree) to 10 (Strongly agree)",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
