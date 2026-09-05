"""Add post-task strategy questions and the four-item CRT-2 to Recourse v10.

The update is idempotent and can be applied either to a freshly generated v10
QSF or to a survey exported from Qualtrics after manual edits.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QSF = REPO_ROOT / "qualtrics" / "Recourse_v10.qsf"

POST_TASK_BLOCK = "BL_10PostTaskReflection"
CRT2_INTRO_BLOCK = "BL_10CRT2Intro"
CRT2_ITEMS_BLOCK = "BL_10CRT2Items"
DEMOGRAPHICS_BLOCK = "BL_d07rL5ybaNgQsMm"

POST_TASK_QUESTION_IDS = ["QID266", "QID36", "QID37"]
CRT2_QUESTION_IDS = ["QID405", "QID406", "QID407", "QID408"]
OBSOLETE_STRATEGY_QUESTION_IDS = {"QID401", "QID402", "QID403"}


def question_elements(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        element["PrimaryAttribute"]: element
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }


def survey_blocks(document: dict[str, Any]) -> list[dict[str, Any]]:
    payload = next(
        element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "BL"
    )
    return list(payload.values()) if isinstance(payload, dict) else payload


def set_survey_blocks(document: dict[str, Any], values: list[dict[str, Any]]) -> None:
    element = next(
        element for element in document["SurveyElements"] if element.get("Element") == "BL"
    )
    element["Payload"] = values


def block(block_id: str, description: str, question_ids: list[str]) -> dict[str, Any]:
    return {
        "Type": "Standard",
        "SubType": "",
        "Description": description,
        "ID": block_id,
        "BlockElements": [
            {"Type": "Question", "QuestionID": question_id}
            for question_id in question_ids
        ],
        "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "false",
            "BlockVisibility": "Collapsed",
        },
    }


def clone_question(
    document: dict[str, Any],
    source: dict[str, Any],
    question_id: str,
    export_tag: str,
    description: str,
) -> dict[str, Any]:
    element = copy.deepcopy(source)
    element["SurveyID"] = document["SurveyEntry"]["SurveyID"]
    element["PrimaryAttribute"] = question_id
    element["SecondaryAttribute"] = description
    payload = element["Payload"]
    payload["QuestionID"] = question_id
    payload["DataExportTag"] = export_tag
    payload["QuestionDescription"] = description
    document["SurveyElements"].append(element)
    return payload


def require_response(payload: dict[str, Any]) -> None:
    payload.setdefault("Validation", {})["Settings"] = {
        "ForceResponse": "ON",
        "ForceResponseType": "ON",
        "Type": "None",
    }


def configure_text_entry(
    payload: dict[str, Any],
    text: str,
    *,
    multiline: bool,
    required: bool = True,
) -> None:
    payload["QuestionText"] = text
    payload["QuestionType"] = "TE"
    payload["Selector"] = "ML" if multiline else "SL"
    payload.pop("SubSelector", None)
    payload.pop("Choices", None)
    payload["ChoiceOrder"] = []
    payload["NextChoiceId"] = 1
    payload["NextAnswerId"] = 1
    payload["QuestionJS"] = ""
    payload.pop("DisplayLogic", None)
    if required:
        require_response(payload)


def configure_display(payload: dict[str, Any], text: str) -> None:
    payload["QuestionText"] = text
    payload["QuestionType"] = "DB"
    payload["Selector"] = "TB"
    payload.pop("SubSelector", None)
    payload.pop("Choices", None)
    payload["ChoiceOrder"] = []
    payload["Validation"] = {"Settings": {"Type": "None"}}
    payload["QuestionJS"] = ""
    payload.pop("DisplayLogic", None)


def update_intro_and_consent(questions: dict[str, dict[str, Any]]) -> None:
    questions["QID286"]["Payload"]["QuestionText"] = """<h1>Research Study on AI Understanding</h1>
<p>Welcome! Thank you for your interest in our research study. <u>If you recently completed this study, please do not participate again.</u> <span style="color:#e74c3c;">Resubmissions may be rejected.</span></p>
<p>We are exploring how people understand and change AI decisions. This study involves:</p>
<ol>
  <li>Consenting to participate.</li>
  <li>Completing a brief tutorial and screening questions.</li>
  <li>A training session involving 10 trials.</li>
  <li>A testing session involving 20 trials.</li>
  <li>Answering short questions about how you approached the task.</li>
  <li>Answering four brief general problem-solving questions.</li>
  <li>Answering demographic questions.</li>
</ol>
<p>The survey takes about 25 minutes to complete.</p>"""

    consent = questions["QID5"]["Payload"]
    text = consent["QuestionText"]
    text = text.replace("Consent \u00adForm", "Consent Form")
    text = text.replace("people\u2019s", "people's").replace("people\ufffds", "people's")
    old_purpose = "This study aims to investigate people's ability in understanding AI decisions."
    new_purpose = (
        "This study aims to investigate how people understand and change AI decisions, "
        "and whether general problem-solving approaches are related to how people perform this task."
    )
    if new_purpose not in text:
        if old_purpose not in text:
            raise RuntimeError("Could not locate the consent purpose text")
        text = text.replace(old_purpose, new_purpose, 1)
    old_procedure = (
        "You will study a tutorial and take a screening test to qualify for the main study. "
        "Then you will go through 3 sessions for the main study. Subsequently, you will answer some demographic questions"
    )
    new_procedure = (
        "You will complete a tutorial and screening test, followed by training and testing tasks involving AI decisions. "
        "After the main task, you will answer questions about how you approached it, four short general problem-solving "
        "questions, and demographic questions."
    )
    if new_procedure not in text:
        if old_procedure not in text:
            raise RuntimeError("Could not locate the consent procedure text")
        text = text.replace(old_procedure, new_procedure, 1)
    consent["QuestionText"] = text


def add_or_update_questions(document: dict[str, Any]) -> None:
    questions = question_elements(document)
    text_template = questions.get("QID37") or questions["QID75"]
    display_template = questions["QID266"]

    if "QID39" not in questions:
        clone_question(
            document,
            display_template,
            "QID39",
            "Demographics_Intro",
            "Demographic questions introduction",
        )

    if "QID36" not in questions:
        clone_question(
            document,
            text_template,
            "QID36",
            "PostTask_MentalModel",
            "Post-task AI rule",
        )
    if "QID37" not in questions:
        clone_question(
            document,
            text_template,
            "QID37",
            "PostTask_StrategyExample",
            "Post-task strategy example",
        )
    questions = question_elements(document)

    questions["QID39"]["Payload"]["DataExportTag"] = "Demographics_Intro"
    questions["QID39"]["Payload"]["QuestionDescription"] = "Demographic questions introduction"
    questions["QID39"]["SecondaryAttribute"] = "Demographic questions introduction"
    configure_display(
        questions["QID39"]["Payload"],
        "<h1>Demographic Questions</h1>",
    )

    questions["QID266"]["Payload"]["DataExportTag"] = "PostTask_Intro"
    questions["QID266"]["Payload"]["QuestionDescription"] = "Post-task reflection introduction"
    questions["QID266"]["SecondaryAttribute"] = "Post-task reflection introduction"
    configure_display(
        questions["QID266"]["Payload"],
        """<h1>How you approached the AI task</h1>
<p>The main task is complete. Please describe the reasoning you actually used. There are no right or wrong strategies; specific details are more useful than general statements.</p>""",
    )

    questions["QID36"]["Payload"]["DataExportTag"] = "PostTask_MentalModel"
    questions["QID36"]["Payload"]["QuestionDescription"] = "Post-task AI rule"
    questions["QID36"]["SecondaryAttribute"] = "Post-task AI rule"
    configure_text_entry(
        questions["QID36"]["Payload"],
        """What rule do you think the AI used to decide between the two outcomes?<br><br>
Please mention which attributes mattered, whether higher or lower values led toward each outcome, and any thresholds or ranges you noticed.""",
        multiline=True,
    )

    questions["QID37"]["Payload"]["DataExportTag"] = "PostTask_StrategyExample"
    questions["QID37"]["Payload"]["QuestionDescription"] = "Post-task strategy example"
    questions["QID37"]["SecondaryAttribute"] = "Post-task strategy example"
    configure_text_entry(
        questions["QID37"]["Payload"],
        """Thinking across the testing profiles, how did you decide:<ol>
  <li>which attributes to change,</li>
  <li>what new values to use, and</li>
  <li>when you had changed enough?</li>
</ol>
If your approach varied between profiles, explain what made it vary.""",
        multiline=True,
    )

    questions = question_elements(document)
    if "QID404" not in questions:
        clone_question(
            document,
            display_template,
            "QID404",
            "CRT2_Intro",
            "General problem-solving questions introduction",
        )
    questions = question_elements(document)
    configure_display(
        questions["QID404"]["Payload"],
        """<h1>Short problem-solving questions</h1>
<p>Next are four brief, self-contained questions about everyday situations. They help us examine whether general approaches to reasoning are associated with how people approached the AI task.</p>
<p>These questions are <b>not an intelligence test</b> and do not affect your eligibility, payment, or performance on the main task. Please answer on your own, without searching or using external tools.</p>""",
    )

    crt_items = {
        "QID405": (
            "CRT2_Race",
            "CRT-2 race item",
            "If you're running a race and you pass the person in second place, what place are you in?",
        ),
        "QID406": (
            "CRT2_Sheep",
            "CRT-2 sheep item",
            "A farmer had 15 sheep and all but 8 died. How many are left?",
        ),
        "QID407": (
            "CRT2_Emily",
            "CRT-2 daughters item",
            "Emily's father has three daughters. The first two are named April and May. What is the third daughter's name?",
        ),
        "QID408": (
            "CRT2_Hole",
            "CRT-2 hole item",
            "How many cubic feet of dirt are there in a hole that is 3' deep × 3' wide × 3' long?",
        ),
    }
    for question_id, (export_tag, description, text) in crt_items.items():
        questions = question_elements(document)
        if question_id not in questions:
            clone_question(document, text_template, question_id, export_tag, description)
        questions = question_elements(document)
        payload = questions[question_id]["Payload"]
        payload["DataExportTag"] = export_tag
        payload["QuestionDescription"] = description
        questions[question_id]["SecondaryAttribute"] = description
        configure_text_entry(payload, text, multiline=False)


def update_blocks_and_flow(document: dict[str, Any]) -> None:
    values = [
        value
        for value in survey_blocks(document)
        if value.get("ID") not in {POST_TASK_BLOCK, CRT2_INTRO_BLOCK, CRT2_ITEMS_BLOCK}
    ]
    demographics = next(value for value in values if value.get("ID") == DEMOGRAPHICS_BLOCK)
    values.remove(demographics)
    demographics["Description"] = "Demographic Questions"
    demographics["BlockElements"] = [
        {"Type": "Question", "QuestionID": "QID39"},
        {"Type": "Question", "QuestionID": "QID71"},
        {"Type": "Question", "QuestionID": "QID70"},
        {"Type": "Question", "QuestionID": "QID72"},
        {"Type": "Question", "QuestionID": "QID73"},
        {"Type": "Question", "QuestionID": "QID74"},
        {"Type": "Question", "QuestionID": "QID75"},
    ]
    post_task = block(POST_TASK_BLOCK, "Post-task strategy questions", POST_TASK_QUESTION_IDS)
    values.extend([
        post_task,
        block(CRT2_INTRO_BLOCK, "General problem-solving introduction", ["QID404"]),
        block(CRT2_ITEMS_BLOCK, "CRT-2: Four reasoning questions", CRT2_QUESTION_IDS),
        demographics,
    ])
    set_survey_blocks(document, values)

    flow_payload = next(
        element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "FL"
    )
    flow = [
        item
        for item in flow_payload["Flow"]
        if item.get("ID") not in {POST_TASK_BLOCK, CRT2_INTRO_BLOCK, CRT2_ITEMS_BLOCK}
    ]
    demographics_flow = next(item for item in flow if item.get("ID") == DEMOGRAPHICS_BLOCK)
    flow.remove(demographics_flow)
    flow.extend([
        {"Type": "Standard", "ID": POST_TASK_BLOCK, "FlowID": "FL_37", "Autofill": []},
        {"Type": "Standard", "ID": CRT2_INTRO_BLOCK, "FlowID": "FL_38", "Autofill": []},
        {"Type": "Standard", "ID": CRT2_ITEMS_BLOCK, "FlowID": "FL_39", "Autofill": []},
        demographics_flow,
    ])
    flow_payload["Flow"] = flow
    flow_payload.setdefault("Properties", {})["Count"] = max(
        int(flow_payload.get("Properties", {}).get("Count", 0)), 39
    )


def update_question_count(document: dict[str, Any]) -> None:
    count = sum(element.get("Element") == "SQ" for element in document["SurveyElements"])
    qc = next(element for element in document["SurveyElements"] if element.get("Element") == "QC")
    qc["SecondaryAttribute"] = str(count)


def apply_post_task_and_crt2(document: dict[str, Any]) -> None:
    document["SurveyElements"] = [
        element
        for element in document["SurveyElements"]
        if element.get("PrimaryAttribute") not in OBSOLETE_STRATEGY_QUESTION_IDS
    ]
    questions = question_elements(document)
    update_intro_and_consent(questions)
    add_or_update_questions(document)
    update_blocks_and_flow(document)
    update_question_count(document)
    document["SurveyEntry"]["SurveyDescription"] = (
        "Qualtrics-hosted study for housing, drink-driving, and diabetes warning domains; "
        "10 training and 20 testing cases, detailed post-task strategy questions, CRT-2, "
        "and two-attempt screening."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_QSF)
    parser.add_argument("--output", type=Path, default=DEFAULT_QSF)
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8-sig"))
    apply_post_task_and_crt2(document)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
