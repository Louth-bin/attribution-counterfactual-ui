"""Randomize all Recourse_v08 case loops and synchronize their shared script."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QSF_PATH = REPO_ROOT / "qualtrics" / "Recourse_v08.qsf"
FRAME_JS_PATH = REPO_ROOT / "qualtrics" / "qualtrics-frame.js"
LOOP_BLOCK_IDS = {
    "BL_3fJJUKH6EXaeRIq",
    "BL_5aT4RYQul04UaHA",
    "BL_3W1PIk3tbtEX0BE",
}
FRAME_QUESTION_IDS = {"QID12", "QID376", "QID379"}


def main() -> None:
    document = json.loads(QSF_PATH.read_text(encoding="utf-8-sig"))
    frame_js = FRAME_JS_PATH.read_text(encoding="utf-8")

    block_payload = next(
        element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "BL"
    )
    blocks = block_payload.values() if isinstance(block_payload, dict) else block_payload
    updated_blocks = set()
    for block in blocks:
        block_id = block.get("ID")
        if block_id not in LOOP_BLOCK_IDS:
            continue
        loop_options = block["Options"]["LoopingOptions"]
        rows = loop_options["Static"]
        if len(rows) != 10 or [rows[str(i)]["1"] for i in range(1, 11)] != [
            str(i) for i in range(10)
        ]:
            raise RuntimeError(f"{block_id} is not the expected ten-row loop")
        loop_options["Randomization"] = "All"
        updated_blocks.add(block_id)
    if updated_blocks != LOOP_BLOCK_IDS:
        raise RuntimeError(f"Missing loop blocks: {LOOP_BLOCK_IDS - updated_blocks}")

    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    for question_id in FRAME_QUESTION_IDS:
        questions[question_id]["QuestionJS"] = frame_js

    QSF_PATH.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(QSF_PATH)


if __name__ == "__main__":
    main()
