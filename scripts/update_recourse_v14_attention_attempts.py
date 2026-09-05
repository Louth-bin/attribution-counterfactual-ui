"""Inject the shared frame script with attempt logging and MCQ checks into Recourse v1.4."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_recourse_v12 import replace_diabetes_ids


QSF = ROOT / "qualtrics" / "Recourse_v1.4.qsf"
FRAME_JS = ROOT / "qualtrics" / "qualtrics-frame.js"
STATIC_JSON = ROOT / "static" / "experiment-data.json"
QUESTION_IDS = ("QID12", "QID409", "QID376", "QID379")


def main() -> None:
    document = json.loads(QSF.read_text(encoding="utf-8"))
    static = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
    metadata = static["datasets"]["diabetes"]["metadata"]
    training_ids = [int(value) for value in metadata["qualtrics_v1_6_training_ids"]]
    testing_ids = {
        int(label): [int(value) for value in values]
        for label, values in metadata["qualtrics_v1_6_testing_ids_by_prediction"].items()
    }
    frame_js = replace_diabetes_ids(FRAME_JS.read_text(encoding="utf-8"), training_ids, testing_ids)

    questions = {
        element["PrimaryAttribute"]: element["Payload"]
        for element in document["SurveyElements"]
        if element.get("Element") == "SQ"
    }
    missing = [question_id for question_id in QUESTION_IDS if question_id not in questions]
    if missing:
        raise RuntimeError(f"Missing expected frame question(s): {missing}")

    for question_id in QUESTION_IDS:
        questions[question_id]["QuestionJS"] = frame_js

    document["SurveyEntry"]["LastModified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    QSF.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps({
        "qsf": str(QSF),
        "updated_question_js": list(QUESTION_IDS),
        "training_ids": training_ids,
        "testing_ids": testing_ids,
    }, indent=2))


if __name__ == "__main__":
    main()
