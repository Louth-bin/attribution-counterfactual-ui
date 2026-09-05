"""Promote the validated regularized-MLP preview into existing Qualtrics slots.

The current QSF already references 130100-130111 for training and
130200-130209/130300-130309 for testing, so no QSF re-import is required.
Generated files are written to the system temp directory for a safe copy step.
"""

from __future__ import annotations

import copy
import json
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static" / "experiment-data.json"
PREVIEW = ROOT / "analysis" / "regularized_mlp_two_cluster_preview.json"
TEMP = Path(tempfile.gettempdir())

TRAIN_IDS = list(range(130100, 130112))
TEST_IDS = {0: list(range(130200, 130210)), 1: list(range(130300, 130310))}


def main() -> None:
    bundle = json.loads(STATIC.read_text(encoding="utf-8"))
    preview = json.loads(PREVIEW.read_text(encoding="utf-8"))
    source = preview["datasets"]["diabetes"]
    diabetes = bundle["datasets"]["diabetes"]

    training = sorted(
        (copy.deepcopy(case) for case in source["training_pool"]),
        key=lambda case: (
            int(case["selection_cluster"]),
            int(case["prediction"]["value"]),
            str(case["source_split"]),
            int(case["source_instance_id"]),
        ),
    )
    if len(training) != 12:
        raise RuntimeError(f"Expected 12 training cases, found {len(training)}")
    for assigned_id, case in zip(TRAIN_IDS, training):
        case["instance_id"] = assigned_id

    testing = []
    for prediction in (0, 1):
        group = sorted(
            (
                copy.deepcopy(case)
                for case in source["test_pool"]
                if int(case["prediction"]["value"]) == prediction
            ),
            key=lambda case: (
                int(case["selection_cluster"]),
                str(case["source_split"]),
                int(case["source_instance_id"]),
            ),
        )
        if len(group) != 10:
            raise RuntimeError(f"Expected 10 testing cases for prediction {prediction}, found {len(group)}")
        for assigned_id, case in zip(TEST_IDS[prediction], group):
            case["instance_id"] = assigned_id
        testing.extend(group)

    experiment_ids = set(TRAIN_IDS + TEST_IDS[0] + TEST_IDS[1])
    diabetes["training_pool"] = [
        case for case in diabetes["training_pool"]
        if int(case["instance_id"]) not in experiment_ids
    ] + training
    diabetes["test_pool"] = [
        case for case in diabetes["test_pool"]
        if int(case["instance_id"]) not in experiment_ids
    ] + testing
    diabetes["browser_model"] = source["browser_model"]

    metadata = diabetes["metadata"]
    metadata.update({
        "model": "regularized_mlp_alpha_0.01",
        "xai_methods": ["lime"],
        "static_training_pool_count": len(diabetes["training_pool"]),
        "static_test_pool_count": len(diabetes["test_pool"]),
        "qualtrics_v1_6_training_ids": TRAIN_IDS,
        "qualtrics_v1_6_testing_ids_by_prediction": {
            str(prediction): ids for prediction, ids in TEST_IDS.items()
        },
        "qualtrics_v1_6_training_blocks": {
            "glucose_bmi": TRAIN_IDS[:6],
            "blood_pressure_insulin": TRAIN_IDS[6:],
        },
        "qualtrics_v1_6_minimum_normalized_change": 0.10,
        "qualtrics_v1_6_selection": (
            "DBSCAN on label-adjusted LIME values and counterfactual endpoints; "
            "training originals required on opposite sides of the common endpoint centre"
        ),
    })
    bundle["version"] = "static-experiment-v16-regularized-mlp-lime-min10"
    bundle["generated_at"] = date.today().isoformat()

    cases_by_id = {
        int(case["instance_id"]): case
        for case in diabetes["training_pool"] + diabetes["test_pool"]
    }
    missing = sorted(experiment_ids - set(cases_by_id))
    if missing:
        raise RuntimeError(f"Missing promoted IDs: {missing}")
    if Counter(cases_by_id[case_id]["experimental_phase"] for case_id in TRAIN_IDS) != {"training": 12}:
        raise RuntimeError("Training phase mismatch")
    if Counter(cases_by_id[case_id]["experimental_phase"] for case_id in TEST_IDS[0] + TEST_IDS[1]) != {"testing": 20}:
        raise RuntimeError("Testing phase mismatch")

    manifest = []
    for case_id in TRAIN_IDS + TEST_IDS[0] + TEST_IDS[1]:
        case = cases_by_id[case_id]
        manifest.append({
            "domain": "diabetes",
            "experimental_phase": case["experimental_phase"],
            "qualtrics_instance_id": case_id,
            "source_split": case["source_split"],
            "source_instance_id": case["source_instance_id"],
            "source_row_id": case.get("source_row_id"),
            "prediction": int(case["prediction"]["value"]),
            "prediction_label": case["prediction"]["label"],
            "selection_cluster": int(case["selection_cluster"]),
            "feature_pair_key": case["feature_pair_key"],
            "minimum_normalized_change": 0.10,
        })

    json_text = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
    (TEMP / "experiment-data-v16.json").write_text(json_text, encoding="utf-8")
    (TEMP / "experiment-data-v16.js").write_text(
        "window.EXPERIMENT_DATA = "
        + json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    (TEMP / "case-manifest-v1.6.json").write_text(
        json.dumps({"version": "1.6", "cases": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    (TEMP / "diabetes-experiment-bundle-v1.6.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "version": bundle["version"],
        "training_ids": TRAIN_IDS,
        "testing_ids": TEST_IDS,
        "qsf_reimport_required": False,
        "temp_json": str(TEMP / "experiment-data-v16.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
