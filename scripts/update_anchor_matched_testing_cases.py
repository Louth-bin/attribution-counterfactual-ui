"""Replace weak v1.4 tests with explanation-matched candidates.

The Qualtrics case IDs are deliberately retained, so the existing QSF does not
need to be imported again.  The replacement rule uses only the currently
observed counterfactual-condition performance to decide which slots are weak;
candidate matching itself uses the algorithmic explanation data: identical
feature pair and prediction plus proximity to a successful case's normalized
minimal-counterfactual change vector.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_recourse_v12 import build_payload
from scripts.generate_static_experiment import build_metadata
from src.pipeline import ExplanationPipeline


STATIC_JSON = ROOT / "static" / "experiment-data.json"
STATIC_JS = ROOT / "static" / "experiment-data.js"
CURRENT_SELECTION = ROOT / "analysis" / "diabetes_hierarchical_oriented_v6_preserved.csv"
CANDIDATE_FILES = (
    ROOT / "analysis" / "diabetes_hierarchical_oriented_v6_reselected_candidate_shortlist.csv",
    ROOT / "analysis" / "diabetes_hierarchical_oriented_q90_anchor_search_candidate_shortlist.csv",
)
OUTPUT_SELECTION = ROOT / "analysis" / "diabetes_anchor_matched_testing_v1.5.csv"
CURRENT_MANIFEST = ROOT / "qualtrics" / "case-manifest-v1.4.json"
OUTPUT_MANIFEST = ROOT / "qualtrics" / "case-manifest-v1.5.json"
OUTPUT_ANALYSIS = ROOT / "analysis" / "diabetes_experiment_bundle_v1.5.json"


# Weak on both observed success and successful-edit proximity.  Replacements
# were chosen as the nearest unused normalized minimal-CF vector to a strong
# case in the same glucose/insulin and prediction stratum.
REPLACEMENTS = {
    130200: ("dev", 41),
    130206: ("train", 62),
    130307: ("dev", 47),
    130308: ("train", 272),
}

ANCHORS = {
    130200: 130207,
    130206: 130207,
    130307: 130300,
    130308: 130309,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def source_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["source_split"]), int(row["instance_id"])


def main() -> None:
    bundle = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
    diabetes = bundle["datasets"]["diabetes"]
    existing_by_id = {
        int(case["instance_id"]): case for case in diabetes["test_pool"]
    }
    manifest = json.loads(CURRENT_MANIFEST.read_text(encoding="utf-8"))["cases"]
    qid_by_source = {
        (str(row["source_split"]), int(row["source_instance_id"])):
            int(row["qualtrics_instance_id"])
        for row in manifest
    }

    current_rows = read_csv(CURRENT_SELECTION)
    candidate_by_source: dict[tuple[str, int], dict[str, str]] = {}
    for candidate_file in CANDIDATE_FILES:
        candidate_by_source.update({
            source_key(row): row for row in read_csv(candidate_file)
        })
    replacement_rows: dict[int, dict[str, str]] = {}
    pipeline = ExplanationPipeline()

    # Reconstruct the complete v1.4 test set first.  This makes rerunning this
    # script idempotent even if the static bundle already contains an earlier
    # proposed replacement set.
    for row in current_rows:
        if row["experimental_phase"] != "testing":
            continue
        qualtrics_id = qid_by_source[source_key(row)]
        original = dict(row)
        original["cluster"] = original["subcluster"]
        original["cluster_rank"] = original["main_cluster_rank"]
        existing_by_id[qualtrics_id] = build_payload(pipeline, original, qualtrics_id)

    for qualtrics_id, candidate_key in REPLACEMENTS.items():
        old_case = existing_by_id[qualtrics_id]
        row = dict(candidate_by_source[candidate_key])
        if row["main_pair"].replace(" ", "") != old_case["feature_pair_key"]:
            raise RuntimeError(f"Feature-pair mismatch for {qualtrics_id}")
        if int(row["prediction"]) != int(old_case["prediction"]["value"]):
            raise RuntimeError(f"Prediction mismatch for {qualtrics_id}")
        row["experimental_phase"] = "testing"
        row["cluster"] = str(old_case["selection_cluster"])
        row["cluster_rank"] = str(old_case["selection_cluster_rank"])
        row["selection_role"] = (
            f"anchor_matched_to_{ANCHORS[qualtrics_id]}_by_normalized_minimal_cf"
        )
        replacement_rows[qualtrics_id] = row
        existing_by_id[qualtrics_id] = build_payload(pipeline, row, qualtrics_id)

    selected_output: list[dict[str, Any]] = []
    for row in current_rows:
        key = source_key(row)
        qualtrics_id = qid_by_source[key]
        selected = dict(replacement_rows.get(qualtrics_id, row))
        selected["qualtrics_instance_id"] = qualtrics_id
        selected["anchor_instance_id"] = ANCHORS.get(qualtrics_id, "")
        selected_output.append(selected)

    fields = list(selected_output[0])
    with OUTPUT_SELECTION.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_output)

    all_test_ids = list(range(130200, 130210)) + list(range(130300, 130310))
    testing = [existing_by_id[case_id] for case_id in all_test_ids]
    diabetes["test_pool"] = [
        case for case in diabetes["test_pool"]
        if int(case["instance_id"]) not in set(all_test_ids)
    ] + testing
    metadata = diabetes["metadata"]
    metadata["static_test_pool_count"] = len(diabetes["test_pool"])
    metadata["qualtrics_v1_5_testing_ids_by_prediction"] = {
        "0": list(range(130200, 130210)),
        "1": list(range(130300, 130310)),
    }
    metadata["qualtrics_v1_5_replaced_test_ids"] = sorted(REPLACEMENTS)
    metadata["qualtrics_v1_5_anchor_for_replacement"] = {
        str(case_id): anchor for case_id, anchor in ANCHORS.items()
    }
    bundle["version"] = "static-experiment-v15-anchor-matched-tests"
    bundle["generated_at"] = date.today().isoformat()
    STATIC_JSON.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    STATIC_JS.write_text(
        "window.EXPERIMENT_DATA = "
        + json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )

    training_ids = metadata["qualtrics_v1_4_training_ids"]
    training_by_id = {
        int(case["instance_id"]): case for case in diabetes["training_pool"]
    }
    training = [training_by_id[int(case_id)] for case_id in training_ids]
    analysis_dataset = {
        "metadata": build_metadata(pipeline, "diabetes", training, testing),
        "browser_model": diabetes["browser_model"],
        "training_pool": training,
        "test_pool": testing,
    }
    OUTPUT_ANALYSIS.write_text(
        json.dumps({
            "version": "diabetes-experiment-v1.5-anchor-matched-tests",
            "generated_at": date.today().isoformat(),
            "default_model": "mlp",
            "datasets": {"diabetes": analysis_dataset},
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_rows = []
    for case in training + testing:
        manifest_rows.append({
            "domain": "diabetes",
            "experimental_phase": case["experimental_phase"],
            "qualtrics_instance_id": case["instance_id"],
            "source_split": case["source_split"],
            "source_instance_id": case["source_instance_id"],
            "prediction": int(case["prediction"]["value"]),
            "prediction_label": case["prediction"]["label"],
            "selection_cluster": case["selection_cluster"],
            "selection_cluster_rank": case["selection_cluster_rank"],
            "feature_pair_key": case["feature_pair_key"],
            "anchor_instance_id": ANCHORS.get(int(case["instance_id"])),
        })
    OUTPUT_MANIFEST.write_text(
        json.dumps({"version": "1.5", "cases": manifest_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "replacements": {
            str(case_id): {
                "source": f"{source[0]}:{source[1]}",
                "anchor": ANCHORS[case_id],
            }
            for case_id, source in REPLACEMENTS.items()
        },
        "kept_test_cases": 20 - len(REPLACEMENTS),
        "qsf_reimport_required": False,
    }, indent=2))


if __name__ == "__main__":
    main()
