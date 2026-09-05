"""Prepare the mixed old/new-instance inputs for qualtrics_results_v1.5.

Participants completing after the recorded deployment cutoff saw four changed
profiles under the same Qualtrics IDs.  Their testing logs are remapped to new
analysis-only IDs, while pre-deployment responses retain the original IDs.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "qualtrics" / "raw_output_v1.4 (2 clusters 20 instances).csv"
EXISTING = ROOT / "qualtrics" / "qualtrics_results_v1.4.csv"
OLD_BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.4.json"
LIVE_BUNDLE = ROOT / "static" / "experiment-data.json"
OUTPUT_RAW = ROOT / "qualtrics" / "v1.5_new_raw_remapped.csv"
OUTPUT_BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.5_results.json"
OUTPUT_AUDIT = ROOT / "qualtrics" / "v1.5_instance_id_remap.json"

CUTOFF = datetime.fromisoformat("2026-08-27 12:09:54")
ID_REMAP = {
    130200: 140200,
    130206: 140206,
    130307: 140307,
    130308: 140308,
}


def completed_at(row: dict[str, str]) -> datetime:
    return datetime.fromisoformat(row["EndDate"])


def main() -> None:
    with EXISTING.open(encoding="utf-8-sig", newline="") as source:
        old_participants = {row["participant"] for row in csv.DictReader(source)}

    with RAW.open(encoding="utf-8-sig", newline="") as source:
        raw_rows = list(csv.reader(source))
    headers = raw_rows[0]
    records = [dict(zip(headers, row)) for row in raw_rows[3:]]
    new_records = [
        row for row in records
        if row.get("Finished") == "1"
        and row.get("ResponseId")
        and row["ResponseId"] not in old_participants
        and len(json.loads(row.get("training_log_json") or "[]")) == 12
        and len(json.loads(row.get("testing_log_json") or "[]")) == 20
    ]
    if len(new_records) != 14:
        raise RuntimeError(f"Expected 14 complete-task new participants, found {len(new_records)}")

    post_cutoff = []
    pre_cutoff = []
    for row in new_records:
        participant = row["ResponseId"]
        if completed_at(row) > CUTOFF:
            logs = json.loads(row["testing_log_json"])
            seen = set()
            for log in logs:
                old_id = int(log["instanceId"])
                if old_id in ID_REMAP:
                    log["instanceId"] = ID_REMAP[old_id]
                    seen.add(old_id)
            if seen != set(ID_REMAP):
                raise RuntimeError(
                    f"Post-cutoff participant {participant} did not see all swapped IDs: {sorted(seen)}"
                )
            row["testing_log_json"] = json.dumps(logs, separators=(",", ":"))
            post_cutoff.append(participant)
        else:
            pre_cutoff.append(participant)

    with OUTPUT_RAW.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerows(raw_rows[:3])
        writer.writerows([[row.get(header, "") for header in headers] for row in new_records])

    old_bundle = json.loads(OLD_BUNDLE.read_text(encoding="utf-8"))
    live_bundle = json.loads(LIVE_BUNDLE.read_text(encoding="utf-8"))
    old_dataset = old_bundle["datasets"]["diabetes"]
    live_cases = {
        int(case["instance_id"]): case
        for case in live_bundle["datasets"]["diabetes"]["test_pool"]
    }
    existing_ids = {int(case["instance_id"]) for case in old_dataset["test_pool"]}
    for old_id, new_id in ID_REMAP.items():
        if old_id not in live_cases or new_id in existing_ids:
            raise RuntimeError(f"Cannot construct remapped case {old_id} -> {new_id}")
        case = deepcopy(live_cases[old_id])
        case["instance_id"] = new_id
        case["analysis_instance_id"] = new_id
        case["qualtrics_instance_id"] = old_id
        case["collection_wave"] = "post-2026-08-27-120954"
        old_dataset["test_pool"].append(case)
    old_dataset["metadata"]["results_instance_id_remap_v1_5"] = {
        str(old_id): new_id for old_id, new_id in ID_REMAP.items()
    }
    old_bundle["version"] = "diabetes-experiment-v1.5-results-mixed-instance-waves"
    OUTPUT_BUNDLE.write_text(
        json.dumps(old_bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    audit = {
        "deployment_cutoff_singapore": "2026-08-27 12:09:54 +08:00",
        "new_participants": len(new_records),
        "pre_cutoff_participants": pre_cutoff,
        "post_cutoff_participants": post_cutoff,
        "instance_id_remap_post_cutoff": {str(k): v for k, v in ID_REMAP.items()},
    }
    OUTPUT_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
