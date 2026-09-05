"""Isolate and remap the eight new-instance complete responses for v1.8 conversion."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "qualtrics" / "raw_output_v1.5 (2 clusters new clustering).csv"
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
MANIFEST = ROOT / "qualtrics" / "case-manifest-v1.6.json"
PRESERVED = (
    ROOT / "qualtrics" / "qualtrics_results_v1.5.csv",
    ROOT / "qualtrics" / "v1.6_new_processed.csv",
)
OUTPUT = ROOT / "qualtrics" / "v1.8_new_raw_remapped.csv"
SUMMARY = ROOT / "qualtrics" / "v1.8_new_raw_remapped_summary.json"
EXPECTED_NEW = 8


def participant_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return {row["participant"] for row in csv.DictReader(source)}


def case_id_remap() -> dict[int, int]:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    bundle_cases = list(bundle["training_pool"]) + list(bundle["test_pool"])
    by_source = {
        (
            case["experimental_phase"],
            case["source_split"],
            int(case["source_instance_id"]),
        ): int(case["qualtrics_instance_id"])
        for case in manifest
    }
    mapping = {
        by_source[
            (
                case["experimental_phase"],
                case["source_split"],
                int(case["source_instance_id"]),
            )
        ]: int(case["instance_id"])
        for case in bundle_cases
    }
    if len(mapping) != 32 or len(set(mapping.values())) != 32:
        raise RuntimeError(f"Expected a one-to-one 32-case v1.6 mapping, found {len(mapping)}")
    return mapping


def main() -> None:
    old_participants: set[str] = set()
    for path in PRESERVED:
        old_participants.update(participant_ids(path))

    with RAW.open(encoding="utf-8-sig", newline="") as source:
        raw_rows = list(csv.reader(source))
    headers = raw_rows[0]
    records = [dict(zip(headers, row)) for row in raw_rows[3:]]
    candidates = [row for row in records if row.get("ResponseId") not in old_participants]
    complete = [
        row
        for row in candidates
        if row.get("Finished") == "1"
        and row.get("ResponseId")
        and len(json.loads(row.get("training_log_json") or "[]")) == 12
        and len(json.loads(row.get("testing_log_json") or "[]")) == 20
    ]
    if len(complete) != EXPECTED_NEW:
        raise RuntimeError(
            f"Expected {EXPECTED_NEW} new complete participants, found {len(complete)}"
        )

    id_remap = case_id_remap()
    remap_counts: Counter[int] = Counter()
    for row in complete:
        participant_seen: set[int] = set()
        for field in ("training_log_json", "testing_log_json"):
            logs = json.loads(row[field])
            for log in logs:
                old_id = int(log["instanceId"])
                if old_id not in id_remap:
                    raise RuntimeError(
                        f"Participant {row['ResponseId']} contains unmapped case {old_id}"
                    )
                log["instanceId"] = id_remap[old_id]
                participant_seen.add(old_id)
                remap_counts[old_id] += 1
            row[field] = json.dumps(logs, separators=(",", ":"))
        if participant_seen != set(id_remap):
            raise RuntimeError(
                f"Participant {row['ResponseId']} did not see all v1.6 cases"
            )

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerows(raw_rows[:3])
        writer.writerows([[row.get(header, "") for header in headers] for row in complete])

    summary = {
        "raw_records": len(records),
        "preserved_participants": len(old_participants),
        "unseen_records": len(candidates),
        "excluded_empty_log_records": len(candidates) - len(complete),
        "new_complete_participants": len(complete),
        "xai_participants": dict(sorted(Counter(row["xaiType"] for row in complete).items())),
        "source_bundle": str(BUNDLE.relative_to(ROOT)),
        "analysis_id_range": [min(id_remap.values()), max(id_remap.values())],
        "remapped_case_rows": {
            str(case_id): remap_counts[case_id] for case_id in sorted(id_remap)
        },
        "participants": [row["ResponseId"] for row in complete],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

