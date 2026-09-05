"""Isolate the new complete response from raw v1.5 for v1.6 conversion."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "qualtrics" / "raw_output_v1.5 (2 clusters new clustering).csv"
EXISTING = ROOT / "qualtrics" / "qualtrics_results_v1.5.csv"
OUTPUT = ROOT / "qualtrics" / "v1.6_new_raw_remapped.csv"
ID_REMAP = {130200: 140200, 130206: 140206, 130307: 140307, 130308: 140308}


def main() -> None:
    with EXISTING.open(encoding="utf-8-sig", newline="") as source:
        old = {row["participant"] for row in csv.DictReader(source)}
    with RAW.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
    headers = rows[0]
    records = [dict(zip(headers, row)) for row in rows[3:]]
    new = [
        row for row in records
        if row.get("Finished") == "1"
        and row.get("ResponseId") not in old
        and len(json.loads(row.get("training_log_json") or "[]")) == 12
        and len(json.loads(row.get("testing_log_json") or "[]")) == 20
    ]
    if len(new) != 1:
        raise RuntimeError(f"Expected one new complete participant, found {len(new)}")
    logs = json.loads(new[0]["testing_log_json"])
    seen = set()
    for log in logs:
        case_id = int(log["instanceId"])
        if case_id in ID_REMAP:
            log["instanceId"] = ID_REMAP[case_id]
            seen.add(case_id)
    if seen != set(ID_REMAP):
        raise RuntimeError(f"New participant did not see all swapped cases: {sorted(seen)}")
    new[0]["testing_log_json"] = json.dumps(logs, separators=(",", ":"))
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerows(rows[:3])
        writer.writerow([new[0].get(header, "") for header in headers])
    print(json.dumps({"participant": new[0]["ResponseId"], "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
