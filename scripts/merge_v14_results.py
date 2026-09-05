"""Append new v1.4-survey participants to qualtrics_results_v1.3."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "qualtrics" / "qualtrics_results_v1.3.csv"
NEW = ROOT / "qualtrics" / "v1.4_new_processed.csv"
OUTPUT = ROOT / "qualtrics" / "qualtrics_results_v1.4.csv"
SUMMARY = ROOT / "qualtrics" / "v1.4_results_summary.json"


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    old_columns, old_rows = read(OLD)
    new_columns, new_rows = read(NEW)
    if old_columns != new_columns:
        raise RuntimeError("Column schema/order mismatch")
    old_ids = {row["participant"] for row in old_rows}
    new_ids = {row["participant"] for row in new_rows}
    if old_ids & new_ids:
        raise RuntimeError(f"Participant overlap: {sorted(old_ids & new_ids)}")
    rows = old_rows + new_rows
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=old_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    participant_rows = Counter(row["participant"] for row in rows)
    summary = {
        "rows": len(rows),
        "columns": len(old_columns),
        "participants": len(participant_rows),
        "preserved_v1_3_participants": len(old_ids),
        "new_participants": len(new_ids),
        "participant_row_counts": dict(Counter(participant_rows.values())),
        "phase_rows": dict(Counter(row["phase"] for row in rows)),
        "xai_participants": {
            xai: len({row["participant"] for row in rows if row["xai"] == xai})
            for xai in sorted({row["xai"] for row in rows})
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
