"""Append the one new complete response to qualtrics_results_v1.5."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "qualtrics" / "qualtrics_results_v1.5.csv"
NEW = ROOT / "qualtrics" / "v1.6_new_processed.csv"
OUTPUT = ROOT / "qualtrics" / "qualtrics_results_v1.6.csv"
SUMMARY = ROOT / "qualtrics" / "v1.6_results_summary.json"


def read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    columns, old_rows = read(OLD)
    new_columns, new_rows = read(NEW)
    if columns != new_columns:
        raise RuntimeError("Column schema/order mismatch")
    old_ids = {row["participant"] for row in old_rows}
    new_ids = {row["participant"] for row in new_rows}
    if old_ids & new_ids:
        raise RuntimeError("Participant overlap")
    rows = old_rows + new_rows
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["participant"] for row in rows)
    summary = {
        "rows": len(rows), "columns": len(columns), "participants": len(counts),
        "new_participants": len(new_ids), "new_rows": len(new_rows),
        "participant_row_counts": dict(Counter(counts.values())),
        "xai_participants": {x: len({r["participant"] for r in rows if r["xai"] == x}) for x in sorted({r["xai"] for r in rows})},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
