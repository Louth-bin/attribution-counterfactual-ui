"""Append the remapped v1.5 collection wave to qualtrics_results_v1.4."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "qualtrics" / "qualtrics_results_v1.4.csv"
NEW = ROOT / "qualtrics" / "v1.5_new_processed.csv"
OUTPUT = ROOT / "qualtrics" / "qualtrics_results_v1.5.csv"
SUMMARY = ROOT / "qualtrics" / "v1.5_results_summary.json"
REMAPPED_IDS = {140200, 140206, 140307, 140308}
NEW_STRATEGY_COLUMNS = {
    "reasoning strategy",
    "best forward mental model",
    "forward mental model fit conclusion",
    "attribute selection method",
    "attribute selection ranking score",
    "change amount method",
    "change amount normalized MAE",
    "best-supported attribute selection method",
    "strategy: xai",
}


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    columns, old_rows = read(OLD)
    new_columns, new_rows = read(NEW)
    if columns != new_columns:
        raise RuntimeError("Column schema/order mismatch")
    old_participants = {row["participant"] for row in old_rows}
    new_participants = {row["participant"] for row in new_rows}
    if old_participants & new_participants:
        raise RuntimeError("Participant overlap between collection waves")

    # Strategy/mental-model classification is intentionally deferred for this
    # collection wave. Preserve prior coding, but leave all new rows blank.
    for row in new_rows:
        for column in NEW_STRATEGY_COLUMNS:
            row[column] = ""

    for row in old_rows:
        if row["instance id"] and int(float(row["instance id"])) in REMAPPED_IDS:
            raise RuntimeError("A remapped ID appears in the preserved old wave")
    remap_counts = Counter(
        int(float(row["instance id"])) for row in new_rows
        if row["phase"] == "testing" and int(float(row["instance id"])) in REMAPPED_IDS
    )
    if remap_counts != Counter({case_id: len(new_participants) for case_id in REMAPPED_IDS}):
        raise RuntimeError(f"Unexpected remapped-instance coverage: {remap_counts}")

    rows = old_rows + new_rows
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    participant_rows = Counter(row["participant"] for row in rows)
    summary = {
        "rows": len(rows),
        "columns": len(columns),
        "participants": len(participant_rows),
        "preserved_v1_4_participants": len(old_participants),
        "new_participants": len(new_participants),
        "participant_row_counts": dict(Counter(participant_rows.values())),
        "phase_rows": dict(Counter(row["phase"] for row in rows)),
        "xai_participants": {
            xai: len({row["participant"] for row in rows if row["xai"] == xai})
            for xai in sorted({row["xai"] for row in rows})
        },
        "new_instance_ids": sorted(REMAPPED_IDS),
        "new_instance_rows_each": len(new_participants),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
