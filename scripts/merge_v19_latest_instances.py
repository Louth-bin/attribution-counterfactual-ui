"""Combine every participant who saw the v1.6 latest-instance set."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    ROOT / "qualtrics" / "v1.8_new_processed.csv",
    ROOT / "qualtrics" / "v1.9_incremental_processed.csv",
    ROOT / "qualtrics" / "v1.9_new_processed.csv",
)
OUTPUT = ROOT / "qualtrics" / "qualtrics_results_v1.9.csv"
LEGACY_OUTPUT = ROOT / "qualtrics" / "v1.9_latest_processed.csv"
SUMMARY = ROOT / "qualtrics" / "v1.9_results_summary.json"
LEGACY_SUMMARY = ROOT / "qualtrics" / "v1.9_latest_processed_summary.json"
EXPECTED_TRAINING_IDS = set(range(160100, 160112))
EXPECTED_TESTING_IDS = set(range(160200, 160220))


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    columns: list[str] | None = None
    rows: list[dict[str, str]] = []
    participants: set[str] = set()
    input_summary: dict[str, dict[str, int]] = {}
    for path in INPUTS:
        new_columns, new_rows = read(path)
        if columns is None:
            columns = new_columns
        elif columns != new_columns:
            raise RuntimeError(f"Column schema/order mismatch in {path.name}")
        new_participants = {row["participant"] for row in new_rows}
        overlap = participants & new_participants
        if overlap:
            raise RuntimeError(f"Participant overlap in {path.name}: {sorted(overlap)}")
        participants.update(new_participants)
        rows.extend(new_rows)
        input_summary[path.name] = {
            "rows": len(new_rows),
            "participants": len(new_participants),
        }

    assert columns is not None
    if len(participants) != 36 or len(rows) != 1152:
        raise RuntimeError(
            f"Expected 36 participants and 1152 rows, found {len(participants)} and {len(rows)}"
        )
    participant_rows = Counter(row["participant"] for row in rows)
    if set(participant_rows.values()) != {32}:
        raise RuntimeError(f"Unexpected participant row counts: {Counter(participant_rows.values())}")
    training_ids = {
        int(float(row["instance id"])) for row in rows if row["phase"] == "training"
    }
    testing_ids = {
        int(float(row["instance id"])) for row in rows if row["phase"] == "testing"
    }
    if training_ids != EXPECTED_TRAINING_IDS or testing_ids != EXPECTED_TESTING_IDS:
        raise RuntimeError("Rows are not restricted to the v1.6 latest-instance IDs")

    mirrors_written = []
    for path in (OUTPUT, LEGACY_OUTPUT):
        try:
            with path.open("w", encoding="utf-8-sig", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            mirrors_written.append(path.name)
        except PermissionError:
            if path == OUTPUT:
                raise
            print(f"warning: could not update locked compatibility mirror {path}")

    summary = {
        "scope": "all participants who saw the v1.6 latest instances",
        "rows": len(rows),
        "columns": len(columns),
        "participants": len(participants),
        "input_waves": input_summary,
        "participant_row_counts": dict(Counter(participant_rows.values())),
        "phase_rows": dict(Counter(row["phase"] for row in rows)),
        "xai_participants": {
            xai: len({row["participant"] for row in rows if row["xai"] == xai})
            for xai in sorted({row["xai"] for row in rows})
        },
        "training_instance_ids": sorted(training_ids),
        "testing_instance_ids": sorted(testing_ids),
        "output_files_written": mirrors_written,
    }
    summary_text = json.dumps(summary, indent=2) + "\n"
    SUMMARY.write_text(summary_text, encoding="utf-8")
    try:
        LEGACY_SUMMARY.write_text(summary_text, encoding="utf-8")
    except PermissionError:
        print(f"warning: could not update locked compatibility summary {LEGACY_SUMMARY}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
