"""Reconstruct the cumulative v1.7 results and append five new responses."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    ROOT / "qualtrics" / "qualtrics_results_v1.5.csv",
    ROOT / "qualtrics" / "v1.6_new_processed.csv",
    ROOT / "qualtrics" / "v1.7_new_processed.csv",
)
OUTPUT = ROOT / "qualtrics" / "qualtrics_results_v1.7.csv"
SUMMARY = ROOT / "qualtrics" / "v1.7_results_summary.json"
V15_REMAPPED_IDS = {140200, 140206, 140307, 140308}
V16_TEST_IDS = set(range(160200, 160220))
STRATEGY_COLUMNS = {
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
    columns: list[str] | None = None
    rows: list[dict[str, str]] = []
    wave_counts: dict[str, dict[str, int]] = {}
    participants_seen: set[str] = set()

    for index, path in enumerate(INPUTS):
        wave_columns, wave_rows = read(path)
        if columns is None:
            columns = wave_columns
        elif columns != wave_columns:
            raise RuntimeError(f"Column schema/order mismatch in {path.name}")
        wave_participants = {row["participant"] for row in wave_rows}
        overlap = participants_seen & wave_participants
        if overlap:
            raise RuntimeError(f"Participant overlap in {path.name}: {sorted(overlap)}")
        if index > 0:
            for row in wave_rows:
                for column in STRATEGY_COLUMNS:
                    row[column] = ""
        rows.extend(wave_rows)
        participants_seen.update(wave_participants)
        wave_counts[path.name] = {
            "rows": len(wave_rows),
            "participants": len(wave_participants),
        }

    assert columns is not None
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    participant_rows = Counter(row["participant"] for row in rows)
    remap_counts = Counter(
        int(float(row["instance id"]))
        for row in rows
        if row["phase"] == "testing"
        and row["instance id"]
        and int(float(row["instance id"])) in V15_REMAPPED_IDS | V16_TEST_IDS
    )
    summary = {
        "rows": len(rows),
        "columns": len(columns),
        "participants": len(participant_rows),
        "input_waves": wave_counts,
        "participant_row_counts": dict(Counter(participant_rows.values())),
        "phase_rows": dict(Counter(row["phase"] for row in rows)),
        "xai_participants": {
            xai: len({row["participant"] for row in rows if row["xai"] == xai})
            for xai in sorted({row["xai"] for row in rows})
        },
        "remapped_instance_rows": {
            str(case_id): remap_counts[case_id]
            for case_id in sorted(V15_REMAPPED_IDS | V16_TEST_IDS)
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
