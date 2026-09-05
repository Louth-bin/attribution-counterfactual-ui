"""Finalize v2.2 by retaining existing fits and adding newly converted participants."""

from __future__ import annotations

import csv
import json
import argparse
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD = ROOT / "qualtrics" / "qualtrics_results_v2.1.csv"
DEFAULT_CONVERTED = ROOT / "qualtrics" / "qualtrics_results_v2.2.converted.csv"
DEFAULT_OUTPUT = ROOT / "qualtrics" / "qualtrics_results_v2.2.csv"

FIT_COLUMNS = (
    "cognitive model v0.1 family",
    "explanation reliance (η)",
    "global relevance reliance (α)",
    "additive change margin (ρ)",
    "exemplar locality (λ)",
    "remembered-change reliance (β)",
    "age treated as actionable (0/1)",
)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--converted", type=Path, default=DEFAULT_CONVERTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    old = args.base if args.base.is_absolute() else ROOT / args.base
    converted = args.converted if args.converted.is_absolute() else ROOT / args.converted
    output = args.output if args.output.is_absolute() else ROOT / args.output
    audit_path = output.with_suffix(".audit.json")

    old_fields, old_rows = read_rows(old)
    new_fields, new_rows = read_rows(converted)
    if old_fields != new_fields:
        raise ValueError("v2.1 and converted v2.2 schemas do not match exactly")

    old_participants = {row["participant"] for row in old_rows}
    fit_by_participant: dict[str, dict[str, str]] = {}
    for row in old_rows:
        participant = row["participant"]
        fit_by_participant.setdefault(
            participant, {column: row[column] for column in FIT_COLUMNS}
        )

    for row in new_rows:
        fit = fit_by_participant.get(row["participant"])
        if fit:
            row.update(fit)

    with output.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(new_rows)

    participants = {row["participant"] for row in new_rows}
    new_participants = participants - old_participants
    counts = Counter(row["participant"] for row in new_rows)
    audit = {
        "output": str(output.relative_to(ROOT)),
        "base": str(old.relative_to(ROOT)),
        "converted_raw": str(converted.relative_to(ROOT)),
        "rows": len(new_rows),
        "columns": len(new_fields),
        "participants": len(participants),
        "existing_participants_retained": len(participants & old_participants),
        "new_participants_appended": len(new_participants),
        "new_participant_ids": sorted(new_participants),
        "new_rows_appended": sum(counts[p] for p in new_participants),
        "participant_row_counts": dict(sorted(Counter(counts.values()).items())),
        "phase_rows": dict(Counter(row["phase"] for row in new_rows)),
        "xai_participants": {
            xai: len({row["participant"] for row in new_rows if row["xai"] == xai})
            for xai in sorted({row["xai"] for row in new_rows})
        },
        "cognitive_fits_preserved_for_existing_participants": len(
            fit_by_participant
        ),
        "cognitive_fits_computed_for_new_participants": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
