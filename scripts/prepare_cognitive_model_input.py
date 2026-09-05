"""Prepare compact Qualtrics results for the cognitive-model fitters.

The v2.4+ exports store the single testing edit as an attribute name and an
amount.  The fitters expect five feature-change columns.  This converter also
supports fitting only participants absent from an earlier fit file, which is
how incremental recruitment cohorts are defined.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FEATURE_COLUMNS = {
    "glucose": "x_1_change",
    "blood pressure": "x_2_change",
    "insulin": "x_3_change",
    "bmi": "x_4_change",
    "age": "x_5_change",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-fit-participants",
        type=Path,
        help="Keep only participants not present in this earlier fit CSV.",
    )
    args = parser.parse_args()

    rows = read_rows(args.responses)
    excluded: set[str] = set()
    if args.exclude_fit_participants:
        excluded = {
            row["participant"]
            for row in read_rows(args.exclude_fit_participants)
            if row.get("participant")
        }
    rows = [row for row in rows if row.get("participant") not in excluded]
    if not rows:
        raise SystemExit("No response rows remain after participant filtering")

    for row in rows:
        row["case"] = row.get("trial number", "")
        for column in FEATURE_COLUMNS.values():
            row[column] = "0"
        if row.get("phase", "").strip().casefold() != "testing":
            continue
        feature = row.get("attribute changed", "").strip().casefold()
        if not feature:
            continue
        try:
            column = FEATURE_COLUMNS[feature]
        except KeyError as error:
            raise ValueError(f"Unknown changed attribute: {feature!r}") from error
        row[column] = row.get("amount changed", "0") or "0"

    fields = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    participants = {row["participant"] for row in rows}
    print(f"wrote {len(rows)} rows for {len(participants)} participants")


if __name__ == "__main__":
    main()
