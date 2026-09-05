"""Add the randomized first training block to the v2.3 CSV and JMP table.

The converted analysis table uses canonical instance IDs and trial numbers, so
its training rows no longer encode the randomized block order.  The original
Qualtrics ``training_log_json`` retains that order.  This script reconstructs
the first block for each participant and copies the participant-level value to
every corresponding training and testing row.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "qualtrics" / "raw_output_v2.3.csv"
DEFAULT_CSV = ROOT / "qualtrics" / "qualtrics_results_v2.3.csv"
DEFAULT_JMP = ROOT / "qualtrics" / "qualtrics_results_v2.3.jmp"
COLUMN = "first training block"

BLOCKS = {
    "Glucose + Age": set(range(130100, 130106)),
    "Glucose + Insulin": set(range(130106, 130112)),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--jmp", type=Path, default=DEFAULT_JMP)
    parser.add_argument("--csv-only", action="store_true")
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_first_blocks(raw_path: Path) -> dict[str, str]:
    first_blocks: dict[str, str] = {}
    with raw_path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            participant = str(row.get("ResponseId", "")).strip()
            try:
                logs = json.loads(row.get("training_log_json", ""))
            except (TypeError, json.JSONDecodeError):
                continue
            if not participant or not isinstance(logs, list) or len(logs) != 12:
                continue

            ids = [int(log["instanceId"]) for log in logs]
            first_ids, second_ids = set(ids[:6]), set(ids[6:])
            matches = [name for name, expected in BLOCKS.items() if first_ids == expected]
            if len(matches) != 1:
                raise ValueError(
                    f"Could not identify first training block for {participant}: {ids[:6]}"
                )
            first_block = matches[0]
            other_block = next(name for name in BLOCKS if name != first_block)
            if second_ids != BLOCKS[other_block]:
                raise ValueError(
                    f"Second training block is invalid for {participant}: {ids[6:]}"
                )
            first_blocks[participant] = first_block

    if not first_blocks:
        raise ValueError(f"No complete training logs found in {raw_path}")
    return first_blocks


def add_csv_column(csv_path: Path, first_blocks: dict[str, str]) -> tuple[int, int]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "participant" not in fields:
        raise ValueError(f"Missing participant column in {csv_path}")

    participants = {row["participant"] for row in rows}
    missing = participants - set(first_blocks)
    extra = set(first_blocks) - participants
    if missing or extra:
        raise ValueError(
            f"Raw/results participant mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    row_counts = Counter(row["participant"] for row in rows)
    unexpected = {participant: count for participant, count in row_counts.items() if count != 32}
    if unexpected:
        raise ValueError(f"Expected 32 rows per participant: {unexpected}")

    if COLUMN not in fields:
        insertion = fields.index("xai") + 1 if "xai" in fields else len(fields)
        fields.insert(insertion, COLUMN)
    for row in rows:
        row[COLUMN] = first_blocks[row["participant"]]

    backup = csv_path.with_name(f"{csv_path.stem}.before_first_training_block{csv_path.suffix}")
    if not backup.exists():
        shutil.copy2(csv_path, backup)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    return len(rows), len(participants)


def jsl_path(path: Path) -> str:
    return path.as_posix().replace('"', '\\"')


def export_jmp(app: object, jmp_path: Path, csv_path: Path) -> None:
    app.RunCommand(
        f'''
Names Default To Here( 1 );
dt = Open( "{jsl_path(jmp_path)}", Invisible );
dt << Save( "{jsl_path(csv_path)}" );
Close( dt, NoSave );
'''
    )


def read_identities(path: Path) -> list[tuple[str, str, str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return [
            (
                row["participant"],
                row["phase"],
                row["trial number"],
                row["instance id"],
            )
            for row in csv.DictReader(source)
        ]


def update_jmp(jmp_path: Path, csv_path: Path) -> None:
    import win32com.client

    app = win32com.client.Dispatch("JMP.Application.19")
    with tempfile.TemporaryDirectory(prefix="v23_first_block_") as directory:
        temporary = Path(directory)
        before = temporary / "before.csv"
        after = temporary / "after.csv"
        export_jmp(app, jmp_path, before)
        if read_identities(before) != read_identities(csv_path):
            raise ValueError("CSV and JMP row identities are not aligned; JMP was not changed")

        backup = jmp_path.with_name(
            f"{jmp_path.stem}.before_first_training_block{jmp_path.suffix}"
        )
        if not backup.exists():
            shutil.copy2(jmp_path, backup)
        app.RunCommand(
            f'''
Names Default To Here( 1 );
dt = Open( "{jsl_path(jmp_path)}", Invisible );
source = Open( "{jsl_path(csv_path)}", Invisible );
If( N Rows( dt ) != N Rows( source ), Throw( "Row-count mismatch" ) );
values = Column( source, "{COLUMN}" ) << Get Values;
columnNames = dt << Get Column Names( "String" );
If(
    Contains( columnNames, "{COLUMN}" ),
    Column( dt, "{COLUMN}" ) << Set Values( values ),
    dt << New Column(
        "{COLUMN}",
        Character,
        Nominal,
        Set Values( values )
    )
);
dt << Save;
dt << Save As( "{jsl_path(after)}" );
Close( source, NoSave );
Close( dt, NoSave );
'''
        )

        with after.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        expected = {}
        with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
            expected_rows = list(csv.DictReader(source))
        if [row.get(COLUMN) for row in rows] != [row[COLUMN] for row in expected_rows]:
            raise AssertionError("JMP round-trip values do not match the v2.3 CSV")


def main() -> None:
    args = arguments()
    raw_path, csv_path, jmp_path = map(resolved, (args.raw, args.csv, args.jmp))
    first_blocks = read_first_blocks(raw_path)
    rows, participants = add_csv_column(csv_path, first_blocks)
    if not args.csv_only:
        update_jmp(jmp_path, csv_path)
    print(
        json.dumps(
            {
                "column": COLUMN,
                "rows": rows,
                "participants": participants,
                "counts": dict(sorted(Counter(first_blocks.values()).items())),
                "csv": str(csv_path),
                "jmp": None if args.csv_only else str(jmp_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
