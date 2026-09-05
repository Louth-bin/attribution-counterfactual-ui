"""Add displayed training-CF changes to the v2.3 CSV and JMP table."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "qualtrics" / "qualtrics_results_v2.3.csv"
DEFAULT_JMP = ROOT / "qualtrics" / "qualtrics_results_v2.3.jmp"
DEFAULT_BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.5_discrete_age.json"
COLUMNS = [f"x_{index}_train_change" for index in range(1, 6)]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--jmp", type=Path, default=DEFAULT_JMP)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--csv-only", action="store_true")
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def scalar(value: float) -> str:
    return f"{value:.12g}"


def training_changes(bundle_path: Path) -> dict[int, list[float]]:
    experiment = json.loads(bundle_path.read_text(encoding="utf-8"))
    cases = experiment["datasets"]["diabetes"]["training_pool"]
    changes: dict[int, list[float]] = {}
    for case in cases:
        # Some historical bundles expose the endpoint only as feature_values.
        after = case["counterfactual"].get("raw_feature_values")
        if after is None:
            after = case["counterfactual"]["feature_values"]
        source = case["raw_feature_values"]
        ranges = case["raw_feature_ranges"]
        vector = [
            (float(destination) - float(origin)) /
            (float(bounds[1]) - float(bounds[0]))
            for origin, destination, bounds in zip(source, after, ranges)
        ]
        changes[int(case["instance_id"])] = vector
    if len(changes) != 12:
        raise ValueError(f"Expected 12 training cases, found {len(changes)}")
    return changes


def update_csv(csv_path: Path, changes: dict[int, list[float]]) -> tuple[int, int]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not {"phase", "xai", "instance id"}.issubset(fields):
        raise ValueError("The v2.3 CSV is missing required identity columns")
    if not all(column in fields for column in COLUMNS):
        insertion = fields.index("x_1_change")
        for column in reversed(COLUMNS):
            if column not in fields:
                fields.insert(insertion, column)

    populated = 0
    for row in rows:
        is_displayed_cf = row["phase"] == "training" and row["xai"] == "counterfactual"
        if is_displayed_cf:
            instance_id = int(float(row["instance id"]))
            if instance_id not in changes:
                raise ValueError(f"Unknown v2.3 training instance {instance_id}")
            for column, value in zip(COLUMNS, changes[instance_id]):
                row[column] = scalar(value)
            populated += 1
        else:
            for column in COLUMNS:
                row[column] = ""

    expected = sum(
        row["phase"] == "training" and row["xai"] == "counterfactual"
        for row in rows
    )
    if populated != expected:
        raise AssertionError(f"Populated {populated} rows; expected {expected}")

    backup = csv_path.with_name(
        f"{csv_path.stem}.before_training_change_columns{csv_path.suffix}"
    )
    if not backup.exists():
        shutil.copy2(csv_path, backup)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    return len(rows), populated


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


def identities(path: Path) -> list[tuple[str, str, str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return [
            (row["participant"], row["phase"], row["trial number"], row["instance id"])
            for row in csv.DictReader(source)
        ]


def update_jmp(jmp_path: Path, csv_path: Path) -> None:
    import win32com.client

    app = win32com.client.Dispatch("JMP.Application.19")
    with tempfile.TemporaryDirectory(prefix="v23_training_changes_") as directory:
        temporary = Path(directory)
        before = temporary / "before.csv"
        after = temporary / "after.csv"
        export_jmp(app, jmp_path, before)
        if identities(before) != identities(csv_path):
            raise ValueError("CSV and JMP row identities are not aligned; JMP was not changed")

        backup = jmp_path.with_name(
            f"{jmp_path.stem}.before_training_change_columns{jmp_path.suffix}"
        )
        if not backup.exists():
            shutil.copy2(jmp_path, backup)
        statements = []
        for column in COLUMNS:
            statements.append(
                f'''
values = Column( source, "{column}" ) << Get Values;
If(
    Contains( columnNames, "{column}" ),
    Column( dt, "{column}" ) << Set Values( values ),
    dt << New Column(
        "{column}", Numeric, Continuous, Format( "Best", 12 ), Set Values( values )
    )
);'''
            )
        app.RunCommand(
            f'''
Names Default To Here( 1 );
dt = Open( "{jsl_path(jmp_path)}", Invisible );
source = Open( "{jsl_path(csv_path)}", Invisible );
If( N Rows( dt ) != N Rows( source ), Throw( "Row-count mismatch" ) );
columnNames = dt << Get Column Names( "String" );
{''.join(statements)}
dt << Save;
dt << Save As( "{jsl_path(after)}" );
Close( source, NoSave );
Close( dt, NoSave );
'''
        )

        with after.open("r", encoding="utf-8-sig", newline="") as source:
            actual = list(csv.DictReader(source))
        with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
            expected = list(csv.DictReader(source))
        for column in COLUMNS:
            actual_values = [row.get(column, "") for row in actual]
            expected_values = [row[column] for row in expected]
            if actual_values != expected_values:
                # JMP may render numeric strings differently; compare numerically.
                for actual_value, expected_value in zip(actual_values, expected_values):
                    if not actual_value and not expected_value:
                        continue
                    # JMP's Best 12 CSV rendering rounds stored numeric values.
                    if abs(float(actual_value) - float(expected_value)) > 1e-8:
                        raise AssertionError(f"JMP mismatch in {column}")


def main() -> None:
    args = arguments()
    csv_path, jmp_path, bundle_path = map(
        resolved, (args.csv, args.jmp, args.bundle)
    )
    changes = training_changes(bundle_path)
    rows, populated = update_csv(csv_path, changes)
    if not args.csv_only:
        update_jmp(jmp_path, csv_path)
    nonzero = {
        column: sum(abs(values[index]) > 1e-12 for values in changes.values())
        for index, column in enumerate(COLUMNS)
    }
    print(
        json.dumps(
            {
                "columns": COLUMNS,
                "rows": rows,
                "counterfactual_training_rows_populated": populated,
                "nonzero_training_instances": nonzero,
                "csv": str(csv_path),
                "jmp": None if args.csv_only else str(jmp_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
