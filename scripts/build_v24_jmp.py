"""Create and round-trip validate the JMP table for qualtrics_results_v2.4.csv."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import win32com.client


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "qualtrics" / "qualtrics_results_v2.4.csv"
JMP_PATH = ROOT / "qualtrics" / "qualtrics_results_v2.4.jmp"


def jsl_path(path: Path) -> str:
    return path.resolve().as_posix().replace('"', '\\"')


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    expected_columns, expected_rows = read_csv(CSV_PATH)
    app = None
    errors = []
    for program_id in (
        "JMP.Application.19",
        "JMP.Application.18",
        "JMP.Application.17",
        "JMP.Application",
    ):
        try:
            app = win32com.client.Dispatch(program_id)
            break
        except Exception as error:  # COM registration differs by JMP release.
            errors.append(f"{program_id}: {error}")
    if app is None:
        raise RuntimeError("No registered JMP automation server. " + " | ".join(errors))
    app.Visible = False
    with tempfile.TemporaryDirectory(prefix="qualtrics_v24_jmp_") as directory:
        roundtrip = Path(directory) / "roundtrip.csv"
        app.RunCommand(
            f'''
Names Default To Here( 1 );
dt = Open( "{jsl_path(CSV_PATH)}", Invisible );
dt << Save As( "{jsl_path(JMP_PATH)}" );
dt << Save As( "{jsl_path(roundtrip)}" );
Close( dt, NoSave );
'''
        )
        actual_columns, actual_rows = read_csv(roundtrip)

    if actual_columns != expected_columns:
        raise AssertionError("JMP round-trip column names or order differ from the CSV")
    if len(actual_rows) != len(expected_rows):
        raise AssertionError("JMP round-trip row count differs from the CSV")
    identity = ["participant", "phase", "trial number", "instance id", "attribute changed"]
    for column in identity:
        expected = [row[column] for row in expected_rows]
        actual = [row[column] for row in actual_rows]
        if expected != actual:
            raise AssertionError(f"JMP round-trip mismatch in {column}")
    for expected, actual in zip(expected_rows, actual_rows):
        left = expected["amount changed"]
        right = actual["amount changed"]
        if not left and not right:
            continue
        if abs(float(left) - float(right)) > 1e-8:
            raise AssertionError("JMP round-trip mismatch in amount changed")

    print(
        f"JMP validated: rows={len(actual_rows)} columns={len(actual_columns)} "
        f"path={JMP_PATH}"
    )


if __name__ == "__main__":
    main()
