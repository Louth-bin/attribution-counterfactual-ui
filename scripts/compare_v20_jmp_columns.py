"""Compare columns between backed-up and current v2.0 JMP files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
OLD_JMP = ROOT / "qualtrics" / "qualtrics_results_v2.0.before_cf_direction_columns.jmp"
NEW_JMP = ROOT / "qualtrics" / "qualtrics_results_v2.0.jmp"
OLD_CSV = ROOT / "qualtrics" / "qualtrics_results_v2.0.before_cf_direction_columns.roundtrip.csv"
NEW_CSV = ROOT / "qualtrics" / "qualtrics_results_v2.0.current.roundtrip.csv"


def jsl_path(path: Path) -> str:
    return path.as_posix()


def export_jmp_to_csv(jmp_path: Path, csv_path: Path) -> None:
    script = f"""
Names Default To Here( 1 );
dt = Open( "{jsl_path(jmp_path)}", Invisible );
dt << Save( "{jsl_path(csv_path)}" );
Close( dt, NoSave );
"""
    app = win32com.client.Dispatch("JMP.Application.19")
    app.RunCommand(script)


def main() -> None:
    export_jmp_to_csv(OLD_JMP, OLD_CSV)
    export_jmp_to_csv(NEW_JMP, NEW_CSV)

    old = pd.read_csv(OLD_CSV, nrows=1)
    new = pd.read_csv(NEW_CSV, nrows=1)

    old_cols = list(old.columns)
    new_cols = list(new.columns)
    old_set = set(old_cols)
    new_set = set(new_cols)

    added = [column for column in new_cols if column not in old_set]
    removed = [column for column in old_cols if column not in new_set]
    order_changed_existing = [
        column
        for column in old_cols
        if column in new_set and old_cols.index(column) != new_cols.index(column)
    ]

    print(f"old_columns={len(old_cols)}")
    print(f"new_columns={len(new_cols)}")
    print(f"added={added}")
    print(f"removed={removed}")
    print(f"existing_columns_order_changed_count={len(order_changed_existing)}")
    if order_changed_existing:
        print(f"first_order_changed={order_changed_existing[:10]}")
    print(f"old_last_10={[column.encode('ascii', 'backslashreplace').decode('ascii') for column in old_cols[-10:]]}")
    print(f"new_last_10={[column.encode('ascii', 'backslashreplace').decode('ascii') for column in new_cols[-10:]]}")


if __name__ == "__main__":
    main()
