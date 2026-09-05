"""Round-trip the updated v2.0 JMP file to CSV and validate new columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
JMP_PATH = ROOT / "qualtrics" / "qualtrics_results_v2.0.jmp"
CHECK_CSV = ROOT / "qualtrics" / "qualtrics_results_v2.0.jmp_roundtrip_check.csv"


def as_jsl_path(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    script = f"""
Names Default To Here( 1 );
dt = Open( "{as_jsl_path(JMP_PATH)}", Invisible );
dt << Save( "{as_jsl_path(CHECK_CSV)}" );
Close( dt, NoSave );
"""
    app = win32com.client.Dispatch("JMP.Application.19")
    app.RunCommand(script)

    data = pd.read_csv(CHECK_CSV, dtype=str, keep_default_na=False)
    columns = [f"x_{i}_changed_in_training_cf_direction" for i in range(1, 6)]
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise AssertionError(f"Missing columns after JMP round-trip: {missing}")

    print(f"roundtrip rows={len(data)}, columns={len(data.columns)}")
    for column in columns:
        series = data[column].astype(str)
        print(
            f"{column}: blank={(series == '').sum()}, "
            f"1={(series == '1').sum()}, 0={(series == '0').sum()}"
        )


if __name__ == "__main__":
    main()
