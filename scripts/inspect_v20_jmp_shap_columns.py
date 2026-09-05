"""Round-trip v2.0 JMP and report existing SHAP-change columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
JMP_PATH = ROOT / "qualtrics" / "qualtrics_results_v2.0.jmp"
CHECK_CSV = ROOT / "qualtrics" / "qualtrics_results_v2.0.pre_shap_roundtrip_check.csv"


def main() -> None:
    script = f"""
Names Default To Here( 1 );
dt = Open( "{JMP_PATH.as_posix()}", Invisible );
dt << Save( "{CHECK_CSV.as_posix()}" );
Close( dt, NoSave );
"""
    app = win32com.client.Dispatch("JMP.Application.19")
    app.RunCommand(script)
    data = pd.read_csv(CHECK_CSV, nrows=1)
    shap_columns = [column for column in data.columns if "SHAP change" in column or "Δφ" in column]
    print(f"roundtrip rows unknown, columns={len(data.columns)}")
    print("Existing SHAP-change columns:")
    for column in shap_columns:
        print(column)


if __name__ == "__main__":
    main()
