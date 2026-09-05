"""Restore pre-edit v2.0 JMP file, then add only five CF-direction columns.

This avoids rebuilding the JMP table from CSV, so existing JMP columns/table
metadata from the backup are preserved. The five new columns are populated from
the already-enriched CSV in the same row order.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
BACKUP_JMP = ROOT / "qualtrics" / "qualtrics_results_v2.0.before_cf_direction_columns.jmp"
TARGET_JMP = ROOT / "qualtrics" / "qualtrics_results_v2.0.jmp"
ENRICHED_CSV = ROOT / "qualtrics" / "qualtrics_results_v2.0.with_cf_direction_columns.csv"

NEW_COLUMNS = [f"x_{i}_changed_in_training_cf_direction" for i in range(1, 6)]


def jsl_path(path: Path) -> str:
    return path.as_posix()


def to_jsl_value(value: str) -> str:
    value = str(value).strip()
    if value == "":
        return "."
    if value in {"0", "0.0"}:
        return "0"
    if value in {"1", "1.0"}:
        return "1"
    raise ValueError(f"Unexpected direction value: {value!r}")


def main() -> None:
    if not BACKUP_JMP.exists():
        raise FileNotFoundError(BACKUP_JMP)
    if not ENRICHED_CSV.exists():
        raise FileNotFoundError(ENRICHED_CSV)

    enriched = pd.read_csv(ENRICHED_CSV, dtype=str, keep_default_na=False)
    missing_columns = [column for column in NEW_COLUMNS if column not in enriched.columns]
    if missing_columns:
        raise ValueError(f"Missing enriched columns: {missing_columns}")

    # Restore the original JMP file first.
    shutil.copy2(BACKUP_JMP, TARGET_JMP)

    add_column_commands = []
    for column in NEW_COLUMNS:
        values = ", ".join(to_jsl_value(value) for value in enriched[column].tolist())
        add_column_commands.append(
            f'''
If( Contains( dt << Get Column Names( "String" ), "{column}" ),
	dt << Delete Columns( "{column}" )
);
dt << New Column(
	"{column}",
	Numeric,
	Nominal,
	Set Values( {{{values}}} )
);
'''
        )

    script = f'''
Names Default To Here( 1 );
dt = Open( "{jsl_path(TARGET_JMP)}", Invisible );
If( N Rows( dt ) != {len(enriched)},
	Throw( "Row count mismatch: JMP table has " || Char( N Rows( dt ) ) || " rows, expected {len(enriched)}" )
);
{chr(10).join(add_column_commands)}
dt << Save( "{jsl_path(TARGET_JMP)}" );
Close( dt, NoSave );
'''

    app = win32com.client.Dispatch("JMP.Application.19")
    app.RunCommand(script)

    print(f"Restored from {BACKUP_JMP.relative_to(ROOT)}")
    print(f"Added columns to {TARGET_JMP.relative_to(ROOT)}:")
    for column in NEW_COLUMNS:
        series = enriched[column]
        print(
            f"  {column}: blank={(series == '').sum()}, "
            f"1={(series == '1').sum()}, 0={(series == '0').sum()}"
        )


if __name__ == "__main__":
    main()
