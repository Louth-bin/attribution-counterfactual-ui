"""Update the five existing v2.0 JMP CF-direction columns in place."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
JMP_PATH = ROOT / "qualtrics" / "qualtrics_results_v2.0.jmp"
ENRICHED_CSV = ROOT / "qualtrics" / "qualtrics_results_v2.0.with_cf_direction_columns.csv"
COLUMNS = [f"x_{i}_changed_in_training_cf_direction" for i in range(1, 6)]


def jsl_path(path: Path) -> str:
    return path.as_posix()


def values_to_jsl(series: pd.Series) -> str:
    parts = []
    for value in series.astype(str).tolist():
        stripped = value.strip()
        parts.append("." if stripped == "" else stripped)
    return ", ".join(parts)


def main() -> None:
    data = pd.read_csv(ENRICHED_CSV, dtype=str, keep_default_na=False)
    commands = []
    for column in COLUMNS:
        if column not in data.columns:
            raise ValueError(f"Missing column in enriched CSV: {column}")
        commands.append(
            f'Column( dt, "{column}" ) << Set Values( {{{values_to_jsl(data[column])}}} );'
        )

    script = f"""
Names Default To Here( 1 );
dt = Open( "{jsl_path(JMP_PATH)}", Invisible );
If( N Rows( dt ) != {len(data)},
	Throw( "Row count mismatch" )
);
{chr(10).join(commands)}
dt << Save( "{jsl_path(JMP_PATH)}" );
Close( dt, NoSave );
"""
    app = win32com.client.Dispatch("JMP.Application.19")
    app.RunCommand(script)
    print("updated_direction_columns=5")
    for column in COLUMNS:
        series = data[column]
        print(
            f"{column}: blank={(series == '').sum()}, "
            f"1={(series == '1').sum()}, 0={(series == '0').sum()}"
        )


if __name__ == "__main__":
    main()
