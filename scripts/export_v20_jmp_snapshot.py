"""Export the current v2.0 JMP table to a CSV snapshot for reproducible analysis.

The source JMP file is opened invisibly and is never saved or modified.
"""

from pathlib import Path

import win32com.client


ROOT = Path(__file__).resolve().parents[1]
JMP_PATH = ROOT / "qualtrics" / "qualtrics_results_v2.0.jmp"
OUTPUT_DIR = ROOT / "outputs" / "v20-effect-search"
CSV_PATH = OUTPUT_DIR / "qualtrics_results_v2.0_from_jmp.csv"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    script = f'''
Names Default To Here( 1 );
dt = Open( "{JMP_PATH.as_posix()}", Invisible );
dt << Save( "{CSV_PATH.as_posix()}" );
Close( dt, NoSave );
'''
    app = win32com.client.Dispatch("JMP.Application.19")
    app.RunCommand(script)
    print(CSV_PATH)


if __name__ == "__main__":
    main()
