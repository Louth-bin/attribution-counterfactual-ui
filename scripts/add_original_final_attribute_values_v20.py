"""Add normalized original/final attribute values to qualtrics_results_v2.0.jmp.

Rules
-----
* Original normalized values are populated for every training and testing row.
* Training final values come only from displayed counterfactual explanations.
  Attribution and no-explanation training rows remain missing.
* Testing final values come from each participant's generated counterfactual.
* A final-value cell is missing unless that attribute actually changed.

The script validates every row before touching the JMP file, creates a backup,
adds or refreshes the ten columns, exports a round-trip CSV, and verifies the
saved result.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
JMP_PATH = ROOT / "qualtrics" / "qualtrics_results_v2.0.jmp"
BACKUP_PATH = (
    ROOT
    / "qualtrics"
    / "qualtrics_results_v2.0.before_normalized_original_final_values.jmp"
)
OUTPUT_DIR = ROOT / "outputs" / "v20-normalized-original-final-values"
BEFORE_CSV = OUTPUT_DIR / "qualtrics_results_v2.0.before.csv"
SOURCE_CSV = OUTPUT_DIR / "original_final_values_source.csv"
CHECK_CSV = OUTPUT_DIR / "qualtrics_results_v2.0.roundtrip.csv"
AUDIT_JSON = OUTPUT_DIR / "audit.json"

FEATURES = (
    (1, "Glucose"),
    (2, "Blood Pressure"),
    (3, "Insulin"),
    (4, "BMI"),
    (5, "Age"),
)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
LINE_PATTERN = re.compile(
    rf'^"(?P<feature>[^"]+)"\s*-\s*'
    rf"(?P<original_raw>{NUMBER})\s*\(\s*(?P<original_norm>{NUMBER})\s*\)"
    rf"(?:\s*->\s*(?P<final_raw>{NUMBER})\s*"
    rf"\(\s*(?P<final_norm>{NUMBER})\s*\))?\s*$",
    flags=re.MULTILINE,
)


def original_column(index: int, feature: str) -> str:
    return f"x_{index} {feature} original normalized value"


def final_column(index: int, feature: str) -> str:
    return f"x_{index} {feature} final changed normalized value"


def legacy_original_column(index: int, feature: str) -> str:
    return f"x_{index} {feature} original value"


def legacy_final_column(index: int, feature: str) -> str:
    return f"x_{index} {feature} final changed value"


def parse_attribute_lines(value: object) -> dict[str, tuple[float, float | None]]:
    text = "" if pd.isna(value) else str(value)
    parsed: dict[str, tuple[float, float | None]] = {}
    for match in LINE_PATTERN.finditer(text):
        final_text = match.group("final_norm")
        parsed[match.group("feature")] = (
            float(match.group("original_norm")),
            None if final_text is None else float(final_text),
        )
    return parsed


def dispatch_jmp():
    errors: list[str] = []
    for prog_id in ("JMP.Application.19", "JMP.Application"):
        try:
            return win32com.client.Dispatch(prog_id)
        except Exception as exc:  # pragma: no cover - depends on local JMP install
            errors.append(f"{prog_id}: {exc}")
    raise RuntimeError("Could not start JMP automation:\n" + "\n".join(errors))


def run_jsl(app, script: str) -> None:
    app.RunCommand(script)


def export_jmp(app, source: Path, destination: Path) -> None:
    run_jsl(
        app,
        f'''
Names Default To Here( 1 );
dt = Open( "{source.as_posix()}", Invisible );
dt << Save( "{destination.as_posix()}" );
Close( dt, NoSave );
''',
    )


def build_source(before: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {
        "phase",
        "xai",
        "attribute values before and after",
        "explanation",
    }
    missing = sorted(required.difference(before.columns))
    if missing:
        raise ValueError(f"Missing required JMP columns: {missing}")

    output = pd.DataFrame({"_row": range(1, len(before) + 1)})
    original_counts = {feature: 0 for _, feature in FEATURES}
    final_counts = {feature: 0 for _, feature in FEATURES}
    final_counts_by_source = {"training_counterfactual": 0, "testing_user": 0}

    original_values = {feature: [] for _, feature in FEATURES}
    final_values = {feature: [] for _, feature in FEATURES}

    for row_number, row in before.iterrows():
        phase = str(row["phase"]).strip().lower()
        xai = str(row["xai"]).strip().lower()
        if phase not in {"training", "testing"}:
            raise ValueError(f"Unexpected phase on row {row_number + 1}: {phase!r}")

        attributes = parse_attribute_lines(row["attribute values before and after"])
        absent = [feature for _, feature in FEATURES if feature not in attributes]
        if absent:
            raise ValueError(
                f"Could not parse original values on row {row_number + 1}: {absent}"
            )

        if phase == "training" and xai == "counterfactual":
            endpoint_source = parse_attribute_lines(row["explanation"])
            endpoint_kind = "training_counterfactual"
            absent = [feature for _, feature in FEATURES if feature not in endpoint_source]
            if absent:
                raise ValueError(
                    f"Could not parse counterfactual explanation on row "
                    f"{row_number + 1}: {absent}"
                )
        elif phase == "testing":
            endpoint_source = attributes
            endpoint_kind = "testing_user"
        else:
            endpoint_source = {}
            endpoint_kind = None

        for _, feature in FEATURES:
            original = attributes[feature][0]
            original_values[feature].append(original)
            original_counts[feature] += 1

            endpoint = endpoint_source.get(feature, (original, None))[1]
            if endpoint is not None and math.isclose(
                endpoint, original, rel_tol=0.0, abs_tol=1e-12
            ):
                endpoint = None
            final_values[feature].append(endpoint)
            if endpoint is not None:
                final_counts[feature] += 1
                final_counts_by_source[endpoint_kind] += 1

    for index, feature in FEATURES:
        output[original_column(index, feature)] = original_values[feature]
        output[final_column(index, feature)] = final_values[feature]

    audit = {
        "rows": len(before),
        "training_rows": int(before["phase"].eq("training").sum()),
        "testing_rows": int(before["phase"].eq("testing").sum()),
        "training_counterfactual_rows": int(
            (before["phase"].eq("training") & before["xai"].eq("counterfactual")).sum()
        ),
        "original_nonmissing_by_feature": original_counts,
        "final_nonmissing_by_feature": final_counts,
        "final_nonmissing_by_source": final_counts_by_source,
        "added_columns": [
            name
            for index, feature in FEATURES
            for name in (original_column(index, feature), final_column(index, feature))
        ],
    }
    return output, audit


def update_jmp(app, source: pd.DataFrame) -> None:
    column_names = [column for column in source.columns if column != "_row"]
    legacy_column_names = [
        name
        for index, feature in FEATURES
        for name in (
            legacy_original_column(index, feature),
            legacy_final_column(index, feature),
        )
    ]
    jsl_column_list = ",\n\t".join(f'"{name}"' for name in column_names)
    jsl_legacy_column_list = ",\n\t".join(
        f'"{name}"' for name in legacy_column_names
    )
    run_jsl(
        app,
        f'''
Names Default To Here( 1 );
jmpPath = "{JMP_PATH.as_posix()}";
sourcePath = "{SOURCE_CSV.as_posix()}";
checkPath = "{CHECK_CSV.as_posix()}";
columns = {{
\t{jsl_column_list}
}};
legacyColumns = {{
\t{jsl_legacy_column_list}
}};

dt = Open( jmpPath, Invisible );
source = Open( sourcePath, Invisible );

If( N Rows( dt ) != N Rows( source ),
\tClose( source, NoSave );
\tClose( dt, NoSave );
\tThrow( "Row count mismatch" );
);

For( r = 1, r <= N Rows( source ), r++,
\tIf( Column( source, "_row" )[r] != r,
\t\tClose( source, NoSave );
\t\tClose( dt, NoSave );
\t\tThrow( "Source row-key mismatch" );
\t);
);

For( i = 1, i <= N Items( columns ), i++,
\tcolumnName = columns[i];
\tvalues = Column( source, columnName ) << Get Values;
\tIf( Contains( dt << Get Column Names( "String" ), columnName ) == 0,
\t\tdt << New Column(
\t\t\tcolumnName,
\t\t\tNumeric,
\t\t\tContinuous,
\t\t\tFormat( "Best", 12 ),
\t\t\tSet Values( values )
\t\t),
\t\tColumn( dt, columnName ) << Set Values( values )
\t);
);

For( i = 1, i <= N Items( legacyColumns ), i++,
\tcolumnName = legacyColumns[i];
\tIf( Contains( dt << Get Column Names( "String" ), columnName ),
\t\tdt << Delete Columns( columnName )
\t);
);

Close( source, NoSave );
dt << Save( jmpPath );
dt << Save As( checkPath );
Close( dt, NoSave );
''',
    )


def verify_roundtrip(expected: pd.DataFrame, audit: dict) -> None:
    actual = pd.read_csv(CHECK_CSV)
    if len(actual) != audit["rows"]:
        raise AssertionError(f"Round-trip row count changed: {len(actual)}")

    for column in audit["added_columns"]:
        if column not in actual.columns:
            raise AssertionError(f"Missing round-trip column: {column}")
        expected_values = pd.to_numeric(expected[column], errors="coerce").to_numpy(
            dtype=float
        )
        actual_values = pd.to_numeric(actual[column], errors="coerce").to_numpy(
            dtype=float
        )
        nan_mismatch = np.isnan(expected_values) != np.isnan(actual_values)
        numeric_mismatch = ~np.isclose(
            expected_values,
            actual_values,
            rtol=0.0,
            atol=1e-10,
            equal_nan=True,
        )
        difference = nan_mismatch | numeric_mismatch
        if difference.any():
            raise AssertionError(
                f"Round-trip mismatch in {column}: {int(difference.sum())} rows"
            )

    originals = [original_column(index, feature) for index, feature in FEATURES]
    finals = [final_column(index, feature) for index, feature in FEATURES]
    legacy = [
        name
        for index, feature in FEATURES
        for name in (
            legacy_original_column(index, feature),
            legacy_final_column(index, feature),
        )
    ]
    unexpected_legacy = [column for column in legacy if column in actual.columns]
    if unexpected_legacy:
        raise AssertionError(f"Legacy raw-value columns remain: {unexpected_legacy}")
    training_non_cf = actual["phase"].eq("training") & ~actual["xai"].eq(
        "counterfactual"
    )
    if actual.loc[training_non_cf, finals].notna().any().any():
        raise AssertionError("A non-counterfactual training row has a final value")
    if actual.loc[actual["phase"].eq("testing"), originals].isna().any().any():
        raise AssertionError("A testing row is missing an original value")
    if actual.loc[actual["phase"].eq("training"), originals].isna().any().any():
        raise AssertionError("A training row is missing an original value")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = dispatch_jmp()
    export_jmp(app, JMP_PATH, BEFORE_CSV)
    before = pd.read_csv(BEFORE_CSV)
    source, audit = build_source(before)
    source.to_csv(SOURCE_CSV, index=False)

    if not BACKUP_PATH.exists():
        shutil.copy2(JMP_PATH, BACKUP_PATH)

    update_jmp(app, source)
    verify_roundtrip(source, audit)

    audit["jmp_columns_after"] = len(pd.read_csv(CHECK_CSV, nrows=0).columns)
    AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
