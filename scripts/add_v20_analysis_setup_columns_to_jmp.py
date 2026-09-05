"""Add reusable analysis setup columns to v2.0 JMP without rebuilding it.

The script computes row-aligned helper columns from the v2.0 Qualtrics CSV and
adds only missing columns directly to the existing JMP table. It does not
replace or re-import the JMP file, preserving existing JMP table structure.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
JMP_PATH = ROOT / "qualtrics" / "qualtrics_results_v2.0.jmp"
ROUNDTRIP_CSV = ROOT / "qualtrics" / "qualtrics_results_v2.0.analysis_setup_roundtrip.csv"

FEATURES = ["Glucose", "Blood Pressure", "Insulin", "BMI", "Age"]
X_FEATURES = ["x_1", "x_2", "x_3", "x_4", "x_5"]
VALUE_PATTERN = re.compile(
    r'"(?P<feature>[^"]+)"\s*-\s*'
    r'[^(\n]*\((?P<before>[-+0-9.eE]+)\)'
    r'(?:\s*->\s*[^(\n]*\((?P<after>[-+0-9.eE]+)\))?'
)


def jsl_path(path: Path) -> str:
    return path.as_posix()


def parse_profile(text: str, use_after: bool) -> np.ndarray:
    values: list[float] = []
    for match in VALUE_PATTERN.finditer(str(text)):
        before = float(match.group("before"))
        after = match.group("after")
        values.append(float(after) if use_after and after is not None else before)
    if len(values) != 5:
        raise ValueError(f"Expected five feature values, found {len(values)} in {text!r}")
    return np.asarray(values, dtype=float)


def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def clean_float(value: object) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if math.isfinite(output) else float("nan")


def scalar(value: object) -> str:
    number = clean_float(value)
    return "" if not math.isfinite(number) else f"{number:.12g}"


def export_current_jmp_to_csv() -> pd.DataFrame:
    script = f"""
Names Default To Here( 1 );
dt = Open( "{jsl_path(JMP_PATH)}", Invisible );
dt << Save( "{jsl_path(ROUNDTRIP_CSV)}" );
Close( dt, NoSave );
"""
    app = win32com.client.Dispatch("JMP.Application.19")
    app.RunCommand(script)
    return pd.read_csv(ROUNDTRIP_CSV, dtype=str, keep_default_na=False)


def compute_setup_columns(data: pd.DataFrame) -> pd.DataFrame:
    original_profiles = np.vstack(
        [parse_profile(text, use_after=False) for text in data["attribute values before and after"]]
    )
    edited_profiles = np.vstack(
        [parse_profile(text, use_after=True) for text in data["attribute values before and after"]]
    )

    # Unique displayed training instances, because each participant saw the same
    # latest v2.0 training batch.
    training = data.loc[data["phase"].eq("training")].drop_duplicates("instance id")
    training_profiles = {
        str(row["instance id"]): parse_profile(row["attribute values before and after"], use_after=False)
        for _, row in training.iterrows()
    }
    training_labels = {
        str(row["instance id"]): str(row["original label"])
        for _, row in training.iterrows()
    }
    training_matrix = np.vstack(list(training_profiles.values()))
    training_ids = list(training_profiles.keys())

    centroids = {}
    for label in ["Diabetes", "No Diabetes"]:
        label_profiles = [
            profile
            for instance_id, profile in training_profiles.items()
            if training_labels[instance_id] == label
        ]
        centroids[label] = np.mean(np.vstack(label_profiles), axis=0)

    out = pd.DataFrame(index=data.index)
    is_testing = data["phase"].eq("testing")

    nearest_distances: list[str] = []
    nearest_ids: list[str] = []
    opposite_original: list[str] = []
    opposite_new: list[str] = []
    opposite_change: list[str] = []
    same_original: list[str] = []

    for row_index, row in data.iterrows():
        if not is_testing.loc[row_index]:
            nearest_distances.append("")
            nearest_ids.append("")
            opposite_original.append("")
            opposite_new.append("")
            opposite_change.append("")
            same_original.append("")
            continue

        original = original_profiles[row_index]
        edited = edited_profiles[row_index]
        distances = np.sum(np.abs(training_matrix - original), axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_distances.append(scalar(distances[nearest_index]))
        nearest_ids.append(training_ids[nearest_index])

        target_label = str(row["target label"])
        original_label = str(row["original label"])
        opp_o = l1(original, centroids[target_label])
        opp_n = l1(edited, centroids[target_label])
        opposite_original.append(scalar(opp_o))
        opposite_new.append(scalar(opp_n))
        opposite_change.append(scalar(opp_n - opp_o))
        same_original.append(scalar(l1(original, centroids[original_label])))

    out["nearest training instance distance (L1 normalized)"] = nearest_distances
    out["nearest training instance id"] = nearest_ids
    out["opposite-class centroid distance original (L1 normalized)"] = opposite_original
    out["opposite-class centroid distance new (L1 normalized)"] = opposite_new
    out["opposite-class centroid distance change (new - original)"] = opposite_change
    out["same-class centroid distance original (L1 normalized)"] = same_original

    testing = data.loc[is_testing].copy()
    testing["boundary distance new numeric"] = pd.to_numeric(
        testing["boundary distance new"], errors="coerce"
    )

    participant_stats = testing.groupby("participant")["boundary distance new numeric"].agg(
        ["mean", "median", "std"]
    )
    participant_q = testing.groupby("participant")["boundary distance new numeric"].quantile(
        [0.25, 0.75]
    ).unstack()
    participant_iqr = participant_q[0.75] - participant_q[0.25]

    instance_stats = testing.groupby("instance id")["boundary distance new numeric"].agg(
        ["mean", "median", "std", "count"]
    )

    for stat_name, source_name in [
        ("participant mean boundary distance new", "mean"),
        ("participant median boundary distance new", "median"),
        ("participant SD boundary distance new", "std"),
    ]:
        out[stat_name] = data["participant"].map(participant_stats[source_name]).map(scalar)
        out.loc[~is_testing, stat_name] = ""

    out["participant IQR boundary distance new"] = (
        data["participant"].map(participant_iqr).map(scalar)
    )
    out.loc[~is_testing, "participant IQR boundary distance new"] = ""

    for stat_name, source_name in [
        ("instance mean boundary distance new", "mean"),
        ("instance median boundary distance new", "median"),
        ("instance SD boundary distance new", "std"),
        ("instance testing row count", "count"),
    ]:
        out[stat_name] = data["instance id"].map(instance_stats[source_name]).map(scalar)
        out.loc[~is_testing, stat_name] = ""

    for x_name in X_FEATURES:
        change = pd.to_numeric(data[f"{x_name}_change"], errors="coerce")
        changed = pd.to_numeric(data[f"{x_name}_changed"], errors="coerce").eq(1)
        direction_column = f"{x_name}_changed_in_training_cf_direction"
        direction = pd.to_numeric(data.get(direction_column, ""), errors="coerce")
        out[f"{x_name}_absolute_change"] = change.abs().where(changed).map(scalar)
        signed_toward = change.abs().where(direction.eq(1), -change.abs())
        out[f"{x_name}_signed_change_toward_training_cf_direction"] = (
            signed_toward.where(changed).map(scalar)
        )

    return out


def values_to_jsl(values: pd.Series) -> str:
    parts = []
    for value in values.astype(str).tolist():
        stripped = value.strip()
        parts.append("." if stripped == "" else stripped)
    return ", ".join(parts)


def add_missing_columns_to_jmp(setup: pd.DataFrame, existing_columns: set[str]) -> list[str]:
    columns_to_add = [column for column in setup.columns if column not in existing_columns]
    if not columns_to_add:
        return []

    commands = []
    for column in columns_to_add:
        commands.append(
            f'''
dt << New Column(
	"{column}",
	Numeric,
	Continuous,
	Set Values( {{{values_to_jsl(setup[column])}}} )
);
'''
        )

    script = f'''
Names Default To Here( 1 );
dt = Open( "{jsl_path(JMP_PATH)}", Invisible );
If( N Rows( dt ) != {len(setup)},
	Throw( "Row count mismatch: JMP table has " || Char( N Rows( dt ) ) || " rows, expected {len(setup)}" )
);
{chr(10).join(commands)}
dt << Save( "{jsl_path(JMP_PATH)}" );
Close( dt, NoSave );
'''
    app = win32com.client.Dispatch("JMP.Application.19")
    app.RunCommand(script)
    return columns_to_add


def main() -> None:
    source = pd.read_csv(SOURCE_CSV, dtype=str, keep_default_na=False)
    current_jmp = export_current_jmp_to_csv()
    if len(source) != len(current_jmp):
        raise ValueError(f"Row mismatch: source CSV={len(source)}, JMP={len(current_jmp)}")

    # Carry the five direction columns from current JMP if they are not in the
    # source CSV, because the setup signed-change columns depend on them.
    for column in [f"{x}_changed_in_training_cf_direction" for x in X_FEATURES]:
        if column not in source.columns and column in current_jmp.columns:
            source[column] = current_jmp[column]

    setup = compute_setup_columns(source)
    added = add_missing_columns_to_jmp(setup, set(current_jmp.columns))

    print(f"current_jmp_rows={len(current_jmp)}")
    print(f"current_jmp_columns_before={len(current_jmp.columns)}")
    print(f"added_columns={len(added)}")
    for column in added:
        nonblank = int(setup[column].astype(str).str.strip().ne("").sum())
        print(f"  {column}: nonblank={nonblank}")


if __name__ == "__main__":
    main()
