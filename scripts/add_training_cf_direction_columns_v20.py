"""Add training-counterfactual direction columns to v2.0 Qualtrics results.

For each feature x_1..x_5, the new column is blank when that feature was not
changed, 1 when the participant changed it in the direction taught by the
counterfactual explanations during training, and 0 when changed in the
opposite direction.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
OUTPUT = ROOT / "qualtrics" / "qualtrics_results_v2.0.with_cf_direction_columns.csv"

FEATURES = [
    ("x_1", "Glucose"),
    ("x_2", "Blood Pressure"),
    ("x_3", "Insulin"),
    ("x_4", "BMI"),
    ("x_5", "Age"),
]

VALUE_PATTERN = re.compile(
    r'"(?P<feature>[^"]+)"\s*-\s*'
    r'(?P<before_raw>[^(\n]+)\((?P<before_norm>[-+0-9.eE]+)\)'
    r'(?:\s*->\s*(?P<after_raw>[^(\n]+)\((?P<after_norm>[-+0-9.eE]+)\))?'
)


def infer_taught_directions(data: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Infer feature directions by target class from training counterfactuals.

    Training rows have an original label but not a filled target-label column,
    so the training target is the opposite class. The CFs teach that risk-feature
    values increase when targeting Diabetes and decrease when targeting
    No Diabetes. Age is not one of the displayed two-feature CF pairs in v2.0,
    so it is assigned the same target-class sign convention.
    """
    signs: dict[tuple[str, str], list[int]] = defaultdict(list)
    rows_by_instance = data.loc[
        data["phase"].eq("training") & data["xai"].eq("counterfactual"),
        ["instance id", "original label", "explanation"],
    ].drop_duplicates("instance id")
    for _, row in rows_by_instance.iterrows():
        original_label = str(row["original label"])
        target_label = "Diabetes" if original_label == "No Diabetes" else "No Diabetes"
        explanation = str(row["explanation"] or "")
        for match in VALUE_PATTERN.finditer(explanation):
            after_raw = match.group("after_raw")
            if after_raw is None:
                continue
            before_value = float(match.group("before_raw").strip())
            after_value = float(after_raw.strip())
            delta = after_value - before_value
            if abs(delta) < 1e-12:
                continue
            signs[(match.group("feature"), target_label)].append(1 if delta > 0 else -1)

    directions: dict[tuple[str, str], int] = {}
    inconsistent: dict[tuple[str, str], list[int]] = {}
    for _, feature in FEATURES:
        for target_label in ("Diabetes", "No Diabetes"):
            key = (feature, target_label)
            unique_signs = sorted(set(signs.get(key, [])))
            if len(unique_signs) == 1:
                directions[key] = unique_signs[0]
            elif len(unique_signs) > 1:
                inconsistent[key] = signs[key]

    # Age is not directly manipulated in the v2.0 training CF examples. Use the
    # same risk-feature convention taught by the CFs: higher values move toward
    # Diabetes and lower values move toward No Diabetes.
    directions.setdefault(("Age", "Diabetes"), 1)
    directions.setdefault(("Age", "No Diabetes"), -1)

    missing_non_age = []
    for _, feature in FEATURES:
        if feature == "Age":
            continue
        for target_label in ("Diabetes", "No Diabetes"):
            key = (feature, target_label)
            if key not in directions:
                missing_non_age.append(key)
    if missing_non_age:
        raise ValueError(f"Could not infer training-CF directions for: {missing_non_age}")

    if inconsistent:
        details = {
            f"{feature} -> {target}": {"n": len(values), "signs": sorted(set(values))}
            for (feature, target), values in inconsistent.items()
        }
        raise ValueError(f"Inconsistent training-CF directions: {details}")

    return directions


def add_direction_columns(data: pd.DataFrame, directions: dict[tuple[str, str], int]) -> pd.DataFrame:
    updated = data.copy()
    parsed_raw_signs: list[dict[str, int]] = []
    for text in updated["attribute values before and after"].fillna(""):
        row_signs: dict[str, int] = {}
        for match in VALUE_PATTERN.finditer(str(text)):
            after_raw = match.group("after_raw")
            if after_raw is None:
                continue
            delta = float(after_raw.strip()) - float(match.group("before_raw").strip())
            if abs(delta) > 1e-12:
                row_signs[match.group("feature")] = 1 if delta > 0 else -1
        parsed_raw_signs.append(row_signs)

    for x_name, feature in FEATURES:
        changed_col = f"{x_name}_changed"
        output_col = f"{x_name}_changed_in_training_cf_direction"

        changed = pd.to_numeric(updated[changed_col], errors="coerce").eq(1)
        values = pd.Series("", index=updated.index, dtype=object)
        for target_label in ("Diabetes", "No Diabetes"):
            taught_sign = directions[(feature, target_label)]
            mask = changed & updated["target label"].eq(target_label)
            raw_sign = pd.Series(
                [row_signs.get(feature, 0) for row_signs in parsed_raw_signs],
                index=updated.index,
            )
            correct = raw_sign == taught_sign
            values.loc[mask] = correct.loc[mask].astype(int).astype(str)
        updated[output_col] = values

    return updated


def main() -> None:
    data = pd.read_csv(INPUT)
    directions = infer_taught_directions(data)
    updated = add_direction_columns(data, directions)
    updated.to_csv(OUTPUT, index=False)

    print("Inferred training-counterfactual direction signs:")
    for x_name, feature in FEATURES:
        diabetes = "increase" if directions[(feature, "Diabetes")] > 0 else "decrease"
        no_diabetes = "increase" if directions[(feature, "No Diabetes")] > 0 else "decrease"
        age_note = " (age sign extrapolated; age was not shown in training CFs)" if feature == "Age" else ""
        print(f"  {x_name} {feature}: toward Diabetes={diabetes}, toward No Diabetes={no_diabetes}{age_note}")

    print("\nNew columns:")
    for x_name, _ in FEATURES:
        col = f"{x_name}_changed_in_training_cf_direction"
        nonblank = int(updated[col].astype(str).ne("").sum())
        ones = int(updated[col].astype(str).eq("1").sum())
        zeros = int(updated[col].astype(str).eq("0").sum())
        print(f"  {col}: nonblank={nonblank}, 1={ones}, 0={zeros}")

    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
