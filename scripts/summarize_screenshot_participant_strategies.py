"""Summarize strategies/comments for participant IDs read from screenshots."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "qualtrics" / "qualtrics_results_v2.0.current.roundtrip.csv"
BASE_RESULTS = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
RAW = ROOT / "qualtrics" / "raw_output_v1.9 (2 clusters new clustering).csv"

PARTICIPANTS = [
    "R_6dM4kpwwdD2TmDK",
    "R_6upJU84Jtf3Faf5",
    "R_1hF6UnZQsvvwBCX",
    "R_1jEfVgE4WoI6fVO",
    "R_1ihbKBZG0EAXF8D",
    "R_771uyV9h8uinmh3",
    "R_7QQowZQ43kZcCj7",
    "R_3CPW0AmZIfESbrT",
    "R_5faag2PFpaGyI6V",
]

FEATURE_LABELS = {
    1: "Glucose",
    2: "Blood Pressure",
    3: "Insulin",
    4: "BMI",
    5: "Age",
}


def num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def nums(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main() -> None:
    data = pd.read_csv(RESULTS, dtype=str, keep_default_na=False)
    base = pd.read_csv(BASE_RESULTS, dtype=str, keep_default_na=False)
    for column in [
        "reasoning strategy",
        "participant training accuracy",
        "cognitive model v0.1 family",
    ]:
        if column not in data.columns and column in base.columns and len(base) == len(data):
            data[column] = base[column]
    raw = pd.read_csv(RAW, skiprows=[1, 2], dtype=str, keep_default_na=False)
    raw_by_id = raw.set_index("ResponseId", drop=False)

    for participant in PARTICIPANTS:
        rows = data[data["participant"].eq(participant)].copy()
        print("=" * 88)
        print(participant)
        if rows.empty:
            print("NOT FOUND in v2.0 results")
            continue
        testing = rows[rows["phase"].eq("testing")].copy()
        for col in [
            "boundary distance new",
            "successful counterfactual (0/1)",
            "proximity",
            "move towards target (0/1)",
            "participant training accuracy",
            "response time (seconds)",
        ]:
            if col in testing.columns:
                testing[col] = pd.to_numeric(testing[col], errors="coerce")

        print("xai:", rows["xai"].iloc[0])
        print("reasoning strategy:", rows["reasoning strategy"].replace("", pd.NA).dropna().iloc[0] if rows["reasoning strategy"].replace("", pd.NA).dropna().size else "")
        print(
            "metrics:",
            {
                "mean_boundary": round(float(testing["boundary distance new"].mean()), 3),
                "median_boundary": round(float(testing["boundary distance new"].median()), 3),
                "sd_boundary": round(float(testing["boundary distance new"].std()), 3),
                "success": round(float(testing["successful counterfactual (0/1)"].mean()), 3),
                "move_toward_target": round(float(testing["move towards target (0/1)"].mean()), 3),
                "mean_proximity": round(float(testing["proximity"].mean()), 3),
                "training_accuracy": round(float(pd.to_numeric(rows["participant training accuracy"], errors="coerce").mean()), 3),
                "median_testing_time_s": round(float(testing["response time (seconds)"].median()), 1),
            },
        )

        changed = {}
        mean_signed = {}
        direction_rate = {}
        for i in range(1, 6):
            changed_col = f"x_{i}_changed"
            signed_col = f"x_{i}_signed_change_toward_training_cf_direction"
            dir_col = f"x_{i}_changed_in_training_cf_direction"
            changed[FEATURE_LABELS[i]] = round(float(num(testing, changed_col).mean()), 2)
            if signed_col in testing.columns:
                mean_signed[FEATURE_LABELS[i]] = round(float(num(testing, signed_col).mean()), 3)
            if dir_col in testing.columns:
                vals = nums(testing.loc[testing[dir_col].ne(""), dir_col])
                direction_rate[FEATURE_LABELS[i]] = None if vals.empty else round(float(vals.mean()), 2)
        print("feature changed prevalence:", changed)
        print("mean signed change toward CF direction:", mean_signed)
        print("direction-correct rate when changed:", direction_rate)

        if participant in raw_by_id.index:
            raw_row = raw_by_id.loc[participant]
            print("PostTask_MentalModel:")
            print(str(raw_row.get("PostTask_MentalModel", "")).strip() or "[blank]")
            print("PostTask_StrategyExample:")
            print(str(raw_row.get("PostTask_StrategyExample", "")).strip() or "[blank]")
        else:
            print("Raw comments: participant not found in raw output")


if __name__ == "__main__":
    main()
