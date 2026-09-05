from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QUALTRICS = ROOT / "qualtrics"

FEATURES = [
    ("x_1", "Glucose", "glucose"),
    ("x_2", "Blood Pressure", "blood_pressure"),
    ("x_3", "Insulin", "insulin"),
    ("x_4", "BMI", "bmi"),
    ("x_5", "Age", "age"),
]

DIRECT_SHAP_COLS = [
    f"{prefix} {label} SHAP change Δφ" for prefix, label, _raw in FEATURES
]
TARGET_HELPFUL_SHAP_COLS = [
    f"{prefix} {label} target-helpful SHAP change" for prefix, label, _raw in FEATURES
]
TRAIN_CHANGE_COLS = [f"{prefix}_train_change" for prefix, _label, _raw in FEATURES]


def _load_training_changes(bundle_path: Path) -> dict[int, list[float]]:
    """Return instance_id -> signed normalized CF delta for each feature."""
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    training_pool = bundle["datasets"]["diabetes"]["training_pool"]
    out: dict[int, list[float]] = {}
    for case in training_pool:
        instance_id = int(case["instance_id"])
        before = case["feature_values"]
        after = case["counterfactual"]["feature_values"]
        ranges = case["feature_ranges"]
        changes = []
        for b, a, r in zip(before, after, ranges):
            span = float(r[1]) - float(r[0])
            changes.append((float(a) - float(b)) / span if span else 0.0)
        out[instance_id] = changes
    return out


def _blank_if_unchanged(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    masked = pd.DataFrame(index=df.index)
    for i, col in enumerate(columns, start=1):
        changed = pd.to_numeric(df[f"x_{i}_changed"], errors="coerce").fillna(0).astype(float)
        values = pd.to_numeric(df[col], errors="coerce")
        masked[col] = values.where(changed == 1, np.nan)
    return masked


def _training_change_frame(df: pd.DataFrame, training_changes: dict[int, list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(index=df.index)
    phase = df["phase"].astype(str).str.lower()
    instance_ids = pd.to_numeric(df["instance id"], errors="coerce")
    for feature_idx, col in enumerate(TRAIN_CHANGE_COLS):
        values = []
        for is_training, instance_id in zip(phase.eq("training"), instance_ids):
            if is_training and pd.notna(instance_id) and int(instance_id) in training_changes:
                change = training_changes[int(instance_id)][feature_idx]
                values.append(change if abs(change) > 1e-12 else np.nan)
            else:
                values.append(np.nan)
        frame[col] = values
    return frame


def prepare_v20() -> None:
    df = pd.read_csv(QUALTRICS / "qualtrics_results_v2.0_with_shap_changes.csv")
    training_changes = _load_training_changes(ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json")

    out = pd.DataFrame({"_row": np.arange(1, len(df) + 1)})
    out = pd.concat(
        [
            out,
            _blank_if_unchanged(df, DIRECT_SHAP_COLS),
            _blank_if_unchanged(df, TARGET_HELPFUL_SHAP_COLS),
            _training_change_frame(df, training_changes),
        ],
        axis=1,
    )
    out.to_csv(QUALTRICS / "v2.0_jmp_extra_columns_source.csv", index=False, encoding="utf-8-sig")


def prepare_v14() -> None:
    df = pd.read_csv(QUALTRICS / "qualtrics_results_v1.4.csv")
    training_changes = _load_training_changes(ROOT / "analysis" / "diabetes_experiment_bundle_v1.4.json")

    out = pd.DataFrame({"_row": np.arange(1, len(df) + 1)})
    out = pd.concat([out, _training_change_frame(df, training_changes)], axis=1)
    out.to_csv(QUALTRICS / "v1.4_jmp_train_change_columns_source.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    prepare_v20()
    prepare_v14()
    print("Wrote v2.0_jmp_extra_columns_source.csv and v1.4_jmp_train_change_columns_source.csv")
