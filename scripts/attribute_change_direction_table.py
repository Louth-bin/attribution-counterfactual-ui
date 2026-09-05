"""Attribute-level change prevalence, direction, success, and boundary table."""

from __future__ import annotations

import re

import joblib
import numpy as np
import pandas as pd


FEATURES = [
    ("Glucose", "glucose", "x_1"),
    ("Blood Pressure", "blood_pressure", "x_2"),
    ("Insulin", "insulin", "x_3"),
    ("BMI", "bmi", "x_4"),
    ("Age", "age", "x_5"),
]
CONDITIONS = ["none", "attribution", "counterfactual"]
LABEL_TO_INDEX = {"Diabetes": 0, "No Diabetes": 1}
VALUE_RE = re.compile(r'"(.+?)" - ([^()]+)\([^)]*\)(?: -> ([^()]+)\([^)]*\))?')


def parse_values(text: str) -> dict[str, tuple[float, float]]:
    values: dict[str, tuple[float, float]] = {}
    for match in VALUE_RE.finditer(str(text)):
        before = float(match.group(2))
        after = float(match.group(3)) if match.group(3) is not None else before
        values[match.group(1)] = (before, after)
    return values


def main() -> None:
    model = joblib.load("analysis/diabetes_mlp_regularized.joblib")
    data = pd.read_csv("qualtrics/qualtrics_results_v2.0.csv")
    testing = data.loc[data["phase"].eq("testing")].copy()
    for _, _, prefix in FEATURES:
        testing[f"{prefix}_changed"] = pd.to_numeric(
            testing[f"{prefix}_changed"], errors="coerce"
        )
    for column in [
        "successful counterfactual (0/1)",
        "boundary distance new",
        "proximity",
    ]:
        testing[column] = pd.to_numeric(testing[column], errors="coerce")

    parsed = [parse_values(value) for value in testing["attribute values before and after"]]
    original = np.asarray(
        [[row[display][0] for display, _, _ in FEATURES] for row in parsed],
        dtype=float,
    )
    after = np.asarray(
        [[row[display][1] for display, _, _ in FEATURES] for row in parsed],
        dtype=float,
    )
    raw_names = [raw for _, raw, _ in FEATURES]
    target = testing["target label"].map(LABEL_TO_INDEX).to_numpy()
    base_probabilities = model.predict_proba(pd.DataFrame(original, columns=raw_names))

    rows = []
    for index, (display, _, prefix) in enumerate(FEATURES):
        changed = testing[f"{prefix}_changed"].eq(1).to_numpy()
        single_feature_after = original.copy()
        single_feature_after[:, index] = after[:, index]
        probabilities = model.predict_proba(
            pd.DataFrame(single_feature_after, columns=raw_names)
        )
        correct_direction = np.asarray(
            [
                probabilities[row_index, target[row_index]]
                > base_probabilities[row_index, target[row_index]] + 1e-12
                for row_index in range(len(testing))
            ],
            dtype=bool,
        )
        testing[f"correct_direction_{prefix}"] = np.where(
            changed, correct_direction.astype(float), np.nan
        )

        for condition in CONDITIONS:
            group = testing.loc[testing["xai"].eq(condition)]
            changed_group = group.loc[group[f"{prefix}_changed"].eq(1)]
            unchanged_group = group.loc[group[f"{prefix}_changed"].eq(0)]
            rows.append(
                {
                    "attribute": display,
                    "condition": condition,
                    "changed rows": len(changed_group),
                    "prevalence changed": len(changed_group) / len(group),
                    "correct direction when changed": changed_group[
                        f"correct_direction_{prefix}"
                    ].mean(),
                    "success when changed": changed_group[
                        "successful counterfactual (0/1)"
                    ].mean(),
                    "boundary distance when changed": changed_group[
                        "boundary distance new"
                    ].mean(),
                    "edit distance when changed": changed_group["proximity"].mean(),
                    "success when not changed": unchanged_group[
                        "successful counterfactual (0/1)"
                    ].mean(),
                    "boundary distance when not changed": unchanged_group[
                        "boundary distance new"
                    ].mean(),
                }
            )

    output = pd.DataFrame(rows)
    display = output.copy()
    for column in [
        "prevalence changed",
        "correct direction when changed",
        "success when changed",
        "success when not changed",
    ]:
        display[column] = (display[column] * 100).round(1)
    for column in [
        "boundary distance when changed",
        "edit distance when changed",
        "boundary distance when not changed",
    ]:
        display[column] = display[column].round(3)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
