"""Independently audit original-versus-submitted diabetes plausibility in v0.9."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import calculate_v09_plausibility as plausibility_calc


OUTPUT_PATH = Path(tempfile.gettempdir()) / "v09_diabetes_plausibility_audit.json"
RAW_VALUE_RE = re.compile(
    r'-\s*(-?(?:\d+(?:\.\d*)?|\.\d+))(?:\([^)]*\))?'
    r'(?:\s*->\s*(-?(?:\d+(?:\.\d*)?|\.\d+)))?'
)


def parsed_profile(text: str, *, final: bool) -> np.ndarray:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 5:
        raise ValueError(f"Expected five lines, got {len(lines)}")
    result = []
    for line in lines:
        values = plausibility_calc.NORMALIZED_VALUE_RE.findall(line)
        if not values:
            raise ValueError(f"No normalized value in {line!r}")
        result.append(float(values[-1] if final else values[0]))
    return np.asarray(result, dtype=float)


def raw_profile(text: str, *, final: bool) -> np.ndarray:
    lines = [line for line in text.splitlines() if line.strip()]
    result = []
    for line in lines:
        match = RAW_VALUE_RE.search(line)
        if match is None:
            raise ValueError(f"No raw value in {line!r}")
        before, after = match.groups()
        result.append(float(after if final and after is not None else before))
    return np.asarray(result, dtype=float)


def normalize_matrix(
    matrix: np.ndarray, ranges: np.ndarray, *, clamp: bool
) -> np.ndarray:
    normalized = (matrix - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])
    return np.clip(normalized, 0, 1) if clamp else normalized


def nearest_l1(reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    return cKDTree(reference).query(queries, k=1, p=1)[0]


def main() -> None:
    experiment = json.loads(
        plausibility_calc.EXPERIMENT_DATA.read_text(encoding="utf-8")
    )
    diabetes = experiment["datasets"]["diabetes"]
    representative = diabetes["test_pool"][0]
    feature_names = list(plausibility_calc.DOMAIN_FEATURES["diabetes"])
    ranges = np.asarray(representative["raw_feature_ranges"], dtype=float)

    train_frame = pd.read_csv(
        plausibility_calc.REPO_ROOT / "src" / "data" / "diabetes" / "train.csv"
    )
    test_frame = pd.read_csv(
        plausibility_calc.REPO_ROOT / "src" / "data" / "diabetes" / "test.csv"
    )
    train_raw = train_frame[feature_names].to_numpy(float)
    test_raw = test_frame[feature_names].to_numpy(float)
    train_clipped = normalize_matrix(train_raw, ranges, clamp=True)
    test_clipped = normalize_matrix(test_raw, ranges, clamp=True)
    train_unclipped = normalize_matrix(train_raw, ranges, clamp=False)
    test_unclipped = normalize_matrix(test_raw, ranges, clamp=False)

    survey = pd.read_csv(plausibility_calc.INPUT_CSV)
    survey = survey.loc[
        (survey["phase"] == "testing") & (survey["domain"] == "diabetes")
    ].copy()
    originals = np.vstack(
        [
            parsed_profile(text, final=False)
            for text in survey["attribute values before and after"]
        ]
    )
    finals = np.vstack(
        [
            parsed_profile(text, final=True)
            for text in survey["attribute values before and after"]
        ]
    )
    original_raw = np.vstack(
        [
            raw_profile(text, final=False)
            for text in survey["attribute values before and after"]
        ]
    )
    final_raw = np.vstack(
        [
            raw_profile(text, final=True)
            for text in survey["attribute values before and after"]
        ]
    )

    expected_originals = test_clipped[survey["instance id"].astype(int).to_numpy()]
    if not np.allclose(originals, expected_originals, atol=1e-10):
        raise ValueError("Parsed original normalized profiles do not match test.csv")
    expected_original_raw = test_raw[survey["instance id"].astype(int).to_numpy()]
    if not np.allclose(original_raw, expected_original_raw, atol=1e-10):
        raise ValueError("Parsed original raw profiles do not match test.csv")

    original_distance = nearest_l1(train_clipped, originals)
    final_distance = nearest_l1(train_clipped, finals)
    survey["original_distance"] = original_distance
    survey["final_distance"] = final_distance
    survey["distance_delta"] = final_distance - original_distance

    # Recalculate with no quantile-range clipping, using the raw values in the rows.
    original_unclipped = normalize_matrix(original_raw, ranges, clamp=False)
    final_unclipped = normalize_matrix(final_raw, ranges, clamp=False)
    original_unclipped_distance = nearest_l1(train_unclipped, original_unclipped)
    final_unclipped_distance = nearest_l1(train_unclipped, final_unclipped)
    survey["unclipped_delta"] = (
        final_unclipped_distance - original_unclipped_distance
    )

    condition = (
        survey.groupby("xai")
        .agg(
            rows=("distance_delta", "size"),
            participants=("participant", "nunique"),
            original_mean=("original_distance", "mean"),
            final_mean=("final_distance", "mean"),
            delta_mean=("distance_delta", "mean"),
            delta_median=("distance_delta", "median"),
            rows_improved=("distance_delta", lambda values: int((values < -1e-12).sum())),
            rows_unchanged=("distance_delta", lambda values: int((values.abs() <= 1e-12).sum())),
            rows_worsened=("distance_delta", lambda values: int((values > 1e-12).sum())),
            unclipped_delta_mean=("unclipped_delta", "mean"),
        )
        .reset_index()
    )

    case_condition = (
        survey.groupby(["xai", "instance id"])
        .agg(
            original_distance=("original_distance", "first"),
            final_distance=("final_distance", "mean"),
            delta=("distance_delta", "mean"),
        )
        .reset_index()
    )
    case_summary = (
        case_condition.groupby("xai")
        .agg(
            cases=("delta", "size"),
            case_mean_delta=("delta", "mean"),
            cases_improved=("delta", lambda values: int((values < 0).sum())),
            cases_worsened=("delta", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )

    selected_ids = sorted(survey["instance id"].astype(int).unique())
    selected_distances = nearest_l1(train_clipped, test_clipped[selected_ids])
    all_test_distances = nearest_l1(train_clipped, test_clipped)
    all_test_unclipped_distances = nearest_l1(train_unclipped, test_unclipped)

    train_tree = cKDTree(train_clipped)
    train_leave_one_out = train_tree.query(train_clipped, k=2, p=1)[0][:, 1]

    outside_selected = (
        (test_raw[selected_ids] < ranges[:, 0])
        | (test_raw[selected_ids] > ranges[:, 1])
    )

    report = {
        "verification": {
            "survey_originals_match_test_csv": True,
            "maximum_original_normalized_mismatch": float(
                np.abs(originals - expected_originals).max()
            ),
            "maximum_original_raw_mismatch": float(
                np.abs(original_raw - expected_original_raw).max()
            ),
            "final_scores_recomputed_directly": True,
        },
        "reference": {
            "training_rows": int(len(train_clipped)),
            "training_leave_one_out_mean": float(train_leave_one_out.mean()),
            "training_leave_one_out_median": float(np.median(train_leave_one_out)),
            "all_test_rows": int(len(test_clipped)),
            "all_test_original_mean": float(all_test_distances.mean()),
            "all_test_original_median": float(np.median(all_test_distances)),
            "all_test_unclipped_original_mean": float(
                all_test_unclipped_distances.mean()
            ),
            "selected_test_cases": len(selected_ids),
            "selected_original_mean": float(selected_distances.mean()),
            "selected_original_median": float(np.median(selected_distances)),
            "selected_values_outside_ui_range": int(outside_selected.sum()),
            "selected_cases_with_any_value_outside_ui_range": int(
                outside_selected.any(axis=1).sum()
            ),
        },
        "condition_summary": condition.round(12).to_dict(orient="records"),
        "case_summary": case_summary.round(12).to_dict(orient="records"),
        "case_details": case_condition.sort_values(
            ["xai", "delta"]
        ).round(12).to_dict(orient="records"),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
