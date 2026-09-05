"""Quickly compare shortest two-feature CF pair vs attribution top-2 pair.

This is intentionally approximate: for each pair, scan L1-normalized rays from
the original point, then bisect crossing rays to estimate the smallest
two-feature edit that flips to the opposite class.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
MODEL = ROOT / "analysis" / "diabetes_mlp_regularized.joblib"
OUTDIR = ROOT / "outputs" / "v20-two-feature-minimal-vs-top2"


def make_directions(n: int = 241) -> np.ndarray:
    vals: list[tuple[float, float]] = []
    for a in np.linspace(-1.0, 1.0, n):
        b_abs = 1.0 - abs(float(a))
        vals.append((float(a), b_abs))
        if b_abs > 1e-12:
            vals.append((float(a), -b_abs))
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.sum(np.abs(arr), axis=1) > 0]
    return arr / np.sum(np.abs(arr), axis=1, keepdims=True)


def pair_distance(
    model,
    case: dict,
    pair: tuple[int, int],
    directions: np.ndarray,
) -> tuple[float, float, np.ndarray | None]:
    """Return approximate min normalized L1 distance, target prob, delta."""
    feature_names = list(case["raw_feature_names"])
    ranges = np.asarray(case["raw_feature_ranges"], dtype=float)
    original = np.asarray(case["raw_feature_values"], dtype=float)
    target = int(case["counterfactual"]["target_prediction"]["value"])

    z0 = np.asarray(
        [
            (original[i] - ranges[i, 0]) / (ranges[i, 1] - ranges[i, 0])
            for i in pair
        ],
        dtype=float,
    )

    max_r = np.full(len(directions), np.inf, dtype=float)
    for j in range(2):
        step = directions[:, j]
        pos = step > 0
        neg = step < 0
        max_r[pos] = np.minimum(max_r[pos], (1.0 - z0[j]) / step[pos])
        max_r[neg] = np.minimum(max_r[neg], (0.0 - z0[j]) / step[neg])
    keep = np.isfinite(max_r) & (max_r > 1e-10)
    if not np.any(keep):
        return math.inf, math.nan, None

    dirs = directions[keep]
    high = max_r[keep]

    def probs_at(radius: np.ndarray) -> np.ndarray:
        z = np.clip(z0[None, :] + radius[:, None] * dirs, 0.0, 1.0)
        values = np.repeat(original[None, :], len(z), axis=0)
        for pos, idx in enumerate(pair):
            values[:, idx] = ranges[idx, 0] + z[:, pos] * (ranges[idx, 1] - ranges[idx, 0])
        return np.asarray(model.predict_proba(pd.DataFrame(values, columns=feature_names)))[:, target]

    end_probs = probs_at(high)
    crossing = end_probs >= 0.5
    if not np.any(crossing):
        return math.inf, float(np.nanmax(end_probs)), None

    dirs = dirs[crossing]
    high = high[crossing]
    low = np.zeros_like(high)
    for _ in range(28):
        mid = (low + high) / 2.0
        crossed = probs_at(mid) >= 0.5
        high[crossed] = mid[crossed]
        low[~crossed] = mid[~crossed]

    probs = probs_at(high)
    best = int(np.argmin(high))
    delta = dirs[best] * high[best]
    return float(high[best]), float(probs[best]), delta


def main() -> None:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    model = joblib.load(MODEL)
    directions = make_directions()
    raw_names = list(bundle["training_pool"][0]["raw_feature_names"])
    pretty_names = list(bundle["training_pool"][0]["feature_names"])
    pretty = dict(zip(raw_names, pretty_names))
    all_pairs = list(itertools.combinations(range(len(raw_names)), 2))

    rows = []
    for pool_name in ("training_pool", "test_pool"):
        for case in bundle[pool_name]:
            top2_raw = tuple(
                sorted(
                    [
                        name
                        for name, _ in sorted(
                            case["feature_importance_by_name"].items(),
                            key=lambda kv: abs(float(kv[1])),
                            reverse=True,
                        )[:2]
                    ]
                )
            )
            top2_idx = tuple(sorted(raw_names.index(name) for name in top2_raw))
            shown_raw = tuple(sorted(case["counterfactual"]["raw_selected_feature_names"]))
            shown_idx = tuple(sorted(raw_names.index(name) for name in shown_raw))

            pair_rows = []
            for pair in all_pairs:
                dist, prob, delta = pair_distance(model, case, pair, directions)
                pair_rows.append((pair, dist, prob, delta))
            finite = [r for r in pair_rows if math.isfinite(r[1])]
            if not finite:
                continue
            best_pair, best_dist, best_prob, best_delta = min(finite, key=lambda r: r[1])
            top2_dist, top2_prob, top2_delta = next((d, p, de) for pa, d, p, de in pair_rows if pa == top2_idx)
            shown_dist, shown_prob, shown_delta = next((d, p, de) for pa, d, p, de in pair_rows if pa == shown_idx)
            gb_idx = tuple(sorted(raw_names.index(name) for name in ("glucose", "bmi")))
            gb_dist, gb_prob, gb_delta = next((d, p, de) for pa, d, p, de in pair_rows if pa == gb_idx)

            def pair_label(pair: tuple[int, int]) -> str:
                return "|".join(pretty[raw_names[i]] for i in pair)

            rows.append(
                {
                    "pool": pool_name.replace("_pool", ""),
                    "instance_id": int(case["instance_id"]),
                    "original_label": case["prediction"]["label"],
                    "target_label": case["counterfactual"]["target_prediction"]["label"],
                    "top2_attribution_pair": pair_label(top2_idx),
                    "shown_cf_pair": pair_label(shown_idx),
                    "minimal_two_feature_pair": pair_label(best_pair),
                    "minimal_pair_L1": best_dist,
                    "top2_pair_L1": top2_dist,
                    "shown_pair_L1": shown_dist,
                    "glucose_bmi_pair_L1": gb_dist,
                    "top2_excess_L1": top2_dist - best_dist,
                    "shown_excess_L1": shown_dist - best_dist,
                    "glucose_bmi_excess_L1": gb_dist - best_dist,
                    "top2_is_minimal_pair": top2_idx == best_pair,
                    "shown_is_minimal_pair": shown_idx == best_pair,
                    "minimal_pair_target_probability": best_prob,
                    "top2_pair_target_probability": top2_prob,
                    "shown_pair_target_probability": shown_prob,
                }
            )

    df = pd.DataFrame(rows)
    output_path = OUTDIR / "quick_two_feature_minimal_pair_vs_attribution_top2.csv"
    try:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        output_note = f"Output: {output_path}"
    except PermissionError:
        output_note = "Output CSV was not saved because Windows denied access to the output folder."

    print("Approximate all-pairs two-feature minimal CF search.")
    print(output_note)
    print()
    print("Coincidence counts")
    summary = (
        df.groupby("pool")
        .agg(
            n=("instance_id", "size"),
            top2_is_minimal=("top2_is_minimal_pair", "sum"),
            shown_is_minimal=("shown_is_minimal_pair", "sum"),
            mean_top2_excess=("top2_excess_L1", "mean"),
            median_top2_excess=("top2_excess_L1", "median"),
            max_top2_excess=("top2_excess_L1", "max"),
        )
        .round(4)
    )
    print(summary.to_string())
    print()
    for tol in (0.001, 0.005, 0.01):
        tol_summary = (
            df.assign(top2_within_tol=df["top2_excess_L1"] <= tol)
            .groupby("pool")
            .agg(n=("instance_id", "size"), top2_within_tol=("top2_within_tol", "sum"))
        )
        print(f"Top-2 attribution pair within {tol:g} normalized L1 of best pair")
        print(tol_summary.to_string())
    print()
    print("By target label")
    target_summary = (
        df.groupby(["pool", "target_label"])
        .agg(
            n=("instance_id", "size"),
            top2_is_minimal=("top2_is_minimal_pair", "sum"),
            mean_top2_excess=("top2_excess_L1", "mean"),
        )
        .round(4)
    )
    print(target_summary.to_string())
    print()
    print("Largest top2-vs-minimal mismatches")
    mismatch = df[~df["top2_is_minimal_pair"]].sort_values("top2_excess_L1", ascending=False)
    cols = [
        "pool",
        "instance_id",
        "target_label",
        "top2_attribution_pair",
        "minimal_two_feature_pair",
        "minimal_pair_L1",
        "top2_pair_L1",
        "top2_excess_L1",
    ]
    print(mismatch[cols].round(4).head(15).to_string(index=False))
    print()
    print("Clean cases where attribution top-2 DOES equal minimal pair")
    clean = df[df["top2_is_minimal_pair"]].sort_values(["pool", "minimal_pair_L1"])
    print(clean[cols].round(4).head(20).to_string(index=False))
    print()
    print("Where the obvious Glucose|BMI route is much worse than the shortest two-feature route")
    gb_cols = [
        "pool",
        "instance_id",
        "target_label",
        "top2_attribution_pair",
        "minimal_two_feature_pair",
        "minimal_pair_L1",
        "glucose_bmi_pair_L1",
        "glucose_bmi_excess_L1",
    ]
    print(df.sort_values("glucose_bmi_excess_L1", ascending=False)[gb_cols].round(4).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
