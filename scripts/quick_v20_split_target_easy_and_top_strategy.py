from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore")
RESULTS = ROOT / "qualtrics" / "qualtrics_results_v2.0_with_shap_changes.csv"
if not RESULTS.exists():
    RESULTS = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
RAW = ROOT / "qualtrics" / "raw_output_v1.9 (2 clusters new clustering).csv"
OUTDIR = ROOT / "outputs" / "v20-target-split-top-strategies"
OUTDIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    ("x_1", "Glucose"),
    ("x_2", "Blood Pressure"),
    ("x_3", "Insulin"),
    ("x_4", "BMI"),
    ("x_5", "Age"),
]
COND_ORDER = {"none": 0, "attribution": 1, "counterfactual": 2}
TEXT_FIELDS = [
    "PostTask_MentalModel",
    "PostTask_StrategyExample",
    "Q70_3_TEXT",
    "Q73_6_TEXT",
    "Q74_7_TEXT",
]


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def prep_testing() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    t = df[df["phase"].eq("testing")].copy()
    t["boundary"] = numeric(t["boundary distance new"])
    t["orig_boundary"] = numeric(t["boundary distance original"])
    t["improvement"] = t["orig_boundary"] - t["boundary"]
    t["success"] = numeric(t["successful counterfactual (0/1)"])
    t["prox"] = numeric(t["proximity"])
    t["spar"] = numeric(t["sparsity"])
    t["move_target"] = numeric(t["move towards target (0/1)"])
    if "target-helpful SHAP change" in t:
        t["helpful_shap"] = numeric(t["target-helpful SHAP change"])
    else:
        t["helpful_shap"] = np.nan
    t["n_features_changed"] = 0.0
    for x, name in FEATURES:
        t[f"{name}_changed"] = numeric(t[f"{x}_changed"]).fillna(0)
        t[f"{name}_abs_change"] = numeric(t[f"{x}_change"]).abs()
        t[f"{name}_signed_change"] = numeric(t[f"{x}_change"])
        t["n_features_changed"] += t[f"{name}_changed"]
    return t


def participant_summary(t: pd.DataFrame) -> pd.DataFrame:
    aggs = {
        "boundary": "mean",
        "orig_boundary": "mean",
        "improvement": "mean",
        "success": "mean",
        "prox": "mean",
        "spar": "mean",
        "move_target": "mean",
        "helpful_shap": "mean",
        "n_features_changed": "mean",
    }
    for _, name in FEATURES:
        aggs[f"{name}_changed"] = "mean"
        aggs[f"{name}_abs_change"] = "mean"
        aggs[f"{name}_signed_change"] = "mean"
    out = t.groupby(["xai", "participant"], as_index=False).agg(aggs)
    out = out.rename(
        columns={
            "boundary": "mean_boundary",
            "orig_boundary": "mean_orig_boundary",
            "improvement": "mean_improvement",
            "success": "success_rate",
            "prox": "mean_proximity",
            "spar": "mean_sparsity",
            "move_target": "move_toward_target_rate",
            "helpful_shap": "mean_helpful_shap",
            "n_features_changed": "mean_n_features_changed",
        }
    )
    out["within_condition_boundary_rank"] = out.groupby("xai")["mean_boundary"].rank(
        method="first", ascending=True
    )
    return out.sort_values(["xai", "within_condition_boundary_rank"])


def load_comments() -> pd.DataFrame:
    if not RAW.exists():
        return pd.DataFrame(columns=["participant", *TEXT_FIELDS, "all_comments"])
    raw = pd.read_csv(RAW, skiprows=[1, 2], dtype=str, keep_default_na=False)
    cols = ["ResponseId"] + [c for c in TEXT_FIELDS if c in raw.columns]
    raw = raw[cols].copy()
    raw = raw.rename(columns={"ResponseId": "participant"})
    comments = []
    for _, row in raw.iterrows():
        bits = []
        for c in TEXT_FIELDS:
            if c in row and str(row[c]).strip():
                bits.append(f"{c}: {str(row[c]).strip()}")
        comments.append(" | ".join(bits))
    raw["all_comments"] = comments
    return raw


def cluster_top(top: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_cols = [
        "success_rate",
        "mean_proximity",
        "move_toward_target_rate",
        "mean_n_features_changed",
        "Glucose_changed",
        "Blood Pressure_changed",
        "Insulin_changed",
        "BMI_changed",
        "Age_changed",
        "Glucose_abs_change",
        "Blood Pressure_abs_change",
        "Insulin_abs_change",
        "BMI_abs_change",
        "Age_abs_change",
    ]
    x = top[cluster_cols].fillna(0).to_numpy(float)
    x = StandardScaler().fit_transform(x)
    diagnostics = []
    labels_by_k = {}
    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = km.fit_predict(x)
        labels_by_k[k] = labels + 1
        diagnostics.append(
            {
                "k": k,
                "silhouette": silhouette_score(x, labels),
                "sizes": dict(pd.Series(labels + 1).value_counts().sort_index()),
            }
        )
    best_k = max(diagnostics, key=lambda d: d["silhouette"])["k"]
    top = top.copy()
    top["strategy_cluster"] = labels_by_k[best_k]
    diag = pd.DataFrame(diagnostics)
    diag["best_k"] = best_k
    return top, diag


def main() -> None:
    t = prep_testing()

    # Instance summaries split by target label.
    inst = (
        t.groupby(["target label", "instance id", "original label"], as_index=False)
        .agg(
            mean_boundary=("boundary", "mean"),
            median_boundary=("boundary", "median"),
            mean_orig_boundary=("orig_boundary", "mean"),
            mean_improvement=("improvement", "mean"),
            success_rate=("success", "mean"),
            mean_proximity=("prox", "mean"),
            mean_n_features_changed=("n_features_changed", "mean"),
            move_toward_target_rate=("move_target", "mean"),
            mean_helpful_shap=("helpful_shap", "mean"),
            glucose_changed=("Glucose_changed", "mean"),
            bp_changed=("Blood Pressure_changed", "mean"),
            insulin_changed=("Insulin_changed", "mean"),
            bmi_changed=("BMI_changed", "mean"),
            age_changed=("Age_changed", "mean"),
            glucose_signed=("Glucose_signed_change", "mean"),
            bp_signed=("Blood Pressure_signed_change", "mean"),
            insulin_signed=("Insulin_signed_change", "mean"),
            bmi_signed=("BMI_signed_change", "mean"),
            age_signed=("Age_signed_change", "mean"),
        )
        .sort_values(["target label", "mean_boundary"])
    )
    inst.to_csv(OUTDIR / "instance_summary_split_by_target.csv", index=False)

    # Easy/hard within each target.
    easy_hard_rows = []
    for target, g in inst.groupby("target label"):
        for label, sub in [("easy_low_boundary", g.head(3)), ("hard_high_boundary", g.tail(3))]:
            rec = {
                "target label": target,
                "group": label,
                "instance_ids": ",".join(map(str, sub["instance id"].astype(int).tolist())),
            }
            for c in [
                "mean_boundary",
                "mean_orig_boundary",
                "mean_improvement",
                "success_rate",
                "mean_proximity",
                "mean_n_features_changed",
                "move_toward_target_rate",
                "mean_helpful_shap",
                "glucose_changed",
                "bp_changed",
                "insulin_changed",
                "bmi_changed",
                "age_changed",
                "glucose_signed",
                "bp_signed",
                "insulin_signed",
                "bmi_signed",
                "age_signed",
            ]:
                rec[c] = sub[c].mean()
            easy_hard_rows.append(rec)
    easy_hard = pd.DataFrame(easy_hard_rows)
    easy_hard.to_csv(OUTDIR / "easy_hard_within_target_summary.csv", index=False)

    participants = participant_summary(t)
    comments = load_comments()
    participants = participants.merge(comments, on="participant", how="left")
    participants.to_csv(OUTDIR / "participant_strategy_summary_with_comments.csv", index=False)
    top = participants[participants["within_condition_boundary_rank"] <= 10].copy()
    top_clustered, diag = cluster_top(top)
    top_clustered.to_csv(OUTDIR / "top10_by_condition_clustered_with_comments.csv", index=False)
    diag.to_csv(OUTDIR / "top10_strategy_cluster_diagnostics.csv", index=False)

    cluster_summary = (
        top_clustered.groupby("strategy_cluster")
        .agg(
            n=("participant", "size"),
            xai_counts=(
                "xai",
                lambda s: json.dumps(
                    {str(k): int(v) for k, v in s.value_counts().sort_index().items()}
                ),
            ),
            mean_boundary=("mean_boundary", "mean"),
            success_rate=("success_rate", "mean"),
            mean_proximity=("mean_proximity", "mean"),
            move_toward_target_rate=("move_toward_target_rate", "mean"),
            mean_n_features_changed=("mean_n_features_changed", "mean"),
            glucose_changed=("Glucose_changed", "mean"),
            bp_changed=("Blood Pressure_changed", "mean"),
            insulin_changed=("Insulin_changed", "mean"),
            bmi_changed=("BMI_changed", "mean"),
            age_changed=("Age_changed", "mean"),
            glucose_abs=("Glucose_abs_change", "mean"),
            bp_abs=("Blood Pressure_abs_change", "mean"),
            insulin_abs=("Insulin_abs_change", "mean"),
            bmi_abs=("BMI_abs_change", "mean"),
            age_abs=("Age_abs_change", "mean"),
        )
        .reset_index()
    )
    cluster_summary.to_csv(OUTDIR / "top10_strategy_cluster_summary.csv", index=False)

    print("INSTANCE SUMMARY SPLIT BY TARGET")
    for target, g in inst.groupby("target label"):
        print(f"\nTARGET: {target} -- easiest")
        print(
            g.head(5)[
                [
                    "instance id",
                    "original label",
                    "mean_boundary",
                    "mean_orig_boundary",
                    "success_rate",
                    "mean_proximity",
                    "move_toward_target_rate",
                    "glucose_signed",
                    "bp_signed",
                    "insulin_signed",
                    "bmi_signed",
                    "age_signed",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
        print(f"\nTARGET: {target} -- hardest")
        print(
            g.tail(5)[
                [
                    "instance id",
                    "original label",
                    "mean_boundary",
                    "mean_orig_boundary",
                    "success_rate",
                    "mean_proximity",
                    "move_toward_target_rate",
                    "glucose_signed",
                    "bp_signed",
                    "insulin_signed",
                    "bmi_signed",
                    "age_signed",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )

    print("\nEASY/HARD WITHIN TARGET AGGREGATE")
    print(easy_hard.round(3).to_string(index=False))

    print("\nTOP-10 STRATEGY CLUSTER DIAGNOSTICS")
    print(diag.to_string(index=False))
    print("\nTOP-10 STRATEGY CLUSTER SUMMARY")
    print(cluster_summary.round(3).to_string(index=False))

    print("\nTOP PARTICIPANTS WITH COMMENTS")
    cols = [
        "xai",
        "within_condition_boundary_rank",
        "participant",
        "strategy_cluster",
        "mean_boundary",
        "success_rate",
        "mean_proximity",
        "mean_n_features_changed",
        "move_toward_target_rate",
        "all_comments",
    ]
    print(top_clustered[cols].round(3).to_string(index=False, max_colwidth=180))
    print(f"\nWrote {OUTDIR}")


if __name__ == "__main__":
    main()
