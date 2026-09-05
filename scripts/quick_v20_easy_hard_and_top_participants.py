from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "qualtrics" / "qualtrics_results_v2.0_with_shap_changes.csv"
    if (ROOT / "qualtrics" / "qualtrics_results_v2.0_with_shap_changes.csv").exists()
    else ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
)
OUTDIR = ROOT / "outputs" / "v20-easy-hard-participants"
OUTDIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    ("x_1", "Glucose"),
    ("x_2", "Blood Pressure"),
    ("x_3", "Insulin"),
    ("x_4", "BMI"),
    ("x_5", "Age"),
]

VALUE_RE = re.compile(
    r'"(?P<feature>[^"]+)"\s*-\s*'
    r'(?P<before_raw>[-+0-9.eE]+)\((?P<before_norm>[-+0-9.eE]+)\)'
    r'(?:\s*->\s*(?P<after_raw>[-+0-9.eE]+)\((?P<after_norm>[-+0-9.eE]+)\))?'
)


def parse_norms(text: str, after: bool) -> dict[str, float]:
    values = {}
    for match in VALUE_RE.finditer(str(text)):
        feature = match.group("feature")
        before = float(match.group("before_norm"))
        after_text = match.group("after_norm")
        values[feature] = float(after_text) if after and after_text is not None else before
    return values


def mean_abs(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").abs().mean())


def mean_num(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean())


def add_common_columns(t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy()
    out["boundary"] = pd.to_numeric(out["boundary distance new"], errors="coerce")
    out["orig_boundary"] = pd.to_numeric(out["boundary distance original"], errors="coerce")
    out["improvement"] = out["orig_boundary"] - out["boundary"]
    out["success"] = pd.to_numeric(out["successful counterfactual (0/1)"], errors="coerce")
    out["prox"] = pd.to_numeric(out["proximity"], errors="coerce")
    out["spar"] = pd.to_numeric(out["sparsity"], errors="coerce")
    out["move_target"] = pd.to_numeric(out["move towards target (0/1)"], errors="coerce")
    if "target-helpful SHAP change" in out.columns:
        out["helpful_shap"] = pd.to_numeric(out["target-helpful SHAP change"], errors="coerce")
    else:
        out["helpful_shap"] = np.nan
    out["n_features_changed"] = sum(
        pd.to_numeric(out[f"{x}_changed"], errors="coerce").fillna(0.0)
        for x, _ in FEATURES
    )
    for x, name in FEATURES:
        out[f"{name}_abs_change"] = pd.to_numeric(out[f"{x}_change"], errors="coerce").abs()
        out[f"{name}_signed_change"] = pd.to_numeric(out[f"{x}_change"], errors="coerce")
        out[f"{name}_changed"] = pd.to_numeric(out[f"{x}_changed"], errors="coerce")
    return out


def summarize_group(g: pd.DataFrame) -> dict[str, float]:
    rec = {
        "n_rows": len(g),
        "n_participants": g["participant"].nunique(),
        "mean_boundary": mean_num(g["boundary"]),
        "median_boundary": float(g["boundary"].median()),
        "mean_orig_boundary": mean_num(g["orig_boundary"]),
        "mean_improvement": mean_num(g["improvement"]),
        "success_rate": mean_num(g["success"]),
        "mean_proximity": mean_num(g["prox"]),
        "mean_sparsity": mean_num(g["spar"]),
        "mean_n_features_changed": mean_num(g["n_features_changed"]),
        "move_toward_target_rate": mean_num(g["move_target"]),
        "mean_helpful_shap": mean_num(g["helpful_shap"]),
    }
    for _, name in FEATURES:
        rec[f"{name}_change_prevalence"] = mean_num(g[f"{name}_changed"])
        rec[f"{name}_mean_abs_change"] = mean_num(g[f"{name}_abs_change"])
        rec[f"{name}_mean_signed_change"] = mean_num(g[f"{name}_signed_change"])
    return rec


def main() -> None:
    df = pd.read_csv(SOURCE)
    t = add_common_columns(df[df["phase"].eq("testing")].copy())

    # Instance-level summaries.
    instance_records = []
    for instance_id, g in t.groupby("instance id"):
        before = parse_norms(g["attribute values before and after"].iloc[0], after=False)
        row = {
            "instance id": int(instance_id),
            "original label": g["original label"].iloc[0],
            "target label": g["target label"].iloc[0],
            **summarize_group(g),
        }
        for _, name in FEATURES:
            row[f"original_{name}_norm"] = before.get(name, np.nan)
        instance_records.append(row)
    inst = pd.DataFrame(instance_records).sort_values("mean_boundary")
    inst["easy_rank"] = np.arange(1, len(inst) + 1)
    inst.to_csv(OUTDIR / "instance_easy_difficult_summary.csv", index=False)

    easy_ids = set(inst.head(5)["instance id"])
    hard_ids = set(inst.tail(5)["instance id"])
    eh = []
    for label, ids in [("easy_lowest_boundary", easy_ids), ("difficult_highest_boundary", hard_ids)]:
        subset = t[t["instance id"].isin(ids)]
        row = {"group": label, "instance_ids": ",".join(map(str, sorted(ids)))}
        row.update(summarize_group(subset))
        eh.append(row)
    easy_hard = pd.DataFrame(eh)
    easy_hard.to_csv(OUTDIR / "easy_vs_difficult_aggregate.csv", index=False)

    # Top participants within condition.
    participant_rows = []
    for (xai, participant), g in t.groupby(["xai", "participant"]):
        row = {
            "xai": xai,
            "participant": participant,
            **summarize_group(g),
            "reasoning_strategy": str(g["reasoning strategy"].dropna().iloc[0])
            if g["reasoning strategy"].notna().any()
            else "",
        }
        participant_rows.append(row)
    part = pd.DataFrame(participant_rows).sort_values(["xai", "mean_boundary"])
    part["within_condition_boundary_rank"] = part.groupby("xai")["mean_boundary"].rank(
        method="first",
        ascending=True,
    )
    part.to_csv(OUTDIR / "participant_boundary_strategy_summary.csv", index=False)
    top = part[part["within_condition_boundary_rank"] <= 10].copy()
    top.to_csv(OUTDIR / "top10_participants_by_condition.csv", index=False)

    top_vs_rest = []
    for xai, g in part.groupby("xai"):
        for label, sub in [
            ("top10", g[g["within_condition_boundary_rank"] <= 10]),
            ("rest", g[g["within_condition_boundary_rank"] > 10]),
        ]:
            rec = {"xai": xai, "participant_group": label, "n_participants": len(sub)}
            for col in [
                "mean_boundary",
                "success_rate",
                "mean_proximity",
                "mean_sparsity",
                "mean_n_features_changed",
                "move_toward_target_rate",
                "mean_helpful_shap",
                "Glucose_change_prevalence",
                "Blood Pressure_change_prevalence",
                "Insulin_change_prevalence",
                "BMI_change_prevalence",
                "Age_change_prevalence",
                "Glucose_mean_abs_change",
                "Blood Pressure_mean_abs_change",
                "Insulin_mean_abs_change",
                "BMI_mean_abs_change",
                "Age_mean_abs_change",
            ]:
                rec[col] = mean_num(sub[col])
            top_vs_rest.append(rec)
    top_rest = pd.DataFrame(top_vs_rest)
    top_rest.to_csv(OUTDIR / "top10_vs_rest_by_condition.csv", index=False)

    print("EASIEST INSTANCES")
    print(
        inst.head(8)[
            [
                "instance id",
                "original label",
                "target label",
                "mean_boundary",
                "mean_orig_boundary",
                "success_rate",
                "mean_proximity",
                "mean_n_features_changed",
                "move_toward_target_rate",
                "Glucose_change_prevalence",
                "Blood Pressure_change_prevalence",
                "Insulin_change_prevalence",
                "BMI_change_prevalence",
                "Age_change_prevalence",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\nHARDEST INSTANCES")
    print(
        inst.tail(8)[
            [
                "instance id",
                "original label",
                "target label",
                "mean_boundary",
                "mean_orig_boundary",
                "success_rate",
                "mean_proximity",
                "mean_n_features_changed",
                "move_toward_target_rate",
                "Glucose_change_prevalence",
                "Blood Pressure_change_prevalence",
                "Insulin_change_prevalence",
                "BMI_change_prevalence",
                "Age_change_prevalence",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\nEASY VS HARD AGGREGATE")
    print(easy_hard.round(3).to_string(index=False))
    print("\nTOP 10 VS REST BY CONDITION")
    print(top_rest.round(3).to_string(index=False))
    print("\nTOP PARTICIPANTS")
    print(
        top[
            [
                "xai",
                "within_condition_boundary_rank",
                "participant",
                "mean_boundary",
                "success_rate",
                "mean_proximity",
                "mean_n_features_changed",
                "move_toward_target_rate",
                "mean_helpful_shap",
                "reasoning_strategy",
            ]
        ]
        .round(3)
        .to_string(index=False, max_colwidth=90)
    )
    print(f"\nWrote: {OUTDIR}")


if __name__ == "__main__":
    main()
