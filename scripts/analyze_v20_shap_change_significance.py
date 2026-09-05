"""Calculate v2.0 row-wise SHAP changes and scan significance routes.

Outputs:
- qualtrics/qualtrics_results_v2.0_with_shap_changes.csv
- outputs/v20-shap-change-significance/shap_change_condition_summary.csv
- outputs/v20-shap-change-significance/shap_change_models.txt
- outputs/v20-shap-change-significance/instance_cf_advantage_scan.csv
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ExplanationPipeline  # noqa: E402
from src.xai_methods.shap_method import REGULARIZED_FEATURE_COUNT  # noqa: E402


INPUT = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
OUTPUT = ROOT / "qualtrics" / "qualtrics_results_v2.0_with_shap_changes.csv"
OUTDIR = ROOT / "outputs" / "v20-shap-change-significance"

DISPLAY_TO_MODEL = {
    "Glucose": "glucose",
    "Blood Pressure": "blood_pressure",
    "Insulin": "insulin",
    "BMI": "bmi",
    "Age": "age",
}
MODEL_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_MODEL.items()}
FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
X_NAMES = ["x_1", "x_2", "x_3", "x_4", "x_5"]

VALUE_RE = re.compile(
    r'"(?P<feature>[^"]+)"\s*-\s*'
    r'(?P<before_raw>[-+0-9.eE]+)\((?P<before_norm>[-+0-9.eE]+)\)'
    r'(?:\s*->\s*(?P<after_raw>[-+0-9.eE]+)\((?P<after_norm>[-+0-9.eE]+)\))?'
)


def parse_profile(text: str, *, after: bool) -> dict[str, float]:
    values: dict[str, float] = {}
    for match in VALUE_RE.finditer(str(text)):
        display = match.group("feature")
        feature = DISPLAY_TO_MODEL.get(display)
        if not feature:
            continue
        raw = match.group("after_raw") if after and match.group("after_raw") is not None else match.group("before_raw")
        values[feature] = float(raw)
    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        raise ValueError(f"Missing {missing} from attribute text: {text!r}")
    return values


def condition_order(value: str) -> int:
    return {"none": 0, "attribution": 1, "counterfactual": 2}.get(str(value), 99)


def calculate_shap_values(points: pd.DataFrame) -> pd.DataFrame:
    """Batch KernelSHAP using the same regularized top-2 setup as UI attribution."""
    # Compatibility for older SHAP releases under newer numpy.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]

    import shap  # noqa: WPS433

    pipe = ExplanationPipeline()
    prepared = pipe.prepare_assets("diabetes", "mlp")
    dataset = prepared.dataset
    estimator = prepared.model_artifact.estimator
    background = dataset.train_df[dataset.feature_names].sample(
        n=min(len(dataset.train_df), 50),
        random_state=42,
    )
    points = points[dataset.feature_names].astype(float)

    explainer = shap.KernelExplainer(
        lambda rows: estimator.predict_proba(pd.DataFrame(rows, columns=dataset.feature_names)),
        background.astype(float).to_numpy(),
        feature_names=dataset.feature_names,
    )
    state = np.random.get_state()
    np.random.seed(42)
    try:
        shap_values = explainer.shap_values(
            points.astype(float).to_numpy(),
            nsamples=2 ** len(dataset.feature_names),
            l1_reg=f"num_features({REGULARIZED_FEATURE_COUNT})",
            silent=True,
        )
    finally:
        np.random.set_state(state)

    if isinstance(shap_values, list):
        class1 = np.asarray(shap_values[1], dtype=float)
    else:
        arr = np.asarray(shap_values, dtype=float)
        class1 = arr[:, :, 1] if arr.ndim == 3 else arr

    return pd.DataFrame(class1, columns=dataset.feature_names, index=points.index)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled else float("nan")


def fit_formula(formula: str, df: pd.DataFrame, cluster_col: str = "participant") -> tuple[str, dict[str, float]]:
    import statsmodels.formula.api as smf

    model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    wanted = [
        "C(xai, Treatment(reference=\"none\"))[T.attribution]",
        "C(xai, Treatment(reference=\"none\"))[T.counterfactual]",
        "target_helpful_shap_delta",
        "abs_total_shap_delta",
    ]
    terms = {}
    for term in wanted:
        if term in model.params.index:
            terms[f"{term}:coef"] = float(model.params[term])
            terms[f"{term}:p"] = float(model.pvalues[term])
    return model.summary().as_text(), terms


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    before_rows = []
    after_rows = []
    for _, row in df.iterrows():
        before_rows.append(parse_profile(row["attribute values before and after"], after=False))
        after_rows.append(parse_profile(row["attribute values before and after"], after=True))
    before = pd.DataFrame(before_rows, index=df.index)[FEATURES]
    after = pd.DataFrame(after_rows, index=df.index)[FEATURES]

    unique_points = pd.concat([before, after], axis=0).drop_duplicates().reset_index(drop=True)
    shap_unique = calculate_shap_values(unique_points)
    key_to_i = {tuple(row): i for i, row in unique_points.iterrows()}

    before_idx = [key_to_i[tuple(row)] for _, row in before.iterrows()]
    after_idx = [key_to_i[tuple(row)] for _, row in after.iterrows()]
    shap_before = shap_unique.iloc[before_idx].reset_index(drop=True)
    shap_after = shap_unique.iloc[after_idx].reset_index(drop=True)
    shap_delta = shap_after - shap_before

    enriched = df.copy()
    for x_name, feature in zip(X_NAMES, FEATURES):
        display = MODEL_TO_DISPLAY[feature]
        enriched[f"{x_name} {display} SHAP original φ"] = shap_before[feature].to_numpy()
        enriched[f"{x_name} {display} SHAP final φ"] = shap_after[feature].to_numpy()
        enriched[f"{x_name} {display} SHAP change Δφ"] = shap_delta[feature].to_numpy()

    enriched["total SHAP change ΔΣφ"] = shap_delta[FEATURES].sum(axis=1).to_numpy()
    # Class 1 is No Diabetes. A helpful movement is positive if target is No Diabetes,
    # and negative if target is Diabetes.
    sign = np.where(enriched["target label"].astype(str).eq("No Diabetes"), 1.0, -1.0)
    enriched["target-helpful SHAP change"] = sign * enriched["total SHAP change ΔΣφ"].astype(float)
    enriched["absolute total SHAP change"] = np.abs(enriched["total SHAP change ΔΣφ"].astype(float))
    for x_name, feature in zip(X_NAMES, FEATURES):
        display = MODEL_TO_DISPLAY[feature]
        enriched[f"{x_name} {display} target-helpful SHAP change"] = sign * shap_delta[feature].to_numpy()

    enriched.to_csv(OUTPUT, index=False)

    testing = enriched[enriched["phase"].eq("testing")].copy()
    testing["boundary"] = pd.to_numeric(testing["boundary distance new"], errors="coerce")
    testing["success"] = pd.to_numeric(testing["successful counterfactual (0/1)"], errors="coerce")
    testing["boundary_improvement"] = (
        pd.to_numeric(testing["boundary distance original"], errors="coerce")
        - pd.to_numeric(testing["boundary distance new"], errors="coerce")
    )
    testing["target_helpful_shap_delta"] = pd.to_numeric(testing["target-helpful SHAP change"], errors="coerce")
    testing["abs_total_shap_delta"] = pd.to_numeric(testing["absolute total SHAP change"], errors="coerce")

    summary = (
        testing.groupby("xai")
        .agg(
            n_rows=("participant", "size"),
            n_participants=("participant", "nunique"),
            mean_boundary=("boundary", "mean"),
            mean_success=("success", "mean"),
            mean_target_helpful_shap_delta=("target_helpful_shap_delta", "mean"),
            sd_target_helpful_shap_delta=("target_helpful_shap_delta", "std"),
            mean_abs_total_shap_delta=("abs_total_shap_delta", "mean"),
            mean_boundary_improvement=("boundary_improvement", "mean"),
        )
        .reset_index()
    )
    summary["xai_order"] = summary["xai"].map(condition_order)
    summary = summary.sort_values("xai_order").drop(columns=["xai_order"])
    summary.to_csv(OUTDIR / "shap_change_condition_summary.csv", index=False)

    formulas = {
        "condition_predicts_target_helpful_shap_delta": 'target_helpful_shap_delta ~ C(xai, Treatment(reference="none")) + C(Q("instance id"))',
        "target_helpful_shap_delta_predicts_boundary": 'boundary ~ target_helpful_shap_delta + C(xai, Treatment(reference="none")) + C(Q("instance id"))',
        "condition_predicts_boundary": 'boundary ~ C(xai, Treatment(reference="none")) + C(Q("instance id"))',
        "condition_predicts_success_linear": 'success ~ C(xai, Treatment(reference="none")) + C(Q("instance id"))',
    }
    text_parts = []
    compact_terms = {}
    for name, formula in formulas.items():
        text, terms = fit_formula(formula, testing)
        text_parts.append("\n" + "=" * 80 + f"\n{name}\n{formula}\n" + text)
        compact_terms[name] = terms

    # Quick instance-level scan: where counterfactual differs most from none/attribution.
    instance_rows = []
    for instance_id, g in testing.groupby("instance id"):
        by = {xai: part for xai, part in g.groupby("xai")}
        if not {"none", "attribution", "counterfactual"}.issubset(by):
            continue
        cf = by["counterfactual"]
        for comparator in ["none", "attribution"]:
            other = by[comparator]
            for outcome, lower_better in [
                ("boundary", True),
                ("target_helpful_shap_delta", False),
                ("boundary_improvement", False),
            ]:
                cf_values = cf[outcome].to_numpy(float)
                other_values = other[outcome].to_numpy(float)
                diff = float(np.nanmean(cf_values) - np.nanmean(other_values))
                beneficial_diff = -diff if lower_better else diff
                instance_rows.append(
                    {
                        "instance id": instance_id,
                        "comparison": f"counterfactual - {comparator}",
                        "outcome": outcome,
                        "cf_mean": float(np.nanmean(cf_values)),
                        "other_mean": float(np.nanmean(other_values)),
                        "raw_difference": diff,
                        "beneficial_difference": beneficial_diff,
                        "cohens_d_raw": cohens_d(cf_values, other_values),
                        "n_cf": int(np.isfinite(cf_values).sum()),
                        f"n_{comparator}": int(np.isfinite(other_values).sum()),
                    }
                )
    inst = pd.DataFrame(instance_rows).sort_values(["outcome", "beneficial_difference"], ascending=[True, False])
    inst.to_csv(OUTDIR / "instance_cf_advantage_scan.csv", index=False)

    payload = {
        "output_csv": str(OUTPUT),
        "unique_profiles_shap_evaluated": int(len(unique_points)),
        "condition_summary": summary.to_dict(orient="records"),
        "compact_model_terms": compact_terms,
        "note": "SHAP values are class-1 (No Diabetes) regularized KernelSHAP using l1_reg=num_features(2), matching UI attribution regularization.",
    }
    (OUTDIR / "shap_change_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUTDIR / "shap_change_models.txt").write_text("\n".join(text_parts), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print("Top counterfactual-beneficial instance differences:")
    print(inst.groupby("outcome").head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
