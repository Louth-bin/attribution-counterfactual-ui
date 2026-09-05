from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "qualtrics" / "qualtrics_results_v2.0_with_shap_changes.csv")
t = df[df["phase"].eq("testing")].copy()

rows = []
features = [
    ("x_1", "Glucose"),
    ("x_2", "Blood Pressure"),
    ("x_3", "Insulin"),
    ("x_4", "BMI"),
    ("x_5", "Age"),
]
for x_name, display in features:
    col = f"{x_name} {display} target-helpful SHAP change"
    t["y"] = pd.to_numeric(t[col], errors="coerce")
    model = smf.ols(
        'y ~ C(xai, Treatment(reference="none")) + C(Q("instance id"))',
        data=t,
    ).fit(cov_type="cluster", cov_kwds={"groups": t["participant"]})
    means = t.groupby("xai")["y"].mean()
    rows.append(
        {
            "feature": display,
            "none_mean": means.get("none", np.nan),
            "attribution_mean": means.get("attribution", np.nan),
            "counterfactual_mean": means.get("counterfactual", np.nan),
            "attr_minus_none_coef": model.params.get(
                'C(xai, Treatment(reference="none"))[T.attribution]', np.nan
            ),
            "attr_p": model.pvalues.get(
                'C(xai, Treatment(reference="none"))[T.attribution]', np.nan
            ),
            "cf_minus_none_coef": model.params.get(
                'C(xai, Treatment(reference="none"))[T.counterfactual]', np.nan
            ),
            "cf_p": model.pvalues.get(
                'C(xai, Treatment(reference="none"))[T.counterfactual]', np.nan
            ),
        }
    )

out = pd.DataFrame(rows)
path = ROOT / "outputs" / "v20-shap-change-significance" / "feature_target_helpful_shap_condition_scan.csv"
out.to_csv(path, index=False)
print(out.round(4).to_string(index=False))
print(path)
