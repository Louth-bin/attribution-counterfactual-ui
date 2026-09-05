from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "qualtrics" / "qualtrics_results_v2.0_with_shap_changes.csv"
if not source.exists():
    source = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"

df = pd.read_csv(source)
t = df[(df["phase"].eq("testing")) & (df["target label"].eq("No Diabetes"))].copy()
t["success"] = pd.to_numeric(t["successful counterfactual (0/1)"], errors="coerce")
t = t[t["success"].eq(1)].copy()
t["bd"] = pd.to_numeric(t["boundary distance new"], errors="coerce")
t["prox"] = pd.to_numeric(t["proximity"], errors="coerce")
t["improve"] = pd.to_numeric(t["boundary distance original"], errors="coerce") - t["bd"]

rows = []
for y in ["bd", "prox", "improve"]:
    for reference in ["none", "attribution"]:
        model = smf.ols(
            f'{y} ~ C(xai, Treatment(reference="{reference}")) + C(Q("instance id"))',
            data=t,
        ).fit(cov_type="cluster", cov_kwds={"groups": t["participant"]})
        for term in model.params.index:
            if "xai" in term:
                rows.append(
                    {
                        "outcome": y,
                        "reference": reference,
                        "term": term,
                        "coef": model.params[term],
                        "p": model.pvalues[term],
                    }
                )

out = pd.DataFrame(rows)
print(out.round(4).to_string(index=False))
