from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "qualtrics" / "qualtrics_results_v2.0_with_shap_changes.csv"
if not source.exists():
    source = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"

df = pd.read_csv(source)
t = df[df["phase"].eq("testing")].copy()
screen = pd.read_csv(
    ROOT / "outputs" / "v20-mlp-two-cluster-design-test" / "model_screened_current_test_instances.csv"
)
top8 = set(screen.head(8)["instance id"].astype(int))
samepair = {160201, 160203, 160205, 160208, 160210, 160212, 160213, 160218}

t["model_screened_top8"] = t["instance id"].astype(int).isin(top8)
t["same_pair_AB"] = t["instance id"].astype(int).isin(samepair)
t["bd"] = pd.to_numeric(t["boundary distance new"], errors="coerce")
t["success"] = pd.to_numeric(t["successful counterfactual (0/1)"], errors="coerce")
t["prox"] = pd.to_numeric(t["proximity"], errors="coerce")
t["improve"] = pd.to_numeric(t["boundary distance original"], errors="coerce") - t["bd"]

rows = []
for subset_name, mask in [
    ("model_screened_top8", t["model_screened_top8"]),
    ("model_screened_out", ~t["model_screened_top8"]),
    ("same_pair_AB", t["same_pair_AB"]),
    ("off_pair", ~t["same_pair_AB"]),
]:
    sub = t[mask].copy()
    for y in ["bd", "improve", "success", "prox"]:
        for ref in ["none", "attribution"]:
            model = smf.ols(
                f'{y} ~ C(xai, Treatment(reference="{ref}")) + C(Q("instance id"))',
                data=sub,
            ).fit(cov_type="cluster", cov_kwds={"groups": sub["participant"]})
            for term in model.params.index:
                if "xai" in term:
                    rows.append(
                        {
                            "subset": subset_name,
                            "outcome": y,
                            "reference": ref,
                            "term": term,
                            "coef": model.params[term],
                            "p": model.pvalues[term],
                            "n_rows": len(sub),
                            "n_participants": sub["participant"].nunique(),
                        }
                    )

out = pd.DataFrame(rows)
path = ROOT / "outputs" / "v20-mlp-two-cluster-design-test" / "screened_subset_empirical_models.csv"
try:
    out.to_csv(path, index=False)
except PermissionError as error:
    print(f"Could not save CSV: {error}")
print(out.round(4).to_string(index=False))
print(path)
