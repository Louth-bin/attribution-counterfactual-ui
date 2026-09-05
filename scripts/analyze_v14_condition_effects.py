"""Exploratory XAI-condition comparisons for qualtrics_results_v1.4.csv."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "qualtrics" / "qualtrics_results_v1.4.csv"


def q(name: str) -> str:
    return f'Q("{name}")'


def main() -> None:
    df = pd.read_csv(DATA)
    df = df.loc[df["phase"].str.lower().eq("testing")].copy()
    df["analysis log response time"] = np.log1p(
        pd.to_numeric(df["response time (seconds)"], errors="coerce")
    )
    order = ["none", "attribution", "counterfactual"]
    df["xai"] = pd.Categorical(df["xai"], categories=order, ordered=True)

    metrics = {
        "success": ("successful counterfactual (0/1)", "binary", "higher"),
        "move toward target": ("move towards target (0/1)", "binary", "higher"),
        "sparsity": ("sparsity", "continuous", "higher"),
        "proximity (distance)": ("proximity", "continuous", "lower"),
        "plausibility": ("plausibility", "continuous", "higher"),
        "actionability": ("actionability (0/1)", "binary", "higher"),
        "boundary distance": ("boundary distance new", "continuous", "lower"),
        "confidence improvement": ("delta confidence of target label", "continuous", "higher"),
        "response time (log)": ("analysis log response time", "continuous", "lower"),
    }

    print(f"testing rows={len(df)}, participants={df['participant'].nunique()}")
    print("participants by condition:")
    print(df.groupby("xai", observed=True)["participant"].nunique().to_string())

    print("\nRAW PARTICIPANT-MEAN SUMMARIES")
    results = []
    for label, (column, kind, better) in metrics.items():
        use = df[["participant", "xai", column]].dropna()
        pm = use.groupby(["participant", "xai"], observed=True)[column].mean().reset_index()
        summary = pm.groupby("xai", observed=True)[column].agg(["mean", "std", "count"])
        summary["se"] = summary["std"] / np.sqrt(summary["count"])
        print(f"\n{label} ({better} is better)")
        print(summary[["mean", "se", "count"]].round(4).to_string())

        # Participant is the unit for the rough omnibus test.
        ols = smf.ols(f"{q(column)} ~ C(xai)", data=pm).fit(cov_type="HC3")
        hypothesis = "C(xai)[T.attribution] = 0, C(xai)[T.counterfactual] = 0"
        participant_p = float(ols.f_test(hypothesis).pvalue)

        # Adjust for case difficulty while clustering repeated rows by participant.
        formula = f"{q(column)} ~ C(xai) + C({q('instance id')})"
        if kind == "binary":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.gee(
                    formula,
                    groups="participant",
                    data=df.dropna(subset=[column]),
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable(),
                ).fit()
            names = ["C(xai)[T.attribution]", "C(xai)[T.counterfactual]"]
            restriction = np.zeros((2, len(fit.params)))
            for row, name in enumerate(names):
                restriction[row, fit.params.index.get_loc(name)] = 1
            adjusted_p = float(fit.wald_test(restriction, scalar=True).pvalue)
        else:
            fit = smf.ols(formula, data=df.dropna(subset=[column])).fit(
                cov_type="cluster", cov_kwds={"groups": df.dropna(subset=[column])["participant"]}
            )
            adjusted_p = float(
                fit.f_test("C(xai)[T.attribution] = 0, C(xai)[T.counterfactual] = 0").pvalue
            )

        results.append((label, participant_p, adjusted_p))

    print("\nOMNIBUS CONDITION TESTS")
    print("metric\tparticipant-mean p\tinstance-adjusted/participant-clustered p")
    for label, p_pm, p_adj in results:
        print(f"{label}\t{p_pm:.4f}\t{p_adj:.4f}")

    # Direction-specific success and movement summaries.
    print("\nDIRECTION-SPECIFIC RATES")
    for outcome in ["successful counterfactual (0/1)", "move towards target (0/1)"]:
        tab = df.groupby(["xai", "target label"], observed=True)[outcome].mean().unstack()
        print(f"\n{outcome}")
        print(tab.round(3).to_string())

    print("\nWITHIN-PARTICIPANT TARGET-DIRECTION DIFFERENCES")
    for label, (column, _kind, _better) in metrics.items():
        paired = (
            df.groupby(["participant", "target label"])[column]
            .mean()
            .unstack()
            .dropna()
        )
        if {"Diabetes", "No Diabetes"}.issubset(paired.columns):
            diff = paired["No Diabetes"] - paired["Diabetes"]
            test = stats.ttest_1samp(diff, 0, nan_policy="omit")
            print(
                f"{label}\tNo Diabetes - Diabetes={diff.mean():.4f}"
                f"\tt={test.statistic:.3f}\tp={test.pvalue:.4g}\tn={len(diff)}"
            )


if __name__ == "__main__":
    main()
