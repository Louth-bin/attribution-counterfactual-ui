"""Test XAI-condition effects in participant-level simulated outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "qualtrics" / "v1.4_all_old_domains_model_vs_participant_statistics.csv"
RAW = ROOT / "qualtrics" / "qualtrics_results_v1.1.schema-export.csv"


def condition_vector(parameter_names: list[str], level: str) -> np.ndarray:
    vector = np.zeros(len(parameter_names))
    vector[0] = 1.0
    key = f"C(xai)[T.{level}]"
    if key in parameter_names:
        vector[parameter_names.index(key)] = 1.0
    return vector


def main() -> None:
    frame = pd.read_csv(STATS)
    raw = pd.read_csv(RAW, low_memory=False)
    raw = raw[(raw["phase"] == "testing") & raw["participant"].isin(frame["participant"])]
    changed = [f"x_{index}_changed" for index in range(1, 6)]
    raw["complexity"] = raw[changed].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    complexity = raw.groupby("participant", as_index=False)["complexity"].mean()
    frame = frame.merge(complexity, on="participant", how="left")

    levels = ("none", "attribution", "counterfactual")
    output = {}
    for outcome in ("model validity", "model boundary distance", "model plausibility", "complexity"):
        additive_plain = smf.ols(f'Q("{outcome}") ~ C(xai) + C(domain)', data=frame).fit()
        additive = additive_plain.get_robustcov_results(cov_type="HC3")
        names = list(additive_plain.params.index)
        condition_indices = [index for index, name in enumerate(names) if name.startswith("C(xai)")]
        restriction = np.zeros((len(condition_indices), len(names)))
        for row_index, parameter_index in enumerate(condition_indices):
            restriction[row_index, parameter_index] = 1.0
        omnibus_p = float(additive.wald_test(restriction, scalar=True).pvalue)

        contrasts = []
        for first, second in (
            ("attribution", "none"),
            ("counterfactual", "none"),
            ("counterfactual", "attribution"),
        ):
            contrast = condition_vector(names, first) - condition_vector(names, second)
            test = additive.t_test(contrast)
            contrasts.append({
                "contrast": f"{first} - {second}",
                "difference": float(np.asarray(test.effect).item()),
                "p": float(test.pvalue),
            })
        adjusted = multipletests([row["p"] for row in contrasts], method="holm")[1]
        for row, adjusted_p in zip(contrasts, adjusted):
            row["Holm p"] = float(adjusted_p)

        interaction = smf.ols(f'Q("{outcome}") ~ C(xai) * C(domain)', data=frame).fit()
        interaction_p = float(anova_lm(additive_plain, interaction).iloc[1]["Pr(>F)"])
        within_domain = {}
        for domain in ("diabetes", "housing", "safelimit"):
            subset = frame[frame["domain"] == domain]
            domain_plain = smf.ols(f'Q("{outcome}") ~ C(xai)', data=subset).fit()
            domain_robust = domain_plain.get_robustcov_results(cov_type="HC3")
            domain_names = list(domain_plain.params.index)
            domain_indices = [index for index, name in enumerate(domain_names) if name.startswith("C(xai)")]
            domain_restriction = np.zeros((len(domain_indices), len(domain_names)))
            for row_index, parameter_index in enumerate(domain_indices):
                domain_restriction[row_index, parameter_index] = 1.0
            within_domain[domain] = {
                "p": float(domain_robust.wald_test(domain_restriction, scalar=True).pvalue),
                "means": {
                    level: float(subset.loc[subset["xai"] == level, outcome].mean())
                    for level in levels
                },
            }
        output[outcome] = {
            "means": {level: float(frame.loc[frame["xai"] == level, outcome].mean()) for level in levels},
            "condition omnibus HC3 p": omnibus_p,
            "condition by domain interaction p": interaction_p,
            "within-domain condition tests": within_domain,
            "contrasts": contrasts,
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
