"""Plot per-attribute correct-direction rates by XAI condition."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attribute_change_direction_table import CONDITIONS, FEATURES, LABEL_TO_INDEX, parse_values

import joblib


OUTPUT_DIR = Path("outputs") / "v20-quick-boundary-diagnostics"
RAW_NAMES = [raw for _, raw, _ in FEATURES]


def main() -> None:
    model = joblib.load("analysis/diabetes_mlp_regularized.joblib")
    data = pd.read_csv("qualtrics/qualtrics_results_v2.0.csv")
    testing = data.loc[data["phase"].eq("testing")].copy()

    for _, _, prefix in FEATURES:
        testing[f"{prefix}_changed"] = pd.to_numeric(
            testing[f"{prefix}_changed"], errors="coerce"
        )

    parsed = [parse_values(value) for value in testing["attribute values before and after"]]
    original = np.asarray(
        [[row[display][0] for display, _, _ in FEATURES] for row in parsed],
        dtype=float,
    )
    after = np.asarray(
        [[row[display][1] for display, _, _ in FEATURES] for row in parsed],
        dtype=float,
    )
    target = testing["target label"].map(LABEL_TO_INDEX).to_numpy()
    base_probabilities = model.predict_proba(pd.DataFrame(original, columns=RAW_NAMES))

    rows = []
    for index, (display, _, prefix) in enumerate(FEATURES):
        changed = testing[f"{prefix}_changed"].eq(1).to_numpy()
        single_feature_after = original.copy()
        single_feature_after[:, index] = after[:, index]
        probabilities = model.predict_proba(
            pd.DataFrame(single_feature_after, columns=RAW_NAMES)
        )
        correct_direction = np.asarray(
            [
                probabilities[row_index, target[row_index]]
                > base_probabilities[row_index, target[row_index]] + 1e-12
                for row_index in range(len(testing))
            ],
            dtype=bool,
        )
        testing[f"correct_direction_{prefix}"] = np.where(
            changed, correct_direction.astype(float), np.nan
        )

        for condition in CONDITIONS:
            group = testing.loc[
                testing["xai"].eq(condition) & testing[f"{prefix}_changed"].eq(1)
            ]
            n = len(group)
            p = float(group[f"correct_direction_{prefix}"].mean())
            se = float(np.sqrt(p * (1.0 - p) / n)) if n else np.nan
            rows.append(
                {
                    "attribute": display,
                    "condition": condition,
                    "changed rows": n,
                    "correct direction rate": p,
                    "95% CI low": max(0.0, p - 1.96 * se),
                    "95% CI high": min(1.0, p + 1.96 * se),
                    "95% CI half-width": 1.96 * se,
                }
            )

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUTPUT_DIR / "attribute_correct_direction_plot_data_v2.csv"
    png_path = OUTPUT_DIR / "attribute_correct_direction_by_condition_v2.png"
    pdf_path = OUTPUT_DIR / "attribute_correct_direction_by_condition_v2.pdf"
    summary.to_csv(data_path, index=False)

    x = np.arange(len(FEATURES), dtype=float)
    offsets = {"none": -0.18, "attribution": 0.0, "counterfactual": 0.18}
    colors = {"none": "#6b7280", "attribution": "#0072b2", "counterfactual": "#d55e00"}
    markers = {"none": "o", "attribution": "s", "counterfactual": "^"}

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for condition in CONDITIONS:
        condition_rows = summary.loc[summary["condition"].eq(condition)]
        y = condition_rows["correct direction rate"].to_numpy(float)
        yerr = condition_rows["95% CI half-width"].to_numpy(float)
        ax.errorbar(
            x + offsets[condition],
            y,
            yerr=yerr,
            marker=markers[condition],
            color=colors[condition],
            linestyle="-",
            linewidth=1.6,
            capsize=4,
            label=condition,
        )
        for xpos, ypos, n in zip(x + offsets[condition], y, condition_rows["changed rows"]):
            ax.text(xpos, ypos + 0.045, f"n={int(n)}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Correct Direction Among Attribute Changes")
    ax.set_ylabel("Correct direction rate")
    ax.set_xlabel("Changed attribute")
    ax.set_xticks(x)
    ax.set_xticklabels([display for display, _, _ in FEATURES])
    ax.set_ylim(0.35, 1.02)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Explanation condition", frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    print(summary.to_string(index=False))
    print(png_path)
    print(pdf_path)
    print(data_path)


if __name__ == "__main__":
    main()
