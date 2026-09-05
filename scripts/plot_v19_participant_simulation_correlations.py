"""Create paper-ready observed-versus-simulated performance plots for v1.9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "qualtrics" / "v1.9_latest_model_simulation_participants.csv"
OUTPUT_DIR = ROOT / "outputs" / "v19-additive-rho-plots"
PNG_OUTPUT = OUTPUT_DIR / "participant_simulation_correlations.png"
PDF_OUTPUT = OUTPUT_DIR / "participant_simulation_correlations.pdf"
SUMMARY_OUTPUT = OUTPUT_DIR / "participant_simulation_correlations.json"

METRICS = (
    ("success", "Counterfactual success rate"),
    ("target confidence gain", "Target-confidence gain"),
    ("boundary improvement", "Boundary-distance improvement"),
    ("plausibility", "Plausibility"),
    ("actionability", "Actionability rate"),
    ("edit L1", "Normalized edit distance (L1)"),
)

CONDITIONS = {
    "none": ("#0072B2", "o", "None"),
    "attribution": ("#D55E00", "s", "Attribution"),
    "counterfactual": ("#009E73", "^", "Counterfactual"),
}


def limits(observed: np.ndarray, simulated: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([observed, simulated])
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, 0.05)
    padding = 0.10 * span
    return low - padding, high + padding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--exclude-participant", action="append", default=[])
    parser.add_argument("--filename-prefix", default="")
    parser.add_argument("--figure-title", default="")
    parser.add_argument(
        "--model-label",
        default="strict conditional additive-rho probabilistic attribute-selection model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    if args.exclude_participant:
        data = data.loc[~data["participant"].isin(args.exclude_participant)].copy()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    png_output = output_dir / f"{args.filename_prefix}{PNG_OUTPUT.name}"
    pdf_output = output_dir / f"{args.filename_prefix}{PDF_OUTPUT.name}"
    summary_output = output_dir / f"{args.filename_prefix}{SUMMARY_OUTPUT.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(10.2, 7.3), constrained_layout=False)
    figure.subplots_adjust(
        left=0.07, right=0.99, bottom=0.08, top=0.82, wspace=0.26, hspace=0.34
    )
    if args.figure_title:
        figure.suptitle(args.figure_title, fontsize=12, y=0.965)
    summaries = {}

    for axis, (column, title) in zip(axes.flat, METRICS):
        observed = data[f"observed {column}"].to_numpy(dtype=float)
        simulated = data[f"simulated {column}"].to_numpy(dtype=float)
        pearson = stats.pearsonr(observed, simulated)
        spearman = stats.spearmanr(observed, simulated)
        mae = float(np.mean(np.abs(observed - simulated)))
        summaries[column] = {
            "label": title,
            "n": int(len(data)),
            "observed mean": float(np.mean(observed)),
            "simulated mean": float(np.mean(simulated)),
            "pearson r": float(pearson.statistic),
            "pearson p": float(pearson.pvalue),
            "spearman rho": float(spearman.statistic),
            "spearman p": float(spearman.pvalue),
            "MAE": mae,
        }

        low, high = limits(observed, simulated)
        axis.plot(
            [low, high], [low, high],
            color="#777777", linewidth=1.0, linestyle="--", zorder=1,
            label="Perfect agreement",
        )
        if float(np.ptp(observed)) > 1e-12:
            slope, intercept = np.polyfit(observed, simulated, 1)
            grid = np.linspace(low, high, 100)
            axis.plot(
                grid, intercept + slope * grid,
                color="#222222", linewidth=1.2, zorder=2,
                label="Linear fit",
            )

        for condition, (color, marker, label) in CONDITIONS.items():
            subset = data["xai"] == condition
            axis.scatter(
                observed[subset], simulated[subset],
                s=34, marker=marker, color=color, edgecolor="white",
                linewidth=0.5, alpha=0.90, zorder=3, label=label,
            )

        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#DDDDDD", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.set_title(title, pad=6)
        axis.set_xlabel("Observed participant performance")
        axis.set_ylabel("Simulated performance")
        axis.text(
            0.04, 0.96,
            f"Pearson r = {pearson.statistic:.2f}\np = {pearson.pvalue:.3g}\nMAE = {mae:.3f}",
            transform=axis.transAxes, va="top", ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(
        by_label.values(), by_label.keys(),
        loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5, frameon=False,
    )
    figure.savefig(png_output, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_output, bbox_inches="tight")
    plt.close(figure)

    summary_output.write_text(
        json.dumps(
            {
                "model": args.model_label,
                "participants": int(data["participant"].nunique()),
                "excluded participants": args.exclude_participant,
                "metrics": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(png_output)
    print(pdf_output)
    print(summary_output)


if __name__ == "__main__":
    main()
