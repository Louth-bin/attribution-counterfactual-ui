"""Add winning cognitive-model v0.1 parameters to every v2.0 result row."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
FITS = ROOT / "qualtrics" / "v20_cognitive_model_v01_fits.csv"
SUMMARY = ROOT / "qualtrics" / "v20_cognitive_model_v01_parameters_summary.json"

COLUMN_MAP = {
    "cognitive model v0.1 family": "model family",
    "explanation reliance (η)": "eta",
    "global relevance reliance (α)": "alpha",
    "additive change margin (ρ)": "rho",
    "exemplar locality (λ)": "lambda",
    "remembered-change reliance (β)": "beta",
    "age treated as actionable (0/1)": "age actionable",
}

PARAMETER_DESCRIPTIONS = {
    "explanation reliance (η)": (
        "Balance between explanation-derived relevance (1) and label-based "
        "diagnosticity (0); fixed at 0 in the no-explanation condition."
    ),
    "global relevance reliance (α)": (
        "Feature-contribution family only: balance between global learned "
        "relevance (1) and instance-local opposition (0)."
    ),
    "additive change margin (ρ)": (
        "Extra range-normalized movement added in the predicted direction for "
        "each selected feature."
    ),
    "exemplar locality (λ)": (
        "Weighted-examples family only: higher values concentrate retrieval on "
        "more similar remembered examples."
    ),
    "remembered-change reliance (β)": (
        "Weighted-examples family: balance between copying remembered change "
        "vectors (1) and moving toward remembered endpoints (0)."
    ),
    "age treated as actionable (0/1)": (
        "Whether Age was allowed among candidate edited features in the fitted model."
    ),
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def cleaned(value: str) -> str:
    try:
        return "" if math.isnan(float(value)) else value
    except (TypeError, ValueError):
        return value


def main() -> None:
    columns, rows = read_csv(RESULTS)
    _, fit_rows = read_csv(FITS)
    winners = [row for row in fit_rows if row["selected family by CV"] == "1"]
    winner_counts = Counter(row["participant"] for row in winners)
    if set(winner_counts.values()) != {1}:
        raise RuntimeError(f"Expected one winning family per participant: {winner_counts}")
    fits_by_participant = {row["participant"]: row for row in winners}
    result_participants = {row["participant"] for row in rows}
    if result_participants != set(fits_by_participant):
        raise RuntimeError(
            "Result and fit participant sets differ: "
            f"missing fits={sorted(result_participants - set(fits_by_participant))}, "
            f"unexpected fits={sorted(set(fits_by_participant) - result_participants)}"
        )

    output_columns = [column for column in columns if column not in COLUMN_MAP]
    output_columns.extend(COLUMN_MAP)
    for row in rows:
        fit = fits_by_participant[row["participant"]]
        for output_name, fit_name in COLUMN_MAP.items():
            row[output_name] = cleaned(fit[fit_name])

    with RESULTS.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=output_columns,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_version": "cognitive model v0.1",
        "responses": str(RESULTS.relative_to(ROOT)),
        "detailed_fits": str(FITS.relative_to(ROOT)),
        "participants": len(result_participants),
        "rows": len(rows),
        "winning_model_families": dict(
            sorted(Counter(row["model family"] for row in winners).items())
        ),
        "winning_model_families_by_xai": {
            xai: dict(
                sorted(
                    Counter(
                        row["model family"] for row in winners if row["xai"] == xai
                    ).items()
                )
            )
            for xai in sorted({row["xai"] for row in winners})
        },
        "columns_added": list(COLUMN_MAP),
        "parameter_descriptions": PARAMETER_DESCRIPTIONS,
        "selection_rule": (
            "Minimum five-fold cross-validated selection NLL, then conditional "
            "amount MAE, as frozen in cognitive model v0.1."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
