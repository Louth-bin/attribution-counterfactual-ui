"""Populate v0.1 winning cognitive-model parameters in v2.3 results."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "qualtrics" / "qualtrics_results_v2.3.csv"
FITS = ROOT / "qualtrics" / "v2.3_cognitive_model_v0.1_fits.csv"

COLUMN_MAP = {
    "cognitive model v0.1 family": "model family",
    "explanation reliance (η)": "eta",
    "global relevance reliance (α)": "alpha",
    "additive change margin (ρ)": "rho",
    "exemplar locality (λ)": "lambda",
    "remembered-change reliance (β)": "beta",
    "age treated as actionable (0/1)": "age actionable",
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
    _, fits = read_csv(FITS)
    winners = [row for row in fits if int(float(row["selected family by CV"])) == 1]
    winners_by_participant = {row["participant"]: row for row in winners}
    participants = {row["participant"] for row in rows}

    if len(rows) != 1920 or len(participants) != 60:
        raise RuntimeError(f"Unexpected v2.3 dimensions: rows={len(rows)}, participants={len(participants)}")
    if len(winners) != 60 or set(winners_by_participant) != participants:
        raise RuntimeError("Expected exactly one fitted winner for every v2.3 participant")
    for row in rows:
        fit = winners_by_participant[row["participant"]]
        if row["xai"] != fit["xai"]:
            raise RuntimeError(f"Condition mismatch for {row['participant']}")
        for output_column, fit_column in COLUMN_MAP.items():
            row[output_column] = cleaned(fit[fit_column])

    missing_columns = [column for column in COLUMN_MAP if column not in columns]
    output_columns = columns + missing_columns
    temporary = RESULTS.with_suffix(".cognitive-refit.tmp.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESULTS)
    print(f"updated={RESULTS} rows={len(rows)} participants={len(participants)}")


if __name__ == "__main__":
    main()
