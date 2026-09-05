"""Merge independently sampled cognitive-model v0.2 winners into v2.3 rows."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


COLUMN_MAP = {
    "cognitive model v0.2 revision": None,
    "cognitive model v0.2 family": "variant",
    "cognitive model v0.2 representation": "representation",
    "cognitive model v0.2 scope": "scope",
    "cognitive model v0.2 global feature lapse": "lapse_global",
    "cognitive model v0.2 memory decay": "decay",
    "cognitive model v0.2 exemplar distance sensitivity": "distance",
    "cognitive model v0.2 selection spread": "selection_spread",
    "cognitive model v0.2 counterfactual edit blend": "edit_blend",
    "cognitive model v0.2 age actionable (0/1)": "age_actionable",
    "cognitive model v0.2 edit margin": "margin",
    "cognitive model v0.2 CV subset NLL": "cv_selection_nll",
    "cognitive model v0.2 CV expected selection F1": "cv_expected_selection_f1",
    "cognitive model v0.2 CV MAP selection F1": "cv_map_selection_f1",
    "cognitive model v0.2 CV exact-subset accuracy": "cv_exact_subset_accuracy",
    "cognitive model v0.2 CV cardinality MAE": "cv_cardinality_mae",
    "cognitive model v0.2 CV conditional amount MAE": "cv_amount_mae",
    "cognitive model v0.2 CV conditional direction accuracy": "cv_direction_accuracy",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    columns, rows = read_csv(arguments.results)
    _, fit_rows = read_csv(arguments.fits)
    winners = [row for row in fit_rows if int(float(row["selected_by_cv"])) == 1]
    winners_by_participant = {row["participant"]: row for row in winners}
    participants = {row["participant"] for row in rows}

    if len(rows) != 1920 or len(participants) != 60:
        raise RuntimeError(
            f"Unexpected v2.3 dimensions: rows={len(rows)}, participants={len(participants)}"
        )
    if len(winners) != 60 or set(winners_by_participant) != participants:
        raise RuntimeError("Expected exactly one v0.2 winner for every v2.3 participant")

    for row in rows:
        fit = winners_by_participant[row["participant"]]
        if row["xai"] != fit["xai"]:
            raise RuntimeError(f"Condition mismatch for {row['participant']}")
        for output_column, fit_column in COLUMN_MAP.items():
            if output_column == "cognitive model v0.2 revision":
                value = "independent-feature"
            elif (
                output_column == "cognitive model v0.2 counterfactual edit blend"
                and fit["representation"] != "counterfactual"
            ):
                value = ""
            else:
                value = cleaned(fit[fit_column])
            row[output_column] = value

    output_columns = columns + [column for column in COLUMN_MAP if column not in columns]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    reread_columns, reread_rows = read_csv(arguments.output)
    if len(reread_rows) != len(rows) or reread_columns != output_columns:
        raise RuntimeError("Written CSV failed dimension or schema verification")
    populated = {
        row["participant"] for row in reread_rows
        if row["cognitive model v0.2 family"]
    }
    if populated != participants:
        raise RuntimeError("Not every participant received a v0.2 winner")
    print(
        f"output={arguments.output} rows={len(rows)} participants={len(participants)} "
        f"columns_added={len([column for column in COLUMN_MAP if column not in columns])}"
    )


if __name__ == "__main__":
    main()
