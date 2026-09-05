"""Filter v1.9 participants by repeated non-increase in target confidence."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs" / "v19-probability-weighted-36"
TRIAL_INPUT = SOURCE_DIR / "probability_weighted_trial_metrics.csv"
PARTICIPANT_INPUT = SOURCE_DIR / "probability_weighted_participant_metrics.csv"
OUTPUT_DIR = SOURCE_DIR / "target-confidence-filter"
THRESHOLD = 8


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_columns, trials = read(TRIAL_INPUT)
    participant_columns, participants = read(PARTICIPANT_INPUT)
    by_participant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trials:
        by_participant[row["participant"]].append(row)

    nonincrease_counts = {
        participant: sum(
            float(row["observed target confidence gain"]) <= 0.0 for row in rows
        )
        for participant, rows in by_participant.items()
    }
    excluded = {
        participant
        for participant, count in nonincrease_counts.items()
        if count >= THRESHOLD
    }
    retained = set(by_participant) - excluded
    if any(len(rows) != 20 for rows in by_participant.values()):
        raise RuntimeError("Expected 20 testing trials per participant")

    retained_participants = []
    excluded_participants = []
    for row in participants:
        participant = row["participant"]
        enriched = {
            **row,
            "observed target-confidence non-increase trials": nonincrease_counts[participant],
            "observed target-confidence increase rate": (
                20 - nonincrease_counts[participant]
            ) / 20,
        }
        if participant in retained:
            retained_participants.append(enriched)
        else:
            excluded_participants.append(enriched)
    retained_trials = [row for row in trials if row["participant"] in retained]
    added_columns = [
        "observed target-confidence non-increase trials",
        "observed target-confidence increase rate",
    ]
    write(
        OUTPUT_DIR / "retained_participant_metrics.csv",
        participant_columns + added_columns,
        retained_participants,
    )
    write(
        OUTPUT_DIR / "excluded_participants.csv",
        participant_columns + added_columns,
        excluded_participants,
    )
    write(OUTPUT_DIR / "retained_trial_metrics.csv", trial_columns, retained_trials)

    summary = {
        "exclusion rule": (
            "exclude when observed target-confidence gain is <= 0 on at least "
            f"{THRESHOLD} of 20 testing trials"
        ),
        "full participants": len(participants),
        "retained participants": len(retained),
        "excluded participants": len(excluded),
        "retained trials": len(retained_trials),
        "retained condition counts": dict(
            sorted(Counter(row["xai"] for row in retained_participants).items())
        ),
        "excluded condition counts": dict(
            sorted(Counter(row["xai"] for row in excluded_participants).items())
        ),
        "excluded participant details": [
            {
                "participant": row["participant"],
                "xai": row["xai"],
                "non-increase trials": int(
                    row["observed target-confidence non-increase trials"]
                ),
                "increase rate": float(
                    row["observed target-confidence increase rate"]
                ),
            }
            for row in excluded_participants
        ],
    }
    (OUTPUT_DIR / "filter_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
