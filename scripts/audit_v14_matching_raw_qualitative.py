"""Find raw files matching v1.4 processed results and audit qualitative responses."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "qualtrics"
PROCESSED = Q / "qualtrics_results_v1.4.csv"
RAW_CANDIDATES = [
    Q / "raw_output_v1.2 (2 clusters).csv",
    Q / "raw_output_v1.3 (2 clusters 20 instances).csv",
    Q / "raw_output_v1.4 (2 clusters 20 instances).csv",
]


CONFUSION_TERMS = [
    "confus",
    "guess",
    "random",
    "not sure",
    "unsure",
    "no idea",
    "did not understand",
    "dont understand",
    "don't understand",
    "hard to tell",
    "could not",
    "couldn't",
    "unclear",
    "struggl",
    "difficult",
    "hard",
]


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def main() -> None:
    processed = pd.read_csv(PROCESSED, dtype=str)
    condition_by_participant = processed.groupby("participant")["xai"].first().to_dict()
    participants = set(condition_by_participant)

    print(f"Processed v1.4 participants: {len(participants)}")
    print("Processed v1.4 conditions:", processed.groupby("xai")["participant"].nunique().to_dict())
    print()
    print("Raw-file overlap with processed v1.4")
    for raw_path in RAW_CANDIDATES:
        raw = pd.read_csv(raw_path, dtype=str)
        actual = raw[raw["ResponseId"].astype(str).str.startswith("R_", na=False)]
        overlap = set(actual["ResponseId"].dropna().astype(str)) & participants
        print(f"{raw_path.name}: raw actual={len(actual)}, overlap={len(overlap)}")

    # The exact v1.4 processed participant set is covered by the v1.2 raw cohort
    # plus the v1.3/1.4 raw cohort. Use earliest matching raw file per participant.
    records = []
    seen: set[str] = set()
    for raw_path in RAW_CANDIDATES:
        raw = pd.read_csv(raw_path, dtype=str)
        actual = raw[raw["ResponseId"].astype(str).str.startswith("R_", na=False)]
        for _, row in actual.iterrows():
            participant = str(row["ResponseId"])
            if participant not in participants or participant in seen:
                continue
            seen.add(participant)
            mental_model = str(row.get("PostTask_MentalModel", "") or "").strip()
            strategy = str(row.get("PostTask_StrategyExample", "") or "").strip()
            text = f"{mental_model} {strategy}".lower()
            records.append(
                {
                    "participant": participant,
                    "xai": condition_by_participant.get(participant, ""),
                    "raw_file": raw_path.name,
                    "mental_model": mental_model,
                    "strategy": strategy,
                    "uncertain_or_confused": any(term in text for term in CONFUSION_TERMS),
                    "vague_short": word_count(mental_model) < 8 or word_count(strategy) < 8,
                    "mentions_training_example_or_memory": any(
                        term in text
                        for term in ["counter", "example", "training", "practice", "remember", "explanation"]
                    ),
                    "range_or_threshold_language": any(
                        term in text
                        for term in [
                            "threshold",
                            "range",
                            "ranges",
                            "normal",
                            "optimum",
                            "around",
                            "over",
                            "under",
                            "higher",
                            "lower",
                            "high",
                            "low",
                        ]
                    ),
                }
            )

    out = pd.DataFrame(records).sort_values(["xai", "participant"])
    print()
    print(f"Qualitative records recovered: {len(out)}")
    print(f"Missing participants: {sorted(participants - seen)}")
    print()
    print("Summary by condition")
    print(
        out.groupby("xai")
        .agg(
            n=("participant", "size"),
            uncertain_or_confused=("uncertain_or_confused", "sum"),
            vague_short=("vague_short", "sum"),
            mentions_training_example_or_memory=("mentions_training_example_or_memory", "sum"),
            range_or_threshold_language=("range_or_threshold_language", "sum"),
        )
        .to_string()
    )
    print()
    print("All matched qualitative responses")
    for record in out.to_dict("records"):
        print()
        print(
            f"--- {record['participant']} | {record['xai']} | {record['raw_file']} | "
            f"uncertain={record['uncertain_or_confused']} | vague={record['vague_short']}"
        )
        print(f"Mental model: {record['mental_model']}")
        print(f"Strategy: {record['strategy']}")


if __name__ == "__main__":
    main()
