"""Compare qualitative confusion/strategy comments in v1.4 vs v2.0."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "qualtrics"

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
RANGE_TERMS = [
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
    "increase",
    "decrease",
]
CF_TERMS = ["counter", "changed to", "change to", "example", "training", "practice", "remember", "explanation"]
PRECISE_TERMS = ["threshold", "around", "range", "between", "over", "under", "near", "close", "enough"]


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def recover_records(version: str) -> pd.DataFrame:
    if version == "v1.4":
        processed = pd.read_csv(Q / "qualtrics_results_v1.4.csv", dtype=str)
        raw_paths = [
            Q / "raw_output_v1.2 (2 clusters).csv",
            Q / "raw_output_v1.3 (2 clusters 20 instances).csv",
            Q / "raw_output_v1.4 (2 clusters 20 instances).csv",
        ]
    elif version == "v2.0":
        processed = pd.read_csv(Q / "qualtrics_results_v2.0.csv", dtype=str)
        raw_paths = [Q / "raw_output_v1.9 (2 clusters new clustering).csv"]
    else:
        raise ValueError(version)

    condition_by_participant = processed.groupby("participant")["xai"].first().to_dict()
    participants = set(condition_by_participant)
    records = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        raw = pd.read_csv(raw_path, dtype=str)
        actual = raw[raw["ResponseId"].astype(str).str.startswith("R_", na=False)]
        for _, row in actual.iterrows():
            pid = str(row["ResponseId"])
            if pid not in participants or pid in seen:
                continue
            seen.add(pid)
            mental = str(row.get("PostTask_MentalModel", "") or "").strip()
            strategy = str(row.get("PostTask_StrategyExample", "") or "").strip()
            text = f"{mental} {strategy}".lower()
            records.append(
                {
                    "version": version,
                    "participant": pid,
                    "xai": condition_by_participant.get(pid, ""),
                    "raw_file": raw_path.name,
                    "mental_model": mental,
                    "strategy": strategy,
                    "uncertain_or_confused": any(t in text for t in CONFUSION_TERMS),
                    "vague_short": word_count(mental) < 8 or word_count(strategy) < 8,
                    "range_threshold_or_direction": any(t in text for t in RANGE_TERMS),
                    "mentions_training_example_cf": any(t in text for t in CF_TERMS),
                    "precise_boundary_language": any(t in text for t in PRECISE_TERMS),
                    "word_count_total": word_count(text),
                }
            )
    out = pd.DataFrame(records)
    missing = participants - seen
    print(version, "processed participants", len(participants), "comments recovered", len(out), "missing", len(missing))
    if missing:
        print("missing", sorted(missing)[:20])
    return out


def main() -> None:
    both = pd.concat([recover_records("v1.4"), recover_records("v2.0")], ignore_index=True)
    flags = [
        "uncertain_or_confused",
        "vague_short",
        "range_threshold_or_direction",
        "mentions_training_example_cf",
        "precise_boundary_language",
    ]
    print()
    print("Overall by version")
    overall = both.groupby("version").agg(
        n=("participant", "size"),
        uncertain_or_confused=("uncertain_or_confused", "mean"),
        vague_short=("vague_short", "mean"),
        range_threshold_or_direction=("range_threshold_or_direction", "mean"),
        mentions_training_example_cf=("mentions_training_example_cf", "mean"),
        precise_boundary_language=("precise_boundary_language", "mean"),
        mean_words=("word_count_total", "mean"),
    )
    print(overall.round(3).to_string())
    print()
    print("By version and condition")
    by = both.groupby(["version", "xai"]).agg(
        n=("participant", "size"),
        uncertain_or_confused=("uncertain_or_confused", "sum"),
        vague_short=("vague_short", "sum"),
        range_threshold_or_direction=("range_threshold_or_direction", "sum"),
        mentions_training_example_cf=("mentions_training_example_cf", "sum"),
        precise_boundary_language=("precise_boundary_language", "sum"),
        mean_words=("word_count_total", "mean"),
    )
    print(by.round(1).to_string())
    print()
    print("Counterfactual condition only: flagged/uncertain examples")
    cf = both[both["xai"].eq("counterfactual")]
    examples = cf[cf["uncertain_or_confused"] | cf["vague_short"]].sort_values(["version", "participant"])
    for r in examples.to_dict("records"):
        print()
        print(f"--- {r['version']} {r['participant']} | uncertain={r['uncertain_or_confused']} vague={r['vague_short']}")
        print("Mental:", r["mental_model"])
        print("Strategy:", r["strategy"])
    print()
    print("Counterfactual condition only: non-flagged representative strategy snippets")
    for version in ["v1.4", "v2.0"]:
        sample = cf[(cf["version"].eq(version)) & ~(cf["uncertain_or_confused"] | cf["vague_short"])].head(6)
        print()
        print(version)
        for r in sample.to_dict("records"):
            print(f"- {r['participant']}: {r['strategy'][:220]}")


if __name__ == "__main__":
    main()
