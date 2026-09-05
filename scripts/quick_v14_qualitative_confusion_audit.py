"""Audit v1.4 raw qualitative responses for saved participants."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "qualtrics" / "raw_output_v1.4 (2 clusters 20 instances).csv"
PROCESSED = ROOT / "qualtrics" / "qualtrics_results_v1.4.csv"


def main() -> None:
    raw = pd.read_csv(RAW, dtype=str)
    processed = pd.read_csv(PROCESSED, dtype=str)

    raw_actual = raw[raw["ResponseId"].astype(str).str.startswith("R_", na=False)].copy()
    saved = set(processed["participant"].dropna().astype(str))
    raw_actual["saved_in_v14"] = raw_actual["ResponseId"].isin(saved)
    xai_by_participant = processed.groupby("participant")["xai"].first().to_dict()

    confusion_terms = [
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
    ]

    rows = []
    for _, row in raw_actual[raw_actual["saved_in_v14"]].iterrows():
        pid = str(row["ResponseId"])
        mental_model = str(row.get("PostTask_MentalModel", "") or "").strip()
        strategy = str(row.get("PostTask_StrategyExample", "") or "").strip()
        text = f"{mental_model} {strategy}".lower()
        confusion_flag = any(term in text for term in confusion_terms)
        vague_short_flag = len(re.findall(r"\w+", mental_model)) < 8 or len(re.findall(r"\w+", strategy)) < 8
        rows.append(
            {
                "participant": pid,
                "xai": xai_by_participant.get(pid, ""),
                "mental_model": mental_model,
                "strategy": strategy,
                "confusion_flag": confusion_flag,
                "vague_short_flag": vague_short_flag,
            }
        )

    out = pd.DataFrame(rows).sort_values(["xai", "participant"])
    print(
        f"raw actual rows={len(raw_actual)}; saved raw participants={raw_actual['saved_in_v14'].sum()}; "
        f"processed v1.4 participants={len(saved)}"
    )
    print()
    print("Summary by condition")
    print(
        out.groupby("xai")
        .agg(
            n=("participant", "size"),
            confusion_flags=("confusion_flag", "sum"),
            vague_short_flags=("vague_short_flag", "sum"),
        )
        .to_string()
    )
    print()
    print("Saved v1.4 raw qualitative responses")
    for record in out.to_dict("records"):
        print()
        print(
            f"--- {record['participant']} | {record['xai']} | "
            f"confusion={record['confusion_flag']} | vague_short={record['vague_short_flag']}"
        )
        print(f"Mental model: {record['mental_model']}")
        print(f"Strategy: {record['strategy']}")


if __name__ == "__main__":
    main()
