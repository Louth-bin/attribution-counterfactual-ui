"""Print participant-level evidence for diabetes reasoning-strategy coding.

This is a read-only audit helper.  It combines the two open-ended Qualtrics
answers with the recorded testing edits and, for the attribution condition,
checks whether edited attributes overlap the two attributes shown by the
explanation for each instance.
"""

from __future__ import annotations

import csv
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "qualtrics" / "raw_output_v0.9.csv"
STATIC = ROOT / "static" / "experiment-data.json"
FEATURES = ("Blood Glucose", "Blood Pressure", "Insulin", "BMI", "Age")


def load_qualtrics(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))
    headers = rows[0]
    return [dict(zip(headers, row)) for row in rows[3:]]


def static_cases() -> dict[int, dict]:
    payload = json.loads(STATIC.read_text(encoding="utf-8"))
    cases: dict[int, dict] = {}
    dataset = payload["datasets"]["diabetes"]
    for phase in ("training_pool", "test_pool"):
        phase_data = dataset[phase]
        if isinstance(phase_data, dict):
            groups = phase_data.values()
        else:
            groups = (phase_data,)
        for group in groups:
            for case in group:
                cases[int(case["instance_id"])] = case
    return cases


def compact(text: str) -> str:
    return " ".join(text.split()) or "[blank]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args()
    cases = static_cases()
    participants = []
    for row in load_qualtrics(args.raw):
        raw_log = row.get("testing_log_json", "").strip()
        if not raw_log:
            continue
        logs = json.loads(raw_log)
        if not logs or logs[0].get("domain") != "diabetes":
            continue
        participants.append((row, logs))

    participants.sort(key=lambda item: (item[0]["xaiType"], item[0]["ResponseId"]))
    for row, logs in participants:
        feature_counts = Counter()
        feature_counts_by_direction: dict[str, Counter] = defaultdict(Counter)
        endpoint_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        subset_counts = Counter()
        edit_counts = Counter()
        attribution_cases_with_overlap = 0
        attribution_selected_total = 0
        attribution_selected_shown = 0

        for log in logs:
            direction = str(log["direction"])
            selected = tuple(change["attributeName"] for change in log["changes"])
            subset_counts[selected] += 1
            edit_counts[len(selected)] += 1
            selected_set = set(selected)
            for change in log["changes"]:
                feature = str(change["attributeName"])
                feature_counts[feature] += 1
                feature_counts_by_direction[direction][feature] += 1
                endpoint_values[(direction, feature)].append(float(change["newValue"]))

            if row["xaiType"] == "attribution":
                case = cases[int(log["instanceId"])]
                shown = {
                    case["feature_names"][int(index)]
                    for index in case["attribution"]["shown_feature_indices"][:2]
                }
                overlap = selected_set & shown
                attribution_cases_with_overlap += bool(overlap)
                attribution_selected_total += len(selected_set)
                attribution_selected_shown += len(overlap)

        print(f"\n{row['ResponseId']} [{row['xaiType']}]")
        print(
            "  Diagnosis answer: "
            + compact(row.get("PostTask_MentalModel", row.get("Q36", "")))
        )
        print(
            "  Change answer: "
            + compact(row.get("PostTask_StrategyExample", row.get("Q37", "")))
        )
        print(
            "  Number changed per case: "
            + ", ".join(f"{number}={count}" for number, count in sorted(edit_counts.items()))
        )
        print(
            "  Feature edit counts: "
            + ", ".join(f"{feature}={feature_counts[feature]}" for feature in FEATURES)
        )
        print(
            "  Most common subsets: "
            + "; ".join(
                f"{'+'.join(subset) or '[none]'}={count}"
                for subset, count in subset_counts.most_common(4)
            )
        )
        for direction in sorted(feature_counts_by_direction):
            counts = feature_counts_by_direction[direction]
            print(
                f"  {direction}: "
                + ", ".join(f"{feature}={counts[feature]}" for feature in FEATURES)
            )
            summaries = []
            for feature in FEATURES:
                values = endpoint_values.get((direction, feature), [])
                if values:
                    common = Counter(values).most_common(1)[0]
                    summaries.append(
                        f"{feature} median={median(values):g}, "
                        f"mode={common[0]:g} ({common[1]}/{len(values)}), "
                        f"range={min(values):g}-{max(values):g}"
                    )
            print("    Endpoints: " + "; ".join(summaries))
        if row["xaiType"] == "attribution":
            print(
                "  Attribution overlap: "
                f"any shown attribute edited in {attribution_cases_with_overlap}/{len(logs)} cases; "
                f"{attribution_selected_shown}/{attribution_selected_total} edited attributes were shown"
            )


if __name__ == "__main__":
    main()
