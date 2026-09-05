"""Quick exploratory boundary diagnostics for v2.0.

This script intentionally leaves the original Qualtrics CSV untouched. It adds
nearest-training-distance to a copy and writes small instance/participant
summaries for fast interactive discussion.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
OUTPUT_DIR = ROOT / "outputs" / "v20-quick-boundary-diagnostics"
FEATURES = ("glucose", "blood_pressure", "insulin", "bmi", "age")
NORMALIZED_VALUE_RE = re.compile(
    r"\((-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)"
)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def numeric(row: dict[str, str], column: str) -> float:
    value = row.get(column, "")
    return float(value) if value not in ("", "nan", "NaN") else float("nan")


def normalized_profile(attribute_text: str, which: str = "last") -> np.ndarray:
    values: list[float] = []
    for line in attribute_text.splitlines():
        if not line.strip():
            continue
        matches = NORMALIZED_VALUE_RE.findall(line)
        if matches:
            values.append(float(matches[0] if which == "first" else matches[-1]))
    if len(values) != 5:
        raise ValueError(f"Expected five normalized values, found {len(values)}")
    return np.asarray(values, dtype=float)


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else float("nan")


def median(values: list[float]) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return float("nan")
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2


def sd(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return float("nan")
    return float(np.std(clean, ddof=1))


def sem(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return float("nan")
    return sd(clean) / math.sqrt(len(clean))


def welch_z(group_a: list[float], group_b: list[float]) -> tuple[float, float]:
    a = [value for value in group_a if math.isfinite(value)]
    b = [value for value in group_b if math.isfinite(value)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    diff = mean(a) - mean(b)
    se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    if se == 0:
        return diff, float("nan")
    z = diff / se
    p = 2 * (1 - NormalDist().cdf(abs(z)))
    return diff, p


def summarize_group(rows: list[dict[str, str]]) -> dict[str, object]:
    testing = [row for row in rows if row["phase"] == "testing"]
    return {
        "rows": len(testing),
        "participants": len({row["participant"] for row in testing}),
        "mean boundary distance new": mean(
            [numeric(row, "boundary distance new") for row in testing]
        ),
        "median boundary distance new": median(
            [numeric(row, "boundary distance new") for row in testing]
        ),
        "mean boundary distance change (new - original)": mean(
            [numeric(row, "boundary distance change (new - original)") for row in testing]
        ),
        "success rate": mean(
            [numeric(row, "successful counterfactual (0/1)") for row in testing]
        ),
        "target-directed rate": mean(
            [numeric(row, "move towards target (0/1)") for row in testing]
        ),
        "mean normalized edit distance": mean([numeric(row, "proximity") for row in testing]),
        "mean target confidence gain": mean(
            [numeric(row, "delta confidence of target label") for row in testing]
        ),
    }


def main() -> None:
    columns, rows = read_rows(INPUT)
    training_by_participant: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if row["phase"] == "training":
            training_by_participant[row["participant"]].append(
                normalized_profile(row["attribute values before and after"])
            )

    enriched: list[dict[str, object]] = []
    testing_rows: list[dict[str, str]] = []
    for row in rows:
        output = dict(row)
        if row["phase"] == "testing":
            profile = normalized_profile(row["attribute values before and after"], which="first")
            training_profiles = training_by_participant[row["participant"]]
            distances = [float(np.sum(np.abs(profile - train))) for train in training_profiles]
            nearest = min(distances) if distances else float("nan")
            output["nearest training instance distance (L1 normalized)"] = scalar(nearest)
            row["nearest training instance distance (L1 normalized)"] = scalar(nearest)
            testing_rows.append(row)
        else:
            output["nearest training instance distance (L1 normalized)"] = ""
        enriched.append(output)

    added_column = "nearest training instance distance (L1 normalized)"
    write_rows(OUTPUT_DIR / "qualtrics_results_v2.0_with_nearest_training_distance.csv",
               columns + [added_column], enriched)

    by_instance: list[dict[str, object]] = []
    for instance_id in sorted({row["instance id"] for row in testing_rows}, key=lambda x: int(float(x))):
        instance_rows = [row for row in testing_rows if row["instance id"] == instance_id]
        record = {
            "instance id": instance_id,
            "case": instance_rows[0]["case"],
            "original label": instance_rows[0]["original label"],
            "target label": instance_rows[0]["target label"],
            "participants": len({row["participant"] for row in instance_rows}),
            "nearest training distance": mean(
                [numeric(row, added_column) for row in instance_rows]
            ),
            **summarize_group(instance_rows),
        }
        by_xai = {}
        for xai in ("counterfactual", "attribution", "none"):
            xai_rows = [row for row in instance_rows if row["xai"] == xai]
            by_xai[xai] = [numeric(row, "boundary distance new") for row in xai_rows]
            record[f"{xai} mean boundary distance new"] = mean(by_xai[xai])
            record[f"{xai} success rate"] = mean(
                [numeric(row, "successful counterfactual (0/1)") for row in xai_rows]
            )
            record[f"{xai} target-directed rate"] = mean(
                [numeric(row, "move towards target (0/1)") for row in xai_rows]
            )
        record["CF minus attribution boundary distance"] = mean(by_xai["counterfactual"]) - mean(by_xai["attribution"])
        record["CF minus none boundary distance"] = mean(by_xai["counterfactual"]) - mean(by_xai["none"])
        by_instance.append(record)
    by_instance.sort(key=lambda row: row["mean boundary distance new"])
    instance_columns = list(by_instance[0].keys())
    write_rows(OUTPUT_DIR / "instance_boundary_summary.csv", instance_columns, by_instance)

    by_participant: list[dict[str, object]] = []
    for participant in sorted({row["participant"] for row in rows}):
        participant_rows = [row for row in rows if row["participant"] == participant]
        testing = [row for row in participant_rows if row["phase"] == "testing"]
        changed_counts = [
            sum(numeric(row, f"x_{index}_changed") for index in range(1, 6))
            for row in testing
        ]
        feature_change_means = {
            f"mean abs normalized change: {feature}": mean(
                [abs(numeric(row, f"x_{index}_change")) for row in testing]
            )
            for index, feature in enumerate(FEATURES, start=1)
        }
        record = {
            "participant": participant,
            "xai": participant_rows[0]["xai"],
            "reasoning strategy": participant_rows[0].get("reasoning strategy", ""),
            "cognitive model family": participant_rows[0].get("cognitive model v0.1 family", ""),
            "explanation reliance eta": participant_rows[0].get("explanation reliance (η)", ""),
            "global relevance reliance alpha": participant_rows[0].get("global relevance reliance (α)", ""),
            "additive change margin rho": participant_rows[0].get("additive change margin (ρ)", ""),
            "participant training accuracy": participant_rows[0].get("participant training accuracy", ""),
            "mean changed features": mean(changed_counts),
            **summarize_group(testing),
            **feature_change_means,
        }
        by_participant.append(record)
    by_participant.sort(key=lambda row: row["mean boundary distance new"])
    participant_columns = list(by_participant[0].keys())
    write_rows(OUTPUT_DIR / "participant_boundary_summary.csv", participant_columns, by_participant)

    distance_values = [numeric(row, added_column) for row in testing_rows]
    threshold = median(distance_values)
    near_rows = [row for row in testing_rows if numeric(row, added_column) <= threshold]
    far_rows = [row for row in testing_rows if numeric(row, added_column) > threshold]
    near_far_records: list[dict[str, object]] = []
    for bucket, bucket_rows in (("near training", near_rows), ("far from training", far_rows)):
        for xai in ("counterfactual", "attribution", "none"):
            xai_rows = [row for row in bucket_rows if row["xai"] == xai]
            near_far_records.append({"training-distance bucket": bucket, "xai": xai, **summarize_group(xai_rows)})
    write_rows(OUTPUT_DIR / "nearest_training_by_condition_summary.csv",
               list(near_far_records[0].keys()), near_far_records)

    condition_rows: list[dict[str, object]] = []
    condition_values: dict[str, list[float]] = {}
    for xai in ("counterfactual", "attribution", "none"):
        xai_testing = [row for row in testing_rows if row["xai"] == xai]
        condition_values[xai] = [numeric(row, "boundary distance new") for row in xai_testing]
        condition_rows.append({"xai": xai, **summarize_group(xai_testing)})
    write_rows(OUTPUT_DIR / "condition_boundary_summary.csv", list(condition_rows[0].keys()), condition_rows)

    cf_attr_diff, cf_attr_p = welch_z(condition_values["counterfactual"], condition_values["attribution"])
    cf_none_diff, cf_none_p = welch_z(condition_values["counterfactual"], condition_values["none"])

    close_participants = by_participant[:10]
    poor_participants = by_participant[-10:]
    close_strategy_counts = Counter(row["reasoning strategy"] for row in close_participants)
    poor_strategy_counts = Counter(row["reasoning strategy"] for row in poor_participants)
    close_xai_counts = Counter(row["xai"] for row in close_participants)
    poor_xai_counts = Counter(row["xai"] for row in poor_participants)

    distance = np.asarray([numeric(row, added_column) for row in testing_rows], dtype=float)
    boundary = np.asarray([numeric(row, "boundary distance new") for row in testing_rows], dtype=float)
    success = np.asarray([numeric(row, "successful counterfactual (0/1)") for row in testing_rows], dtype=float)
    finite = np.isfinite(distance) & np.isfinite(boundary)
    distance_boundary_corr = float(np.corrcoef(distance[finite], boundary[finite])[0, 1])
    finite_success = np.isfinite(distance) & np.isfinite(success)
    distance_success_corr = float(np.corrcoef(distance[finite_success], success[finite_success])[0, 1])

    summary = {
        "scope": "quick exploratory v2.0 testing rows",
        "participants": len({row["participant"] for row in rows}),
        "testing rows": len(testing_rows),
        "condition summary": condition_rows,
        "boundary distance lower is better": True,
        "close participant definition": "lowest mean boundary distance new across testing rows",
        "top 10 close participant xai counts": dict(close_xai_counts),
        "bottom 10 participant xai counts": dict(poor_xai_counts),
        "top 10 close participant strategy counts": dict(close_strategy_counts),
        "bottom 10 participant strategy counts": dict(poor_strategy_counts),
        "median nearest training distance": threshold,
        "nearest training distance vs boundary distance correlation": distance_boundary_corr,
        "nearest training distance vs success correlation": distance_success_corr,
        "CF minus attribution boundary distance quick Welch z approximation": {
            "difference": cf_attr_diff,
            "p": cf_attr_p,
            "interpretation": "negative favors counterfactual because lower boundary distance is better",
        },
        "CF minus none boundary distance quick Welch z approximation": {
            "difference": cf_none_diff,
            "p": cf_none_p,
            "interpretation": "negative favors counterfactual because lower boundary distance is better",
        },
        "best instances by mean boundary distance": by_instance[:5],
        "worst instances by mean boundary distance": by_instance[-5:],
        "outputs": {
            "enriched rows": str((OUTPUT_DIR / "qualtrics_results_v2.0_with_nearest_training_distance.csv").relative_to(ROOT)),
            "instance summary": str((OUTPUT_DIR / "instance_boundary_summary.csv").relative_to(ROOT)),
            "participant summary": str((OUTPUT_DIR / "participant_boundary_summary.csv").relative_to(ROOT)),
            "nearest training summary": str((OUTPUT_DIR / "nearest_training_by_condition_summary.csv").relative_to(ROOT)),
            "condition summary": str((OUTPUT_DIR / "condition_boundary_summary.csv").relative_to(ROOT)),
        },
    }
    (OUTPUT_DIR / "quick_boundary_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
