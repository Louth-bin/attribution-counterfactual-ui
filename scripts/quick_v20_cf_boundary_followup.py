"""Follow-up diagnostics for CF nearest-training and participant boundary groups."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "v20-quick-boundary-diagnostics" / "qualtrics_results_v2.0_with_nearest_training_distance.csv"
OUTPUT_DIR = ROOT / "outputs" / "v20-quick-boundary-diagnostics"
DIST = "nearest training instance distance (L1 normalized)"
BOUNDARY = "boundary distance new"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def num(row: dict[str, str], column: str) -> float:
    value = row.get(column, "")
    return float(value) if value not in ("", "nan", "NaN") else float("nan")


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(np.mean(clean)) if clean else float("nan")


def sd(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan")


def corr(rows: list[dict[str, str]], x: str, y: str) -> float:
    xs = np.asarray([num(row, x) for row in rows], dtype=float)
    ys = np.asarray([num(row, y) for row in rows], dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if int(finite.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(xs[finite], ys[finite])[0, 1])


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "participants": len({row["participant"] for row in rows}),
        "mean nearest training distance": mean([num(row, DIST) for row in rows]),
        "mean boundary distance new": mean([num(row, BOUNDARY) for row in rows]),
        "sd boundary distance new": sd([num(row, BOUNDARY) for row in rows]),
        "success rate": mean([num(row, "successful counterfactual (0/1)") for row in rows]),
        "target-directed rate": mean([num(row, "move towards target (0/1)") for row in rows]),
        "mean normalized edit distance": mean([num(row, "proximity") for row in rows]),
        "mean changed features": mean([
            sum(num(row, f"x_{index}_changed") for index in range(1, 6))
            for row in rows
        ]),
        "distance-boundary correlation": corr(rows, DIST, BOUNDARY),
    }


def kmeans2(matrix: np.ndarray, seed: int = 20260901) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = matrix[rng.choice(len(matrix), size=2, replace=False)].copy()
    labels = np.zeros(len(matrix), dtype=int)
    for _ in range(100):
        distances = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for label in (0, 1):
            if np.any(labels == label):
                centers[label] = matrix[labels == label].mean(axis=0)
    return labels


def main() -> None:
    _, rows = read_rows(INPUT)
    testing = [row for row in rows if row["phase"] == "testing"]
    cf_rows = [row for row in testing if row["xai"] == "counterfactual"]
    distance_values = [num(row, DIST) for row in testing if math.isfinite(num(row, DIST))]
    median_distance = float(np.median(distance_values))
    condition_close_summary = []
    for xai in ("counterfactual", "attribution", "none"):
        condition_rows = [
            row for row in testing
            if row["xai"] == xai and num(row, DIST) <= median_distance
        ]
        condition_close_summary.append({"xai": xai, **summarize(condition_rows)})

    participant_close_rows = []
    for participant in sorted({row["participant"] for row in testing}):
        participant_rows = [
            row for row in testing
            if row["participant"] == participant and num(row, DIST) <= median_distance
        ]
        if not participant_rows:
            continue
        participant_close_rows.append({
            "participant": participant,
            "xai": participant_rows[0]["xai"],
            **summarize(participant_rows),
        })
    participant_close_rows.sort(key=lambda row: row["mean boundary distance new"])
    write_rows(
        OUTPUT_DIR / "participant_near_training_boundary_summary.csv",
        list(participant_close_rows[0].keys()),
        participant_close_rows,
    )

    condition_cluster_rows = []
    participant_feature_rows = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for participant in sorted({row["participant"] for row in testing}):
        participant_rows = [row for row in testing if row["participant"] == participant]
        correct = [row for row in participant_rows if num(row, "successful counterfactual (0/1)") == 1]
        wrong = [row for row in participant_rows if num(row, "successful counterfactual (0/1)") == 0]
        feature_row = {
            "participant": participant,
            "xai": participant_rows[0]["xai"],
            "mean boundary distance new": mean([num(row, BOUNDARY) for row in participant_rows]),
            "sd boundary distance new": sd([num(row, BOUNDARY) for row in participant_rows]),
            "mean boundary distance when correct": mean([num(row, BOUNDARY) for row in correct]),
            "mean boundary distance when wrong": mean([num(row, BOUNDARY) for row in wrong]),
            "success rate": mean([num(row, "successful counterfactual (0/1)") for row in participant_rows]),
            "target-directed rate": mean([num(row, "move towards target (0/1)") for row in participant_rows]),
            "mean normalized edit distance": mean([num(row, "proximity") for row in participant_rows]),
            "mean changed features": mean([
                sum(num(row, f"x_{index}_changed") for index in range(1, 6))
                for row in participant_rows
            ]),
            "rho": participant_rows[0].get("additive change margin (rho)", "")
                or participant_rows[0].get("additive change margin (ρ)", ""),
            "model family": participant_rows[0].get("cognitive model v0.1 family", ""),
        }
        participant_feature_rows.append(feature_row)
        grouped[participant_rows[0]["xai"]].append(feature_row)

    for xai, records in grouped.items():
        features = np.asarray([
            [
                float(record["mean boundary distance new"]),
                float(record["sd boundary distance new"]),
                float(record["success rate"]),
                float(record["mean normalized edit distance"]),
                float(record["mean changed features"]),
            ]
            for record in records
        ], dtype=float)
        mu = features.mean(axis=0)
        sigma = features.std(axis=0, ddof=1)
        sigma[sigma == 0] = 1
        labels = kmeans2((features - mu) / sigma)
        means = {
            label: mean([
                float(record["mean boundary distance new"])
                for record, assigned in zip(records, labels)
                if assigned == label
            ])
            for label in (0, 1)
        }
        close_label = min(means, key=means.get)
        for record, label in zip(records, labels):
            cluster = "closer-boundary group" if label == close_label else "farther/spread group"
            record["within-condition boundary group"] = cluster
        for cluster in ("closer-boundary group", "farther/spread group"):
            cluster_records = [record for record in records if record["within-condition boundary group"] == cluster]
            condition_cluster_rows.append({
                "xai": xai,
                "boundary group": cluster,
                "participants": len(cluster_records),
                "mean boundary distance new": mean([float(r["mean boundary distance new"]) for r in cluster_records]),
                "mean sd boundary distance new": mean([float(r["sd boundary distance new"]) for r in cluster_records]),
                "mean boundary distance when correct": mean([float(r["mean boundary distance when correct"]) for r in cluster_records]),
                "mean boundary distance when wrong": mean([float(r["mean boundary distance when wrong"]) for r in cluster_records]),
                "success rate": mean([float(r["success rate"]) for r in cluster_records]),
                "target-directed rate": mean([float(r["target-directed rate"]) for r in cluster_records]),
                "mean normalized edit distance": mean([float(r["mean normalized edit distance"]) for r in cluster_records]),
                "mean changed features": mean([float(r["mean changed features"]) for r in cluster_records]),
                "model family counts": dict(Counter(str(r["model family"]) for r in cluster_records)),
            })

    participant_feature_rows.sort(key=lambda row: (row["xai"], row["within-condition boundary group"], row["mean boundary distance new"]))
    write_rows(
        OUTPUT_DIR / "participant_boundary_groups_by_condition.csv",
        list(participant_feature_rows[0].keys()),
        participant_feature_rows,
    )
    write_rows(
        OUTPUT_DIR / "condition_boundary_group_summary.csv",
        list(condition_cluster_rows[0].keys()),
        condition_cluster_rows,
    )

    summary = {
        "CF rows": len(cf_rows),
        "CF participants": len({row["participant"] for row in cf_rows}),
        "CF nearest-training distance vs boundary distance r": corr(cf_rows, DIST, BOUNDARY),
        "CF nearest-training distance vs success r": corr(cf_rows, DIST, "successful counterfactual (0/1)"),
        "median nearest training distance across testing rows": median_distance,
        "condition summary on close-to-training instances": condition_close_summary,
        "top 10 participants on close-to-training instances": participant_close_rows[:10],
        "top 10 close-to-training participant xai counts": dict(Counter(row["xai"] for row in participant_close_rows[:10])),
        "within-condition two-group summary": condition_cluster_rows,
        "outputs": {
            "participant near-training summary": str((OUTPUT_DIR / "participant_near_training_boundary_summary.csv").relative_to(ROOT)),
            "participant boundary groups": str((OUTPUT_DIR / "participant_boundary_groups_by_condition.csv").relative_to(ROOT)),
            "condition boundary group summary": str((OUTPUT_DIR / "condition_boundary_group_summary.csv").relative_to(ROOT)),
        },
    }
    (OUTPUT_DIR / "cf_boundary_followup_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
