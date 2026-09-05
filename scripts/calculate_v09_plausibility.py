"""Calculate v0.9 plausibility as distance to the nearest training record.

Aryal and Keane (2024) operationalize semi-factual plausibility as distance to
the single nearest training instance (k=1), where lower is better.  This script
uses the experiment's existing five-feature normalized L1 distance so the
result is on the same scale as the JMP proximity column.  Numerical values are
range-normalized and clamped to [0, 1]; the binary categorical features use
0/1 mismatch distance.  Training-phase result rows remain missing.
"""

from __future__ import annotations

import csv
import json
import re
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "qualtrics" / "qualtrics_results_v0.9.for-plausibility.csv"
EXPERIMENT_DATA = REPO_ROOT / "static" / "experiment-data.json"
TEMP_DIR = Path(tempfile.gettempdir())
OUTPUT_CSV = TEMP_DIR / "plausibility_v09_values_calculated.csv"
SUMMARY_JSON = TEMP_DIR / "plausibility_v09_summary_calculated.json"

DOMAIN_FEATURES = {
    "housing": ("sqft_living", "bedrooms", "bathrooms", "floors", "grade"),
    "diabetes": ("glucose", "blood_pressure", "insulin", "bmi", "age"),
    "safelimit": ("units", "weight", "duration", "gender", "stomach_fullness"),
}

NORMALIZED_VALUE_RE = re.compile(
    r"\((-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)"
)


def normalize(value: str, feature_type: str, feature_range: list[object]) -> float:
    if feature_type == "categorical":
        normalized = value.casefold()
        categories = [str(category).casefold() for category in feature_range]
        if normalized not in categories:
            raise ValueError(f"Unknown categorical value {value!r} for {categories!r}")
        if len(categories) <= 1:
            return 0.0
        return categories.index(normalized) / (len(categories) - 1)

    low, high = float(feature_range[0]), float(feature_range[1])
    if high == low:
        return 0.0
    return min(1.0, max(0.0, (float(value) - low) / (high - low)))


def final_normalized_profile(attribute_text: str) -> np.ndarray:
    lines = [line for line in attribute_text.splitlines() if line.strip()]
    if len(lines) != 5:
        raise ValueError(f"Expected five attribute lines, found {len(lines)}")
    values: list[float] = []
    for line in lines:
        matches = NORMALIZED_VALUE_RE.findall(line)
        if not matches:
            raise ValueError(f"No normalized value found in {line!r}")
        values.append(float(matches[-1]))
    return np.asarray(values, dtype=float)


def load_reference_matrices() -> dict[str, np.ndarray]:
    experiment = json.loads(EXPERIMENT_DATA.read_text(encoding="utf-8"))
    matrices: dict[str, np.ndarray] = {}

    for domain, feature_names in DOMAIN_FEATURES.items():
        case = experiment["datasets"][domain]["test_pool"][0]
        case_feature_names = tuple(case["raw_feature_names"])
        if case_feature_names != feature_names:
            raise ValueError(
                f"{domain}: expected features {feature_names}, found {case_feature_names}"
            )
        feature_types = list(case["feature_types"])
        feature_ranges = list(case["raw_feature_ranges"])
        train_path = REPO_ROOT / "src" / "data" / domain / "train.csv"

        normalized_rows: list[list[float]] = []
        with train_path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                normalized_rows.append(
                    [
                        normalize(row[name], feature_types[index], feature_ranges[index])
                        for index, name in enumerate(feature_names)
                    ]
                )
        matrix = np.asarray(normalized_rows, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 5 or matrix.shape[0] == 0:
            raise ValueError(f"{domain}: invalid training matrix shape {matrix.shape}")
        matrices[domain] = matrix

    return matrices


def main() -> None:
    reference_matrices = load_reference_matrices()
    output_rows: list[dict[str, str]] = []
    scores_by_domain: dict[str, list[float]] = {
        domain: [] for domain in DOMAIN_FEATURES
    }
    testing_rows = 0
    training_rows = 0

    with INPUT_CSV.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    for row_number, row in enumerate(rows, start=1):
        domain = row["domain"].strip().casefold()
        phase = row["phase"].strip().casefold()
        if domain not in reference_matrices:
            raise ValueError(f"Row {row_number}: unknown domain {domain!r}")

        score_text = ""
        if phase == "testing":
            testing_rows += 1
            profile = final_normalized_profile(row["attribute values before and after"])
            distances = np.abs(reference_matrices[domain] - profile).sum(axis=1)
            score = float(distances.min())
            scores_by_domain[domain].append(score)
            score_text = f"{score:.12g}"
        elif phase == "training":
            training_rows += 1
        else:
            raise ValueError(f"Row {row_number}: unknown phase {phase!r}")

        output_rows.append(
            {
                "row_number": str(row_number),
                "participant": row["participant"],
                "domain": domain,
                "phase": phase,
                "plausibility": score_text,
            }
        )

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "definition": "normalized L1 distance to the single nearest training instance",
        "neighbors": 1,
        "direction": "lower is more plausible",
        "row_count": len(output_rows),
        "testing_scored": testing_rows,
        "training_missing": training_rows,
        "reference_training_rows": {
            domain: int(matrix.shape[0])
            for domain, matrix in reference_matrices.items()
        },
        "testing_score_summary": {
            domain: {
                "count": len(scores),
                "min": min(scores),
                "median": float(np.median(scores)),
                "mean": float(np.mean(scores)),
                "max": max(scores),
            }
            for domain, scores in scores_by_domain.items()
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
