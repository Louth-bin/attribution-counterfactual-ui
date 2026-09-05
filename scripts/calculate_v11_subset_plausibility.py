"""Calculate plausibility relative to the 10 displayed training profiles.

The metric mirrors the higher-is-better v1.0/v1.1 plausibility definition:
one minus the minimum five-feature Gower distance (k=1). The only difference
is that the reference set is the experiment bundle's 10-item training_pool for
the row's domain rather than the complete model-training split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_VALUE_RE = re.compile(
    r"\((-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)"
)


def normalized_profile(attribute_text: str) -> np.ndarray:
    lines = [line for line in attribute_text.splitlines() if line.strip()]
    if len(lines) != 5:
        raise ValueError(f"Expected five attribute lines, found {len(lines)}")
    values: list[float] = []
    for line in lines:
        matches = NORMALIZED_VALUE_RE.findall(line)
        if not matches:
            raise ValueError(f"No normalized value in {line!r}")
        values.append(float(matches[-1]))
    return np.asarray(values, dtype=float)


def normalize_value(value: object, kind: str, levels: list[object]) -> float:
    if kind == "categorical":
        categories = [str(item).casefold() for item in levels]
        index = categories.index(str(value).casefold())
        return index / (len(categories) - 1) if len(categories) > 1 else 0.0
    low, high = float(levels[0]), float(levels[1])
    if math.isclose(low, high):
        return 0.0
    return min(1.0, max(0.0, (float(value) - low) / (high - low)))


def displayed_training_matrices(bundle_path: Path) -> dict[str, np.ndarray]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    matrices: dict[str, np.ndarray] = {}
    for domain, dataset in bundle["datasets"].items():
        training_pool = dataset["training_pool"]
        if len(training_pool) != 10:
            raise ValueError(
                f"Expected 10 displayed training profiles for {domain}, "
                f"found {len(training_pool)}"
            )
        rows: list[list[float]] = []
        for case in training_pool:
            rows.append(
                [
                    normalize_value(value, kind, list(levels))
                    for value, kind, levels in zip(
                        case["raw_feature_values"],
                        case["feature_types"],
                        case["raw_feature_ranges"],
                    )
                ]
            )
        matrices[domain.casefold()] = np.asarray(rows, dtype=float)
    return matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--bundle", type=Path, default=ROOT / "static" / "experiment-data.json"
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    matrices = displayed_training_matrices(args.bundle)

    output_rows: list[dict[str, object]] = []
    values: list[float] = []
    training_nonmissing = 0
    for row_number, row in enumerate(rows, start=1):
        value: str | float = ""
        if row["phase"].casefold() == "testing":
            profile = normalized_profile(row["attribute values before and after"])
            reference = matrices[row["domain"].casefold()]
            nearest_gower = float(np.abs(reference - profile).mean(axis=1).min())
            score = min(1.0, max(0.0, 1.0 - nearest_gower))
            value = f"{score:.12g}"
            values.append(score)
        elif value != "":
            training_nonmissing += 1
        output_rows.append(
            {
                "row_number": row_number,
                "participant": row["participant"],
                "phase": row["phase"],
                "plausibility (subset)": value,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "rows": len(rows),
        "testing_rows": len(values),
        "training_rows": len(rows) - len(values),
        "training_nonmissing": training_nonmissing,
        "reference_profiles_per_domain": {
            domain: int(matrix.shape[0]) for domain, matrix in matrices.items()
        },
        "definition": "1 - Gower distance to nearest displayed domain training profile (k=1); higher is more plausible",
        "testing_range": [min(values), max(values)],
        "testing_mean": float(np.mean(values)),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
