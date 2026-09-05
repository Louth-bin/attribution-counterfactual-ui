"""Add a manually verified dominant counterfactual-reasoning strategy column.

The categories summarize feature selection, amount, endpoint, and stopping
patterns across all testing cases.  They are intentionally broad because a
single participant can show elements of more than one process.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLUMN = "reasoning strategy"

SAME_ATTRIBUTES = "Same attributes across cases"
MOST_INFLUENTIAL = "Most influential attributes"
TARGET_RANGE = "Values outside the target range"
SMALLEST_EASIEST = "Smallest or easiest changes"
HARD_TO_UNDERSTAND = "Random or hard to understand"

# Diabetes categories use both the recorded edits and the two open-ended
# responses (Q36 and Q37).  The labels state the dominant attribute-selection
# rule.  Number of edits and endpoints are supporting evidence rather than
# separate, overlapping categories.
CASE_INFLUENCE = "Change the most influential attributes for the case"
CASE_OUTLIERS = "Change the most out-of-range attributes for the case"
TARGET_PROFILE = "Change all mismatching attributes toward a target profile"
FIXED_ATTRIBUTES = "Always change the same primary attribute regardless of the case"
NO_CLEAR_RULE = "No consistent rule or guessed changes"


V04_STRATEGIES = {
    # Housing: attribution
    "R_1eUSbs1pQPimBTZ": MOST_INFLUENTIAL,
    "R_322r5wcn8EpZV6e": TARGET_RANGE,
    "R_322vaHesZ9mABo8": SAME_ATTRIBUTES,
    "R_39eI4Ru6gkJHOk9": SMALLEST_EASIEST,
    "R_3fTrP1eNSLr6OFA": SMALLEST_EASIEST,
    "R_5jOrV0ED2kEcpj3": SAME_ATTRIBUTES,
    "R_5JrAewWP4QfNMDy": MOST_INFLUENTIAL,
    "R_67vRcoYqomc0aH9": MOST_INFLUENTIAL,
    "R_6DJgrDPENvbubax": MOST_INFLUENTIAL,
    "R_7CymkLq1W5EbO9c": MOST_INFLUENTIAL,
    "R_7o6kIdn4EaINkfo": SAME_ATTRIBUTES,
    "R_7rMsMTKGzMUtNfr": SAME_ATTRIBUTES,
    # Housing: counterfactual
    "R_10ZHzHEdKnH9gjk": TARGET_RANGE,
    "R_3L5KFiVFwyHWfy9": SMALLEST_EASIEST,
    "R_3nxy8fTXACyezRw": TARGET_RANGE,
    "R_5aFAlypirbiSy54": TARGET_RANGE,
    "R_5g6goAxW0POnfXS": SMALLEST_EASIEST,
    "R_6aVQRGrKd153TVx": HARD_TO_UNDERSTAND,
    "R_6D8FEAVXW6pNKgX": SMALLEST_EASIEST,
    "R_6vdUndtY54RqpVv": SMALLEST_EASIEST,
    "R_7m6pHZPK4Sqdz6p": HARD_TO_UNDERSTAND,
    "R_7ouaHYEFOkTbFrn": TARGET_RANGE,
    # Housing: none
    "R_12qUeHEbPUo2PTn": TARGET_RANGE,
    "R_3da8jwTxaAuNxsI": TARGET_RANGE,
    "R_3FJQV7viaHYDHi3": TARGET_RANGE,
    "R_3rxDc2Hw1SSJ2Es": TARGET_RANGE,
    "R_51KsCVvwgplPovt": TARGET_RANGE,
    "R_5ARKBiJcbeCMbPI": TARGET_RANGE,
    "R_5DSzaEugkyhB8He": SMALLEST_EASIEST,
    "R_5g60ppMS88e6WxS": TARGET_RANGE,
    "R_5lApGQB2048vlh4": TARGET_RANGE,
    "R_63dAINefVgDYtEz": SAME_ATTRIBUTES,
    "R_6KVGHDWJFFdRBWL": TARGET_RANGE,
    "R_6N8hq2tClpLxNP1": HARD_TO_UNDERSTAND,
    # SafeLimit: attribution
    "R_12bqooCAAJESsRr": MOST_INFLUENTIAL,
    "R_3OLJbpwtIN5r62B": MOST_INFLUENTIAL,
    "R_5qAk52BxKeKkOOJ": SAME_ATTRIBUTES,
    "R_7oigZfUmLeZhBs7": MOST_INFLUENTIAL,
    # SafeLimit: counterfactual
    "R_1knCAqvmyTwFTer": SMALLEST_EASIEST,
    "R_31GpbVwSgVHpLam": HARD_TO_UNDERSTAND,
    "R_5fiJADPk00t6Se9": SAME_ATTRIBUTES,
    "R_5rPkDbCA9RBiFjd": SAME_ATTRIBUTES,
    "R_6DZbalJ8TCZtARb": TARGET_RANGE,
    # SafeLimit: none
    "R_1J8ShU0r192gzAd": SAME_ATTRIBUTES,
    "R_6t07wdUJUABPKIE": SAME_ATTRIBUTES,
    "R_71opu9CilxP5KrN": TARGET_RANGE,
}


V07_STRATEGIES = {
    # Diabetes: attribution
    "R_1eygULWOJvBHm3R": CASE_INFLUENCE,
    "R_1LdrH2cHZN4yUZH": CASE_OUTLIERS,
    "R_3obc8KIkN9SGl0t": TARGET_PROFILE,
    "R_3RUOoa2ojaPNwoF": NO_CLEAR_RULE,
    "R_5j6OULl151xAZzJ": CASE_OUTLIERS,
    "R_5JCJbR8oOgkjlA0": TARGET_PROFILE,
    "R_5kGrfdcFvaTZHov": TARGET_PROFILE,
    "R_5QGGlhAzSiOVePL": CASE_OUTLIERS,
    "R_5z49Mi5OsGgJ6pz": CASE_OUTLIERS,
    "R_61yCzDc7u4eGlCE": CASE_INFLUENCE,
    "R_6wzdpKed5aB6LmF": CASE_INFLUENCE,
    "R_7NDOTmN9jbLS3Fq": CASE_INFLUENCE,
    # Diabetes: counterfactual
    "R_1beGb3seoyBqBAl": TARGET_PROFILE,
    "R_2wBl4JWxZiMKmQY": NO_CLEAR_RULE,
    "R_3B7CxgDgEzWSvrb": NO_CLEAR_RULE,
    "R_3MQqO18M5uhCVdT": CASE_INFLUENCE,
    "R_3oGxWzvzrZAX87o": CASE_OUTLIERS,
    "R_55tIhWzHfQVe7nG": CASE_OUTLIERS,
    "R_5q8M1s4dsLm5hDO": CASE_INFLUENCE,
    "R_65WwRg0sK4kNw2R": CASE_OUTLIERS,
    "R_6a9mdRXLV87vbOx": NO_CLEAR_RULE,
    "R_7kHwLYEKytNtFsY": NO_CLEAR_RULE,
    # Diabetes: none
    "R_1QAMxqJ6UaP9XPc": CASE_OUTLIERS,
    "R_5b2MBKWDgnmpE3f": CASE_OUTLIERS,
    "R_5OT6oo9gKTZyooh": FIXED_ATTRIBUTES,
    "R_68NgnXPoKxlotUv": NO_CLEAR_RULE,
    "R_6HeiSeEgD99blZN": CASE_INFLUENCE,
    "R_6K9YdDlQJKgwWKY": CASE_OUTLIERS,
    "R_6TYmqMi3LX02S5Z": CASE_OUTLIERS,
    "R_71lEhxrOvvuiitz": FIXED_ATTRIBUTES,
    "R_7e3EuifgObjNXJm": CASE_OUTLIERS,
    "R_7iwBjEgQvGNOP4l": TARGET_PROFILE,
    "R_7JSvmv0y2BQADx7": TARGET_PROFILE,
}


def add_strategies(path: Path, mapping: dict[str, str]) -> dict[str, object]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        fieldnames = list(reader.fieldnames)

    participants = {row["participant"] for row in rows}
    missing = participants - set(mapping)
    extra = set(mapping) - participants
    if missing or extra:
        raise ValueError(
            f"{path.name} strategy mapping mismatch: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    if COLUMN not in fieldnames:
        insertion_point = fieldnames.index("x_5_changed") + 1
        fieldnames.insert(insertion_point, COLUMN)
    for row in rows:
        row[COLUMN] = mapping[row["participant"]]

    temporary_path = path.with_name(
        f"{path.stem}.reasoning-strategy-tmp{path.suffix}"
    )
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)

    participant_counts = Counter(mapping.values())
    return {
        "file": path.name,
        "rows": len(rows),
        "participants": len(participants),
        "strategy_counts": dict(sorted(participant_counts.items())),
    }


def main() -> None:
    reports = [
        add_strategies(
            ROOT / "qualtrics" / "qualtrics_results_v0.4.csv", V04_STRATEGIES
        ),
        add_strategies(
            ROOT / "qualtrics" / "qualtrics_results_v0.7.csv", V07_STRATEGIES
        ),
    ]
    for report in reports:
        print(report)


if __name__ == "__main__":
    main()
