import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "qualtrics" / "qualtrics_results_v2.0.csv")
training = (
    df[(df["phase"].eq("training")) & (df["xai"].eq("counterfactual"))]
    .drop_duplicates("instance id")
    .sort_values("instance id")
)

pattern = re.compile(
    r'"(?P<feature>[^"]+)"\s*-\s*'
    r'[-+0-9.eE]+\((?P<before>[-+0-9.eE]+)\)'
    r'(?:\s*->\s*[-+0-9.eE]+\((?P<after>[-+0-9.eE]+)\))?'
)

rows = []
for _, row in training.iterrows():
    record = {
        "instance id": int(row["instance id"]),
        "original label": row["original label"],
    }
    changed = []
    total = 0.0
    for match in pattern.finditer(str(row["explanation"])):
        feature = match.group("feature")
        before = float(match.group("before"))
        after_text = match.group("after")
        change = 0.0 if after_text is None else float(after_text) - before
        record[feature] = change
        total += abs(change)
        if abs(change) > 1e-9:
            changed.append(feature)
    record["cf_L1_norm_change"] = total
    record["changed_features"] = ", ".join(changed)
    rows.append(record)

out = pd.DataFrame(rows)
cols = [
    "instance id",
    "original label",
    "changed_features",
    "cf_L1_norm_change",
    "Glucose",
    "Blood Pressure",
    "Insulin",
    "BMI",
    "Age",
]
print(out[cols].round(3).to_string(index=False))
print()
print(
    out.groupby("changed_features")["cf_L1_norm_change"]
    .agg(["count", "mean", "std", "min", "max"])
    .round(3)
    .to_string()
)
