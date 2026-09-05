"""Preserve the v1.3 cases and add eight q70 testing cases for v1.4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "analysis" / "diabetes_hierarchical_oriented_v5.csv"
CANDIDATES = ROOT / "analysis" / "diabetes_hierarchical_oriented_v6_reselected_candidate_shortlist.csv"
OUTPUT = ROOT / "analysis" / "diabetes_hierarchical_oriented_v6_preserved.csv"
SUMMARY = ROOT / "analysis" / "diabetes_hierarchical_oriented_v6_preserved.json"
PAIRS = ("glucose | insulin", "blood_pressure | age")


def vector(frame: pd.DataFrame) -> np.ndarray:
    attributes = np.vstack(frame["pair_oriented_attributions"].map(json.loads))
    changes = np.vstack(frame["pair_oriented_changes"].map(json.loads))
    return np.hstack((attributes, changes))


def main() -> None:
    old = pd.read_csv(OLD)
    candidates = pd.read_csv(CANDIDATES)
    key_columns = ["source_split", "instance_id"]
    old_keys = set(map(tuple, old[key_columns].itertuples(index=False, name=None)))
    additions = []

    for rank, pair in enumerate(PAIRS, start=1):
        group = candidates[candidates["main_pair"] == pair].copy()
        values = vector(group)
        means = values.mean(axis=0)
        scales = values.std(axis=0)
        scales[scales < 1e-12] = 1.0
        group["existing_pattern_distance"] = np.nan

        for prediction in (0, 1):
            existing = old[
                (old["main_pair"] == pair) & (old["prediction"] == prediction)
            ]
            existing_keys = set(
                map(tuple, existing[key_columns].itertuples(index=False, name=None))
            )
            positions = [
                index
                for index, key in enumerate(
                    map(tuple, group[key_columns].itertuples(index=False, name=None))
                )
                if key in existing_keys
            ]
            if len(positions) != 6:
                raise RuntimeError(f"Expected six preserved {pair}, prediction-{prediction} cases")
            standardized = (values - means) / scales
            centroid = standardized[positions].mean(axis=0)
            distances = np.linalg.norm(standardized - centroid, axis=1)
            eligible = group[group["prediction"] == prediction].copy()
            eligible["existing_pattern_distance"] = distances[
                group["prediction"].to_numpy() == prediction
            ]
            eligible = eligible[
                ~eligible[key_columns].apply(tuple, axis=1).isin(old_keys)
            ].sort_values(
                ["existing_pattern_distance", "local_boundary_distance", "instance_id"]
            )
            chosen = eligible.head(2).copy()
            if len(chosen) != 2:
                raise RuntimeError(f"Not enough additions for {pair}, prediction {prediction}")
            chosen["experimental_phase"] = "testing"
            chosen["selection_role"] = "expanded_boundary_nearest_existing_explanation_centroid"
            chosen["training_direction_pattern"] = str(
                existing["training_direction_pattern"].dropna().iloc[0]
            )
            chosen["main_cluster_rank"] = rank
            additions.append(chosen)

    added = pd.concat(additions, ignore_index=True)
    result = pd.concat([old, added], ignore_index=True, sort=False)
    result = result.sort_values(
        ["experimental_phase", "main_cluster_rank", "prediction", "instance_id"]
    )
    output_keys = set(map(tuple, result[key_columns].itertuples(index=False, name=None)))
    if not old_keys.issubset(output_keys) or len(result) != 32:
        raise RuntimeError("Preserved selection validation failed")
    if result["experimental_phase"].value_counts().to_dict() != {"testing": 20, "training": 12}:
        raise RuntimeError("Unexpected phase counts")
    result.to_csv(OUTPUT, index=False)
    summary = {
        "training": 12,
        "testing": 20,
        "preserved_cases": len(old),
        "added_cases": added[
            ["main_pair", "prediction", "source_split", "instance_id", "local_boundary_distance", "existing_pattern_distance"]
        ].to_dict("records"),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
