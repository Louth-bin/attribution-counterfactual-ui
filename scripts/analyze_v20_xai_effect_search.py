"""Exploratory, multiplicity-aware XAI effect search for Qualtrics v2.0.

The canonical input is a fresh CSV snapshot exported from
qualtrics_results_v2.0.jmp.  Treatment is assigned between participants, so all
inferential tests use participants as the independent units.  The script is
deliberately broad, but keeps the search auditable through deterministic
permutations, false-discovery-rate columns, wave splits, and sensitivity checks.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "v20-effect-search"
SOURCE = OUTDIR / "qualtrics_results_v2.0_from_jmp.csv"
OLD_WAVE_SOURCE = ROOT / "qualtrics" / "v1.9_latest_processed.csv"
BUNDLE_SOURCE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
SEED = 20260902
N_PERM = 9_999
N_BOOT = 4_000

FEATURES = ["Glucose", "Blood Pressure", "Insulin", "BMI", "Age"]
XCOLS = [f"x_{i}" for i in range(1, 6)]
CONDITIONS = ["none", "attribution", "counterfactual"]
COMPARISONS = [
    ("counterfactual", "none"),
    ("attribution", "none"),
    ("counterfactual", "attribution"),
]

VALUE_RE = re.compile(
    r'"(?P<feature>[^"]+)"\s*-\s*'
    r'(?P<before_raw>[-+0-9.eE]+)\((?P<before_norm>[-+0-9.eE]+)\)'
    r'(?:\s*->\s*(?P<after_raw>[-+0-9.eE]+)\((?P<after_norm>[-+0-9.eE]+)\))?'
)
ATTR_RE = re.compile(
    r'"(?P<feature>[^"]+)"\s*-\s*(?P<weight>[-+0-9.eE]+)%\s+toward\s+(?P<label>Diabetes|No Diabetes)'
)


OUTCOMES: dict[str, dict[str, object]] = {
    "boundary_new": {"better": -1, "family": "performance"},
    "boundary_improvement": {"better": 1, "family": "performance"},
    "relative_boundary_improvement": {"better": 1, "family": "performance"},
    "success": {"better": 1, "family": "performance"},
    "confidence_gain": {"better": 1, "family": "performance"},
    "move_target": {"better": 1, "family": "performance"},
    "edit_l1": {"better": -1, "family": "cost"},
    "n_changed": {"better": -1, "family": "cost"},
    "distance_to_minimal_cf": {"better": -1, "family": "cost"},
    "plausibility": {"better": 1, "family": "quality"},
    "subset_plausibility": {"better": 1, "family": "quality"},
    "actionability": {"better": 1, "family": "quality"},
    "target_centroid_progress": {"better": 1, "family": "geometry"},
    "target_centroid_efficiency": {"better": 1, "family": "geometry"},
    "nearest_target_train_progress": {"better": 1, "family": "geometry"},
    "target_helpful_shap": {"better": 1, "family": "attribution_alignment"},
    "helpful_shap_efficiency": {"better": 1, "family": "attribution_alignment"},
    "attribution_feature_overlap": {"better": 1, "family": "attribution_alignment"},
    "attribution_global_relevance": {"better": 1, "family": "attribution_alignment"},
    "training_direction_agreement": {"better": 1, "family": "counterfactual_alignment"},
    "training_direction_magnitude_index": {"better": 1, "family": "counterfactual_alignment"},
    "cf_demo_cosine": {"better": 1, "family": "counterfactual_alignment"},
    "cf_demo_feature_jaccard": {"better": 1, "family": "counterfactual_alignment"},
    "cf_demo_sign_agreement": {"better": 1, "family": "counterfactual_alignment"},
    "cf_demo_delta_l1_error": {"better": -1, "family": "counterfactual_alignment"},
    "target_demo_endpoint_distance": {"better": -1, "family": "counterfactual_alignment"},
    "response_time": {"better": -1, "family": "process"},
}


KEY_MODERATOR_OUTCOMES = [
    "boundary_new",
    "boundary_improvement",
    "success",
    "confidence_gain",
    "edit_l1",
    "target_centroid_progress",
    "target_helpful_shap",
    "cf_demo_cosine",
    "cf_demo_feature_jaccard",
    "attribution_feature_overlap",
]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_profile(text: object, use_after: bool) -> np.ndarray:
    found: dict[str, float] = {}
    for match in VALUE_RE.finditer(str(text)):
        before = float(match.group("before_norm"))
        after = match.group("after_norm")
        found[match.group("feature")] = float(after) if use_after and after is not None else before
    if set(found) != set(FEATURES):
        raise ValueError(f"Could not parse five-feature profile: {text!r}")
    return np.asarray([found[name] for name in FEATURES], dtype=float)


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    a = np.asarray(numerator, dtype=float)
    b = np.asarray(denominator, dtype=float)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=np.abs(b) > 1e-12)


def bh_adjust(values: pd.Series) -> pd.Series:
    p = numeric(values).to_numpy(float)
    result = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return pd.Series(result, index=values.index)
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(len(pv))
    out[order] = np.minimum(adjusted, 1.0)
    result[np.where(valid)[0]] = out
    return pd.Series(result, index=values.index)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_var = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (
        len(a) + len(b) - 2
    )
    if pooled_var <= 0:
        return float("nan")
    d = (np.mean(a) - np.mean(b)) / math.sqrt(pooled_var)
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    return float(d * correction)


def pairwise_test(
    frame: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_a: object,
    group_b: object,
    *,
    strata_col: str | None = "wave",
    n_perm: int = N_PERM,
    n_boot: int = N_BOOT,
    seed_offset: int = 0,
) -> dict[str, float | int]:
    cols = [value_col, group_col] + ([strata_col] if strata_col and strata_col in frame else [])
    work = frame[cols].copy()
    work[value_col] = numeric(work[value_col])
    work = work[work[group_col].isin([group_a, group_b]) & work[value_col].notna()].reset_index(drop=True)
    a = work.loc[work[group_col].eq(group_a), value_col].to_numpy(float)
    b = work.loc[work[group_col].eq(group_b), value_col].to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return {
            "n_a": len(a), "n_b": len(b), "mean_a": np.nan, "mean_b": np.nan,
            "difference": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "permutation_p": np.nan, "welch_p": np.nan, "hedges_g": np.nan,
        }
    observed = float(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(SEED + seed_offset)
    labels = work[group_col].to_numpy(object)
    values = work[value_col].to_numpy(float)
    extreme = 0
    strata = work[strata_col].to_numpy(object) if strata_col and strata_col in work else None
    for _ in range(n_perm):
        permuted = labels.copy()
        if strata is None:
            permuted = rng.permutation(permuted)
        else:
            for level in np.unique(strata):
                idx = np.flatnonzero(strata == level)
                permuted[idx] = rng.permutation(permuted[idx])
        diff = values[permuted == group_a].mean() - values[permuted == group_b].mean()
        extreme += abs(diff) >= abs(observed) - 1e-15
    permutation_p = (extreme + 1) / (n_perm + 1)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = rng.choice(a, size=len(a), replace=True).mean() - rng.choice(
            b, size=len(b), replace=True
        ).mean()
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "difference": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "permutation_p": float(permutation_p),
        "welch_p": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
        "hedges_g": hedges_g(a, b),
    }


def participant_slope(group: pd.DataFrame, outcome: str, x_col: str) -> float:
    y = numeric(group[outcome]).to_numpy(float)
    x = numeric(group[x_col]).to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4 or np.nanstd(x[valid]) <= 1e-12:
        return float("nan")
    x = (x[valid] - np.mean(x[valid])) / np.std(x[valid])
    return float(np.polyfit(x, y[valid], 1)[0])


def add_geometry(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    before = np.vstack([parse_profile(text, False) for text in data["attribute values before and after"]])
    after = np.vstack([parse_profile(text, True) for text in data["attribute values before and after"]])
    for i, name in enumerate(FEATURES):
        data[f"orig_{i+1}"] = before[:, i]
        data[f"new_{i+1}"] = after[:, i]
        data[f"delta_{i+1}"] = after[:, i] - before[:, i]

    training = data[data["phase"].eq("training")].copy()
    unique_training = training.drop_duplicates("instance id")
    train_profiles = {
        int(row["instance id"]): parse_profile(row["attribute values before and after"], False)
        for _, row in unique_training.iterrows()
    }
    train_labels = {
        int(row["instance id"]): str(row["original label"])
        for _, row in unique_training.iterrows()
    }

    cf_training = training[training["xai"].eq("counterfactual")].drop_duplicates("instance id")
    demo_endpoints: dict[int, np.ndarray] = {}
    demo_deltas: dict[int, np.ndarray] = {}
    for _, row in cf_training.iterrows():
        iid = int(row["instance id"])
        endpoint = parse_profile(row["explanation"], True)
        demo_endpoints[iid] = endpoint
        demo_deltas[iid] = endpoint - train_profiles[iid]

    attr_training = training[training["xai"].eq("attribution")].drop_duplicates("instance id")
    attr_features: dict[int, set[int]] = {}
    attr_weights: dict[int, np.ndarray] = {}
    for _, row in attr_training.iterrows():
        iid = int(row["instance id"])
        weights = np.zeros(len(FEATURES), dtype=float)
        shown: set[int] = set()
        for match in ATTR_RE.finditer(str(row["explanation"])):
            if match.group("feature") in FEATURES:
                idx = FEATURES.index(match.group("feature"))
                shown.add(idx)
                weights[idx] = abs(float(match.group("weight"))) / 100.0
        attr_features[iid] = shown
        attr_weights[iid] = weights

    global_relevance: dict[str, np.ndarray] = {}
    for original_label in ["Diabetes", "No Diabetes"]:
        ids = [iid for iid, label in train_labels.items() if label == original_label]
        vector = np.sum([attr_weights[iid] for iid in ids], axis=0)
        global_relevance[original_label] = vector / vector.sum() if vector.sum() else vector

    centroids: dict[str, np.ndarray] = {}
    endpoint_centroids: dict[str, np.ndarray] = {}
    for label in ["Diabetes", "No Diabetes"]:
        centroids[label] = np.mean(
            [profile for iid, profile in train_profiles.items() if train_labels[iid] == label], axis=0
        )
        # A demonstration endpoint has the label opposite its source label.
        endpoint_centroids[label] = np.mean(
            [demo_endpoints[iid] for iid in demo_endpoints if train_labels[iid] != label], axis=0
        )

    instance_records: dict[int, dict[str, object]] = {}
    testing_unique = data[data["phase"].eq("testing")].drop_duplicates("instance id")
    for idx, row in testing_unique.iterrows():
        iid = int(row["instance id"])
        original_label = str(row["original label"])
        target_label = str(row["target label"])
        original = before[idx]
        same_ids = [tid for tid, label in train_labels.items() if label == original_label]
        distances = {tid: float(np.abs(original - train_profiles[tid]).sum()) for tid in same_ids}
        nearest_id = min(distances, key=distances.get)
        target_train_ids = [tid for tid, label in train_labels.items() if label == target_label]
        nearest_target_distance = min(float(np.abs(original - train_profiles[tid]).sum()) for tid in target_train_ids)
        instance_records[iid] = {
            "instance id": iid,
            "nearest_same_training_id": nearest_id,
            "nearest_same_training_distance": distances[nearest_id],
            "nearest_target_training_distance_original": nearest_target_distance,
            "original_to_target_centroid": float(np.abs(original - centroids[target_label]).sum()),
            "original_to_source_centroid": float(np.abs(original - centroids[original_label]).sum()),
            "demo_delta_l1": float(np.abs(demo_deltas[nearest_id]).sum()),
            "original label": original_label,
            "target label": target_label,
            **{f"original_{name}": original[j] for j, name in enumerate(FEATURES)},
        }

    instance_meta = pd.DataFrame(instance_records.values()).set_index("instance id", drop=False)

    for idx, row in data.iterrows():
        if row["phase"] != "testing":
            continue
        iid = int(row["instance id"])
        meta = instance_records[iid]
        nearest_id = int(meta["nearest_same_training_id"])
        original = before[idx]
        endpoint = after[idx]
        delta = endpoint - original
        demo_delta = demo_deltas[nearest_id]
        changed = np.abs(delta) > 1e-10
        demo_changed = np.abs(demo_delta) > 1e-10
        union = changed | demo_changed
        overlap = changed & demo_changed
        denom = np.linalg.norm(delta) * np.linalg.norm(demo_delta)
        cosine = float(np.dot(delta, demo_delta) / denom) if denom > 1e-12 else np.nan
        sign_agreement = (
            float(np.mean(np.sign(delta[overlap]) == np.sign(demo_delta[overlap])))
            if overlap.any() else 0.0
        )
        attr_set = attr_features[nearest_id]
        selected = set(np.flatnonzero(changed).tolist())
        attr_union = selected | attr_set
        data.loc[idx, "cf_demo_cosine"] = cosine
        data.loc[idx, "cf_demo_feature_jaccard"] = float(overlap.sum() / union.sum()) if union.any() else np.nan
        data.loc[idx, "cf_demo_sign_agreement"] = sign_agreement
        data.loc[idx, "cf_demo_delta_l1_error"] = float(np.abs(delta - demo_delta).sum())
        data.loc[idx, "target_demo_endpoint_distance"] = float(
            np.abs(endpoint - endpoint_centroids[str(row["target label"])]).sum()
        )
        data.loc[idx, "attribution_feature_overlap"] = (
            float(len(selected & attr_set) / len(attr_union)) if attr_union else np.nan
        )
        relevance = global_relevance[str(row["original label"])]
        data.loc[idx, "attribution_global_relevance"] = (
            float(relevance[list(selected)].mean()) if selected else 0.0
        )
        target_centroid = centroids[str(row["target label"])]
        data.loc[idx, "target_centroid_progress"] = float(
            np.abs(original - target_centroid).sum() - np.abs(endpoint - target_centroid).sum()
        )
        target_train_ids = [tid for tid, label in train_labels.items() if label == str(row["target label"])]
        original_nearest = min(float(np.abs(original - train_profiles[tid]).sum()) for tid in target_train_ids)
        endpoint_nearest = min(float(np.abs(endpoint - train_profiles[tid]).sum()) for tid in target_train_ids)
        data.loc[idx, "nearest_target_train_progress"] = original_nearest - endpoint_nearest

    return data, instance_meta


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Run export_v20_jmp_snapshot.py first: {SOURCE}")
    data = pd.read_csv(SOURCE, low_memory=False)
    if len(data) != 1984 or data["participant"].nunique() != 62:
        raise ValueError(f"Unexpected canonical JMP snapshot shape: {data.shape}")
    if data.duplicated(["participant", "phase", "instance id"]).any():
        raise ValueError("Duplicate participant-phase-instance rows in JMP snapshot")

    old_ids = set(pd.read_csv(OLD_WAVE_SOURCE, usecols=["participant"])["participant"].astype(str))
    data["wave"] = np.where(data["participant"].astype(str).isin(old_ids), "original_36", "new_26")
    data, instance_meta = add_geometry(data)

    testing = data[data["phase"].eq("testing")].copy()
    testing["boundary_original"] = numeric(testing["abs dist original to boundary"])
    testing["boundary_new"] = numeric(testing["abs dist user CF to boundary"])
    testing["boundary_improvement"] = testing["boundary_original"] - testing["boundary_new"]
    testing["relative_boundary_improvement"] = np.clip(
        safe_divide(testing["boundary_improvement"], testing["boundary_original"]), -5, 5
    )
    testing["success"] = numeric(testing["successful CF"])
    testing["confidence_gain"] = numeric(testing["delta confidence of target label"])
    testing["move_target"] = numeric(testing["user CF increase target label confidence?"])
    testing["edit_l1"] = np.abs(testing[[f"delta_{i}" for i in range(1, 6)]].to_numpy(float)).sum(axis=1)
    testing["n_changed"] = (np.abs(testing[[f"delta_{i}" for i in range(1, 6)]].to_numpy(float)) > 1e-10).sum(axis=1)
    testing["distance_to_minimal_cf"] = numeric(testing["distance to closest minimal counterfactual"])
    testing["plausibility"] = numeric(testing["plausibility"])
    testing["subset_plausibility"] = numeric(testing["plausibility (subset)"])
    testing["actionability"] = numeric(testing["actionability (0/1)"])
    testing["response_time"] = numeric(testing["response time (seconds)"])
    testing["target_centroid_efficiency"] = np.clip(
        safe_divide(testing["target_centroid_progress"], testing["edit_l1"]), -5, 5
    )

    helpful_cols = [f"x_{i} {FEATURES[i-1]} target-helpful SHAP change" for i in range(1, 6)]
    testing["target_helpful_shap"] = testing[helpful_cols].apply(numeric).fillna(0).sum(axis=1)
    testing["helpful_shap_efficiency"] = np.clip(
        safe_divide(testing["target_helpful_shap"], testing["edit_l1"]), -5, 5
    )
    direction_cols = [f"x_{i}_changed_in_training_cf_direction" for i in range(1, 6)]
    direction = testing[direction_cols].apply(numeric).to_numpy(float)
    changed = testing[[f"x_{i}_changed" for i in range(1, 6)]].apply(numeric).to_numpy(float)
    aligned = np.nansum(np.where(changed > 0, direction, np.nan), axis=1)
    testing["training_direction_agreement"] = np.divide(
        aligned, testing["n_changed"], out=np.zeros(len(testing)), where=testing["n_changed"].to_numpy() > 0
    )
    abs_change = np.abs(testing[[f"delta_{i}" for i in range(1, 6)]].to_numpy(float))
    signed_alignment = np.where(direction == 1, abs_change, np.where(direction == 0, -abs_change, 0.0))
    testing["training_direction_magnitude_index"] = np.divide(
        np.nansum(signed_alignment, axis=1),
        testing["edit_l1"],
        out=np.zeros(len(testing)),
        where=testing["edit_l1"].to_numpy() > 1e-12,
    )

    bundle = json.loads(BUNDLE_SOURCE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    cases = {int(case["instance_id"]): case for case in bundle["test_pool"]}
    seen_pairs = {case["feature_pair_key"] for case in bundle["training_pool"]}
    training_pairs = {int(case["instance_id"]): case["feature_pair_key"] for case in bundle["training_pool"]}
    for iid, case in cases.items():
        instance_meta.loc[iid, "feature_pair"] = case["feature_pair_key"]
        instance_meta.loc[iid, "pair_seen_in_training"] = int(case["feature_pair_key"] in seen_pairs)
        nearest_demo_id = int(instance_meta.loc[iid, "nearest_same_training_id"])
        instance_meta.loc[iid, "nearest_demo_pair"] = training_pairs[nearest_demo_id]
        instance_meta.loc[iid, "oracle_pair_matches_nearest_demo"] = int(
            case["feature_pair_key"] == training_pairs[nearest_demo_id]
        )
        instance_meta.loc[iid, "design_cluster"] = int(case["selection_cluster"])
        instance_meta.loc[iid, "bundle_nearest_training_distance"] = float(
            case["profile_distance_to_nearest_training"]
        )
        instance_meta.loc[iid, "prediction_confidence"] = float(
            max(p["value"] for p in case["prediction"]["probabilities"])
        )
        instance_meta.loc[iid, "attribution_local_fidelity"] = float(case["attribution"]["local_fidelity"])
        pair = set(case["feature_pair_names"])
        for feature in FEATURES:
            instance_meta.loc[iid, f"oracle_pair_{feature}"] = int(feature in pair)

    for column in instance_meta.columns:
        if column not in testing.columns and column != "instance id":
            testing[column] = testing["instance id"].map(instance_meta[column])

    # Participant-level outcome and behavioral summaries.
    aggregate = {outcome: "mean" for outcome in OUTCOMES}
    aggregate.update({
        "participant training accuracy": "first",
        "CRT-2 score (0-4)": "first",
        "cognitive model v0.1 family": "first",
        "explanation reliance (η)": "first",
        "global relevance reliance (α)": "first",
        "additive change margin (ρ)": "first",
        "exemplar locality (λ)": "first",
        "remembered-change reliance (β)": "first",
        "age treated as actionable (0/1)": "first",
        "median response time < 10": "first",
        "wave": "first",
    })
    participant = testing.groupby(["participant", "xai"], as_index=False).agg(aggregate)
    participant = participant.rename(columns={
        "participant training accuracy": "training_accuracy",
        "CRT-2 score (0-4)": "crt",
        "cognitive model v0.1 family": "cognitive_family",
        "explanation reliance (η)": "eta",
        "global relevance reliance (α)": "alpha",
        "additive change margin (ρ)": "rho",
        "exemplar locality (λ)": "lambda",
        "remembered-change reliance (β)": "beta",
        "age treated as actionable (0/1)": "age_actionable",
        "median response time < 10": "fast_participant",
    })

    for i, feature in enumerate(FEATURES, start=1):
        temp = testing.assign(
            selected=(np.abs(testing[f"delta_{i}"]) > 1e-10).astype(float),
            abs_delta=np.abs(testing[f"delta_{i}"]),
            signed_delta=testing[f"delta_{i}"],
        ).groupby("participant")[["selected", "abs_delta", "signed_delta"]].mean()
        participant[f"select_{feature}"] = participant["participant"].map(temp["selected"])
        participant[f"abs_change_{feature}"] = participant["participant"].map(temp["abs_delta"])
        participant[f"signed_change_{feature}"] = participant["participant"].map(temp["signed_delta"])

    extra_records = []
    for pid, group in testing.groupby("participant"):
        subsets = []
        signs = []
        counts = np.zeros(5, dtype=float)
        for _, row in group.iterrows():
            delta = np.asarray([row[f"delta_{i}"] for i in range(1, 6)], dtype=float)
            selected = tuple(np.flatnonzero(np.abs(delta) > 1e-10).tolist())
            subsets.append(selected)
            counts[list(selected)] += 1
            signs.extend(np.sign(delta[np.abs(delta) > 1e-10]).tolist())
        probabilities = counts[counts > 0] / counts.sum() if counts.sum() else np.asarray([])
        entropy = -float(np.sum(probabilities * np.log(probabilities))) / math.log(5) if len(probabilities) else 0.0
        subset_share = Counter(subsets).most_common(1)[0][1] / len(subsets)
        sign_consistency = abs(float(np.mean(signs))) if signs else 0.0
        extra_records.append({
            "participant": pid,
            "feature_selection_entropy": entropy,
            "modal_subset_share": subset_share,
            "global_sign_consistency": sign_consistency,
        })
    participant = participant.merge(pd.DataFrame(extra_records), on="participant", how="left")
    return testing, participant, instance_meta.reset_index(drop=True)


def overall_effects(participant: pd.DataFrame) -> pd.DataFrame:
    records = []
    seed_offset = 0
    for outcome, metadata in OUTCOMES.items():
        for group_a, group_b in COMPARISONS:
            result = pairwise_test(
                participant, outcome, "xai", group_a, group_b, seed_offset=seed_offset
            )
            seed_offset += 1
            records.append({
                "outcome": outcome,
                "family": metadata["family"],
                "comparison": f"{group_a} - {group_b}",
                "group_a": group_a,
                "group_b": group_b,
                **result,
                "beneficial_difference": result["difference"] * int(metadata["better"]),
                "beneficial_hedges_g": result["hedges_g"] * int(metadata["better"]),
            })
    out = pd.DataFrame(records)
    out["q_global"] = bh_adjust(out["permutation_p"])
    out["q_within_family"] = out.groupby("family", group_keys=False)["permutation_p"].apply(bh_adjust)
    return out.sort_values(["q_global", "permutation_p", "outcome"])


def wave_and_sensitivity_effects(participant: pd.DataFrame) -> pd.DataFrame:
    contexts = {
        "full": np.ones(len(participant), dtype=bool),
        "original_wave": participant["wave"].eq("original_36").to_numpy(),
        "new_wave": participant["wave"].eq("new_26").to_numpy(),
        "exclude_fast_under_10s": ~numeric(participant["fast_participant"]).eq(1).to_numpy(),
        "training_accuracy_at_least_50pct": numeric(participant["training_accuracy"]).ge(0.5).to_numpy(),
    }
    records = []
    seed_offset = 20_000
    for context, mask in contexts.items():
        sub = participant.loc[mask]
        for outcome in KEY_MODERATOR_OUTCOMES:
            for group_a, group_b in COMPARISONS:
                result = pairwise_test(
                    sub,
                    outcome,
                    "xai",
                    group_a,
                    group_b,
                    strata_col="wave" if context not in {"original_wave", "new_wave"} else None,
                    n_perm=3_999,
                    n_boot=2_000,
                    seed_offset=seed_offset,
                )
                seed_offset += 1
                records.append({
                    "context": context,
                    "outcome": outcome,
                    "comparison": f"{group_a} - {group_b}",
                    **result,
                    "beneficial_difference": result["difference"] * int(OUTCOMES[outcome]["better"]),
                })
    out = pd.DataFrame(records)
    out["q_within_context"] = out.groupby("context", group_keys=False)["permutation_p"].apply(bh_adjust)
    return out.sort_values(["context", "q_within_context", "permutation_p"])


def create_instance_moderators(testing: pd.DataFrame, instance_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = instance_meta.copy().set_index("instance id", drop=False)
    control = testing[testing["xai"].eq("none")].groupby("instance id").agg(
        control_boundary=("boundary_new", "mean"),
        control_success=("success", "mean"),
        control_edit_l1=("edit_l1", "mean"),
    )
    meta = meta.join(control, how="left")

    def median_split(column: str, low: str, high: str) -> pd.Series:
        threshold = meta[column].median()
        return pd.Series(np.where(meta[column] <= threshold, low, high), index=meta.index)

    moderators: dict[str, pd.Series] = {
        "target_label_group": meta["target label"].astype(str),
        "original_boundary_distance_bucket": median_split(
            "boundary_original", "closer_original_boundary", "farther_original_boundary"
        ) if "boundary_original" in meta else pd.Series(index=meta.index, dtype=str),
        "nearest_same_training_distance_bucket": median_split(
            "nearest_same_training_distance", "near_training", "far_training"
        ),
        "pair_seen_in_training_group": meta["pair_seen_in_training"].map({1: "seen_pair", 0: "novel_pair"}),
        "nearest_demo_pair_match_group": meta["oracle_pair_matches_nearest_demo"].map(
            {1: "pair_matches_nearest_demo", 0: "pair_differs_from_nearest_demo"}
        ),
        "design_cluster_group": meta["design_cluster"].map({1: "cluster_1", 2: "cluster_2"}),
        "prediction_confidence_bucket": median_split("prediction_confidence", "lower_confidence", "higher_confidence"),
        "control_empirical_difficulty_group": median_split("control_boundary", "control_easy", "control_hard"),
        "oracle_pair_uses_age_group": meta["oracle_pair_Age"].map({1.0: "age_pair", 0.0: "no_age_pair"}),
    }

    # Exogenous geometry clustering of the 20 test instances.
    geometry_cols = [
        *[f"original_{name}" for name in FEATURES],
        "nearest_same_training_distance",
        "original_to_target_centroid",
        "original_to_source_centroid",
        "demo_delta_l1",
        "prediction_confidence",
        "attribution_local_fidelity",
        *[f"oracle_pair_{name}" for name in FEATURES],
    ]
    x = StandardScaler().fit_transform(meta[geometry_cols].astype(float))
    diagnostics = []
    labels_by_k: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(SEED + 30000)
    for k in range(2, 6):
        base = KMeans(n_clusters=k, n_init=50, random_state=SEED + k).fit_predict(x)
        labels_by_k[k] = base
        aris = []
        for _ in range(100):
            cols = rng.choice(x.shape[1], size=x.shape[1], replace=True)
            boot = KMeans(n_clusters=k, n_init=20, random_state=int(rng.integers(1_000_000))).fit_predict(x[:, cols])
            aris.append(adjusted_rand_score(base, boot))
        diagnostics.append({
            "cluster_type": "exogenous_instance_geometry",
            "k": k,
            "silhouette": silhouette_score(x, base),
            "feature_bootstrap_ari": float(np.mean(aris)),
            "minimum_cluster_size": int(pd.Series(base).value_counts().min()),
            "cluster_sizes": json.dumps({int(key + 1): int(value) for key, value in pd.Series(base).value_counts().sort_index().items()}),
        })
    eligible = [row for row in diagnostics if row["minimum_cluster_size"] >= 4]
    best = max(eligible, key=lambda row: (row["silhouette"] + row["feature_bootstrap_ari"]) / 2)
    moderators["exogenous_geometry_best_cluster"] = pd.Series(
        [f"geometry_{label + 1}" for label in labels_by_k[int(best["k"])]], index=meta.index
    )
    moderators["exogenous_geometry_k2"] = pd.Series(
        [f"geometry_{label + 1}" for label in labels_by_k[2]], index=meta.index
    )

    for name, values in moderators.items():
        meta[name] = values
        testing[name] = testing["instance id"].map(values)
    return meta.reset_index(drop=True), pd.DataFrame(diagnostics)


def moderator_effects(testing: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    moderator_names = [
        "target_label_group",
        "original_boundary_distance_bucket",
        "nearest_same_training_distance_bucket",
        "pair_seen_in_training_group",
        "nearest_demo_pair_match_group",
        "design_cluster_group",
        "prediction_confidence_bucket",
        "control_empirical_difficulty_group",
        "oracle_pair_uses_age_group",
        "exogenous_geometry_k2",
    ]
    subgroup_records = []
    interaction_records = []
    seed_offset = 40_000
    for moderator in moderator_names:
        levels = [str(level) for level in pd.Series(testing[moderator].dropna().unique()).sort_values()]
        if len(levels) != 2:
            continue
        level_a, level_b = levels
        instance_counts = testing.drop_duplicates("instance id")[moderator].value_counts()
        for outcome in KEY_MODERATOR_OUTCOMES:
            per_level = testing.groupby(["participant", "xai", "wave", moderator], as_index=False)[outcome].mean()
            for level in levels:
                sub = per_level[per_level[moderator].astype(str).eq(level)]
                for group_a, group_b in COMPARISONS:
                    result = pairwise_test(
                        sub, outcome, "xai", group_a, group_b,
                        n_perm=2_999, n_boot=1_500, seed_offset=seed_offset,
                    )
                    seed_offset += 1
                    subgroup_records.append({
                        "moderator": moderator,
                        "level": level,
                        "n_instances": int(instance_counts.get(level, 0)),
                        "outcome": outcome,
                        "comparison": f"{group_a} - {group_b}",
                        **result,
                        "beneficial_difference": result["difference"] * int(OUTCOMES[outcome]["better"]),
                    })

            pivot = per_level.pivot_table(
                index=["participant", "xai", "wave"], columns=moderator, values=outcome
            ).reset_index()
            if level_a not in pivot or level_b not in pivot:
                continue
            pivot["within_person_level_contrast"] = pivot[level_a] - pivot[level_b]
            for group_a, group_b in COMPARISONS:
                result = pairwise_test(
                    pivot,
                    "within_person_level_contrast",
                    "xai",
                    group_a,
                    group_b,
                    n_perm=4_999,
                    n_boot=2_000,
                    seed_offset=seed_offset,
                )
                seed_offset += 1
                interaction_records.append({
                    "moderator": moderator,
                    "level_contrast": f"{level_a} - {level_b}",
                    "instances_level_a": int(instance_counts.get(level_a, 0)),
                    "instances_level_b": int(instance_counts.get(level_b, 0)),
                    "outcome": outcome,
                    "condition_comparison": f"{group_a} - {group_b}",
                    **result,
                })
    subgroup = pd.DataFrame(subgroup_records)
    interaction = pd.DataFrame(interaction_records)
    subgroup["q_global"] = bh_adjust(subgroup["permutation_p"])
    subgroup["q_within_moderator"] = subgroup.groupby("moderator", group_keys=False)["permutation_p"].apply(bh_adjust)
    interaction["q_global"] = bh_adjust(interaction["permutation_p"])
    interaction["q_within_moderator"] = interaction.groupby("moderator", group_keys=False)["permutation_p"].apply(bh_adjust)
    return subgroup.sort_values(["q_global", "permutation_p"]), interaction.sort_values(["q_global", "permutation_p"])


def instance_effects(testing: pd.DataFrame) -> pd.DataFrame:
    records = []
    seed_offset = 80_000
    for iid, group in testing.groupby("instance id"):
        for outcome in KEY_MODERATOR_OUTCOMES:
            participant_rows = group[["participant", "xai", "wave", outcome]].copy()
            for group_a, group_b in COMPARISONS:
                result = pairwise_test(
                    participant_rows, outcome, "xai", group_a, group_b,
                    n_perm=1_999, n_boot=1_000, seed_offset=seed_offset,
                )
                seed_offset += 1
                wave_diffs = {}
                for wave in ["original_36", "new_26"]:
                    w = participant_rows[participant_rows["wave"].eq(wave)]
                    av = numeric(w.loc[w["xai"].eq(group_a), outcome]).dropna()
                    bv = numeric(w.loc[w["xai"].eq(group_b), outcome]).dropna()
                    wave_diffs[wave] = float(av.mean() - bv.mean()) if len(av) and len(bv) else np.nan
                records.append({
                    "instance id": int(iid),
                    "target label": str(group["target label"].iloc[0]),
                    "feature_pair": str(group["feature_pair"].iloc[0]),
                    "design_cluster": str(group["design_cluster"].iloc[0]),
                    "outcome": outcome,
                    "comparison": f"{group_a} - {group_b}",
                    **result,
                    "beneficial_difference": result["difference"] * int(OUTCOMES[outcome]["better"]),
                    "original_wave_difference": wave_diffs["original_36"],
                    "new_wave_difference": wave_diffs["new_26"],
                    "same_direction_both_waves": bool(
                        np.isfinite(wave_diffs["original_36"])
                        and np.isfinite(wave_diffs["new_26"])
                        and np.sign(wave_diffs["original_36"]) == np.sign(wave_diffs["new_26"])
                    ),
                })
    out = pd.DataFrame(records)
    out["q_global"] = bh_adjust(out["permutation_p"])
    out["q_within_outcome_comparison"] = out.groupby(
        ["outcome", "comparison"], group_keys=False
    )["permutation_p"].apply(bh_adjust)
    return out.sort_values(["q_global", "permutation_p"])


def participant_clusters(participant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_features = [
        *[f"select_{name}" for name in FEATURES],
        *[f"abs_change_{name}" for name in FEATURES],
        *[f"signed_change_{name}" for name in FEATURES],
        "training_direction_agreement",
        "feature_selection_entropy",
        "modal_subset_share",
        "global_sign_consistency",
    ]
    x_frame = participant[cluster_features].apply(numeric)
    x_frame = x_frame.fillna(x_frame.median())
    x = StandardScaler().fit_transform(x_frame)
    diagnostics = []
    labels_by_k: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(SEED + 100_000)
    for k in range(2, 7):
        base = KMeans(n_clusters=k, n_init=80, random_state=SEED + k).fit_predict(x)
        labels_by_k[k] = base
        aris = []
        for _ in range(100):
            cols = rng.choice(x.shape[1], size=x.shape[1], replace=True)
            alt = KMeans(n_clusters=k, n_init=30, random_state=int(rng.integers(1_000_000))).fit_predict(x[:, cols])
            aris.append(adjusted_rand_score(base, alt))
        sizes = pd.Series(base).value_counts().sort_index()
        diagnostics.append({
            "cluster_type": "participant_edit_strategy",
            "k": k,
            "silhouette": silhouette_score(x, base),
            "feature_bootstrap_ari": float(np.mean(aris)),
            "minimum_cluster_size": int(sizes.min()),
            "cluster_sizes": json.dumps({int(key + 1): int(value) for key, value in sizes.items()}),
        })
    eligible = [row for row in diagnostics if row["minimum_cluster_size"] >= 6]
    best = max(eligible, key=lambda row: (row["silhouette"] + row["feature_bootstrap_ari"]) / 2)
    best_k = int(best["k"])
    clustered = participant.copy()
    clustered["strategy_cluster"] = labels_by_k[best_k] + 1

    summary_aggs = {
        "participant": "size",
        **{column: "mean" for column in cluster_features},
        **{outcome: "mean" for outcome in KEY_MODERATOR_OUTCOMES},
        "xai": lambda s: json.dumps({str(k): int(v) for k, v in s.value_counts().sort_index().items()}),
        "cognitive_family": lambda s: json.dumps({str(k): int(v) for k, v in s.value_counts().sort_index().items()}),
    }
    summary = clustered.groupby("strategy_cluster").agg(summary_aggs).rename(
        columns={"participant": "n", "xai": "xai_counts", "cognitive_family": "cognitive_family_counts"}
    ).reset_index()

    table = pd.crosstab(clustered["xai"], clustered["strategy_cluster"])
    chi2 = float(stats.chi2_contingency(table, correction=False)[0])
    rng = np.random.default_rng(SEED + 110_000)
    labels = clustered["xai"].to_numpy(object)
    clusters = clustered["strategy_cluster"].to_numpy(int)
    extreme = 0
    for _ in range(N_PERM):
        perm = rng.permutation(labels)
        perm_table = pd.crosstab(perm, clusters)
        stat = float(stats.chi2_contingency(perm_table, correction=False)[0])
        extreme += stat >= chi2 - 1e-15
    assoc = pd.DataFrame([{
        "best_k": best_k,
        "chi_square": chi2,
        "permutation_p": (extreme + 1) / (N_PERM + 1),
        "cramers_v": math.sqrt(chi2 / (len(clustered) * min(table.shape[0] - 1, table.shape[1] - 1))),
        "table": json.dumps({str(i): {str(j): int(table.loc[i, j]) for j in table.columns} for i in table.index}),
    }])

    enrichment = []
    for cluster in sorted(clustered["strategy_cluster"].unique()):
        in_cluster = clustered["strategy_cluster"].eq(cluster)
        for condition in CONDITIONS:
            is_condition = clustered["xai"].eq(condition)
            contingency = np.asarray([
                [np.sum(in_cluster & is_condition), np.sum(in_cluster & ~is_condition)],
                [np.sum(~in_cluster & is_condition), np.sum(~in_cluster & ~is_condition)],
            ])
            odds, p = stats.fisher_exact(contingency)
            enrichment.append({
                "strategy_cluster": int(cluster),
                "condition": condition,
                "cluster_n": int(in_cluster.sum()),
                "condition_in_cluster": int(np.sum(in_cluster & is_condition)),
                "condition_rate_in_cluster": float(np.mean(is_condition[in_cluster])),
                "condition_rate_outside": float(np.mean(is_condition[~in_cluster])),
                "odds_ratio": float(odds),
                "fisher_p": float(p),
            })
    enrichment = pd.DataFrame(enrichment)
    enrichment["q"] = bh_adjust(enrichment["fisher_p"])
    return clustered, pd.DataFrame(diagnostics), summary, pd.concat([assoc, enrichment], ignore_index=True, sort=False)


def cognitive_subgroups(participant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = [
        ("counterfactual", "cognitive_family", "weighted examples", "feature contribution"),
        ("counterfactual", "eta", 1.0, 0.5),
        ("counterfactual", "age_actionable", 1.0, 0.0),
        ("attribution", "cognitive_family", "weighted examples", "feature contribution"),
        ("attribution", "eta", 1.0, 0.0),
        ("none", "cognitive_family", "weighted examples", "feature contribution"),
    ]
    records = []
    seed_offset = 120_000
    for condition, parameter, a, b in definitions:
        sub = participant[participant["xai"].eq(condition)].copy()
        for outcome in KEY_MODERATOR_OUTCOMES + [
            "training_direction_agreement", "feature_selection_entropy", "modal_subset_share"
        ]:
            result = pairwise_test(
                sub, outcome, parameter, a, b, strata_col="wave",
                n_perm=4_999, n_boot=2_000, seed_offset=seed_offset,
            )
            seed_offset += 1
            records.append({
                "xai": condition,
                "parameter": parameter,
                "comparison": f"{a} - {b}",
                "outcome": outcome,
                **result,
            })
    subgroups = pd.DataFrame(records)
    subgroups["q_global"] = bh_adjust(subgroups["permutation_p"])
    subgroups["q_within_condition_parameter"] = subgroups.groupby(
        ["xai", "parameter"], group_keys=False
    )["permutation_p"].apply(bh_adjust)

    correlation_records = []
    parameters = ["eta", "alpha", "rho", "lambda", "beta", "age_actionable"]
    for condition in CONDITIONS:
        sub = participant[participant["xai"].eq(condition)]
        for parameter in parameters:
            for outcome in KEY_MODERATOR_OUTCOMES:
                work = sub[[parameter, outcome]].apply(numeric).dropna()
                if len(work) < 8 or work[parameter].nunique() < 2 or work[outcome].nunique() < 2:
                    continue
                rho, p = stats.spearmanr(work[parameter], work[outcome])
                correlation_records.append({
                    "xai": condition,
                    "parameter": parameter,
                    "outcome": outcome,
                    "n": len(work),
                    "spearman_rho": float(rho),
                    "p": float(p),
                })
    correlations = pd.DataFrame(correlation_records)
    correlations["q_global"] = bh_adjust(correlations["p"])
    correlations["q_within_xai"] = correlations.groupby("xai", group_keys=False)["p"].apply(bh_adjust)
    return subgroups.sort_values(["q_global", "permutation_p"]), correlations.sort_values(["q_global", "p"])


def trends(testing: pd.DataFrame, data_participant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    slope_specs = {
        "test_trial": ("trial number", [
            "boundary_new", "boundary_improvement", "success", "confidence_gain", "edit_l1",
            "target_helpful_shap", "cf_demo_cosine", "attribution_feature_overlap", "response_time",
        ]),
        "nearest_training_distance": ("nearest_same_training_distance", KEY_MODERATOR_OUTCOMES),
        "original_boundary_distance": ("boundary_original", KEY_MODERATOR_OUTCOMES),
    }
    slopes = []
    for slope_type, (x_col, outcomes) in slope_specs.items():
        for (pid, xai, wave), group in testing.groupby(["participant", "xai", "wave"]):
            for outcome in outcomes:
                slopes.append({
                    "participant": pid,
                    "xai": xai,
                    "wave": wave,
                    "slope_type": slope_type,
                    "outcome": outcome,
                    "slope": participant_slope(group, outcome, x_col),
                })
    slopes = pd.DataFrame(slopes)
    records = []
    seed_offset = 140_000
    for (slope_type, outcome), group in slopes.groupby(["slope_type", "outcome"]):
        for group_a, group_b in COMPARISONS:
            result = pairwise_test(
                group, "slope", "xai", group_a, group_b,
                n_perm=3_999, n_boot=2_000, seed_offset=seed_offset,
            )
            seed_offset += 1
            records.append({
                "slope_type": slope_type,
                "outcome": outcome,
                "comparison": f"{group_a} - {group_b}",
                **result,
            })
    effects = pd.DataFrame(records)
    effects["q_global"] = bh_adjust(effects["permutation_p"])
    effects["q_within_slope_type"] = effects.groupby("slope_type", group_keys=False)["permutation_p"].apply(bh_adjust)
    return slopes, effects.sort_values(["q_global", "permutation_p"])


def jackknife(testing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in KEY_MODERATOR_OUTCOMES:
        for group_a, group_b in COMPARISONS:
            estimates = []
            for omitted in sorted(testing["instance id"].unique()):
                p = testing[testing["instance id"].ne(omitted)].groupby(["participant", "xai"], as_index=False)[outcome].mean()
                a = p.loc[p["xai"].eq(group_a), outcome].mean()
                b = p.loc[p["xai"].eq(group_b), outcome].mean()
                estimates.append(a - b)
            full = testing.groupby(["participant", "xai"], as_index=False)[outcome].mean()
            full_diff = full.loc[full["xai"].eq(group_a), outcome].mean() - full.loc[full["xai"].eq(group_b), outcome].mean()
            estimates = np.asarray(estimates, dtype=float)
            rows.append({
                "outcome": outcome,
                "comparison": f"{group_a} - {group_b}",
                "full_difference": full_diff,
                "leave_one_instance_min": float(np.nanmin(estimates)),
                "leave_one_instance_max": float(np.nanmax(estimates)),
                "same_sign_fraction": float(np.mean(np.sign(estimates) == np.sign(full_diff))),
            })
    return pd.DataFrame(rows)


def make_figures(overall: pd.DataFrame, interactions: pd.DataFrame, clusters: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    focus = overall[
        overall["outcome"].isin(KEY_MODERATOR_OUTCOMES)
        & overall["comparison"].isin(["counterfactual - none", "attribution - none"])
    ].copy()
    focus["beneficial_ci_low"] = np.where(
        focus["beneficial_difference"].ge(0),
        np.minimum(focus["ci_low"] * focus["outcome"].map(lambda x: OUTCOMES[x]["better"]), focus["ci_high"] * focus["outcome"].map(lambda x: OUTCOMES[x]["better"])),
        np.minimum(focus["ci_low"] * focus["outcome"].map(lambda x: OUTCOMES[x]["better"]), focus["ci_high"] * focus["outcome"].map(lambda x: OUTCOMES[x]["better"])),
    )
    focus["beneficial_ci_high"] = np.maximum(
        focus["ci_low"] * focus["outcome"].map(lambda x: OUTCOMES[x]["better"]),
        focus["ci_high"] * focus["outcome"].map(lambda x: OUTCOMES[x]["better"]),
    )
    focus = focus.sort_values(["comparison", "beneficial_hedges_g"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, comparison in zip(axes, ["counterfactual - none", "attribution - none"]):
        plot = focus[focus["comparison"].eq(comparison)].sort_values("beneficial_hedges_g")
        ax.scatter(plot["beneficial_hedges_g"], plot["outcome"], c=np.where(plot["permutation_p"] < .05, "#b13a2f", "#355c7d"))
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(comparison)
        ax.set_xlabel("Hedges g, positive favors first condition")
    fig.suptitle("Exploratory participant-level XAI effects")
    fig.tight_layout()
    fig.savefig(OUTDIR / "overall_effect_sizes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    heat = interactions[interactions["condition_comparison"].isin([
        "counterfactual - none", "attribution - none"
    ])].copy()
    if not heat.empty:
        heat["signed_score"] = np.sign(heat["difference"]) * -np.log10(heat["permutation_p"].clip(lower=1e-4))
        pivot = heat.pivot_table(
            index=["moderator", "outcome"], columns="condition_comparison", values="signed_score", aggfunc="first"
        )
        fig, ax = plt.subplots(figsize=(8, max(8, len(pivot) * .22)))
        sns.heatmap(pivot, center=0, cmap="vlag", ax=ax, cbar_kws={"label": "signed -log10(permutation p)"})
        ax.set_title("Matched moderator interactions (exploratory)")
        fig.tight_layout()
        fig.savefig(OUTDIR / "moderator_interaction_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    profile_cols = [f"select_{name}" for name in FEATURES] + [
        "training_direction_agreement", "feature_selection_entropy", "modal_subset_share"
    ]
    if not clusters.empty:
        profile = clusters.set_index("strategy_cluster")[profile_cols]
        fig, ax = plt.subplots(figsize=(10, max(3, len(profile) * .8)))
        sns.heatmap(profile, annot=True, fmt=".2f", cmap="crest", ax=ax)
        ax.set_title("Participant strategy-cluster profiles")
        fig.tight_layout()
        fig.savefig(OUTDIR / "participant_cluster_profiles.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    testing, participant, instance_meta = prepare_data()
    instance_meta["boundary_original"] = instance_meta["instance id"].map(
        testing.drop_duplicates("instance id").set_index("instance id")["boundary_original"]
    )

    overall = overall_effects(participant)
    sensitivity = wave_and_sensitivity_effects(participant)
    instance_meta, instance_cluster_diagnostics = create_instance_moderators(testing, instance_meta)
    subgroup, interactions = moderator_effects(testing, instance_meta)
    per_instance = instance_effects(testing)
    clustered_participants, participant_cluster_diagnostics, cluster_summary, cluster_association = participant_clusters(participant)
    cognitive_groups, cognitive_correlations = cognitive_subgroups(participant)
    slope_rows, trend_effects = trends(testing, participant)
    jackknife_results = jackknife(testing)

    testing.to_csv(OUTDIR / "testing_rows_enriched.csv", index=False)
    participant.to_csv(OUTDIR / "participant_summary.csv", index=False)
    instance_meta.to_csv(OUTDIR / "instance_metadata_and_clusters.csv", index=False)
    overall.to_csv(OUTDIR / "overall_condition_effects.csv", index=False)
    sensitivity.to_csv(OUTDIR / "wave_and_sensitivity_effects.csv", index=False)
    subgroup.to_csv(OUTDIR / "moderator_subgroup_effects.csv", index=False)
    interactions.to_csv(OUTDIR / "moderator_interactions.csv", index=False)
    per_instance.to_csv(OUTDIR / "instance_specific_effects.csv", index=False)
    clustered_participants.to_csv(OUTDIR / "participant_cluster_assignments.csv", index=False)
    participant_cluster_diagnostics.to_csv(OUTDIR / "participant_cluster_diagnostics.csv", index=False)
    instance_cluster_diagnostics.to_csv(OUTDIR / "instance_cluster_diagnostics.csv", index=False)
    cluster_summary.to_csv(OUTDIR / "participant_cluster_summary.csv", index=False)
    cluster_association.to_csv(OUTDIR / "participant_cluster_condition_association.csv", index=False)
    cognitive_groups.to_csv(OUTDIR / "cognitive_parameter_subgroups.csv", index=False)
    cognitive_correlations.to_csv(OUTDIR / "cognitive_parameter_correlations.csv", index=False)
    slope_rows.to_csv(OUTDIR / "participant_slopes.csv", index=False)
    trend_effects.to_csv(OUTDIR / "trend_and_distance_slope_effects.csv", index=False)
    jackknife_results.to_csv(OUTDIR / "leave_one_instance_out_effects.csv", index=False)
    make_figures(overall, interactions, cluster_summary)

    summary = {
        "canonical_source": str(SOURCE.relative_to(ROOT)),
        "rows": int(len(testing)),
        "participants": int(participant["participant"].nunique()),
        "condition_counts": {str(k): int(v) for k, v in participant["xai"].value_counts().items()},
        "wave_counts": {str(k): int(v) for k, v in participant["wave"].value_counts().items()},
        "overall_nominal_p_under_05": int((overall["permutation_p"] < .05).sum()),
        "overall_q_under_10": int((overall["q_global"] < .10).sum()),
        "moderator_nominal_p_under_05": int((interactions["permutation_p"] < .05).sum()),
        "moderator_q_under_10": int((interactions["q_global"] < .10).sum()),
        "instance_nominal_p_under_05": int((per_instance["permutation_p"] < .05).sum()),
        "instance_q_under_10": int((per_instance["q_global"] < .10).sum()),
        "participant_cluster_best_k": int(cluster_association.iloc[0]["best_k"]),
        "participant_cluster_condition_p": float(cluster_association.iloc[0]["permutation_p"]),
    }
    (OUTDIR / "analysis_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nTOP OVERALL EFFECTS")
    print(overall.head(20).round(4).to_string(index=False))
    print("\nTOP MODERATOR INTERACTIONS")
    print(interactions.head(20).round(4).to_string(index=False))
    print("\nTOP COGNITIVE SUBGROUP DIFFERENCES")
    print(cognitive_groups.head(20).round(4).to_string(index=False))
    print("\nTOP TREND/DISTANCE SLOPE EFFECTS")
    print(trend_effects.head(20).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
