"""Select reciprocal diabetes groups using discrete LIME explanations.

This is deliberately a shortlist, not the final experimental selection. It uses
discrete LIME, the existing exact-two-feature counterfactual
optimizer, and the user's perpendicular-plane criterion.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xai_methods.counterfactual import generate_counterfactual  # noqa: E402


FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
DISPLAY = ["Glucose", "Blood Pressure", "Insulin", "BMI", "Age"]
LABELS = ["Diabetes", "No Diabetes"]
NUM_SAMPLES = 500
SHORTLIST_PER_SIDE = None
TARGET_GROUP_PER_SIDE = 3
SELECTED_CLUSTER_COUNT = 2
MIN_NORMALIZED_CHANGE = 0.10
MAX_ACCURACY_DROP = 0.02


def normalized(values: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    return (values - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])


def main() -> None:
    frames = []
    for split in ("train", "dev", "test"):
        frame = pd.read_csv(ROOT / f"src/data/diabetes/{split}.csv")
        frame["split"] = split
        frame["split_index"] = np.arange(len(frame), dtype=int)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    train = frames[0]
    baseline_model = joblib.load(ROOT / "src/ai_models/saved_models/diabetes_mlp.joblib")
    baseline_scores = {
        split: accuracy_score(frame["target"].astype(int), baseline_model.predict(frame[FEATURES]))
        for split, frame in zip(("train", "dev", "test"), frames)
    }
    cached_model = ROOT / "analysis/diabetes_mlp_regularized.joblib"
    if cached_model.exists():
        model = joblib.load(cached_model)
        chosen_alpha = float(model.named_steps["model"].alpha)
        chosen_scores = {
            split: accuracy_score(frame["target"].astype(int), model.predict(frame[FEATURES]))
            for split, frame in zip(("train", "dev", "test"), frames)
        }
        accuracy_rows = [(chosen_alpha, model, chosen_scores)]
    else:
        accuracy_rows = []
        for alpha in (0.002, 0.005, 0.01):
            candidate_model = clone(baseline_model)
            candidate_model.set_params(model__alpha=alpha)
            candidate_model.fit(train[FEATURES], train["target"].astype(int))
            scores = {
                split: accuracy_score(frame["target"].astype(int), candidate_model.predict(frame[FEATURES]))
                for split, frame in zip(("train", "dev", "test"), frames)
            }
            accuracy_rows.append((alpha, candidate_model, scores))
        acceptable = [
            row for row in accuracy_rows
            if row[2]["dev"] >= baseline_scores["dev"] - MAX_ACCURACY_DROP
            and row[2]["test"] >= baseline_scores["test"] - MAX_ACCURACY_DROP
        ]
        chosen_alpha, model, chosen_scores = max(
            acceptable or accuracy_rows,
            key=lambda row: (row[0] if acceptable else row[2]["dev"], row[2]["test"]),
        )
    model_output = Path.home() / "AppData/Local/Temp/diabetes_mlp_regularized.joblib"
    joblib.dump(model, model_output)
    with (ROOT / "static/experiment-data.json").open(encoding="utf-8") as handle:
        bundle = json.load(handle)
    ranges = np.asarray(bundle["datasets"]["diabetes"]["training_pool"][0]["raw_feature_ranges"], dtype=float)

    X = data[FEATURES].copy()
    predictions = np.asarray(model.predict(X), dtype=int)
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    data["prediction"] = predictions
    data["boundary_margin"] = np.abs(probabilities[:, 0] - 0.5)

    def predict_array(values: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(values, columns=FEATURES))

    explainer = LimeTabularExplainer(
        train[FEATURES].to_numpy(float),
        feature_names=DISPLAY,
        class_names=LABELS,
        mode="classification",
        discretize_continuous=True,
        sample_around_instance=True,
        feature_selection="lasso_path",
        random_state=42,
    )

    lime_rows = []
    for index, row in data.iterrows():
        # Reset the generator per instance so results do not depend on dataset order.
        explainer.random_state = np.random.RandomState(42)
        predicted = int(row["prediction"])
        explanation = explainer.explain_instance(
            row[FEATURES].to_numpy(float),
            predict_array,
            labels=(predicted,),
            num_features=2,
            num_samples=NUM_SAMPLES,
        )
        weights = sorted(explanation.local_exp[predicted], key=lambda item: -abs(item[1]))[:2]
        pair = tuple(sorted(int(item[0]) for item in weights))
        weight_vector = np.zeros(len(FEATURES), dtype=float)
        for feature_index, weight in weights:
            weight_vector[int(feature_index)] = float(weight)
        lime_rows.append({
            "index": int(index),
            "pair": pair,
            "weights": weight_vector.tolist(),
            "conditions": list(explanation.domain_mapper.discretized_feature_names),
            "score": float(explanation.score),
        })

    counts = Counter((record["pair"], int(data.loc[record["index"], "prediction"])) for record in lime_rows)
    pair_totals = []
    for pair in sorted({record["pair"] for record in lime_rows}):
        side_counts = [counts[(pair, side)] for side in (0, 1)]
        if min(side_counts) >= 6:
            pair_totals.append((min(side_counts), sum(side_counts), pair, side_counts))
    pair_totals.sort(reverse=True)
    candidate_pairs = [record[2] for record in pair_totals]

    lime_by_index = {record["index"]: record for record in lime_rows}
    candidates = []
    for pair in candidate_pairs:
        for side in (0, 1):
            eligible = [
                record for record in lime_rows
                if record["pair"] == pair and int(data.loc[record["index"], "prediction"]) == side
            ]
            # Exhaustive within every sufficiently populated LIME pair; no
            # boundary-distance ranking or cutoff is applied.
            eligible.sort(key=lambda record: record["index"])
            for record in eligible:
                index = record["index"]
                reference = data.loc[[index], FEATURES].copy()
                target = 1 - side
                target_frame = data.loc[data["prediction"] == target, FEATURES].copy()
                cf = generate_counterfactual(
                    estimator=model,
                    reference_frame=reference,
                    target_distribution_frame=target_frame,
                    feature_names=FEATURES,
                    feature_types=["numerical"] * len(FEATURES),
                    feature_ranges=ranges.tolist(),
                    class_labels=LABELS,
                    shap_values=record["weights"],
                    top_k=2,
                    selected_feature_indices=list(pair),
                    generation_mode="minimal",
                )
                if cf is None:
                    continue
                original = reference.iloc[0].to_numpy(float)
                counterfactual = np.asarray(cf["feature_values"], dtype=float)
                # Make both edits visually observable: retain the optimizer's
                # direction but expand each selected edit to at least 5% of its
                # feature range. Reject it if the expanded edit stops helping or
                # the combined profile no longer flips.
                for feature_index in pair:
                    raw_delta = counterfactual[feature_index] - original[feature_index]
                    if abs(raw_delta) <= 1e-12:
                        counterfactual = None
                        break
                    span = ranges[feature_index, 1] - ranges[feature_index, 0]
                    minimum = MIN_NORMALIZED_CHANGE * span
                    if abs(raw_delta) < minimum:
                        expanded = original[feature_index] + np.sign(raw_delta) * minimum
                        if pd.api.types.is_integer_dtype(data[FEATURES[feature_index]].dtype):
                            expanded = round(expanded)
                        counterfactual[feature_index] = np.clip(
                            expanded, ranges[feature_index, 0], ranges[feature_index, 1]
                        )
                if counterfactual is None:
                    continue
                base_target_probability = float(model.predict_proba(reference)[0, target])
                individually_supporting = True
                for feature_index in pair:
                    single = original.copy()
                    single[feature_index] = counterfactual[feature_index]
                    single_frame = pd.DataFrame([single], columns=FEATURES)
                    if float(model.predict_proba(single_frame)[0, target]) <= base_target_probability:
                        individually_supporting = False
                        break
                cf_frame = pd.DataFrame([counterfactual], columns=FEATURES)
                if not individually_supporting or int(model.predict(cf_frame)[0]) != target:
                    continue
                norm_original = normalized(original, ranges)
                norm_cf = normalized(counterfactual, ranges)
                delta = norm_cf - norm_original
                oriented_delta = delta[list(pair)] if side == 0 else -delta[list(pair)]
                candidates.append({
                    "index": index,
                    "split": str(data.loc[index, "split"]),
                    "row_id": int(data.loc[index, "row_id"]),
                    "source_instance_id": int(data.loc[index, "split_index"]),
                    "prediction": side,
                    "pair": pair,
                    "lime_values": (
                        np.asarray(record["weights"], dtype=float)
                        * (-1.0 if side == 0 else 1.0)
                    ).tolist(),
                    "lime_feature_conditions": record["conditions"],
                    "lime_score": record["score"],
                    "margin": float(data.loc[index, "boundary_margin"]),
                    "original": original,
                    "counterfactual": counterfactual,
                    "norm_original": norm_original,
                    "norm_cf": norm_cf,
                    "delta": delta,
                    "oriented_sign": tuple(int(np.sign(value)) for value in oriented_delta),
                    "target_probability": float(model.predict_proba(cf_frame)[0, target]),
                })

    by_pair = defaultdict(list)
    for candidate in candidates:
        by_pair[candidate["pair"]].append(candidate)

    pair_results = []
    for pair, rows in by_pair.items():
        if min(sum(row["prediction"] == side for row in rows) for side in (0, 1)) < TARGET_GROUP_PER_SIDE:
            continue
        # Cluster the requested joint representation: label-adjusted LIME
        # coefficients plus the normalized endpoint on the two changed features.
        vectors = np.asarray([
            [*np.asarray(row["lime_values"])[list(pair)], *row["norm_cf"][list(pair)]]
            for row in rows
        ], dtype=float)
        scaled = StandardScaler().fit_transform(vectors)
        scaled_index_by_candidate = {row["index"]: index for index, row in enumerate(rows)}
        seen_groups = set()
        for eps in np.linspace(0.45, 2.5, 18):
            labels = DBSCAN(eps=float(eps), min_samples=6).fit_predict(scaled)
            for cluster_label in sorted(set(labels) - {-1}):
                member_indices = np.flatnonzero(labels == cluster_label)
                raw_members = [rows[index] for index in member_indices]
                for sign in sorted({row["oriented_sign"] for row in raw_members}):
                    if 0 in sign:
                        continue
                    members = [row for row in raw_members if row["oriented_sign"] == sign]
                    if min(sum(row["prediction"] == side for row in members) for side in (0, 1)) < TARGET_GROUP_PER_SIDE:
                        continue
                    # Preserve the earlier reciprocal-side requirement: relative
                    # to a common endpoint centre, Diabetes originals must be on
                    # the pre-change side in both features and No-Diabetes
                    # originals on the opposite side.
                    sign_array = np.asarray(sign, dtype=float)
                    for _ in range(3):
                        endpoint_centre = np.mean([
                            row["norm_cf"][list(pair)] for row in members
                        ], axis=0)
                        filtered = []
                        for row in members:
                            position = (
                                row["norm_original"][list(pair)] - endpoint_centre
                            ) * sign_array
                            proper_side = np.all(position < 0) if row["prediction"] == 0 else np.all(position > 0)
                            if proper_side:
                                filtered.append(row)
                        if len(filtered) == len(members):
                            break
                        members = filtered
                    if min(sum(row["prediction"] == side for row in members) for side in (0, 1)) < TARGET_GROUP_PER_SIDE:
                        continue
                    group_key = tuple(sorted(row["index"] for row in members))
                    if group_key in seen_groups:
                        continue
                    seen_groups.add(group_key)
                    selected_indices = np.asarray([
                        scaled_index_by_candidate[row["index"]] for row in members
                    ], dtype=int)
                    centroid = np.mean(scaled[selected_indices], axis=0)
                    distances = np.linalg.norm(scaled[selected_indices] - centroid, axis=1)
                    for row, distance in zip(members, distances):
                        row["cluster_distance"] = float(distance)
                    pair_results.append({
                        "pair": pair,
                        "sign": sign,
                        "endpoint_centre": endpoint_centre,
                        "rows": members,
                        "eps": float(eps),
                        "compactness": float(np.mean(distances)),
                        "radius": float(np.max(distances)),
                        "selected": {
                            side: sorted(
                                [row for row in members if row["prediction"] == side],
                                key=lambda row: row["cluster_distance"],
                            )
                            for side in (0, 1)
                        },
                    })

    pair_results.sort(
        key=lambda result: (
            result["compactness"],
            -min(len(result["selected"][0]), len(result["selected"][1])),
        )
    )
    if len(pair_results) < SELECTED_CLUSTER_COUNT:
        raise RuntimeError("No shared-threshold group with at least three profiles per prediction found")
    age_result = next(
        (result for result in pair_results if FEATURES.index("age") in result["pair"]),
        None,
    )
    if age_result is None:
        raise RuntimeError("No qualifying cluster changes Age")
    second_result = next(
        (result for result in pair_results if result["pair"] != age_result["pair"]),
        None,
    )
    if second_result is None:
        raise RuntimeError("No second qualifying feature-pair cluster found")
    selected_results = [age_result, second_result]

    # Jointly choose each stratum's three training demonstrations and five
    # testing cases to make the nearest-demonstration edit reusable. Candidate
    # training cases must be DBSCAN members; tests must share the exact
    # discrete-LIME feature pair and counterfactual direction pattern.
    test_quotas = {0: [5, 5], 1: [5, 5]}
    cluster_training = defaultdict(dict)
    testing_assignments = defaultdict(list)
    copy_validation = []
    for cluster_number, result in enumerate(selected_results, start=1):
        for side in (0, 1):
            pair = result["pair"]
            full_pool = [
                row for row in candidates
                if row["prediction"] == side
                and row["pair"] == pair
                and row["oriented_sign"] == result["sign"]
            ]
            quota = test_quotas[side][cluster_number - 1]
            best_design = None
            for training_rows_tuple in combinations(result["selected"][side], TARGET_GROUP_PER_SIDE):
                training_rows = list(training_rows_tuple)
                training_ids = {row["index"] for row in training_rows}
                ranked = []
                for row in full_pool:
                    if row["index"] in training_ids:
                        continue
                    profile_distances = np.asarray([
                        np.mean(np.abs(row["norm_original"] - train_row["norm_original"]))
                        for train_row in training_rows
                    ])
                    nearest = int(np.argmin(profile_distances))
                    copied_delta = training_rows[nearest]["delta"]
                    copied_profile = np.clip(row["norm_original"] + copied_delta, 0.0, 1.0)
                    raw_copied = ranges[:, 0] + copied_profile * (ranges[:, 1] - ranges[:, 0])
                    target = 1 - side
                    copied_frame = pd.DataFrame([raw_copied], columns=FEATURES)
                    copied_probability = float(model.predict_proba(copied_frame)[0, target])
                    copied_success = int(model.predict(copied_frame)[0]) == target
                    delta_distance = float(np.mean(np.abs(row["delta"] - copied_delta)))
                    rank_key = (
                        -int(copied_success),
                        delta_distance,
                        float(profile_distances[nearest]),
                        abs(copied_probability - 0.55),
                    )
                    ranked.append((rank_key, row, nearest, copied_probability, delta_distance))
                if len(ranked) < quota:
                    continue
                chosen = sorted(ranked, key=lambda item: item[0])[:quota]
                successes = sum(int(item[3] >= 0.5) for item in chosen)
                score = (
                    successes,
                    -float(np.mean([item[4] for item in chosen])),
                    -float(np.mean([item[0][2] for item in chosen])),
                    -float(np.mean([row["cluster_distance"] for row in training_rows])),
                )
                if best_design is None or score > best_design[0]:
                    best_design = (score, training_rows, chosen)
            if best_design is None:
                raise RuntimeError(f"No feasible copy-edit design for cluster {cluster_number}, prediction {side}")
            _, training_rows, chosen_rows = best_design
            cluster_training[cluster_number][side] = training_rows
            for _, row, nearest, copied_probability, delta_distance in chosen_rows:
                profile_distance = float(np.mean(np.abs(
                    row["norm_original"] - training_rows[nearest]["norm_original"]
                )))
                testing_assignments[(cluster_number, side)].append((profile_distance, nearest, row))
                copy_validation.append({
                    "cluster": cluster_number,
                    "prediction": side,
                    "source": f"{row['split']}:{row['row_id']}",
                    "copy_target_probability": copied_probability,
                    "copy_success": int(copied_probability >= 0.5),
                    "normalized_delta_distance": delta_distance,
                })

    output_rows = []
    selected_cluster_summaries = []
    for cluster_number, result in enumerate(selected_results, start=1):
        pair = result["pair"]
        training_by_side = cluster_training[cluster_number]
        testing_by_side = {
            side: sorted(testing_assignments[(cluster_number, side)], key=lambda item: item[0])
            for side in (0, 1)
        }

        def append_row(row, role, similarity=None):
            record = {
                "role": role,
                "cluster": cluster_number,
                "prediction": LABELS[row["prediction"]],
                "source": f"{row['split']}:{row['row_id']}",
                "source_split": row["split"],
                "source_row_id": row["row_id"],
                "source_instance_id": row["source_instance_id"],
                "training_cluster_pair": " + ".join(DISPLAY[index] for index in pair),
                "profile_lime_pair": " + ".join(DISPLAY[index] for index in row["pair"]),
                "original": {DISPLAY[i]: round(float(row["original"][i]), 2) for i in range(len(FEATURES))},
                "counterfactual": {DISPLAY[i]: round(float(row["counterfactual"][i]), 2) for i in range(len(FEATURES))},
                "counterfactual_changes": {
                    DISPLAY[i]: round(float(row["counterfactual"][i] - row["original"][i]), 2)
                    for i in row["pair"]
                },
                "boundary_margin": round(row["margin"], 4),
                "target_probability": round(row["target_probability"], 4),
                "lime_fidelity": round(row["lime_score"], 4),
                "lime_values_no_diabetes_direction": [
                    float(value) for value in row["lime_values"]
                ],
                "lime_feature_conditions": row["lime_feature_conditions"],
                "joint_cluster_distance": round(float(row["cluster_distance"]), 4)
                if role == "training" else None,
            }
            if similarity is not None:
                record.update({
                    "profile_distance_to_nearest_training": round(float(similarity[0]), 4),
                    "nearest_training_source": (
                        f"{training_by_side[row['prediction']][similarity[1]]['split']}:"
                        f"{training_by_side[row['prediction']][similarity[1]]['row_id']}"
                    ),
                })
            output_rows.append(record)

        for side in (0, 1):
            for row in training_by_side[side]:
                append_row(row, "training")
            for option in testing_by_side[side]:
                append_row(option[2], "testing", option)

        selected_cluster_summaries.append({
            "cluster": cluster_number,
            "pair": " + ".join(DISPLAY[i] for i in pair),
            "oriented_diabetes_to_no_diabetes_sign": list(result["sign"]),
            "mean_label_adjusted_lime": {
                DISPLAY[index]: round(float(np.mean([
                    row["lime_values"][index] for row in result["rows"]
                ])), 4) for index in pair
            },
            "mean_counterfactual_endpoint": {
                DISPLAY[index]: round(float(np.mean([
                    row["counterfactual"][index] for row in result["rows"]
                ])), 2) for index in pair
            },
            "common_endpoint_centre": {
                DISPLAY[index]: round(float(
                    ranges[index, 0]
                    + result["endpoint_centre"][offset] * (ranges[index, 1] - ranges[index, 0])
                ), 2)
                for offset, index in enumerate(pair)
            },
            "dbscan_eps": round(result["eps"], 3),
            "joint_compactness": round(result["compactness"], 4),
            "training_per_prediction": TARGET_GROUP_PER_SIDE,
            "testing_per_prediction": {
                "Diabetes": test_quotas[0][cluster_number - 1],
                "No Diabetes": test_quotas[1][cluster_number - 1],
            },
        })

    summary = {
        "method": {
            "lime_num_samples": NUM_SAMPLES,
            "minimum_normalized_change_per_selected_feature": MIN_NORMALIZED_CHANGE,
            "candidate_pairs_checked": [" + ".join(DISPLAY[i] for i in pair) for pair in candidate_pairs],
            "shortlist_per_prediction_per_pair": "all",
            "clustering": "DBSCAN within each exact LIME pair on standardized [two label-adjusted LIME values, two normalized counterfactual endpoint values].",
            "baseline_accuracy": baseline_scores,
            "regularized_candidates": [
                {"alpha": alpha, **scores} for alpha, _, scores in accuracy_rows
            ],
            "chosen_alpha": chosen_alpha,
            "chosen_accuracy": chosen_scores,
            "regularized_model": str(model_output),
        },
        "lime_pair_counts": [
            {
                "pair": " + ".join(DISPLAY[i] for i in pair),
                "diabetes": counts[(pair, 0)],
                "no_diabetes": counts[(pair, 1)],
            }
            for _, _, pair, _ in pair_totals
        ],
        "selected_clusters": selected_cluster_summaries,
        "qualifying_groups": len(pair_results),
        "qualifying_group_details": [
            {
                "pair": " + ".join(DISPLAY[i] for i in result["pair"]),
                "dbscan_eps": round(result["eps"], 3),
                "joint_compactness": round(result["compactness"], 4),
                "profiles_available": {
                    "diabetes": len(result["selected"][0]),
                    "no_diabetes": len(result["selected"][1]),
                },
            }
            for result in pair_results
        ],
        "selected_training_count": sum(row["role"] == "training" for row in output_rows),
        "selected_testing_count": sum(row["role"] == "testing" for row in output_rows),
        "copy_strategy_validation": {
            "testing_cases": len(copy_validation),
            "success_rate": float(np.mean([row["copy_success"] for row in copy_validation])),
            "mean_target_probability": float(np.mean([row["copy_target_probability"] for row in copy_validation])),
            "mean_normalized_delta_distance": float(np.mean([row["normalized_delta_distance"] for row in copy_validation])),
            "trials": copy_validation,
        },
        "selected_instances": output_rows,
    }

    output_directory = ROOT / "outputs/v15-discrete-lime-age"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_json = output_directory / "selected_clusters.json"
    output_csv = output_directory / "selected_instances.csv"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(output_rows).to_csv(output_csv, index=False)
    print(json.dumps({
        "selected_clusters": summary["selected_clusters"],
        "qualifying_groups": summary["qualifying_groups"],
        "training": summary["selected_training_count"],
        "testing": summary["selected_testing_count"],
        "json": str(output_json),
        "csv": str(output_csv),
    }, indent=2))


if __name__ == "__main__":
    main()
