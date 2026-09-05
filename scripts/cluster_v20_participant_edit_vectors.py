"""Cluster v2.0 participants by their 20 x 5 signed edit vectors."""

from __future__ import annotations

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
OUTPUT_DIR = ROOT / "outputs" / "v20-participant-edit-vector-clusters"
FEATURES = [f"x_{i}_change" for i in range(1, 6)]


def bootstrap_stability(x: np.ndarray, k: int, repeats: int = 8) -> float:
    labels = []
    rng = np.random.default_rng(20260901 + k)
    n_features = x.shape[1]
    for _ in range(repeats):
        cols = rng.choice(n_features, size=n_features, replace=True)
        labels.append(
            KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1_000_000)))
            .fit_predict(x[:, cols])
        )
    aris = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            aris.append(adjusted_rand_score(labels[i], labels[j]))
    return float(np.mean(aris))


def main() -> None:
    data = pd.read_csv(INPUT)
    testing = data.loc[data["phase"].eq("testing")].copy()
    testing["instance id"] = testing["instance id"].astype(int)
    for column in FEATURES:
        testing[column] = pd.to_numeric(testing[column], errors="coerce").fillna(0.0)

    expected_instances = sorted(testing["instance id"].unique())
    rows = []
    matrix = []
    for participant, group in testing.groupby("participant"):
        group = group.sort_values("instance id")
        if group["instance id"].tolist() != expected_instances:
            raise RuntimeError(f"{participant} does not have all test instances in order")
        vector = group[FEATURES].to_numpy(float).reshape(-1)
        matrix.append(vector)
        rows.append(
            {
                "participant": participant,
                "xai": group["xai"].iloc[0],
                "mean_boundary_distance": group["boundary distance new"].astype(float).mean(),
                "median_boundary_distance": group["boundary distance new"].astype(float).median(),
                "success_rate": group["successful counterfactual (0/1)"].astype(float).mean(),
                "mean_proximity": group["proximity"].astype(float).mean(),
                "direction_rate": group["move towards target (0/1)"].astype(float).mean(),
                **{
                    f"feature_change_prevalence_{i}": group[f"x_{i}_changed"].astype(float).mean()
                    for i in range(1, 6)
                },
                **{
                    f"mean_signed_change_x{i}": group[f"x_{i}_change"].astype(float).mean()
                    for i in range(1, 6)
                },
            }
        )
    participants = pd.DataFrame(rows)
    x = np.vstack(matrix)
    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=5, random_state=0).fit(x_scaled)
    pcs = pca.transform(x_scaled)
    participants["PC1"] = pcs[:, 0]
    participants["PC2"] = pcs[:, 1]
    participants["PC3"] = pcs[:, 2]

    diagnostics = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=30, random_state=42)
        labels = km.fit_predict(x_scaled)
        diagnostics.append(
            {
                "k": k,
                "silhouette": silhouette_score(x_scaled, labels),
                "bootstrap_ARI_mean": bootstrap_stability(x_scaled, k),
                "cluster_sizes": dict(pd.Series(labels).value_counts().sort_index()),
            }
        )
        participants[f"kmeans_k{k}"] = labels + 1

    # Also include a hierarchical 2-cluster label as a sanity check.
    participants["agglomerative_k2"] = (
        AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(x_scaled) + 1
    )

    best_k = max(diagnostics, key=lambda d: d["silhouette"])["k"]
    cluster_col = f"kmeans_k{best_k}"

    cluster_summaries = []
    for cluster, group in participants.groupby(cluster_col):
        record = {
            "best_k": best_k,
            "cluster": int(cluster),
            "n": len(group),
            "xai_counts": dict(group["xai"].value_counts()),
            "mean_boundary_distance": group["mean_boundary_distance"].mean(),
            "success_rate": group["success_rate"].mean(),
            "mean_proximity": group["mean_proximity"].mean(),
            "direction_rate": group["direction_rate"].mean(),
        }
        for i in range(1, 6):
            record[f"feature_change_prevalence_{i}"] = group[
                f"feature_change_prevalence_{i}"
            ].mean()
            record[f"mean_signed_change_x{i}"] = group[f"mean_signed_change_x{i}"].mean()
        cluster_summaries.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    participants.to_csv(OUTPUT_DIR / "participant_edit_vector_clusters.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT_DIR / "cluster_diagnostics.csv", index=False)
    pd.DataFrame(cluster_summaries).to_csv(OUTPUT_DIR / "best_cluster_summary.csv", index=False)

    print("participants", len(participants), "features", x.shape[1])
    print("PCA explained variance first 5:", [round(v, 3) for v in pca.explained_variance_ratio_])
    print("diagnostics:")
    for d in diagnostics:
        print(
            f"k={d['k']} silhouette={d['silhouette']:.3f} "
            f"bootstrap_ARI={d['bootstrap_ARI_mean']:.3f} sizes={d['cluster_sizes']}"
        )
    print("best_k_by_silhouette", best_k)
    print(pd.DataFrame(cluster_summaries).round(3).to_string(index=False))
    print(OUTPUT_DIR / "participant_edit_vector_clusters.csv")


if __name__ == "__main__":
    main()
