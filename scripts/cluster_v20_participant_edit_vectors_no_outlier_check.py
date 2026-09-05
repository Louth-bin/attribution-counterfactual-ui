import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


QUALTRICS = Path("qualtrics/qualtrics_results_v2.0.csv")
OUTLIER = "R_7L78tZJ3cQDgpFH"
FEATURES = [f"x_{i}" for i in range(1, 6)]
CHANGE_COLS = [f"{f}_change" for f in FEATURES]


def bootstrap_ari(X, labels, k, repeats=20, seed=123):
    rng = np.random.default_rng(seed)
    aris = []
    n, p = X.shape
    for _ in range(repeats):
        cols = rng.choice(p, p, replace=True)
        labels_b = KMeans(n_clusters=k, random_state=int(rng.integers(1_000_000)), n_init=20).fit_predict(X[:, cols])
        aris.append(adjusted_rand_score(labels, labels_b))
    return float(np.mean(aris))


def build_vectors(df):
    test = df[df["phase"].astype(str).str.lower().eq("testing")].copy()
    test = test.sort_values(["participant", "instance id"])

    rows = []
    vectors = []
    for pid, g in test.groupby("participant", sort=True):
        if len(g) != 20:
            continue
        g = g.sort_values("instance id")
        vector = g[CHANGE_COLS].to_numpy(dtype=float).reshape(-1)
        if np.isfinite(vector).all():
            rows.append(
                {
                    "participant": pid,
                    "xai": g["xai"].iloc[0],
                    "mean_boundary_distance": g["boundary distance new"].mean(),
                    "median_boundary_distance": g["boundary distance new"].median(),
                    "success_rate": g["successful counterfactual (0/1)"].astype(float).mean(),
                    "mean_proximity": g["proximity"].mean(),
                    "direction_rate": g["move towards target (0/1)"].astype(float).mean(),
                }
            )
            vectors.append(vector)
    return pd.DataFrame(rows), np.vstack(vectors)


def run(label, rows, X_raw):
    print(f"\n{label}")
    print(f"participants={len(rows)} dimensions={X_raw.shape[1]}")

    X = StandardScaler().fit_transform(X_raw)
    pca = PCA(n_components=min(8, X.shape[0], X.shape[1]), random_state=1).fit_transform(X)
    evr = PCA(n_components=min(8, X.shape[0], X.shape[1]), random_state=1).fit(X).explained_variance_ratio_
    print("pca_explained_first5=" + ",".join(f"{v:.3f}" for v in evr[:5]))

    diag = []
    for k in range(2, min(7, len(rows))):
        km = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        ari = bootstrap_ari(X, labels, k)
        sizes = pd.Series(labels + 1).value_counts().sort_index().to_dict()
        diag.append((k, sil, ari, sizes, labels))
        print(f"k={k} silhouette={sil:.3f} bootstrap_ARI={ari:.3f} sizes={sizes}")

    best = max(diag, key=lambda z: z[1])
    rows2 = rows.copy()
    rows2["best_kmeans_cluster"] = best[4] + 1
    rows2["agglo_k2"] = AgglomerativeClustering(n_clusters=2).fit_predict(X) + 1
    rows2["PC1"] = pca[:, 0]
    rows2["PC2"] = pca[:, 1]

    print(f"best_k_by_silhouette={best[0]}")
    print("best_k_xai_crosstab:")
    print(pd.crosstab(rows2["best_kmeans_cluster"], rows2["xai"]).to_string())
    print("best_k_summary:")
    print(
        rows2.groupby("best_kmeans_cluster")
        .agg(
            n=("participant", "size"),
            mean_boundary_distance=("mean_boundary_distance", "mean"),
            success_rate=("success_rate", "mean"),
            mean_proximity=("mean_proximity", "mean"),
            direction_rate=("direction_rate", "mean"),
        )
        .round(3)
        .to_string()
    )


def main():
    df = pd.read_csv(QUALTRICS)
    rows, X = build_vectors(df)
    run("ALL PARTICIPANTS", rows, X)

    keep = rows["participant"] != OUTLIER
    run("EXCLUDING SINGLE EXTREME OUTLIER", rows.loc[keep].reset_index(drop=True), X[keep.to_numpy()])


if __name__ == "__main__":
    main()
