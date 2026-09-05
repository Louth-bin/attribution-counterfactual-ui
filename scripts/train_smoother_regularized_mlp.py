"""Train and select a smaller, more regularized diabetes MLP.

Hyperparameters are selected without using the test split.  The selection score
combines development log loss, train/development accuracy gap, and the rate of
one-feature response curves that cross the 0.5 boundary more than once.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
ARCHITECTURES = [(4,), (8,), (12,), (8, 4)]
ALPHAS = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
SEEDS = [11, 29, 42]
MODEL_OUT = ROOT / "analysis" / "diabetes_mlp_smoother_regularized.joblib"
RESULTS_OUT = ROOT / "analysis" / "diabetes_mlp_smoother_regularized_search.csv"
SUMMARY_OUT = ROOT / "analysis" / "diabetes_mlp_smoother_regularized_summary.json"


def load_split(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(ROOT / f"src/data/diabetes/{name}.csv")
    return frame[FEATURES], frame["target"].to_numpy(int)


def make_model(hidden_layers: tuple[int, ...], alpha: float, seed: int) -> Pipeline:
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", StandardScaler(), FEATURES),
        ], remainder="drop")),
        ("model", MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=30,
            tol=1e-4,
            random_state=seed,
        )),
    ])


def repeated_crossing_rate(model: Pipeline, profiles: pd.DataFrame, ranges: np.ndarray) -> tuple[float, float]:
    crossing_counts = []
    originals = profiles[FEATURES].to_numpy(float)
    point_count = 201
    for index in range(len(FEATURES)):
        values = np.linspace(ranges[index, 0], ranges[index, 1], point_count)
        candidates = np.repeat(originals, point_count, axis=0)
        candidates[:, index] = np.tile(values, len(originals))
        probabilities = model.predict_proba(pd.DataFrame(candidates, columns=FEATURES))[:, 1]
        classes = (probabilities >= 0.5).reshape(len(originals), point_count)
        crossing_counts.extend(np.sum(classes[:, 1:] != classes[:, :-1], axis=1).astype(int).tolist())
    counts = np.asarray(crossing_counts)
    return float(np.mean(counts > 1)), float(np.mean(counts))


def metrics(model: Pipeline, x: pd.DataFrame, y: np.ndarray, prefix: str) -> dict[str, float]:
    probability = model.predict_proba(x)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y, prediction)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        f"{prefix}_log_loss": float(log_loss(y, probability, labels=[0, 1])),
        f"{prefix}_brier": float(brier_score_loss(y, probability)),
        f"{prefix}_auc": float(roc_auc_score(y, probability)),
    }


def main() -> None:
    x_train, y_train = load_split("train")
    x_dev, y_dev = load_split("dev")
    x_test, y_test = load_split("test")
    with (ROOT / "static" / "experiment-data.json").open(encoding="utf-8") as source:
        ranges = np.asarray(json.load(source)["datasets"]["diabetes"]["training_pool"][0]["raw_feature_ranges"], float)

    candidates: list[tuple[dict[str, object], Pipeline]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for hidden_layers in ARCHITECTURES:
            for alpha in ALPHAS:
                for seed in SEEDS:
                    model = make_model(hidden_layers, alpha, seed)
                    model.fit(x_train, y_train)
                    row: dict[str, object] = {
                        "hidden_layers": "x".join(map(str, hidden_layers)),
                        "alpha": alpha,
                        "seed": seed,
                        "iterations": int(model.named_steps["model"].n_iter_),
                    }
                    row.update(metrics(model, x_train, y_train, "train"))
                    row.update(metrics(model, x_dev, y_dev, "dev"))
                    repeated_rate, mean_crossings = repeated_crossing_rate(model, x_dev, ranges)
                    row["dev_repeated_crossing_rate"] = repeated_rate
                    row["dev_mean_boundary_crossings"] = mean_crossings
                    row["accuracy_gap"] = float(row["train_accuracy"] - row["dev_accuracy"])
                    row["selection_score"] = (
                        float(row["dev_log_loss"])
                        + 0.75 * repeated_rate
                        + 0.35 * max(0.0, float(row["accuracy_gap"]))
                    )
                    row["eligible"] = (
                        float(row["train_accuracy"]) <= 0.95
                        and float(row["dev_balanced_accuracy"]) >= 0.72
                    )
                    candidates.append((row, model))

    eligible = [(row, model) for row, model in candidates if row["eligible"]]
    if not eligible:
        raise RuntimeError("No candidate met the generalization eligibility thresholds")
    selected_row, selected_model = min(eligible, key=lambda item: (item[0]["selection_score"], -item[0]["dev_auc"]))
    selected_row.update(metrics(selected_model, x_test, y_test, "test"))
    test_repeated_rate, test_mean_crossings = repeated_crossing_rate(selected_model, x_test, ranges)
    selected_row["test_repeated_crossing_rate"] = test_repeated_rate
    selected_row["test_mean_boundary_crossings"] = test_mean_crossings

    with RESULTS_OUT.open("w", encoding="utf-8-sig", newline="") as destination:
        rows = [row for row, _ in sorted(candidates, key=lambda item: item[0]["selection_score"])]
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    joblib.dump(selected_model, MODEL_OUT)
    summary = {
        "model_path": str(MODEL_OUT),
        "selection_uses_test_data": False,
        "selection_rule": "minimum dev log loss + 0.75 repeated-crossing rate + 0.35 positive train/dev accuracy gap; train accuracy <= .95 and dev balanced accuracy >= .72",
        "selected": selected_row,
        "search_candidates": len(candidates),
        "eligible_candidates": len(eligible),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
