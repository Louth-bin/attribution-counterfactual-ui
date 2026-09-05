"""Compare top-two SHAP truncation with two-feature Kernel SHAP regularization.

This analysis uses only the 90 classifier profiles in the static experiment
bundle. It does not read participant data and does not modify the classifier or
the experiment bundle.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ATTRIBUTION_RANKING_DECIMALS, ExplanationPipeline  # noqa: E402
from src.xai_methods.shap_method import (  # noqa: E402
    _decode_rows,
    _encode_frames,
    _kernel_shap_sample_count,
)


BUNDLE_PATH = ROOT / "static" / "experiment-data.json"
OUTPUT_DIR = ROOT / "analysis"
INSTANCE_OUTPUT = OUTPUT_DIR / "attribution_top2_regularization_instances.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "attribution_top2_regularization_summary.csv"
FEATURE_OUTPUT = OUTPUT_DIR / "attribution_top2_regularization_feature_counts.csv"
TRANSITION_OUTPUT = OUTPUT_DIR / "attribution_top2_regularization_pair_transitions.csv"
REPORT_OUTPUT = OUTPUT_DIR / "attribution_top2_regularization_report.md"

DOMAINS = ("housing", "safelimit", "diabetes")
NONZERO_TOLERANCE = 1e-12


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def positive_class_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[min(1, len(shap_values) - 1)], dtype=float)
    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3:
        return values[:, :, min(1, values.shape[2] - 1)]
    if values.ndim == 2:
        return values
    raise ValueError(f"Unexpected SHAP output shape {values.shape}")


def top_two_indices(values: np.ndarray) -> tuple[int, ...]:
    ranked = sorted(
        range(len(values)),
        key=lambda index: (
            round(abs(float(values[index])), ATTRIBUTION_RANKING_DECIMALS),
            -index,
        ),
        reverse=True,
    )
    return tuple(index for index in ranked if abs(float(values[index])) > 0)[:2]


def pair_label(indices: Iterable[int], names: list[str]) -> str:
    selected = set(indices)
    return " | ".join(name for index, name in enumerate(names) if index in selected)


def explain_batch(
    estimator: Any,
    background: pd.DataFrame,
    instances: pd.DataFrame,
    l1_reg: str | float,
) -> np.ndarray:
    encoded_background, encoded_instances, category_maps = _encode_frames(
        background, instances
    )

    def wrapped_predict(encoded_rows: np.ndarray) -> np.ndarray:
        decoded = _decode_rows(
            encoded_rows=encoded_rows,
            reference_columns=list(instances.columns),
            reference_background=background,
            category_maps=category_maps,
        )
        return estimator.predict_proba(decoded)

    explainer = shap.KernelExplainer(
        wrapped_predict,
        encoded_background.to_numpy(dtype=float),
        feature_names=list(instances.columns),
    )
    random_state = np.random.get_state()
    np.random.seed(42)
    try:
        values = explainer.shap_values(
            encoded_instances.to_numpy(dtype=float),
            nsamples=_kernel_shap_sample_count(encoded_background.shape[1]),
            l1_reg=l1_reg,
            silent=True,
        )
    finally:
        np.random.set_state(random_state)
    return positive_class_values(values)


def group_summary(
    instance_rows: list[dict[str, Any]], domain: str, split: str
) -> dict[str, Any]:
    rows = [
        row
        for row in instance_rows
        if (domain == "all" or row["domain"] == domain)
        and (split == "all" or row["split"] == split)
    ]
    total = len(rows)
    exact = sum(row["exact_pair_match"] for row in rows)
    overlap_counts = Counter(row["overlap_count"] for row in rows)
    return {
        "domain": domain,
        "split": split,
        "instances": total,
        "exact_pair_matches": exact,
        "exact_pair_match_rate": exact / total,
        "pair_changed_count": total - exact,
        "pair_changed_rate": (total - exact) / total,
        "both_features_retained_count": overlap_counts[2],
        "one_feature_retained_count": overlap_counts[1],
        "no_features_retained_count": overlap_counts[0],
        "selected_feature_slot_retention_rate": sum(
            row["overlap_count"] for row in rows
        )
        / (2 * total),
        "regularized_exactly_two_nonzero_count": sum(
            row["regularized_nonzero_count"] == 2 for row in rows
        ),
        "unregularized_pair_reproduction_count": sum(
            row["unregularized_pair_reproduced"] for row in rows
        ),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def make_report(
    instance_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
) -> str:
    overall = next(
        row for row in summary_rows if row["domain"] == "all" and row["split"] == "all"
    )
    domain_rows = [
        row for row in summary_rows if row["domain"] != "all" and row["split"] == "all"
    ]
    summary_table = [
        [
            row["domain"],
            row["instances"],
            row["exact_pair_matches"],
            f"{row['exact_pair_match_rate']:.1%}",
            row["one_feature_retained_count"],
            row["no_features_retained_count"],
            f"{row['selected_feature_slot_retention_rate']:.1%}",
        ]
        for row in domain_rows
    ]
    feature_table = [
        [
            row["domain"],
            row["feature"],
            row["truncated_top2_selected_count"],
            row["regularized_selected_count"],
            row["regularized_minus_truncated"],
        ]
        for row in feature_rows
    ]
    changed_transitions = [
        row
        for row in transition_rows
        if row["truncated_pair"] != row["regularized_pair"]
    ]
    transition_table = [
        [
            row["domain"],
            row["truncated_pair"],
            row["regularized_pair"],
            row["instances"],
        ]
        for row in sorted(
            changed_transitions,
            key=lambda row: (-row["instances"], row["domain"], row["truncated_pair"]),
        )[:20]
    ]
    return "\n".join(
        [
            "# Two-feature Kernel SHAP regularization comparison",
            "",
            "## Question",
            "",
            (
                "How much does feature selection change if the current procedure—"
                "compute all five Kernel SHAP values with `l1_reg=0`, then display "
                "the two largest absolute values—is replaced by Kernel SHAP's "
                "intrinsic `l1_reg=\"num_features(2)\"` selection?"
            ),
            "",
            "No participant data is used. The comparison covers the 90 static experiment profiles: 10 training and 20 testing profiles in each of three domains.",
            "",
            "The classifier, 50-row training background, exact 32-coalition Kernel SHAP evaluation, positive-class output, and random seed are held fixed. Only `l1_reg` changes. This regularizes the local SHAP surrogate; it does not retrain or regularize the neural classifier.",
            "",
            "## Result",
            "",
            (
                f"Across all {overall['instances']} profiles, the exact selected "
                f"feature pair agrees on {overall['exact_pair_matches']} "
                f"({overall['exact_pair_match_rate']:.1%}) and changes on "
                f"{overall['pair_changed_count']} ({overall['pair_changed_rate']:.1%}). "
                f"Across the 180 selected feature slots, "
                f"{overall['selected_feature_slot_retention_rate']:.1%} are retained."
            ),
            "",
            markdown_table(
                [
                    "Domain",
                    "N",
                    "Exact pairs",
                    "Exact rate",
                    "One retained",
                    "None retained",
                    "Feature-slot retention",
                ],
                summary_table,
            ),
            "",
            (
                f"The regularized solution has exactly two nonzero attributions "
                f"for {overall['regularized_exactly_two_nonzero_count']} of "
                f"{overall['instances']} profiles. Recomputed unregularized top-two "
                f"pairs reproduce the stored experiment pair for "
                f"{overall['unregularized_pair_reproduction_count']} of "
                f"{overall['instances']} profiles."
            ),
            "",
            "## Feature selection counts",
            "",
            markdown_table(
                ["Domain", "Feature", "Truncated top-2", "Regularized", "Difference"],
                feature_table,
            ),
            "",
            "## Most frequent changed pair transitions",
            "",
            markdown_table(
                ["Domain", "Truncated pair", "Regularized pair", "Instances"],
                transition_table,
            ),
            "",
            "## Interpretation",
            "",
            (
                "Top-two truncation ranks coefficients from the full five-feature "
                "local explanation. `num_features(2)` instead chooses two features "
                "while fitting the sparse local surrogate, then refits their values. "
                "Consequently, it can select a different pair when features are "
                "correlated or provide substitutable local effects."
            ),
            "",
        ]
    )


def main() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))["datasets"]
    pipeline = ExplanationPipeline()
    instance_rows: list[dict[str, Any]] = []

    for domain in DOMAINS:
        assets = pipeline.prepare_assets(domain, "mlp")
        dataset = assets.dataset
        names = list(dataset.feature_names)
        cases = [
            ("training", case) for case in bundle[domain]["training_pool"]
        ] + [("testing", case) for case in bundle[domain]["test_pool"]]
        instances = pd.DataFrame(
            [case["raw_feature_values"] for _, case in cases], columns=names
        ).astype(dataset.train_df[names].dtypes.to_dict())
        background = dataset.train_df[names].sample(
            n=min(len(dataset.train_df), 50), random_state=42
        )

        unregularized = explain_batch(
            assets.model_artifact.estimator, background, instances, l1_reg=0.0
        )
        regularized = explain_batch(
            assets.model_artifact.estimator,
            background,
            instances,
            l1_reg="num_features(2)",
        )

        for (split, case), unregularized_values, regularized_values in zip(
            cases, unregularized, regularized
        ):
            stored_indices = tuple(case["attribution"]["shown_feature_indices"][:2])
            recomputed_indices = top_two_indices(unregularized_values)
            regularized_indices = tuple(
                int(index)
                for index in np.flatnonzero(
                    np.abs(regularized_values) > NONZERO_TOLERANCE
                ).tolist()
            )
            overlap = set(stored_indices).intersection(regularized_indices)
            instance_rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "instance_id": case["instance_id"],
                    "prediction_label": case["prediction"]["label"],
                    "truncated_top2_pair": pair_label(stored_indices, names),
                    "regularized_pair": pair_label(regularized_indices, names),
                    "exact_pair_match": int(set(stored_indices) == set(regularized_indices)),
                    "overlap_count": len(overlap),
                    "retained_features": " | ".join(
                        name for index, name in enumerate(names) if index in overlap
                    ),
                    "dropped_features": " | ".join(
                        name
                        for index, name in enumerate(names)
                        if index in set(stored_indices).difference(regularized_indices)
                    ),
                    "added_features": " | ".join(
                        name
                        for index, name in enumerate(names)
                        if index in set(regularized_indices).difference(stored_indices)
                    ),
                    "regularized_nonzero_count": len(regularized_indices),
                    "unregularized_pair_reproduced": int(
                        set(stored_indices) == set(recomputed_indices)
                    ),
                    "stored_raw_shap_values": json.dumps(
                        case["attribution"]["raw_values"], separators=(",", ":")
                    ),
                    "recomputed_unregularized_values": json.dumps(
                        unregularized_values.tolist(), separators=(",", ":")
                    ),
                    "regularized_values": json.dumps(
                        regularized_values.tolist(), separators=(",", ":")
                    ),
                }
            )

    summary_rows = [group_summary(instance_rows, "all", "all")]
    summary_rows.extend(group_summary(instance_rows, domain, "all") for domain in DOMAINS)
    summary_rows.extend(
        group_summary(instance_rows, domain, split)
        for domain in DOMAINS
        for split in ("training", "testing")
    )

    feature_rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        domain_rows = [row for row in instance_rows if row["domain"] == domain]
        names = list(bundle[domain]["training_pool"][0]["raw_feature_names"])
        for feature in names:
            truncated_count = sum(
                feature in row["truncated_top2_pair"].split(" | ") for row in domain_rows
            )
            regularized_count = sum(
                feature in row["regularized_pair"].split(" | ") for row in domain_rows
            )
            feature_rows.append(
                {
                    "domain": domain,
                    "feature": feature,
                    "instances": len(domain_rows),
                    "truncated_top2_selected_count": truncated_count,
                    "regularized_selected_count": regularized_count,
                    "regularized_minus_truncated": regularized_count - truncated_count,
                }
            )

    transition_counts: Counter[tuple[str, str, str]] = Counter(
        (
            row["domain"],
            row["truncated_top2_pair"],
            row["regularized_pair"],
        )
        for row in instance_rows
    )
    transition_rows = [
        {
            "domain": domain,
            "truncated_pair": truncated_pair,
            "regularized_pair": regularized_pair,
            "instances": count,
        }
        for (domain, truncated_pair, regularized_pair), count in sorted(
            transition_counts.items()
        )
    ]

    write_csv(INSTANCE_OUTPUT, instance_rows, instance_rows[0].keys())
    write_csv(SUMMARY_OUTPUT, summary_rows, summary_rows[0].keys())
    write_csv(FEATURE_OUTPUT, feature_rows, feature_rows[0].keys())
    write_csv(TRANSITION_OUTPUT, transition_rows, transition_rows[0].keys())
    REPORT_OUTPUT.write_text(
        make_report(instance_rows, summary_rows, feature_rows, transition_rows),
        encoding="utf-8",
    )
    print(f"instances={len(instance_rows)}")
    print(f"summary={SUMMARY_OUTPUT}")
    print(f"report={REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
