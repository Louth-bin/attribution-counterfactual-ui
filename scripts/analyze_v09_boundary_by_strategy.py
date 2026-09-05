"""Summarize diabetes boundary distance by participant strategy in v0.9."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "qualtrics" / "qualtrics_results_v0.9.for-plausibility.csv"
BOUNDARY_PATH = Path(tempfile.gettempdir()) / "boundary_distance_v09_values.csv"
FITS_PATH = ROOT / "qualtrics" / "v0.9_participant_counterfactual_strategy_fits.csv"
OUTPUT_PATH = Path(tempfile.gettempdir()) / "v09_boundary_by_strategy.json"


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    result = frame.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(8)
    return result.replace({np.nan: None}).to_dict(orient="records")


def main() -> None:
    base = pd.read_csv(BASE_PATH)
    base.insert(0, "row_number", np.arange(1, len(base) + 1))
    boundary = pd.read_csv(BOUNDARY_PATH)
    fits = pd.read_csv(FITS_PATH)
    data = base.merge(
        boundary,
        on=["row_number", "participant", "domain", "phase"],
        validate="one_to_one",
    ).merge(
        fits,
        on=["participant", "domain", "xai"],
        how="left",
        validate="many_to_one",
    )
    data = data.loc[
        (data["phase"] == "testing") & (data["domain"] == "diabetes")
    ].copy()
    data["boundary_new"] = pd.to_numeric(
        data["boundary distance new"], errors="coerce"
    )
    data["valid"] = pd.to_numeric(
        data["valid counterfactual (0/1)"], errors="coerce"
    )
    data["proximity"] = pd.to_numeric(data["proximity"], errors="coerce")
    data["sparsity"] = pd.to_numeric(data["sparsity"], errors="coerce")

    participant = (
        data.groupby(["xai", "participant"], observed=True)
        .agg(
            cases=("boundary_new", "size"),
            boundary_mean=("boundary_new", "mean"),
            boundary_median=("boundary_new", "median"),
            valid_rate=("valid", "mean"),
            valid_boundary_mean=(
                "boundary_new",
                lambda values: float(
                    data.loc[values.index]
                    .loc[data.loc[values.index, "valid"] == 1, "boundary_new"]
                    .mean()
                ),
            ),
            proximity_mean=("proximity", "mean"),
            attributes_changed_mean=("sparsity", "mean"),
            qualitative_strategy=("reasoning strategy", "first"),
            fitted_strategy=("best_counterfactual_strategy", "first"),
            fitted_model=("counterfactual_implied_mental_model", "first"),
            fit_conclusion=("counterfactual_strategy_fit_conclusion", "first"),
            supported_strategies=("supported_counterfactual_strategies_one_se", "first"),
            cv_loss=("best_strategy_cv_mean_normalized_l1", "first"),
            fitted_k=("best_strategy_full_data_k", "first"),
            fixed_amount=("best_strategy_full_data_fixed_amount", "first"),
        )
        .reset_index()
    )

    condition = (
        participant.groupby("xai", observed=True)
        .agg(
            participants=("participant", "size"),
            participant_boundary_mean=("boundary_mean", "mean"),
            participant_boundary_median=("boundary_mean", "median"),
            participant_valid_rate=("valid_rate", "mean"),
        )
        .reset_index()
    )
    none_mean = float(
        participant.loc[participant["xai"] == "none", "boundary_mean"].mean()
    )
    attribution_mean = float(
        participant.loc[
            participant["xai"] == "attribution", "boundary_mean"
        ].mean()
    )
    cf = participant.loc[participant["xai"] == "counterfactual"].copy()
    cf_mean = float(cf["boundary_mean"].mean())

    def summarize_strategy(column: str) -> pd.DataFrame:
        result = (
            cf.groupby(column, dropna=False, observed=True)
            .agg(
                participants=("participant", "size"),
                boundary_mean=("boundary_mean", "mean"),
                boundary_median=("boundary_mean", "median"),
                valid_boundary_mean=("valid_boundary_mean", "mean"),
                validity_rate=("valid_rate", "mean"),
                proximity_mean=("proximity_mean", "mean"),
                attributes_changed_mean=("attributes_changed_mean", "mean"),
                cv_loss_mean=("cv_loss", "mean"),
            )
            .reset_index()
        )
        result["difference_from_none"] = result["boundary_mean"] - none_mean
        result["difference_from_attribution"] = (
            result["boundary_mean"] - attribution_mean
        )
        result["weighted_contribution_to_cf_minus_none"] = (
            result["participants"]
            / len(cf)
            * result["difference_from_none"]
        )
        remaining_means = []
        for strategy in result[column]:
            retained = cf.loc[cf[column] != strategy, "boundary_mean"]
            remaining_means.append(float(retained.mean()) if len(retained) else np.nan)
        result["cf_mean_if_group_removed"] = remaining_means
        return result.sort_values("boundary_mean")

    report = {
        "condition_summary": records(condition),
        "counterfactual_condition_mean": cf_mean,
        "none_condition_mean": none_mean,
        "attribution_condition_mean": attribution_mean,
        "by_fitted_strategy": records(summarize_strategy("fitted_strategy")),
        "by_fitted_mental_model": records(summarize_strategy("fitted_model")),
        "by_qualitative_strategy": records(
            summarize_strategy("qualitative_strategy")
        ),
        "qualitative_strategy_by_condition": records(
            participant.groupby(
                ["xai", "qualitative_strategy"],
                dropna=False,
                observed=True,
            )
            .agg(
                participants=("participant", "size"),
                boundary_mean=("boundary_mean", "mean"),
                valid_boundary_mean=("valid_boundary_mean", "mean"),
                validity_rate=("valid_rate", "mean"),
                proximity_mean=("proximity_mean", "mean"),
                attributes_changed_mean=("attributes_changed_mean", "mean"),
            )
            .reset_index()
            .sort_values(["qualitative_strategy", "xai"])
        ),
        "counterfactual_participants": records(
            cf.sort_values("boundary_mean")[
                [
                    "participant",
                    "boundary_mean",
                    "valid_boundary_mean",
                    "valid_rate",
                    "proximity_mean",
                    "attributes_changed_mean",
                    "qualitative_strategy",
                    "fitted_strategy",
                    "fitted_model",
                    "fit_conclusion",
                    "supported_strategies",
                    "cv_loss",
                    "fitted_k",
                    "fixed_amount",
                ]
            ]
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
