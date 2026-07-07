from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .runtime_config import get_dataset_runtime_config


DIABETES_SOURCE_VERSION = "kaggle_mathchi_pima_flipped_labels_v1"
DIABETES_SOURCE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
CERAMIC_SOURCE_VERSION = "synthetic_ceramic_tile_firing_deformation_n2000_v1"
SAFELIMIT_SOURCE_VERSION = "widmark_synthetic_safelimit_n2000_drunk_zero_v2"


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetBundle:
    dataset_name: str
    dataset_dir: Path
    full_df: pd.DataFrame
    train_df: pd.DataFrame
    dev_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_names: list[str]
    feature_display_names: list[str]
    feature_types: list[str]
    feature_ranges: list[list[Any]]
    display_feature_ranges: list[list[Any]]
    friendly_category_names: dict[str, dict[str, str]]
    category_orders: dict[str, tuple[str, ...]]
    target_column: str
    class_labels: list[str]

    @property
    def available_instance_count(self) -> int:
        return len(self.test_df)


def _dataset_dir(dataset_name: str) -> Path:
    dataset_dir = DATA_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def _base_csv_path(dataset_name: str) -> Path:
    return _dataset_dir(dataset_name) / f"{dataset_name}.csv"


def _metadata_path(dataset_name: str) -> Path:
    return _dataset_dir(dataset_name) / "metadata.json"


def _source_csv_path(dataset_name: str, filename: str = "source.csv") -> Path:
    return _dataset_dir(dataset_name) / filename


def _split_paths(dataset_name: str) -> dict[str, Path]:
    dataset_dir = _dataset_dir(dataset_name)
    return {
        "train": dataset_dir / "train.csv",
        "dev": dataset_dir / "dev.csv",
        "test": dataset_dir / "test.csv",
    }


def _write_csv_atomically(dataframe: pd.DataFrame, destination: Path) -> None:
    temporary_path = destination.with_suffix(f"{destination.suffix}.tmp")
    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(destination)


def _write_json_atomically(payload: dict[str, Any], destination: Path) -> None:
    temporary_path = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(destination)


def _load_csv_from_local_or_remote(
    dataset_name: str,
    source_url: str,
    filename: str = "source.csv",
    read_csv_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    source_csv_path = _source_csv_path(dataset_name, filename)
    read_csv_kwargs = read_csv_kwargs or {}
    if source_csv_path.exists():
        return pd.read_csv(source_csv_path, **read_csv_kwargs)

    try:
        dataframe = pd.read_csv(source_url, **read_csv_kwargs)
    except Exception as error:
        raise ValueError(
            "The source file is not available locally and could not be downloaded. "
            f"Expected a CSV at {source_csv_path}."
        ) from error

    _write_csv_atomically(dataframe, source_csv_path)
    return dataframe


def _generate_diabetes_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_df = _load_csv_from_local_or_remote(
        dataset_name="diabetes",
        source_url=DIABETES_SOURCE_URL,
        filename="kaggle_source.csv",
        read_csv_kwargs={
            "header": None,
            "names": [
                "glucose",
                "blood_pressure",
                "skin_thickness",
                "insulin",
                "bmi",
                "age",
                "target",
            ],
            "usecols": [1, 2, 3, 4, 5, 7, 8],
        },
    ).copy()
    raw_df.insert(0, "row_id", range(len(raw_df)))
    raw_df["target"] = raw_df["target"].map({0: 1, 1: 0})

    metadata = {
        "target_column": "target",
        "source_format": DIABETES_SOURCE_VERSION,
        "class_labels": [
            "Diabetes",
            "No Diabetes",
        ],
        "feature_types": {
            "glucose": "numerical",
            "blood_pressure": "numerical",
            "skin_thickness": "numerical",
            "insulin": "numerical",
            "bmi": "numerical",
            "age": "numerical",
        },
    }
    return raw_df, metadata


def _generate_ceramic_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(219)
    row_count = 2000

    body_moisture_pct = np.clip(rng.normal(5.8, 0.85, row_count), 3.0, 8.5)
    pressing_pressure_mpa = np.clip(rng.normal(31.0, 4.8, row_count), 18.0, 44.0)
    green_density_g_cm3 = np.clip(
        1.84
        + (0.006 * (pressing_pressure_mpa - 31.0))
        - (0.011 * (body_moisture_pct - 5.8))
        + rng.normal(0.0, 0.022, row_count),
        1.65,
        2.02,
    )
    particle_d50_um = np.clip(rng.normal(13.5, 2.8, row_count), 6.0, 23.0)
    feldspar_pct = np.clip(rng.normal(22.0, 3.2, row_count), 12.0, 32.0)
    peak_temperature_c = np.clip(rng.normal(1185.0, 24.0, row_count), 1120.0, 1250.0)
    heating_rate_c_min = np.clip(rng.normal(7.3, 1.7, row_count), 2.5, 12.5)
    soak_time_min = np.clip(rng.normal(40.0, 10.0, row_count), 15.0, 70.0)
    cooling_rate_c_min = np.clip(rng.normal(6.2, 1.5, row_count), 2.0, 11.0)

    deformation_score = (
        0.048 * (peak_temperature_c - 1185.0)
        + 0.050 * (soak_time_min - 40.0)
        + 0.180 * (feldspar_pct - 22.0)
        + 0.310 * (heating_rate_c_min - 7.3)
        + 0.240 * (cooling_rate_c_min - 6.2)
        - 12.0 * (green_density_g_cm3 - 1.84)
        + 0.090 * (particle_d50_um - 13.5)
        + 0.150 * (body_moisture_pct - 5.8)
        + rng.normal(0.0, 0.85, row_count)
    )
    target = (deformation_score >= np.median(deformation_score)).astype(int)

    raw_df = pd.DataFrame(
        {
            "row_id": range(row_count),
            "peak_temperature_c": np.round(peak_temperature_c, 1),
            "heating_rate_c_min": np.round(heating_rate_c_min, 2),
            "soak_time_min": np.round(soak_time_min, 1),
            "cooling_rate_c_min": np.round(cooling_rate_c_min, 2),
            "body_moisture_pct": np.round(body_moisture_pct, 2),
            "pressing_pressure_mpa": np.round(pressing_pressure_mpa, 1),
            "green_density_g_cm3": np.round(green_density_g_cm3, 3),
            "particle_d50_um": np.round(particle_d50_um, 2),
            "feldspar_pct": np.round(feldspar_pct, 2),
            "target": target,
        }
    )

    metadata = {
        "target_column": "target",
        "source_format": CERAMIC_SOURCE_VERSION,
        "source": "Synthetic technical ceramic tile firing batches generated from process and green-body measurements; outcome indicates post-firing dimensional deformation.",
        "class_labels": [
            "Dimensionally Stable",
            "Warped After Firing",
        ],
        "feature_types": {
            "peak_temperature_c": "numerical",
            "heating_rate_c_min": "numerical",
            "soak_time_min": "numerical",
            "cooling_rate_c_min": "numerical",
            "body_moisture_pct": "numerical",
            "pressing_pressure_mpa": "numerical",
            "green_density_g_cm3": "numerical",
            "particle_d50_um": "numerical",
            "feldspar_pct": "numerical",
        },
    }
    return raw_df, metadata


def _generate_safelimit_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(88)
    row_count = 2000

    units = np.clip(rng.normal(loc=5.5, scale=2.0, size=row_count), 0.5, 12.0)
    weight = np.clip(rng.normal(loc=74.0, scale=14.0, size=row_count), 45.0, 120.0)
    duration = np.clip(rng.normal(loc=150.0, scale=70.0, size=row_count), 15.0, 420.0)
    gender = rng.choice(["female", "male"], size=row_count, p=[0.5, 0.5])
    stomach_fullness = rng.choice(["empty", "full"], size=row_count, p=[0.45, 0.55])

    widmark_r = np.where(gender == "male", 0.68, 0.55)
    stomach_absorption = np.where(stomach_fullness == "full", 0.82, 1.0)
    alcohol_grams = units * 8.0
    elapsed_hours = duration / 60.0

    bac_percent = (
        ((alcohol_grams * stomach_absorption) / (weight * 1000.0 * widmark_r)) * 100.0
        - (0.015 * elapsed_hours)
    )
    bac_percent = np.clip(bac_percent, 0.0, None)
    target = (bac_percent < 0.08).astype(int)

    raw_df = pd.DataFrame(
        {
            "row_id": range(row_count),
            "units": np.round(units, 1),
            "weight": np.round(weight, 1),
            "duration": np.round(duration).astype(int),
            "gender": gender,
            "stomach_fullness": stomach_fullness,
            "target": target,
        }
    )

    metadata = {
        "target_column": "target",
        "source_format": SAFELIMIT_SOURCE_VERSION,
        "source": "Synthetic SafeLimit data generated from the Widmark BAC equation described by Warren et al. (IUI 2023 / TiiS 2024).",
        "bac_threshold": 0.08,
        "class_labels": [
            "Above Limit",
            "Below Limit",
        ],
        "feature_types": {
            "units": "numerical",
            "weight": "numerical",
            "duration": "numerical",
            "gender": "categorical",
            "stomach_fullness": "categorical",
        },
        "categorical_levels": {
            "gender": ["female", "male"],
            "stomach_fullness": ["empty", "full"],
        },
    }
    return raw_df, metadata


BUILTIN_DATASET_GENERATORS = {
    "diabetes": _generate_diabetes_dataset,
    "ceramic": _generate_ceramic_dataset,
    "safelimit": _generate_safelimit_dataset,
}

BUILTIN_DATASET_SOURCE_VERSIONS = {
    "diabetes": DIABETES_SOURCE_VERSION,
    "ceramic": CERAMIC_SOURCE_VERSION,
    "safelimit": SAFELIMIT_SOURCE_VERSION,
}


def ensure_base_dataset(dataset_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_name = dataset_name.lower()
    dataset_dir = _dataset_dir(dataset_name)
    csv_path = _base_csv_path(dataset_name)
    metadata_path = _metadata_path(dataset_name)

    if csv_path.exists():
        dataframe = pd.read_csv(csv_path)
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_source_version = BUILTIN_DATASET_SOURCE_VERSIONS.get(dataset_name)
        if dataset_name in BUILTIN_DATASET_GENERATORS and (
            metadata.get("source_format") != expected_source_version
            or "target" not in dataframe.columns
        ):
            dataframe, metadata = BUILTIN_DATASET_GENERATORS[dataset_name]()
            _write_csv_atomically(dataframe, csv_path)
            _write_json_atomically(metadata, metadata_path)
        return dataframe, metadata

    if dataset_name not in BUILTIN_DATASET_GENERATORS:
        raise ValueError(
            f"Dataset '{dataset_name}' is unavailable. Add {csv_path.name} "
            "and an optional metadata.json file under src/data/<dataset_name>/."
        )

    dataframe, metadata = BUILTIN_DATASET_GENERATORS[dataset_name]()
    _write_csv_atomically(dataframe, csv_path)
    _write_json_atomically(metadata, metadata_path)
    return dataframe, metadata


def _infer_feature_types_and_ranges(
    dataframe: pd.DataFrame,
    feature_names: list[str],
    metadata: dict[str, Any],
    category_orders: dict[str, tuple[str, ...]] | None = None,
) -> tuple[list[str], list[list[Any]]]:
    feature_type_map = metadata.get("feature_types", {})
    categorical_levels = metadata.get("categorical_levels", {})
    category_orders = category_orders or {}
    feature_types: list[str] = []
    feature_ranges: list[list[Any]] = []

    for feature_name in feature_names:
        explicit_type = feature_type_map.get(feature_name)
        series = dataframe[feature_name]

        if explicit_type == "categorical" or series.dtype == object:
            feature_types.append("categorical")
            if feature_name in categorical_levels:
                feature_ranges.append(list(categorical_levels[feature_name]))
            elif feature_name in category_orders:
                observed_values = set(series.dropna().astype(str).unique().tolist())
                ordered_values = [
                    value
                    for value in category_orders[feature_name]
                    if value in observed_values
                ]
                remaining_values = [
                    value
                    for value in series.dropna().astype(str).unique().tolist()
                    if value not in set(ordered_values)
                ]
                feature_ranges.append(ordered_values + remaining_values)
            else:
                feature_ranges.append([str(value) for value in series.dropna().unique().tolist()])
            continue

        feature_types.append("numerical")
        min_value = float(series.quantile(0.02))
        max_value = float(series.quantile(0.98))
        if min_value == max_value:
            min_value = float(np.min(series))
            max_value = float(np.max(series))
        feature_ranges.append([min_value, max_value])

    return feature_types, feature_ranges


def _create_sequential_splits(
    dataframe: pd.DataFrame,
    dataset_name: str,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_paths = _split_paths(dataset_name)
    total_rows = len(dataframe)
    if total_rows < 3:
        raise ValueError("Dataset must contain at least three rows to build train/dev/test splits.")

    if target_column and target_column in dataframe.columns:
        train_df, dev_df, test_df = _create_stratified_splits(dataframe, target_column)
        _write_csv_atomically(train_df, split_paths["train"])
        _write_csv_atomically(dev_df, split_paths["dev"])
        _write_csv_atomically(test_df, split_paths["test"])
        return train_df, dev_df, test_df

    train_end = max(int(total_rows * 0.7), 1)
    dev_end = max(train_end + int(total_rows * 0.15), train_end + 1)
    dev_end = min(dev_end, total_rows - 1)

    train_df = dataframe.iloc[:train_end].reset_index(drop=True)
    dev_df = dataframe.iloc[train_end:dev_end].reset_index(drop=True)
    test_df = dataframe.iloc[dev_end:].reset_index(drop=True)

    if dev_df.empty or test_df.empty:
        raise ValueError("Sequential split produced an empty dev or test partition.")

    _write_csv_atomically(train_df, split_paths["train"])
    _write_csv_atomically(dev_df, split_paths["dev"])
    _write_csv_atomically(test_df, split_paths["test"])
    return train_df, dev_df, test_df


def _create_stratified_splits(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_parts = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for _, class_df in dataframe.groupby(target_column, sort=True):
        class_df = class_df.reset_index(drop=True)
        class_count = len(class_df)
        if class_count < 3:
            return _create_sequential_split_frames(dataframe)

        train_end = max(int(class_count * 0.7), 1)
        dev_end = max(train_end + int(class_count * 0.15), train_end + 1)
        dev_end = min(dev_end, class_count - 1)

        split_parts["train"].append(class_df.iloc[:train_end])
        split_parts["dev"].append(class_df.iloc[train_end:dev_end])
        split_parts["test"].append(class_df.iloc[dev_end:])

    return tuple(
        pd.concat(split_parts[split_name], ignore_index=True)
        .sort_values("row_id" if "row_id" in dataframe.columns else dataframe.columns[0])
        .reset_index(drop=True)
        for split_name in ("train", "dev", "test")
    )


def _create_sequential_split_frames(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_rows = len(dataframe)
    train_end = max(int(total_rows * 0.7), 1)
    dev_end = max(train_end + int(total_rows * 0.15), train_end + 1)
    dev_end = min(dev_end, total_rows - 1)

    train_df = dataframe.iloc[:train_end].reset_index(drop=True)
    dev_df = dataframe.iloc[train_end:dev_end].reset_index(drop=True)
    test_df = dataframe.iloc[dev_end:].reset_index(drop=True)
    return train_df, dev_df, test_df


def ensure_dataset_bundle(
    dataset_name: str,
    force_resplit: bool = False,
) -> DatasetBundle:
    dataset_name = dataset_name.lower()
    full_df, metadata = ensure_base_dataset(dataset_name)
    split_paths = _split_paths(dataset_name)

    should_resplit = force_resplit or not all(path.exists() for path in split_paths.values())

    if not should_resplit:
        train_df = pd.read_csv(split_paths["train"])
        dev_df = pd.read_csv(split_paths["dev"])
        test_df = pd.read_csv(split_paths["test"])
        combined_split_df = pd.concat(
            [train_df, dev_df, test_df],
            ignore_index=True,
        )
        split_row_count = len(train_df) + len(dev_df) + len(test_df)
        split_columns = set(train_df.columns) | set(dev_df.columns) | set(test_df.columns)
        full_columns = set(full_df.columns)
        if "row_id" in full_df.columns and "row_id" in combined_split_df.columns:
            split_rows_match_full = (
                sorted(combined_split_df["row_id"].tolist())
                == sorted(full_df["row_id"].tolist())
            )
        else:
            split_rows_match_full = combined_split_df.equals(full_df.reset_index(drop=True))
        should_resplit = (
            split_row_count != len(full_df)
            or split_columns != full_columns
            or not split_rows_match_full
        )

    target_column = metadata.get("target_column", "target")
    if target_column not in full_df.columns:
        target_column = full_df.columns[-1]

    if should_resplit:
        train_df, dev_df, test_df = _create_sequential_splits(
            full_df,
            dataset_name,
            target_column=target_column,
        )

    feature_names = [
        column
        for column in full_df.columns
        if column not in {target_column, "row_id"}
    ]
    runtime_config = get_dataset_runtime_config(dataset_name)
    friendly_feature_names = runtime_config.friendly_feature_names
    category_orders = runtime_config.category_orders
    feature_display_names = [
        friendly_feature_names.get(feature_name, feature_name.replace("_", " ").title())
        for feature_name in feature_names
    ]
    feature_types, feature_ranges = _infer_feature_types_and_ranges(
        train_df,
        feature_names,
        metadata,
        category_orders=category_orders,
    )
    friendly_category_names = runtime_config.friendly_category_names
    display_feature_ranges = [
        _get_display_feature_range(
            feature_name=feature_name,
            feature_type=feature_type,
            feature_range=feature_range,
            friendly_category_names=friendly_category_names,
        )
        for feature_name, feature_type, feature_range in zip(
            feature_names,
            feature_types,
            feature_ranges,
        )
    ]

    class_labels = metadata.get("class_labels")
    if not class_labels:
        unique_targets = sorted(full_df[target_column].dropna().unique().tolist())
        class_labels = [f"Class {value}" for value in unique_targets]

    return DatasetBundle(
        dataset_name=dataset_name,
        dataset_dir=_dataset_dir(dataset_name),
        full_df=full_df,
        train_df=train_df,
        dev_df=dev_df,
        test_df=test_df,
        feature_names=feature_names,
        feature_display_names=feature_display_names,
        feature_types=feature_types,
        feature_ranges=feature_ranges,
        display_feature_ranges=display_feature_ranges,
        friendly_category_names=friendly_category_names,
        category_orders=category_orders,
        target_column=target_column,
        class_labels=list(class_labels),
    )


def subset_dataset_bundle(
    dataset_bundle: DatasetBundle,
    selected_feature_names: list[str],
) -> DatasetBundle:
    feature_type_by_name = dict(zip(dataset_bundle.feature_names, dataset_bundle.feature_types))
    feature_range_by_name = dict(zip(dataset_bundle.feature_names, dataset_bundle.feature_ranges))
    display_feature_range_by_name = dict(
        zip(dataset_bundle.feature_names, dataset_bundle.display_feature_ranges)
    )
    feature_display_name_by_name = dict(
        zip(dataset_bundle.feature_names, dataset_bundle.feature_display_names)
    )

    return replace(
        dataset_bundle,
        feature_names=list(selected_feature_names),
        feature_display_names=[
            feature_display_name_by_name[feature_name]
            for feature_name in selected_feature_names
        ],
        feature_types=[
            feature_type_by_name[feature_name]
            for feature_name in selected_feature_names
        ],
        feature_ranges=[
            feature_range_by_name[feature_name]
            for feature_name in selected_feature_names
        ],
        display_feature_ranges=[
            display_feature_range_by_name[feature_name]
            for feature_name in selected_feature_names
        ],
    )


def exclude_features_from_dataset_bundle(
    dataset_bundle: DatasetBundle,
    excluded_feature_names: list[str] | tuple[str, ...],
) -> DatasetBundle:
    excluded_feature_name_set = set(excluded_feature_names)
    retained_feature_names = [
        feature_name
        for feature_name in dataset_bundle.feature_names
        if feature_name not in excluded_feature_name_set
    ]

    if not retained_feature_names:
        raise ValueError(
            f"All features for dataset '{dataset_bundle.dataset_name}' were excluded."
        )

    return subset_dataset_bundle(
        dataset_bundle=dataset_bundle,
        selected_feature_names=retained_feature_names,
    )


def display_feature_value(
    dataset_bundle: DatasetBundle,
    feature_name: str,
    value: Any,
) -> Any:
    category_name_map = dataset_bundle.friendly_category_names.get(feature_name, {})
    return category_name_map.get(str(value), value)


def display_feature_values(
    dataset_bundle: DatasetBundle,
    feature_names: list[str],
    values: list[Any],
) -> list[Any]:
    return [
        display_feature_value(dataset_bundle, feature_name, value)
        for feature_name, value in zip(feature_names, values)
    ]


def _get_display_feature_range(
    feature_name: str,
    feature_type: str,
    feature_range: list[Any],
    friendly_category_names: dict[str, dict[str, str]],
) -> list[Any]:
    if feature_type != "categorical":
        return feature_range

    category_name_map = friendly_category_names.get(feature_name, {})
    return [
        category_name_map.get(str(value), value)
        for value in feature_range
    ]
