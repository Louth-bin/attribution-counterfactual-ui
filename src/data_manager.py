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
GERMAN_CREDIT_SOURCE_VERSION = "openml_credit_g_flipped_labels_v1"


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
                "pregnancies",
                "glucose",
                "blood_pressure",
                "skin_thickness",
                "insulin",
                "bmi",
                "diabetes_pedigree_function",
                "age",
                "target",
            ],
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
            "pregnancies": "numerical",
            "glucose": "numerical",
            "blood_pressure": "numerical",
            "skin_thickness": "numerical",
            "insulin": "numerical",
            "bmi": "numerical",
            "diabetes_pedigree_function": "numerical",
            "age": "numerical",
        },
    }
    return raw_df, metadata


def _generate_german_credit_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    source_csv_path = _source_csv_path("german_credit", "openml_source.csv")
    if source_csv_path.exists():
        raw_df = pd.read_csv(source_csv_path)
    else:
        try:
            from sklearn.datasets import fetch_openml

            raw_df = fetch_openml(
                name="credit-g",
                version=1,
                as_frame=True,
            ).frame
        except Exception as error:
            raise ValueError(
                "The German credit source file is not available locally and could not be downloaded. "
                f"Expected a CSV at {source_csv_path}."
            ) from error

        _write_csv_atomically(raw_df, source_csv_path)

    raw_df = raw_df.copy()
    raw_df.insert(0, "row_id", range(len(raw_df)))
    raw_df["target"] = raw_df["class"].map({"bad": 0, "good": 1})
    raw_df = raw_df.drop(columns=["class"])

    feature_type_overrides = {
        "duration": "numerical",
        "credit_amount": "numerical",
        "installment_commitment": "numerical",
        "residence_since": "numerical",
        "age": "numerical",
        "existing_credits": "numerical",
        "num_dependents": "numerical",
    }
    feature_types = {
        column: feature_type_overrides.get(column, "categorical")
        for column in raw_df.columns
        if column not in {"row_id", "target"}
    }

    metadata = {
        "target_column": "target",
        "source_format": GERMAN_CREDIT_SOURCE_VERSION,
        "class_labels": [
            "Bad Credit Risk",
            "Good Credit Risk",
        ],
        "feature_types": feature_types,
    }
    return raw_df, metadata


BUILTIN_DATASET_GENERATORS = {
    "diabetes": _generate_diabetes_dataset,
    "german_credit": _generate_german_credit_dataset,
}

BUILTIN_DATASET_SOURCE_VERSIONS = {
    "diabetes": DIABETES_SOURCE_VERSION,
    "german_credit": GERMAN_CREDIT_SOURCE_VERSION,
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
        min_value = float(np.min(series))
        max_value = float(np.max(series))
        feature_ranges.append([min_value, max_value])

    return feature_types, feature_ranges


def _create_sequential_splits(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_paths = _split_paths(dataset_name)
    total_rows = len(dataframe)
    if total_rows < 3:
        raise ValueError("Dataset must contain at least three rows to build train/dev/test splits.")

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
        should_resplit = (
            split_row_count != len(full_df)
            or split_columns != full_columns
            or not combined_split_df.equals(full_df.reset_index(drop=True))
        )

    if should_resplit:
        train_df, dev_df, test_df = _create_sequential_splits(full_df, dataset_name)

    target_column = metadata.get("target_column", "target")
    if target_column not in full_df.columns:
        target_column = full_df.columns[-1]

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
        full_df,
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
