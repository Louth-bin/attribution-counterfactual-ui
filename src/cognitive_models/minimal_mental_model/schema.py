from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    """One user-visible feature; categorical levels are encoded one-hot."""

    name: str
    kind: str = "numerical"
    low: float | None = None
    high: float | None = None
    categories: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"numerical", "categorical"}:
            raise ValueError(f"Unsupported feature kind {self.kind!r}")
        if self.kind == "numerical":
            if self.low is None or self.high is None or self.high <= self.low:
                raise ValueError(f"Numerical feature {self.name!r} needs low < high")
        elif len(self.categories) < 2:
            raise ValueError(f"Categorical feature {self.name!r} needs at least two levels")

    @classmethod
    def numerical(cls, name: str, low: float, high: float) -> "FeatureSpec":
        return cls(name=name, kind="numerical", low=float(low), high=float(high))

    @classmethod
    def categorical(cls, name: str, categories: Iterable[Any]) -> "FeatureSpec":
        return cls(name=name, kind="categorical", categories=tuple(categories))


class FeatureSpace:
    """Encode profiles and compute feature-level range-normalized distances."""

    def __init__(self, features: Iterable[FeatureSpec]) -> None:
        self.features = tuple(features)
        if not self.features:
            raise ValueError("At least one feature is required")
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique")
        self._slices: dict[str, slice] = {}
        start = 0
        for feature in self.features:
            width = 1 if feature.kind == "numerical" else len(feature.categories)
            self._slices[feature.name] = slice(start, start + width)
            start += width
        self.encoded_size = start

    @classmethod
    def from_dataset_fields(
        cls,
        feature_names: Iterable[str],
        feature_types: Iterable[str],
        feature_ranges: Iterable[Iterable[Any]],
    ) -> "FeatureSpace":
        """Build directly from the fields exposed by ``DatasetBundle``."""
        feature_names = tuple(feature_names)
        feature_types = tuple(feature_types)
        feature_ranges = tuple(feature_ranges)
        if not (len(feature_names) == len(feature_types) == len(feature_ranges)):
            raise ValueError("Feature names, types, and ranges must have equal lengths")
        specs: list[FeatureSpec] = []
        for name, kind, values in zip(feature_names, feature_types, feature_ranges):
            values = tuple(values)
            if kind == "categorical":
                specs.append(FeatureSpec.categorical(name, values))
            else:
                if len(values) != 2:
                    raise ValueError(f"Numerical range for {name!r} must contain low and high")
                specs.append(FeatureSpec.numerical(name, float(values[0]), float(values[1])))
        return cls(specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    def feature_slice(self, name: str) -> slice:
        return self._slices[name]

    def spec(self, name: str) -> FeatureSpec:
        return next(feature for feature in self.features if feature.name == name)

    def encode(self, profile: Mapping[str, Any]) -> np.ndarray:
        missing = [name for name in self.names if name not in profile]
        if missing:
            raise ValueError(f"Profile is missing features: {missing}")
        vector = np.zeros(self.encoded_size, dtype=float)
        for feature in self.features:
            feature_slice = self.feature_slice(feature.name)
            value = profile[feature.name]
            if feature.kind == "numerical":
                assert feature.low is not None and feature.high is not None
                number = float(value)
                vector[feature_slice.start] = np.clip(
                    (number - feature.low) / (feature.high - feature.low), 0.0, 1.0
                )
            else:
                try:
                    category_index = feature.categories.index(value)
                except ValueError as error:
                    raise ValueError(
                        f"Unknown level {value!r} for feature {feature.name!r}"
                    ) from error
                vector[feature_slice.start + category_index] = 1.0
        return vector

    def decode(self, vector: np.ndarray) -> dict[str, Any]:
        vector = self.clip_vector(vector)
        profile: dict[str, Any] = {}
        for feature in self.features:
            values = vector[self.feature_slice(feature.name)]
            if feature.kind == "numerical":
                assert feature.low is not None and feature.high is not None
                profile[feature.name] = float(
                    feature.low + values[0] * (feature.high - feature.low)
                )
            else:
                profile[feature.name] = feature.categories[int(np.argmax(values))]
        return profile

    def clip_vector(self, vector: np.ndarray) -> np.ndarray:
        clipped = np.asarray(vector, dtype=float).copy()
        if clipped.shape != (self.encoded_size,):
            raise ValueError(
                f"Expected encoded shape {(self.encoded_size,)}, got {clipped.shape}"
            )
        for feature in self.features:
            feature_slice = self.feature_slice(feature.name)
            if feature.kind == "numerical":
                clipped[feature_slice] = np.clip(clipped[feature_slice], 0.0, 1.0)
            else:
                winner = int(np.argmax(clipped[feature_slice]))
                clipped[feature_slice] = 0.0
                clipped[feature_slice.start + winner] = 1.0
        return clipped

    def per_feature_distance(self, left: np.ndarray, right: np.ndarray) -> dict[str, float]:
        distances: dict[str, float] = {}
        for feature in self.features:
            feature_slice = self.feature_slice(feature.name)
            if feature.kind == "numerical":
                distances[feature.name] = float(abs(left[feature_slice][0] - right[feature_slice][0]))
            else:
                distances[feature.name] = float(
                    np.argmax(left[feature_slice]) != np.argmax(right[feature_slice])
                )
        return distances

    def distance(
        self,
        left: np.ndarray,
        right: np.ndarray,
        feature_names: Iterable[str] | None = None,
    ) -> float:
        distances = self.per_feature_distance(left, right)
        selected = self.names if feature_names is None else tuple(feature_names)
        if not selected:
            raise ValueError("At least one feature is needed for distance")
        return float(np.mean([distances[name] for name in selected]))
