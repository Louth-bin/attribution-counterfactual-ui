from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .model import MentalModel, Prediction
from .schema import FeatureSpace


EPSILON = 1e-12


@dataclass(frozen=True)
class RememberedInstance:
    profile: dict[str, Any]
    encoded: np.ndarray
    label: Any
    explanation: str
    attributions: dict[str, float] | None = None


@dataclass(frozen=True)
class RememberedChange:
    original: dict[str, Any]
    edited: dict[str, Any]
    original_encoded: np.ndarray
    edited_encoded: np.ndarray
    source_label: Any
    target_label: Any


class PerfectMemoryCognitiveModel:
    """Parameter-free perfect-memory baseline with shared feature relevance.

    Relevance is one non-negative, normalized feature vector used by all three
    mental models. Its evidence source depends on the explanation condition:
    class diagnosticity for no XAI, absolute displayed attribution for
    attribution XAI, and changed-feature frequency for counterfactual XAI.
    """

    def __init__(self, feature_space: FeatureSpace, class_labels: Sequence[Any]) -> None:
        if len(class_labels) != 2 or class_labels[0] == class_labels[1]:
            raise ValueError("PerfectMemoryCognitiveModel requires two class labels")
        self.feature_space = feature_space
        self.class_labels = tuple(class_labels)
        self.instances: list[RememberedInstance] = []
        self.changes: list[RememberedChange] = []
        self._explicit_relevance_sum = np.zeros(len(feature_space.features), dtype=float)
        self._explicit_relevance_count = 0
        self._derived_cache: dict[Any, Any] = {}

    def class_sign(self, label: Any) -> int:
        if label == self.class_labels[0]:
            return -1
        if label == self.class_labels[1]:
            return 1
        raise ValueError(f"Unknown class label {label!r}")

    def observe(
        self,
        profile: Mapping[str, Any],
        label: Any,
        *,
        explanation: str = "none",
        attributions: Mapping[str, float] | None = None,
        counterfactual_profile: Mapping[str, Any] | None = None,
        counterfactual_target_label: Any | None = None,
    ) -> None:
        self._derived_cache.clear()
        self.class_sign(label)
        if explanation not in {"none", "attribution", "counterfactual", "example"}:
            raise ValueError(f"Unknown explanation condition {explanation!r}")
        encoded = self.feature_space.encode(profile)
        attribution_copy = None if attributions is None else {
            name: float(value) for name, value in attributions.items()
        }
        self.instances.append(
            RememberedInstance(
                profile=dict(profile),
                encoded=encoded,
                label=label,
                explanation=explanation,
                attributions=attribution_copy,
            )
        )

        if explanation == "attribution" and attribution_copy:
            signal = np.array(
                [abs(attribution_copy.get(feature.name, 0.0)) for feature in self.feature_space.features],
                dtype=float,
            )
            self._add_relevance_signal(signal)

        if explanation == "counterfactual":
            if counterfactual_profile is None or counterfactual_target_label is None:
                raise ValueError(
                    "Counterfactual learning requires a profile and target label"
                )
            self.class_sign(counterfactual_target_label)
            edited_encoded = self.feature_space.encode(counterfactual_profile)
            change = RememberedChange(
                original=dict(profile),
                edited=dict(counterfactual_profile),
                original_encoded=encoded,
                edited_encoded=edited_encoded,
                source_label=label,
                target_label=counterfactual_target_label,
            )
            self.changes.append(change)
            # A counterfactual explanation supplies a second labeled example:
            # the edited profile is presented as an instance of the target class.
            # Retaining it lets exemplar and prototype representations learn from
            # the same explanation that supplies the attribution model's change
            # direction evidence.
            self.instances.append(
                RememberedInstance(
                    profile=dict(counterfactual_profile),
                    encoded=edited_encoded,
                    label=counterfactual_target_label,
                    explanation=explanation,
                )
            )
            changed_signal = np.array(
                list(
                    self.feature_space.per_feature_distance(
                        encoded, edited_encoded
                    ).values()
                ),
                dtype=float,
            )
            changed_signal = (changed_signal > EPSILON).astype(float)
            self._add_relevance_signal(changed_signal)

    def predict(
        self,
        profile: Mapping[str, Any],
        mental_model: MentalModel | str,
    ) -> Prediction:
        model = MentalModel(mental_model)
        encoded = self.feature_space.encode(profile)
        if model is MentalModel.EXEMPLAR:
            return self._predict_exemplar(encoded)
        if model is MentalModel.ATTRIBUTION:
            return self._predict_attribution(encoded)
        return self._predict_prototype(encoded)

    def relevance(self) -> np.ndarray:
        if "relevance" in self._derived_cache:
            return self._derived_cache["relevance"].copy()
        if self._explicit_relevance_count:
            values = self._explicit_relevance_sum / self._explicit_relevance_count
            result = self._normalize_relevance(values)
        else:
            result = self._data_relevance()
        self._derived_cache["relevance"] = result
        return result.copy()

    def polarity(self) -> np.ndarray:
        """Return attribution direction for each encoded dimension."""
        if "polarity" in self._derived_cache:
            return self._derived_cache["polarity"].copy()
        data_direction = self._data_direction()
        middle = self.attribution_middle()
        attribution_score = np.zeros(self.feature_space.encoded_size, dtype=float)
        counterfactual_score = np.zeros(self.feature_space.encoded_size, dtype=float)

        for instance in self.instances:
            if not instance.attributions:
                continue
            deviation = instance.encoded - middle
            for feature in self.feature_space.features:
                value = float(instance.attributions.get(feature.name, 0.0))
                feature_slice = self.feature_space.feature_slice(feature.name)
                attribution_score[feature_slice] += value * deviation[feature_slice]

        for change in self.changes:
            target_sign = self.class_sign(change.target_label)
            counterfactual_score += target_sign * (
                change.edited_encoded - change.original_encoded
            )

        if np.any(np.abs(attribution_score) > EPSILON):
            direction_score = attribution_score
        elif np.any(np.abs(counterfactual_score) > EPSILON):
            direction_score = counterfactual_score
        else:
            direction_score = data_direction
        direction_score = np.where(
            np.abs(direction_score) > EPSILON, direction_score, data_direction
        )
        result = np.sign(direction_score)
        self._derived_cache["polarity"] = result
        return result.copy()

    def attribution_middle(self) -> np.ndarray:
        if "attribution_middle" in self._derived_cache:
            return self._derived_cache["attribution_middle"].copy()
        grouped = self._instances_by_class()
        if not grouped[self.class_labels[0]] or not grouped[self.class_labels[1]]:
            if not self.instances:
                result = np.full(self.feature_space.encoded_size, 0.5)
            else:
                result = np.mean([instance.encoded for instance in self.instances], axis=0)
        else:
            negative = np.mean(grouped[self.class_labels[0]], axis=0)
            positive = np.mean(grouped[self.class_labels[1]], axis=0)
            result = (negative + positive) / 2.0
        self._derived_cache["attribution_middle"] = result
        return result.copy()

    def attribution_weights(self) -> np.ndarray:
        if "attribution_weights" in self._derived_cache:
            return self._derived_cache["attribution_weights"].copy()
        feature_relevance = self.relevance()
        direction = self.polarity()
        weights = np.zeros(self.feature_space.encoded_size, dtype=float)
        for index, feature in enumerate(self.feature_space.features):
            feature_slice = self.feature_space.feature_slice(feature.name)
            width = feature_slice.stop - feature_slice.start
            weights[feature_slice] = feature_relevance[index] * direction[feature_slice] / width
        self._derived_cache["attribution_weights"] = weights
        return weights.copy()

    def attribution_contributions(self, profile: Mapping[str, Any]) -> dict[str, float]:
        encoded = self.feature_space.encode(profile)
        products = self.attribution_weights() * (encoded - self.attribution_middle())
        return {
            feature.name: float(np.sum(products[self.feature_space.feature_slice(feature.name)]))
            for feature in self.feature_space.features
        }

    def global_importance(self) -> dict[str, float]:
        return {
            feature.name: float(value)
            for feature, value in zip(self.feature_space.features, self.relevance())
        }

    def prototype_profile(self, label: Any) -> dict[str, Any]:
        cache_key = ("prototype_profile", label)
        if cache_key in self._derived_cache:
            return dict(self._derived_cache[cache_key])
        values = self._class_matrix(label)
        if values.size == 0:
            raise ValueError(f"No remembered instances for class {label!r}")
        centre = np.median(values, axis=0)
        for feature in self.feature_space.features:
            if feature.kind == "categorical":
                feature_slice = self.feature_space.feature_slice(feature.name)
                winner = int(np.argmax(np.mean(values[:, feature_slice], axis=0)))
                centre[feature_slice] = 0.0
                centre[feature_slice.start + winner] = 1.0
        result = self.feature_space.decode(centre)
        self._derived_cache[cache_key] = result
        return dict(result)

    def prototype_ranges(self, label: Any) -> dict[str, Any]:
        cache_key = ("prototype_ranges", label)
        if cache_key in self._derived_cache:
            return dict(self._derived_cache[cache_key])
        values = self._class_matrix(label)
        if values.size == 0:
            raise ValueError(f"No remembered instances for class {label!r}")
        lower = np.quantile(values, 0.25, axis=0)
        upper = np.quantile(values, 0.75, axis=0)
        centre = np.median(values, axis=0)
        result: dict[str, Any] = {}
        for feature in self.feature_space.features:
            feature_slice = self.feature_space.feature_slice(feature.name)
            if feature.kind == "numerical":
                low_vector = centre.copy()
                high_vector = centre.copy()
                low_vector[feature_slice] = lower[feature_slice]
                high_vector[feature_slice] = upper[feature_slice]
                result[feature.name] = (
                    self.feature_space.decode(low_vector)[feature.name],
                    self.feature_space.decode(high_vector)[feature.name],
                )
            else:
                frequencies = np.mean(values[:, feature_slice], axis=0)
                maximum = np.max(frequencies)
                result[feature.name] = tuple(
                    category
                    for category, frequency in zip(feature.categories, frequencies)
                    if np.isclose(frequency, maximum)
                )
        self._derived_cache[cache_key] = result
        return dict(result)

    def prototype_mismatches(
        self, profile: Mapping[str, Any], label: Any
    ) -> dict[str, float]:
        values = self._class_matrix(label)
        if values.size == 0:
            raise ValueError(f"No remembered instances for class {label!r}")
        encoded = self.feature_space.encode(profile)
        lower = np.quantile(values, 0.25, axis=0)
        upper = np.quantile(values, 0.75, axis=0)
        mismatches: dict[str, float] = {}
        for feature in self.feature_space.features:
            feature_slice = self.feature_space.feature_slice(feature.name)
            if feature.kind == "numerical":
                current = encoded[feature_slice][0]
                mismatches[feature.name] = float(
                    max(
                        lower[feature_slice][0] - current,
                        current - upper[feature_slice][0],
                        0.0,
                    )
                )
            else:
                frequencies = np.mean(values[:, feature_slice], axis=0)
                observed = int(np.argmax(encoded[feature_slice]))
                mismatches[feature.name] = float(
                    not np.isclose(frequencies[observed], np.max(frequencies))
                )
        return mismatches

    def _predict_exemplar(self, encoded: np.ndarray) -> Prediction:
        if not self.instances:
            return self._uniform_prediction(MentalModel.EXEMPLAR)
        relevance = self.relevance()
        totals = {label: 1.0 for label in self.class_labels}  # fixed symmetric prior
        for instance in self.instances:
            per_feature = self.feature_space.per_feature_distance(
                encoded, instance.encoded
            )
            distance = sum(
                relevance[index] * per_feature[feature.name]
                for index, feature in enumerate(self.feature_space.features)
            )
            totals[instance.label] += math.exp(-distance)
        denominator = sum(totals.values())
        probabilities = {label: totals[label] / denominator for label in self.class_labels}
        label = max(self.class_labels, key=probabilities.get)
        evidence = probabilities[self.class_labels[1]] - probabilities[self.class_labels[0]]
        return Prediction(MentalModel.EXEMPLAR, label, probabilities, float(evidence))

    def _predict_attribution(self, encoded: np.ndarray) -> Prediction:
        weights = self.attribution_weights()
        score = float(np.dot(weights, encoded - self.attribution_middle()))
        if not np.any(np.abs(weights) > EPSILON):
            return self._class_prior_prediction(MentalModel.ATTRIBUTION)
        positive_probability = 1.0 / (1.0 + math.exp(-score))
        probabilities = {
            self.class_labels[0]: 1.0 - positive_probability,
            self.class_labels[1]: positive_probability,
        }
        label = self.class_labels[1] if score >= 0 else self.class_labels[0]
        return Prediction(MentalModel.ATTRIBUTION, label, probabilities, score)

    def _predict_prototype(self, encoded: np.ndarray) -> Prediction:
        if any(self._class_matrix(label).size == 0 for label in self.class_labels):
            return self._class_prior_prediction(MentalModel.PROTOTYPE)
        relevance = self.relevance()
        distances = {
            label: self._prototype_distance(encoded, label, relevance)
            for label in self.class_labels
        }
        similarities = {label: math.exp(-distance) for label, distance in distances.items()}
        denominator = sum(similarities.values())
        probabilities = {
            label: similarities[label] / denominator for label in self.class_labels
        }
        label = min(self.class_labels, key=distances.get)
        evidence = distances[self.class_labels[0]] - distances[self.class_labels[1]]
        return Prediction(MentalModel.PROTOTYPE, label, probabilities, float(evidence))

    def _prototype_distance(
        self, encoded: np.ndarray, label: Any, relevance: np.ndarray
    ) -> float:
        mismatches = self.prototype_mismatches(
            self.feature_space.decode(encoded), label
        )
        return float(
            sum(
                relevance[index] * mismatches[feature.name]
                for index, feature in enumerate(self.feature_space.features)
            )
        )

    def _data_relevance(self) -> np.ndarray:
        grouped = self._instances_by_class()
        if not grouped[self.class_labels[0]] or not grouped[self.class_labels[1]]:
            return np.full(len(self.feature_space.features), 1.0 / len(self.feature_space.features))
        negative = np.asarray(grouped[self.class_labels[0]], dtype=float)
        positive = np.asarray(grouped[self.class_labels[1]], dtype=float)
        signals: list[float] = []
        for feature in self.feature_space.features:
            feature_slice = self.feature_space.feature_slice(feature.name)
            mean_difference = np.sum(
                np.abs(
                    np.mean(positive[:, feature_slice], axis=0)
                    - np.mean(negative[:, feature_slice], axis=0)
                )
            )
            spread = np.sum(
                np.std(positive[:, feature_slice], axis=0)
                + np.std(negative[:, feature_slice], axis=0)
            )
            signals.append(float(mean_difference / (spread + EPSILON)))
        return self._normalize_relevance(np.asarray(signals, dtype=float))

    def _data_direction(self) -> np.ndarray:
        grouped = self._instances_by_class()
        if not grouped[self.class_labels[0]] or not grouped[self.class_labels[1]]:
            return np.zeros(self.feature_space.encoded_size, dtype=float)
        negative = np.mean(grouped[self.class_labels[0]], axis=0)
        positive = np.mean(grouped[self.class_labels[1]], axis=0)
        return positive - negative

    def _instances_by_class(self) -> dict[Any, list[np.ndarray]]:
        if "instances_by_class" in self._derived_cache:
            return self._derived_cache["instances_by_class"]
        grouped = {label: [] for label in self.class_labels}
        for instance in self.instances:
            grouped[instance.label].append(instance.encoded)
        self._derived_cache["instances_by_class"] = grouped
        return grouped

    def _class_matrix(self, label: Any) -> np.ndarray:
        cache_key = ("class_matrix", label)
        if cache_key in self._derived_cache:
            return self._derived_cache[cache_key]
        values = [instance.encoded for instance in self.instances if instance.label == label]
        if not values:
            result = np.empty((0, self.feature_space.encoded_size), dtype=float)
        else:
            result = np.asarray(values, dtype=float)
        self._derived_cache[cache_key] = result
        return result

    def _add_relevance_signal(self, signal: np.ndarray) -> None:
        normalized = self._normalize_relevance(signal)
        self._explicit_relevance_sum += normalized
        self._explicit_relevance_count += 1

    def _normalize_relevance(self, values: np.ndarray) -> np.ndarray:
        values = np.maximum(np.asarray(values, dtype=float), 0.0)
        total = float(np.sum(values))
        if total <= EPSILON:
            return np.full(len(self.feature_space.features), 1.0 / len(self.feature_space.features))
        return values / total

    def _class_prior_prediction(self, model: MentalModel) -> Prediction:
        counts = {
            label: 1 + sum(instance.label == label for instance in self.instances)
            for label in self.class_labels
        }
        denominator = sum(counts.values())
        probabilities = {label: counts[label] / denominator for label in self.class_labels}
        label = max(self.class_labels, key=probabilities.get)
        evidence = probabilities[self.class_labels[1]] - probabilities[self.class_labels[0]]
        return Prediction(model, label, probabilities, float(evidence))

    def _uniform_prediction(self, model: MentalModel) -> Prediction:
        probabilities = {label: 0.5 for label in self.class_labels}
        return Prediction(model, self.class_labels[0], probabilities, 0.0)
