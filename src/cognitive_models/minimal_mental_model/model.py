from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from .memory import DeclarativeMemory
from .schema import FeatureSpace


EPSILON = 1e-9


class MentalModel(str, Enum):
    EXEMPLAR = "exemplar"
    ATTRIBUTION = "attribution"
    PROTOTYPE = "prototype"


@dataclass(frozen=True)
class Prediction:
    mental_model: MentalModel
    label: Any
    probabilities: dict[Any, float]
    evidence: float


@dataclass
class PrototypeState:
    count: int
    center: np.ndarray
    squared_deviation: np.ndarray

    @classmethod
    def empty(cls, size: int) -> "PrototypeState":
        return cls(count=0, center=np.zeros(size), squared_deviation=np.zeros(size))

    def update(self, value: np.ndarray) -> None:
        self.count += 1
        delta = value - self.center
        self.center += delta / self.count
        second_delta = value - self.center
        self.squared_deviation += delta * second_delta

    @property
    def deviation(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros_like(self.center)
        return np.sqrt(np.maximum(self.squared_deviation / (self.count - 1), 0.0))


class MinimalCognitiveModel:
    """Shared memory with exemplar, attribution, and prototype mental models.

    Only ``learning_rate`` and ``retrieval_threshold`` are intended to be fit
    per participant. Range scaling, ACT-R decay (0.5), deterministic retrieval,
    and all strategy rules are structural choices.
    """

    def __init__(
        self,
        feature_space: FeatureSpace,
        class_labels: Sequence[Any],
        *,
        learning_rate: float = 0.25,
        retrieval_threshold: float = -2.0,
    ) -> None:
        if len(class_labels) != 2 or class_labels[0] == class_labels[1]:
            raise ValueError("The minimal model currently requires two distinct class labels")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        self.feature_space = feature_space
        self.class_labels = tuple(class_labels)
        self.learning_rate = float(learning_rate)
        self.memory = DeclarativeMemory(
            retrieval_threshold=retrieval_threshold,
            decay=0.5,
        )
        self.prototypes = {
            label: PrototypeState.empty(feature_space.encoded_size)
            for label in self.class_labels
        }
        self._explanation_weights = np.zeros(feature_space.encoded_size, dtype=float)
        self._has_explanation_weights = False

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
    ) -> None:
        """Learn one labeled profile, optionally with signed feature attributions."""
        self.class_sign(label)
        if explanation not in {"none", "example", "attribution", "counterfactual"}:
            raise ValueError(f"Unknown explanation type {explanation!r}")
        encoded = self.feature_space.encode(profile)
        self.memory.advance()
        chunk = self.memory.store(
            "instance",
            slots={"label": label, "explanation": explanation},
            payload={"profile": dict(profile), "encoded": encoded.copy()},
        )
        if explanation == "example":
            # Seeing the case as an explanation supplies one parameter-free
            # extra rehearsal, increasing its ACT-R base-level activation.
            self.memory.touch(chunk)
        self.prototypes[label].update(encoded)
        if explanation == "attribution" and attributions is not None:
            self._learn_from_attributions(encoded, attributions)

    def observe_counterfactual(
        self,
        original: Mapping[str, Any],
        edited: Mapping[str, Any],
        source_label: Any,
        target_label: Any,
        *,
        attributions: Mapping[str, float] | None = None,
    ) -> None:
        """Store an original/edited pair and learn direction and importance from it."""
        if source_label == target_label:
            raise ValueError("A counterfactual must have different source and target labels")
        self.observe(original, source_label, explanation="counterfactual")
        self.observe(edited, target_label, explanation="counterfactual")
        original_encoded = self.feature_space.encode(original)
        edited_encoded = self.feature_space.encode(edited)
        self.memory.store(
            "change",
            slots={"source_label": source_label, "target_label": target_label},
            payload={
                "original": dict(original),
                "edited": dict(edited),
                "original_encoded": original_encoded,
                "edited_encoded": edited_encoded,
            },
        )
        self._learn_from_counterfactual(original_encoded, edited_encoded, target_label)
        if attributions is not None:
            self._learn_from_attributions(original_encoded, attributions)

    def predict(
        self,
        profile: Mapping[str, Any],
        mental_model: MentalModel | str,
    ) -> Prediction:
        model = MentalModel(mental_model)
        encoded = self.feature_space.encode(profile)
        self.memory.advance()
        if model is MentalModel.EXEMPLAR:
            return self._predict_exemplar(encoded)
        if model is MentalModel.ATTRIBUTION:
            return self._predict_attribution(encoded)
        return self._predict_prototype(encoded)

    def attribution_weights(self) -> np.ndarray:
        """Return explanation weights, or standardized class contrast as fallback."""
        if self._has_explanation_weights and np.linalg.norm(
            self._explanation_weights, ord=1
        ) > EPSILON:
            weights = self._explanation_weights.copy()
        else:
            negative = self.prototypes[self.class_labels[0]]
            positive = self.prototypes[self.class_labels[1]]
            if negative.count == 0 or positive.count == 0:
                return np.zeros(self.feature_space.encoded_size)
            scale = negative.deviation + positive.deviation
            scale = np.where(scale > EPSILON, scale, 1.0)
            weights = (positive.center - negative.center) / scale
        norm = np.linalg.norm(weights, ord=1)
        return weights if norm <= EPSILON else weights / norm

    def attribution_middle(self) -> np.ndarray:
        negative = self.prototypes[self.class_labels[0]]
        positive = self.prototypes[self.class_labels[1]]
        if negative.count == 0 or positive.count == 0:
            return np.full(self.feature_space.encoded_size, 0.5)
        return (negative.center + positive.center) / 2.0

    def attribution_contributions(self, profile: Mapping[str, Any]) -> dict[str, float]:
        encoded = self.feature_space.encode(profile)
        products = self.attribution_weights() * (encoded - self.attribution_middle())
        return {
            feature.name: float(np.sum(products[self.feature_space.feature_slice(feature.name)]))
            for feature in self.feature_space.features
        }

    def global_importance(self) -> dict[str, float]:
        weights = self.attribution_weights()
        return {
            feature.name: float(
                np.sum(np.abs(weights[self.feature_space.feature_slice(feature.name)]))
            )
            for feature in self.feature_space.features
        }

    def prototype_profile(self, label: Any) -> dict[str, Any]:
        state = self.prototypes[label]
        if state.count == 0:
            raise ValueError(f"No remembered examples for class {label!r}")
        return self.feature_space.decode(state.center)

    def prototype_ranges(self, label: Any) -> dict[str, Any]:
        state = self.prototypes[label]
        if state.count == 0:
            raise ValueError(f"No remembered examples for class {label!r}")
        ranges: dict[str, Any] = {}
        for feature in self.feature_space.features:
            feature_slice = self.feature_space.feature_slice(feature.name)
            if feature.kind == "numerical":
                center = state.center[feature_slice][0]
                deviation = state.deviation[feature_slice][0]
                low_profile = self.feature_space.decode(
                    self._replace_numeric(state.center, feature.name, center - deviation)
                )
                high_profile = self.feature_space.decode(
                    self._replace_numeric(state.center, feature.name, center + deviation)
                )
                ranges[feature.name] = (
                    low_profile[feature.name],
                    high_profile[feature.name],
                )
            else:
                values = state.center[feature_slice]
                maximum = np.max(values)
                ranges[feature.name] = tuple(
                    category
                    for category, value in zip(feature.categories, values)
                    if np.isclose(value, maximum)
                )
        return ranges

    def prototype_mismatches(
        self, profile: Mapping[str, Any], label: Any
    ) -> dict[str, float]:
        """Return each feature's normalized distance outside a class's typical range."""
        state = self.prototypes[label]
        if state.count == 0:
            raise ValueError(f"No remembered examples for class {label!r}")
        encoded = self.feature_space.encode(profile)
        mismatches: dict[str, float] = {}
        for feature in self.feature_space.features:
            feature_slice = self.feature_space.feature_slice(feature.name)
            if feature.kind == "numerical":
                value = encoded[feature_slice][0]
                center = state.center[feature_slice][0]
                deviation = state.deviation[feature_slice][0]
                lower = max(0.0, center - deviation)
                upper = min(1.0, center + deviation)
                mismatches[feature.name] = float(max(lower - value, value - upper, 0.0))
            else:
                observed = int(np.argmax(encoded[feature_slice]))
                frequencies = state.center[feature_slice]
                mismatches[feature.name] = float(
                    not np.isclose(frequencies[observed], np.max(frequencies))
                )
        return mismatches

    def _replace_numeric(self, vector: np.ndarray, name: str, value: float) -> np.ndarray:
        replaced = vector.copy()
        replaced[self.feature_space.feature_slice(name)] = np.clip(value, 0.0, 1.0)
        return replaced

    def _predict_exemplar(self, encoded: np.ndarray) -> Prediction:
        retrieved = self.memory.retrieve_many(kind="instance")
        if not retrieved:
            return self._uniform_prediction(MentalModel.EXEMPLAR)
        totals = {label: 0.0 for label in self.class_labels}
        best_chunk = None
        best_weight = -math.inf
        for item in retrieved:
            exemplar = np.asarray(item.chunk.payload["encoded"], dtype=float)
            distance = self.feature_space.distance(encoded, exemplar)
            weight = math.exp(-distance + item.activation)
            label = item.chunk.slots["label"]
            totals[label] += weight
            if weight > best_weight:
                best_weight = weight
                best_chunk = item.chunk
        if best_chunk is not None:
            self.memory.touch(best_chunk)
        denominator = sum(totals.values())
        probabilities = {label: totals[label] / denominator for label in self.class_labels}
        label = max(self.class_labels, key=probabilities.get)
        evidence = probabilities[self.class_labels[1]] - probabilities[self.class_labels[0]]
        return Prediction(MentalModel.EXEMPLAR, label, probabilities, float(evidence))

    def _predict_attribution(self, encoded: np.ndarray) -> Prediction:
        weights = self.attribution_weights()
        score = float(np.dot(weights, encoded - self.attribution_middle()))
        positive_probability = 1.0 / (1.0 + math.exp(-float(np.clip(score, -700, 700))))
        probabilities = {
            self.class_labels[0]: 1.0 - positive_probability,
            self.class_labels[1]: positive_probability,
        }
        label = self.class_labels[1] if score >= 0 else self.class_labels[0]
        return Prediction(MentalModel.ATTRIBUTION, label, probabilities, score)

    def _predict_prototype(self, encoded: np.ndarray) -> Prediction:
        if any(state.count == 0 for state in self.prototypes.values()):
            return self._uniform_prediction(MentalModel.PROTOTYPE)
        distances = {
            label: self._distance_to_prototype_range(encoded, self.prototypes[label])
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

    def _distance_to_prototype_range(
        self, encoded: np.ndarray, state: PrototypeState
    ) -> float:
        feature_distances: list[float] = []
        for feature in self.feature_space.features:
            feature_slice = self.feature_space.feature_slice(feature.name)
            if feature.kind == "numerical":
                value = encoded[feature_slice][0]
                center = state.center[feature_slice][0]
                deviation = state.deviation[feature_slice][0]
                lower = max(0.0, center - deviation)
                upper = min(1.0, center + deviation)
                feature_distances.append(max(lower - value, value - upper, 0.0))
            else:
                observed = int(np.argmax(encoded[feature_slice]))
                frequencies = state.center[feature_slice]
                feature_distances.append(
                    float(not np.isclose(frequencies[observed], np.max(frequencies)))
                )
        return float(np.mean(feature_distances))

    def _uniform_prediction(self, model: MentalModel) -> Prediction:
        probabilities = {label: 0.5 for label in self.class_labels}
        return Prediction(model, self.class_labels[0], probabilities, 0.0)

    def _learn_from_attributions(
        self,
        encoded: np.ndarray,
        attributions: Mapping[str, float],
    ) -> None:
        middle = self.attribution_middle()
        signal = np.zeros(self.feature_space.encoded_size, dtype=float)
        for feature in self.feature_space.features:
            if feature.name not in attributions:
                continue
            feature_slice = self.feature_space.feature_slice(feature.name)
            deviation = encoded[feature_slice] - middle[feature_slice]
            active = int(np.argmax(np.abs(deviation)))
            denominator = deviation[active]
            value = float(attributions[feature.name])
            signal[feature_slice.start + active] = (
                value / denominator if abs(denominator) > EPSILON else value
            )
        self._blend_explanation_weights(signal)

    def _learn_from_counterfactual(
        self,
        original: np.ndarray,
        edited: np.ndarray,
        target_label: Any,
    ) -> None:
        per_feature = self.feature_space.per_feature_distance(original, edited)
        changed = {name: amount for name, amount in per_feature.items() if amount > EPSILON}
        if not changed:
            return
        inverse_total = sum(1.0 / (amount + EPSILON) for amount in changed.values())
        target_sign = self.class_sign(target_label)
        signal = np.zeros(self.feature_space.encoded_size, dtype=float)
        for name, amount in changed.items():
            feature_slice = self.feature_space.feature_slice(name)
            direction = edited[feature_slice] - original[feature_slice]
            direction_norm = np.linalg.norm(direction, ord=1)
            if direction_norm <= EPSILON:
                continue
            importance = (1.0 / (amount + EPSILON)) / inverse_total
            signal[feature_slice] = target_sign * importance * direction / direction_norm
        self._blend_explanation_weights(signal)

    def _blend_explanation_weights(self, signal: np.ndarray) -> None:
        norm = np.linalg.norm(signal, ord=1)
        if norm <= EPSILON:
            return
        signal = signal / norm
        self._explanation_weights = (
            (1.0 - self.learning_rate) * self._explanation_weights
            + self.learning_rate * signal
        )
        self._has_explanation_weights = True
