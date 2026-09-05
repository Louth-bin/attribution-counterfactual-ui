from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .model import EPSILON, MentalModel, MinimalCognitiveModel
from .perfect_memory import PerfectMemoryCognitiveModel


@dataclass(frozen=True)
class EditProposal:
    strategy: str
    mental_model: MentalModel
    original: dict[str, Any]
    edited: dict[str, Any]
    changed_features: tuple[str, ...]
    predicted_before: Any
    predicted_after: Any
    target_label: Any

    @property
    def flips_mental_prediction(self) -> bool:
        return self.predicted_after == self.target_label


class CounterfactualStrategies:
    """Counterfactual editing strategies supported by the three mental models."""

    SUPPORT = {
        "change_most_contributing_attributes": {
            MentalModel.EXEMPLAR: "partial",
            MentalModel.ATTRIBUTION: "strong",
            MentalModel.PROTOTYPE: "partial",
        },
        "change_most_influential_attributes_to_flip": {
            MentalModel.EXEMPLAR: "weak",
            MentalModel.ATTRIBUTION: "strong",
            MentalModel.PROTOTYPE: "partial",
        },
        "change_most_influential_attributes_by_fixed_amount": {
            MentalModel.EXEMPLAR: "weak",
            MentalModel.ATTRIBUTION: "strong",
            MentalModel.PROTOTYPE: "partial",
        },
        "set_attributes_to_target_profile_values": {
            MentalModel.EXEMPLAR: "weak",
            MentalModel.ATTRIBUTION: "weak",
            MentalModel.PROTOTYPE: "strong",
        },
        "change_attributes_not_matching_target_profile": {
            MentalModel.EXEMPLAR: "partial",
            MentalModel.ATTRIBUTION: "weak",
            MentalModel.PROTOTYPE: "strong",
        },
        "change_largest_target_profile_mismatches": {
            MentalModel.EXEMPLAR: "partial",
            MentalModel.ATTRIBUTION: "partial",
            MentalModel.PROTOTYPE: "strong",
        },
        "change_toward_remembered_exemplar": {
            MentalModel.EXEMPLAR: "strong",
            MentalModel.ATTRIBUTION: "weak",
            MentalModel.PROTOTYPE: "weak",
        },
        "copy_changes_from_remembered_example": {
            MentalModel.EXEMPLAR: "shared memory",
            MentalModel.ATTRIBUTION: "shared memory",
            MentalModel.PROTOTYPE: "shared memory",
        },
    }

    def __init__(
        self,
        model: MinimalCognitiveModel | PerfectMemoryCognitiveModel,
        *,
        evaluate_predictions: bool = True,
    ) -> None:
        self.model = model
        self.evaluate_predictions = bool(evaluate_predictions)

    # These public names follow the terminology of the study hypotheses.  The
    # shorter original methods below remain as backward-compatible aliases.

    def change_most_contributing_attributes(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        max_changes: int = 1,
    ) -> EditProposal:
        return self.most_contributing(
            profile, target_label, max_changes=max_changes
        )

    def change_most_influential_attributes_to_flip(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        max_changes: int = 1,
    ) -> EditProposal:
        return self.global_influence(
            profile, target_label, mode="flip", max_changes=max_changes
        )

    def change_most_influential_attributes_by_fixed_amount(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        max_changes: int = 1,
        fixed_amount: float | None = None,
    ) -> EditProposal:
        return self.global_influence(
            profile,
            target_label,
            mode="fixed_delta",
            max_changes=max_changes,
            fixed_amount=fixed_amount,
        )

    def set_attributes_to_target_profile_values(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        max_changes: int | None = None,
    ) -> EditProposal:
        target_profile = self.model.prototype_profile(target_label)
        mismatches = self.model.prototype_mismatches(profile, target_label)
        order = sorted(mismatches, key=mismatches.get, reverse=True)
        limit = len(order) if max_changes is None else max_changes
        edited = dict(profile)
        for name in order[:limit]:
            if mismatches[name] > EPSILON:
                edited[name] = target_profile[name]
        return self._proposal(
            "set_attributes_to_target_profile_values",
            MentalModel.PROTOTYPE,
            profile,
            edited,
            target_label,
        )

    def change_attributes_not_matching_target_profile(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
    ) -> EditProposal:
        return self.prototype_mismatch(profile, target_label, max_changes=None)

    def change_largest_target_profile_mismatches(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        max_changes: int = 1,
    ) -> EditProposal:
        return self.prototype_mismatch(
            profile, target_label, max_changes=max_changes
        )

    def change_toward_remembered_exemplar(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        max_changes: int | None = None,
    ) -> EditProposal:
        return self.remembered_exemplar(
            profile, target_label, max_changes=max_changes
        )

    def copy_changes_from_remembered_example(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        max_changes: int | None = None,
    ) -> EditProposal:
        return self.copy_remembered_change(
            profile, target_label, max_changes=max_changes
        )

    def most_contributing(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        max_changes: int = 1,
    ) -> EditProposal:
        contributions = self.model.attribution_contributions(profile)
        target_sign = self.model.class_sign(target_label)
        order = sorted(
            contributions,
            key=lambda name: -target_sign * contributions[name],
            reverse=True,
        )
        edited = self._attribution_flip(profile, target_label, order, max_changes)
        return self._proposal(
            "change_most_contributing_attributes",
            MentalModel.ATTRIBUTION,
            profile,
            edited,
            target_label,
        )

    def global_influence(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        mode: str = "flip",
        max_changes: int = 1,
        fixed_amount: float | None = None,
    ) -> EditProposal:
        if mode not in {"flip", "fixed_delta", "fixed_value"}:
            raise ValueError("mode must be 'flip', 'fixed_delta', or 'fixed_value'")
        importance = self.model.global_importance()
        order = sorted(importance, key=importance.get, reverse=True)
        if mode == "flip":
            edited = self._attribution_flip(profile, target_label, order, max_changes)
        elif mode == "fixed_value":
            prototype = self.model.prototype_profile(target_label)
            edited = dict(profile)
            for name in order[:max_changes]:
                edited[name] = prototype[name]
        else:
            edited = self._fixed_delta(
                profile, target_label, order[:max_changes], fixed_amount
            )
        return self._proposal(
            {
                "flip": "change_most_influential_attributes_to_flip",
                "fixed_delta": "change_most_influential_attributes_by_fixed_amount",
                "fixed_value": "set_attributes_to_target_profile_values",
            }[mode],
            MentalModel.ATTRIBUTION,
            profile,
            edited,
            target_label,
        )

    def prototype_mismatch(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        max_changes: int | None = None,
    ) -> EditProposal:
        target_profile = self.model.prototype_profile(target_label)
        ranges = self.model.prototype_ranges(target_label)
        mismatch = self.model.prototype_mismatches(profile, target_label)
        order = sorted(mismatch, key=mismatch.get, reverse=True)
        limit = len(order) if max_changes is None else max_changes
        edited = dict(profile)
        for name in order[:limit]:
            if mismatch[name] <= EPSILON:
                continue
            spec = self.model.feature_space.spec(name)
            if spec.kind == "numerical":
                low, high = ranges[name]
                edited[name] = float(np.clip(float(profile[name]), low, high))
            else:
                edited[name] = target_profile[name]
        return self._proposal(
            (
                "change_attributes_not_matching_target_profile"
                if max_changes is None
                else "change_largest_target_profile_mismatches"
            ),
            MentalModel.PROTOTYPE,
            profile,
            edited,
            target_label,
        )

    def remembered_exemplar(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        max_changes: int | None = None,
    ) -> EditProposal:
        encoded = self.model.feature_space.encode(profile)
        if isinstance(self.model, PerfectMemoryCognitiveModel):
            candidates = [
                instance
                for instance in self.model.instances
                if instance.label == target_label
            ]
            if not candidates:
                raise ValueError(f"No remembered exemplar for target class {target_label!r}")
            closest = min(
                candidates,
                key=lambda instance: self.model.feature_space.distance(
                    encoded, instance.encoded
                ),
            )
            exemplar = dict(closest.profile)
            exemplar_encoded = closest.encoded
        else:
            candidates = self.model.memory.retrieve_many(
                kind="instance", cue={"label": target_label}, exact=True
            )
            if not candidates:
                raise ValueError(f"No retrievable exemplar for target class {target_label!r}")
            closest = min(
                candidates,
                key=lambda item: self.model.feature_space.distance(
                    encoded, np.asarray(item.chunk.payload["encoded"], dtype=float)
                ),
            )
            self.model.memory.touch(closest.chunk)
            exemplar = dict(closest.chunk.payload["profile"])
            exemplar_encoded = np.asarray(closest.chunk.payload["encoded"], dtype=float)
        mismatch = self.model.feature_space.per_feature_distance(encoded, exemplar_encoded)
        order = sorted(mismatch, key=mismatch.get, reverse=True)
        limit = len(order) if max_changes is None else max_changes
        edited = dict(profile)
        for name in order[:limit]:
            if mismatch[name] > EPSILON:
                edited[name] = exemplar[name]
        return self._proposal(
            "change_toward_remembered_exemplar",
            MentalModel.EXEMPLAR,
            profile,
            edited,
            target_label,
        )

    def copy_remembered_change(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        *,
        max_changes: int | None = None,
    ) -> EditProposal:
        if isinstance(self.model, PerfectMemoryCognitiveModel):
            candidates = [
                change
                for change in self.model.changes
                if change.target_label == target_label
            ]
            if not candidates:
                raise ValueError(f"No remembered change for target class {target_label!r}")
            current_encoded = self.model.feature_space.encode(profile)
            retrieved = min(
                candidates,
                key=lambda change: self.model.feature_space.distance(
                    current_encoded, change.original_encoded
                ),
            )
            stored_original = retrieved.original_encoded
            stored_edited = retrieved.edited_encoded
            stored_edited_profile = retrieved.edited
        else:
            memory_result = self.model.memory.retrieve_one(
                kind="change", cue={"target_label": target_label}, exact=True
            )
            if memory_result is None:
                raise ValueError(f"No retrievable change for target class {target_label!r}")
            payload = memory_result.chunk.payload
            stored_original = np.asarray(payload["original_encoded"], dtype=float)
            stored_edited = np.asarray(payload["edited_encoded"], dtype=float)
            stored_edited_profile = payload["edited"]
        current = self.model.feature_space.encode(profile)
        amounts = self.model.feature_space.per_feature_distance(stored_original, stored_edited)
        order = sorted(amounts, key=amounts.get, reverse=True)
        limit = len(order) if max_changes is None else max_changes
        copied = current.copy()
        for name in order[:limit]:
            if amounts[name] <= EPSILON:
                continue
            spec = self.model.feature_space.spec(name)
            feature_slice = self.model.feature_space.feature_slice(name)
            if spec.kind == "numerical":
                copied[feature_slice] += stored_edited[feature_slice] - stored_original[feature_slice]
            else:
                copied[feature_slice] = self.model.feature_space.encode(
                    {**profile, name: stored_edited_profile[name]}
                )[feature_slice]
        decoded = self.model.feature_space.decode(copied)
        edited = dict(profile)
        for name in order[:limit]:
            feature_slice = self.model.feature_space.feature_slice(name)
            if amounts[name] > EPSILON and np.any(
                np.abs(copied[feature_slice] - current[feature_slice]) > EPSILON
            ):
                edited[name] = decoded[name]
        return self._proposal(
            "copy_changes_from_remembered_example",
            MentalModel.EXEMPLAR,
            profile,
            edited,
            target_label,
        )

    def _attribution_flip(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        order: list[str],
        max_changes: int,
    ) -> dict[str, Any]:
        if max_changes < 1:
            raise ValueError("max_changes must be at least 1")
        vector = self.model.feature_space.encode(profile)
        original_vector = vector.copy()
        weights = self.model.attribution_weights()
        middle = self.model.attribution_middle()
        target_sign = self.model.class_sign(target_label)
        for name in order[:max_changes]:
            feature = self.model.feature_space.spec(name)
            feature_slice = self.model.feature_space.feature_slice(name)
            score = float(np.dot(weights, vector - middle))
            if target_sign * score > 0:
                break
            if feature.kind == "numerical":
                weight = weights[feature_slice][0]
                if abs(weight) <= EPSILON:
                    continue
                required = vector[feature_slice][0] + (
                    target_sign * EPSILON - score
                ) / weight
                endpoint = 1.0 if target_sign * weight > 0 else 0.0
                if not 0.0 <= required <= 1.0:
                    required = endpoint
                vector[feature_slice] = np.clip(required, 0.0, 1.0)
            else:
                best_vector = vector.copy()
                best_score = target_sign * score
                for category_index in range(len(feature.categories)):
                    candidate = vector.copy()
                    candidate[feature_slice] = 0.0
                    candidate[feature_slice.start + category_index] = 1.0
                    candidate_score = target_sign * float(np.dot(weights, candidate - middle))
                    if candidate_score > best_score:
                        best_vector = candidate
                        best_score = candidate_score
                vector = best_vector
        decoded = self.model.feature_space.decode(vector)
        edited = dict(profile)
        for name in order[:max_changes]:
            feature_slice = self.model.feature_space.feature_slice(name)
            if np.any(
                np.abs(vector[feature_slice] - original_vector[feature_slice]) > EPSILON
            ):
                edited[name] = decoded[name]
        return edited

    def _fixed_delta(
        self,
        profile: Mapping[str, Any],
        target_label: Any,
        feature_names: list[str],
        fixed_amount: float | None,
    ) -> dict[str, Any]:
        if fixed_amount is None:
            fixed_amount = self._remembered_change_amount(target_label)
        if not 0 < fixed_amount <= 1:
            raise ValueError("fixed_amount must be in normalized units (0, 1]")
        vector = self.model.feature_space.encode(profile)
        original_vector = vector.copy()
        weights = self.model.attribution_weights()
        target_sign = self.model.class_sign(target_label)
        prototype = self.model.prototype_profile(target_label)
        for name in feature_names:
            feature = self.model.feature_space.spec(name)
            feature_slice = self.model.feature_space.feature_slice(name)
            if feature.kind == "numerical":
                weight = weights[feature_slice][0]
                if abs(weight) > EPSILON:
                    direction = 1.0 if target_sign * weight > 0 else -1.0
                else:
                    target_vector = self.model.feature_space.encode(prototype)
                    direction = (
                        1.0
                        if target_vector[feature_slice][0] >= vector[feature_slice][0]
                        else -1.0
                    )
                vector[feature_slice] += direction * fixed_amount
            else:
                vector[feature_slice] = self.model.feature_space.encode(
                    {**profile, name: prototype[name]}
                )[feature_slice]
        decoded = self.model.feature_space.decode(vector)
        edited = dict(profile)
        for name in feature_names:
            feature_slice = self.model.feature_space.feature_slice(name)
            if np.any(
                np.abs(vector[feature_slice] - original_vector[feature_slice]) > EPSILON
            ):
                edited[name] = decoded[name]
        return edited

    def _remembered_change_amount(self, target_label: Any) -> float:
        amounts: list[float] = []
        if isinstance(self.model, PerfectMemoryCognitiveModel):
            changes = [
                (change.original_encoded, change.edited_encoded)
                for change in self.model.changes
                if change.target_label == target_label
            ]
        else:
            changes = [
                (
                    np.asarray(item.chunk.payload["original_encoded"], dtype=float),
                    np.asarray(item.chunk.payload["edited_encoded"], dtype=float),
                )
                for item in self.model.memory.retrieve_many(
                    kind="change", cue={"target_label": target_label}, exact=True
                )
            ]
        for before, after in changes:
            amounts.extend(
                amount
                for amount in self.model.feature_space.per_feature_distance(before, after).values()
                if amount > EPSILON
            )
        if not amounts:
            raise ValueError(
                "fixed_amount was omitted and no remembered target-class changes are available"
            )
        return float(np.median(amounts))

    def _proposal(
        self,
        strategy: str,
        mental_model: MentalModel,
        original: Mapping[str, Any],
        edited: Mapping[str, Any],
        target_label: Any,
    ) -> EditProposal:
        original_dict = dict(original)
        edited_dict = dict(edited)
        changed = tuple(
            name
            for name in self.model.feature_space.names
            if original_dict[name] != edited_dict[name]
        )
        return EditProposal(
            strategy=strategy,
            mental_model=mental_model,
            original=original_dict,
            edited=edited_dict,
            changed_features=changed,
            predicted_before=(
                self.model.predict(original_dict, mental_model).label
                if self.evaluate_predictions
                else None
            ),
            predicted_after=(
                self.model.predict(edited_dict, mental_model).label
                if self.evaluate_predictions
                else None
            ),
            target_label=target_label,
        )
