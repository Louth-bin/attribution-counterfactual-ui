"""Calculate unrestricted nearest-counterfactual distance for v0.9 results.

For each testing row, this script measures the normalized L1 distance from
(a) the original profile and (b) the participant-edited profile to the nearest
profile on the other side of the model's 0.5 decision boundary.  All five
features may vary; no two-feature or actionability constraint is imposed.

The browser models are ReLU networks.  A Wachter-style loss directly optimizes
profiles toward the 0.5 output while penalizing summed normalized L1 change.
Multiple deterministic random starts discover relevant local linear regions;
within each discovered region the closest boundary point is solved exactly by
linear programming.  Numerical features are continuous within the experiment's
configured ranges.  Categorical features are enumerated over their configured
levels and use 0/1 mismatch distance.  Distances therefore use the same summed
normalized L1 geometry as the study's existing proximity variables (range 0--5
for five features).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "qualtrics" / "qualtrics_results_v0.9.for-plausibility.csv"
DEFAULT_EXPERIMENT_DATA = REPO_ROOT / "static" / "experiment-data.json"
DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "boundary_distance_v09_values.csv"
DEFAULT_SUMMARY = Path(tempfile.gettempdir()) / "boundary_distance_v09_summary.json"

ORIGINAL_COLUMN = "boundary distance original"
EDITED_COLUMN = "boundary distance new"
DELTA_COLUMN = "boundary distance change (new - original)"
BOUNDARY_LOGIT_EPSILON = 1e-7
NORMALIZED_VALUE_RE = re.compile(
    r"\((-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)"
)


def stable_sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def category_index(value: Any, categories: list[Any]) -> int:
    normalized = str(value).casefold()
    for index, category in enumerate(categories):
        if str(category).casefold() == normalized:
            return index
    raise ValueError(f"Category {value!r} is not in {categories!r}")


def normalized_value(value: Any, feature_type: str, feature_range: list[Any]) -> float:
    if feature_type == "categorical":
        if len(feature_range) <= 1:
            return 0.0
        return category_index(value, feature_range) / (len(feature_range) - 1)
    low, high = float(feature_range[0]), float(feature_range[1])
    if math.isclose(low, high):
        return 0.0
    return min(1.0, max(0.0, (float(value) - low) / (high - low)))


def final_normalized_profile(attribute_text: str) -> np.ndarray:
    lines = [line for line in attribute_text.splitlines() if line.strip()]
    if len(lines) != 5:
        raise ValueError(f"Expected five attribute lines, found {len(lines)}")
    values: list[float] = []
    for line in lines:
        matches = NORMALIZED_VALUE_RE.findall(line)
        if not matches:
            raise ValueError(f"No normalized value found in {line!r}")
        values.append(float(matches[-1]))
    return np.asarray(values, dtype=float)


def model_logit(model: dict[str, Any], normalized_profile: np.ndarray, case: dict[str, Any]) -> float:
    raw_values: list[Any] = []
    for value, feature_type, feature_range in zip(
        normalized_profile, case["feature_types"], case["raw_feature_ranges"]
    ):
        if feature_type == "categorical":
            category_position = int(round(value * (len(feature_range) - 1)))
            raw_values.append(feature_range[category_position])
        else:
            low, high = float(feature_range[0]), float(feature_range[1])
            raw_values.append(low + float(value) * (high - low))

    preprocessing = model["preprocessing"]
    if preprocessing["type"] == "standard-scaler":
        layer_values = [
            (float(value) - float(mean)) / float(scale)
            for value, mean, scale in zip(
                raw_values, preprocessing["mean"], preprocessing["scale"]
            )
        ]
    elif preprocessing["type"] == "column-transformer-v1":
        by_name = dict(zip(model["feature_names"], raw_values))
        numeric = preprocessing.get("numeric", {})
        layer_values = [
            (float(by_name[name]) - float(mean)) / float(scale)
            for name, mean, scale in zip(
                numeric.get("feature_names", []),
                numeric.get("mean", []),
                numeric.get("scale", []),
            )
        ]
        categorical = preprocessing.get("categorical", {})
        for name, categories in zip(
            categorical.get("feature_names", []),
            categorical.get("categories", []),
        ):
            layer_values.extend(
                1.0 if str(category) == str(by_name[name]) else 0.0
                for category in categories
            )
    else:
        raise ValueError(f"Unsupported preprocessing type {preprocessing['type']!r}")

    current = np.asarray(layer_values, dtype=float)
    for layer_index, layer in enumerate(model["layers"]):
        current = current @ np.asarray(layer["weights"], dtype=float) + np.asarray(
            layer["biases"], dtype=float
        )
        if layer_index < len(model["layers"]) - 1:
            current = np.maximum(current, 0.0)
    if current.shape != (1,):
        raise ValueError(f"Expected a scalar model output, found {current.shape}")
    return float(current[0])


@dataclass(frozen=True)
class BoundarySolution:
    distance: float
    profile: tuple[float, ...]
    logit: float
    target_class: int

class WachterBoundarySearch:
    def __init__(
        self,
        model: dict[str, Any],
        case: dict[str, Any],
        *,
        allow_reference_outside_bounds: bool = False,
    ) -> None:
        self.model = model
        self.case = case
        self.allow_reference_outside_bounds = bool(allow_reference_outside_bounds)
        self.feature_types = list(case["feature_types"])
        self.feature_ranges = list(case["raw_feature_ranges"])
        self.numeric_indices = [
            index for index, kind in enumerate(self.feature_types) if kind != "categorical"
        ]
        self.categorical_indices = [
            index for index, kind in enumerate(self.feature_types) if kind == "categorical"
        ]

    def _categorical_cost(
        self, reference: np.ndarray, categorical_values: tuple[int, ...]
    ) -> float:
        cost = 0.0
        for category_position, feature_index in enumerate(self.categorical_indices):
            levels = len(self.feature_ranges[feature_index])
            reference_level = int(round(reference[feature_index] * (levels - 1)))
            cost += (
                abs(categorical_values[category_position] - reference_level) / (levels - 1)
                if levels > 1
                else 0.0
            )
        return cost

    def _with_categories(
        self, profile: np.ndarray, categorical_values: tuple[int, ...]
    ) -> np.ndarray:
        result = profile.copy()
        for category_position, feature_index in enumerate(self.categorical_indices):
            levels = len(self.feature_ranges[feature_index])
            result[feature_index] = (
                categorical_values[category_position] / (levels - 1) if levels > 1 else 0.0
            )
        return result

    def _numeric_logits_and_gradients(
        self,
        numeric_values: np.ndarray,
        categorical_values: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        input_constant, input_coefficients = self._preprocessed_affine(
            categorical_values
        )
        layers = self.model["layers"]
        first_weights = np.asarray(layers[0]["weights"], dtype=float)
        first_biases = np.asarray(layers[0]["biases"], dtype=float)
        second_weights = np.asarray(layers[1]["weights"], dtype=float)
        second_biases = np.asarray(layers[1]["biases"], dtype=float)
        output_weights = np.asarray(layers[2]["weights"], dtype=float)[:, 0]
        output_bias = float(layers[2]["biases"][0])

        model_inputs = input_constant + numeric_values @ input_coefficients
        first_preactivation = model_inputs @ first_weights + first_biases
        first_activation = np.maximum(first_preactivation, 0.0)
        second_preactivation = first_activation @ second_weights + second_biases
        second_activation = np.maximum(second_preactivation, 0.0)
        logits = second_activation @ output_weights + output_bias

        second_gradient = (
            (second_preactivation > 0.0).astype(float) * output_weights[None, :]
        )
        first_gradient = (second_gradient @ second_weights.T) * (
            first_preactivation > 0.0
        )
        input_gradient = first_gradient @ first_weights.T
        numeric_gradient = input_gradient @ input_coefficients.T
        return logits, numeric_gradient

    def _boundary_anchors(
        self,
        reference: np.ndarray,
        target_class: int,
        categorical_values: tuple[int, ...],
    ) -> list[np.ndarray]:
        base = self._with_categories(reference, categorical_values)
        base_logit = model_logit(self.model, base, self.case)
        if int(base_logit > 0.0) == target_class:
            return [base]

        numeric_reference = reference[self.numeric_indices]
        seed_material = (
            np.round(reference, 12).tobytes()
            + bytes(int(value) for value in categorical_values)
        )
        seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], "little", signed=False
        )
        generator = np.random.default_rng(seed)
        starts = np.vstack(
            (
                numeric_reference[None, :],
                generator.random((15, len(self.numeric_indices))),
            )
        )

        optimized_batches: list[np.ndarray] = []
        for loss_weight in (10.0, 100.0, 1000.0, 10000.0):
            positions = starts.copy()
            first_moment = np.zeros_like(positions)
            second_moment = np.zeros_like(positions)
            for step in range(1, 121):
                logits, logit_gradients = self._numeric_logits_and_gradients(
                    positions, categorical_values
                )
                probabilities = 1.0 / (
                    1.0 + np.exp(-np.clip(logits, -700.0, 700.0))
                )
                probability_gradients = (
                    probabilities * (1.0 - probabilities)
                )[:, None] * logit_gradients
                output_loss_gradient = (
                    2.0
                    * loss_weight
                    * (probabilities - 0.5)[:, None]
                    * probability_gradients
                )
                differences = positions - numeric_reference[None, :]
                distance_gradient = differences / np.sqrt(
                    differences * differences + 1e-10
                )
                gradient = output_loss_gradient + distance_gradient
                gradient_norm = np.linalg.norm(gradient, axis=1, keepdims=True)
                gradient = gradient / np.maximum(1.0, gradient_norm / 25.0)

                first_moment = 0.9 * first_moment + 0.1 * gradient
                second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
                corrected_first = first_moment / (1.0 - 0.9**step)
                corrected_second = second_moment / (1.0 - 0.999**step)
                learning_rate = 0.03 * (0.25 + 0.75 * (1.0 - step / 120.0))
                positions -= learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + 1e-8
                )
                positions = np.clip(positions, 0.0, 1.0)
            optimized_batches.append(positions.copy())

        optimized = np.vstack(optimized_batches)
        optimized_logits, _ = self._numeric_logits_and_gradients(
            optimized, categorical_values
        )
        optimized_distances = np.abs(
            optimized - numeric_reference[None, :]
        ).sum(axis=1)
        ranking = np.lexsort((optimized_distances, np.abs(optimized_logits)))

        anchors = [base]
        for candidate_index in ranking[:32]:
            candidate = base.copy()
            candidate[self.numeric_indices] = optimized[int(candidate_index)]
            anchors.append(candidate)

        corners = np.asarray(
            list(product((0.0, 1.0), repeat=len(self.numeric_indices))),
            dtype=float,
        )
        corner_logits, _ = self._numeric_logits_and_gradients(
            corners, categorical_values
        )
        target_corner_indices = np.flatnonzero(
            (corner_logits > 0.0) == bool(target_class)
        )
        corner_anchors: list[tuple[float, np.ndarray, np.ndarray]] = []
        if len(target_corner_indices):
            target_corners = corners[target_corner_indices]
            target_corner_distances = np.abs(
                target_corners - numeric_reference[None, :]
            ).sum(axis=1)
            target_corners = target_corners[
                np.argsort(target_corner_distances)[:8]
            ]
            lower = np.zeros(len(target_corners), dtype=float)
            upper = np.ones(len(target_corners), dtype=float)
            boundary_numeric = target_corners.copy()
            for _ in range(60):
                midpoint = (lower + upper) / 2.0
                candidates = numeric_reference[None, :] + midpoint[:, None] * (
                    target_corners - numeric_reference[None, :]
                )
                candidate_logits, _ = self._numeric_logits_and_gradients(
                    candidates, categorical_values
                )
                successful = (
                    ((candidate_logits > 0.0) == bool(target_class))
                    & (np.abs(candidate_logits) >= BOUNDARY_LOGIT_EPSILON)
                )
                upper[successful] = midpoint[successful]
                boundary_numeric[successful] = candidates[successful]
                lower[~successful] = midpoint[~successful]

            for boundary_values, target_values in zip(
                boundary_numeric, target_corners
            ):
                boundary_candidate = base.copy()
                boundary_candidate[self.numeric_indices] = boundary_values
                target = base.copy()
                target[self.numeric_indices] = target_values
                boundary_distance = float(
                    np.abs(boundary_values - numeric_reference).sum()
                )
                corner_anchors.append(
                    (boundary_distance, boundary_candidate, target)
                )
        for _, boundary_candidate, target in sorted(
            corner_anchors, key=lambda item: item[0]
        )[:4]:
            anchors.extend((boundary_candidate, target))
        return anchors

    def _preprocessed_affine(self, categorical_values: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        numeric_position = {feature_index: position for position, feature_index in enumerate(self.numeric_indices)}
        raw_constant: dict[str, float] = {}
        raw_coefficient: dict[str, np.ndarray] = {}
        categorical_raw: dict[str, Any] = {}

        categorical_position = {feature_index: position for position, feature_index in enumerate(self.categorical_indices)}
        for feature_index, name in enumerate(self.model["feature_names"]):
            if feature_index in numeric_position:
                low, high = [float(value) for value in self.feature_ranges[feature_index]]
                coefficient = np.zeros(len(self.numeric_indices), dtype=float)
                coefficient[numeric_position[feature_index]] = high - low
                raw_constant[name] = low
                raw_coefficient[name] = coefficient
            else:
                level = categorical_values[categorical_position[feature_index]]
                categorical_raw[name] = self.feature_ranges[feature_index][level]

        preprocessing = self.model["preprocessing"]
        constants: list[float] = []
        coefficients: list[np.ndarray] = []
        if preprocessing["type"] == "standard-scaler":
            for feature_index, name in enumerate(self.model["feature_names"]):
                scale = float(preprocessing["scale"][feature_index])
                constants.append((raw_constant[name] - float(preprocessing["mean"][feature_index])) / scale)
                coefficients.append(raw_coefficient[name] / scale)
        elif preprocessing["type"] == "column-transformer-v1":
            numeric = preprocessing.get("numeric", {})
            for name, mean, scale in zip(
                numeric.get("feature_names", []), numeric.get("mean", []), numeric.get("scale", [])
            ):
                constants.append((raw_constant[name] - float(mean)) / float(scale))
                coefficients.append(raw_coefficient[name] / float(scale))
            categorical = preprocessing.get("categorical", {})
            for name, categories in zip(
                categorical.get("feature_names", []), categorical.get("categories", [])
            ):
                for category in categories:
                    constants.append(1.0 if str(category) == str(categorical_raw[name]) else 0.0)
                    coefficients.append(np.zeros(len(self.numeric_indices), dtype=float))
        else:
            raise ValueError(f"Unsupported preprocessing type {preprocessing['type']!r}")
        return np.asarray(constants), np.asarray(coefficients).T

    def _solve_category(
        self,
        reference: np.ndarray,
        target_class: int,
        categorical_values: tuple[int, ...],
    ) -> BoundarySolution | None:
        anchors = self._boundary_anchors(reference, target_class, categorical_values)
        if not anchors:
            return None
        categorical_cost = self._categorical_cost(reference, categorical_values)
        if int(model_logit(self.model, anchors[0], self.case) > 0.0) == target_class:
            feasible_logit = model_logit(self.model, anchors[0], self.case)
            return BoundarySolution(
                categorical_cost,
                tuple(float(value) for value in anchors[0]),
                feasible_logit,
                target_class,
            )

        input_constant, input_coefficients = self._preprocessed_affine(categorical_values)
        layers = self.model["layers"]
        first_weights = np.asarray(layers[0]["weights"], dtype=float)
        first_biases = np.asarray(layers[0]["biases"], dtype=float)
        second_weights = np.asarray(layers[1]["weights"], dtype=float)
        second_biases = np.asarray(layers[1]["biases"], dtype=float)
        output_weights = np.asarray(layers[2]["weights"], dtype=float)[:, 0]
        output_bias = float(layers[2]["biases"][0])

        first_constant = first_biases + input_constant @ first_weights
        first_coefficients = input_coefficients @ first_weights
        n_numeric = len(self.numeric_indices)
        n_first = len(first_biases)
        n_second = len(second_biases)
        z_start = 0
        t_start = z_start + n_numeric
        first_y_start = t_start + n_numeric
        second_y_start = first_y_start + n_first
        variable_count = second_y_start + n_second

        objective = np.zeros(variable_count)
        objective[t_start : t_start + n_numeric] = 1.0
        numeric_bounds = []
        for feature_index in self.numeric_indices:
            reference_value = float(reference[feature_index])
            numeric_bounds.append(
                (
                    min(0.0, reference_value)
                    if self.allow_reference_outside_bounds
                    else 0.0,
                    max(1.0, reference_value)
                    if self.allow_reference_outside_bounds
                    else 1.0,
                )
            )
        base_bounds: list[tuple[float | None, float | None]] = (
            numeric_bounds
            + [(0.0, None)] * n_numeric
            + [(0.0, None)] * n_first
            + [(0.0, None)] * n_second
        )
        solutions: list[BoundarySolution] = []
        seen_patterns: set[tuple[tuple[bool, ...], tuple[bool, ...]]] = set()

        for anchor in anchors:
            anchor_numeric = anchor[self.numeric_indices]
            first_preactivation = first_constant + anchor_numeric @ first_coefficients
            first_activation = np.maximum(first_preactivation, 0.0)
            second_preactivation = second_biases + first_activation @ second_weights
            pattern = (
                tuple(bool(value > 0.0) for value in first_preactivation),
                tuple(bool(value > 0.0) for value in second_preactivation),
            )
            if pattern in seen_patterns:
                continue
            seen_patterns.add(pattern)

            inequalities: list[np.ndarray] = []
            inequality_bounds: list[float] = []
            equalities: list[np.ndarray] = []
            equality_bounds: list[float] = []
            bounds = list(base_bounds)

            for numeric_position, feature_index in enumerate(self.numeric_indices):
                reference_value = float(reference[feature_index])
                row = np.zeros(variable_count)
                row[z_start + numeric_position] = 1.0
                row[t_start + numeric_position] = -1.0
                inequalities.append(row)
                inequality_bounds.append(reference_value)
                row = np.zeros(variable_count)
                row[z_start + numeric_position] = -1.0
                row[t_start + numeric_position] = -1.0
                inequalities.append(row)
                inequality_bounds.append(-reference_value)

            for neuron, active in enumerate(pattern[0]):
                if active:
                    row = np.zeros(variable_count)
                    row[z_start : z_start + n_numeric] = -first_coefficients[:, neuron]
                    row[first_y_start + neuron] = 1.0
                    equalities.append(row)
                    equality_bounds.append(float(first_constant[neuron]))
                else:
                    bounds[first_y_start + neuron] = (0.0, 0.0)
                    row = np.zeros(variable_count)
                    row[z_start : z_start + n_numeric] = first_coefficients[:, neuron]
                    inequalities.append(row)
                    inequality_bounds.append(float(-first_constant[neuron]))

            for neuron, active in enumerate(pattern[1]):
                if active:
                    row = np.zeros(variable_count)
                    row[first_y_start : first_y_start + n_first] = -second_weights[:, neuron]
                    row[second_y_start + neuron] = 1.0
                    equalities.append(row)
                    equality_bounds.append(float(second_biases[neuron]))
                else:
                    bounds[second_y_start + neuron] = (0.0, 0.0)
                    row = np.zeros(variable_count)
                    row[first_y_start : first_y_start + n_first] = second_weights[:, neuron]
                    inequalities.append(row)
                    inequality_bounds.append(float(-second_biases[neuron]))

            row = np.zeros(variable_count)
            row[second_y_start : second_y_start + n_second] = (
                -output_weights if target_class == 1 else output_weights
            )
            inequalities.append(row)
            inequality_bounds.append(
                output_bias - BOUNDARY_LOGIT_EPSILON
                if target_class == 1
                else -BOUNDARY_LOGIT_EPSILON - output_bias
            )

            result = linprog(
                objective,
                A_ub=np.asarray(inequalities),
                b_ub=np.asarray(inequality_bounds),
                A_eq=np.asarray(equalities),
                b_eq=np.asarray(equality_bounds),
                bounds=bounds,
                method="highs",
            )
            if not result.success or result.x is None:
                continue

            profile = self._with_categories(reference, categorical_values)
            profile[self.numeric_indices] = result.x[z_start : z_start + n_numeric]
            logit = model_logit(self.model, profile, self.case)
            if (target_class == 1 and logit < -1e-6) or (
                target_class == 0 and logit > 1e-6
            ):
                raise ValueError(
                    f"Local-region LP/model disagreement: target={target_class}, logit={logit}"
                )
            distance = float(result.fun + categorical_cost)
            direct_distance = sum(abs(float(a) - float(b)) for a, b in zip(reference, profile))
            if not math.isclose(distance, direct_distance, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(
                    f"Distance mismatch: objective={distance}, direct={direct_distance}"
                )
            solutions.append(
                BoundarySolution(
                    distance,
                    tuple(float(value) for value in profile),
                    logit,
                    target_class,
                )
            )

        return min(solutions, key=lambda solution: solution.distance) if solutions else None

    def solve(self, reference: np.ndarray) -> BoundarySolution:
        current_logit = model_logit(self.model, reference, self.case)
        current_class = int(current_logit > 0.0)
        target_class = 1 - current_class
        category_options = [range(len(self.feature_ranges[index])) for index in self.categorical_indices]
        combinations = list(product(*category_options)) if category_options else [tuple()]
        solutions = [
            solution
            for combination in combinations
            if (solution := self._solve_category(reference, target_class, combination)) is not None
        ]
        if not solutions:
            raise RuntimeError("No unrestricted counterfactual was found")
        return min(solutions, key=lambda solution: solution.distance)


def case_index(experiment_data: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for domain, dataset in experiment_data["datasets"].items():
        for case in dataset["test_pool"]:
            key = (domain, int(case["instance_id"]))
            if key in index:
                raise ValueError(f"Duplicate test case {key}")
            index[key] = case
    return index


def cache_key(domain: str, profile: np.ndarray) -> tuple[str, tuple[float, ...]]:
    return domain, tuple(round(float(value), 12) for value in profile)


def calculate(
    input_path: Path,
    experiment_data_path: Path,
    output_path: Path,
    summary_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    experiment = json.loads(experiment_data_path.read_text(encoding="utf-8"))
    cases = case_index(experiment)
    solvers = {}
    for domain, dataset in experiment["datasets"].items():
        representative_case = dataset["test_pool"][0]
        solvers[domain] = WachterBoundarySearch(
            dataset["browser_model"],
            representative_case,
        )
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    output_rows: list[dict[str, str]] = []
    cache: dict[tuple[str, tuple[float, ...]], BoundarySolution] = {}
    testing_processed = 0
    domain_values: dict[str, dict[str, list[float]]] = {
        domain: {"original": [], "new": [], "delta": []} for domain in solvers
    }

    for row_number, row in enumerate(rows, start=1):
        domain = row["domain"].strip().casefold()
        phase = row["phase"].strip().casefold()
        original_text = edited_text = delta_text = ""
        if phase == "testing" and (limit is None or testing_processed < limit):
            instance_id = int(row["instance id"])
            case = cases[(domain, instance_id)]
            original = np.asarray(
                [
                    normalized_value(value, feature_type, feature_range)
                    for value, feature_type, feature_range in zip(
                        case["raw_feature_values"], case["feature_types"], case["raw_feature_ranges"]
                    )
                ],
                dtype=float,
            )
            edited = final_normalized_profile(row["attribute values before and after"])
            for profile in (original, edited):
                key = cache_key(domain, profile)
                if key not in cache:
                    cache[key] = solvers[domain].solve(profile)
            original_distance = cache[cache_key(domain, original)].distance
            edited_distance = cache[cache_key(domain, edited)].distance
            delta = edited_distance - original_distance
            original_text = f"{original_distance:.12g}"
            edited_text = f"{edited_distance:.12g}"
            delta_text = f"{delta:.12g}"
            domain_values[domain]["original"].append(original_distance)
            domain_values[domain]["new"].append(edited_distance)
            domain_values[domain]["delta"].append(delta)
            testing_processed += 1
            if testing_processed % 100 == 0:
                print(
                    f"scored_testing_rows={testing_processed} unique_profiles={len(cache)}",
                    flush=True,
                )

        output_rows.append(
            {
                "row_number": str(row_number),
                "participant": row["participant"],
                "domain": domain,
                "phase": phase,
                ORIGINAL_COLUMN: original_text,
                EDITED_COLUMN: edited_text,
                DELTA_COLUMN: delta_text,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    def summarize(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
        array = np.asarray(values)
        return {
            "count": len(values),
            "min": float(array.min()),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "max": float(array.max()),
        }

    summary = {
        "definition": "minimum summed range-normalized L1 distance to an opposite-prediction profile",
        "boundary": "browser-model probability 0.5 (logit 0)",
        "feature_constraint": "all five features free",
        "actionability_constraint": "none",
        "numeric_features": "continuous within configured experiment ranges",
        "categorical_features": "enumerated configured levels with normalized mismatch distance",
        "solver": "Wachter-style multi-start optimization with range-normalized L1; exact LP within each discovered ReLU region",
        "input_rows": len(rows),
        "testing_scored": testing_processed,
        "unique_profiles_solved": len(cache),
        "by_domain": {
            domain: {measure: summarize(values) for measure, values in measures.items()}
            for domain, measures in domain_values.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--experiment-data", type=Path, default=DEFAULT_EXPERIMENT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    summary = calculate(args.input, args.experiment_data, args.output, args.summary, args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
