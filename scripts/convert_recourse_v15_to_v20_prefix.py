"""Convert a Recourse v1.5 Qualtrics export to the current v2.0 JMP prefix.

The output schema is the exact 65-column order currently present in
qualtrics_results_v2.0.jmp before ``nearest training instance id``.  The
converter deliberately stops after the nearest-training distance column.

Example:
    python scripts/convert_recourse_v15_to_v20_prefix.py raw_export.csv \
        --output qualtrics/qualtrics_results_v1.5_new_prefix.csv

Use ``--skip-boundary`` for a quick structural conversion or
``--skip-cognitive-fit`` while checking an incomplete/pilot export.  The
corresponding output columns remain present but blank.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calculate_v09_boundary_distance import calculate as calculate_boundaries
from scripts.convert_qualtrics_counterfactuals import convert, write_output
from scripts.fit_parsimonious_weighted_models import observed_trials
from scripts.fit_probabilistic_attribute_selection import fit_family


BUNDLE = ROOT / "analysis" / "diabetes_experiment_bundle_v1.5_discrete_age.json"
ASSEMBLY_REFERENCE = ROOT / "qualtrics" / "qualtrics_results_v2.0.csv"
ASSEMBLER = ROOT / "scripts" / "build_v12_diabetes_clustering_results.py"
DEFAULT_OUTPUT = ROOT / "qualtrics" / "qualtrics_results_v1.5_new_prefix.csv"

# Captured from the current qualtrics_results_v2.0.jmp (2026-09-02), not from
# the older 63-column CSV. These include the user's renamed JMP columns.
JMP_PREFIX_COLUMNS = [
    "participant",
    "domain",
    "xai",
    "phase",
    "original label",
    "target label",
    "trial number",
    "instance id",
    "training response",
    "training correct (0/1)",
    "participant training accuracy",
    "response time (seconds)",
    "log(time)",
    "participant median time",
    "attribute values before and after",
    "explanation",
    "sparsity",
    "dist user CF from orig",
    "user CF label",
    "successful CF",
    "distance to closest minimal counterfactual",
    "confidence for target label original",
    "confidence for target label user CF",
    "delta confidence of target label",
    "user CF increase target label confidence?",
    "x_1_change",
    "x_2_change",
    "x_3_change",
    "x_4_change",
    "x_5_change",
    "x_1_changed",
    "x_2_changed",
    "x_3_changed",
    "x_4_changed",
    "x_5_changed",
    "delta confidence/change amount",
    "confidence of counterfactual - 50%",
    "close to boundary old",
    "distance to boundary",
    "CRT-2 score (0-4)",
    "successful counterfactual (continuous)",
    "actionability (0/1)",
    "plausibility",
    "abs dist original to boundary",
    "abs dist user CF to boundary",
    "sign dist user CF to boundary",
    "boundary dist change (user - original)",
    "plausibility (subset)",
    "participant level correct direction change",
    "% x4_changed > 50",
    "cognitive model v0.1 family",
    "explanation reliance (η)",
    "global relevance reliance (α)",
    "additive change margin (ρ)",
    "exemplar locality (λ)",
    "remembered-change reliance (β)",
    "age treated as actionable (0/1)",
    "mean participant CF dist to orig",
    "Norm(proximity)",
    "x_1_changed_in_training_cf_direction",
    "x_2_changed_in_training_cf_direction",
    "x_3_changed_in_training_cf_direction",
    "x_4_changed_in_training_cf_direction",
    "x_5_changed_in_training_cf_direction",
    "dist to nearest training instance (L1 normalized)",
]

RAW_ALIASES = {
    "ResponseId": ("ResponseId", "Response ID", "response_id", "participant"),
    "Finished": ("Finished", "Complete", "Completed", "is_finished"),
    "xaiType": ("xaiType", "XAI Type", "explanation type", "condition", "xai"),
    "training_log_json": (
        "training_log_json", "training log json", "training log", "training_log"
    ),
    "testing_log_json": (
        "testing_log_json", "testing log json", "testing log", "testing_log"
    ),
    "CRT2_Race": ("CRT2_Race", "CRT2 Race", "QID405"),
    "CRT2_Sheep": ("CRT2_Sheep", "CRT2 Sheep", "QID406"),
    "CRT2_Emily": ("CRT2_Emily", "CRT2 Emily", "QID407"),
    "CRT2_Hole": ("CRT2_Hole", "CRT2 Hole", "QID408"),
}

SOURCE_TO_JMP = {
    "case": "trial number",
    "proximity": "dist user CF from orig",
    "counterfactual label": "user CF label",
    "successful counterfactual (0/1)": "successful CF",
    "confidence for target label counterfactual": "confidence for target label user CF",
    "move towards target (0/1)": "user CF increase target label confidence?",
    "boundary distance original": "abs dist original to boundary",
    "boundary distance new": "abs dist user CF to boundary",
    "boundary distance change (new - original)": "boundary dist change (user - original)",
}

FEATURE_NAMES = ["Glucose", "Blood Pressure", "Insulin", "BMI", "Age"]


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def resolve_raw_columns(header_rows: list[list[str]]) -> dict[str, int]:
    width = len(header_rows[0])
    descriptors = []
    for index in range(width):
        descriptors.append(
            " | ".join(
                row[index] if index < len(row) else "" for row in header_rows[:3]
            )
        )

    resolved: dict[str, int] = {}
    for canonical, aliases in RAW_ALIASES.items():
        alias_keys = {normalized_name(alias) for alias in aliases}
        exact = [
            index for index, header in enumerate(header_rows[0])
            if normalized_name(header) in alias_keys
        ]
        if len(exact) == 1:
            resolved[canonical] = exact[0]
            continue
        contains = [
            index for index, description in enumerate(descriptors)
            if any(key and key in normalized_name(description) for key in alias_keys)
        ]
        if len(contains) == 1:
            resolved[canonical] = contains[0]
            continue
        if canonical.startswith("CRT2_"):
            continue
        raise ValueError(
            f"Could not uniquely map required Qualtrics column {canonical!r}; "
            f"matches={contains or exact}. Available headers were saved in the audit output."
        )
    return resolved


def truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "finished", "complete"}


def normalize_xai(value: Any) -> str:
    result = str(value or "").strip().casefold()
    return "counterfactual" if result == "counterfactuals" else result


def prepare_raw_export(
    input_path: Path, output_path: Path, bundle: dict[str, Any]
) -> dict[str, Any]:
    with input_path.open(encoding="utf-8-sig", newline="") as source:
        raw_rows = list(csv.reader(source))
    if not raw_rows:
        raise ValueError(f"Empty input: {input_path}")
    header_row_count = 3 if len(raw_rows) >= 3 else 1
    header_rows = raw_rows[:header_row_count]
    while len(header_rows) < 3:
        header_rows.append([""] * len(header_rows[0]))
    resolved = resolve_raw_columns(header_rows)

    training = bundle["training_pool"]
    testing = bundle["test_pool"]
    training_order = {int(case["instance_id"]): index + 1 for index, case in enumerate(training)}
    test_by_prediction: dict[int, list[int]] = defaultdict(list)
    for case in testing:
        test_by_prediction[int(case["prediction"]["value"])].append(int(case["instance_id"]))
    test_order = {
        instance_id: index + 1
        for ids in test_by_prediction.values()
        for index, instance_id in enumerate(ids)
    }
    expected_training = set(training_order)
    expected_testing = set(test_order)

    canonical_headers = list(RAW_ALIASES)
    selected: list[dict[str, str]] = []
    excluded = Counter()
    seen_ids: set[str] = set()
    for raw in raw_rows[header_row_count:]:
        if len(raw) < len(header_rows[0]):
            raw = raw + [""] * (len(header_rows[0]) - len(raw))
        values = {
            name: raw[index] if index < len(raw) else ""
            for name, index in resolved.items()
        }
        response_id = values.get("ResponseId", "").strip()
        if not response_id:
            excluded["missing_response_id"] += 1
            continue
        if response_id in seen_ids:
            excluded["duplicate_response_id"] += 1
            continue
        if not truthy(values.get("Finished", "")):
            excluded["unfinished"] += 1
            continue
        try:
            training_logs = json.loads(values.get("training_log_json") or "[]")
            testing_logs = json.loads(values.get("testing_log_json") or "[]")
        except json.JSONDecodeError:
            excluded["invalid_log_json"] += 1
            continue
        if len(training_logs) != 12 or len(testing_logs) != 20:
            excluded["not_12_training_20_testing"] += 1
            continue
        training_ids = {int(log["instanceId"]) for log in training_logs}
        testing_ids = {int(log["instanceId"]) for log in testing_logs}
        if training_ids != expected_training or testing_ids != expected_testing:
            excluded["unexpected_instance_set"] += 1
            continue

        for log in training_logs:
            log["caseNumber"] = training_order[int(log["instanceId"])]
            log["explanation"] = normalize_xai(log.get("explanation") or values.get("xaiType"))
        for log in testing_logs:
            log["caseNumberWithinDirection"] = test_order[int(log["instanceId"])]
            log["explanation"] = normalize_xai(log.get("explanation") or values.get("xaiType"))
        values["xaiType"] = normalize_xai(values.get("xaiType"))
        values["training_log_json"] = json.dumps(training_logs, separators=(",", ":"))
        values["testing_log_json"] = json.dumps(testing_logs, separators=(",", ":"))
        selected.append({header: values.get(header, "") for header in canonical_headers})
        seen_ids.add(response_id)

    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=canonical_headers, lineterminator="\n")
        writer.writeheader()
        writer.writerow({header: header for header in canonical_headers})
        writer.writerow({header: header for header in canonical_headers})
        writer.writerows(selected)

    return {
        "input": str(input_path),
        "raw_columns": header_rows[0],
        "resolved_columns": {
            canonical: header_rows[0][index] for canonical, index in resolved.items()
        },
        "complete_participants": len(selected),
        "excluded": dict(excluded),
        "xai_participants": dict(Counter(row["xaiType"] for row in selected)),
    }


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def scalar(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "" if not math.isfinite(number) else f"{number:.12g}"


def fit_cognitive_models(
    base_rows: list[dict[str, Any]], bundle: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        grouped[str(row["participant"])].append(row)
    case_map = {int(case["instance_id"]): case for case in bundle["test_pool"]}
    training = bundle["training_pool"]
    winners: dict[str, dict[str, Any]] = {}
    fit_rows: list[dict[str, Any]] = []
    for participant, rows in sorted(grouped.items()):
        condition = str(rows[0]["xai"])
        trials = observed_trials(rows, case_map)
        fitted = []
        for family in ("feature contribution", "weighted examples"):
            parameters, insample, cv = fit_family(family, condition, training, trials)
            result = {
                "participant": participant,
                "xai": condition,
                "model family": family,
                **parameters,
                **{f"in-sample {name}": value for name, value in insample.items()},
                **{f"5-fold CV {name}": value for name, value in cv.items()},
            }
            fitted.append(result)
        winner = min(
            fitted,
            key=lambda row: (
                row["5-fold CV selection_nll"], row["5-fold CV amount_mae"]
            ),
        )["model family"]
        for result in fitted:
            result["selected family by CV"] = int(result["model family"] == winner)
            fit_rows.append(result)
            if result["selected family by CV"]:
                winners[participant] = result
    return winners, fit_rows


def taught_directions(bundle: dict[str, Any]) -> dict[tuple[int, str], int]:
    signs: dict[tuple[int, str], list[int]] = defaultdict(list)
    for case in bundle["training_pool"]:
        target = str(case["counterfactual"]["prediction"]["label"])
        for index, (before, after) in enumerate(
            zip(case["raw_feature_values"], case["counterfactual"]["raw_feature_values"])
        ):
            delta = float(after) - float(before)
            if abs(delta) > 1e-9:
                signs[(index, target)].append(1 if delta > 0 else -1)
    result = {}
    for key, values in signs.items():
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        if positive == negative:
            raise ValueError(f"Tied training-counterfactual direction for {key}: {values}")
        result[key] = 1 if positive > negative else -1
    return result


def assemble_jmp_prefix(
    assembled: pd.DataFrame,
    base_rows: list[dict[str, Any]],
    bundle: dict[str, Any],
    cognitive: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    renamed = assembled.rename(columns=SOURCE_TO_JMP).copy()
    result = pd.DataFrame("", index=renamed.index, columns=JMP_PREFIX_COLUMNS)
    for column in result.columns:
        if column in renamed.columns:
            result[column] = renamed[column].astype(str)

    is_testing = result["phase"].eq("testing")
    times = pd.to_numeric(result["response time (seconds)"], errors="coerce")
    result["log(time)"] = np.log(times.where(times > 0)).map(scalar)
    training_medians = (
        pd.DataFrame({"participant": result["participant"], "time": times})
        .loc[result["phase"].eq("training")]
        .groupby("participant")["time"]
        .median()
    )
    result["participant median time"] = result["participant"].map(training_medians).map(scalar)

    validity = pd.to_numeric(result["successful CF"], errors="coerce")
    boundary_new = pd.to_numeric(result["abs dist user CF to boundary"], errors="coerce")
    result.loc[is_testing, "sign dist user CF to boundary"] = (
        boundary_new.loc[is_testing] * (2 * validity.loc[is_testing] - 1)
    ).map(scalar)

    direction_success = pd.to_numeric(
        result["user CF increase target label confidence?"], errors="coerce"
    )
    participant_direction = direction_success.loc[is_testing].groupby(
        result.loc[is_testing, "participant"]
    ).mean()
    result["participant level correct direction change"] = (
        result["participant"].map(participant_direction).map(scalar)
    )

    x4 = pd.to_numeric(result["x_4_changed"], errors="coerce")
    participant_x4 = x4.loc[is_testing].groupby(result.loc[is_testing, "participant"]).mean()
    result["% x4_changed > 50"] = [
        "1" if result.at[index, "xai"] != "none" and participant_x4.get(participant, 0) > 0.5 else "0"
        for index, participant in enumerate(result["participant"])
    ]

    proximity = pd.to_numeric(result["dist user CF from orig"], errors="coerce")
    participant_proximity = proximity.loc[is_testing].groupby(
        result.loc[is_testing, "participant"]
    ).mean()
    result["mean participant CF dist to orig"] = (
        result["participant"].map(participant_proximity).map(scalar)
    )
    for condition, indices in result.loc[is_testing].groupby("xai").groups.items():
        values = proximity.loc[indices]
        low, high = float(values.min()), float(values.max())
        normalized = (values - low) / (high - low) if high > low else values * 0
        result.loc[indices, "Norm(proximity)"] = normalized.map(scalar)

    directions = taught_directions(bundle)
    for index in range(5):
        changed = pd.to_numeric(result[f"x_{index + 1}_changed"], errors="coerce").eq(1)
        change = pd.to_numeric(result[f"x_{index + 1}_change"], errors="coerce")
        output = f"x_{index + 1}_changed_in_training_cf_direction"
        for row_index in result.index[is_testing & changed]:
            taught = directions.get((index, result.at[row_index, "target label"]))
            if taught is not None and abs(change.at[row_index]) > 1e-12:
                result.at[row_index, output] = str(int(np.sign(change.at[row_index]) == taught))

    training_cases = bundle["training_pool"]
    training_matrix = np.asarray(
        [
            [
                (float(value) - float(bounds[0])) / (float(bounds[1]) - float(bounds[0]))
                for value, bounds in zip(case["raw_feature_values"], case["raw_feature_ranges"])
            ]
            for case in training_cases
        ]
    )
    test_cases = {int(case["instance_id"]): case for case in bundle["test_pool"]}
    for row_index in result.index[is_testing]:
        case = test_cases[int(float(result.at[row_index, "instance id"]))]
        original = np.asarray(
            [
                (float(value) - float(bounds[0])) / (float(bounds[1]) - float(bounds[0]))
                for value, bounds in zip(case["raw_feature_values"], case["raw_feature_ranges"])
            ]
        )
        distance = float(np.min(np.sum(np.abs(training_matrix - original), axis=1)))
        result.at[row_index, "dist to nearest training instance (L1 normalized)"] = scalar(distance)

    cognitive_map = {
        "cognitive model v0.1 family": "model family",
        "explanation reliance (η)": "eta",
        "global relevance reliance (α)": "alpha",
        "additive change margin (ρ)": "rho",
        "exemplar locality (λ)": "lambda",
        "remembered-change reliance (β)": "beta",
        "age treated as actionable (0/1)": "age actionable",
    }
    for output, source in cognitive_map.items():
        result[output] = [
            scalar(cognitive.get(participant, {}).get(source, ""))
            for participant in result["participant"]
        ]

    return result


def write_fit_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, nargs="?", help="Downloaded Qualtrics CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--skip-cognitive-fit", action="store_true")
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()

    if args.schema_only:
        print(json.dumps({"columns": JMP_PREFIX_COLUMNS, "count": len(JMP_PREFIX_COLUMNS)}, indent=2))
        return
    if args.raw is None:
        parser.error("raw is required unless --schema-only is used")

    experiment = json.loads(args.bundle.read_text(encoding="utf-8"))
    bundle = experiment["datasets"]["diabetes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = args.output.with_suffix(".audit.json")
    fits_path = args.output.with_suffix(".cognitive_fits.csv")

    with tempfile.TemporaryDirectory(prefix="recourse_v15_") as temporary:
        work = Path(temporary)
        canonical_raw = work / "canonical_raw.csv"
        base_csv = work / "base.csv"
        boundary_csv = work / "boundary.csv"
        boundary_summary = work / "boundary_summary.json"
        assembled_csv = work / "assembled.csv"
        assembly_summary = work / "assembly_summary.json"

        raw_audit = prepare_raw_export(args.raw, canonical_raw, bundle)
        if raw_audit["complete_participants"] == 0:
            raise ValueError("No complete 12-training/20-testing participants were found")

        base_rows, conversion_audit = convert([canonical_raw], args.bundle)
        write_output(base_csv, base_rows)

        if args.skip_boundary:
            base = read_csv(base_csv)
            boundary = base.copy()
            for column in (
                "boundary distance original",
                "boundary distance new",
                "boundary distance change (new - original)",
            ):
                boundary[column] = ""
            boundary.to_csv(boundary_csv, index=False, encoding="utf-8-sig")
        else:
            calculate_boundaries(base_csv, args.bundle, boundary_csv, boundary_summary)

        subprocess.run(
            [
                sys.executable,
                str(ASSEMBLER),
                "--base", str(base_csv),
                "--reference-csv", str(ASSEMBLY_REFERENCE),
                "--raw", str(canonical_raw),
                "--boundary", str(boundary_csv),
                "--bundle", str(args.bundle),
                "--output", str(assembled_csv),
                "--summary", str(assembly_summary),
                "--continuous-validity-column", "successful counterfactual (continuous)",
                "--skip-strategies",
            ],
            check=True,
        )

        cognitive: dict[str, dict[str, Any]] = {}
        fit_rows: list[dict[str, Any]] = []
        if not args.skip_cognitive_fit:
            cognitive, fit_rows = fit_cognitive_models(base_rows, bundle)
            write_fit_rows(fits_path, fit_rows)

        output = assemble_jmp_prefix(read_csv(assembled_csv), base_rows, bundle, cognitive)
        if list(output.columns) != JMP_PREFIX_COLUMNS:
            raise AssertionError("Output does not match the captured JMP prefix schema")
        output.to_csv(args.output, index=False, encoding="utf-8-sig", lineterminator="\n")

    participant_counts = output.groupby("participant").size().to_dict()
    if set(participant_counts.values()) != {32}:
        raise ValueError(f"Expected 32 rows per participant: {Counter(participant_counts.values())}")
    audit = {
        "output": str(args.output),
        "schema_source": "qualtrics/qualtrics_results_v2.0.jmp",
        "stops_before": "nearest training instance id",
        "columns": len(output.columns),
        "rows": len(output),
        "participants": len(participant_counts),
        "participant_row_counts": dict(Counter(participant_counts.values())),
        "phase_rows": dict(Counter(output["phase"])),
        "raw": raw_audit,
        "conversion": conversion_audit,
        "boundary_computed": not args.skip_boundary,
        "cognitive_fit_computed": not args.skip_cognitive_fit,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
