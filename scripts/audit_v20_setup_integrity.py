"""Audit v2.0 experiment setup, generated data, and core derived columns.

Read-only audit. It checks instance restrictions, counterfactual generation
invariants, Qualtrics row parsing, row-level change columns, and current JMP
column preservation/additions.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QUALTRICS = ROOT / "qualtrics"
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
MANIFEST = QUALTRICS / "case-manifest-v1.6.json"
RESULTS = QUALTRICS / "qualtrics_results_v2.0.csv"
SUMMARY = QUALTRICS / "v2.0_results_summary.json"
CURRENT_JMP_ROUNDTRIP = QUALTRICS / "qualtrics_results_v2.0.current.roundtrip.csv"
OLD_JMP_ROUNDTRIP = QUALTRICS / "qualtrics_results_v2.0.before_cf_direction_columns.roundtrip.csv"
RAW = QUALTRICS / "raw_output_v1.9 (2 clusters new clustering).csv"

EXPECTED_TRAINING_IDS = set(range(160100, 160112))
EXPECTED_TESTING_IDS = set(range(160200, 160220))
FEATURES = ["Glucose", "Blood Pressure", "Insulin", "BMI", "Age"]
RAW_FEATURES = ["glucose", "blood_pressure", "insulin", "bmi", "age"]
DISPLAY_TO_RAW = dict(zip(FEATURES, RAW_FEATURES))
VALUE_PATTERN = re.compile(
    r'"(?P<feature>[^"]+)"\s*-\s*'
    r'(?P<before_raw>[^(\n]+)\((?P<before_norm>[-+0-9.eE]+)\)'
    r'(?:\s*->\s*(?P<after_raw>[^(\n]+)\((?P<after_norm>[-+0-9.eE]+)\))?'
)


class Audit:
    def __init__(self) -> None:
        self.pass_count = 0
        self.warnings: list[str] = []
        self.failures: list[str] = []
        self.info: list[str] = []

    def check(self, condition: bool, message: str, detail: str = "") -> None:
        if condition:
            self.pass_count += 1
        else:
            suffix = f" — {detail}" if detail else ""
            self.failures.append(f"{message}{suffix}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.info.append(message)


def load_bundle() -> dict[str, Any]:
    return json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]


def load_manifest() -> list[dict[str, Any]]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]


def normalized_value(value: Any, low_high: list[Any]) -> float:
    low, high = float(low_high[0]), float(low_high[1])
    return min(1.0, max(0.0, (float(value) - low) / (high - low)))


def parse_attribute_text(text: str) -> dict[str, tuple[float, float | None]]:
    out: dict[str, tuple[float, float | None]] = {}
    for match in VALUE_PATTERN.finditer(str(text)):
        before = float(match.group("before_norm"))
        after = match.group("after_norm")
        out[match.group("feature")] = (before, float(after) if after is not None else None)
    return out


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def audit_bundle_and_manifest(audit: Audit) -> tuple[dict[int, dict[str, Any]], dict[int, int]]:
    data = load_bundle()
    manifest = load_manifest()
    cases = data["training_pool"] + data["test_pool"]

    case_key = lambda c: (c["experimental_phase"], c["source_split"], int(c["source_instance_id"]))
    cases_by_source = {case_key(case): case for case in cases}
    manifest_by_qualtrics = {int(row["qualtrics_instance_id"]): row for row in manifest}
    cases_by_qualtrics = {
        int(row["qualtrics_instance_id"]): cases_by_source[
            (row["experimental_phase"], row["source_split"], int(row["source_instance_id"]))
        ]
        for row in manifest
    }
    remap = {
        qualtrics_id: int(case["instance_id"])
        for qualtrics_id, case in cases_by_qualtrics.items()
    }

    audit.check(len(manifest) == 32, "manifest has exactly 32 cases", str(len(manifest)))
    audit.check(len(cases) == 32, "bundle has exactly 32 cases", str(len(cases)))
    audit.check(len(set(remap.values())) == 32, "manifest→analysis instance remap is one-to-one")
    audit.check(
        {int(c["instance_id"]) for c in data["training_pool"]} == EXPECTED_TRAINING_IDS,
        "bundle training instance IDs are v2.0 latest 160100–160111",
    )
    audit.check(
        {int(c["instance_id"]) for c in data["test_pool"]} == EXPECTED_TESTING_IDS,
        "bundle testing instance IDs are v2.0 latest 160200–160219",
    )
    audit.check(
        Counter(int(c["prediction"]["value"]) for c in data["test_pool"]) == {0: 10, 1: 10},
        "test pool has balanced original predictions",
    )

    bad_feature_schema = [
        c["instance_id"]
        for c in cases
        if c["feature_names"] != FEATURES or c["raw_feature_names"] != RAW_FEATURES
    ]
    audit.check(not bad_feature_schema, "all cases use expected five diabetes features", str(bad_feature_schema[:5]))

    cf_changed_counts = []
    cf_nonflip = []
    cf_not_same_selected = []
    attribution_count_bad = []
    for case in cases:
        original = case["raw_feature_values"]
        counterfactual = case["counterfactual"]["raw_feature_values"]
        changed = [
            index
            for index, (before, after) in enumerate(zip(original, counterfactual))
            if abs(float(before) - float(after)) > 1e-9
        ]
        cf_changed_counts.append(len(changed))
        if int(case["counterfactual"]["prediction"]["value"]) == int(case["prediction"]["value"]):
            cf_nonflip.append(case["instance_id"])
        selected = {
            DISPLAY_TO_RAW.get(name, name)
            for name in case["counterfactual"].get("selected_feature_names", [])
        }
        raw_changed = {RAW_FEATURES[index] for index in changed}
        if selected and selected != raw_changed:
            cf_not_same_selected.append((case["instance_id"], sorted(selected), sorted(raw_changed)))
        if len(case["attribution"].get("shown_feature_indices", [])) != 2:
            attribution_count_bad.append(case["instance_id"])

    audit.check(set(cf_changed_counts) == {2}, "all bundled CFs change exactly two features", str(Counter(cf_changed_counts)))
    audit.check(not cf_nonflip, "all bundled CFs flip model prediction", str(cf_nonflip[:5]))
    audit.check(not cf_not_same_selected, "CF selected features match actually changed features", str(cf_not_same_selected[:3]))
    audit.check(not attribution_count_bad, "all attributions show exactly two features", str(attribution_count_bad[:5]))

    pair_counts = Counter(case["feature_pair_key"] for case in cases)
    audit.note(f"bundle feature_pair_key counts: {dict(sorted(pair_counts.items()))}")
    manifest_phase_ids = {
        phase: sorted(int(row["qualtrics_instance_id"]) for row in manifest if row["experimental_phase"] == phase)
        for phase in ("training", "testing")
    }
    audit.note(f"browser/Qualtrics manifest ids by phase: {manifest_phase_ids}")
    audit.note(f"analysis remapped ids: training={sorted(EXPECTED_TRAINING_IDS)}, testing={sorted(EXPECTED_TESTING_IDS)}")
    return cases_by_qualtrics, remap


def audit_results(audit: Audit) -> pd.DataFrame:
    results = pd.read_csv(RESULTS, dtype=str, keep_default_na=False)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    audit.check(len(results) == 1984, "v2.0 CSV has 1,984 rows", str(len(results)))
    audit.check(len(results.columns) == 63, "v2.0 CSV has 63 columns after cognitive model merge", str(len(results.columns)))
    participant_counts = results.groupby("participant").size()
    audit.check(len(participant_counts) == 62, "v2.0 CSV has 62 participants", str(len(participant_counts)))
    audit.check(set(participant_counts.unique()) == {32}, "every participant has 32 rows", str(Counter(participant_counts)))
    audit.check(
        Counter(results["phase"]) == {"testing": 1240, "training": 744},
        "phase row counts are 744 training / 1240 testing",
        str(Counter(results["phase"])),
    )
    xai_by_participant = results.groupby("participant")["xai"].nunique()
    audit.check(xai_by_participant.max() == 1, "each participant has exactly one XAI condition")
    xai_counts = results.drop_duplicates("participant")["xai"].value_counts().to_dict()
    audit.check(xai_counts == {"counterfactual": 27, "none": 19, "attribution": 16}, "XAI participant counts are expected", str(xai_counts))

    training_ids = set(map(int, results.loc[results["phase"].eq("training"), "instance id"]))
    testing_ids = set(map(int, results.loc[results["phase"].eq("testing"), "instance id"]))
    audit.check(training_ids == EXPECTED_TRAINING_IDS, "CSV training rows restricted to latest instance IDs")
    audit.check(testing_ids == EXPECTED_TESTING_IDS, "CSV testing rows restricted to latest instance IDs")
    audit.check(
        summary.get("training_instance_ids") == sorted(EXPECTED_TRAINING_IDS)
        and summary.get("testing_instance_ids") == sorted(EXPECTED_TESTING_IDS),
        "v2.0 summary lists latest instance IDs",
    )

    # Training/test field missingness.
    training = results["phase"].eq("training")
    testing = results["phase"].eq("testing")
    metric_columns = [
        "sparsity",
        "proximity",
        "plausibility",
        "plausibility (subset)",
        "actionability (0/1)",
        "boundary distance original",
        "boundary distance new",
    ]
    training_nonblank = int(results.loc[training, metric_columns].astype(str).ne("").sum().sum())
    testing_blank = int(results.loc[testing, metric_columns].astype(str).eq("").sum().sum())
    audit.check(training_nonblank == 0, "training rows leave testing metrics blank", str(training_nonblank))
    audit.check(testing_blank == 0, "testing rows have all core metrics populated", str(testing_blank))

    # Core change columns from attribute text.
    mismatch_examples = []
    raw_changed_norm_zero_examples = []
    for idx, row in results.loc[testing].iterrows():
        parsed = parse_attribute_text(row["attribute values before and after"])
        if set(parsed) != set(FEATURES):
            mismatch_examples.append((idx, "parse_features", sorted(parsed)))
            continue
        for feature_index, feature in enumerate(FEATURES, start=1):
            before, after = parsed[feature]
            expected_changed = 0 if after is None else int(abs(after - before) > 1e-9)
            expected_change = 0.0 if after is None else after - before
            actual_changed = int(float(row[f"x_{feature_index}_changed"]))
            actual_change = float(row[f"x_{feature_index}_change"])
            if actual_changed == 1 and expected_changed == 0 and abs(actual_change) <= 1e-12:
                raw_changed_norm_zero_examples.append((idx, feature))
                continue
            if actual_changed != expected_changed or abs(actual_change - expected_change) > 1e-8:
                mismatch_examples.append(
                    (idx, feature, expected_changed, actual_changed, expected_change, actual_change)
                )
                break
    audit.check(not mismatch_examples, "x_i_change/x_i_changed match parsed before-after text", str(mismatch_examples[:5]))
    if raw_changed_norm_zero_examples:
        audit.warn(
            "Some raw edits are recorded as changed but have zero clipped normalized change; "
            f"examples={raw_changed_norm_zero_examples[:5]}"
        )

    changed_count = sum(numeric(results[f"x_{i}_changed"]) for i in range(1, 6))
    sparsity_expected = 1 - changed_count / 5
    sparsity_mismatch = int(((numeric(results["sparsity"]) - sparsity_expected).abs() > 1e-9).loc[testing].sum())
    audit.check(sparsity_mismatch == 0, "sparsity equals 1 - changed_feature_count/5", str(sparsity_mismatch))

    proximity_expected = sum(numeric(results[f"x_{i}_change"]).abs() for i in range(1, 6))
    proximity_mismatch = int(((numeric(results["proximity"]) - proximity_expected).abs() > 1e-8).loc[testing].sum())
    audit.check(proximity_mismatch == 0, "proximity equals normalized L1 edit distance", str(proximity_mismatch))

    action_expected = numeric(results["x_5_changed"]).eq(0).astype(int)
    action_mismatch = int((numeric(results["actionability (0/1)"]) != action_expected).loc[testing].sum())
    audit.check(action_mismatch == 0, "actionability is 1 iff Age/x_5 was not changed", str(action_mismatch))

    target_opposite = results.loc[testing, ["original label", "target label"]].apply(
        lambda r: set(r) == {"Diabetes", "No Diabetes"}, axis=1
    )
    audit.check(bool(target_opposite.all()), "testing target label is opposite of original label")

    success_expected = results["counterfactual label"].eq(results["target label"]).astype(int)
    success_mismatch = int((numeric(results["successful counterfactual (0/1)"]) != success_expected).loc[testing].sum())
    audit.check(success_mismatch == 0, "success flag matches counterfactual label == target label", str(success_mismatch))

    delta_expected = numeric(results["confidence for target label counterfactual"]) - numeric(results["confidence for target label original"])
    delta_mismatch = int(((numeric(results["delta confidence of target label"]) - delta_expected).abs() > 1e-8).loc[testing].sum())
    audit.check(delta_mismatch == 0, "delta confidence equals edited target confidence - original target confidence", str(delta_mismatch))

    move_expected = delta_expected.gt(0).astype(int)
    move_mismatch = int((numeric(results["move towards target (0/1)"]) != move_expected).loc[testing].sum())
    audit.check(move_mismatch == 0, "move-towards-target flag equals positive target-confidence delta", str(move_mismatch))

    bd_delta_expected = numeric(results["boundary distance new"]) - numeric(results["boundary distance original"])
    bd_delta_mismatch = int(((numeric(results["boundary distance change (new - original)"]) - bd_delta_expected).abs() > 1e-8).loc[testing].sum())
    audit.check(bd_delta_mismatch == 0, "boundary distance change equals new - original", str(bd_delta_mismatch))

    # Existing direction columns are raw-direction based. Age is extrapolated.
    for feature_index, feature in enumerate(FEATURES, start=1):
        column = f"x_{feature_index}_changed_in_training_cf_direction"
        if column not in results.columns:
            continue
        bad_direction = []
        for idx, row in results.loc[testing].iterrows():
            value = str(row[column]).strip()
            changed = int(float(row[f"x_{feature_index}_changed"]))
            if not changed:
                if value != "":
                    bad_direction.append((idx, "expected blank", value))
                continue
            parsed = parse_attribute_text(row["attribute values before and after"])
            # recover raw sign from text with a direct second parse
            raw_sign = 0
            for match in VALUE_PATTERN.finditer(row["attribute values before and after"]):
                if match.group("feature") == feature and match.group("after_raw") is not None:
                    delta = float(match.group("after_raw").strip()) - float(match.group("before_raw").strip())
                    raw_sign = 1 if delta > 0 else -1 if delta < 0 else 0
                    break
            target_sign = 1 if row["target label"] == "Diabetes" else -1
            expected = "1" if raw_sign == target_sign else "0"
            if value not in {"0", "1"} or value != expected:
                bad_direction.append((idx, expected, value))
        audit.check(not bad_direction, f"{column} matches raw target-specific direction rule", str(bad_direction[:5]))

    return results


def audit_raw_logs(
    audit: Audit,
    cases_by_qualtrics: dict[int, dict[str, Any]],
    remap: dict[int, int],
    results: pd.DataFrame,
) -> None:
    with RAW.open(encoding="utf-8-sig", newline="") as source:
        raw_rows = list(csv.reader(source))
    headers = raw_rows[0]
    records = [dict(zip(headers, row)) for row in raw_rows[3:]]

    complete_latest = []
    stale_or_bad = 0
    for row in records:
        try:
            training_logs = json.loads(row.get("training_log_json") or "[]")
            testing_logs = json.loads(row.get("testing_log_json") or "[]")
        except json.JSONDecodeError:
            continue
        if row.get("Finished") != "1" or len(training_logs) != 12 or len(testing_logs) != 20:
            continue
        training_ids = [int(log["instanceId"]) for log in training_logs]
        testing_ids = [int(log["instanceId"]) for log in testing_logs]
        if set(training_ids + testing_ids) != set(remap) or len(set(training_ids)) != 12 or len(set(testing_ids)) != 20:
            stale_or_bad += 1
            continue
        ok = True
        for log in training_logs:
            case = cases_by_qualtrics[int(log["instanceId"])]
            ok = ok and int(log["correctPrediction"]) == int(case["prediction"]["value"])
        for log in testing_logs:
            case = cases_by_qualtrics[int(log["instanceId"])]
            ok = ok and int(log["originalPrediction"]["value"]) == int(case["prediction"]["value"])
            final_values = log.get("changedRawFeatureValues") or {}
            changes = {int(c["attributeIndex"]): c for c in log.get("changes", [])}
            for index, (feature, expected) in enumerate(zip(case["raw_feature_names"], case["raw_feature_values"])):
                actual = changes[index]["originalValue"] if index in changes else final_values.get(feature)
                if actual is None or not math.isclose(float(actual), float(expected), abs_tol=1e-9, rel_tol=0.0):
                    ok = False
                    break
        if ok:
            complete_latest.append(row["ResponseId"])
        else:
            stale_or_bad += 1

    audit.check(len(set(complete_latest)) >= 62, "raw output contains at least the 62 latest complete participants", str(len(set(complete_latest))))
    audit.note(f"raw latest complete participant candidates={len(set(complete_latest))}; stale_or_bad_complete_candidates={stale_or_bad}")

    result_testing = {
        (row["participant"], int(row["instance id"])): row
        for _, row in results.loc[results["phase"].eq("testing")].iterrows()
    }
    raw_compare_mismatches = []
    result_participants = set(results["participant"])
    for row in records:
        participant = row.get("ResponseId", "")
        if participant not in result_participants:
            continue
        testing_logs = json.loads(row.get("testing_log_json") or "[]")
        for log in testing_logs:
            raw_id = int(log["instanceId"])
            if raw_id not in remap:
                continue
            instance_id = remap[raw_id]
            result_row = result_testing.get((participant, instance_id))
            if result_row is None:
                raw_compare_mismatches.append((participant, instance_id, "missing_result_row"))
                continue
            case = cases_by_qualtrics[raw_id]
            final_values = log.get("changedRawFeatureValues") or {}
            original_values = case["raw_feature_values"]
            edited_values = [final_values[name] for name in RAW_FEATURES]
            for feature_index, (before, after, feature_range) in enumerate(
                zip(original_values, edited_values, case["raw_feature_ranges"]), start=1
            ):
                expected_change = normalized_value(after, feature_range) - normalized_value(before, feature_range)
                expected_changed = int(abs(float(after) - float(before)) > 1e-9)
                actual_change = float(result_row[f"x_{feature_index}_change"])
                actual_changed = int(float(result_row[f"x_{feature_index}_changed"]))
                if abs(actual_change - expected_change) > 1e-8 or actual_changed != expected_changed:
                    raw_compare_mismatches.append(
                        (
                            participant,
                            instance_id,
                            f"x_{feature_index}",
                            expected_change,
                            actual_change,
                            expected_changed,
                            actual_changed,
                        )
                    )
                    break
    audit.check(
        not raw_compare_mismatches,
        "converted testing change columns match raw Qualtrics logs after v2.0 remap",
        str(raw_compare_mismatches[:5]),
    )


def audit_current_jmp(audit: Audit) -> None:
    if not CURRENT_JMP_ROUNDTRIP.exists() or not OLD_JMP_ROUNDTRIP.exists():
        audit.warn("JMP roundtrip CSVs not found; skip current JMP structural audit")
        return
    old = pd.read_csv(OLD_JMP_ROUNDTRIP, dtype=str, keep_default_na=False, nrows=1)
    current = pd.read_csv(CURRENT_JMP_ROUNDTRIP, dtype=str, keep_default_na=False, nrows=1)
    old_cols = list(old.columns)
    current_cols = list(current.columns)
    added = [col for col in current_cols if col not in old_cols]
    removed = [col for col in old_cols if col not in current_cols]
    order_changed = [
        col for col in old_cols if col in current_cols and old_cols.index(col) != current_cols.index(col)
    ]
    audit.check(not removed, "current JMP removed no original columns", str(removed))
    audit.check(not order_changed, "current JMP preserves original column order", str(order_changed[:5]))
    audit.note(f"current JMP columns={len(current_cols)}; original backup columns={len(old_cols)}; appended columns={len(added)}")


def main() -> None:
    audit = Audit()
    cases_by_qualtrics, remap = audit_bundle_and_manifest(audit)
    results = audit_results(audit)
    audit_raw_logs(audit, cases_by_qualtrics, remap, results)
    audit_current_jmp(audit)

    report = {
        "status": "fail" if audit.failures else ("warn" if audit.warnings else "pass"),
        "passes": audit.pass_count,
        "failures": audit.failures,
        "warnings": audit.warnings,
        "info": audit.info,
    }
    print(json.dumps(report, indent=2))
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
