"""Build v2.0 from every complete participant using the latest instances.

The v1.9 cohort is preserved and newly collected complete responses are
converted into the same 56-column schema before the two cohorts are merged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from calculate_v09_boundary_distance import calculate as calculate_boundaries
from convert_qualtrics_counterfactuals import convert, write_output


ROOT = Path(__file__).resolve().parents[1]
QUALTRICS = ROOT / "qualtrics"
RAW = QUALTRICS / "raw_output_v1.9 (2 clusters new clustering).csv"
V19 = QUALTRICS / "qualtrics_results_v1.9.csv"
BUNDLE = ROOT / "analysis" / "diabetes-experiment-bundle-v1.6.json"
MANIFEST = QUALTRICS / "case-manifest-v1.6.json"
OUTPUT = QUALTRICS / "qualtrics_results_v2.0.csv"
SUMMARY = QUALTRICS / "v2.0_results_summary.json"
EXPECTED_TRAINING_IDS = set(range(160100, 160112))
EXPECTED_TESTING_IDS = set(range(160200, 160220))


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def existing_participants() -> set[str]:
    _, rows = read_rows(V19)
    return {row["participant"] for row in rows}


def latest_cases_by_qualtrics_id() -> dict[int, dict[str, object]]:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))["datasets"]["diabetes"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    cases = list(bundle["training_pool"]) + list(bundle["test_pool"])
    cases_by_source = {
        (
            case["experimental_phase"],
            case["source_split"],
            int(case["source_instance_id"]),
        ): case
        for case in cases
    }
    cases_by_qualtrics_id = {
        int(case["qualtrics_instance_id"]): cases_by_source[
            (
                case["experimental_phase"],
                case["source_split"],
                int(case["source_instance_id"]),
            )
        ]
        for case in manifest
    }
    if len(cases_by_qualtrics_id) != 32:
        raise RuntimeError(
            f"Expected 32 latest Qualtrics cases, found {len(cases_by_qualtrics_id)}"
        )
    return cases_by_qualtrics_id


def case_id_remap() -> dict[int, int]:
    cases_by_qualtrics_id = latest_cases_by_qualtrics_id()
    mapping = {
        qualtrics_id: int(case["instance_id"])
        for qualtrics_id, case in cases_by_qualtrics_id.items()
    }
    if len(set(mapping.values())) != 32:
        raise RuntimeError("Latest analysis instance IDs are not one-to-one")
    return mapping


def has_latest_profiles(row: dict[str, str]) -> bool:
    """Verify the profiles actually shown, since Qualtrics IDs were reused."""
    cases_by_qualtrics_id = latest_cases_by_qualtrics_id()
    expected_training_ids = {
        qualtrics_id
        for qualtrics_id, case in cases_by_qualtrics_id.items()
        if case["experimental_phase"] == "training"
    }
    expected_testing_ids = {
        qualtrics_id
        for qualtrics_id, case in cases_by_qualtrics_id.items()
        if case["experimental_phase"] == "testing"
    }
    training_logs = json.loads(row["training_log_json"])
    testing_logs = json.loads(row["testing_log_json"])
    training_ids = [int(log["instanceId"]) for log in training_logs]
    testing_ids = [int(log["instanceId"]) for log in testing_logs]
    if (
        set(training_ids) != expected_training_ids
        or len(training_ids) != len(set(training_ids))
        or set(testing_ids) != expected_testing_ids
        or len(testing_ids) != len(set(testing_ids))
    ):
        return False

    for log in training_logs:
        case = cases_by_qualtrics_id[int(log["instanceId"])]
        if int(log["correctPrediction"]) != int(case["prediction"]["value"]):
            return False

    for log in testing_logs:
        case = cases_by_qualtrics_id[int(log["instanceId"])]
        if int(log["originalPrediction"]["value"]) != int(
            case["prediction"]["value"]
        ):
            return False
        final_values = log.get("changedRawFeatureValues") or {}
        changes = {
            int(change["attributeIndex"]): change
            for change in (log.get("changes") or [])
        }
        for index, (feature_name, expected_value) in enumerate(
            zip(case["raw_feature_names"], case["raw_feature_values"])
        ):
            actual_value = (
                changes[index]["originalValue"]
                if index in changes
                else final_values.get(feature_name)
            )
            if actual_value is None or not math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                return False
    return True


def isolate_and_remap(output: Path) -> dict[str, object]:
    with RAW.open(encoding="utf-8-sig", newline="") as source:
        raw_rows = list(csv.reader(source))
    headers = raw_rows[0]
    records = [dict(zip(headers, row)) for row in raw_rows[3:]]
    old = existing_participants()

    complete_candidates: list[dict[str, str]] = []
    for row in records:
        try:
            training_count = len(json.loads(row.get("training_log_json") or "[]"))
            testing_count = len(json.loads(row.get("testing_log_json") or "[]"))
        except json.JSONDecodeError:
            continue
        if (
            row.get("Finished") == "1"
            and row.get("ResponseId")
            and training_count == 12
            and testing_count == 20
        ):
            complete_candidates.append(row)

    latest_complete = [row for row in complete_candidates if has_latest_profiles(row)]
    latest_participant_ids = {row["ResponseId"] for row in latest_complete}
    missing_preserved = old - latest_participant_ids
    if missing_preserved:
        raise RuntimeError(
            "Preserved v1.9 participants fail the latest-profile audit: "
            f"{sorted(missing_preserved)}"
        )
    complete = [row for row in latest_complete if row["ResponseId"] not in old]

    mapping = case_id_remap()
    expected_source_ids = set(mapping)
    for row in complete:
        participant_ids: set[int] = set()
        for field in ("training_log_json", "testing_log_json"):
            logs = json.loads(row[field])
            for log in logs:
                source_id = int(log["instanceId"])
                if source_id not in mapping:
                    raise RuntimeError(
                        f"Participant {row['ResponseId']} contains unmapped case {source_id}"
                    )
                participant_ids.add(source_id)
                log["instanceId"] = mapping[source_id]
            row[field] = json.dumps(logs, separators=(",", ":"))
        if participant_ids != expected_source_ids:
            raise RuntimeError(
                f"Participant {row['ResponseId']} did not see all 32 latest instances"
            )

    with output.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerows(raw_rows[:3])
        writer.writerows([[row.get(header, "") for header in headers] for row in complete])

    return {
        "raw_records": len(records),
        "complete_responses": len(complete_candidates),
        "latest_profile_participants": len(latest_complete),
        "stale_profile_participants_excluded": len(complete_candidates)
        - len(latest_complete),
        "incomplete_or_empty_records_excluded": len(records)
        - len(complete_candidates),
        "v1.9_latest_participants_preserved": len(old),
        "new_complete_participants": len(complete),
        "new_xai_participants": dict(
            sorted(Counter(row["xaiType"] for row in complete).items())
        ),
    }


def boundary_worker(arguments: tuple[str, str, str, str]) -> dict[str, object]:
    input_path, experiment_path, output_path, summary_path = map(Path, arguments)
    return calculate_boundaries(
        input_path, experiment_path, output_path, summary_path
    )


def calculate_boundaries_parallel(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    work: Path,
) -> dict[str, object]:
    columns, rows = read_rows(input_path)
    participant_order = list(dict.fromkeys(row["participant"] for row in rows))
    worker_count = min(8, len(participant_order))
    chunk_size = math.ceil(len(participant_order) / worker_count)
    chunks = [
        participant_order[index : index + chunk_size]
        for index in range(0, len(participant_order), chunk_size)
    ]
    chunk_jobs: list[tuple[Path, Path, Path]] = []
    for index, participants in enumerate(chunks):
        participant_set = set(participants)
        chunk_input = work / f"boundary_input_{index:02d}.csv"
        chunk_output = work / f"boundary_output_{index:02d}.csv"
        chunk_summary = work / f"boundary_summary_{index:02d}.json"
        with chunk_input.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                row for row in rows if row["participant"] in participant_set
            )
        chunk_jobs.append((chunk_input, chunk_output, chunk_summary))

    summaries: list[dict[str, object] | None] = [None] * len(chunk_jobs)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        pending = {
            executor.submit(
                boundary_worker,
                (str(chunk_input), str(BUNDLE), str(chunk_output), str(chunk_summary)),
            ): index
            for index, (chunk_input, chunk_output, chunk_summary) in enumerate(chunk_jobs)
        }
        for future in as_completed(pending):
            index = pending[future]
            summaries[index] = future.result()
            print(f"boundary_chunk_completed={index + 1}/{len(chunk_jobs)}", flush=True)

    merged_rows: list[dict[str, str]] = []
    boundary_columns: list[str] | None = None
    for _, chunk_output, _ in chunk_jobs:
        new_columns, new_rows = read_rows(chunk_output)
        if boundary_columns is None:
            boundary_columns = new_columns
        elif boundary_columns != new_columns:
            raise RuntimeError("Parallel boundary chunks have inconsistent schemas")
        merged_rows.extend(new_rows)
    assert boundary_columns is not None
    for index, row in enumerate(merged_rows, start=1):
        row["row_number"] = str(index)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=boundary_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged_rows)

    completed = [summary for summary in summaries if summary is not None]
    first = completed[0]
    values: dict[str, dict[str, list[float]]] = {}
    for row in merged_rows:
        if row["phase"] != "testing":
            continue
        domain = row["domain"]
        measures = values.setdefault(
            domain, {"original": [], "new": [], "delta": []}
        )
        measures["original"].append(float(row["boundary distance original"]))
        measures["new"].append(float(row["boundary distance new"]))
        measures["delta"].append(float(row["boundary distance change (new - original)"]))

    def summarize(items: list[float]) -> dict[str, float | int]:
        ordered = sorted(items)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return {
            "count": len(items),
            "min": min(items),
            "median": median,
            "mean": sum(items) / len(items),
            "max": max(items),
        }

    summary = {
        key: first[key]
        for key in (
            "definition",
            "boundary",
            "feature_constraint",
            "actionability_constraint",
            "numeric_features",
            "categorical_features",
            "solver",
        )
    }
    summary.update(
        {
            "parallel_workers": worker_count,
            "parallel_chunks": len(chunk_jobs),
            "input_rows": len(rows),
            "testing_scored": sum(
                int(item["testing_scored"]) for item in completed
            ),
            "unique_profiles_solved_across_chunks": sum(
                int(item["unique_profiles_solved"]) for item in completed
            ),
            "by_domain": {
                domain: {
                    measure: summarize(items)
                    for measure, items in measures.items()
                }
                for domain, measures in values.items()
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def validate_output(path: Path, expected_participants: int) -> dict[str, object]:
    v19_columns, v19_rows = read_rows(V19)
    columns, rows = read_rows(path)
    if columns != v19_columns:
        raise RuntimeError("v2.0 column schema/order does not exactly match v1.9")

    participants = {row["participant"] for row in rows}
    overlap = participants & {row["participant"] for row in v19_rows}
    if overlap:
        raise RuntimeError(f"v2.0 contains v1.9 participants: {sorted(overlap)}")
    if len(participants) != expected_participants:
        raise RuntimeError(
            f"Expected {expected_participants} participants, found {len(participants)}"
        )

    participant_rows = Counter(row["participant"] for row in rows)
    if set(participant_rows.values()) != {32}:
        raise RuntimeError(f"Unexpected rows per participant: {Counter(participant_rows.values())}")
    training_ids = {
        int(float(row["instance id"])) for row in rows if row["phase"] == "training"
    }
    testing_ids = {
        int(float(row["instance id"])) for row in rows if row["phase"] == "testing"
    }
    if training_ids != EXPECTED_TRAINING_IDS or testing_ids != EXPECTED_TESTING_IDS:
        raise RuntimeError("v2.0 is not restricted to the latest 12+20 instances")

    return {
        "scope": "complete participants collected after v1.9 using the latest instances",
        "rows": len(rows),
        "columns": len(columns),
        "participants": len(participants),
        "participant_row_counts": dict(Counter(participant_rows.values())),
        "phase_rows": dict(Counter(row["phase"] for row in rows)),
        "xai_participants": {
            xai: len({row["participant"] for row in rows if row["xai"] == xai})
            for xai in sorted({row["xai"] for row in rows})
        },
        "training_instance_ids": sorted(training_ids),
        "testing_instance_ids": sorted(testing_ids),
        "v1.9_participant_overlap": 0,
    }


def merge_with_v19(new_path: Path) -> dict[str, object]:
    columns, old_rows = read_rows(V19)
    new_columns, new_rows = read_rows(new_path)
    if columns != new_columns:
        raise RuntimeError("New cohort schema/order does not exactly match v1.9")

    old_participants = {row["participant"] for row in old_rows}
    new_participants = {row["participant"] for row in new_rows}
    overlap = old_participants & new_participants
    if overlap:
        raise RuntimeError(f"Participant overlap while merging v2.0: {sorted(overlap)}")

    rows = old_rows + new_rows
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    participant_rows = Counter(row["participant"] for row in rows)
    if set(participant_rows.values()) != {32}:
        raise RuntimeError(f"Unexpected rows per merged participant: {Counter(participant_rows.values())}")
    training_ids = {
        int(float(row["instance id"])) for row in rows if row["phase"] == "training"
    }
    testing_ids = {
        int(float(row["instance id"])) for row in rows if row["phase"] == "testing"
    }
    if training_ids != EXPECTED_TRAINING_IDS or testing_ids != EXPECTED_TESTING_IDS:
        raise RuntimeError("Merged v2.0 contains rows outside the latest instance set")

    return {
        "scope": "all complete participants using the latest 12 training and 20 testing instances",
        "rows": len(rows),
        "columns": len(columns),
        "participants": len(participant_rows),
        "v1.9_participants": len(old_participants),
        "new_participants_after_v1.9": len(new_participants),
        "participant_row_counts": dict(Counter(participant_rows.values())),
        "phase_rows": dict(Counter(row["phase"] for row in rows)),
        "xai_participants": {
            xai: len({row["participant"] for row in rows if row["xai"] == xai})
            for xai in sorted({row["xai"] for row in rows})
        },
        "training_instance_ids": sorted(training_ids),
        "testing_instance_ids": sorted(testing_ids),
        "participant_overlap_between_waves": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merge-existing-output",
        action="store_true",
        help="Treat the existing v2.0 file as the already processed new-only cohort and merge v1.9 into it.",
    )
    args = parser.parse_args()

    if args.merge_existing_output:
        _, existing_rows = read_rows(OUTPUT)
        expected = len({row["participant"] for row in existing_rows})
        new_summary = validate_output(OUTPUT, expected)
        combined_summary = merge_with_v19(OUTPUT)
        prior = (
            json.loads(SUMMARY.read_text(encoding="utf-8"))
            if SUMMARY.exists()
            else {}
        )
        final_summary = {
            **combined_summary,
            "source_raw": str(RAW.relative_to(ROOT)),
            "schema_reference": str(V19.relative_to(ROOT)),
            "new_cohort_validation": new_summary,
            "new_cohort_processing": prior,
        }
        SUMMARY.write_text(json.dumps(final_summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(final_summary, indent=2))
        return

    with tempfile.TemporaryDirectory(prefix="v20_") as temporary:
        work = Path(temporary)
        remapped = work / "new_raw_remapped.csv"
        base = work / "new_base.csv"
        boundary = work / "new_boundary.csv"
        boundary_summary = work / "boundary_summary.json"
        processed_summary = work / "processed_summary.json"
        new_processed = work / "new_processed.csv"

        print("stage=isolate_and_remap", flush=True)
        cohort_summary = isolate_and_remap(remapped)
        print("stage=convert", flush=True)
        base_rows, conversion_summary = convert([remapped], BUNDLE)
        write_output(base, base_rows)
        print("stage=boundary_parallel", flush=True)
        calculate_boundaries_parallel(base, boundary, boundary_summary, work)

        print("stage=assemble_56_columns", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_v12_diabetes_clustering_results.py"),
                "--base",
                str(base),
                "--reference-csv",
                str(V19),
                "--raw",
                str(remapped),
                "--boundary",
                str(boundary),
                "--bundle",
                str(BUNDLE),
                "--output",
                str(new_processed),
                "--summary",
                str(processed_summary),
                "--continuous-validity-column",
                "successful counterfactual (continuous)",
                "--skip-strategies",
            ],
            check=True,
        )

        print("stage=validate_and_merge", flush=True)
        new_summary = validate_output(
            new_processed, int(cohort_summary["new_complete_participants"])
        )
        final_summary = {
            **merge_with_v19(new_processed),
            "source_raw": str(RAW.relative_to(ROOT)),
            "schema_reference": str(V19.relative_to(ROOT)),
            "new_cohort_validation": new_summary,
            "cohort_filter": cohort_summary,
            "conversion": conversion_summary,
            "boundary_metrics": json.loads(boundary_summary.read_text(encoding="utf-8")),
        }
        SUMMARY.write_text(json.dumps(final_summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
