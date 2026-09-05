# Repository structure and data lineage

## Main folders

- `src/`: application pipeline, XAI methods, and cognitive-model implementations.
- `scripts/`: reusable survey builders, converters, validators, simulations, and analyses. Versioned scripts are retained because older raw/result files depend on their contemporary schemas.
- `static/`: generated browser payload used by the interface.
- `qualtrics/`: main QSF survey versions, case manifests, current preview/interface assets, all main `raw_output_*` exports, and all main `qualtrics_results_*` tables.
- `analysis/`: ignored working assets. The retained files are the diabetes experiment bundles needed to map survey IDs back to dataset rows and the fitted MLP models needed to reproduce predictions.
- `outputs/v25-cognitive-model-fits/`: fits for the first 42 participants in v2.5.
- `outputs/v25-new-cohort-cognitive-model-fits/`: fits for the 21 participants added after the first fit.

Generated exploratory tables, plots, round-trip copies, staging tables, and QSF backups were removed after snapshot commit `aadc90d`. They can be restored from that commit if needed.

## Diabetes instance identifiers

There are three identifiers in the pipeline:

1. `row_id` is the closest thing to a global ID in the prepared diabetes dataset. It is assigned before the train/dev/test split and is stored in each split CSV.
2. `source_instance_id` is a zero-based row position **within one split**. Its full key is therefore `(dataset, source_split, source_instance_id)`, such as `diabetes:test:10`. The regular pipeline API also calls this split-local row position `instance_id`.
3. `instance_id` values `130100`-`130309` are stable survey/UI IDs assigned to the curated experiment. They are not original dataset row numbers:
   - `130100`-`130111`: 12 training cases;
   - `130200`-`130209`: 10 testing cases initially predicted Diabetes;
   - `130300`-`130309`: 10 testing cases initially predicted No Diabetes.

The experiment bundle records the mapping from the stable survey ID to `source_split` and `source_instance_id`. The current v1.7 bundle does not copy `row_id` into the UI payload, but `row_id` can be recovered by looking up the recorded split-local position in the corresponding split CSV. The `13` prefix is an experiment namespace chosen to avoid collisions; it is not a meaningful source-dataset number.

## Current diabetes instance organization

The v1.7 experiment has 12 training and 20 testing cases. Training is balanced across prediction direction and two explanation-edit clusters:

- cluster 1: Glucose + BMI (three Diabetes and three No Diabetes profiles);
- cluster 2: Blood Pressure + Insulin (three Diabetes and three No Diabetes profiles).

Testing has ten profiles in each prediction direction. Profiles were selected near same-label training profiles and have a one-feature reference recourse; the interface locks Glucose and allows the participant to edit only one of Blood Pressure, Insulin, BMI, or Age. The stable survey IDs, source IDs, cluster, nearest training profile, and reference edit are stored in `analysis/diabetes_experiment_bundle_v1.7_actionable.json`.

## SafeLimit provenance

The current SafeLimit rows were generated locally and deterministically by `src/data_manager.py`; they were not copied row-for-row from an external dataset. The generator uses NumPy seed 88 to create 2,000 profiles with alcohol units, weight, drinking duration, gender, and stomach fullness. It then calculates BAC using a Widmark-style equation with sex-specific distribution constants (`0.68` male, `0.55` female), a stomach-fullness absorption factor (`0.82` full, `1.0` empty), elimination of `0.015` BAC per hour, and a `0.08` threshold.

The generator and metadata describe this as an adaptation of the SafeLimit BAC task from Warren et al. (IUI 2023 / TiiS 2024). `src/data/safelimit/IUI_dataset.xlsx` is a copy of the authors' prior-study participant-response workbook (sheets `Experiment 1` and `Experiment 2`); the current synthetic feature rows are not read from that workbook.
