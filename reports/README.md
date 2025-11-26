# ABOUTME: Describes model outputs stored under reports/.
# ABOUTME: Defines naming conventions for metrics, plots, and exports.

## Required Artifacts

- `item_params.parquet`: item difficulty, discrimination, and guessing-like parameters from Wide & Deep IRT.
- `item_drift.parquet`: rolling comparisons of parameter stability (e.g., weekly deltas).
- `behavior_slices.md`: markdown summary of clickstream behaviors stratified by ability groups.
- `student_state.parquet`: per-student mastery vectors exported from SAKT.
- `next_correct_predictions.parquet`: chronological predictions with timestamps for calibration.
- `demo_trace/`: directory for serialized demo payloads (JSON) used by `scripts/demo_trace.py`.

## Metrics and Plots

Produce evaluation metrics under `reports/metrics/<engine>_<dataset>_<seed>.json` and plots under `reports/plots/<engine>_<dataset>_<seed>/`. Each plot should be accompanied by a short markdown snippet describing interpretation.

## Versioning

Each artifact name must include the `run_name` from its config when multiple experiments are executed on the same dataset. Example: `reports/item_params_wdirt_edm_seed42.parquet`.
