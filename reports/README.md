# ABOUTME: Describes model outputs stored under reports/.
# ABOUTME: Defines naming conventions for metrics, plots, and exports.

## Required Artifacts

- `item_params.parquet`: item difficulty, discrimination, and guessing-like parameters from Wide & Deep IRT.
- `item_drift.parquet`: rolling comparisons of parameter stability (e.g., weekly deltas).
- `behavior_slices.md`: markdown summary of clickstream behaviors stratified by ability groups.
- `sakt_student_state.parquet`: per-interaction mastery vectors exported from SAKT.
- `sakt_predictions.parquet`: chronological predictions with timestamps for calibration.
- `sakt_attention.parquet`: attention-derived top influences for explainability.
- `skill_mastery.parquet`: aggregated mastery by `(user_id, skill)` used by the demo CLI.
- `bandit_state.npz`: serialized LinUCB weights/covariance for RL recommendations.
- `checkpoints/`: Lightning.ai-trained checkpoint directories (e.g., `sakt_edm/sakt_edm_seed42_best.pt`, `wd_irt_edm/latest.ckpt`) kept in repo so exports can run offline.

## Metrics and Plots

Produce evaluation metrics under `reports/metrics/<engine>_<dataset>_<seed>.json` and plots under `reports/plots/<engine>_<dataset>_<seed>/`. Each plot should be accompanied by a short markdown snippet describing interpretation.

## Versioning

Each artifact name must include the `run_name` from its config when multiple experiments are executed on the same dataset. Example: `reports/item_params_wdirt_edm_seed42.parquet`.
