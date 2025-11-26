# Wide & Deep IRT Phase 1 (EDM Cup Reproduction)

This ExecPlan follows `PLANS.md` and must remain a living document. Update `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` as work proceeds.

## Purpose / Big Picture

Replicate the Wide & Deep IRT pipeline described in “Predicting Students’ Future Success: Harnessing Clickstream Data with Wide & Deep Item Response Theory” using the EDM Cup 2023 dataset. At completion, we will: (1) generate clickstream-derived features matching the paper’s Student Action taxonomy, (2) train a Wide & Deep model that outputs correctness probabilities plus item ability parameters, (3) match the paper’s reported metrics (AUC, etc.), and (4) export the Item Health Lab artifacts (`item_params.parquet`, `item_drift.parquet`, `behavior_slices.md`) for future UI work. This establishes confidence that our implementation is faithful before we extend or integrate.

## Progress

- [x] (2025-11-26 03:25Z) Reviewed OSF archive (`osfstorage-archive/edm2023`) to capture feature/model expectations (TF-based WideDeep architecture with item beta & guessing weights, action encoders, schema utilities).
- [x] (2025-11-26 03:45Z) Added history feature encoding utilities (`FeatureConfig`, `FeatureStats`, `encode_history_sequences`) plus unit tests validating padding, action codes, and latency bucketization.
- [x] (2025-11-26 04:10Z) Implemented `EdmClickstreamDataset` with data joins across assignment details/relationships/scores, caching, and dataset unit test covering sample generation.
- [x] (2025-11-26 04:30Z) Added PyTorch Lightning `WideDeepIrtModule`, training CLI (`src/wd_irt/train.py`), YAML trainer options, and basic test coverage for features/dataset to unblock training.
- [ ] Build PyTorch Lightning module + trainer in `src/wd_irt/model.py` / `train.py`, wired to configs and data loaders; add tests for shape/forward-pass sanity.
- [ ] Run training on EDM Cup, log metrics, and compare to reported AUC; iterate until alignment (document results).
- [ ] Implement exporters producing `item_params.parquet`, `item_drift.parquet`, `behavior_slices.md`; validate file schemas.
- [ ] Update README/execplan docs, capture final metrics, and note outstanding gaps in Outcomes section.

## Surprises & Discoveries

- Observation: Reference implementation is TensorFlow/Keras; we must port the architecture to PyTorch Lightning while keeping math consistent (wide component = sigmoid ability minus difficulty + guessing mix).

## Decision Log

- Record each significant choice (e.g., optimizer settings, batching strategy, handling missing actions, drift window definitions) with rationale and date.

## Outcomes & Retrospective

Will capture once phase concludes: summarize achieved metrics vs paper, artifact locations, and follow-up actions.

## Context and Orientation

Repository structure:

- `src/wd_irt/` currently contains stubs (`datasets.py`, `features.py`, `model.py`, `train.py`, `export.py`). These must be filled.
- `configs/wd_irt_edm.yaml` offers initial hyperparameter placeholders; we’ll update to match the paper/OSF defaults.
- `data/interim/edm_cup_2023_<seed>_events.parquet` already exists from the data pipeline; this is the input.
- `reports/` is the destination for metrics and artifact outputs.
- OSF bundle downloaded from `osfstorage-archive/` contains reference implementation (files: `preprocessing.py` with one-hot utilities, `dataset.py` describing raw/processed schemas and relationships, `model.py` implementing the Wide & Deep architecture in TensorFlow).

Constraints:

- Match paper’s feature definitions and evaluation metrics before customizing.
- Keep training reproducible (seeded, deterministic). Use the uv environment.
- Document any deviations or assumptions (e.g., if OSF code lacks clarity).

## Plan of Work

### 1. Reference Acquisition & Study

1. Extract OSF archive (`osfstorage-archive/edm2023/...`) and identify scripts/notebooks implementing features, model, evaluation.
2. Read relevant sections: feature engineering (student actions, sliding windows), model architecture (wide + deep structure, ability head), training procedure (optimizer, loss, regularization), evaluation metrics (AUC, AP, calibration). Summarize in ExecPlan for quick reference.

### 2. Feature Engineering Implementation

1. Design a schema for intermediate feature tables (per student-problem interaction). Variables include counts of student actions (hints, answer requests), time features (latency, time since start), aggregated behavior statistics. Use the canonical events parquet plus `action_logs.csv` as needed.
2. Implement functions in `src/wd_irt/features.py`:
   - `build_student_action_features(events: pd.DataFrame, raw_clickstream_path: Path) -> pd.DataFrame`
   - `assemble_feature_frame(...)` combining wide (IRT-like) and deep features.
3. Add `tests/test_wd_irt_features.py` covering:
   - Counting action types per window.
   - Handling missing actions / zero padding.
   - Matching sample outputs from OSF reference (create small fixtures).

### 3. Dataset & DataLoader Setup

1. Implement `src/wd_irt/datasets.py` to load features into PyTorch tensors:
   - `WideDeepDataset` returning `(wide_features, deep_features, labels, metadata)`.
   - Support train/val/test splits using `data/splits/edm_cup_2023_<seed>.json`.
2. Provide a collate function that handles variable-length sequences if needed (e.g., for behavioral histories).

### 4. Model & Training

1. Flesh out `src/wd_irt/model.py`:
   - `WideDeepConfig` extended with parameters from `wd_irt_edm.yaml`.
   - `WideDeepIrtModule` inheriting from `pl.LightningModule`, with:
     - Wide branch modeling item ability/difficulty (IRT-like linear component).
     - Deep branch (MLP) over clickstream features.
     - Outputs: probability of correctness and latent ability estimates.
     - Loss combining binary cross-entropy with ability regularizer.
   - Methods for optimizer configuration, training/validation/test steps logging metrics.
2. Build training CLI in `src/wd_irt/train.py`:
   - Parse config (use YAML loader).
   - Instantiate dataset/dataloaders.
   - Configure Lightning Trainer (seed, callbacks, checkpointing).
   - Save metrics to `reports/metrics/...`.
3. Add unit/integration tests:
   - Model forward pass with dummy tensors (check shapes).
   - Training loop smoke test on a tiny subset (few batches).

### 5. Evaluation & Metrics

1. Implement evaluation utilities (maybe under `src/common/evaluation.py` or within `wd_irt`) to compute AUC, AP, calibration ECE as per paper.
2. Ensure validation/test metrics logged match the reported format (per-split results).
3. Compare metrics to OSF outputs; iterate on hyperparameters if necessary. Document actual values.

### 6. Exporters / Item Health Lab

1. Implement `src/wd_irt/export.py` with functions:
   - `export_item_parameters(checkpoint, output_path)`: extract item difficulty/discrimination/guessing (as available from model weights).
   - `export_item_drift(events, parameters, window="weekly"/"monthly")`: compute drift scores using sliding windows.
   - `generate_behavior_slices(features, ability_predictions, output_md)`: summarize behaviors across ability quantiles (mirroring paper’s analysis).
2. Add CLI or Makefile target hooking into exports after training. Ensure outputs land in `reports/`.
3. Validate artifact schemas (e.g., columns for `item_params.parquet`) and include sample rows in ExecPlan.

### 7. Documentation & Verification

1. Update `README.md` (and possibly a dedicated `reports/ITEM_HEALTH.md`) to describe how to train WD-IRT and where outputs are stored.
2. Summarize steps, hyperparameters, and metrics in this ExecPlan’s `Outcomes` section.
3. Consider adding a model card entry describing WD-IRT assumptions/limitations.

## Concrete Steps

1. Inspect OSF archive: `ls osfstorage-archive/edm2023`.
2. Create fixtures and tests for feature engineering.
3. Implement features/datasets/models incrementally, running unit tests after each major addition (`uv run pytest tests/test_wd_irt_features.py`, etc.).
4. Execute training: `uv run python -m src.wd_irt.train --config configs/wd_irt_edm.yaml`.
5. Run evaluation/export commands; verify outputs exist and contain expected columns.
6. Capture metrics (e.g., `Validation AUC = 0.82` vs target) in ExecPlan.

## Validation and Acceptance

- Unit tests for feature extraction and model components pass.
- Training CLI completes on EDM dataset, producing checkpoints and metrics equivalent (or close) to paper’s results (document target vs actual).
- `reports/item_params.parquet`, `reports/item_drift.parquet`, and `reports/behavior_slices.md` exist with meaningful data (spot-check a few rows).
- README instructions allow another contributor to rerun training/evaluation/export without ambiguity.
- ExecPlan updated with decisions, surprises, artifacts, and final outcomes.

## Idempotence and Recovery

- Training scripts should respect config seeds and log directories, avoiding overwriting unless `--run-name` or output paths are unique.
- Exporters should write to new files or support `--force`.
- If training crashes, checkpoints in `reports/checkpoints/wd_irt_edm/` allow resuming (consider enabling Lightning checkpointing).
- Keep data pipeline outputs untouched; training consumes `data/interim/edm_cup_2023_*` and `data/splits/...`.

## Artifacts and Notes

- Add sample metrics output in ExecPlan once obtained, e.g.:

    ```
    Epoch 10: val_auc=0.823, val_ap=0.611, test_auc=0.821
    ```

- Document command snippets:

    ```
    uv run python -m src.wd_irt.train --config configs/wd_irt_edm.yaml
    uv run python -m src.wd_irt.export --checkpoint reports/checkpoints/wd_irt_edm/epoch=10.ckpt --output-dir reports/
    ```

- Capture excerpts of exported parquet files (schema, first few rows) for clarity.

## Interfaces and Dependencies

- Dependencies: PyTorch 2.3, PyTorch Lightning 2.4, pandas/pyarrow (already installed).
- Feature builder functions accept pandas DataFrames from canonical events; dataset class should rely on numpy/torch tensors.
- Model interface: `WideDeepIrtModule.forward(wide_x, deep_x)` returning logits and ability estimates; training step expects `(wide_x, deep_x, y)`.
- Export functions need access to trained module parameters; design so they can load from checkpoint path via Lightning.
