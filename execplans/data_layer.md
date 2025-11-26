# Data Layer for Canonical Learning Events

This ExecPlan is a living document maintained under `execplans/` per `PLANS.md`.

## Purpose / Big Picture

Enable both engines (Wide & Deep IRT and SAKT) to consume the same canonical learning-event representation. After this work, contributors can run a single CLI to transform EDM Cup 2023 clickstream CSVs (already stored in `edm-cup-2023/`) into normalized parquet files plus deterministic train/val/test splits in `data/`. The same pipeline will later extend to ASSISTments2009, but this milestone focuses on EDM Cup so model development can proceed with real data.

## Progress

- [x] (2025-11-26 01:05Z) Moved EDM Cup CSV bundle into `data/raw/edm_cup_2023/`, recorded SHA256 guidance, and documented files in `data/README.md`.
- [x] (2025-11-26 01:45Z) Added `src/common/data_pipeline.py` with schema enforcement, Arrow-based ingestion, deterministic split generator, and Typer CLI; created fixture-backed unit tests.
- [x] (2025-11-26 01:55Z) Wiring into Makefile (`make data`) and README quickstart instructions, plus uv-based dependency instructions.
- [x] (2025-11-26 02:10Z) Executed pipeline on full EDM dataset producing `data/interim/edm_cup_2023_42_events.parquet` (≈5.17M rows) and split manifest; captured stats in Surprises/Artifacts.

## Surprises & Discoveries

- Observation: Reading `action_logs.csv` via standard pandas was extremely slow (~30 minutes). Switched to `pyarrow.csv` with column filtering + action pre-filtering to cut load time and memory usage. Evidence: optimized run generated 5,167,603 events in ~6 minutes instead of >30.
- Observation: uv-managed environment lacks pandas/pyarrow by default; had to initialize `.venv` via `uv venv` then `uv pip install ...` before running tests/pipeline.

## Decision Log

- Decision: Represent skill tags as list-of-strings by splitting on commas/semicolons, defaulting to empty list when metadata missing. Rationale: keeps schema stable for both models. Date: 2025-11-26.
- Decision: Build help flag via cumulative sum over filtered actions (`PROBLEM_START`, helps, responses) after pre-filtering with PyArrow to minimize memory. Rationale: reduces dataset size before handing to pandas while preserving necessary ordering info. Date: 2025-11-26.
- Decision: Use JSON split manifest with explicit train/val/test arrays instead of storing user assignments in parquet to keep CLI simple. Rationale: human-readable and matches README spec. Date: 2025-11-26.

## Outcomes & Retrospective

Pipeline now reproducibly generates canonical EDM events and deterministic splits. UV-based Makefile target (`make data split_seed=42`) writes `data/interim/edm_cup_2023_42_events.parquet` (~87 MB) and `data/splits/edm_cup_2023_42.json` (train 24,634 / val 5,278 / test 5,280). Remaining work: extend pipeline to additional datasets (ASSISTments) and enrich events with more context (e.g., sequence metadata) as future phases.

## Context and Orientation

Current assets:

- Raw EDM Cup CSVs located at `/Users/nainy/Documents/Personal/deepKT+IRT/edm-cup-2023/`.
- Repository scaffolding already expects canonical events in `data/interim/<dataset>_<seed>_events.parquet` and split manifests under `data/splits/`.
- `src/common/schemas.py` defines the `LearningEvent` dataclass.
- `Makefile` target `make data` references a non-existent `src.common.data_pipeline` module; this plan will implement it.

Constraints:

- Must preserve deterministic splits keyed by student/user id.
- Keep environment reproducible; no heavy dependencies beyond pandas/pyarrow/numpy already in `environment.yml`.
- Document any preprocessing assumptions (e.g., filtering columns) in README updates.

## Plan of Work

1. **Raw data organization**: Move or symlink `edm-cup-2023/` into `data/raw/edm_cup_2023/`. Update `data/README.md` to describe the new folder, expected file list, and checksums. Record SHA256 sums for each CSV to ensure integrity. ✅
2. **Schema contract**: Create `src/common/schema_validation.py` (or extend `schemas.py`) with helpers to validate and coerce raw DataFrames into the `LearningEvent` structure. Include conversions for timestamps, skill list parsing, and boolean normalization. (Handled within `data_pipeline.py` to keep scope focused.) ✅
3. **Preprocessing module**: Implement `src/common/data_pipeline.py` containing functions:
   - `load_raw_edm(raw_dir: Path) -> pd.DataFrame`: read `action_logs.csv`, join with metadata (assignment_details, problem_details, etc.).
   - `normalize_events(df) -> pd.DataFrame`: project to canonical columns, apply data cleaning (drop missing user_id/item_id, enforce chronological ordering).
   - `write_events(df, out_path)`.
   - `generate_splits(df, seed) -> Dict[str, List[str]]` splitting unique `user_id` into train/val/test (e.g., 70/15/15).
   - Typer CLI `main` entry that accepts `--raw-dir`, `--output-path`, `--split-manifest`, `--seed`, `--dataset`.
4. **Unit tests**: Under `tests/` (create if absent), write tests verifying:
   - Schema validation rejects missing columns.
   - Split generator is deterministic for a fixed seed.
   - Learning events are sorted per user by timestamp.
   Use small fixture CSVs stored under `tests/fixtures/`.
5. **Makefile integration**: Update `Makefile data` target to call the new CLI with defaults pointing to `data/raw/edm_cup_2023` and output `data/interim/edm_cup_2023_<seed>_events.parquet`.
6. **Documentation**: Expand `README.md` and `data/README.md` describing the pipeline invocation (`make data split_seed=42`), output filenames, and verification steps (row counts, sample commands). ✅
7. **Verification workflow**: Provide sample commands (e.g., `python3 -m src.common.data_pipeline --help`, `python3 -m src.common.data_pipeline --raw-dir data/raw/edm_cup_2023 --seed 42 --dataset edm_cup_2023`). Document expected console output and resulting files. ✅

## Concrete Steps

1. `mv edm-cup-2023 data/raw/edm_cup_2023` (or create symlink) to align with repo structure.
2. Extend `data/README.md` with table listing raw files and checksums.
3. Create `src/common/data_pipeline.py` with Typer CLI and helper functions described above.
4. Add schema validation helper(s) and import them where needed.
5. Introduce `tests/` directory containing fixture CSVs and pytest cases (document test command in README).
6. Update `Makefile` data target to call `$(PYTHON) -m src.common.data_pipeline ...`.
7. Run pipeline locally to ensure `data/interim/edm_cup_2023_42_events.parquet` and `data/splits/edm_cup_2023_42.json` appear.
8. Capture outputs (ls listings, head of parquet via `pyarrow.parquet`) in ExecPlan `Artifacts` section.

## Validation and Acceptance

- `uv run python -m src.common.data_pipeline --raw-dir data/raw/edm_cup_2023 --dataset edm_cup_2023 --seed 42 --events-out data/interim/edm_cup_2023_42_events.parquet --splits-out data/splits/edm_cup_2023_42.json` succeeds (produced 5,167,603 events; command output captured below).
- Generated parquet contains required columns; inspected via pandas/pandas head.
- Split manifest counts: `{'train': 24634, 'val': 5278, 'test': 5280}` with no overlap.
- Rerunning with the same seed yields identical manifest (verified by JSON diff).
- README instructions document uv setup + `make data`.

## Idempotence and Recovery

- Pipeline should overwrite existing outputs atomically (write to temp file, then rename) to avoid partial data.
 - Provide `--force` flag or timestamped outputs to prevent accidental overwrites if needed.
- If command fails mid-run, deleting partially written files and rerunning should restore consistency since raw data remains untouched.

## Artifacts and Notes

```
$ make data split_seed=42
[data] Building canonical events from data/raw/edm_cup_2023
[data] Writing 5167603 events to data/interim/edm_cup_2023_42_events.parquet
[data] Train=24634 Val=5278 Test=5280
```

```
$ ls -lh data/interim/edm_cup_2023_42_events.parquet
-rw-r--r--  87M Nov 25 20:39 data/interim/edm_cup_2023_42_events.parquet
```

```
$ head data/splits/edm_cup_2023_42.json
{
  "train": [
    "21FBAKBQSD",
    "1LGLR1BC8H",
    ...
```

## Interfaces and Dependencies

- Dependencies: pandas, pyarrow (already in environment), typer for CLI (already used).
- Exposed CLI: `src.common.data_pipeline` Typer app with options `--raw-dir`, `--dataset`, `--seed`, `--events-out`, `--splits-out`, `--train-ratio`, `--val-ratio`.
- Helper functions should accept/return pandas DataFrames to keep compatibility with future feature builders.
