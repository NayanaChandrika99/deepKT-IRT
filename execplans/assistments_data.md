# ASSISTments Skill Builder Canonicalization

This ExecPlan is maintained per `PLANS.md`; keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` up to date as work proceeds.

## Purpose / Big Picture

Transform `data/skill_builder_data.csv` (ASSISTments skill builder dataset) into the canonical learning-event schema shared by both engines. After implementing this plan, running `make data dataset=assistments_skill_builder split_seed=42` will generate `data/interim/assistments_skill_builder_42_events.parquet` and a deterministic user split manifest. This unlocks SAKT training via pyKT and keeps the data layer consistent with the EDM Cup pipeline already in place.

## Progress

- [x] (2025-11-26 02:25Z) Drafted ExecPlan covering raw organization, preprocessing, tests, Makefile/docs updates, and validation strategy.
- [x] (2025-11-26 02:32Z) Moved `skill_builder_data.csv` to `data/raw/assistments_skill_builder/`, generated `sha256sum.txt`, and documented the dataset in `data/README.md`.
- [x] (2025-11-26 03:05Z) Extended `src/common/data_pipeline.py` with dataset dispatch + ASSISTments preprocessing, plus new fixture-based unit tests verifying schema/help/skills.
- [x] (2025-11-26 03:10Z) Updated README instructions, reran unit tests, and executed `make data dataset=assistments_skill_builder split_seed=42` to produce canonical parquet/split outputs.

## Surprises & Discoveries

- Observation: `skill_builder_data.csv` lacks absolute timestamps; we synthesize them using `order_id` offsets from `2009-01-01` to preserve ordering. Evidence: unit test confirmed monotonic timestamps per user.
- Observation: Some records omit `skill_id` but include `skill_name`, so fallback logic maps the skill name into `skill_ids` to avoid empty lists. Evidence: fixture row without `skill_id` produced `["Decimals"]`.

## Decision Log

- Decision: Use `problem_id` as canonical `item_id` for ASSISTments so both pipelines reference the same identifier format. Rationale: `problem_id` aligns with pyKT configs and is stable, whereas `assistment_id` can be shared across problems. Date: 2025-11-26.
- Decision: Set synthetic timestamps by interpreting `order_id` as seconds offset from `2009-01-01` UTC. Rationale: dataset lacks absolute times; this method yields deterministic ordering while keeping values within datetime range. Date: 2025-11-26.
- Decision: Flag `help_requested` when any of `hint_count`, `hint_total`, or `bottom_hint` > 0 to ensure all assistance signals are captured. Date: 2025-11-26.

## Outcomes & Retrospective

ASSISTments skill builder ingestion now mirrors the EDM path. Running `make data dataset=assistments_skill_builder split_seed=42` produces `data/interim/assistments_skill_builder_42_events.parquet` (~5.3 MB, 525,534 rows) plus `data/splits/assistments_skill_builder_42.json` (train 2,951 / val 632 / test 634). Docs explain the dataset override, and tests cover both pipelines. Future iterations can enrich events with class/sequence metadata if modeling needs arise.

## Context and Orientation

Relevant files/directories:

- `data/skill_builder_data.csv`: raw ASSISTments dataset currently living at repo root `data/`. Needs relocation under `data/raw/`.
- `src/common/data_pipeline.py`: existing EDM Cup preprocessing pipeline using PyArrow for fast ingestion; will be extended with dataset-specific code paths.
- `tests/test_data_pipeline.py`: holds current unit tests for EDM pipeline; we will add tests for the ASSISTments helpers.
- `Makefile`: `make data` currently assumes EDM Cup. We will expose `dataset` parameter already wired (default `edm_cup_2023`) so nothing else required beyond documentation.
- `README.md` and `data/README.md`: must describe new dataset workflow and CLI usage.

## Plan of Work

1. **Raw data organization**: Create `data/raw/assistments_skill_builder/` and move `skill_builder_data.csv` there. Generate `sha256sum.txt` for traceability. Update `data/README.md` with a table describing the ASSISTments file and instructions for acquiring it.
2. **Schema mapping**: Analyze columns (e.g., `user_id`, `problem_id`, `skill_id`, `correct`, `ms_first_response`, `order_id`, `assignment_id`, `hint_count`, `hint_total`). Define mappings:
   - `user_id` → canonical `user_id`.
   - `problem_id` or `assistment_id` as `item_id` (decide and document).
   - `skill_ids` derived from `skill_id` (split on underscores if multiple) while falling back to skill name if needed.
   - `timestamp` synthesized from `order_id` (converted to UTC timestamp) or derived from `ms_first_response` relative to a reference start; document assumption.
   - `correct` from `correct` column (binary).
   - `action_sequence_id` using `assignment_id`.
   - `latency_ms` from `ms_first_response`.
   - `help_requested` flagged if `hint_count > 0`, `hint_total > 0`, or `bottom_hint` truthy.
3. **Pipeline extension**: Inside `src/common/data_pipeline.py`, add functions:
   - `_load_assistments_skill_builder(path: Path) -> pd.DataFrame`.
   - `_prepare_assistments_events(df: pd.DataFrame) -> pd.DataFrame`.
   - Branch within `prepare_learning_events` based on `dataset` argument; optionally split the function into dataset-specific handlers for clarity.
4. **Unit tests**: Extend `tests/test_data_pipeline.py` (or create a new module) with fixtures covering ASSISTments rows. Tests should confirm:
   - Column mapping produces expected canonical schema.
   - `help_requested` reflects hint usage.
   - Artificial timestamps are monotonic per user.
   - Splits remain deterministic for the new dataset.
5. **CLI & docs**: Ensure `Makefile` dataset parameter works (already defined). Update `README.md` quickstart to mention `make data dataset=assistments_skill_builder`. Document the dataset option plus any assumptions in `data/README.md`.
6. **Validation**: Run `make data dataset=assistments_skill_builder split_seed=42` to generate outputs. Record file sizes, row counts, and sample manifest entries in this ExecPlan’s `Artifacts` section along with observed runtime.

## Concrete Steps

1. Move raw CSV into `data/raw/assistments_skill_builder/` and regenerate checksums.
2. Modify `data/README.md` to include instructions for both datasets, referencing new directory.
3. Update `src/common/data_pipeline.py`: refactor to dispatch by dataset ID, implement ASSISTments-specific logic, and ensure CLI accepts new dataset choice.
4. Expand tests with synthetic ASSISTments fixtures.
5. Re-run `uv run python -m unittest tests.test_data_pipeline`.
6. Execute `make data dataset=assistments_skill_builder split_seed=42` and verify outputs.
7. Update ExecPlan progress, surprises, decision log, and artifacts with empirical data (row counts, sample JSON).

## Validation and Acceptance

- Unit tests pass and cover both EDM and ASSISTments code paths.
- Running `make data dataset=assistments_skill_builder split_seed=42` produces canonical events parquet + split manifest with >0 rows and no schema gaps.
- README/Data README describe how to run the pipeline for either dataset and list assumptions (e.g., timestamp derivation).
- ExecPlan includes evidence of the run (command output, file sizes, split counts).

## Idempotence and Recovery

- Pipeline overwrites previous outputs safely (same as EDM path).
- Users can rerun command with new seeds/dataset without manual cleanup.
- If dataset relocation fails mid-step, file remains in original path; rerun the move after cleaning up partially created directories.

## Artifacts and Notes

```
$ make data dataset=assistments_skill_builder split_seed=42
[data] Building canonical events from data/raw/assistments_skill_builder for dataset='assistments_skill_builder'
[data] Writing 525534 events to data/interim/assistments_skill_builder_42_events.parquet
[data] Train=2951 Val=632 Test=634
```

```
$ ls -lh data/interim/assistments_skill_builder_42_events.parquet
-rw-r--r--  5.3M Nov 25 20:50 data/interim/assistments_skill_builder_42_events.parquet
```

```
$ head data/splits/assistments_skill_builder_42.json
{
  "train": [
    "79962",
    "87225",
    "87312",
    ...
```

## Interfaces and Dependencies

- Reuse pandas + pyarrow for ingestion (ASSISTments CSV is manageable; start with pandas, but read via `pyarrow.csv` if necessary).
- Maintain canonical schema defined in `src/common/schemas.py`.
- CLI remains `python -m src.common.data_pipeline` with required options (`--raw-dir`, `--dataset`, `--events-out`, etc.). Document accepted dataset values: `edm_cup_2023`, `assistments_skill_builder`.
