# Repo Skeleton for Clickstream Twin Engines

This ExecPlan is a living document. Maintain it in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

The repository must host two complementary learning analytics engines: Wide & Deep IRT for item health and SAKT for student readiness. After implementing this plan, a contributor can check out the repo, install dependencies from `environment.yml`, inspect clear directories for data, configs, and source packages, and run placeholder commands that outline the final train/eval/export surfaces. The skeleton makes it obvious where to plug in datasets, training scripts, and reporting artifacts without writing actual model code yet.

## Progress

- [x] (2025-11-26 00:40Z) Created ExecPlan and base directory scaffolding (`data`, `src/common`, `src/wd_irt`, `src/sakt_kt`, `reports`, `configs`, `scripts`, `execplans`).
- [x] (2025-11-26 01:05Z) Authored root README plus Makefile overview describing quickstart, directory layout, and workflow targets.
- [x] (2025-11-26 01:12Z) Added `environment.yml`, configs for WD-IRT and SAKT, along with per-directory READMEs documenting data and report contracts.
- [x] (2025-11-26 01:22Z) Implemented stub Python modules with ABOUTME headers across `src/common`, `src/wd_irt`, and `src/sakt_kt`.
- [x] (2025-11-26 01:28Z) Built Typer-based demo CLI plus documentation updates referencing the command.

## Surprises & Discoveries

- Observation: macOS sandbox blocks `ps` inside Homebrew shellenv script, but it only emits a warning and does not impact commands. Evidence: repeated `/bin/ps: Operation not permitted` lines preceding successful command output. No mitigation required now.
- Observation: System Python on host lacks Typer, so running the demo CLI requires creating the project environment first. Evidence: `ModuleNotFoundError: No module named 'typer'` when executing `python3 scripts/demo_trace.py --help`.

## Decision Log

- Decision: Use `execplans/repo_skeleton.md` to track this work so the executable spec sits alongside the implementation. Rationale: there was no prior plan file for this scope, and PLANS.md requires self-contained specs. Date/Author: 2025-11-26 / Claude.
- Decision: Introduce placeholder README files within each major directory to keep empty folders under version control once git exists, rather than `.gitkeep`. Rationale: README text conveys intent while satisfying structural requirements. Date/Author: 2025-11-26 / Claude.
- Decision: Implement Python stubs that raise `NotImplementedError` instead of pseudo-logic to avoid misleading behavior while still defining interfaces. Rationale: reinforces YAGNI and keeps expectations explicit. Date/Author: 2025-11-26 / Claude.
- Decision: Use Typer + Rich for the demo CLI now to lock the user interface before models exist. Rationale: gives an immediately runnable command and guides future work on data joins. Date/Author: 2025-11-26 / Claude.
- Decision: Default Makefile `PYTHON` to `python3` because the host environment lacks a `python` shim. Rationale: ensures `make demo` resolves the interpreter without additional setup. Date/Author: 2025-11-26 / Claude.

## Outcomes & Retrospective

Repository skeleton now includes reproducibility artifacts (README, environment file, Makefile), directory READMEs explaining expected contents, YAML configs for both engines, Typer CLI, and stub Python modules with ABOUTME headers. Remaining work involves filling in preprocessing, model code, and exporters, but contributors can already navigate the structure and run `make demo` to see the intended interface.

## Context and Orientation

The repository currently contains meta-docs (`AGENTS.md`, `CLAUDE.md`, `plan.md`, `PLANS.md`) but no runnable code. `plan.md` describes a phased build culminating in two engines plus shared data schema. This plan will add project scaffolding:

- `README.md` at root: one-stop orientation, deterministic setup, make targets, and dataset split policy.
- `environment.yml`: explicit Conda/uv-style environment capturing Python version plus base libs (numpy, pandas, pytorch, pytorch-lightning, pykt, pyarrow, rich, typer).
- `data/` tree with subdirectories (`raw`, `interim`, `processed`) and `data/README.md` describing ingest + versioning contract.
- `src/` packages: `src/common` (schemas, eval, feature utils), `src/wd_irt` (feature pipelines, model driver, exporters), `src/sakt_kt` (pyKT adapters, training CLI). Each package will ship placeholder modules with ABOUTME headers and `NotImplementedError` stubs to show intended responsibilities.
- `configs/` with YAML stubs for `wd_irt_edm.yaml` and `sakt_assist2009.yaml`.
- `reports/README.md` describing metrics/plots deposit.
- `scripts/demo_trace.py` as future CLI entry point, currently echoing TODO but demonstrating interface.
- `Makefile` exposing `data`, `train_wdirt`, `train_sakt`, `export`, and `demo` targets that simply print guidance for now.

## Plan of Work

1. Author `README.md` summarizing the twin-engine vision, setup via `conda env create -f environment.yml`, data split policy, and directory responsibilities. Include Makefile commands and verification instructions.
2. Create `environment.yml` pinning Python 3.11 and baseline libraries (numpy, pandas, pytorch cpu, pytorch-lightning, pyarrow, scipy, typer, rich, pykt). Keep versions realistic but minimal.
3. Add `Makefile` skeleton with phony targets for downloading data, training WD-IRT, training SAKT, exporting outputs, and running the demo script. Each target should call the placeholder Python entrypoints or print TODO statements referencing `scripts/demo_trace.py`.
4. Populate `data/README.md`, `configs/README.md`, `reports/README.md`, and `scripts/README.md` (if needed) describing expectations per directory, dataset references, and naming conventions for artifacts like `item_params.parquet`.
5. Under `src/common`, create modules: `schemas.py`, `features.py`, `evaluation.py`, plus `__init__.py`. Each module begins with the two ABOUTME lines and defines lightweight dataclasses, Protocol stubs, or helper signatures raising `NotImplementedError`.
6. Under `src/wd_irt`, create `__init__.py`, `datasets.py`, `features.py`, `model.py`, `train.py`, `export.py`. Provide placeholder classes/functions referencing the EDM Cup pipeline and hooking into config dataclasses from `src/common`.
7. Under `src/sakt_kt`, create `__init__.py`, `datasets.py`, `train.py`, `export.py`, `adapters.py`. Mirror the placeholder approach referencing pyKT dataset builder and demonstrating how to call pyKT entrypoints later.
8. Define YAML config stubs: `configs/wd_irt_edm.yaml` outlining dataset paths, feature flags, model hyperparameters, and evaluation windows; `configs/sakt_assist2009.yaml` capturing dataset, embedding dim, num heads, dropout, checkpoint path. Comments should explain each field without referencing historical context.
9. Create `scripts/demo_trace.py` using `typer` to declare CLI arguments `student_id`, `topic`, `time_window`. Print stub output referencing the planned joined artifacts.
10. Update ExecPlan `Progress`, `Surprises`, and `Decision Log` as each milestone completes.

## Concrete Steps

1. While in `/Users/nainy/Documents/Personal/deepKT+IRT`, create or edit files via `apply_patch` to keep diffs reviewable.
2. Use `mkdir -p` only when introducing new directory trees; avoid touching hidden system files.
3. For YAML and Markdown files, rely on ASCII-only text.
4. After creating skeleton files, run `ls -R` selectively (e.g., `ls src`, `ls configs`) to verify structure and reference in documentation.

## Validation and Acceptance

Acceptance hinges on documentation and placeholder commands rather than runnable training:

- `conda env create -f environment.yml` should succeed because dependencies are standard (validation deferred until real install).
- Running `make demo student_id=123 topic=fractions` should call the Typer script and print stub text describing the planned join of mastery + item health.
- Directory READMEs must clearly state what artifacts live where and how to name them.
- Source modules should import standard libraries only and raise `NotImplementedError` where logic will eventually reside.

## Idempotence and Recovery

Re-running `make` targets is safe because they only echo instructions. Editing README/config files is additive and reversible. If a placeholder Python module needs to be regenerated, simply overwrite it with the templates described above; there are no migrations or external side effects.

## Artifacts and Notes

Key verification snippet:

    $ ls -R src
    common
    sakt_kt
    wd_irt

    src/common:
    __init__.py
    evaluation.py
    features.py
    schemas.py

    src/sakt_kt:
    __init__.py
    adapters.py
    datasets.py
    export.py
    train.py

    src/wd_irt:
    __init__.py
    datasets.py
    export.py
    features.py
    model.py
    train.py

## Interfaces and Dependencies

- CLI tooling uses `typer` (Typer builds CLIs from Python functions). Typer expects application objects created via `typer.Typer()` and decorated command functions.
- PyTorch and PyTorch Lightning will power both models later; config stubs should expose keys like `optimizer`, `learning_rate`, `max_epochs`.
- pyKT will manage SAKT specifics; placeholder adapters should mention hooking into `pykt.models` once logic is written.
- Use `pyarrow` for parquet outputs to align with plan requirements for `item_params.parquet`, `item_drift.parquet`, and `student_state.parquet`.
