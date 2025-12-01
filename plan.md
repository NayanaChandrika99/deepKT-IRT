Here’s a clean build plan for the **two-engine system** (no Streamlit/UI yet): **SAKT “Student Readiness”** + **Wide & Deep IRT “Item Health”**, based directly on the Wide&Deep IRT paper/code and pyKT’s SAKT support. ([jedm.educationaldatamining.org][1])

---

## Status Snapshot (2025‑11‑28)

- ✅ Canonical ingestion pipeline, SAKT, and Wide & Deep IRT engines are implemented and covered by tests (`src/common`, `src/sakt_kt`, `src/wd_irt`, `tests/`).
- ✅ Demo + analytics surfaces (`scripts/demo_trace.py`, explainability, RL bandit, gaming detection) are wired to the exported parquet artifacts.
- ✅ Training runs were executed on Lightning.ai remote GPUs; resulting checkpoints are committed so local demos don’t require retraining unless desired:
  - `reports/checkpoints/sakt_edm/sakt_edm_seed42_best.pt`
  - `reports/checkpoints/sakt_assist2009/sakt_assist2009_seed42_best.pt`
  - `reports/checkpoints/wd_irt_edm/latest.ckpt` (replace with the newest Lightning.ai export when retraining)
- ✅ Exports (`reports/sakt_*.parquet`, `reports/item_*.parquet`, `reports/behavior_slices.md`) are generated from those checkpoints and stay in sync with the demo CLI.

---

## Phase 0 — Repo + reproducibility (set the project up like an MLE would)

**Artifacts**

* `README.md` with one-command runs (train/eval/export).
* `environment.yml` (or `uv.lock/poetry.lock`) + deterministic seeds.
* Data versioning notes (raw vs processed), and a fixed train/val/test split protocol.

**Suggested structure**

* `data/` (download scripts + raw pointers, not giant files)
* `src/common/` (schemas, feature utils, eval utils)
* `src/wd_irt/` (Wide&Deep IRT pipeline)
* `src/sakt_kt/` (pyKT + dataset adapter + training)
* `reports/` (auto-generated metrics + plots)
* `configs/` (YAML configs for runs)

---

## Phase 1 — Wide & Deep IRT engine (reproduce paper behavior + outputs)

You want to **first replicate the authors’ results** on the *EDM Cup 2023 clickstream dataset*, because the model’s whole point is leveraging clickstream to predict performance and expose behavior patterns. ([jedm.educationaldatamining.org][1])

### 1.1 Get the reference implementation + dataset

* Download the **Wide & Deep IRT paper** and the **OSF code** linked by the authors. The paper explicitly states the source code is on OSF and the experiments use EDM Cup 2023 clickstream. ([jedm.educationaldatamining.org][1])
* Pull **EDM Cup 2023** data (hosted via Kaggle; the EDM page describes the task and that it uses ASSISTments clickstream across in-unit assignments to predict end-of-unit outcomes). ([educationaldatamining.org][2])

### 1.2 Implement the pipeline exactly as the paper describes

* **Feature engineering**: implement “student action / clickstream” features the way the paper does (they have a dedicated feature engineering section starting from clickstream “student actions”). ([jedm.educationaldatamining.org][1])
* **Model training**: Wide component is IRT-like; deep component consumes clickstream-derived features; output is correctness probability and an ability estimate. ([jedm.educationaldatamining.org][1])
* **Evaluation**: compute AUC (they report test AUC and compare vs IRT + KT baselines). ([jedm.educationaldatamining.org][1])

### 1.3 Produce the “Item Health Lab” outputs (no UI)

Generate machine-readable outputs you’ll later wire into a UI (already in place via the Lightning.ai training run, see `reports/item_params.parquet`, `reports/item_drift.parquet`, and `reports/behavior_slices.md`):

* `item_params.parquet`: item difficulty/discrimination/guessing-like parameters (whatever the reference code exports)
* `item_drift.parquet`: drift flags across time windows (weekly/monthly)
* `behavior_slices.md`: behavior insights by ability group (the paper analyzes patterns like “request answer rapidly” vs ability). ([jedm.educationaldatamining.org][1])

---

## Phase 2 — SAKT engine (via pyKT) on a known benchmark first

Before adapting to EDM Cup, start with a “boring but reliable” KT benchmark to prove your training/eval loop works.

### 2.1 Run SAKT using pyKT’s standard pipeline

* pyKT is explicitly designed to standardize preprocessing across common KT datasets and includes SAKT. ([pykt-toolkit.readthedocs.io][3])
* Use **ASSISTments2009** (pyKT documents it and links the dataset). ([pykt-toolkit.readthedocs.io][4])
* Train SAKT and export:

  * `student_state.parquet`: per-student mastery vectors over time (skill/topic-level)
  * `next_correct_predictions.parquet`: per interaction predicted probability
    SAKT is defined as a self-attention KT model capturing relevance between knowledge components and past interactions. ([pykt-toolkit.readthedocs.io][5])

**Artifacts**

* `train_sakt.py` (wrap pyKT config + logging)
* `eval_sakt.py` (AUC, calibration plots, subgroup metrics)
* Lightning.ai job artifacts synced to `reports/checkpoints/sakt_*/*.pt`, enabling exports without re-running training locally.

---

## Phase 3 — Unify the data (so both engines run on the same reality)

This is the key step that makes your “symbiotic ecosystem” pitch real.

### 3.1 Define a shared “Learning Event” schema

Create one canonical event format:

* `user_id, item_id, skill_id(s), timestamp, correctness`
* clickstream-derived fields: `action_sequence_id`, `time_to_first_action`, `time_to_answer_request`, etc.

EDM Cup 2023 is explicitly clickstream-heavy and includes curriculum/assignment/problem context—perfect for this shared schema. ([educationaldatamining.org][2])

### 3.2 Build an EDM Cup → KT adapter

Transform EDM Cup logs into the standard KT interaction sequence format (user-ordered attempts with skill tags), so you can train **SAKT on EDM Cup** too (not just ASSIST2009). This lets both models operate on the same population, same items, same time windows.

---

## Phase 4 — Integration without UI: build the “join” artifacts + demo script

Now create the “wow” connection as *data products* first:

### 4.1 Join outputs

* `student_topic_mastery.parquet` from SAKT
* `item_health.parquet` from Wide&Deep IRT
* Join key: `(skill/topic, item_id)` plus time window

### 4.2 “One command demo”

A CLI script like:

* `python demo_trace.py --student_id 123 --topic fractions`
  Outputs:
* Top weak skills (SAKT)
* Items recommended next (SAKT)
* For the items the student struggled on: item parameters + drift flags + suspicious clickstream patterns (Wide&Deep IRT)

This demonstrates the ecosystem *without* building a frontend yet.

---

## Phase 5 — MLE-grade polish (what makes this interview-ready)

**Minimum set of engineering artifacts**

* `Makefile` / `justfile` with:

  * `make data`
  * `make train_wdirt`
  * `make train_sakt`
  * `make export`
* Unit tests for:

  * schema validation
  * sequence building determinism
  * feature extraction correctness
* Model cards:

  * what task each model was trained for (KT vs post-test prediction) and limits (distribution shift, missing clickstream, etc.)

---

If you want, I can draft the exact **repo skeleton + command interface** (filenames, CLI arguments, and what each script outputs) so you can start implementing immediately from those two references.

[1]: https://jedm.educationaldatamining.org/index.php/JEDM/article/download/677/211 "Predicting Students’ Future Success: Harnessing Clickstream Data with Wide & Deep Item Response Theory"
[2]: https://educationaldatamining.org/edm2023/edm-cup-2023/ "
  EDM Cup 2023:
Educational Data Mining 2024-
"
[3]: https://pykt-toolkit.readthedocs.io/?utm_source=chatgpt.com "Welcome to pyKT's documentation! — pykt-toolkit 0.0.37 ..."
[4]: https://pykt-toolkit.readthedocs.io/en/latest/datasets.html "Datasets — pykt-toolkit 0.0.37 documentation"
[5]: https://pykt-toolkit.readthedocs.io/en/latest/models.html "Models — pykt-toolkit 0.0.37 documentation"
