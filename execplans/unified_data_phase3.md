# Unified Data Layer (Phase 3)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document must be maintained in accordance with `PLANS.md` at the repository root.


## Purpose / Big Picture

After completing this work, both engines (SAKT and Wide & Deep IRT) will operate on the **same dataset** (EDM Cup 2023), enabling meaningful joins between their outputs. Currently:
- Wide & Deep IRT runs on EDM Cup 2023 → `item_params.parquet`, `item_drift.parquet`
- SAKT runs on ASSISTments → `sakt_predictions.parquet`, `sakt_student_state.parquet`

These outputs can't be joined because they cover different students, items, and time periods. Phase 3 bridges this gap by training SAKT on EDM Cup, producing outputs with shared keys (`user_id`, `item_id`, `timestamp`).

The key insight from `plan.md`:
> "This lets both models operate on the same population, same items, same time windows."

After Phase 3, running `demo_trace.py --student-id X --topic Y` will pull both student mastery (SAKT) and item health (WD-IRT) for the same student's actual interactions.


## Progress

- [x] (2025-11-27 02:00Z) Milestone 1: Created `configs/sakt_edm.yaml`, verified data adapter works. EDM Cup: 5.1M events, 35K users, 36K items, vocab=36,801. Ready for training.
- [x] (2025-11-27 14:50Z) Milestone 1: Training complete! Best val_auc=0.696 at epoch 9 (early stopped at 14). Checkpoint synced to local.
- [ ] Milestone 2: Create unified config and export pipeline
- [ ] Milestone 3: Validate output joinability


## Surprises & Discoveries

- Observation: EDM Cup has larger vocabulary than ASSISTments (36,801 vs 26,689 items). This will increase model size but shouldn't affect training convergence.
  Evidence: Data prep output shows `num_q=36801, num_c=36801` for EDM Cup.

- Observation: EDM Cup has better skill coverage (1.0 skills/event vs ASSISTments ~0.0). This means we won't need the item_id fallback, resulting in more meaningful concept embeddings.
  Evidence: `events["skill_ids"].apply(len).mean() = 1.00` for EDM Cup.


## Decision Log

- Decision: Reuse existing SAKT adapter rather than creating EDM-specific code
  Rationale: The adapter in `src/sakt_kt/adapters.py` works on canonical `LearningEvent` format, which both EDM Cup and ASSISTments produce. No EDM-specific logic needed.
  Date/Author: 2025-11-27 / Initial planning


## Outcomes & Retrospective

(To be completed after Phase 3)


## Context and Orientation

### Current State

The repository has two working engines:

1. **Wide & Deep IRT** (`src/wd_irt/`)
   - Trained on: EDM Cup 2023
   - Outputs: `item_params.parquet` (difficulty, guessing by item), `item_drift.parquet`, `behavior_slices.md`
   - Key insight: Learns item-level parameters from clickstream + response data

2. **SAKT** (`src/sakt_kt/`)
   - Trained on: ASSISTments skill_builder
   - Outputs: `sakt_predictions.parquet`, `sakt_student_state.parquet`, `sakt_mastery_report.md`
   - Key insight: Tracks per-student mastery over time

### The Gap

The outputs can't be joined because:
- Different user populations (ASSISTments students vs EDM Cup students)
- Different item pools (~26K ASSISTments problems vs ~3K EDM Cup problems)
- Different time periods (2009 vs 2023)

### The Solution

Train SAKT on EDM Cup canonical events. The data pipeline already produces `data/interim/edm_cup_2023_42_events.parquet` in the same `LearningEvent` format as ASSISTments. The SAKT adapter (`canonical_to_pykt_csv`) doesn't care about the source dataset—it just needs the canonical columns.

### Key Files

| File | Purpose |
|------|---------|
| `data/interim/edm_cup_2023_42_events.parquet` | EDM Cup in canonical format |
| `data/splits/edm_cup_2023_42.json` | User splits for EDM Cup |
| `src/sakt_kt/adapters.py` | Converts canonical → pyKT (dataset-agnostic) |
| `configs/sakt_assist2009.yaml` | Current SAKT config (ASSISTments-focused) |


## Plan of Work

### Milestone 1: Verify SAKT Trains on EDM Cup

Confirm that the existing SAKT pipeline can train on EDM Cup data without modification.

**Steps:**

1. Create `configs/sakt_edm.yaml` pointing to EDM Cup data paths
2. Run SAKT training: `python -m src.sakt_kt.train --config configs/sakt_edm.yaml`
3. Verify training completes and produces reasonable AUC (>0.55 baseline)
4. Check that vocabulary size is sensible (~3K items for EDM vs ~26K for ASSISTments)

**Expected Issues:**
- EDM Cup has richer skill annotations than ASSISTments—may affect num_c vs num_q
- Sequence lengths may differ (EDM Cup is unit-based assignments)

**Success Criteria:**
- Training runs without errors
- Validation AUC > 0.55
- Checkpoint saved to `reports/checkpoints/sakt_edm/`

### Milestone 2: Create Unified Config and Export Pipeline

Ensure both engines produce outputs with consistent schemas that can be joined.

**Steps:**

1. Run WD-IRT export: `python -m src.wd_irt.train export --config configs/wd_irt_edm.yaml`
2. Run SAKT export: `python -m src.sakt_kt.train export --config configs/sakt_edm.yaml`
3. Verify output schemas match expected join keys

**Output Schema Alignment:**

| Artifact | SAKT | WD-IRT | Join Key |
|----------|------|--------|----------|
| Student mastery | `sakt_student_state.parquet` | N/A | `user_id` |
| Predictions | `sakt_predictions.parquet` | N/A | `user_id`, `item_id` |
| Item params | N/A | `item_params.parquet` | `item_id` |
| Item drift | N/A | `item_drift.parquet` | `item_id` |

**Success Criteria:**
- Both exports complete without errors
- Outputs contain overlapping `user_id` and `item_id` values
- Timestamps are in same timezone (UTC)

### Milestone 3: Validate Output Joinability

Create a validation script that proves the outputs can be meaningfully joined.

**Steps:**

1. Create `scripts/validate_join.py` that:
   - Loads all four parquet files
   - Computes join coverage (% of SAKT predictions with matching WD-IRT item params)
   - Samples 10 random students and shows their mastery + item health for items attempted
   - Reports any schema mismatches

2. Run validation: `python scripts/validate_join.py --reports-dir reports`

3. Update `demo_trace.py` to actually load and join the data (replace placeholder logic)

**Success Criteria:**
- Join coverage > 80% (most items in SAKT predictions have WD-IRT params)
- Sample output shows meaningful combinations
- `demo_trace.py` produces real output instead of placeholder text


## Concrete Steps for Milestone 1

### Step 1: Create EDM SAKT Config

Create `configs/sakt_edm.yaml`:

```yaml
# SAKT configuration for EDM Cup 2023 dataset
run_name: sakt_edm
seed: 42

data:
  raw_dir: data/raw/edm_cup_2023
  events_path: data/interim/edm_cup_2023_42_events.parquet
  splits_path: data/splits/edm_cup_2023_42.json
  pykt_dir: data/processed/edm_pykt

model:
  emb_size: 64
  num_attn_heads: 4
  dropout: 0.2
  seq_len: 200

training:
  batch_size: 64
  learning_rate: 0.001
  max_epochs: 30
  accelerator: auto
  early_stopping_patience: 5
  early_stopping_metric: val_auc
  early_stopping_mode: max

outputs:
  checkpoint_dir: reports/checkpoints/sakt_edm
  metrics_dir: reports/metrics
  predictions_path: reports/sakt_edm_predictions.parquet
  student_state_path: reports/sakt_edm_student_state.parquet
```

### Step 2: Verify Data Stats

Run a quick check to understand EDM Cup data characteristics:

```bash
python -c "
import pandas as pd
events = pd.read_parquet('data/interim/edm_cup_2023_42_events.parquet')
print(f'Events: {len(events):,}')
print(f'Users: {events[\"user_id\"].nunique():,}')
print(f'Items: {events[\"item_id\"].nunique():,}')
print(f'Skills coverage: {events[\"skill_ids\"].apply(len).mean():.2f} skills/event')
print(f'Avg events/user: {len(events) / events[\"user_id\"].nunique():.1f}')
"
```

### Step 3: Train SAKT on EDM Cup

```bash
python -m src.sakt_kt.train --config configs/sakt_edm.yaml
```

Monitor for:
- Vocabulary size (num_q, num_c) in logs
- Training loss convergence
- Validation AUC improvement

### Step 4: Export and Verify

```bash
python -m src.sakt_kt.train export \
    --checkpoint reports/checkpoints/sakt_edm/sakt_edm_seed42_best.pt \
    --config configs/sakt_edm.yaml \
    --output-dir reports
```


## Validation and Acceptance

Milestone 1 is complete when:
- `configs/sakt_edm.yaml` exists and is valid
- SAKT training on EDM Cup completes without errors
- Validation AUC > 0.55 (baseline performance)
- Checkpoint exists at `reports/checkpoints/sakt_edm/`

Milestone 2 is complete when:
- Both WD-IRT and SAKT exports exist in `reports/`
- Output files have compatible schemas

Milestone 3 is complete when:
- `scripts/validate_join.py` shows >80% join coverage
- `demo_trace.py` produces real output for a valid student/topic


## Idempotence and Recovery

All data preparation is idempotent. Training checkpoints are timestamped. If training fails:
- Check EDM Cup events parquet exists and has expected columns
- Verify SAKT adapter handles EDM Cup's skill_ids format
- Check GPU memory if running on Lightning AI


## Interfaces and Dependencies

No new dependencies required. Uses existing:
- `src/sakt_kt/adapters.py` (canonical → pyKT)
- `src/sakt_kt/train.py` (training CLI)
- `src/sakt_kt/export.py` (artifact generation)

New files to create:
- `configs/sakt_edm.yaml`
- `scripts/validate_join.py`


## Timeline Estimate

- Milestone 1: ~1 hour (config + training + verification)
- Milestone 2: ~30 minutes (run exports, check schemas)
- Milestone 3: ~1 hour (validation script + demo update)

Total: ~2.5 hours


---

## Revision Log

- 2025-11-27: Initial draft based on plan.md Phase 3 requirements

