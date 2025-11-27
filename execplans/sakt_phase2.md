# SAKT Knowledge Tracing Engine (Phase 2)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document must be maintained in accordance with `PLANS.md` at the repository root.


## Purpose / Big Picture

After completing this work, the repository will have a functioning SAKT (Self-Attentive Knowledge Tracing) engine that predicts student mastery over skills. Given a sequence of student interactions (which problems they attempted and whether they got them right), the model outputs the probability of answering the next problem correctly. This complements the Wide & Deep IRT engine (Phase 1) which focuses on item-level health — together they form the "twin-engine" learning analytics system.

The user can run training with a single command and obtain two key artifacts: (1) `sakt_student_state.parquet` containing per-student mastery vectors over time, and (2) `sakt_predictions.parquet` with predicted probabilities for each interaction. These will later be joined with item health outputs from Phase 1 to power the demo CLI.


## Progress

- [x] (2025-11-26 23:00Z) Milestone 1: pyKT integration spike — verified pyKT's SAKT model works with ASSISTments data. Achieved 0.573 AUC in 3 epochs.
- [x] (2025-11-27 00:00Z) Milestone 2: Implemented dataset adapter — 17 unit tests passing. Created adapters.py and datasets.py.
- [x] (2025-11-27 00:30Z) Milestone 3: Implemented training CLI with train_sakt(), early stopping, checkpointing, metrics logging.
- [ ] Milestone 4: Implement export functionality for student mastery and predictions
- [ ] Milestone 5: End-to-end validation and documentation


## Surprises & Discoveries

- Observation: pyKT's SAKT forward signature is `forward(q, r, qry)` not `forward(q, c, r)`. The concepts tensor is not used — only questions, responses, and a shifted query sequence.
  Evidence: Inspecting pykt/models/sakt.py line 40: `def forward(self, q, r, qry, qtest=False)`

- Observation: SAKT's embedding index calculation is `x = q + num_c * r`, so if we pass concept IDs where responses belong, indices explode beyond embedding size.
  Evidence: First saw `ind >=0 && ind < ind_dim_size` CUDA assertion failures until we fixed argument order.

- Observation: pyKT expects 1-indexed IDs with 0 reserved for padding. Using 0-indexed IDs caused out-of-bounds errors.
  Evidence: Validation output showed max_q=26688 with num_c=26689 after adding +1 offset.

- Observation: ASSISTments skill_builder data has `skill_ids` mostly empty/unpopulated, so we fall back to using `item_id` as the concept. This results in ~26K concepts instead of the expected ~100 skills.
  Evidence: Spike prints "Warning: No skills found, using item_id as concept"


## Decision Log

- Decision: Use pyKT's built-in SAKT implementation rather than writing custom PyTorch code
  Rationale: pyKT is a mature, well-tested library specifically designed for knowledge tracing. It provides standardized preprocessing, training loops, and evaluation. Reinventing this would be time-consuming and error-prone.
  Date/Author: 2025-11-26 / Initial planning

- Decision: Create shifted query sequence `qryseqs` in our code rather than relying on pyKT to do it
  Rationale: pyKT's SAKT expects `(q, r, qry)` where `qry` is `q` shifted right by one position with zero padding at the start. We must build this ourselves when calling the model directly.
  Date/Author: 2025-11-26 / During spike debugging

- Decision: Use item_id as concept fallback when skill_ids are missing
  Rationale: ASSISTments skill_builder dataset has sparse skill annotations. Using item_id ensures every interaction has a concept, though it increases embedding size significantly (~26K vs ~100).
  Date/Author: 2025-11-26 / During spike implementation


## Outcomes & Retrospective

### Milestone 1 Retrospective (2025-11-26)

Milestone 1 successfully validated pyKT integration. Key learnings:
- SAKT model initializes with ~5.2M parameters for 26K concepts
- Training converges within 3 epochs to ~0.57 AUC (expected baseline for this data)
- Data format conversion is straightforward but requires careful attention to 1-indexing
- The spike code in `scripts/pykt_sakt_spike.py` can serve as reference for production adapter


## Context and Orientation

The repository already has Phase 1 complete: Wide & Deep IRT is trained and exports item parameters. Phase 2 adds the second engine.

Key terms:
- **SAKT (Self-Attentive Knowledge Tracing)**: A transformer-based model that uses self-attention to capture relationships between past interactions and predict future performance. It learns which historical interactions are most relevant for predicting the current question.
- **pyKT**: A Python library (pykt-toolkit) that provides standardized implementations of knowledge tracing models including SAKT, DKT, AKT, and others.
- **Knowledge Component (KC)**: A skill or concept that a problem tests. In ASSISTments, this is called `skill_id`.

Repository structure relevant to this plan:
- `src/sakt_kt/` — Contains stubbed modules for SAKT integration
  - `train.py` — CLI entrypoint (currently raises NotImplementedError)
  - `adapters.py` — Bridges canonical events to pyKT format (stub)
  - `datasets.py` — Prepares sequences for SAKT (stub)
  - `export.py` — Exports student mastery artifacts (stub)
- `configs/sakt_assist2009.yaml` — Training configuration (updated with validated hyperparameters)
- `scripts/pykt_sakt_spike.py` — Working reference implementation from Milestone 1
- `data/raw/assistments_skill_builder/skill_builder_data.csv` — Raw ASSISTments data (~400K interactions)
- `data/interim/assistments_skill_builder_42_events.parquet` — Preprocessed canonical events
- `data/splits/assistments_skill_builder_42.json` — Train/val/test user split

The data pipeline from Phase 0 already normalizes ASSISTments into the canonical `LearningEvent` format with fields: `user_id`, `item_id`, `skill_ids`, `timestamp`, `correct`, `latency_ms`, `help_requested`.


## Plan of Work

### Milestone 2: Dataset Adapter

This milestone extracts the data conversion logic from the spike into a reusable module. The adapter converts canonical events to pyKT's expected format and builds the necessary config dictionaries.

At the end of this milestone, calling `prepare_pykt_data(events_parquet, output_dir)` will produce the correctly formatted CSV and data_config.json that pyKT expects.

Implementation plan:

1. **Create `src/sakt_kt/adapters.py`** with two functions:
   - `canonical_to_pykt_csv()`: Groups events by user, encodes IDs with 1-indexing (0=padding), pads sequences, outputs CSV
   - `build_data_config()`: Returns dict with num_q, num_c, emb_path, etc.

2. **Create `src/sakt_kt/datasets.py`** with:
   - `PyKTDataset` class: Wraps the CSV data, returns tensors for q, r, qry, mask
   - `build_shifted_query()`: Helper to create the shifted question sequence
   - `prepare_dataloaders()`: Creates train/val DataLoaders from config

3. **Add unit tests** in `tests/test_sakt_adapter.py`:
   - Test 1-indexing is applied correctly (min ID = 1, not 0)
   - Test padding with 0 works
   - Test sequences don't exceed max_seq_len
   - Test data_config has required keys

The adapter must handle:
- Empty skill_ids (fallback to item_id as concept)
- Variable-length user sequences (pad to max_seq_len)
- 1-based indexing with 0 reserved for padding
- Building the shifted query sequence for SAKT

### Milestone 3: Training CLI

Wire `src/sakt_kt/train.py` to load config, prepare data via adapter, initialize pyKT SAKT model, run training loop with validation, save checkpoints and metrics.

### Milestone 4: Export Functionality

Implement `src/sakt_kt/export.py` to run inference and save student mastery and prediction artifacts.

### Milestone 5: End-to-End Validation

Full pipeline test, verify AUC > 0.70, update documentation.


## Concrete Steps for Milestone 2

All commands assume working directory is the repository root.

### Step 1: Create adapters.py

Create `src/sakt_kt/adapters.py` with the conversion logic extracted from the spike. The key insight is that SAKT only uses questions (not concepts separately), so we use item_id as the "question" and fall back to item_id as "concept" when skills are missing.

    # File: src/sakt_kt/adapters.py
    # Key functions:
    # - canonical_to_pykt_csv(events_df, output_dir, max_seq_len=200) -> (Path, dict)
    # - build_data_config(num_items, num_concepts) -> dict

### Step 2: Create datasets.py

Create `src/sakt_kt/datasets.py` with PyTorch Dataset and DataLoader creation.

    # File: src/sakt_kt/datasets.py
    # Key classes/functions:
    # - PyKTDataset(csv_path, fold, is_train) -> Dataset returning (qseqs, rseqs, qryseqs, masks)
    # - build_shifted_query(qseqs) -> tensor
    # - prepare_dataloaders(config) -> (train_loader, val_loader)

### Step 3: Add tests

    # Run tests
    uv run pytest tests/test_sakt_adapter.py -v

Expected output:

    tests/test_sakt_adapter.py::test_one_indexing PASSED
    tests/test_sakt_adapter.py::test_padding_with_zero PASSED
    tests/test_sakt_adapter.py::test_max_seq_len_enforced PASSED
    tests/test_sakt_adapter.py::test_data_config_keys PASSED

### Step 4: Verify adapter matches spike

    # Quick verification that adapter produces same output as spike
    python -c "
    from src.sakt_kt.adapters import canonical_to_pykt_csv
    from pathlib import Path
    import pandas as pd
    
    events = pd.read_parquet('data/interim/assistments_skill_builder_42_events.parquet')
    csv_path, config = canonical_to_pykt_csv(events, Path('data/processed/assistments_pykt_test'))
    print(f'CSV: {csv_path}')
    print(f'Config: {config}')
    
    # Verify format
    df = pd.read_csv(csv_path)
    print(f'Rows: {len(df)}, Columns: {list(df.columns)}')
    
    # Check 1-indexing
    qs = [int(x) for x in df.iloc[0].questions.split(',')]
    print(f'First user q range: [{min(qs)}, {max(qs)}]')
    assert min([q for q in qs if q > 0]) >= 1, '1-indexing failed'
    print('✅ Adapter produces correct format')
    "


## Validation and Acceptance for Milestone 2

Milestone 2 is complete when:

1. `src/sakt_kt/adapters.py` exists with `canonical_to_pykt_csv` and `build_data_config` functions
2. `src/sakt_kt/datasets.py` exists with `PyKTDataset` class and `prepare_dataloaders` function
3. All tests in `tests/test_sakt_adapter.py` pass
4. Running the verification snippet above prints "✅ Adapter produces correct format"


## Idempotence and Recovery

Data preparation is idempotent — rerunning overwrites existing files safely.

If tests fail, check:
- Events parquet exists at expected path
- Output directory is writable
- No pandas version incompatibilities


## Interfaces and Dependencies

Dependencies (already in requirements.txt):
- `pykt-toolkit>=0.0.37`
- `torch>=2.0`
- `pandas>=2.0`

Module interfaces after Milestone 2:

In `src/sakt_kt/adapters.py`:

    def canonical_to_pykt_csv(
        events_df: pd.DataFrame,
        output_dir: Path,
        max_seq_len: int = 200
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Convert canonical events to pyKT CSV format.
        
        Returns:
            csv_path: Path to the generated train_valid_sequences.csv
            data_config: Dict with num_q, num_c, emb_path, etc.
        """

    def build_data_config(
        num_questions: int,
        num_concepts: int
    ) -> Dict[str, Any]:
        """Build the data_config dict pyKT expects."""

In `src/sakt_kt/datasets.py`:

    class PyKTDataset(torch.utils.data.Dataset):
        """Dataset that loads pyKT CSV and returns tensors."""
        
        def __init__(self, csv_path: Path, fold: int = 0, is_train: bool = True):
            ...
        
        def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
            """Returns dict with qseqs, rseqs, qryseqs, masks."""

    def build_shifted_query(qseqs: torch.Tensor) -> torch.Tensor:
        """Create shifted query sequence (prepend 0, drop last)."""

    def prepare_dataloaders(
        config: Dict[str, Any],
        csv_path: Path
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation DataLoaders."""


---

## Revision Log

- 2025-11-26 23:30Z: Updated Progress to mark Milestone 1 complete. Added Surprises & Discoveries from spike debugging. Added Decision Log entries. Added Milestone 1 Retrospective. Expanded Milestone 2 plan with concrete implementation details based on spike learnings.
