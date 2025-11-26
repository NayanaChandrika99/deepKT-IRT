# SAKT Knowledge Tracing Engine (Phase 2)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document must be maintained in accordance with `PLANS.md` at the repository root.


## Purpose / Big Picture

After completing this work, the repository will have a functioning SAKT (Self-Attentive Knowledge Tracing) engine that predicts student mastery over skills. Given a sequence of student interactions (which problems they attempted and whether they got them right), the model outputs the probability of answering the next problem correctly. This complements the Wide & Deep IRT engine (Phase 1) which focuses on item-level health — together they form the "twin-engine" learning analytics system.

The user can run training with a single command and obtain two key artifacts: (1) `student_state.parquet` containing per-student mastery vectors over time, and (2) `next_correct_predictions.parquet` with predicted probabilities for each interaction. These will later be joined with item health outputs from Phase 1 to power the demo CLI.


## Progress

- [ ] Milestone 1: pyKT integration spike — verify pyKT's SAKT model works with ASSISTments data
- [ ] Milestone 2: Implement dataset adapter to convert canonical events to pyKT format
- [ ] Milestone 3: Wire training CLI (`src/sakt_kt/train.py`) to pyKT's training loop
- [ ] Milestone 4: Implement export functionality for student mastery and predictions
- [ ] Milestone 5: End-to-end validation and documentation


## Surprises & Discoveries

(To be populated during implementation)


## Decision Log

- Decision: Use pyKT's built-in SAKT implementation rather than writing custom PyTorch code
  Rationale: pyKT is a mature, well-tested library specifically designed for knowledge tracing. It provides standardized preprocessing, training loops, and evaluation. Reinventing this would be time-consuming and error-prone.
  Date/Author: 2025-11-26 / Initial planning


## Outcomes & Retrospective

(To be completed after implementation)


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
- `configs/sakt_assist2009.yaml` — Training configuration (already exists)
- `data/raw/assistments_skill_builder/skill_builder_data.csv` — Raw ASSISTments data (~400K interactions)
- `data/interim/assistments_skill_builder_42_events.parquet` — Preprocessed canonical events
- `data/splits/assistments_skill_builder_42.json` — Train/val/test user split

The data pipeline from Phase 0 already normalizes ASSISTments into the canonical `LearningEvent` format with fields: `user_id`, `item_id`, `skill_ids`, `timestamp`, `correct`, `latency_ms`, `help_requested`.


## Plan of Work

### Milestone 1: pyKT Integration Spike

Before writing production code, verify that pyKT's SAKT model works with our data format. This milestone produces a standalone script that:
1. Converts our canonical events to pyKT's expected format
2. Trains a SAKT model for a few epochs
3. Gets predictions and AUC metrics

The spike lives in `scripts/pykt_sakt_spike.py` and will be discarded or refactored after proving feasibility.

pyKT expects data in a specific CSV format with columns: `uid`, `questions`, `concepts`, `responses`, `timestamps` where each row is a user and the columns contain comma-separated sequences. The preprocessing step must:
1. Group events by user
2. Order by timestamp
3. Encode skill_ids as concept IDs
4. Encode problem_ids as question IDs
5. Format as comma-separated strings

### Milestone 2: Dataset Adapter

Implement `src/sakt_kt/adapters.py` with functions:
- `canonical_to_pykt_format(events_df, output_path)`: Converts canonical events DataFrame to pyKT's CSV format
- `build_pykt_data_config(num_questions, num_concepts)`: Creates the data_config dict pyKT expects

Implement `src/sakt_kt/datasets.py` with:
- `prepare_sakt_dataset(config_path)`: Orchestrates data preparation from config

The pyKT data_config needs these fields:
    
    {
        "num_q": <number of unique questions>,
        "num_c": <number of unique concepts/skills>,
        "max_concepts": 1,  # concepts per question (1 for ASSISTments)
        "input_type": ["questions", "concepts"]
    }

### Milestone 3: Training CLI

Wire `src/sakt_kt/train.py` to:
1. Parse the YAML config
2. Call adapter to prepare data in pyKT format
3. Initialize pyKT's SAKT model using `init_model`
4. Initialize dataloaders using `init_dataset4train`
5. Set up optimizer and train using `train_model`
6. Save checkpoints to `reports/checkpoints/sakt_assist2009/`
7. Log metrics to `reports/metrics/sakt_assist2009_metrics.json`

The training loop uses pyKT's built-in `train_model` function which handles epochs, validation, early stopping, and checkpointing.

### Milestone 4: Export Functionality

Implement `src/sakt_kt/export.py` with:
- `export_student_mastery(model, events_df, output_path)`: Run inference to get hidden states representing student knowledge, save as `student_state.parquet`
- `export_predictions(model, events_df, output_path)`: Run inference to get predicted probabilities for each interaction, save as `next_correct_predictions.parquet`

The student_state.parquet schema:
    
    user_id: string
    skill_id: string
    timestamp: datetime
    mastery: float (0-1, predicted probability of correct on this skill)

The predictions.parquet schema:
    
    user_id: string
    item_id: string
    timestamp: datetime
    actual: int (0 or 1)
    predicted: float (probability 0-1)

### Milestone 5: End-to-End Validation

1. Run the full pipeline on ASSISTments data
2. Verify AUC is reasonable (>0.7 expected for SAKT on ASSISTments)
3. Verify exported artifacts have correct schemas
4. Update README with SAKT training instructions
5. Update execplan with final metrics and lessons learned


## Concrete Steps

All commands assume working directory is the repository root.

### Milestone 1 Commands

    # Ensure ASSISTments data is preprocessed (already done in Phase 0)
    make data dataset=assistments_skill_builder split_seed=42

    # Create and run the spike script
    python scripts/pykt_sakt_spike.py

Expected output:

    Loading canonical events from data/interim/assistments_skill_builder_42_events.parquet
    Converting to pyKT format...
    Saved pyKT format to data/processed/assistments_pykt/
    Initializing SAKT model...
    Training for 3 epochs...
    Epoch 1: train_loss=0.XXX, val_auc=0.XXX
    Epoch 2: train_loss=0.XXX, val_auc=0.XXX
    Epoch 3: train_loss=0.XXX, val_auc=0.XXX
    ✅ Spike successful! SAKT integration verified.

### Milestone 3 Commands

    # Train SAKT
    python -m src.sakt_kt.train --config configs/sakt_assist2009.yaml

Expected output:

    [sakt] Loading config from configs/sakt_assist2009.yaml
    [sakt] Preparing dataset...
    [sakt] Initializing SAKT model...
    [sakt] Training for 30 epochs...
    Epoch 1: train_loss=X.XXX, val_auc=0.XXX
    ...
    Epoch 30: train_loss=X.XXX, val_auc=0.XXX
    [sakt] Best validation AUC: 0.XXX at epoch N
    [sakt] Checkpoint saved to reports/checkpoints/sakt_assist2009/
    [sakt] Metrics saved to reports/metrics/sakt_assist2009_metrics.json

### Milestone 4 Commands

    # Export student mastery and predictions
    python -m src.sakt_kt.train export \
        --checkpoint reports/checkpoints/sakt_assist2009/best.ckpt \
        --config configs/sakt_assist2009.yaml \
        --output-dir reports

Expected output:

    ✅ Exported student mastery to reports/student_state.parquet (N students, M skills)
    ✅ Exported predictions to reports/next_correct_predictions.parquet (K interactions)


## Validation and Acceptance

The implementation is complete when:

1. `python -m src.sakt_kt.train --config configs/sakt_assist2009.yaml` completes without errors and achieves validation AUC > 0.70

2. `reports/student_state.parquet` exists with columns: user_id, skill_id, timestamp, mastery

3. `reports/next_correct_predictions.parquet` exists with columns: user_id, item_id, timestamp, actual, predicted

4. Running verification:

        python -c "import pandas as pd; \
            sm = pd.read_parquet('reports/student_state.parquet'); \
            print(f'Student mastery: {len(sm)} rows, {sm.user_id.nunique()} users'); \
            pred = pd.read_parquet('reports/next_correct_predictions.parquet'); \
            print(f'Predictions: {len(pred)} rows'); \
            from sklearn.metrics import roc_auc_score; \
            auc = roc_auc_score(pred.actual, pred.predicted); \
            print(f'Prediction AUC: {auc:.4f}')"

   Should output reasonable numbers (thousands of rows, AUC > 0.7)


## Idempotence and Recovery

All data preparation steps are idempotent — rerunning `make data` overwrites existing files safely.

Training can be resumed from checkpoints if interrupted. pyKT's `train_model` function saves the best model automatically.

If the pyKT installation fails, try:

    pip install pykt-toolkit --upgrade

If CUDA issues occur, the config has `trainer_accelerator: cpu` as fallback.


## Artifacts and Notes

### pyKT Data Format

pyKT expects a specific CSV format. Example of the required format:

    uid,questions,concepts,responses,timestamps
    1,"101,102,103,104","1,2,1,3","1,0,1,1","0,1,2,3"
    2,"101,105,102","1,4,2","1,1,0","0,1,2"

Each row is one user. The columns are comma-separated sequences of equal length representing the user's interaction history.

### pyKT Config Structure

pyKT needs two config dictionaries:

1. data_config (per dataset):

        {
            "num_q": 1234,       # unique questions
            "num_c": 100,        # unique concepts/skills
            "max_concepts": 1,   # concepts per question
            "input_type": ["questions", "concepts"]
        }

2. model_config (for SAKT):

        {
            "emb_size": 128,
            "num_attn_heads": 4,
            "dropout": 0.2
        }


## Interfaces and Dependencies

Dependencies:
- `pykt-toolkit>=0.0.37` (already in requirements.txt)
- PyTorch (already installed)

Key pyKT imports:

    from pykt.models import init_model, train_model
    from pykt.datasets import init_dataset4train
    from pykt.utils import set_seed

Module interfaces after implementation:

In `src/sakt_kt/adapters.py`:

    def canonical_to_pykt_format(
        events_df: pd.DataFrame,
        output_dir: Path,
        max_seq_len: int = 200
    ) -> Tuple[Path, Dict[str, int]]:
        """
        Convert canonical events to pyKT CSV format.
        Returns (csv_path, data_config_dict).
        """

    def build_pykt_data_config(
        num_questions: int,
        num_concepts: int,
        max_concepts: int = 1
    ) -> Dict[str, Any]:
        """Build the data_config dict pyKT expects."""

In `src/sakt_kt/train.py`:

    def train_sakt(config_path: Path) -> None:
        """Train SAKT model from config YAML."""

    def export_sakt(
        checkpoint: Path,
        config_path: Path,
        output_dir: Path
    ) -> None:
        """Export student mastery and predictions."""

In `src/sakt_kt/export.py`:

    def export_student_mastery(
        model: nn.Module,
        events_df: pd.DataFrame,
        output_path: Path
    ) -> None:
        """Export per-student, per-skill mastery estimates."""

    def export_predictions(
        model: nn.Module,
        events_df: pd.DataFrame,
        output_path: Path
    ) -> None:
        """Export predicted probabilities for each interaction."""

