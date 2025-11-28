# ABOUTME: Introduces the twin-engine learning analytics project and how to run it.
# ABOUTME: Documents setup, directory layout, and commands for both SAKT and WD-IRT workflows.

# deepKT + Wide&Deep IRT

This repository implements two complementary analytics engines for educational data:

1. **Student Readiness (SAKT)** — Sequential Attention Knowledge Tracing via pyKT. Predicts student mastery over skills based on interaction history.
2. **Item Health (Wide & Deep IRT)** — Fuses clickstream behavior with psychometrics to estimate item difficulty, discrimination, and guessing parameters.

Both engines consume the same canonical learning-event schema so their outputs can be joined for comprehensive learning analytics.

## Quick Start

### Environment Setup

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate deepkt
```

### Data Preparation

```bash
# Prepare EDM Cup 2023 data
make data dataset=edm_cup_2023 split_seed=42

# Prepare ASSISTments data
make data dataset=assistments_skill_builder split_seed=42
```

### Training

#### SAKT (Student Readiness)

```bash
# Train SAKT model
python -m src.sakt_kt.train --config configs/sakt_assist2009.yaml

# Export student mastery and predictions
python -m src.sakt_kt.train export \
    --checkpoint reports/checkpoints/sakt_assist2009/sakt_assist2009_seed42_best.pt \
    --config configs/sakt_assist2009.yaml \
    --output-dir reports
```

**Expected outputs:**
- `reports/sakt_predictions.parquet` — Predicted vs actual correctness
- `reports/sakt_student_state.parquet` — Per-interaction mastery estimates
- `reports/sakt_attention.parquet` — Attention weights for explainability
- `reports/sakt_mastery_report.md` — Summary statistics

**Validated performance:** AUC 0.74 on ASSISTments skill_builder dataset.

#### Wide & Deep IRT (Item Health)

```bash
# Train Wide & Deep IRT model
python -m src.wd_irt.train --config configs/wd_irt_edm.yaml

# Export item parameters
python -m src.wd_irt.train export \
    --checkpoint reports/checkpoints/wd_irt_edm/best.ckpt \
    --config configs/wd_irt_edm.yaml \
    --output-dir reports
```

**Expected outputs:**
- `reports/item_params.parquet` — Item difficulty, discrimination, guessing
- `reports/item_drift.parquet` — Temporal drift flags
- `reports/behavior_slices.md` — Item health summary by topic

## Directory Layout

```
.
├── configs/           # YAML configs for WD-IRT and SAKT runs
├── data/
│   ├── raw/           # Immutable source data (gitignored)
│   ├── interim/       # Canonical learning events
│   ├── processed/     # Model-ready features (pyKT format, etc.)
│   └── splits/        # Train/val/test user split manifests
├── execplans/         # ExecPlan documents per PLANS.md
├── reports/           # Generated metrics, checkpoints, exports
├── scripts/           # Utility scripts (spike tests, demos)
├── src/
│   ├── common/        # Shared schemas, data pipeline
│   ├── sakt_kt/       # SAKT training via pyKT
│   └── wd_irt/        # Wide & Deep IRT models
└── tests/             # Unit tests
```

## Key Modules

### SAKT Engine (`src/sakt_kt/`)

| Module | Purpose |
|--------|---------|
| `adapters.py` | Converts canonical events to pyKT CSV format |
| `datasets.py` | PyTorch Dataset and DataLoader utilities |
| `train.py` | CLI for training and export |
| `export.py` | Generates student mastery and prediction artifacts |

### Wide & Deep IRT Engine (`src/wd_irt/`)

| Module | Purpose |
|--------|---------|
| `features.py` | Clickstream feature engineering |
| `datasets.py` | PyTorch Dataset for EDM data |
| `model.py` | Wide & Deep IRT architecture |
| `train.py` | CLI for training and export |
| `export.py` | Generates item health artifacts |

## Data Schema

Both engines use the canonical `LearningEvent` schema defined in `src/common/schemas.py`:

```python
@dataclass
class LearningEvent:
    user_id: str
    item_id: str
    skill_ids: List[str]
    timestamp: datetime
    correct: bool
    action_sequence_id: Optional[str]
    latency_ms: Optional[int]
    help_requested: Optional[bool]
```

## Configuration

### SAKT Config (`configs/sakt_assist2009.yaml`)

Key hyperparameters:
- `emb_size: 64` — Embedding dimension
- `num_attn_heads: 4` — Attention heads
- `seq_len: 200` — Maximum sequence length
- `learning_rate: 0.001`
- `early_stopping_patience: 5`

### WD-IRT Config (`configs/wd_irt_edm.yaml`)

Key hyperparameters:
- `wide_units: 256` — Wide component size
- `deep_units: [512, 256, 128]` — Deep MLP layers
- `embedding_dim: 128`

## Running Tests

```bash
# Run all tests
PYTHONPATH=. uv run pytest tests/ -v

# Run SAKT adapter tests
PYTHONPATH=. uv run pytest tests/test_sakt_adapter.py -v
```

## Development

This project follows the ExecPlan methodology documented in `PLANS.md`. Each major feature has an associated execution plan in `execplans/`.

### Demo (Phase 4)

Generate skill mastery (if not already present) and emit recommendations by joining SAKT + WD-IRT outputs:

```bash
source .venv/bin/activate
python scripts/demo_trace.py --student-id <user> --topic <skill_code> --time-window <window>
```

Requirements: `reports/sakt_student_state.parquet`, `data/interim/...events.parquet`, and `reports/item_params.parquet` (with optional `reports/item_drift.parquet`). The CLI writes `reports/skill_mastery.parquet` on first run.

### Explainability & Gaming (Phase 5A)

Explain a student's mastery and surface gaming alerts:

```bash
python scripts/demo_trace.py explain --user-id <user> --skill <skill_code>
python scripts/demo_trace.py gaming-check --user-id <user>  # or omit for all students
```

Outputs rely on the same artifacts as the demo; attention parquet is optional (explanations degrade gracefully without it).

### Current Status

- ✅ **SAKT Engine** — Complete (training, export, 0.74 AUC)
- ✅ **Wide & Deep IRT** — Complete (training, export)
- ✅ **Demo CLI (Phase 4)** — Joins both engines' outputs with recommendations
- ✅ **Explainability & Gaming (Phase 5A)** — Attention-based explanations and behavioral alerts

## License

MIT
