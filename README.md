# ABOUTME: Introduces the twin-engine learning analytics project and how to run it.
# ABOUTME: Documents setup, directory layout, and commands for both SAKT and WD-IRT workflows.

# deepKT + Wide&Deep IRT

Twin-engine learning analytics combining knowledge tracing and item response theory:

1. **SAKT (Student Readiness)** — Sequential Attention Knowledge Tracing predicting student mastery over time
2. **Wide & Deep IRT (Item Health)** — Fuses clickstream behavior with psychometrics to estimate item difficulty, discrimination, and guessing

Both engines share a canonical learning-event schema enabling comprehensive joint analytics.

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
# Train and export (local or remote GPU)
make train_sakt SAKT_CONFIG=configs/sakt_edm.yaml
make export_sakt \
    SAKT_CONFIG=configs/sakt_edm.yaml \
    SAKT_CHECKPOINT=reports/checkpoints/sakt_edm/sakt_edm_seed42_best.pt
```

**Outputs:**
- `reports/sakt_predictions.parquet` — Predicted vs actual correctness
- `reports/sakt_student_state.parquet` — Per-interaction mastery estimates
- `reports/sakt_attention.parquet` — Attention weights for explainability
- `reports/sakt_mastery_report.md` — Summary statistics

**Performance:** AUC 0.74 on EDM Cup 2023 (checkpoint trained on Lightning.ai A100)

#### Wide & Deep IRT (Item Health)

```bash
# Train and export
make train_wdirt WD_CONFIG=configs/wd_irt_edm.yaml
make export_wdirt \
    WD_CONFIG=configs/wd_irt_edm.yaml \
    WD_CHECKPOINT=reports/checkpoints/wd_irt_edm/latest.ckpt
```

**Outputs:**
- `reports/item_params.parquet` — Difficulty, discrimination, guessing parameters
- `reports/item_drift.parquet` — Temporal drift flags
- `reports/behavior_slices.md` — Item health by topic

### Pretrained Checkpoints

| Engine | Dataset | Path |
|--------|---------|------|
| SAKT | EDM Cup 2023 | `reports/checkpoints/sakt_edm/sakt_edm_seed42_best.pt` |
| SAKT | ASSISTments 2009 | `reports/checkpoints/sakt_assist2009/sakt_assist2009_seed42_best.pt` |
| Wide & Deep IRT | EDM Cup 2023 | `reports/checkpoints/wd_irt_edm/latest.ckpt` |

Checkpoints trained on Lightning.ai (A100). Use `make export_*` to regenerate artifacts without retraining.

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

Hyperparameters are defined in `configs/`:

**SAKT** (`sakt_*.yaml`): `emb_size: 64`, `num_attn_heads: 4`, `seq_len: 200`, `learning_rate: 0.001`

**WD-IRT** (`wd_irt_*.yaml`): `wide_units: 256`, `deep_units: [512, 256, 128]`, `embedding_dim: 128`

## Running Tests

```bash
# Run all tests
PYTHONPATH=. uv run pytest tests/ -v

# Run SAKT adapter tests
PYTHONPATH=. uv run pytest tests/test_sakt_adapter.py -v
```

## Usage

### Demo & Recommendations

Generate recommendations by joining SAKT + WD-IRT outputs:

```bash
# Basic recommendations
python scripts/demo_trace.py --student-id <user> --topic <skill> --time-window <window>

# With RL bandit (LinUCB)
python scripts/demo_trace.py --student-id <user> --topic <skill> --time-window <window> --use-rl

# Explainability
python scripts/demo_trace.py explain --user-id <user> --skill <skill>

# Gaming detection
python scripts/demo_trace.py gaming-check --user-id <user>
```

**Requirements:** `reports/sakt_student_state.parquet`, `data/interim/*_events.parquet`, `reports/item_params.parquet`

### Explainability & Gaming Detection

**Optional LLM Enhancement**

Enable LLM-generated explanations for more natural language (default uses templates):

```bash
export USE_LLM_EXPLANATIONS=true
export LLM_PROVIDER=openai  # or "anthropic"
export LLM_MODEL=gpt-4o-mini  # or "claude-3-haiku-20240307"
export OPENAI_API_KEY="sk-..."

python scripts/demo_trace.py explain --user-id <user> --skill <skill>
```

Cost: ~$0.00015-0.00025 per explanation. Generates mastery insights, recommendation reasons, and alerts.

### RL Bandit

LinUCB contextual bandit for adaptive item selection:

```bash
# One-time setup: warm-start from historical data
python scripts/warmstart_bandit.py

# Use RL recommendations
python scripts/demo_trace.py --student-id <user> --topic <skill> --time-window <window> --use-rl

# Compare rule-based vs RL
python scripts/demo_trace.py compare-recs --student-id <user> --topic <skill>
```

Balances exploration/exploitation to learn which items work best for student profiles. Outputs include success probability, uncertainty, and reasoning.

## Features

- ✅ SAKT Engine (AUC 0.74 on EDM Cup 2023)
- ✅ Wide & Deep IRT
- ✅ Static GitHub Pages Dashboard (`docs/` with Plotly.js visuals)
- ✅ Attention-based explainability
- ✅ Gaming behavior detection
- ✅ LinUCB contextual bandit

## Development

Project follows ExecPlan methodology. Plans live in `execplans/`. See `PLANS.md` for details.

## License

MIT
