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

You can drive the training/exports either with `make` or by invoking the modules directly.

#### SAKT (Student Readiness)

```bash
# Train SAKT model on the chosen config (local or remote GPU)
make train_sakt SAKT_CONFIG=configs/sakt_edm.yaml

# Export student mastery, predictions, attention weights
make export_sakt \
    SAKT_CONFIG=configs/sakt_edm.yaml \
    SAKT_CHECKPOINT=reports/checkpoints/sakt_edm/sakt_edm_seed42_best.pt
```

**Expected outputs:**
- `reports/sakt_predictions.parquet` — Predicted vs actual correctness
- `reports/sakt_student_state.parquet` — Per-interaction mastery estimates
- `reports/sakt_attention.parquet` — Attention weights for explainability
- `reports/sakt_mastery_report.md` — Summary statistics

**Validated performance:** AUC 0.74 on the EDM Cup skill set. The production checkpoint was trained on Lightning.ai (remote A100) and committed to `reports/checkpoints/sakt_edm/sakt_edm_seed42_best.pt`.

#### Wide & Deep IRT (Item Health)

```bash
# Train Wide & Deep IRT model
make train_wdirt WD_CONFIG=configs/wd_irt_edm.yaml

# Export item parameters and drift
make export_wdirt \
    WD_CONFIG=configs/wd_irt_edm.yaml \
    WD_CHECKPOINT=reports/checkpoints/wd_irt_edm/latest.ckpt
```

**Expected outputs:**
- `reports/item_params.parquet` — Item difficulty, discrimination, guessing
- `reports/item_drift.parquet` — Temporal drift flags
- `reports/behavior_slices.md` — Item health summary by topic

> **Training on Lightning.ai**  
> Both engines were originally trained on Lightning.ai remote GPU sessions to keep local development lean. The resulting checkpoints live in `reports/checkpoints/` so you can run exports or demos without re-training unless you need to fine-tune. When re-training, you can either run locally (if you have GPU access) or trigger a Lightning.ai job and copy the resulting `.pt/.ckpt` files back into this directory.

### Pretrained Checkpoints

| Engine | Dataset | Path | Notes |
|--------|---------|------|-------|
| SAKT | EDM Cup 2023 | `reports/checkpoints/sakt_edm/sakt_edm_seed42_best.pt` | Trained on Lightning.ai (A100) seed 42 job. |
| SAKT | ASSISTments 2009 | `reports/checkpoints/sakt_assist2009/sakt_assist2009_seed42_best.pt` | Baseline benchmark checkpoint. |
| Wide & Deep IRT | EDM Cup 2023 | `reports/checkpoints/wd_irt_edm/latest.ckpt` (update if re-trained) | Produced on Lightning.ai; use `make export_wdirt` to regenerate item health artifacts. |

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

Generate skill mastery (if not already present) and emit recommendations by joining SAKT + WD-IRT outputs. The CLI already layers on Phase 5 features (explainability, gaming detection, RL bandit), so one command shows the full experience:

```bash
source .venv/bin/activate
python scripts/demo_trace.py --student-id <user> --topic <skill_code> --time-window <window>
python scripts/demo_trace.py --student-id <user> --topic <skill_code> --time-window <window> --use-rl  # LinUCB
python scripts/demo_trace.py explain --user-id <user> --skill <skill_code>  # Attention-based insight
python scripts/demo_trace.py gaming-check --user-id <user>  # Clickstream alerting
```

Requirements: `reports/sakt_student_state.parquet`, `data/interim/...events.parquet`, and `reports/item_params.parquet` (with optional `reports/item_drift.parquet`). The CLI writes `reports/skill_mastery.parquet` on first run.

### Explainability & Gaming (Phase 5A)

**LLM-Generated Explanations (Optional)**

All explanations can be generated using LLMs for more natural, contextual language. By default, the system uses template-based explanations. To enable LLM generation:

```bash
# Set environment variable
export USE_LLM_EXPLANATIONS=true

# Choose provider (default: openai)
export LLM_PROVIDER=openai  # or "anthropic"
export LLM_MODEL=gpt-4o-mini  # or "claude-3-haiku-20240307"

# Set API key
export OPENAI_API_KEY="sk-..."  # or ANTHROPIC_API_KEY for Claude

# Now all explanations will use LLM
python scripts/demo_trace.py explain --user-id 1006OOQBE9 --skill "4.NBT.A.1"
```

**What gets LLM-generated:**
- **Mastery explanations** — Insights and recommendations from attention patterns
- **RL recommendation reasons** — Why items are recommended by the bandit
- **Rule-based recommendation reasons** — Why items match student mastery

**Cost:** ~$0.00015-0.00025 per explanation (gpt-4o-mini or claude-3-haiku)

Explain a student's mastery and surface gaming alerts:

```bash
python scripts/demo_trace.py explain --user-id <user> --skill <skill_code>
python scripts/demo_trace.py gaming-check --user-id <user>  # or omit for all students
```

Outputs rely on the same artifacts as the demo; attention parquet is optional (explanations degrade gracefully without it).

### RL Recommendations (Phase 5B)

Use reinforcement learning (LinUCB bandit) for adaptive recommendations:

```bash
# Warm-start the bandit from historical data (one-time setup)
python scripts/warmstart_bandit.py

# Use RL recommendations
python scripts/demo_trace.py trace --student-id <user> --topic <skill> --time-window <window> --use-rl

# Compare rule-based vs RL side-by-side
python scripts/demo_trace.py compare-recs --student-id <user> --topic <skill>
```

The bandit learns which items work best for which student profiles, balancing exploration (trying new items) and exploitation (using best known). Recommendations include expected success probability, uncertainty estimates, and human-readable reasons (LLM-backed when enabled). In practice you can keep the rule-based recommendations as a baseline or let RL fully replace the Phase 4 logic by passing `--use-rl` (and optionally wiring that path up as the default in your product surface).

### Current Status

- ✅ **SAKT Engine** — Complete (training, export, 0.74 AUC)
- ✅ **Wide & Deep IRT** — Complete (training, export)
- ✅ **Static GitHub Pages Dashboard** — `docs/` contains Plotly.js visuals powered by JSON exports (enable GitHub Pages to view)
- ✅ **Explainability & Gaming (Phase 5A)** — Attention-based explanations and behavioral alerts
- ✅ **RL Recommendations (Phase 5B)** — LinUCB contextual bandit for adaptive item selection (fully integrated; toggle via `--use-rl`)

## License

MIT
