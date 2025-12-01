# ABOUTME: Documents utility and CLI scripts housed in scripts/.
# ABOUTME: Explains how demo Trace CLI interacts with model artifacts.

## Scripts

- `demo_trace.py`: Typer CLI that **actually** joins SAKT mastery, WD-IRT item parameters, RL bandit outputs, and explainability/gaming detectors to narrate a learning trace or produce alerts.
- `warmstart_bandit.py`: Bootstraps the LinUCB state from historical events so RL recommendations have sensible priors.
- `validate_join.py`: Verifies that SAKT + WD-IRT artifacts overlap on the same skill/item space.
- `pykt_sakt_spike.py`, `poc_*`: Kept for provenance; use the production commands (`make train_*`, `scripts/demo_trace.py`) for day-to-day workflows.

## Usage

```
# Trace mastery + recommendations (rule-based)
python scripts/demo_trace.py --student-id 123 --topic 7.RP.A.1 --time-window 2023-W15

# RL recommendations with LinUCB
python scripts/demo_trace.py trace --student-id 123 --topic 7.RP.A.1 --time-window 2023-W15 --use-rl

# Explain mastery (uses attention parquet)
python scripts/demo_trace.py explain --user-id 123 --skill 7.RP.A.1

# Gaming detection
python scripts/demo_trace.py gaming-check --user-id 123
python scripts/demo_trace.py gaming-check --output reports/gaming_alerts.parquet
```

All commands consume the parquet artifacts in `reports/` (and auto-generate `reports/skill_mastery.parquet` if needed), so once the pretrained checkpoints have been exported you can immediately demo the system without additional plumbing.
