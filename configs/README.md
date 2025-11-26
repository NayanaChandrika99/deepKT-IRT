# ABOUTME: Defines how experiment configs are structured and applied.
# ABOUTME: Lists required keys for WD-IRT and SAKT configuration files.

Configuration files use YAML and follow this pattern:

```
run_name: wd_irt_edm
seed: 42
data:
  dataset: edm_cup_2023
  events_path: data/interim/edm_cup_2023_42_events.parquet
model:
  module: src.wd_irt.model:WideDeepIrtModule
  params:
    wide_dim: 256
    deep_dim: 128
training:
  batch_size: 2048
  max_epochs: 20
  optimizer: adam
outputs:
  item_params: reports/item_params.parquet
  item_drift: reports/item_drift.parquet
  behavior_slices: reports/behavior_slices.md
```

Guidelines:

- Keep one config per dataset/experiment. Copy existing configs and edit values instead of reusing names.
- Include all absolute or repo-relative paths explicitly.
- For SAKT configs, add a `pykt` section referencing dataset alias, maximum sequence length, embedding dimension, attention heads, dropout, and checkpoint directory.
- Store sensitive credentials via environment variables and reference them in config (e.g., `${ASSIST_TOKEN}`).

Current starter configs:

- `wd_irt_edm.yaml`
- `sakt_assist2009.yaml`

Add notes inside this README whenever you introduce new required keys.
