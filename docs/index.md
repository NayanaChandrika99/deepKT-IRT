# ABOUTME: Landing page for the GitHub Pages dashboard narrative.
# ABOUTME: Mirrors the Streamlit story in a static format.

# deepKT + Wide &amp; Deep IRT Dashboard

<link rel="stylesheet" href="css/styles.css">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" defer></script>
<script src="js/app.js" defer></script>

This microsite summarizes the twin-engine learning analytics system:

1. **Student Readiness** (SAKT via pyKT) – tracks mastery and attention patterns.
2. **Item Health** (Wide &amp; Deep IRT) – monitors item parameters, drift, and clickstream behavior.
3. **Reinforcement Learning recommendations** – LinUCB bandit that balances exploration and exploitation.

## Contents

- [Data Pipeline Flow](#data-pipeline-flow)
- [Student Insights](#student-insights)
- [Recommendation Dynamics](#recommendation-dynamics)
- [Model Performance](#model-performance)
- [Operational Health](#operational-health)

Each section embeds interactive Plotly.js visuals backed by JSON snapshots stored in `docs/data/`. To refresh the data:

1. Install pandas/pyarrow (`pip install pandas pyarrow`).
2. Run `python docs/scripts/export_docs_assets.py`.
3. Commit the updated JSON files.

---

## Data Pipeline Flow

<div id="pipeline-viz" class="viz-container"></div>

This Sankey diagram shows how raw clickstream CSVs flow into canonical events, SAKT prep, and WD-IRT prep. Counts come from `data/interim` and `data/processed`.

---

## Student Insights

<div id="student-viz" class="viz-container"></div>

We highlight a sample student's mastery distribution, timeline, radar, and attention weights using data from `reports/skill_mastery.parquet` and `reports/sakt_attention.parquet`.

---

## Recommendation Dynamics

<div id="recommendation-viz" class="viz-container"></div>

This section breaks down how the LinUCB bandit chooses items, showing expected reward vs. uncertainty. Data source: `reports/bandit_state.npz`.

---

## Model Performance

<div id="model-viz" class="viz-container"></div>

Summaries of SAKT/WD-IRT training metrics, attention clusters, and item health trends derived from `reports/metrics/*.json` and `reports/item_params.parquet`.

---

## Operational Health

<div id="ops-viz" class="viz-container"></div>

Data lineage, throughput estimates, and joinability checks visualized using prepared JSON extracts (`docs/data/ops_summary.json`).
