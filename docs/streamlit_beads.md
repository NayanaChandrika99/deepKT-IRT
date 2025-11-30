# Streamlit Dashboard Bead Plan

This document captures the comprehensive bead (issue) plan for implementing the Streamlit dashboard described in `execplans/streamlit_dashboard.md`. It’s designed to be self-contained so future contributors understand the goals, rationale, and dependencies without needing extra context.

---

## Epic A — Streamlit Foundation & Utilities

- **Why:** Shared loaders/components prevent duplicate logic, simplify maintenance, and enforce consistent visuals.
- **Depends on:** Existing parquet/NPZ artifacts only; no other code prerequisites.
- **Key considerations:** Loaders must degrade gracefully if data is missing; components should encapsulate animation defaults (500 ms transitions, rich hover templates); `streamlit_app/requirements.txt` should pin compatible versions of Streamlit, Plotly, NetworkX, etc.

**Subtasks**
1. Scaffold `streamlit_app/` project structure (`app.py`, `config.py`, `pages/`, `components/`, `data/`, `.streamlit/`).
2. Implement `data/loader.py` with `@st.cache_data` readers for all artifacts (student state, predictions, attention, skill mastery, item params/drift, canonical events, prepared datasets, checkpoint metrics, bandit state). Include metadata (e.g., `last_modified`) and friendly warnings when files are missing.
3. Author reusable components:
   - `components/metrics.py` (metric cards, badges, deltas).
   - `components/charts.py` (timeline, distribution, heatmap, Sankey, radar, gauge, network graph) with built-in animation/hover defaults.
   - `components/tables.py` (filterable tables, summary stats).
4. Add Streamlit-specific requirements (`streamlit_app/requirements.txt`) and document setup/run instructions in `streamlit_app/README.md`.

---

## Epic B — Page 1: Student Insights

- **Why:** Shows stakeholders how the twin-engine system profiles an individual student—mastery trajectory, explanations, gaming alerts.
- **Depends on:** Epic A foundation; SAKT exports (`sakt_*`, `skill_mastery.parquet`) and canonical events for skill lookups/gaming detection.
- **Considerations:** Handle missing attention data gracefully; mention Lightning.ai checkpoint provenance in descriptions.

**Subtasks**
1. Student dashboard & mastery distribution: sidebar selector → metric cards (avg mastery, interaction count, confidence) + histogram from `skill_mastery.parquet`.
2. Animated mastery timeline: Plotly line chart with `animation_frame="sequence_position"` showing mastery evolution per skill.
3. Explainability card: Use `sakt_attention.parquet` to list top influences (enriched with canonical events for skill names, correctness, timestamps).
4. Skill radar chart: Polar chart of mastery snapshot across top skills with smooth transitions when student changes.
5. Gaming detection console: Run rapid-guess/help-abuse/suspicious-pattern heuristics on canonical events; render alerts with severity coloring.
6. Attention heatmap: Zoomable 2D heatmap of attention weights with rich hover showing item IDs and correctness; fallback message if attention data missing.

---

## Epic C — Page 2: Recommendations

- **Why:** Explains how recommendations are produced (rule-based vs LinUCB) and builds trust in RL-driven decisions.
- **Depends on:** Epics A & B (needs skill mastery), `item_params.parquet`, `bandit_state.npz`.
- **Considerations:** Document that RL state can be regenerated via `scripts/warmstart_bandit.py`; highlight exploration vs exploitation logic.

**Subtasks**
1. RL recommendation explorer: Load bandit state, compute expected reward/uncertainty for items in selected skill, show top N with explanation text (template or LLM).
2. Animated UCB confidence gauge: Gauge indicator showing uncertainty vs expected reward for top recommendation with animated needle and textual mode (explore vs exploit).
3. RL vs rule-based comparison: Baseline recommendation (easiest/hardest items based on mastery) vs RL output, highlighting overlaps/divergences and rationale.

---

## Epic D — Page 3: Model Performance

- **Why:** Visualizes training health, attention behavior, and item health to prove the twin engines are reliable.
- **Depends on:** Epic A; Lightning.ai metric CSVs; attention data; item params/drift.
- **Considerations:** Clarify metric provenance (Lightning.ai run names); explain feature-importance methodology.

**Subtasks**
1. Training dashboard (animated): Dual-pane (SAKT + WD-IRT) AUC/loss vs epoch with animation and hover details.
2. Attention mapping network graph: Force-directed graph of interactions with edge thickness representing attention weight (pulsing animation).
3. Item health scatter + drift alerts: Scatter of difficulty vs discrimination colored by drift score; filter for high-drift items.
4. Training curves deep dive: Static line charts with metric picker and optional smoothing/overlay of multiple runs.
5. Feature importance analysis: Load WD-IRT checkpoint, compute relative importance of wide/deep features (e.g., L1 norms), visualize as bar chart with explanatory notes.

---

## Epic E — Page 4: Data Quality

- **Why:** Demonstrates that the canonical data pipeline is healthy (coverage, schema, splits, joinability).
- **Depends on:** Epic A; canonical events; prepared SAKT/WD-IRT datasets; join metrics (from `scripts/validate_join.py` logic).
- **Considerations:** Keep ingestion stats tied to actual `make data` outputs; highlight if splits deviate from expected ratios.

**Subtasks**
1. Animated ingestion flow (Sankey): Raw → canonical → SAKT prep → WD-IRT prep with flowing animation and hover showing record counts + file paths.
2. Coverage heatmap: User vs skill density map with zoom/pan controls and summary stats.
3. Sequence quality metrics: Histogram and statistics of sequence lengths (pre-truncation) to show padding/truncation percentages relative to `seq_len`.
4. Split integrity dashboard: Train/val/test ratios, user-overlap check, and warnings if seeds mismatch configuration.
5. Schema validation status: Run LearningEvent schema validation and display pass/fail per field with remediation guidance.
6. Joinability gauge: Percentage of SAKT predictions overlapping WD-IRT items/skills with sample joined records.

---

## Epic F — Page 5: Pipeline Health

- **Why:** Gives ops stakeholders visibility into artifact lineage, throughput, and drift alerts.
- **Depends on:** Epic A; file metadata; drift history.
- **Considerations:** Provide file paths and timestamps for traceability; emphasize Lightning.ai checkpoint origins.

**Subtasks**
1. Data lineage map: Network graph of artifact dependencies (raw CSV → canonical parquet → reports) with hover showing file size, last modified, producing command.
2. Throughput monitoring: Metrics/table showing event counts and approximate processing times per stage, highlighting bottlenecks.
3. Data join overview (Venn): Visual overlap of user_ids/item_ids across canonical events, SAKT predictions, WD-IRT params with counts and percentages.
4. Model drift alerts (animated): Time-series for high-drift items with animation showing when drift score crossed threshold and recommended actions.

---

## Epic G — Polish, Documentation, Guardrails

- **Why:** Ensures the dashboard is understandable, resilient, and easy to maintain.
- **Depends on:** Completion of Epics A–F.
- **Considerations:** Document static-data assumption, Lightning.ai provenance, and instructions for regenerating artifacts.

**Subtasks**
1. Add descriptions/help text to each page/visualization explaining data sources, rationale, and Lightning.ai context.
2. Implement error handling and empty states—graceful `st.info` guidance when artifacts are missing.
3. Show last-updated timestamps per data source and add a “Refresh data” button that clears Streamlit caches.
4. Write `streamlit_app/README.md` covering setup, data requirements, troubleshooting, and how to rerun exports after retraining.
5. Performance sanity check (ensure cached loads <3s, profile slow queries, adjust caching or filters as needed).

---

## Suggested Workflow for Creating Beads

1. Run `bd doctor --fix` (if needed) to clear daemon warnings after `bd init`.
2. For each epic above, run `bd create "<Epic Title>"` and paste the relevant section as the issue body.
3. Reference dependencies in each issue (e.g., “DependsOn: deepKT+IRT-XX” once IDs exist).
4. Optionally create sub-issues for complex visualization clusters if more granularity is desired.

This plan mirrors the structure in `.beads/README.md` and should stay in sync when scope changes. Update both documents if epics/tasks evolve.
