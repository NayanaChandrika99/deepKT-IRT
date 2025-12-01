# Beads - AI-Native Issue Tracking

Welcome to Beads! This repository uses **Beads** for issue tracking - a modern, AI-native tool designed to live directly in your codebase alongside your code.

## What is Beads?

Beads is issue tracking that lives in your repo, making it perfect for AI coding agents and developers who want their issues close to their code. No web UI required - everything works through the CLI and integrates seamlessly with git.

**Learn more:** [github.com/steveyegge/beads](https://github.com/steveyegge/beads)

## Quick Start

### Essential Commands

```bash
# Create new issues
bd create "Add user authentication"

# View all issues
bd list

# View issue details
bd show <issue-id>

# Update issue status
bd update <issue-id> --status in-progress
bd update <issue-id> --status done

# Sync with git remote
bd sync
```

### Working with Issues

Issues in Beads are:
- **Git-native**: Stored in `.beads/issues.jsonl` and synced like code
- **AI-friendly**: CLI-first design works perfectly with AI coding agents
- **Branch-aware**: Issues can follow your branch workflow
- **Always in sync**: Auto-syncs with your commits

## Why Beads?

✨ **AI-Native Design**
- Built specifically for AI-assisted development workflows
- CLI-first interface works seamlessly with AI coding agents
- No context switching to web UIs

🚀 **Developer Focused**
- Issues live in your repo, right next to your code
- Works offline, syncs when you push
- Fast, lightweight, and stays out of your way

🔧 **Git Integration**
- Automatic sync with git commits
- Branch-aware issue tracking
- Intelligent JSONL merge resolution

## Get Started with Beads

Try Beads in your own projects:

```bash
# Install Beads
curl -sSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash

# Initialize in your repo
bd init

# Create your first issue
bd create "Try out Beads"
```

## Learn More

- **Documentation**: [github.com/steveyegge/beads/docs](https://github.com/steveyegge/beads/tree/main/docs)
- **Quick Start Guide**: Run `bd quickstart`
- **Examples**: [github.com/steveyegge/beads/examples](https://github.com/steveyegge/beads/tree/main/examples)

---

*Beads: Issue tracking that moves at the speed of thought* ⚡

---

# Streamlit Dashboard Workstream (Beads Backlog)

This section documents the end-to-end plan for the Streamlit dashboard described in `execplans/streamlit_dashboard.md`. It is intentionally verbose so our future selves (or other agents) can understand the “why,” the dependencies, and the expected outcomes without reopening other files.

## Goals & Intent

1. **Showcase the twin-engine system** (SAKT + Wide&Deep IRT) with rich visuals that resonate with product/business stakeholders.
2. **Explain how recommendations and analytics work**—training curves, attention maps, RL bandit decisions, data flow health.
3. **Provide operational visibility** into data quality, pipeline lineage, and gaming detection so the system is auditable.
4. **Do everything with static artifacts** (parquet/npz checkpoints) to keep the first release frictionless—no DB wiring yet.

## Target Stack & Constraints

- Streamlit multipage app (`streamlit_app/`) using Plotly for animation/interactivity.
- Data is sourced from existing parquet/npz artifacts in `reports/` + prepared datasets under `data/`.
- App must degrade gracefully if a file is missing (clear message vs crash).
- Visualization count: **24**, spread across 5 pages (Student Insights, Recommendations, Model Performance, Data Quality, Pipeline Health) + home/overview page.

## High-Level Dependencies

| Stage | Depends On | Notes |
|-------|------------|-------|
| Foundation (app scaffolding, loaders, components) | None | Must exist before any page work. |
| Student Insights Page | Foundation, existing SAKT artifacts (`sakt_*`, `skill_mastery.parquet`, canonical events) | Needs attention parquet + gaming detection. |
| Recommendations Page | Foundation, `bandit_state.npz`, `item_params.parquet`, `skill_mastery.parquet` | RL + rule-based comparison logic. |
| Model Performance Page | Foundation, checkpoint metric CSVs, attention parquet, item params/drift | Visualizes training/attention/feature info. |
| Data Quality Page | Foundation, canonical events, prepared datasets, schema validation | Focuses on ingestion stats and coverage. |
| Pipeline Health Page | Foundation, file metadata, drift history, join stats | Highlights lineage, throughput, alerts. |
| Polish & Documentation | All previous stages | Adds help text, error handling, README. |

## Detailed Task Beads

Each bead below should be represented as a Beads issue (or sub-issue) if we want fine-grained tracking; the commentary here captures background + acceptance criteria.

### Bead 1: Streamlit Foundation & Utilities
- **Background:** We need a single place to load/cached data, share styling, and render reusable components before any viz can exist.
- **Tasks:**
  1. Create `streamlit_app/` scaffold (app.py, `pages/`, `components/`, `data/`, etc.).
  2. Implement `data/loader.py` with cached readers for every artifact (student state, predictions, attention, skill mastery, item params, drift, bandit state, canonical events, training metrics, prepared datasets).
  3. Implement reusable components:
     - `components/metrics.py` (metric cards, status badges).
     - `components/charts.py` (timeline builder, distribution chart, heatmap, Sankey, radar, gauge, network graph, etc.).
     - `components/tables.py` (filterable tables, summary stats).
  4. Add `config.py` for paths + theme constants.
  5. Update `streamlit_app/requirements.txt` (streamlit, plotly, networkx, etc.).
- **Success Criteria:** `streamlit run streamlit_app/app.py` boots with navigation skeleton; loader gracefully handles missing files; caching works.

### Bead 2: Student Insights Page (Page 1)
- **Purpose:** Show end users how an individual student looks—mastery trajectory, explainability, gaming alerts.
- **Subtasks:**
  1. Student dashboard metrics + mastery distribution chart (uses `skill_mastery.parquet`).
  2. Animated mastery timeline (Plotly animation over `sakt_student_state.parquet`).
  3. Explainability card pulling top influences from `sakt_attention.parquet` (join against canonical events for skill names).
  4. Skill radar chart (top skills for selected student).
  5. Gaming detection console (run detection helpers on canonical events, highlight severity).
  6. Interactive attention heatmap (zoom/pan, rich hover, smooth transitions).
- **Dependencies:** Foundation bead; attention + canonical data must exist.
- **Notes:** Provide fallback messages when attention data missing; keep GPU training context in help text (“derived from Lightning.ai checkpoint X”).

### Bead 3: Recommendations Page (Page 2)
- **Purpose:** Explain how recommendations are generated (rule-based vs RL) and how confident the system is.
- **Subtasks:**
  1. RL recommendation explorer using `bandit_state.npz` + item params (display top items with expected reward/uncertainty flags).
  2. UCB confidence gauge (animated needle showing exploration vs exploitation ratio).
  3. RL vs rule-based comparison table (hardest-items-from-weakest-skill baseline).
- **Dependencies:** Foundation + Student page (needs skill mastery); ensures RL output is transparent.
- **Notes:** Mention Lightning.ai for RL warmstart? Provide button to load new bandit state when `warmstart_bandit.py` rerun.

### Bead 4: Model Performance Page (Page 3)
- **Purpose:** Give stakeholders insight into training quality, attention behavior, and item health.
- **Subtasks:**
  1. Animated training dashboard (AUC/loss vs epoch for both engines).
  2. Attention mapping network graph (force-directed view of influence edges).
  3. Item health scatter + drift alerts (difficulty vs discrimination colored by drift score).
  4. Deep dive training curves with smoothing/toggles.
  5. Wide & Deep feature importance (analyze checkpoint weights).
- **Dependencies:** Foundation + metric CSVs + attention + item artifacts.
- **Notes:** For feature importance, document assumptions (e.g., L1 norm interpretation) in the UI so future viewers know what they’re seeing.

### Bead 5: Data Quality Page (Page 4)
- **Purpose:** Validate canonical pipeline outputs and highlight potential data hygiene issues.
- **Subtasks:**
  1. Animated canonical event flow (Sankey from raw → canonical → SAKT/WD-IRT prepared data).
  2. Coverage heatmap (user vs skill density).
  3. Sequence quality metrics (length distribution, truncation/padding stats).
  4. Split integrity dashboard (train/val/test ratios, leakage checks).
  5. Schema validation status panel (tie into `src/common/schemas`).
  6. Joinability gauge (percentage of SAKT predictions overlapping WD-IRT items).
- **Dependencies:** Requires canonical events, prepared SAKT/WD-IRT datasets, and join metrics (can reuse logic from `validate_join.py`).
- **Notes:** Document any assumptions (e.g., using seed 42 splits) directly in the page description.

### Bead 6: Pipeline Health Page (Page 5)
- **Purpose:** Communicate lineage, throughput, and drift alerts to ops stakeholders.
- **Subtasks:**
  1. Interactive data lineage map (graph of file dependencies with timestamps).
  2. Throughput monitoring (event counts, processing duration estimates).
  3. Data join overview (Venn diagram of user/item overlaps between data sets).
  4. Model drift alerts (animated trend lines for high-drift items).
- **Dependencies:** Foundation, file metadata (use `Path.stat()`), drift history.
- **Notes:** Provide export path info so someone can locate files quickly (e.g., hover showing actual path). Clarify that throughput is approximated (since we’re reading local artifacts).

### Bead 7: Polish, Docs, & Guard Rails
- **Purpose:** Make the dashboard self-documenting and resilient.
- **Subtasks:**
  1. Add descriptive text/help tooltips for every visualization (include rationale and what data feeds it).
  2. Implement error handling + friendly empty states when artifacts missing.
  3. Add “last updated” timestamps per data source (based on file mod times).
  4. Provide a manual refresh button to clear Streamlit caches.
  5. Write `streamlit_app/README.md` with setup/run instructions, data requirements, and known limitations.
  6. Conduct a pass for performance (ensure caching reduces load times to <3s).
- **Dependencies:** All visualization beads complete.
- **Notes:** Document in README that data is static/refresh-on-demand and how to regenerate artifacts (e.g., rerun exports after retraining on Lightning.ai).

## General Considerations

- **Lightning.ai Context:** Everywhere we visualize training/attention/bandit outputs, mention that the source checkpoints came from Lightning.ai A100 runs synced into `reports/checkpoints/…`. This maintains traceability.
- **Accessibility:** Use color palettes that work for colorblind users (Plotly’s qualitative sets + custom overrides).
- **Testing:** Even though Streamlit is mostly visual, we should add a sanity check (e.g., a pytest that loads `data/loader.py` functions to ensure paths resolve).
- **Extensibility:** Keep components generic so future data sources (e.g., DB hooks) can swap in without rewriting theming/logic.

---

This plan should map one-to-one with forthcoming Beads issues (`bd create ...`). When creating each issue, reference the corresponding bead (e.g., “Bead 3: Recommendations Page – UCB Gauge”). This document stays updated as we learn more or adjust scope. Let’s keep it truthful: if a visualization morphs or is deprioritized, edit here so the backlog remains the single source of truth.
