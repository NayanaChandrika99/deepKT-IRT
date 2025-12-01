# deepKT+IRT GitHub Pages Dashboard Plan

## Overview

The Streamlit track is on hold; we’re focusing on a static GitHub Pages site (`docs/`) that retells the twin-engine analytics story (SAKT + Wide & Deep IRT) using Plotly.js/D3 and pre-exported JSON. The site mirrors the original plan’s five sections:
1. Data Pipeline Flow
2. Student Insights
3. Recommendation Dynamics
4. Model Performance
5. Operational Health

**Tech Stack:** Markdown + HTML + CSS + Plotly.js (or D3.js)
**Data Mode:** Static JSON exports generated from repo artifacts
**Target Audience:** Product/Business Stakeholders
**Scope:** Condensed storytelling (at least one visualization per section) with refresh instructions.

## Animation & Interactivity Strategy

**Philosophy:** Visualizations should tell dynamic stories, not show static snapshots. Use animation to reveal temporal patterns, smooth transitions to guide attention, and rich interactivity to enable exploration.

**Inspired by:** [Central Flows](https://centralflows.github.io/) - animated Sankey diagrams, smooth transitions, interactive filtering

### Animation Techniques

1. **Temporal Animations** (Plotly `animation_frame`):
   - Student mastery timeline: Play through sequence positions like a video
   - Training curves: Animate epoch-by-epoch progress
   - Item drift alerts: Show difficulty changes over time
   - **Implementation:** Add `animation_frame` parameter to plotly charts, include play/pause/slider controls

2. **Transition Animations** (Plotly `layout.transition`):
   - Smooth updates when switching students or filters
   - Animated bar chart growth when loading new data
   - Gauge needle animations for UCB confidence
   - **Implementation:** Set `layout.transition.duration=500ms` for smooth 0.5s transitions

3. **Flow Animations** (Animated Sankey):
   - Canonical Event Flow: Particles flowing through pipeline stages
   - Attention mapping: Weighted edges pulse with attention strength
   - Data lineage: Highlight active data paths
   - **Implementation:** Use Plotly Sankey with custom hover effects, consider plotly-resampler for smooth interactions

4. **Interactive Network Graphs** (Plotly Graph Objects):
   - Attention mapping: Zoom/pan/click nodes to see interaction details
   - Skill dependency graphs (if applicable)
   - **Implementation:** Use `plotly.graph_objects.Scatter` with custom node/edge layouts, enable dragmode='pan'

### Interactivity Enhancements

1. **Rich Hover Details**:
   - Not just tooltips - show mini-cards with full context
   - Example: Hover over attention node → show item text, correctness, timestamp, skill
   - Example: Hover over training curve point → show all metrics at that epoch
   - **Implementation:** Use `hovertemplate` with HTML formatting, include `customdata` for rich info

2. **Linked Selections** (Cross-filtering):
   - Select student in one chart → all charts update
   - Click on item in health dashboard → show full history
   - **Implementation:** Use Streamlit session state to track selections, update all charts reactively

3. **Playable Timelines**:
   - Student mastery timeline: Play button shows mastery evolving like a video
   - Training dashboard: Step through epochs with play/pause/skip controls
   - **Implementation:** Plotly animation controls + Streamlit custom components if needed

4. **Zoom & Pan**:
   - All scatter plots, heatmaps, and network graphs support zoom/pan
   - Double-click to reset view
   - **Implementation:** Set `layout.dragmode='zoom'` by default, add reset button

### Visualization-Specific Animation Plan

**High Priority (Must Animate):**
- ✅ **1.2 Student Mastery Timeline**: Animated playback through sequence positions
- ✅ **3.1 Training Dashboard**: Animate epoch-by-epoch progress for both models
- ✅ **3.2 Attention Mapping**: Pulsing edges, interactive node exploration
- ✅ **4.1 Canonical Event Flow**: Animated Sankey with flowing particles
- ✅ **5.1 Data Lineage Map**: Interactive graph with zoom/pan

**Medium Priority (Should Animate):**
- ⚡ **1.6 Attention Heatmap**: Smooth transitions when switching students
- ⚡ **2.2 UCB Confidence Gauge**: Animated needle movement
- ⚡ **3.3 Item Health Dashboard**: Transition animations when filtering
- ⚡ **5.4 Model Drift Alerts**: Animated trend lines showing drift over time

**Low Priority (Static OK, Add Interactivity):**
- 📊 **1.1 Student Dashboard**: Rich hover on skill bars
- 📊 **1.4 Skill Radar**: Smooth transitions when switching students
- 📊 **4.2 Coverage Heatmap**: Zoom into sparse regions

### Technical Implementation Notes

**Plotly Animation Frame Setup:**
```python
# Example: Animated timeline
fig = px.line(
    df,
    x='sequence_position',
    y='mastery_score',
    color='skill_id',
    animation_frame='sequence_position',  # Creates play controls
    range_y=[0, 1],
    title="Student Mastery Evolution"
)
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300  # 300ms per frame
fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 200  # 200ms transition
```

**Smooth Transitions:**
```python
# Add to all charts
fig.update_layout(
    transition_duration=500,  # 0.5s smooth transitions
    hovermode='closest',
    dragmode='zoom'
)
```

**Rich Hover Templates:**
```python
# Example: Attention node hover
fig.update_traces(
    hovertemplate="<b>Item:</b> %{customdata[0]}<br>" +
                  "<b>Correct:</b> %{customdata[1]}<br>" +
                  "<b>Skill:</b> %{customdata[2]}<br>" +
                  "<b>Attention Weight:</b> %{customdata[3]:.1%}<br>" +
                  "<extra></extra>"  # Hides secondary box
)
```

## Architecture

### Project Structure (GitHub Pages)
```
docs/
├── index.md (or index.html)
├── css/
├── js/
└── data/ (pre-rendered JSON from repo artifacts)
```

### Data Strategy (Static JSON)

- Use helper script (`docs/scripts/export_docs_assets.py` or manual pandas commands) to write compact JSON (≤200 rows) per dataset under `docs/data/`.
- Each JSON should include the fields needed for the Plotly visual (e.g., `mastery_mean`, `skill`, `top_influences`).
- Commit the JSON files so GitHub Pages can serve them statically; update them only when exports change.

## Visualization Plan (GitHub Pages)

We’ll mirror the five Streamlit sections, but each section gets one or two key visuals to keep the page performant. All visuals are driven by JSON (Plotly.js or D3).

### 1. Data Pipeline Flow

- Sankey showing Raw CSV → Canonical Events → SAKT Prep + WD-IRT Prep.
- Source: `docs/data/pipeline_sample.json`.
- Implementation: Plotly.js `Sankey` chart, styled with site CSS.

### 2. Student Insights

- Combined plot containing:
  - Mastery histogram (from `skill_sample.json`).
  - Skill radar (top 6 skills by interaction count).
  - Attention weights bar chart (from `attention_sample.json`).
- Implementation: Plotly.js `histogram` + `scatterpolar` + `bar` (using `grid` layout).
- Optional: Add a textual narrative explaining what the charts show.

### 3. Recommendation Dynamics

- Scatter of expected reward vs. uncertainty (LinUCB) with sample data, or precompute from bandit snapshot if feasible.
- Gauge or metrics showing exploration vs. exploitation ratio (static numbers until we wire JSON).
- Implementation: Plotly.js scatter + custom metric cards.

### 4. Model Performance

- Line chart of sample train/val AUC (since we don’t have streamed metrics yet; future TODO is to export `reports/metrics/*.json`).
- Animated timeline of attention clusters could be approximated with Plotly; placeholder for now.
- Implementation: Plotly.js line chart with hard-coded data (document in JSON if needed).

### 5. Operational Health

- Metric grid summarizing canonical event counts, SAKT prep sample, WD-IRT prep sample.
- Future TODO: add Venn diagram or D3 lineage map once JSON exports are available.

## Implementation Phases (GitHub Pages)

### Phase 1: Scaffolding & Data Exports
1. Create `docs/` structure (CSS, JS, data directories, `index.md`).
2. Generate JSON summaries (`attention_sample`, `skill_sample`, `pipeline_sample`) via pandas script; commit to `docs/data/`.
3. Include Plotly.js CDN + custom JS (`docs/js/app.js`) that loads JSON and renders charts.

### Phase 2: Student Insights + Pipeline Flow
1. Implement Sankey + Student Insights section using JSON.
2. Style with consistent dark theme (`docs/css/styles.css`).
3. Add textual narrative explaining each chart and its data source.

### Phase 3: Recommendation + Model Sections
1. Add RL scatter/gauge (initial sample data, or export from `bandit_state.npz` if possible).
2. Create training curve chart (sample or exported metrics).
3. Document in JS how to refresh the sample data.

### Phase 4: Operational Health Section
1. Render metric cards using pipeline JSON (counts from canonical dataset).
2. Provide instructions for refreshing metrics when pipeline is rerun.
3. Optional: add a D3-based lineage graph if time permits.

### Phase 5: Polish & Deployment
1. Add instructions for regenerating JSON exports (scripts, manual commands).
2. Configure GitHub Pages (branch `main`, folder `/docs`).
3. Add badges/links on the main README pointing to the live site.

## Refresh Workflow
- Run the export script whenever parquet artifacts change:
  ```bash
  python docs/scripts/export_docs_assets.py
  ```
- Commit updated JSONs to the repo; GitHub Pages automatically serves the latest version.
- Update `docs/js/app.js` if the schema changes (e.g., new fields in JSON).

## Success Criteria (GitHub Pages)
- Site loads all sections without JS errors.
- JSON exports stay under ~200 rows each to keep load times fast (<2s).
- Visuals match the narrative described in README/plan.md.
- Stakeholders can navigate sections and understand the key takeaways without needing Streamlit/data access.

**1.1 Student Dashboard**
- **Data:** `skill_mastery.parquet`
- **UI:** Sidebar student selector → 3 metrics (avg mastery, total skills, confidence) + skill mastery bar chart + recent activity table
- **Implementation:**
  ```python
  student_id = st.sidebar.selectbox("Select Student", unique_students)
  student_data = skill_mastery[skill_mastery['user_id'] == student_id]
  col1, col2, col3 = st.columns(3)
  col1.metric("Avg Mastery", f"{student_data['mastery'].mean():.2f}")
  st.plotly_chart(build_distribution_chart(student_data['mastery'], bins=20, title="Skill Mastery Distribution"))
  ```

**1.2 Student Mastery Timeline** ✅ ANIMATED
- **Data:** `sakt_student_state.parquet`
- **UI:** Animated time-series line chart showing mastery evolution per skill for selected student
- **Animation:** Play through sequence positions with play/pause controls, smooth line drawing
- **Implementation:**
  ```python
  fig = px.line(df, x='sequence_position', y='mastery_score', color='skill_id',
                animation_frame='sequence_position', range_y=[0, 1])
  fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300
  ```

**1.3 Explainability Card**
- **Data:** `sakt_attention.parquet` + `canonical_events.parquet` (for item/skill lookup)
- **UI:** Rich text card showing WHY prediction was made - top influential past interactions with weights
- **Implementation:** Display top_influences from attention data, enrich with skill names, format as bullet list with emoji indicators (✅/❌)

**1.4 Skill Radar Chart**
- **Data:** `skill_mastery.parquet`
- **UI:** Radar chart showing student's mastery across multiple skills
- **Implementation:** Select top 8-10 skills by interaction count, plot on polar coordinates

**1.5 Gaming Detection Console**
- **Data:** `canonical_events.parquet` → run gaming_detection.py analysis
- **UI:** Alerts panel showing flagged students + gaming behavior metrics (rapid guess rate, help abuse %, suspicious patterns)
- **Implementation:** Run `detect_gaming()` on canonical events, display results in expandable table with severity colors

**1.6 Attention Heatmap** ⚡ INTERACTIVE
- **Data:** `sakt_attention.parquet`
- **UI:** Interactive 2D heatmap showing attention weights (rows=query positions, cols=key positions) for selected student sequence
- **Interactivity:** Smooth transitions when switching students, zoom/pan, rich hover showing item_ids and correctness
- **Implementation:**
  ```python
  fig = px.imshow(attention_matrix, labels=dict(x="Key Position", y="Query Position"),
                  hover_data={'item_id': item_ids, 'correct': correctness})
  fig.update_layout(transition_duration=500, dragmode='zoom')
  ```

### Page 2: Recommendations (3 visualizations)

**2.1 RL Recommendation Explorer**
- **Data:** `bandit_state.npz` + `item_params.parquet` + `skill_mastery.parquet`
- **UI:** Sidebar student selector → Top 10 recommended items table with (expected reward, uncertainty, is_exploration) + gauge showing exploration ratio
- **Implementation:** Load LinUCB bandit state, compute scores for all items, sort by UCB score, display with color coding

**2.2 UCB Confidence Gauge** ⚡ ANIMATED
- **Data:** `bandit_state.npz`
- **UI:** Animated gauge chart showing model confidence (uncertainty vs expected reward ratio) for current top recommendation
- **Animation:** Needle animates to new position when student changes, smooth 500ms transition
- **Implementation:**
  ```python
  fig = go.Figure(go.Indicator(
      mode="gauge+number+delta",
      value=uncertainty / expected,
      gauge={'axis': {'range': [0, 2]}, 'threshold': {'value': 0.5}}
  ))
  fig.update_layout(transition_duration=500)
  ```

**2.3 RL vs Rule-Based Comparison**
- **Data:** `item_params.parquet` + `skill_mastery.parquet`
- **UI:** Side-by-side table comparing RL recommendations vs simple rule-based (next hardest item in weakest skill)
- **Implementation:** Generate both recommendation lists, highlight differences, show overlap percentage metric

### Page 3: Model Performance (5 visualizations)

**3.1 Training Dashboard** ✅ ANIMATED
- **Data:** `checkpoints/sakt_kt/version_*/metrics.csv` + `checkpoints/wd_irt/version_*/metrics.csv`
- **UI:** Dual-pane showing SAKT metrics (train/val AUC, loss) and WD-IRT metrics (train/val AUC, loss) over epochs
- **Animation:** Play through training epochs with play/pause controls, watch metrics evolve
- **Implementation:**
  ```python
  fig = px.line(metrics_df, x='epoch', y=['train_auc', 'val_auc'],
                animation_frame='epoch', range_y=[0, 1])
  # Add train/val gap shading with fillbetween
  ```

**3.2 Attention Mapping Visualization** ✅ ANIMATED + INTERACTIVE
- **Data:** `sakt_attention.parquet`
- **UI:** Interactive force-directed network graph showing attention flow between interactions
- **Animation:** Edges pulse with attention strength, nodes expand on hover
- **Interactivity:** Click node to see item details, zoom/pan to explore, drag nodes to rearrange
- **Implementation:**
  ```python
  # Use plotly graph_objects for custom network layout
  nodes = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                     marker=dict(size=20, color=correctness_colors))
  edges = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                     line=dict(width=attention_weights*10))  # Line thickness = attention
  fig = go.Figure(data=[edges, nodes])
  fig.update_layout(dragmode='pan', hovermode='closest')
  ```

**3.3 Item Health Dashboard**
- **Data:** `item_params.parquet` + `item_drift.parquet`
- **UI:** Scatter plot (x=difficulty, y=discrimination) with color=drift_score + alerts for items with high drift or poor discrimination
- **Implementation:** Join params + drift, plot scatter, add threshold lines, filterable table below

**3.4 Training Curves (Deep Dive)**
- **Data:** `checkpoints/*/metrics.csv`
- **UI:** Detailed training curves with epoch-by-epoch metrics, smoothing controls, train/val gap visualization
- **Implementation:** Multi-select for metrics, plotly line chart with hover details, gap shading

**3.5 Feature Importance (WD-IRT)**
- **Data:** `checkpoints/wd_irt/version_*/` checkpoint → extract deep network weights
- **UI:** Bar chart showing relative importance of wide features vs deep clickstream features
- **Implementation:** Load checkpoint, compute L1 norms of weight matrices, normalize and display

### Page 4: Data Quality (6 visualizations)

**4.1 Canonical Event Flow (Ingestion DAG)** ✅ ANIMATED
- **Data:** `data/canonical/events.parquet`
- **UI:** Animated Sankey diagram showing raw CSV → canonical parquet → SAKT/WD-IRT prepared data with record counts
- **Animation:** Flow particles animate through pipeline stages (inspired by Central Flows)
- **Interactivity:** Hover shows exact record counts, click to highlight path
- **Implementation:**
  ```python
  fig = go.Figure(go.Sankey(
      node=dict(label=["Raw CSV", "Canonical", "SAKT", "WD-IRT"]),
      link=dict(source=[0,1,1], target=[1,2,3], value=record_counts)
  ))
  # Add custom animations for flowing particles if using D3.js wrapper
  ```

**4.2 Coverage Heatmap**
- **Data:** `data/canonical/events.parquet`
- **UI:** 2D heatmap (rows=users, cols=skills) showing interaction density, highlight sparse regions
- **Implementation:** Pivot table user x skill with count aggregation, plot heatmap with color scale

**4.3 Sequence Quality Metrics**
- **Data:** `data/sakt_prepared/train.csv`
- **UI:** Histogram of sequence lengths + metrics (avg length, median, % padded, % truncated)
- **Implementation:** Parse sequence columns, compute lengths, plot distribution with threshold markers

**4.4 Train/Val/Test Split Integrity**
- **Data:** `data/sakt_prepared/*.csv` + `data/wd_irt_prepared/*.parquet`
- **UI:** Metrics panel showing split ratios, user overlap check, temporal distribution
- **Implementation:** Load all splits, verify no user leakage, compute size ratios, plot user count per split

**4.5 Schema Validation Dashboard**
- **Data:** `data/canonical/events.parquet`
- **UI:** Checklist showing schema compliance (required columns present, types correct, no nulls in required fields, valid ranges)
- **Implementation:** Run schema validation from schemas.py, display results as status badges

**4.6 Joinability Gauge**
- **Data:** `data/canonical/events.parquet` + model predictions
- **UI:** Metrics showing % of events with valid user_id, item_id, skill_id in both canonical and model outputs
- **Implementation:** Compute join success rates, display as percentage metrics with drill-down tables

### Page 5: Pipeline Health (4 visualizations)

**5.1 Data Lineage Map** ✅ INTERACTIVE
- **Data:** File system metadata (file sizes, timestamps)
- **UI:** Interactive dependency graph showing data flow from raw CSV to final artifacts with last updated timestamps
- **Interactivity:** Zoom/pan to explore, click nodes to see file details, color by staleness (green=fresh, red=stale)
- **Implementation:**
  ```python
  # Use networkx for layout, plotly for rendering
  import networkx as nx
  G = nx.DiGraph()
  # Add nodes and edges based on file dependencies
  pos = nx.spring_layout(G)
  # Convert to plotly scatter plot with custom node/edge traces
  ```

**5.2 Throughput Monitoring**
- **Data:** File metadata + event counts
- **UI:** Metrics showing total events processed, processing rate (events/sec if timestamps available), bottleneck identification
- **Implementation:** Compute event counts per stage, estimate processing rates, highlight slowest stage

**5.3 Data Join Overview**
- **Data:** `canonical_events.parquet` + `sakt_predictions.parquet` + `skill_mastery.parquet`
- **UI:** Venn diagram showing overlap between canonical events, SAKT predictions, and mastery snapshots
- **Implementation:** Compute unique user_ids in each dataset, plot overlaps, show counts

**5.4 Model Drift Alerts** ⚡ ANIMATED
- **Data:** `item_drift.parquet` + historical predictions (if available)
- **UI:** Alert panel showing items with significant drift + animated trend charts for flagged items
- **Animation:** Trend lines draw progressively to show difficulty evolution over time
- **Implementation:**
  ```python
  # Filter high-drift items
  high_drift = item_drift[item_drift['drift_score'] > 0.3]
  # Plot difficulty trends with animation
  fig = px.line(drift_history, x='timestamp', y='difficulty', color='item_id',
                animation_frame='timestamp')
  ```

## Implementation Phases

### Phase 1: Foundation (Streamlit Infra)
**Tasks:**
1. Create project structure: `streamlit_app/` directory with all subdirectories
2. Implement `config.py`: Define REPORTS_DIR, DATA_DIR, CHECKPOINTS_DIR paths
3. Implement `data/loader.py`:
   - `load_sakt_student_state() -> pd.DataFrame`
   - `load_sakt_predictions() -> pd.DataFrame`
   - `load_sakt_attention() -> pd.DataFrame`
   - `load_skill_mastery() -> pd.DataFrame`
   - `load_item_params() -> pd.DataFrame`
   - `load_item_drift() -> pd.DataFrame`
   - `load_bandit_state() -> Dict[str, np.ndarray]`
   - `load_canonical_events() -> pd.DataFrame`
   - `load_training_metrics(model_name: str) -> pd.DataFrame`
4. Implement `components/metrics.py`: All 3 metric rendering functions
5. Implement `components/charts.py`: All 6 chart builder functions
6. Implement `components/tables.py`: Both table rendering functions
7. Create `app.py`: Landing page with navigation and overview metrics
8. Update `requirements.txt`: Add streamlit, plotly, numpy (ensure pandas/pyarrow already present)

**Verification:**
- Run `streamlit run streamlit_app/app.py` - should load without errors
- Verify all parquet files load successfully with caching
- Check navigation sidebar appears with 5 pages

### Phase 2: Student Insights Page (Streamlit)
**Tasks:**
1. Create `pages/1_student_insights.py`
2. Implement visualization 1.1: Student Dashboard
3. Implement visualization 1.2: Student Mastery Timeline
4. Implement visualization 1.3: Explainability Card
5. Implement visualization 1.4: Skill Radar Chart
6. Implement visualization 1.5: Gaming Detection Console
7. Implement visualization 1.6: Attention Heatmap

**Verification:**
- All 6 visualizations render without errors
- Student selector filters data correctly
- Attention data joins with canonical events for skill enrichment
- Gaming detection runs on canonical events

### Phase 3: Recommendations Page (Streamlit)
**Tasks:**
1. Create `pages/2_recommendations.py`
2. Implement visualization 2.1: RL Recommendation Explorer
3. Implement visualization 2.2: UCB Confidence Gauge
4. Implement visualization 2.3: RL vs Rule-Based Comparison

**Verification:**
- Bandit state loads from NPZ correctly
- LinUCB recommendations compute for all items
- Rule-based recommendations (hardest item in weakest skill) generate correctly
- Comparison highlights differences clearly

### Phase 4: Model Performance Page (Streamlit)
**Tasks:**
1. Create `pages/3_model_performance.py`
2. Implement visualization 3.1: Training Dashboard
3. Implement visualization 3.2: Attention Mapping Visualization
4. Implement visualization 3.3: Item Health Dashboard
5. Implement visualization 3.4: Training Curves (Deep Dive)
6. Implement visualization 3.5: Feature Importance (WD-IRT)

**Verification:**
- Training metrics load from latest checkpoint versions
- Attention mapping shows network graph with weighted edges
- Item health scatter plot combines params + drift
- Feature importance extracts from WD-IRT checkpoint

### Phase 5: Data Quality Page (Streamlit)
**Tasks:**
1. Create `pages/4_data_quality.py`
2. Implement visualization 4.1: Canonical Event Flow (Ingestion DAG)
3. Implement visualization 4.2: Coverage Heatmap
4. Implement visualization 4.3: Sequence Quality Metrics
5. Implement visualization 4.4: Train/Val/Test Split Integrity
6. Implement visualization 4.5: Schema Validation Dashboard
7. Implement visualization 4.6: Joinability Gauge

**Verification:**
- Sankey diagram shows flow from raw to prepared data
- Coverage heatmap highlights sparse user-skill pairs
- Sequence length distribution matches SAKT padding config
- Split integrity check confirms no user leakage
- Schema validation runs against LearningEvent dataclass

### Phase 6: Pipeline Health Page (Streamlit)
**Tasks:**
1. Create `pages/5_pipeline_health.py`
2. Implement visualization 5.1: Data Lineage Map
3. Implement visualization 5.2: Throughput Monitoring
4. Implement visualization 5.3: Data Join Overview
5. Implement visualization 5.4: Model Drift Alerts

**Verification:**
- Lineage map scans file system for dependencies
- Throughput metrics compute from file metadata
- Venn diagram shows user_id overlaps across datasets
- Drift alerts filter items with high drift scores

### Phase 7: Polish & Documentation (Streamlit)
**Tasks:**
1. Add page descriptions and help text to all visualizations
2. Implement error handling for missing files (friendly messages)
3. Add "Last Updated" timestamps to all pages
4. Create README.md in `streamlit_app/` with setup instructions
5. Add refresh button to reload data without restart
6. Optimize performance: profile slow visualizations, add additional caching
7. Test with product stakeholders: gather feedback on layout and clarity

**Verification:**
- All pages load gracefully with missing data
- README includes clear setup and run instructions
- Refresh button clears cache and reloads parquet files
- Stakeholder feedback incorporated into final adjustments

## GitHub Pages Track

### Goals
- Provide a lightweight, static version of the dashboard so stakeholders can browse the narrative on GitHub Pages (`docs/`).
- Highlight the most important visuals (Sankey pipeline, RL explanation, attention mapping) without requiring Streamlit or parquet files.

### Tasks
1. **Docs scaffolding**: Create `docs/index.md` with a Light/Dark theme-friendly layout, referencing CSS/JS assets.
2. **Data extracts**: Write helper scripts to export small JSON blobs (e.g., top 20 attention influences, RL example runs) to `docs/data/`.
3. **Visualization port**:
   - Use Plotly.js or D3.js to recreate the Sankey flow, attention graph, RL flow, etc., referencing static JSON.
   - Provide explanatory copy for each section (mirroring the Streamlit page descriptions).
4. **Deployment**: Configure GitHub Pages (branch `main`, folder `/docs`) and document the workflow for updating data extracts.

### Considerations
- Since GitHub Pages can’t run Python, precompute any data needed for the visualizations (e.g., using `scripts/export_docs_assets.py`).
- Keep the static site in sync with the Streamlit app by reusing the same JSON exports or referencing the same plan sections.

## Dependencies to Add (Streamlit)

**New packages for `streamlit_app/requirements.txt`:**
```
streamlit>=1.28.0
plotly>=5.17.0
networkx>=3.1  # For graph layouts (lineage map, attention mapping)
kaleido>=0.2.1  # For plotly static image export (optional)
```

**Already available (from main requirements.txt):**
- pandas==2.2.3
- pyarrow==18.1.0
- numpy==2.2.0
- scikit-learn==1.6.0

## Key Design Decisions

1. **Animation-first approach**: 5 high-priority visualizations use Plotly animations to tell dynamic stories (mastery timelines, training curves, attention mapping, event flow, lineage map)
2. **Smooth transitions everywhere**: All interactive charts use 500ms transition animations for professional feel
3. **Rich interactivity**: Zoom/pan on all graphs, rich hover tooltips with full context, linked selections across charts
4. **Multipage app structure**: Uses Streamlit's built-in `pages/` directory for automatic navigation
5. **Centralized data loading**: Single loader module with caching prevents redundant reads
6. **Reusable animated components**: Chart builders with built-in animation support for consistency
7. **Static refresh mode**: Manual refresh button to reload parquet files (no auto-refresh)
8. **Graceful degradation**: Missing data files show friendly messages instead of crashes
9. **Business-focused language**: Avoid technical jargon in titles and descriptions

## Success Metrics

- All 24 visualizations render correctly with production data
- Page load time < 3 seconds for cached data
- Zero errors with missing or incomplete data files
- Product stakeholders can navigate and interpret insights without training
- Gaming detection, explainability, and RL recommendations provide actionable insights

## Future Enhancements (Out of Scope)

- Real-time data refresh from database
- Export visualizations as PDF reports
- A/B testing dashboard for recommendation strategies
- User authentication and role-based access
- Historical trend analysis (requires time-series artifact storage)
