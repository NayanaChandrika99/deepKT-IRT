# deepKT+IRT GitHub Pages Dashboard - Complete Implementation Plan

## Overview

Static GitHub Pages site (`docs/`) showcasing the twin-engine analytics system (SAKT + Wide & Deep IRT) with **24 interactive visualizations** using Plotly.js and pre-exported JSON data.

**Tech Stack:** HTML + CSS + JavaScript (ES6 Modules) + Plotly.js (v2.32+)
**Data Mode:** Static JSON exports from parquet artifacts
**Target Audience:** Product/Business Stakeholders + Technical Teams
**Deployment:** GitHub Pages (main branch, `/docs` folder)
**Architecture:** Modular component library with reusable visualization classes

## Modular Component Architecture

All visualizations are built as **self-contained, reusable JavaScript classes** following a consistent design pattern:

### Component Design Pattern

Each visualization component:
- **Encapsulates** rendering logic, state, and configuration
- **Accepts** a container ID and optional configuration
- **Exposes** `render()`, `update()`, and `destroy()` methods
- **Returns** itself for method chaining
- **Handles** its own error states and loading indicators

**Example Component Structure:**
```javascript
// components/MasteryTimeline.js
class MasteryTimeline {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = {
      animationDuration: 300,
      transitionDuration: 500,
      height: 400,
      ...options
    };
    this.chart = null;
    this.data = null;
  }

  async render(data) {
    this.data = data;
    const trace = {
      x: data.map(d => d.sequence_position),
      y: data.map(d => d.mastery_score),
      type: 'scatter',
      mode: 'lines+markers',
      // ... Plotly config
    };

    const layout = {
      transition: { duration: this.options.transitionDuration },
      height: this.options.height,
      // ... layout config
    };

    this.chart = await Plotly.newPlot(this.containerId, [trace], layout);
    return this;
  }

  update(newData) {
    this.data = newData;
    Plotly.animate(this.containerId, {
      data: [/* transformed newData */],
      traces: [0]
    }, {
      transition: { duration: this.options.transitionDuration }
    });
    return this;
  }

  destroy() {
    if (this.chart) {
      Plotly.purge(this.containerId);
      this.chart = null;
    }
  }
}

export default MasteryTimeline;
```

### Project Structure (Modular)

```
docs/
├── index.html                          # Main landing page
├── css/
│   └── styles.css                      # Dark theme, responsive layout
├── js/
│   ├── app.js                          # Main orchestrator (imports & wires components)
│   ├── components/                     # 24 visualization components
│   │   ├── StudentDashboard.js         # 1.1 - Metrics + bar + table
│   │   ├── MasteryTimeline.js          # 1.2 - Animated timeline
│   │   ├── ExplainabilityCard.js       # 1.3 - Attention reasoning
│   │   ├── SkillRadar.js               # 1.4 - Polar chart
│   │   ├── GamingConsole.js            # 1.5 - Alert panel
│   │   ├── AttentionHeatmap.js         # 1.6 - Interactive heatmap
│   │   ├── RLExplorer.js               # 2.1 - Recommendation table
│   │   ├── UCBGauge.js                 # 2.2 - Animated gauge
│   │   ├── RecComparison.js            # 2.3 - Side-by-side table
│   │   ├── TrainingDashboard.js        # 3.1 - Dual-pane charts
│   │   ├── AttentionNetwork.js         # 3.2 - Force-directed graph
│   │   ├── ItemHealthDashboard.js      # 3.3 - Scatter + alerts
│   │   ├── TrainingCurves.js           # 3.4 - Multi-metric explorer
│   │   ├── FeatureImportance.js        # 3.5 - Horizontal bar
│   │   ├── PipelineFlow.js             # 4.1 - Animated Sankey
│   │   ├── CoverageHeatmap.js          # 4.2 - User×skill density
│   │   ├── SequenceQuality.js          # 4.3 - Histogram + metrics
│   │   ├── SplitIntegrity.js           # 4.4 - Validation metrics
│   │   ├── SchemaValidation.js         # 4.5 - Compliance checklist
│   │   ├── JoinabilityGauge.js         # 4.6 - Venn + metrics
│   │   ├── LineageMap.js               # 5.1 - Dependency graph
│   │   ├── ThroughputMonitoring.js     # 5.2 - Bottleneck metrics
│   │   ├── JoinOverview.js             # 5.3 - 3-way Venn
│   │   └── DriftAlerts.js              # 5.4 - Animated trends
│   └── utils/
│       ├── DataLoader.js               # Fetch & cache JSON
│       ├── PlotlyHelpers.js            # Common configs (themes, transitions)
│       ├── Dropdown.js                 # Reusable dropdown widget
│       └── MetricCard.js               # Reusable metric display
├── data/
│   ├── student_dashboard.json
│   ├── mastery_timeline.json
│   ├── explainability_sample.json
│   └── ... (21 more JSON files)
└── scripts/
    └── export_all_visuals.py
```

### Utility Modules

**DataLoader.js** - Centralized data fetching with caching:
```javascript
class DataLoader {
  constructor() {
    this.cache = new Map();
  }

  async load(filename) {
    if (this.cache.has(filename)) {
      return this.cache.get(filename);
    }

    const response = await fetch(`data/${filename}`);
    if (!response.ok) throw new Error(`Failed to load ${filename}`);

    const data = await response.json();
    this.cache.set(filename, data);
    return data;
  }

  clearCache() {
    this.cache.clear();
  }
}

export default new DataLoader(); // Singleton
```

**PlotlyHelpers.js** - Common configurations:
```javascript
export const darkTheme = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: '#1b1f2a',
  font: { color: '#f5f5f5', family: 'Inter, sans-serif' }
};

export const smoothTransition = {
  transition: { duration: 500, easing: 'cubic-in-out' }
};

export const interactiveConfig = {
  hovermode: 'closest',
  dragmode: 'zoom',
  displayModeBar: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d']
};
```

**Dropdown.js** - Reusable selector:
```javascript
class Dropdown {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = {
      label: 'Select',
      onChange: () => {},
      ...options
    };
    this.element = null;
  }

  render(items, selectedValue = null) {
    const container = document.getElementById(this.containerId);
    container.innerHTML = `
      <label>${this.options.label}</label>
      <select id="${this.containerId}-select">
        ${items.map(item => `
          <option value="${item.value}" ${item.value === selectedValue ? 'selected' : ''}>
            ${item.label}
          </option>
        `).join('')}
      </select>
    `;

    this.element = container.querySelector('select');
    this.element.addEventListener('change', (e) => {
      this.options.onChange(e.target.value);
    });

    return this;
  }
}

export default Dropdown;
```

**MetricCard.js** - Reusable metric display:
```javascript
class MetricCard {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = {
      title: '',
      formatter: (v) => v,
      ...options
    };
  }

  render(value, delta = null) {
    const container = document.getElementById(this.containerId);
    container.className = 'metric-card';
    container.innerHTML = `
      <span class="metric-title">${this.options.title}</span>
      <strong class="metric-value">${this.options.formatter(value)}</strong>
      ${delta !== null ? `<small class="metric-delta ${delta >= 0 ? 'positive' : 'negative'}">${delta >= 0 ? '+' : ''}${delta}</small>` : ''}
    `;
    return this;
  }
}

export default MetricCard;
```

### Main App Orchestration

**app.js** - Wires everything together:
```javascript
import DataLoader from './utils/DataLoader.js';
import Dropdown from './utils/Dropdown.js';
import MasteryTimeline from './components/MasteryTimeline.js';
import SkillRadar from './components/SkillRadar.js';
// ... import all 24 components

class DashboardApp {
  constructor() {
    this.components = {};
    this.currentStudent = null;
  }

  async init() {
    // Initialize student selector
    const students = await DataLoader.load('student_list.json');
    const selector = new Dropdown('student-selector', {
      label: 'Select Student',
      onChange: (studentId) => this.onStudentChange(studentId)
    });
    selector.render(students.map(s => ({
      value: s.user_id,
      label: s.name || s.user_id
    })), students[0].user_id);

    // Initialize Section 1 components
    await this.initSection1();

    // Initialize other sections (lazy-loaded)
    this.setupLazyLoading();
  }

  async initSection1() {
    const data = await DataLoader.load('mastery_timeline.json');

    this.components.masteryTimeline = new MasteryTimeline('mastery-timeline', {
      animationDuration: 300
    });
    await this.components.masteryTimeline.render(data);

    this.components.skillRadar = new SkillRadar('skill-radar');
    await this.components.skillRadar.render(data);

    // ... initialize other Section 1 components
  }

  async onStudentChange(studentId) {
    this.currentStudent = studentId;
    // Reload data for new student and update all components
    const data = await DataLoader.load(`student_${studentId}.json`);

    Object.values(this.components).forEach(component => {
      if (component.update) {
        component.update(data);
      }
    });
  }

  setupLazyLoading() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && !entry.target.dataset.loaded) {
          const section = entry.target.dataset.section;
          this.loadSection(section);
          entry.target.dataset.loaded = 'true';
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('[data-section]').forEach(el => observer.observe(el));
  }

  async loadSection(sectionName) {
    switch(sectionName) {
      case 'recommendations':
        await this.initSection2();
        break;
      case 'models':
        await this.initSection3();
        break;
      // ... other sections
    }
  }
}

// Initialize app when DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new DashboardApp();
  app.init();
});
```

### Benefits of Modular Architecture

1. **Reusability** - Each component can be used multiple times with different data/config
2. **Testability** - Each component can be unit tested independently
3. **Maintainability** - Changes to one component don't affect others
4. **Composability** - Combine components to create complex dashboards
5. **Lazy Loading** - Load sections on-demand for faster initial page load
6. **Configurability** - Each component accepts options for customization
7. **Consistency** - Shared utilities ensure consistent UX (transitions, themes)

## Animation & Interactivity Philosophy

Visualizations tell dynamic stories through:
- **Temporal Animations** - Play through time with controls (mastery timelines, training curves)
- **Smooth Transitions** - 500ms animations when data updates
- **Rich Hover Details** - Full context cards, not just tooltips
- **Zoom & Pan** - All scatter/heatmap/network graphs are explorable
- **Interactive Filters** - Dropdowns to switch students/items/skills

**Inspiration:** [Central Flows](https://centralflows.github.io/) - animated Sankey diagrams with flowing particles

---

## All 24 Visualizations

### Section 1: Student Insights (6 visualizations)

**1.1 Student Dashboard**
- **Type:** Metrics + Bar Chart
- **Data:** `student_dashboard.json` (from `skill_mastery.parquet`)
- **UI:**
  - Dropdown: Select student
  - 3 metric cards: Avg Mastery, Total Skills, Confidence Score
  - Bar chart: Skill mastery distribution (20 bins)
  - Table: Recent activity (last 10 interactions)
- **Interactivity:** Rich hover on skill bars showing skill names and counts

**1.2 Student Mastery Timeline** ✅ ANIMATED
- **Type:** Animated Line Chart
- **Data:** `mastery_timeline.json` (from `sakt_student_state.parquet`)
- **UI:** Time-series showing mastery evolution per skill for selected student
- **Animation:** Play through sequence positions (300ms per frame), play/pause/slider controls
- **Implementation:** `animation_frame='sequence_position'` in Plotly

**1.3 Explainability Card**
- **Type:** Rich Text Card
- **Data:** `explainability_sample.json` (from `sakt_attention.parquet` + `canonical_events.parquet`)
- **UI:** WHY was prediction made? Top 5 influential past interactions with:
  - Item ID + skill name
  - Correctness (✅/❌)
  - Attention weight (%)
  - Time delta from current interaction
- **Styling:** Emoji indicators, gradient bars for weights

**1.4 Skill Radar Chart**
- **Type:** Polar Scatter
- **Data:** `skill_radar.json` (from `skill_mastery.parquet`)
- **UI:** Radar showing student's mastery across top 8-10 skills
- **Interactivity:** Smooth transitions when switching students (500ms)

**1.5 Gaming Detection Console**
- **Type:** Alert Panel + Table
- **Data:** `gaming_alerts.json` (from `canonical_events.parquet` → run `gaming_detection.py`)
- **UI:**
  - Alerts: Flagged students with severity colors (red/yellow/green)
  - Metrics: Rapid guess rate, help abuse %, suspicious patterns
  - Expandable table: Detailed gaming behavior per student
- **Styling:** Severity badges, sortable columns

**1.6 Attention Heatmap** ⚡ INTERACTIVE
- **Type:** 2D Heatmap
- **Data:** `attention_heatmap.json` (from `sakt_attention.parquet`)
- **UI:** Rows=query positions, cols=key positions, color=attention weight
- **Interactivity:**
  - Zoom/pan to explore patterns
  - Rich hover: item_id, correctness, skill, timestamp
  - Smooth transitions when switching students

---

### Section 2: Recommendations (3 visualizations)

**2.1 RL Recommendation Explorer**
- **Type:** Table + Metrics
- **Data:** `rl_recommendations.json` (from `bandit_state.npz` + `item_params.parquet`)
- **UI:**
  - Dropdown: Select student
  - Table: Top 10 items with columns:
    - Item ID
    - Expected reward (%)
    - Uncertainty (%)
    - Mode (Explore/Exploit) - color coded
  - Metric card: Exploration ratio (% explore vs exploit)

**2.2 UCB Confidence Gauge** ⚡ ANIMATED
- **Type:** Gauge Chart
- **Data:** Same as 2.1
- **UI:** Gauge showing confidence (expected - uncertainty) for top recommendation
- **Animation:** Needle animates to new position when student changes (500ms transition)

**2.3 RL vs Rule-Based Comparison**
- **Type:** Side-by-side Table
- **Data:** `rec_comparison.json` (RL from bandit, Rule-based: next hardest item in weakest skill)
- **UI:**
  - Two columns: RL Recommendations | Rule-Based Recommendations
  - Highlight differences (items unique to each approach)
  - Metric: Overlap percentage

---

### Section 3: Model Performance (5 visualizations)

**3.1 Training Dashboard** ✅ ANIMATED
- **Type:** Dual-pane Animated Line Charts
- **Data:** `training_metrics.json` (from `checkpoints/*/metrics.csv`)
- **UI:**
  - Left pane: SAKT (train/val AUC, loss over epochs)
  - Right pane: WD-IRT (train/val AUC, loss over epochs)
- **Animation:** Play through epochs (300ms per frame), show train/val gap with shaded area
- **Interactivity:** Hover to see all metrics at each epoch

**3.2 Attention Mapping Visualization** ✅ ANIMATED + INTERACTIVE
- **Type:** Force-Directed Network Graph
- **Data:** `attention_network.json` (from `sakt_attention.parquet`)
- **UI:**
  - Nodes = interactions (size by recency, color by correctness)
  - Edges = attention weights (thickness proportional to weight)
- **Animation:** Edges pulse with attention strength
- **Interactivity:**
  - Click node → show item details (item_id, skill, timestamp, correctness)
  - Zoom/pan to explore
  - Drag nodes to rearrange (optional)

**3.3 Item Health Dashboard**
- **Type:** Scatter Plot + Alert Table
- **Data:** `item_health.json` (from `item_params.parquet` + `item_drift.parquet`)
- **UI:**
  - Scatter: x=difficulty, y=discrimination, color=drift_score
  - Threshold lines: Min discrimination (0.5), Drift alert (>0.3)
  - Alert table: Items with high drift or poor discrimination (filterable)

**3.4 Training Curves (Deep Dive)**
- **Type:** Multi-Series Line Chart
- **Data:** Same as 3.1 but with more granular control
- **UI:**
  - Multi-select: Choose metrics (AUC, Loss, Accuracy, F1)
  - Toggle: Show train/val gap shading
  - Slider: Smoothing factor (moving average window)
- **Interactivity:** Rich hover with all selected metrics at each epoch

**3.5 Feature Importance (WD-IRT)**
- **Type:** Horizontal Bar Chart
- **Data:** `feature_importance.json` (extract from `checkpoints/wd_irt/*/latest.ckpt`)
- **UI:** Bar chart showing L1 norms of:
  - Wide features (user_id, item_id, skill_id embeddings)
  - Deep features (clickstream sequence embeddings)
- **Computation:** Load checkpoint, compute `L1_norm(weights)` per feature group, normalize

---

### Section 4: Data Quality (6 visualizations)

**4.1 Canonical Event Flow** ✅ ANIMATED
- **Type:** Animated Sankey Diagram
- **Data:** `pipeline_flow.json` (from `data/interim/`, `data/processed/`)
- **UI:** Raw CSV → Canonical → SAKT Prep + WD-IRT Prep (with record counts)
- **Animation:** Flow particles animate through pipeline stages
- **Interactivity:** Hover shows exact counts, click to highlight path

**4.2 Coverage Heatmap**
- **Type:** 2D Heatmap
- **Data:** `coverage_heatmap.json` (from `canonical_events.parquet`)
- **UI:** Rows=users (sampled), cols=skills, color=interaction count
- **Interactivity:**
  - Zoom into sparse regions
  - Hover: user_id, skill, count
- **Insight:** Highlight cold-start users/skills

**4.3 Sequence Quality Metrics**
- **Type:** Histogram + Metrics
- **Data:** `sequence_quality.json` (from `data/sakt_prepared/train.csv`)
- **UI:**
  - Histogram: Sequence lengths (bins=20)
  - Metrics: Avg length, Median, % Padded, % Truncated
  - Threshold markers: Min (10), Max (200), Padding cutoff (50)

**4.4 Train/Val/Test Split Integrity**
- **Type:** Metrics Panel + Bar Chart
- **Data:** `split_integrity.json` (from `data/sakt_prepared/*.csv`, `data/wd_irt_prepared/*.parquet`)
- **UI:**
  - Metrics: Split ratios (train%, val%, test%)
  - Bar chart: User count per split
  - Alerts: User overlap (should be 0), Temporal leakage check
- **Validation:** Ensure no user appears in multiple splits

**4.5 Schema Validation Dashboard**
- **Type:** Checklist + Status Badges
- **Data:** `schema_validation.json` (validate `canonical_events.parquet` against `schemas.py`)
- **UI:** Checklist with ✅/❌ badges:
  - Required columns present
  - Correct data types
  - No nulls in required fields
  - Valid value ranges (timestamps, IDs)
  - Foreign key integrity (item_id, skill_id exist)

**4.6 Joinability Gauge**
- **Type:** Metrics + Venn Diagram (Plotly shapes)
- **Data:** `joinability_metrics.json` (from `canonical_events`, `sakt_predictions`, `skill_mastery`)
- **UI:**
  - Metrics: % events with valid user_id, item_id, skill_id
  - Venn diagram: Overlap between canonical, predictions, mastery (user_id intersections)
  - Drill-down: Show orphaned records (events without predictions)

---

### Section 5: Pipeline Health (4 visualizations)

**5.1 Data Lineage Map** ✅ INTERACTIVE
- **Type:** Interactive Dependency Graph
- **Data:** `lineage_graph.json` (file system metadata: sizes, timestamps, dependencies)
- **UI:**
  - Nodes = files (raw CSVs, parquet, checkpoints)
  - Edges = dependencies (A produces B)
  - Color by staleness: green (fresh <1 day), yellow (1-7 days), red (>7 days)
- **Interactivity:**
  - Zoom/pan to explore
  - Click node → show file details (size, last modified, path)
  - Hover edge → show transformation (e.g., "data_pipeline.py")
- **Implementation:** Use networkx for layout, Plotly Graph Objects for rendering

**5.2 Throughput Monitoring**
- **Type:** Metrics + Bar Chart
- **Data:** `throughput_metrics.json` (event counts, file sizes, timestamps)
- **UI:**
  - Metrics: Total events processed, Processing rate (events/sec), Bottleneck stage
  - Bar chart: Events per pipeline stage (Raw → Canonical → SAKT/WD-IRT)
  - Bottleneck highlight: Stage with lowest throughput ratio

**5.3 Data Join Overview**
- **Type:** Venn Diagram + Metrics
- **Data:** `join_overview.json` (user_ids from `canonical_events`, `sakt_predictions`, `skill_mastery`)
- **UI:**
  - Venn diagram: 3-way overlap (Plotly shapes/annotations)
  - Metrics:
    - Total unique users in canonical
    - Users with predictions (%)
    - Users with mastery snapshots (%)
    - Full coverage (users in all 3): %

**5.4 Model Drift Alerts** ⚡ ANIMATED
- **Type:** Alert Panel + Animated Line Chart
- **Data:** `drift_alerts.json` (from `item_drift.parquet` + historical predictions)
- **UI:**
  - Alert panel: Items with drift_score > 0.3 (color by severity)
  - Animated line chart: Difficulty evolution over time for flagged items
- **Animation:** Trend lines draw progressively (300ms per time step)
- **Interactivity:** Click item in alert → jump to its trend line

---


## Data Export Strategy

All JSON files generated by `docs/scripts/export_all_visuals.py`:

**Input Sources:**
- `reports/skill_mastery.parquet`
- `reports/sakt_student_state.parquet`
- `reports/sakt_predictions.parquet`
- `reports/sakt_attention.parquet`
- `reports/item_params.parquet`
- `reports/item_drift.parquet`
- `reports/bandit_state.npz`
- `data/interim/edm_cup_2023_42_events.parquet`
- `data/processed/sakt_prepared/*.csv`
- `data/processed/wd_irt_prepared/*.parquet`
- `checkpoints/sakt_edm/metrics.csv`
- `checkpoints/wd_irt_edm/metrics.csv`

**Size Limits:**
- Sample 200 rows max per JSON (except metrics/aggregations)
- For large datasets (canonical events, attention), sample representative subset
- Pre-aggregate where possible (e.g., pivot tables, counts, histograms)

**Schema Enforcement:**
- Each JSON has strict schema for Plotly compatibility
- Include metadata: `{data: [...], metadata: {last_updated: "...", row_count: 123}}`

---

## Implementation Phases

### Phase 1: Foundation (CURRENT - Already Complete)
- ✅ `docs/` structure created
- ✅ Basic `index.md` with 5 sections
- ✅ Basic CSS (`styles.css`)
- ✅ Basic JS (`app.js`) with 9 visualizations
- ✅ Initial export script (`export_docs_assets.py`)

### Phase 2: Expand Data Export (NEXT)
**Goal:** Generate all 24 JSON files

**Tasks:**
1. Rewrite `docs/scripts/export_all_visuals.py` to replace `export_docs_assets.py`
2. Implement exporters for each of the 24 visualizations:
   - Student Dashboard (1.1)
   - Mastery Timeline (1.2) - sample 1-2 students with full sequence
   - Explainability (1.3) - top 5 influences per student sample
   - Skill Radar (1.4) - top 10 skills per student
   - Gaming Alerts (1.5) - run gaming_detection.py, export flagged students
   - Attention Heatmap (1.6) - pivot attention weights into matrix
   - RL Recommendations (2.1-2.2) - load bandit, compute top 10 per student
   - Rec Comparison (2.3) - compare RL vs rule-based
   - Training Metrics (3.1, 3.4) - load metrics CSVs
   - Attention Network (3.2) - build graph from attention weights
   - Item Health (3.3) - join params + drift
   - Feature Importance (3.5) - extract from checkpoint (may need PyTorch)
   - Pipeline Flow (4.1) - count records per stage
   - Coverage Heatmap (4.2) - pivot events → user x skill
   - Sequence Quality (4.3) - parse sequences, compute lengths
   - Split Integrity (4.4) - validate splits
   - Schema Validation (4.5) - run validation checks
   - Joinability (4.6) - compute join metrics
   - Lineage Map (5.1) - scan file system, build dependency graph
   - Throughput (5.2) - compute event counts
   - Join Overview (5.3) - Venn diagram data
   - Drift Alerts (5.4) - filter high-drift items

**Verification:**
- Run `python docs/scripts/export_all_visuals.py`
- Check all 24 JSON files created in `docs/data/`
- Each JSON < 500KB
- Valid JSON syntax (test with `jq`)

### Phase 3: Build Utility Modules
**Goal:** Create reusable utility classes (foundation for all components)

**Tasks:**
1. Create `docs/js/utils/DataLoader.js`:
   - Fetch JSON with caching
   - Error handling for missing files
   - Cache invalidation method
2. Create `docs/js/utils/PlotlyHelpers.js`:
   - Dark theme config
   - Smooth transition defaults
   - Interactive config (zoom/pan)
   - Hover template formatters
3. Create `docs/js/utils/Dropdown.js`:
   - Reusable dropdown component
   - Event handling (onChange callback)
   - Styling helpers
4. Create `docs/js/utils/MetricCard.js`:
   - Metric display component
   - Formatting helpers (%, currency, etc.)
   - Delta indicators (positive/negative)

**Verification:**
- All utility modules export correctly (ES6 modules)
- DataLoader caches JSON correctly
- Dropdown triggers onChange callback
- MetricCard renders with correct formatting

### Phase 4: Redesign HTML Landing Page
**Goal:** Replace `index.md` with rich `index.html` supporting modular components

**Tasks:**
1. Create `docs/index.html` with:
   - ES6 module support (`<script type="module">`)
   - Navigation tabs (5 sections)
   - Section containers with component mount points
   - Plotly.js CDN (v2.32)
   - Loading spinners for each section
2. Add interactive navigation:
   - Click tab → scroll to section
   - Sticky header with section links
   - Active tab highlighting
3. Responsive design (mobile/tablet/desktop)
4. Define component mount points (container divs with IDs)

**Verification:**
- HTML loads without errors
- ES6 modules work (check browser console)
- Navigation scrolls to sections
- Responsive on mobile/tablet/desktop

### Phase 5: Build Component Library - Section 1 (Student Insights)
**Goal:** Create all 6 modular visualization components (1.1-1.6)

**Tasks:**
1. Create component classes:
   - `docs/js/components/StudentDashboard.js` (1.1)
   - `docs/js/components/MasteryTimeline.js` (1.2)
   - `docs/js/components/ExplainabilityCard.js` (1.3)
   - `docs/js/components/SkillRadar.js` (1.4)
   - `docs/js/components/GamingConsole.js` (1.5)
   - `docs/js/components/AttentionHeatmap.js` (1.6)
2. Each component implements:
   - `constructor(containerId, options)`
   - `async render(data)` - initial render
   - `update(newData)` - smooth transition update
   - `destroy()` - cleanup
3. Wire components in `app.js`:
   - Initialize all Section 1 components
   - Connect to student dropdown
   - Handle data loading errors

**Verification:**
- All 6 components render without errors
- Student dropdown updates all components
- Animations smooth (300-500ms transitions)
- Hover details show correctly
- No memory leaks (test with Chrome DevTools)

### Phase 6: Build Component Library - Section 2 (Recommendations)
**Goal:** Create all 3 modular visualization components (2.1-2.3)

**Tasks:**
1. Create component classes:
   - `docs/js/components/RLExplorer.js` (2.1)
   - `docs/js/components/UCBGauge.js` (2.2)
   - `docs/js/components/RecComparison.js` (2.3)
2. Wire components in `app.js`:
   - Lazy-load Section 2 when scrolled into view
   - Connect to student dropdown
3. Add gauge animation on student change

**Verification:**
- RL recommendations display correctly
- Gauge animates smoothly (needle transition)
- Comparison highlights differences (color coding)
- Lazy loading works (check Network tab)

### Phase 7: Build Component Library - Section 3 (Model Performance)
**Goal:** Create all 5 modular visualization components (3.1-3.5)

**Tasks:**
1. Create component classes:
   - `docs/js/components/TrainingDashboard.js` (3.1)
   - `docs/js/components/AttentionNetwork.js` (3.2)
   - `docs/js/components/ItemHealthDashboard.js` (3.3)
   - `docs/js/components/TrainingCurves.js` (3.4)
   - `docs/js/components/FeatureImportance.js` (3.5)
2. Implement:
   - `renderTrainingDashboard(data)` - dual-pane animated lines
   - `renderAttentionNetwork(data)` - force-directed graph
   - `renderItemHealth(data)` - scatter + alert table
   - `renderTrainingCurves(data)` - multi-select metrics, smoothing
   - `renderFeatureImportance(data)` - horizontal bar
3. Add play controls for animated training curves
4. Implement network graph with zoom/pan/click

**Verification:**
- Training animations play smoothly
- Network graph is interactive
- Item health scatter highlights drift correctly
- Feature importance loads from checkpoint data

### Phase 7: Implement Section 4 (Data Quality)
**Goal:** Build all 6 visualizations (4.1-4.6)

**Tasks:**
1. Create `docs/js/section4_quality.js`
2. Implement:
   - `renderPipelineFlow(data)` - animated Sankey
   - `renderCoverageHeatmap(data)` - 2D heatmap with zoom
   - `renderSequenceQuality(data)` - histogram + metrics
   - `renderSplitIntegrity(data)` - metrics + bar chart
   - `renderSchemaValidation(data)` - checklist with badges
   - `renderJoinabilityGauge(data)` - metrics + Venn diagram
3. Add flow animation to Sankey
4. Implement Venn diagram using Plotly shapes

**Verification:**
- Sankey flows correctly
- Coverage heatmap highlights sparse regions
- Split integrity shows no user overlap
- Schema validation runs all checks
- Venn diagram renders correctly

### Phase 8: Build Component Library - Section 4 (Data Quality)
**Goal:** Create all 6 modular visualization components (4.1-4.6)

**Tasks:**
1. Create component classes:
   - `docs/js/components/PipelineFlow.js` (4.1)
   - `docs/js/components/CoverageHeatmap.js` (4.2)
   - `docs/js/components/SequenceQuality.js` (4.3)
   - `docs/js/components/SplitIntegrity.js` (4.4)
   - `docs/js/components/SchemaValidation.js` (4.5)
   - `docs/js/components/JoinabilityGauge.js` (4.6)
2. Wire components in `app.js` with lazy loading

**Verification:**
- Sankey flows correctly with animations
- Coverage heatmap highlights sparse regions
- Split integrity shows no user overlap
- Schema validation runs all checks
- Venn diagram renders correctly

### Phase 9: Build Component Library - Section 5 (Pipeline Health)
**Goal:** Create all 4 modular visualization components (5.1-5.4)

**Tasks:**
1. Create component classes:
   - `docs/js/components/LineageMap.js` (5.1)
   - `docs/js/components/ThroughputMonitoring.js` (5.2)
   - `docs/js/components/JoinOverview.js` (5.3)
   - `docs/js/components/DriftAlerts.js` (5.4)
2. Implement:
   - `renderLineageMap(data)` - interactive dependency graph
   - `renderThroughputMonitoring(data)` - metrics + bar chart
   - `renderJoinOverview(data)` - Venn diagram + metrics
   - `renderDriftAlerts(data)` - alert panel + animated trends
3. Use networkx-style layout for lineage map
4. Add progressive line drawing for drift trends

**Verification:**
- Lineage map shows file dependencies correctly
- Throughput identifies bottlenecks
- Join overview shows user overlaps
- Drift alerts highlight high-drift items

### Phase 10: Polish & Optimization
**Goal:** Performance, UX, accessibility

**Tasks:**
1. **Performance:**
   - Lazy-load sections (only render when scrolled into view)
   - Debounce dropdown changes (300ms)
   - Cache rendered Plotly charts
   - Optimize JSON file sizes (gzip if >200KB)
2. **UX:**
   - Add loading spinners while JSONs load
   - Error messages for missing data
   - "Last Updated" timestamp per section
   - Print-friendly CSS (`@media print`)
3. **Accessibility:**
   - ARIA labels on all interactive elements
   - Keyboard navigation (tab through dropdowns)
   - High-contrast mode toggle
4. **Documentation:**
   - Update `docs/README.md` with:
     - How to regenerate data (`python docs/scripts/export_all_visuals.py`)
     - How to deploy (GitHub Pages settings)
     - Troubleshooting (missing data, broken charts)

**Verification:**
- Page loads in <3 seconds
- All sections lazy-load correctly
- No console errors
- Works on mobile/tablet/desktop
- Passes basic accessibility checks (axe DevTools)

### Phase 11: Deploy & Document
**Goal:** Go live on GitHub Pages

**Tasks:**
1. Enable GitHub Pages:
   - Repo settings → Pages → Source: main branch, /docs folder
2. Update main README.md:
   - Add link to live site
   - Badge: `[![Dashboard](https://img.shields.io/badge/Dashboard-Live-blue)](https://nayanachandrika99.github.io/deepKT-IRT/)`
3. Test live site:
   - All 24 visualizations render
   - Navigation works
   - Mobile responsive
4. Create `docs/CHANGELOG.md` to track updates

**Verification:**
- Site accessible at https://nayanachandrika99.github.io/deepKT-IRT/
- All features work in production
- README links correctly

---

## Success Criteria

- ✅ All 24 visualizations implemented and rendering correctly
- ✅ Animations smooth (60fps, no jank)
- ✅ Interactive elements responsive (<100ms)
- ✅ Page load time <3 seconds (with caching)
- ✅ JSON exports automated and documented
- ✅ Mobile/tablet/desktop responsive
- ✅ No JavaScript errors in console
- ✅ Stakeholders can navigate without training
- ✅ Live on GitHub Pages

---

## Technical Notes

### Plotly.js Animation Examples

**Animated Timeline:**
```javascript
const frames = data.map(point => ({
  name: point.sequence_position,
  data: [{x: data.slice(0, point.sequence_position).map(d => d.x),
          y: data.slice(0, point.sequence_position).map(d => d.y)}]
}));

Plotly.newPlot('chart', data, layout, {frames: frames});
```

**Smooth Transitions:**
```javascript
const layout = {
  transition: {duration: 500, easing: 'cubic-in-out'},
  hovermode: 'closest',
  dragmode: 'zoom'
};
```

**Rich Hover Templates:**
```javascript
const trace = {
  x: data.map(d => d.x),
  y: data.map(d => d.y),
  customdata: data.map(d => [d.item_id, d.skill, d.correctness]),
  hovertemplate: '<b>Item:</b> %{customdata[0]}<br>' +
                 '<b>Skill:</b> %{customdata[1]}<br>' +
                 '<b>Correct:</b> %{customdata[2]}<br>' +
                 '<extra></extra>'
};
```

### Student Selector Implementation

Use vanilla JS dropdown that updates all charts in a section:

```javascript
function initStudentSelector(students, onSelect) {
  const select = document.getElementById('student-select');
  select.innerHTML = students.map(s =>
    `<option value="${s.user_id}">${s.name || s.user_id}</option>`
  ).join('');

  select.addEventListener('change', (e) => {
    onSelect(e.target.value);
  });

  // Trigger initial load
  onSelect(students[0].user_id);
}
```

### Lazy Loading Sections

Only render sections when scrolled into view:

```javascript
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting && !entry.target.dataset.loaded) {
      const section = entry.target.id;
      loadSection(section);
      entry.target.dataset.loaded = 'true';
    }
  });
}, {threshold: 0.1});

document.querySelectorAll('.section-container').forEach(el => observer.observe(el));
```

---

## Future Enhancements (Out of Scope)

- Real-time data refresh (WebSocket connection)
- Export charts as PNG/PDF
- User authentication (view student-level data)
- A/B testing dashboard for RL strategies
- Historical trend analysis (time-series storage)
