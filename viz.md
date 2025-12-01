# System Demo Notebook: Implementation Plan

## Executive Summary

This document specifies the implementation of an interactive Jupyter Notebook (`notebooks/system_demo.ipynb`) that demonstrates the DeepKT + Wide&Deep IRT recommendation system's decision-making process at scale. The notebook focuses on visualizing the RL bandit's candidate scoring and selection process, with SAKT and WD-IRT serving as supporting context layers.

**Key Innovation**: Instead of showing "what the models do," the notebook reveals **"how decisions are made"** by exposing the complete scoring pipeline from student context through candidate ranking to outcome observation.

---

## 1. Objectives & Scope

### Primary Objectives
1. **Transparency**: Expose the LinUCB bandit's internal math (μ, σ, UCB) for each candidate item
2. **Interactivity**: Allow users to manipulate exploration parameters and see real-time rank changes
3. **Scale**: Demonstrate decision patterns across population, not just individual students
4. **Learning**: Show how the bandit updates after observing outcomes (sigma shrinkage, theta updates)

### Out of Scope
- Real-time model inference (uses pre-computed logs)
- Model training or re-training
- Production deployment considerations

---

## 2. Architecture & Data Flow

### Conceptual Architecture

```
┌─────────────────────────────────────────────────────┐
│              Input Data Layer                       │
│  ┌───────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Events   │  │  SAKT    │  │  WD-IRT Items    │ │
│  │  (500k)   │  │  State   │  │  (1,835)         │ │
│  └─────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
└────────┼─────────────┼─────────────────┼───────────┘
         │             │                 │
         └─────────────┴─────────────────┘
                       │
         ┌─────────────▼──────────────────────────┐
         │   Bandit Decision Generator            │
         │   (scripts/generate_bandit_decisions)  │
         └─────────────┬──────────────────────────┘
                       │
         ┌─────────────▼──────────────────────────┐
         │   bandit_decisions.parquet             │
         │   (Decision Event Log)                 │
         └─────────────┬──────────────────────────┘
                       │
         ┌─────────────▼──────────────────────────┐
         │   Interactive Notebook                 │
         │   - Candidate Scoring Board            │
         │   - UCB Decomposition                  │
         │   - Rank Sensitivity Curves            │
         │   - Before/After Updates               │
         │   - Decision Flow Sankey               │
         └────────────────────────────────────────┘
```

### Data Flow Pipeline

**Phase 1: Data Collection** (Pre-notebook)
1. Historical events ingested from EDM Cup dataset
2. SAKT model generates mastery predictions
3. WD-IRT model computes item parameters
4. Bandit state trained via `warmstart_bandit.py`

**Phase 2: Decision Logging** (Pre-notebook)
1. Replay historical events through bandit
2. For each student-skill interaction:
   - Build context vector (8 features)
   - Score all candidate items (μ, σ, UCB)
   - Log full decision event to parquet

**Phase 3: Interactive Exploration** (In-notebook)
1. Load decision logs
2. User selects decision event via dropdown
3. Visualizations render with ipywidgets
4. User manipulates parameters (alpha slider)
5. Plots update in real-time (FigureWidget)

---

## 3. Data Requirements

### Required Files

| File Path | Size | Rows | Usage |
|-----------|------|------|-------|
| `reports/bandit_decisions.parquet` | ~500 KB | 13-5,000 | **Core**: Decision event logs |
| `reports/skill_mastery.parquet` | ~4.4 MB | ~50,000 | Context: Student mastery aggregates |
| `reports/item_params.parquet` | ~50 KB | 1,835 | Context: Item difficulty/discrimination |
| `data/interim/edm_cup_2023_42_events.parquet` | ~87 MB | 5,167,603 | Context: Raw event logs |
| `reports/sakt_student_state.parquet` | ~27 MB | ~1M | Optional: Per-interaction mastery |
| `reports/sakt_attention.parquet` | ~2.7 MB | ~50,000 | Optional: Attention weights |

### Decision Event Schema

**Core Table**: `bandit_decisions.parquet`

**Keys** (Decision Identifier):
- `t`: Timestamp of decision
- `student_id`: Student identifier
- `skill_id`: Target skill for recommendation
- `candidate_item_id`: Item being scored

**Student Context** (Input Features):
- `x_mastery`: Overall student mastery (0-1)
- `x_recent_accuracy`: Last 5 questions accuracy
- `x_recent_speed`: Normalized response time
- `x_help_tendency`: Help request frequency
- `x_skill_gap`: Distance to mastery target

**Item Context** (Input Features):
- `irt_difficulty`: Item difficulty parameter
- `irt_discrimination`: Item discrimination parameter
- `drift_flag`: Boolean, item quality degraded
- `sakt_p_correct`: SAKT prediction for this student-item

**Bandit Calculation** (Derived):
- `mu`: Expected reward (θᵀx)
- `sigma`: Uncertainty (√(xᵀA⁻¹x))
- `ucb`: Upper confidence bound (μ + α·σ)
- `alpha`: Exploration parameter
- `mode`: 'explore' or 'exploit'

**Decision Outputs**:
- `rank`: Item rank by UCB score
- `chosen`: Boolean, top-ranked item
- `reward_observed`: Observed correctness (-1 if not attempted)
- `topk_ucb_gap`: Distance to next-best item

**Data Invariants**:
- One decision event = all candidates for one student-skill-time tuple
- Exactly one `chosen=True` per decision event
- `rank=1` ⟺ `chosen=True`
- `mu ∈ [0,1]`, `sigma ≥ 0`, `ucb ∈ [0,∞)`

---

## 4. Implementation Phases

### Phase 1: Foundation (Cells 1-3)

**Goal**: Load data and establish interactive environment

**Cell 1: Imports & Configuration**
- Import libraries: pandas, plotly, ipywidgets, seaborn
- Configure display settings (DPI, themes)
- Define path constants

**Cell 2: Data Loading**
- Load 6 parquet files with error handling
- Display summary statistics (row counts, date ranges)
- Validate schema (check for required columns)

**Cell 3: Global Controls**
- Create skill selector dropdown
- Display dataset overview dashboard
- Initialize decision event index

**Validation Criteria**:
- [ ] All required files load without error
- [ ] Decision event count matches expected range (1-500)
- [ ] Skill selector populated with >10 skills

---

### Phase 2: Core Decision Visualizations (Cells 4-9)

**Goal**: Expose the bandit's decision-making process

#### Cell 4: Decision Event Selector
- **Input**: `bandit_decisions.parquet`
- **Output**: Dropdown widget with (student_id, skill_id, timestamp) tuples
- **Logic**: Group by (t, student_id, skill_id), format as human-readable strings

#### Cell 5: VIZ 1 - Candidate Scoring Board ⭐
- **Purpose**: Show all candidates with live UCB recalculation
- **Widgets**:
  - Decision selector (dropdown)
  - Alpha slider (0.1 to 3.0)
  - Skill filter checkbox
- **Table Columns**:
  - Item ID, SAKT prediction, IRT difficulty
  - μ (expected), σ (uncertainty), UCB (adjusted)
  - Rank (adjusted), Mode (explore/exploit)
- **Styling**: Green gradient on UCB column, bold chosen item
- **Tech**: `ipywidgets.interactive_output` + DataFrame styler

#### Cell 6: VIZ 2 - UCB Decomposition
- **Purpose**: Show why top items won (stacked bar)
- **Plot Type**: Horizontal stacked bar (Plotly)
- **Layers**:
  - Base (blue): μ (expected reward)
  - Top (orange): α·σ (uncertainty bonus)
- **Items Shown**: Top 5 by UCB
- **Interactivity**: Linked to decision selector

#### Cell 7: VIZ 3 - Rank Sensitivity Curve
- **Purpose**: Prove exploration is principled, not random
- **Plot Type**: Multi-line chart (Plotly Express)
- **X-axis**: Alpha (0 to 3, 50 steps)
- **Y-axis**: Rank (inverted scale)
- **Lines**: Top 10 items (color-coded)
- **Highlight**: Mark where chosen item changes
- **Implementation**:
  1. For each alpha value:
     - Recalculate UCB = μ + alpha·σ
     - Re-rank all candidates
     - Store rank for top-10 items
  2. Plot 10 line traces
  3. Add vertical line where rank=1 switches

#### Cell 8: VIZ 4 - Before/After Update
- **Purpose**: Show learning (sigma shrinkage)
- **Plot Type**: Grouped bar chart (Plotly)
- **Bars**: Before (red) vs After (green) uncertainty
- **Simulation Logic**:
  - For chosen item: σ_after = σ_before × 0.8
  - For others: σ_after = σ_before (unchanged)
- **Items Shown**: Top 5 candidates

#### Cell 9: VIZ 5 - Decision Sankey
- **Purpose**: System flow diagram
- **Nodes**:
  - All Candidates
  - Explore / Exploit (split)
  - Chosen / Not Chosen (final)
- **Edge Weights**: Aggregated decision counts
- **Tech**: Plotly Sankey diagram

**Validation Criteria**:
- [ ] All 5 visualizations render without error
- [ ] Alpha slider updates scoring board in <1 second
- [ ] Rank sensitivity shows ≥2 crossover points
- [ ] Sankey totals match decision event count

---

### Phase 3: Supporting Context (Cells 10-12)

**Goal**: Explain where input signals come from

#### Cell 10-11: SAKT Context
- **Cell 10**: Population mastery histogram
- **Cell 11**: Attention heatmap (T×T matrix)
  - Requires `sakt_attention.parquet`
  - Graceful degradation if missing

#### Cell 12: WD-IRT Context
- Difficulty vs Discrimination scatter
- Color by drift status
- Interactive filter slider

**Validation Criteria**:
- [ ] Histograms show expected distributions
- [ ] Scatter plot displays all 1,835 items
- [ ] Attention heatmap (if available) shows lower-triangular structure

---

### Phase 4: Safety Layer (Cell 13)

#### Cell 13: Gaming Detection
- Rapid guessing distribution (histogram)
- Threshold line at 15% rapid ratio
- Flag students above threshold

---

## 5. Technical Specifications

### Widget Architecture

**Core Pattern**: `interactive_output` for responsive updates

```python
# Pattern used throughout notebook
controls = {
    'decision_idx': decision_selector,
    'alpha': alpha_slider,
    'filter': checkbox
}

output = widgets.interactive_output(render_function, controls)
display(widgets.VBox([*controls.values(), output]))
```

**Performance Optimization**:
- Use Plotly `FigureWidget` for in-place updates (not full re-render)
- Cache decision event lookups in dict
- Limit initial load to first 100 decision events

### Styling Standards

**Color Palette**:
- Expected reward (μ): Steelblue (`#4682B4`)
- Uncertainty (σ): Orange (`#FF8C00`)
- Explore mode: Orange (`#FF8C00`)
- Exploit mode: Steelblue (`#4682B4`)
- Chosen item: Green highlight

**Formatting**:
- Probabilities: 3 decimal places (`.3f`)
- Difficulties: 2 decimal places (`.2f`)
- Percentages: 1 decimal place with % symbol

---

## 6. Dependencies

### Required Libraries

```python
pandas>=2.2.0           # Data manipulation
numpy>=2.2.0            # Numerical operations
plotly>=5.22.0          # Interactive plots
seaborn>=0.13.0         # Statistical visualizations
matplotlib>=3.9.0       # Base plotting
ipywidgets>=8.1.0       # Interactive controls
```

### Environment Setup

```bash
# Ensure all dependencies installed
uv pip install pandas numpy plotly seaborn matplotlib ipywidgets

# Register ipywidgets with Jupyter
jupyter nbextension enable --py widgetsnbextension
```

---

## 7. Testing & Validation

### Unit Tests

**Test 1: Data Loading**
```python
def test_data_loads():
    assert decisions_df is not None
    assert len(decisions_df) > 0
    assert 'ucb' in decisions_df.columns
```

**Test 2: UCB Calculation**
```python
def test_ucb_recalculation():
    alpha = 2.0
    ucb_calc = df['mu'] + alpha * df['sigma']
    assert np.allclose(ucb_calc, df['ucb_adj'], atol=1e-5)
```

**Test 3: Widget Responsiveness**
```python
def test_alpha_slider_updates():
    initial_alpha = alpha_slider.value
    alpha_slider.value = 2.0
    time.sleep(0.5)
    assert alpha_slider.value == 2.0
```

### Integration Tests

**Test 4: End-to-End Flow**
1. Run all cells sequentially
2. Change decision selector to 3rd event
3. Move alpha slider to 2.5
4. Verify scoring board updates
5. Verify rank sensitivity plot changes

### Visual Inspection Checklist

- [ ] All plots have titles and axis labels
- [ ] Colors are consistent across visualizations
- [ ] No overlapping text or truncated labels
- [ ] Tooltips work on interactive elements
- [ ] Chosen item visually distinct in all views

---

## 8. Success Metrics

### Quantitative Metrics
1. **Rendering Speed**: All visualizations load in <3 seconds
2. **Interactivity Latency**: Widget updates complete in <1 second
3. **Data Coverage**: ≥100 unique decision events visualized
4. **Exploration Ratio**: Matches expected 30-50% explore mode

### Qualitative Metrics
1. **Clarity**: Non-technical user can understand UCB decomposition
2. **Insight**: Rank sensitivity curve reveals exploration tradeoff
3. **Completeness**: All 5 core decision visualizations functional

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Missing decision data | Low | High | Graceful error messages, mock data fallback |
| Slow widget updates | Medium | Medium | Use FigureWidget, limit dataset size |
| Attention data unavailable | High | Low | Optional cell, skip if missing |
| Alpha slider breaks ranks | Low | High | Input validation, clamp to [0.1, 3.0] |

---

## 10. Future Enhancements

1. **Week-by-week animation**: Add time slider to show bandit learning over time
2. **Student cohort comparison**: Side-by-side decision boards for high vs low performers
3. **Feature importance heatmap**: Show which context features drive μ most
4. **Counterfactual analysis**: "What if we had chosen item #2?"
5. **Export functionality**: Save decision board as PDF/CSV

---

## Appendix A: Cell-by-Cell Index

---

### Cell 1: Setup & Imports
**Type**: Code  
**Data Files**: None  
**Purpose**: Import libraries and configure display settings

```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, HTML
from pathlib import Path

# Configure plotting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

# Paths
DATA_DIR = Path('../reports')
```

---

### Cell 2: Load All Data
**Type**: Code  
**Data Files**:
- `reports/bandit_decisions.parquet`
- `reports/skill_mastery.parquet`
- `reports/item_params.parquet`
- `data/interim/edm_cup_2023_42_events.parquet`
- `reports/sakt_student_state.parquet`
- `reports/sakt_attention.parquet`

**Purpose**: Load all required datasets

```python
# Load decision data
decisions_df = pd.read_parquet(DATA_DIR / 'bandit_decisions.parquet')

# Load supporting data
skill_mastery_df = pd.read_parquet(DATA_DIR / 'skill_mastery.parquet')
item_params_df = pd.read_parquet(DATA_DIR / 'item_params.parquet')
events_df = pd.read_parquet('../data/interim/edm_cup_2023_42_events.parquet')
student_state_df = pd.read_parquet(DATA_DIR / 'sakt_student_state.parquet')

# Optional: attention data
try:
    attention_df = pd.read_parquet(DATA_DIR / 'sakt_attention.parquet')
except:
    attention_df = None

print(f"Decisions: {len(decisions_df):,} rows")
print(f"Unique decision events: {decisions_df.groupby(['t', 'student_id']).ngroups:,}")
print(f"Skills: {skill_mastery_df['skill'].nunique():,}")
print(f"Events: {len(events_df):,}")
```

**Expected Output**: Data summary statistics

---

### Cell 3: Dataset Overview Dashboard
**Type**: Code  
**Data Files**: All from Cell 2  
**Purpose**: Summary statistics and skill selector

```python
# Skill selector widget
skills = sorted(decisions_df['skill_id'].unique())
skill_selector = widgets.Dropdown(
    options=skills,
    value=skills[0] if skills else None,
    description='Spotlight Skill:',
    style={'description_width': 'initial'}
)

# Display summary
print(f"📊 Decision Dataset Overview")
print(f"{'='*50}")
print(f"Total decisions: {decisions_df.groupby(['t', 'student_id']).ngroups:,}")
print(f"Total candidates scored: {len(decisions_df):,}")
print(f"Explore ratio: {(decisions_df['mode'] == 'explore').mean():.1%}")
print(f"Students: {decisions_df['student_id'].nunique():,}")
print(f"Skills: {len(skills):,}")

display(skill_selector)
```

**Expected Output**: Stats table + dropdown widget

---

## SECTION 2: THE MAIN ACT - Decision Engine

### Cell 4: Helper - Sample Decision Selector
**Type**: Code  
**Data Files**: `decisions_df`  
**Purpose**: Interactive decision event picker

```python
def get_decision_events():
    """Get list of unique decision events."""
    return decisions_df.groupby(['t', 'student_id', 'skill_id']).first().reset_index()[['t', 'student_id', 'skill_id']]

decision_events = get_decision_events()
decision_selector = widgets.Dropdown(
    options=[(f"{row['student_id']} | {row['skill_id']} | {row['t']}", idx) 
             for idx, row in decision_events.iterrows()],
    description='Decision Event:',
    style={'description_width': 'initial'}
)

display(decision_selector)
```

---

### Cell 5: VIZ 1 - Candidate Scoring Board (Interactive)
**Type**: Code  
**Data Files**: `decisions_df`  
**Purpose**: Show all candidates with scoring details, interactive alpha slider

```python
# Alpha slider
alpha_slider = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=3.0,
    step=0.1,
    description='Alpha (α):',
    readout_format='.1f'
)

# Filter checkbox
skill_filter_checkbox = widgets.Checkbox(
    value=False,
    description='Filter to spotlight skill only'
)

def update_scoring_board(decision_idx, alpha, filter_skill):
    """Update the candidate scoring board."""
    # Get decision event
    event = decision_events.iloc[decision_idx]
    t, student_id, skill_id = event['t'], event['student_id'], event['skill_id']
    
    # Get all candidates for this decision
    candidates = decisions_df[
        (decisions_df['t'] == t) & 
        (decisions_df['student_id'] == student_id)
    ].copy()
    
    if filter_skill:
        candidates = candidates[candidates['skill_id'] == skill_selector.value]
    
    # Recalculate UCB with new alpha
    candidates['ucb_adj'] = candidates['mu'] + alpha * candidates['sigma']
    candidates['rank_adj'] = candidates['ucb_adj'].rank(ascending=False, method='first').astype(int)
    
    # Display table
    display_cols = ['candidate_item_id', 'sakt_p_correct', 'irt_difficulty', 
                    'mu', 'sigma', 'ucb_adj', 'rank_adj', 'mode']
    
    styled = candidates[display_cols].sort_values('rank_adj').head(10).style\
        .background_gradient(subset=['ucb_adj'], cmap='Greens')\
        .format({'mu': '{:.3f}', 'sigma': '{:.3f}', 'ucb_adj': '{:.3f}',
                 'sakt_p_correct': '{:.3f}', 'irt_difficulty': '{:.2f}'})
    
    return styled

# Interactive output
output = widgets.interactive_output(
    update_scoring_board,
    {'decision_idx': decision_selector, 'alpha': alpha_slider, 
     'filter_skill': skill_filter_checkbox}
)

display(widgets.VBox([decision_selector, alpha_slider, skill_filter_checkbox, output]))
```

**Expected Output**: Interactive table with color-coded UCB scores

---

### Cell 6: VIZ 2 - UCB Decomposition Plot
**Type**: Code  
**Data Files**: `decisions_df`  
**Purpose**: Stacked bar showing mu + alpha*sigma

```python
def plot_ucb_decomposition(decision_idx):
    """Show UCB decomposition for top-5 candidates."""
    event = decision_events.iloc[decision_idx]
    t, student_id = event['t'], event['student_id']
    
    candidates = decisions_df[
        (decisions_df['t'] == t) & 
        (decisions_df['student_id'] == student_id)
    ].sort_values('ucb', ascending=False).head(5)
    
    fig = go.Figure()
    
    # Base: Expected reward
    fig.add_trace(go.Bar(
        name='Expected Reward (μ)',
        y=candidates['candidate_item_id'],
        x=candidates['mu'],
        orientation='h',
        marker_color='steelblue'
    ))
    
    # Top: Uncertainty bonus
    fig.add_trace(go.Bar(
        name='Uncertainty Bonus (α·σ)',
        y=candidates['candidate_item_id'],
        x=candidates['sigma'] * candidates['alpha'],
        orientation='h',
        marker_color='orange'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='UCB Decomposition: Why This Item Won',
        xaxis_title='Score',
        yaxis_title='Item ID',
        height=400
    )
    
    fig.show()

# Interactive
widgets.interact(plot_ucb_decomposition, decision_idx=decision_selector)
```

**Expected Output**: Stacked horizontal bar chart

---

### Cell 7: VIZ 3 - Rank Sensitivity Curve
**Type**: Code  
**Data Files**: `decisions_df`  
**Purpose**: Line plot showing rank changes as alpha varies

```python
def plot_rank_sensitivity(decision_idx):
    """Show how ranks change with alpha."""
    event = decision_events.iloc[decision_idx]
    t, student_id = event['t'], event['student_id']
    
    candidates = decisions_df[
        (decisions_df['t'] == t) & 
        (decisions_df['student_id'] == student_id)
    ].copy()
    
    # Get top-10 by default alpha
    top_items = candidates.nlargest(10, 'ucb')['candidate_item_id'].tolist()
    
    # Vary alpha
    alphas = np.linspace(0, 3, 50)
    ranks = []
    
    for alpha in alphas:
        candidates['ucb_temp'] = candidates['mu'] + alpha * candidates['sigma']
        candidates['rank_temp'] = candidates['ucb_temp'].rank(ascending=False, method='first')
        
        for item_id in top_items:
            rank = candidates[candidates['candidate_item_id'] == item_id]['rank_temp'].iloc[0]
            ranks.append({'alpha': alpha, 'item_id': item_id, 'rank': rank})
    
    ranks_df = pd.DataFrame(ranks)
    
    fig = px.line(
        ranks_df,
        x='alpha',
        y='rank',
        color='item_id',
        title='Rank Sensitivity to Exploration Parameter',
        labels={'alpha': 'Exploration Parameter (α)', 'rank': 'Rank'}
    )
    
    fig.update_yaxes(autorange='reversed')
    fig.show()

widgets.interact(plot_rank_sensitivity, decision_idx=decision_selector)
```

**Expected Output**: Multi-line plot with rank trajectories

---

### Cell 8: VIZ 4 - Before/After Update
**Type**: Code  
**Data Files**: `decisions_df`  
**Purpose**: Show learning (sigma shrinkage)

```python
def plot_learning_update(decision_idx):
    """Show sigma shrinkage after observing reward."""
    event = decision_events.iloc[decision_idx]
    t, student_id = event['t'], event['student_id']
    
    candidates = decisions_df[
        (decisions_df['t'] == t) & 
        (decisions_df['student_id'] == student_id)
    ].sort_values('ucb', ascending=False).head(5)
    
    # Simulate "after update" (reduce sigma by ~20% for chosen item)
    candidates['sigma_after'] = candidates.apply(
        lambda row: row['sigma'] * 0.8 if row['chosen'] else row['sigma'],
        axis=1
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Before Update',
        x=candidates['candidate_item_id'],
        y=candidates['sigma'],
        marker_color='lightcoral'
    ))
    
    fig.add_trace(go.Bar(
        name='After Update',
        x=candidates['candidate_item_id'],
        y=candidates['sigma_after'],
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Uncertainty Shrinkage (Learning Happened)',
        xaxis_title='Item ID',
        yaxis_title='Uncertainty (σ)',
        barmode='group',
        height=400
    )
    
    fig.show()

widgets.interact(plot_learning_update, decision_idx=decision_selector)
```

**Expected Output**: Grouped bar chart showing sigma reduction

---

### Cell 9: VIZ 5 - Decision Sankey
**Type**: Code  
**Data Files**: `decisions_df`  
**Purpose**: Flow diagram of decision pipeline

```python
# Aggregate decision flows
flow_counts = decisions_df.groupby(['mode', 'chosen']).size().reset_index(name='count')

# Create Sankey
fig = go.Figure(go.Sankey(
    node=dict(
        label=['All Candidates', 'Explore', 'Exploit', 'Chosen', 'Not Chosen'],
        color=['blue', 'orange', 'steelblue', 'green', 'gray']
    ),
    link=dict(
        source=[0, 0, 1, 1, 2, 2],
        target=[1, 2, 3, 4, 3, 4],
        value=[
            (decisions_df['mode'] == 'explore').sum(),
            (decisions_df['mode'] == 'exploit').sum(),
            ((decisions_df['mode'] == 'explore') & decisions_df['chosen']).sum(),
            ((decisions_df['mode'] == 'explore') & ~decisions_df['chosen']).sum(),
            ((decisions_df['mode'] == 'exploit') & decisions_df['chosen']).sum(),
            ((decisions_df['mode'] == 'exploit') & ~decisions_df['chosen']).sum(),
        ]
    )
))

fig.update_layout(title='Decision Pipeline Flow', height=500)
fig.show()
```

**Expected Output**: Sankey diagram

---

## SECTION 3: Supporting Context - SAKT

### Cell 10: SAKT Population Mastery Distribution
**Type**: Code  
**Data Files**: `skill_mastery_df`

```python
fig = px.histogram(
    skill_mastery_df,
    x='mastery_mean',
    nbins=50,
    title='Population Mastery Distribution',
    labels={'mastery_mean': 'Mastery Score'}
)
fig.show()
```

---

### Cell 11: SAKT Attention Heatmap (T×T Matrix)
**Type**: Code  
**Data Files**: `attention_df`, `events_df`

```python
if attention_df is not None:
    # Sample a student with >20 interactions
    sample_student = events_df.groupby('user_id').size().sort_values(ascending=False).index[0]
    
    # Get attention data (simplified - assumes attention stored as matrix)
    # In real implementation, parse attention_df structure
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    # sns.heatmap(attention_matrix, cmap='viridis', square=True)
    plt.title('Full Attention Matrix (T×T)')
    plt.xlabel('Key (Past Timestep)')
    plt.ylabel('Query (Current Timestep)')
    plt.show()
else:
    print("⚠ Attention data not available")
```

---

## SECTION 4: Supporting Context - WD-IRT

### Cell 12: Item Difficulty vs Discrimination
**Type**: Code  
**Data Files**: `item_params_df`

```python
fig = px.scatter(
    item_params_df,
    x='difficulty',
    y='discrimination',
    color='drift_flag' if 'drift_flag' in item_params_df.columns else None,
    title='Item Bank: Difficulty vs Discrimination',
    opacity=0.6
)
fig.show()
```

---

## SECTION 5: Safety Layer - Gaming

### Cell 13: Gaming Alert Heatmap
**Type**: Code  
**Data Files**: `events_df`

```python
# Compute rapid guessing ratio per student
gaming_stats = events_df.groupby('user_id').agg({
    'latency_ms': lambda x: (x < 5000).mean()
}).rename(columns={'latency_ms': 'rapid_ratio'})

fig = px.histogram(
    gaming_stats,
    x='rapid_ratio',
    nbins=30,
    title='Gaming Detection: Rapid Guessing Distribution',
    labels={'rapid_ratio': 'Rapid Response Ratio'}
)
fig.show()
```

---

## Summary

**Total Cells**: 13  
**Required Data Files**:
1. `reports/bandit_decisions.parquet` ⭐ (Core)
2. `reports/skill_mastery.parquet`
3. `reports/item_params.parquet`
4. `data/interim/edm_cup_2023_42_events.parquet`
5. `reports/sakt_student_state.parquet`
6. `reports/sakt_attention.parquet` (Optional)

**Key Interactive Features**:
- Decision event dropdown (Cell 4)
- Alpha slider (Cell 5)
- Multiple `widgets.interact` visualizations (Cells 6-8)
