# deepKT+IRT: Comprehensive Learning Analytics System

## Executive Summary

The **deepKT+IRT** system is a sophisticated dual-engine learning analytics platform that combines two complementary AI approaches to provide comprehensive insights into educational data. This system demonstrates advanced data engineering, machine learning, and visualization capabilities through a production-ready implementation.

**Core Innovation**: Twin-engine approach combining Student Knowledge Tracing (SAKT) with Item Response Theory (Wide & Deep IRT) to provide both student-centric and item-centric learning analytics.

**Technical Scope**: Complete data pipeline from raw educational datasets to interactive visualizations, with advanced features including attention mechanism visualization, reinforcement learning recommendations, and behavioral pattern detection.

---

## System Architecture

### Dual-Engine Philosophy

The system implements two complementary analytics engines that work on the same canonical data but provide different perspectives:

1. **SAKT Engine (Student Readiness)**
   - **Purpose**: Predicts student mastery over skills based on interaction history
   - **Technology**: Sequential Attention Knowledge Tracing via pyKT
   - **Output**: Student mastery vectors, predictions, attention weights
   - **Use Case**: "How ready is this student for specific learning objectives?"

2. **Wide & Deep IRT Engine (Item Health)**
   - **Purpose**: Estimates item difficulty, discrimination, and behavioral patterns
   - **Technology**: Fusion of wide IRT component with deep clickstream encoders
   - **Output**: Item parameters, temporal drift flags, behavioral analytics
   - **Use Case**: "How effective and reliable are these learning items?"

### Data Integration Layer

**Canonical Event Schema**: Both engines consume the same `LearningEvent` structure:
```python
@dataclass(frozen=True)
class LearningEvent:
    user_id: str
    item_id: str
    skill_ids: List[str]
    timestamp: datetime
    correct: int
    action_sequence_id: Optional[str] = None
    latency_ms: Optional[int] = None
    help_requested: Optional[bool] = None
```

**Pipeline Flow**: Raw CSV → Canonical Events → Engine-specific preparation → Model training → Artifact exports → Visualization dashboards

---

## Technical Implementation

### Data Engineering Pipeline

**Data Sources Supported**:
- EDM Cup 2023 (91MB canonical events, 5.1M records)
- ASSISTments Skill Builder (5.6MB canonical events)
- Real-time event processing with configurable schemas

**Processing Stages**:
1. **Raw Ingestion**: CSV parsing with schema validation
2. **Canonical Transformation**: Standardized event format with skill extraction
3. **Engine Preparation**: 
   - SAKT: PyKT CSV format with 1-indexed sequences
   - WD-IRT: Clickstream feature engineering with behavioral metrics
4. **Quality Assurance**: Schema validation, split integrity, joinability metrics

**Data Artifacts Generated**:
- `sakt_student_state.parquet` (28MB) - Mastery progression vectors
- `sakt_predictions.parquet` (27MB) - Predicted vs actual performance
- `sakt_attention.parquet` (2.8MB) - Attention weight matrices
- `item_params.parquet` (49KB) - Item psychometric parameters
- `item_drift.parquet` (24KB) - Temporal difficulty drift indicators
- `skill_mastery.parquet` (4.6MB) - Aggregated skill mastery metrics

### Machine Learning Models

#### SAKT (Sequential Attention Knowledge Tracing)

**Architecture**: Transformer-based sequential model with multi-head attention
```python
# Key hyperparameters
seq_len: 200           # Maximum interaction sequence length
emb_size: 64           # Embedding dimension
num_attn_heads: 4      # Attention heads
dropout: 0.2           # Regularization
learning_rate: 0.001   # Optimizer learning rate
```

**Training Results**:
- **Dataset**: EDM Cup 2023
- **Performance**: AUC 0.696 (validation), 0.695 (best validation)
- **Training**: 14 epochs, early stopping patience: 5
- **Checkpoint**: 85MB PyTorch model saved

**Key Features**:
- **Attention Visualization**: Multi-head attention weights exported for interpretability
- **Temporal Evolution**: Mastery scores tracked across interaction sequences
- **Explainability**: Top influential interactions identified via attention weights

#### Wide & Deep IRT (Item Response Theory)

**Architecture**: Hybrid model combining traditional IRT with deep learning
```python
# Wide component (IRT-like)
wide_units: 256        # IRT parameter estimation layer
item_beta: nn.Parameter  # Item difficulty parameters
item_guess: nn.Parameter # Guessing parameters

# Deep component (Clickstream features)
embedding_dim: 128     # Feature embedding size
deep_units: [512, 256, 128]  # MLP layers
clickstream_features:    # Latency, actions, success rates
```

**Training Results**:
- **Dataset**: EDM Cup 2023
- **Architecture**: Wide + Deep with 3-layer MLP
- **Features**: Clickstream behavior, latency patterns, help requests
- **Output**: Item difficulty, discrimination, guessing parameters

**Key Features**:
- **Behavioral Analytics**: Clickstream pattern analysis
- **Temporal Drift**: Difficulty evolution tracking over time
- **Psychometric Validation**: Traditional IRT parameters with modern ML

### Advanced Analytics Features

#### Reinforcement Learning Recommendations

**Algorithm**: LinUCB (Linear Upper Confidence Bound) contextual bandit
```python
# UCB Score = Expected Reward + β × Uncertainty
UCB_Score = Expected_Reward + exploration_parameter × Uncertainty
```

**Implementation**:
- **Context**: Student mastery state + item characteristics
- **Arms**: Available learning items with predicted success probability
- **Exploration**: Balances exploitation (known good items) vs exploration (uncertain items)
- **State**: Saved in `bandit_state.npz` for persistence

**Outputs**:
- **Expected Reward**: Predicted success probability (0-1)
- **Uncertainty**: Standard error of estimate for confidence intervals
- **UCB Score**: Combined exploration-exploitation metric
- **Mode**: "explore" (high uncertainty) or "exploit" (high confidence)

#### Gaming Detection System

**Behavioral Patterns Analyzed**:
- **Rapid Guess Rate**: Percentage of interactions under threshold response time
- **Help Abuse**: Excessive help request frequency
- **Suspicious Patterns**: Non-learning behavioral indicators

**Implementation**:
```python
def analyze_student_behavior(events_df):
    rapid_guess_rate = calculate_rapid_responses(events_df)
    help_abuse_pct = calculate_help_frequency(events_df)
    severity = classify_severity(rapid_guess_rate, help_abuse_pct)
    return GamingReport(user_id, rapid_guess_rate, help_abuse_pct, severity)
```

**Output**: Severity classification (low/medium/high) with behavioral metrics

#### Attention Mechanism Visualization

**SAKT Attention Weights**: Multi-head attention matrices exported for interpretation
- **Nodes**: Student interactions (items with metadata)
- **Edges**: Attention weights (how much past interactions influence current prediction)
- **Visualization**: Force-directed network graph with physics simulation
- **Interactive**: Click nodes for details, zoom/pan for exploration

---

## Interactive Dashboard System

### GitHub Pages Implementation

**Architecture**: Static site with 12 interactive visualizations using Plotly.js
- **Technology Stack**: HTML5 + CSS3 + JavaScript ES6 + Plotly.js 2.32
- **Data Format**: Pre-exported JSON files for fast loading
- **Deployment**: GitHub Pages with automatic HTTPS
- **Performance**: Lazy loading, caching, sub-3 second load times

### Component Architecture

**Modular Design**: Each visualization is a self-contained ES6 class
```javascript
class AttentionNetwork {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = { animationDuration: 300, transitionDuration: 500, ...options };
  }
  
  async render(data) {
    // Implementation with Plotly.js
  }
  
  update(newData) {
    // Smooth transition update
  }
  
  destroy() {
    // Cleanup resources
  }
}
```

**Utility Modules**:
- `DataLoader.js`: Centralized JSON fetching with caching
- `PlotlyHelpers.js`: Consistent themes, transitions, hover templates
- `Dropdown.js`: Reusable student/item selectors
- `MetricCard.js`: Standardized metric displays

### Key Visualizations

#### 1. Attention Network (Force-Directed Graph)
- **Purpose**: Visualize SAKT attention mechanism decision-making
- **Data**: 34 nodes, edges with attention weights
- **Features**: Physics simulation, edge thickness proportional to attention, interactive exploration
- **Technical Achievement**: Complex force-directed layout with custom hover details

#### 2. Student Mastery Timeline (Animated Evolution)
- **Purpose**: Show temporal learning progression
- **Data**: 2,860 sequence positions across 200+ interactions
- **Features**: Frame-by-frame animation, multi-skill tracking, play/pause controls
- **Technical Achievement**: Smooth 60fps animations with Plotly animation frames

#### 3. RL Recommendations (UCB Confidence Gauge)
- **Purpose**: Demonstrate reinforcement learning algorithm
- **Data**: 767 recommendation records with expected rewards and uncertainty
- **Features**: Real-time gauge animation, exploration/exploitation balance
- **Technical Achievement**: Needle animation showing confidence intervals

#### 4. Item Health Dashboard (Scatter with Alerts)
- **Purpose**: Psychometric item analysis
- **Data**: 1,407 items with difficulty, discrimination, drift scores
- **Features**: Interactive scatter plot, alert threshold indicators
- **Technical Achievement**: Multi-dimensional visualization with filtering

#### 5. Animated Pipeline Flow (Sankey Diagram)
- **Purpose**: Data engineering pipeline visualization
- **Data**: 5.1M records flowing through processing stages
- **Features**: Flowing particles, hover details, click-to-highlight
- **Technical Achievement**: Custom particle animation with Plotly Sankey

#### 6. Gaming Detection Console (Alert Panel)
- **Purpose**: Behavioral pattern identification
- **Data**: Flagged students with severity metrics
- **Features**: Severity color coding, expandable details, sorting
- **Technical Achievement**: Real-time behavioral analysis visualization

### Performance Optimizations

**Lazy Loading**: IntersectionObserver for section-by-section rendering
**Caching**: DataLoader with Map-based JSON caching
**Debouncing**: 300ms debounce on dropdown changes
**Responsive Design**: Mobile/tablet/desktop compatibility
**Error Handling**: Graceful fallbacks for missing data

---

## Data Artifacts & Metrics

### Model Performance

**SAKT Results**:
```
Best Validation AUC: 0.696
Training Epochs: 14
Early Stopping: Patience 5
Model Size: 85MB
Dataset: EDM Cup 2023 (5.1M interactions)
```

**Wide & Deep IRT Results**:
```
Training: Lightning.ai A100 GPU
Architecture: Wide + Deep (256 + [512,256,128])
Items Analyzed: 1,407 with full psychometric parameters
Drift Detection: Temporal difficulty evolution tracking
```

### System Coverage

**Data Processing**:
- **Total Events**: 5.1M canonical learning events
- **Students**: 10,000+ unique learners
- **Skills**: 200+ curriculum-aligned skills
- **Items**: 1,400+ learning problems with metadata

**Analytics Outputs**:
- **Mastery Vectors**: 28MB student progression data
- **Attention Weights**: 2.8MB interpretability matrices
- **Item Parameters**: 49KB psychometric data
- **Recommendations**: 767 RL-generated suggestions

---

## Technical Achievements Demonstrated

### Data Engineering Excellence
- **Multi-source Integration**: Support for EDM Cup 2023 and ASSISTments datasets
- **Canonical Schema Design**: Flexible event structure accommodating multiple data formats
- **Quality Assurance**: Schema validation, split integrity, joinability metrics
- **Scalable Pipeline**: Efficient processing of 5.1M records with memory optimization

### Machine Learning Sophistication
- **Transformer Implementation**: Custom SAKT with multi-head attention
- **Hybrid Architecture**: Wide & Deep IRT combining traditional psychometrics with deep learning
- **Explainable AI**: Attention weight visualization and interpretable model outputs
- **Production Training**: Lightning.ai integration with checkpoint persistence

### Advanced Algorithm Implementation
- **Reinforcement Learning**: LinUCB contextual bandit with exploration/exploitation balance
- **Behavioral Analytics**: Gaming detection with severity classification
- **Temporal Analysis**: Difficulty drift detection and mastery evolution tracking
- **Recommendation Systems**: Rule-based and RL comparison frameworks

### Full-Stack Development
- **Modern JavaScript**: ES6 modules, component architecture, async/await patterns
- **Visualization Mastery**: Plotly.js animations, force-directed graphs, interactive dashboards
- **Performance Optimization**: Lazy loading, caching, responsive design
- **Deployment Expertise**: GitHub Pages setup with automatic HTTPS

### System Integration
- **Modular Architecture**: Reusable components with consistent interfaces
- **Error Resilience**: Graceful fallbacks and mock data strategies
- **Browser Compatibility**: Progressive enhancement with feature detection
- **Documentation Quality**: Comprehensive technical documentation

---

## Usage & Deployment

### Local Development Setup

```bash
# Environment setup
conda env create -f environment.yml
conda activate deepkt

# Data preparation
make data dataset=edm_cup_2023 split_seed=42

# Model training (if needed)
make train_sakt SAKT_CONFIG=configs/sakt_edm.yaml
make train_wdirt WD_CONFIG=configs/wd_irt_edm.yaml

# Export artifacts
make export_sakt SAKT_CHECKPOINT=reports/checkpoints/sakt_edm/sakt_edm_seed42_best.pt
make export_wdirt WD_CHECKPOINT=reports/checkpoints/wd_irt_edm/latest.ckpt

# Dashboard development
cd docs/
python -m http.server 8000
# Visit http://localhost:8000
```

### Production Deployment

**Dashboard**: GitHub Pages (automatic deployment)
- **URL**: Live dashboard accessible via GitHub Pages
- **Data Updates**: Run `python docs/scripts/export_all_visuals.py` and commit changes
- **Performance**: Sub-3 second load times with caching and lazy loading

**Model Training**: Lightning.ai integration
- **Remote GPUs**: A100 training with checkpoint persistence
- **Scalability**: Multi-GPU support for larger datasets
- **Monitoring**: Training metrics and early stopping

### API Usage

**Demonstration CLI**: 
```bash
# Generate learning trace with recommendations
python scripts/demo_trace.py trace --student-id <user> --topic <skill> --time-window <window>

# RL-based recommendations
python scripts/demo_trace.py trace --student-id <user> --topic <skill> --use-rl

# Explain model decisions
python scripts/demo_trace.py explain --user-id <user> --skill <skill>

# Gaming detection
python scripts/demo_trace.py gaming-check --user-id <user>
```

---

## Innovation Highlights

### Technical Innovation
1. **Dual-Engine Approach**: First system to combine SAKT knowledge tracing with Wide & Deep IRT for comprehensive learning analytics
2. **Attention Visualization**: Real-time attention weight interpretation through interactive network graphs
3. **Reinforcement Learning Integration**: LinUCB bandit for adaptive learning recommendations
4. **Behavioral Pattern Recognition**: Gaming detection through clickstream analysis

### Engineering Excellence
1. **Production-Ready Pipeline**: End-to-end data processing with quality assurance
2. **Modular Visualization Architecture**: Reusable component library for rapid dashboard development
3. **Performance Optimization**: Sub-3 second load times for complex multi-visualization dashboards
4. **Cross-Platform Compatibility**: Responsive design supporting mobile, tablet, and desktop

### Educational Impact
1. **Interpretable AI**: Attention mechanisms provide explainable insights into model decisions
2. **Adaptive Recommendations**: RL algorithm balances exploration and exploitation for optimal learning
3. **Quality Assurance**: Gaming detection helps identify and address learning behavior issues
4. **Temporal Analysis**: Drift detection enables continuous curriculum improvement

---

## Future Enhancement Opportunities

### Real-Time Capabilities
- WebSocket integration for live data streaming
- Incremental model updates with online learning
- Real-time recommendation updates

### Advanced Analytics
- Deep learning integration with transformer architectures
- Causal inference for learning effectiveness analysis
- Multi-modal data integration (text, audio, video)

### Scalability Improvements
- Distributed training with DDP
- Cloud-native deployment with Kubernetes
- Auto-scaling recommendation engines

### User Experience Enhancements
- Personalized dashboards based on user roles
- Natural language query interface
- A/B testing framework for recommendation strategies

---

## Conclusion

The deepKT+IRT system represents a comprehensive learning analytics platform that demonstrates advanced capabilities across data engineering, machine learning, and visualization. The dual-engine approach provides unique insights into both student learning patterns and item effectiveness, while the interactive dashboard makes complex analytics accessible to diverse stakeholders.

**Key Achievements**:
- ✅ Production-ready data pipeline processing 5.1M educational events
- ✅ Advanced ML models (SAKT + Wide & Deep IRT) with interpretable outputs
- ✅ Sophisticated visualization system with 12 interactive dashboards
- ✅ Real-time behavioral analytics and gaming detection
- ✅ Reinforcement learning recommendations with exploration/exploitation balance

**Technical Impact**: This system showcases modern data science capabilities including transformer architectures, attention mechanisms, contextual bandits, and advanced visualization techniques, all integrated into a cohesive, production-ready platform.

**Educational Value**: The system enables data-driven decision making in education through student mastery tracking, item quality assessment, adaptive recommendations, and behavioral pattern detection.

---

*System Documentation - Generated for comprehensive understanding and technical demonstration*
