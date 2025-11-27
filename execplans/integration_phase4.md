# Integration Demo (Phase 4)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document must be maintained in accordance with `PLANS.md` at the repository root.


## Purpose / Big Picture

After completing this work, the repository will have a working demo CLI that combines outputs from both engines to produce actionable recommendations. Given a student ID and topic, the demo will:

1. Show the student's mastery on that topic (from SAKT practice data)
2. Recommend test items matching their skill level (from WD-IRT item parameters)
3. Flag any item health concerns (drift, low discrimination)

This is the "one command demo" from `plan.md`:
```bash
python demo_trace.py --student-id 123 --topic fractions
```

**Key Insight:** No model rewrites needed. Both engines already produce compatible outputs:
- SAKT: per-interaction mastery (can be aggregated by skill from events' `skill_ids`)
- WD-IRT: per-item difficulty/health (with `topic` column matching skill codes)
- 334 overlapping skills/topics with 96.3% coverage


## Progress

- [ ] Milestone 1: Build skill-level mastery aggregation from SAKT outputs
- [ ] Milestone 2: Implement recommendation engine joining mastery + item health
- [ ] Milestone 3: Update demo_trace.py with real outputs
- [ ] Milestone 4: End-to-end validation and documentation


## Surprises & Discoveries

- Discovery: Skills in events (`skill_ids`) and topics in WD-IRT (`problem_skill_code`) use the same Common Core standard codes (e.g., "7.RP.A.1", "3.OA.C.7-2"). This enables direct joins without mapping tables.
  Evidence: 334/347 (96.3%) of WD-IRT topics have matching practice data in SAKT.

- Discovery: 1,650 WD-IRT items are in overlapping topics, giving rich recommendation options per skill.
  Evidence: Query showing `wdirt_params[wdirt_params['topic'].isin(overlap)]` = 1,650 items.

- Discovery: Our approach aligns with industry implementations. ETS (Educational Testing Service) published the Wide & Deep IRT paper we're implementing (Shi Pu et al., 2024). TrueLearn's SmartBank uses similar "comparative analytics" combining mastery tracking with item-level performance benchmarking.
  Evidence: Research shows ETS uses Wide & Deep IRT for clickstream-based predictions; TrueLearn markets "performance analytics & real-time national benchmarking" which mirrors our twin-engine approach.


## Decision Log

- Decision: Join at skill level rather than item level
  Rationale: Direct item overlap is only 23 items (0.07%), but skill overlap is 334 skills (96.3%). Students practice on many items but are tested on specific items for each skill. Skill-level join captures this real-world pattern.
  Date/Author: 2025-11-27 / Planning

- Decision: Aggregate SAKT mastery from original events rather than model outputs
  Rationale: SAKT model outputs don't include skill_ids (only item_id). We'll join mastery predictions back to events to get skill information. This is a data processing step, not a model change.
  Date/Author: 2025-11-27 / Planning


## Outcomes & Retrospective

(To be completed after Phase 4)


## Competitive Landscape & Similar Implementations

### ETS (Educational Testing Service)

**What they do:** ETS developed the Wide & Deep IRT model we're implementing (Pu et al., 2024). They use it for:
- Predicting student performance from clickstream data
- Item health monitoring (drift detection, behavior pattern analysis)
- High-stakes test scoring (GRE, TOEFL)

**Key insight:** ETS combines IRT with deep learning for clickstream analysis, exactly our approach. Their model placed 2nd/3rd in EDM Cup 2023.

**Reference:** "Predicting Students' Future Success: Harnessing Clickstream Data with Wide & Deep Item Response Theory" (JEDM, 2024)

### TrueLearn SmartBank

**What they do:** Medical exam prep platform (USMLE, NCLEX) with:
- "Performance Analytics & Real-Time National Benchmarking"
- "Comparative Analytics" (benchmarking against peers)
- Item-level difficulty tracking
- Mastery-based recommendations

**Key insight:** They combine mastery tracking (like SAKT) with item-level analytics (like WD-IRT) for personalized recommendations. Their "SmartBank" concept mirrors our twin-engine approach.

**Reference:** TrueLearn.com marketing materials emphasize "data-driven approach" with analytics combining student performance and item characteristics.

### Duolingo English Test

**What they do:** High-stakes language proficiency test using:
- AutoIRT: Automated ML + IRT for item calibration
- Computerized Adaptive Testing (CAT)
- Item parameter estimation from small response sets

**Key insight:** They use ML-enhanced IRT (similar to our Wide & Deep IRT) for faster item calibration and better predictive performance than traditional IRT.

**Reference:** "AutoIRT: Calibrating Item Response Theory Models with Automated Machine Learning" (Sharpnack et al., 2024)

### Academic Research

**Integration approaches:**
- "Integrating knowledge tracing and item response theory: A tale of two frameworks" (Khajah et al., 2014) - Early work combining KT and IRT
- "Deep Knowledge Tracing is an implicit dynamic multidimensional item response theory model" (Vie & Kashima, 2023) - Shows DKT can be viewed as IRT variant
- "Supercharging BKT with Multidimensional Generalizable IRT and Skill Discovery" (Khajah, 2024) - Combines BKT with IRT

**Key insight:** Academic research confirms combining KT and IRT is a valid, well-studied approach. Our twin-engine system follows this pattern.

### What Makes Our Approach Different

| Feature | ETS | TrueLearn | Duolingo | Our System |
|---------|-----|-----------|----------|------------|
| Student Readiness | ✅ (IRT ability) | ✅ (Mastery tracking) | ✅ (Proficiency) | ✅ (SAKT) |
| Item Health | ✅ (WD-IRT) | ✅ (Analytics) | ✅ (AutoIRT) | ✅ (WD-IRT) |
| Clickstream Analysis | ✅ | ❌ | ❌ | ✅ |
| Open Source | ❌ | ❌ | ❌ | ✅ |
| Skill-Level Joins | ❌ | ✅ | ❌ | ✅ |

**Our advantage:** Open-source, reproducible, combines best of both worlds (SAKT for mastery + WD-IRT for item health) with full clickstream analysis.


## Context and Orientation

### Current State

Both engines are trained on EDM Cup 2023:
- **SAKT**: 35,192 students, 3.19M predictions, AUC=0.7528
- **WD-IRT**: 1,835 test items with difficulty/discrimination/guessing params

The challenge: SAKT predictions are per-item (32,887 items), WD-IRT params are for test items only (1,835 items). Only 23 items directly overlap.

### The Solution

Join at skill level:
1. Events have `skill_ids` (Common Core standards like "7.RP.A.1")
2. WD-IRT items have `topic` (same Common Core standards)
3. SAKT mastery can be aggregated by skill using: `mastery_by_skill[user][skill] = avg(mastery for interactions on that skill)`

### Data Flow

```
Events (5.1M)
    │
    ├── skill_ids ──────────────────────┐
    │                                   │
    └── SAKT predictions ──────┐        │
                               │        │
                               ▼        ▼
                         [Aggregate by user+skill]
                               │
                               ▼
                      Skill Mastery Matrix
                      (35K users × 480 skills)
                               │
                               ▼
                         [Join on skill]
                               │
                               ▼
                      Recommendations
                      (user → items by skill + difficulty)
                               ▲
                               │
                        WD-IRT items
                      (1,835 items × topics)
```

### Key Files

| File | Purpose |
|------|---------|
| `data/interim/edm_cup_2023_42_events.parquet` | Events with skill_ids |
| `reports/sakt_predictions.parquet` | Per-interaction predictions |
| `reports/sakt_student_state.parquet` | Per-interaction mastery |
| `reports/item_params.parquet` | WD-IRT item parameters with topic |
| `scripts/demo_trace.py` | Demo CLI (currently placeholder) |


## Plan of Work

### Milestone 1: Build Skill-Level Mastery Aggregation

Create a module that computes per-student, per-skill mastery from SAKT outputs + events.

**Inputs:**
- `sakt_student_state.parquet`: per-interaction mastery
- `edm_cup_2023_42_events.parquet`: events with skill_ids

**Output:**
- `skill_mastery.parquet`: DataFrame with columns `[user_id, skill, mastery_mean, mastery_std, interaction_count]`

**Implementation:**

```python
# src/common/mastery_aggregation.py

def aggregate_skill_mastery(
    mastery_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-interaction mastery to per-skill mastery.
    
    1. Join mastery predictions with events (to get skill_ids)
    2. Explode skill_ids (events can have multiple skills)
    3. Group by (user_id, skill)
    4. Compute mean, std, count
    """
```

**Tests:**
- Test aggregation produces one row per (user, skill)
- Test users with no interactions for a skill are handled
- Test multi-skill events are counted for each skill

### Milestone 2: Implement Recommendation Engine

Create a module that recommends items based on skill mastery and item parameters.

**Inputs:**
- `skill_mastery.parquet`: from M1
- `item_params.parquet`: WD-IRT item parameters

**Logic:**
1. For a given user and target skill:
   - Get user's mastery on that skill
   - Find items matching that skill (by topic)
   - Sort by difficulty (recommend easier items for struggling students)
   - Filter out high-drift items
2. Return top N recommendations with reasoning

**Implementation:**

```python
# src/common/recommendation.py

@dataclass
class ItemRecommendation:
    item_id: str
    topic: str
    difficulty: float
    discrimination: float
    reason: str  # e.g., "Easy item for skill 7.RP.A.1 (mastery: 0.45)"

def recommend_items(
    user_id: str,
    target_skill: str,
    skill_mastery: pd.DataFrame,
    item_params: pd.DataFrame,
    max_items: int = 5,
    exclude_high_drift: bool = True,
) -> List[ItemRecommendation]:
    """
    Recommend items for a student on a specific skill.
    """
```

### Milestone 3: Update demo_trace.py

Replace placeholder output with real recommendations.

**Changes:**
1. Load real data files (skill_mastery, item_params)
2. Call `aggregate_skill_mastery()` if skill_mastery.parquet doesn't exist
3. Call `recommend_items()` with user input
4. Display rich output showing:
   - Student's mastery trajectory on the skill
   - Recommended items with difficulty/health info
   - Any warnings (high drift, low interaction count)

**Example Output:**
```
──────────────── DeepKT + Wide&Deep IRT Trace ────────────────
Student: ABC123
Skill: 7.RP.A.1 (Ratios & Proportional Relationships)
Time Window: 2023-W15

📊 Mastery Analysis (SAKT)
  Current mastery: 0.45 (below threshold 0.60)
  Practice interactions: 23
  Trend: ↗ improving (+0.12 from last week)

📚 Recommended Items (WD-IRT)
  1. ITEM_X | difficulty: 0.32 | Good for current level
  2. ITEM_Y | difficulty: 0.41 | Slight stretch
  3. ITEM_Z | difficulty: 0.48 | Challenge item

⚠️ Item Health Alerts
  - ITEM_W has high drift (0.15) - consider reviewing
```

### Milestone 4: Validation and Documentation

1. Create `tests/test_integration.py` with end-to-end tests
2. Update README.md with demo instructions
3. Document the skill-level join methodology


## Concrete Steps for Milestone 1

### Step 1: Create mastery_aggregation.py

```bash
# Create the module
cat > src/common/mastery_aggregation.py << 'EOF'
# ABOUTME: Aggregates per-interaction mastery to per-skill mastery.
# ABOUTME: Bridges SAKT outputs with skill-level analysis.
...
EOF
```

### Step 2: Verify aggregation logic

```python
# Quick test
events = pd.read_parquet('data/interim/edm_cup_2023_42_events.parquet')
mastery = pd.read_parquet('reports/sakt_student_state.parquet')

# Join on position within user
events_with_pos = events.copy()
events_with_pos['position'] = events_with_pos.groupby('user_id').cumcount()

joined = mastery.merge(
    events_with_pos[['user_id', 'position', 'skill_ids']],
    on=['user_id', 'position']
)
print(f'Joined rows: {len(joined):,}')
```

### Step 3: Add unit tests

```bash
pytest tests/test_mastery_aggregation.py -v
```


## Validation and Acceptance

Milestone 1 complete when:
- `src/common/mastery_aggregation.py` exists
- `aggregate_skill_mastery()` produces valid skill_mastery.parquet
- Unit tests pass

Milestone 2 complete when:
- `src/common/recommendation.py` exists
- `recommend_items()` returns sensible recommendations
- Recommendations respect mastery thresholds and item health

Milestone 3 complete when:
- `scripts/demo_trace.py` produces real output
- Output shows mastery + recommendations + alerts

Milestone 4 complete when:
- End-to-end tests pass
- README documents the demo


## Idempotence and Recovery

All aggregation is idempotent. Re-running regenerates files safely.

If skill_mastery.parquet is missing, demo_trace.py will generate it on first run.


## Interfaces and Dependencies

No new dependencies required. Uses existing pandas, rich, typer.

**New modules:**
- `src/common/mastery_aggregation.py`
- `src/common/recommendation.py`

**Updated files:**
- `scripts/demo_trace.py`


## Timeline Estimate

- Milestone 1: ~1 hour (aggregation logic + tests)
- Milestone 2: ~1 hour (recommendation engine)
- Milestone 3: ~1 hour (demo CLI update)
- Milestone 4: ~30 minutes (validation + docs)

Total: ~3.5 hours


---

## Revision Log

- 2025-11-27: Initial draft based on Phase 3 joinability analysis

