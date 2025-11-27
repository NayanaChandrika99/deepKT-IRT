# Enhancement Priorities for UWorld Demo

## Purpose

This document helps prioritize system enhancements based on UWorld's strategic goals, current data availability, and implementation feasibility.

---

## Quick Decision Matrix

| Enhancement | Impact | Effort | Data Ready? | Demo Value | Priority |
|-------------|--------|--------|-------------|------------|----------|
| Explainable KT | 🟢 High | 2-3 wks | ✅ Yes | Shows "why" | **#1** |
| Gaming Detection | 🟢 High | 1-2 wks | ✅ Yes (basic) | Integrity | **#2** |
| RL Recommendations | 🟢 Very High | 4-6 wks | ✅ Yes | Self-improving | **#3** |
| LLM Item Analysis | 🟢 High | 3-4 wks | ❌ No | Content efficiency | Blocked |
| Fairness Detection | 🟡 Med-High | 3-5 wks | ⚠️ Partial | Compliance | Phase 5C |
| Cognitive Load | 🟡 Med-High | 4-6 wks | ⚠️ Partial | Engagement | Phase 5C |

---

## Priority #1: Explainable Knowledge Tracing

### Why First?
- **Differentiation**: Competitors (TrueLearn, AMBOSS) don't have this
- **Trust Builder**: Students/educators see WHY predictions are made
- **Low Risk**: Uses existing SAKT model, no new data needed
- **Demo-Ready**: Impressive visual output ("Here's why mastery is 0.45")

### UWorld Value
- Increases student engagement with recommendations
- Helps educators understand and trust the system
- Positions UWorld as "transparent AI, not black box"

### Data Status
- ✅ SAKT model with attention mechanism (trained)
- ✅ Mastery predictions (exported)
- ✅ Skill/item metadata (available)

### Implementation
- Extract attention weights from SAKT's transformer layers
- Map attention to specific past interactions
- Generate human-readable explanations

---

## Priority #2: Basic Gaming Detection

### Why Second?
- **Real Problem**: Gaming is prevalent in online learning
- **Quick Win**: 1-2 weeks to implement
- **Uses Existing Data**: latency_ms, help_requested, correct
- **High Visibility**: "Catching cheaters" resonates with educators

### UWorld Value
- Maintains assessment integrity (critical for medical licensing prep)
- Identifies struggling students (gaming often = frustration)
- Competitive advantage (most platforms ignore this)

### Data Status
- ✅ Response latency (latency_ms)
- ✅ Help request flags (help_requested)
- ✅ Correctness patterns (correct)
- ⚠️ Full action logs (not available - future enhancement)

### What We CAN Detect
1. **Rapid Guessing**: < 5 second responses
2. **Help Abuse**: Help requests before attempting
3. **Suspicious Patterns**: Many rapid wrong → sudden correct

### What We CAN'T Detect (Yet)
- Detailed clickstream patterns (need action logs)
- Copy-paste behavior (need keystroke data)
- Screen sharing (need browser monitoring)

---

## Priority #3: RL Recommendations (Phase 5B)

### Why Third?
- **Transformative**: Self-improving recommendations
- **Proven Results**: 14%+ improvement in literature
- **Industry Standard**: Google, Duolingo use this
- **Builds on Phase 4**: Enhances existing recommendation engine

### UWorld Value
- Better learning outcomes (proven)
- Self-optimizing (reduces manual tuning)
- Positions UWorld as cutting-edge

### Data Status
- ✅ Mastery scores (SAKT output)
- ✅ Item parameters (WD-IRT output)
- ✅ Response outcomes (events data)

### Implementation Approach
- Contextual Multi-Armed Bandit (LinUCB or Neural)
- State: student mastery + recent performance
- Actions: recommend item from pool
- Reward: correctness + long-term retention

---

## Blocked Enhancements

### LLM Item Analysis ❌
**Blocked By**: Missing raw item text
**What's Missing**: problem stems, answer options, distractors
**Current**: Only have BERT embeddings (pre-computed)
**Unblock**: UWorld provides item content database

### Fairness Detection (DIF) ⚠️
**Blocked By**: Missing demographic data
**What's Missing**: gender, ethnicity, background
**Unblock**: UWorld provides anonymized demographics

### Full Gaming Detection ⚠️
**Blocked By**: Missing complete action logs
**What's Missing**: all clicks, not just responses
**Current**: Can do basic detection (latency, help flags)
**Unblock**: UWorld provides action-level clickstream

---

## Recommended Phasing

### Phase 5A (NOW - 3-4 weeks)
1. **Explainable KT** - 2-3 weeks
2. **Basic Gaming Detection** - 1-2 weeks
   
Total: 3-4 weeks (can parallelize)

### Phase 5B (NEXT - 4-6 weeks)
3. **RL Recommendations** - 4-6 weeks

### Phase 5C (FUTURE - Needs Data)
4. **LLM Item Analysis** - blocked
5. **Fairness Detection** - blocked
6. **Full Gaming Detection** - blocked

---

## Demo Talking Points

### Current Demo (Phase 4)
> "Our twin-engine system predicts student readiness AND item health."

### Enhanced Demo (Phase 5A)
> "Not only do we predict mastery, we EXPLAIN why. Students see which past interactions drive their scores. Plus, we detect gaming behavior to maintain assessment integrity."

### Future Vision (Phase 5B+)
> "Our RL-powered recommendations continuously improve. The system learns what works for each student profile."

---

## Questions for UWorld

Before finalizing priorities, consider:

1. **Explainability**: How important is transparency to your users?
   - Do students/educators ask "why" about their scores?
   - Is "black box AI" a concern in sales conversations?

2. **Gaming/Cheating**: Is this a known problem?
   - Do proctors report suspicious behavior?
   - Is assessment integrity a selling point?

3. **RL Recommendations**: How much improvement matters?
   - Would 14% better recommendations be a big deal?
   - Is "self-improving" a compelling message?

4. **Data Availability**: Can you provide:
   - Item text/stems for LLM analysis?
   - Demographic data for fairness analysis?
   - Full action logs for better gaming detection?

---

## Revision Log

- 2025-11-27: Initial priorities based on data availability assessment

