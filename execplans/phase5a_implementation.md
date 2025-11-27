# Phase 5A Implementation Plan

## Overview

**Goal**: Implement Explainable Knowledge Tracing and Basic Gaming Detection (3-4 weeks)

**Deliverables**:
1. `src/common/explainability.py` - Attention-based explanations
2. `src/common/gaming_detection.py` - Behavior anomaly detection
3. Enhanced `demo_trace.py` - Show explanations and gaming flags
4. Tests for both modules

---

## Milestone 1: Explainable Knowledge Tracing (2-3 weeks)

### M1.1: Extract SAKT Attention Weights (3-4 days)

**Task**: Modify SAKT export to capture attention weights during inference.

**Files to Modify**:
- `src/sakt_kt/export.py`

**Implementation**:

```python
# In export_student_mastery function, after model forward pass:

def extract_attention_weights(model, batch):
    """Extract attention weights from SAKT transformer layers."""
    with torch.no_grad():
        # Get attention weights from SAKT's attention mechanism
        # pyKT's SAKT uses scaled dot-product attention
        q, r, qry = batch
        
        # Hook to capture attention weights
        attention_weights = []
        def attention_hook(module, input, output):
            # output[1] is attention weights if return_attention=True
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1])
        
        # Register hook on attention layers
        hooks = []
        for layer in model.model:
            if hasattr(layer, 'attention'):
                hooks.append(layer.attention.register_forward_hook(attention_hook))
        
        # Forward pass
        output = model(q, r, qry)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        return attention_weights
```

**Output Schema**:
```python
# sakt_attention.parquet
{
    "user_id": str,
    "prediction_idx": int,  # Which prediction in sequence
    "item_id": str,         # Item being predicted
    "attended_item_ids": list[str],  # Past items with high attention
    "attention_weights": list[float],  # Corresponding weights
}
```

**Acceptance Criteria**:
- [ ] Attention weights extracted for each prediction
- [ ] Weights sum to ~1.0 for each prediction
- [ ] Top-5 attended items identified

---

### M1.2: Build Explanation Generator (3-4 days)

**Task**: Convert attention weights to human-readable explanations.

**New File**: `src/common/explainability.py`

```python
# ABOUTME: Generates human-readable explanations from SAKT attention weights.
# ABOUTME: Maps attention patterns to interpretable factors affecting mastery.

from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

@dataclass
class MasteryExplanation:
    """Human-readable explanation for a mastery prediction."""
    user_id: str
    skill_id: str
    mastery_score: float
    key_factors: List[Dict]  # [{"item_id": X, "weight": Y, "outcome": Z}]
    summary: str  # Natural language summary

def generate_explanation(
    user_id: str,
    skill_id: str,
    mastery_score: float,
    attention_weights: pd.DataFrame,
    events_df: pd.DataFrame,
) -> MasteryExplanation:
    """Generate explanation for a mastery prediction."""
    
    # Get top-5 attended interactions
    user_attention = attention_weights[attention_weights["user_id"] == user_id]
    top_attended = user_attention.nlargest(5, "attention_weight")
    
    key_factors = []
    correct_count = 0
    incorrect_count = 0
    
    for _, row in top_attended.iterrows():
        item_id = row["attended_item_id"]
        weight = row["attention_weight"]
        
        # Get outcome from events
        event = events_df[
            (events_df["user_id"] == user_id) & 
            (events_df["item_id"] == item_id)
        ].iloc[-1]  # Most recent interaction
        
        outcome = "correct" if event["correct"] == 1 else "incorrect"
        if outcome == "correct":
            correct_count += 1
        else:
            incorrect_count += 1
        
        key_factors.append({
            "item_id": item_id,
            "weight": round(weight, 3),
            "outcome": outcome,
            "skill": event.get("skill_ids", ["unknown"])[0] if isinstance(event.get("skill_ids"), list) else "unknown"
        })
    
    # Generate summary
    if mastery_score >= 0.7:
        summary = f"High mastery ({mastery_score:.2f}): "
        summary += f"Strong performance on {correct_count}/5 key interactions."
    elif mastery_score >= 0.4:
        summary = f"Moderate mastery ({mastery_score:.2f}): "
        summary += f"Mixed results - {correct_count} correct, {incorrect_count} incorrect on key interactions."
    else:
        summary = f"Low mastery ({mastery_score:.2f}): "
        summary += f"Struggled with {incorrect_count}/5 key interactions. Focus on foundational skills."
    
    return MasteryExplanation(
        user_id=user_id,
        skill_id=skill_id,
        mastery_score=mastery_score,
        key_factors=key_factors,
        summary=summary
    )

def format_explanation_for_display(explanation: MasteryExplanation) -> str:
    """Format explanation for CLI/display."""
    lines = [
        f"Student: {explanation.user_id}",
        f"Skill: {explanation.skill_id}",
        f"Mastery: {explanation.mastery_score:.2f}",
        "",
        "Key Factors:",
    ]
    
    for i, factor in enumerate(explanation.key_factors, 1):
        emoji = "✅" if factor["outcome"] == "correct" else "❌"
        lines.append(
            f"  {i}. {emoji} {factor['item_id']} "
            f"(attention: {factor['weight']:.3f}, {factor['outcome']})"
        )
    
    lines.append("")
    lines.append(f"Summary: {explanation.summary}")
    
    return "\n".join(lines)
```

**Acceptance Criteria**:
- [ ] Explanations generated for any user/skill
- [ ] Summary matches mastery level (high/medium/low)
- [ ] Key factors include attention weights and outcomes

---

### M1.3: Integrate with Demo CLI (2-3 days)

**Task**: Add explanation output to `demo_trace.py`.

**Modify**: `scripts/demo_trace.py`

```python
# Add new command to demo CLI

@app.command()
def explain(
    user_id: str = typer.Argument(..., help="Student ID to explain"),
    skill_id: str = typer.Argument(..., help="Skill ID to explain"),
    attention_path: Path = typer.Option(
        "reports/sakt_attention.parquet",
        help="Path to attention weights"
    ),
    events_path: Path = typer.Option(
        "data/interim/edm_cup_2023_42_events.parquet",
        help="Path to events"
    ),
):
    """Show explanation for a student's mastery on a skill."""
    from src.common.explainability import generate_explanation, format_explanation_for_display
    
    attention_df = pd.read_parquet(attention_path)
    events_df = pd.read_parquet(events_path)
    mastery_df = pd.read_parquet("reports/sakt_student_state.parquet")
    
    # Get mastery score
    mastery_row = mastery_df[
        (mastery_df["user_id"] == user_id) & 
        (mastery_df["skill_id"] == skill_id)
    ]
    
    if mastery_row.empty:
        console.print(f"[red]No mastery data for {user_id} on {skill_id}[/red]")
        return
    
    mastery_score = mastery_row.iloc[-1]["mastery"]
    
    explanation = generate_explanation(
        user_id=user_id,
        skill_id=skill_id,
        mastery_score=mastery_score,
        attention_weights=attention_df,
        events_df=events_df,
    )
    
    console.print(format_explanation_for_display(explanation))
```

**Sample Output**:
```
Student: ABC123
Skill: 7.RP.A.1
Mastery: 0.45

Key Factors:
  1. ❌ Q12345 (attention: 0.23, incorrect)
  2. ❌ Q23456 (attention: 0.19, incorrect)
  3. ✅ Q34567 (attention: 0.15, correct)
  4. ❌ Q45678 (attention: 0.12, incorrect)
  5. ✅ Q56789 (attention: 0.11, correct)

Summary: Moderate mastery (0.45): Mixed results - 2 correct, 3 incorrect on key interactions.
```

---

## Milestone 2: Basic Gaming Detection (1-2 weeks)

### M2.1: Implement Gaming Detector (3-4 days)

**Task**: Create module to detect gaming behavior from response patterns.

**New File**: `src/common/gaming_detection.py`

```python
# ABOUTME: Detects gaming/cheating behavior from student response patterns.
# ABOUTME: Uses latency, help requests, and correctness sequences.

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

@dataclass 
class GamingAlert:
    """Alert for potential gaming behavior."""
    user_id: str
    alert_type: str  # "rapid_guessing", "help_abuse", "suspicious_pattern"
    severity: str    # "low", "medium", "high"
    evidence: dict   # Supporting data
    recommendation: str

# Thresholds (configurable)
RAPID_RESPONSE_THRESHOLD_MS = 5000  # < 5 seconds
HELP_ABUSE_THRESHOLD = 0.3  # > 30% help requests
RAPID_INCORRECT_STREAK = 5  # 5+ rapid incorrect answers

def detect_rapid_guessing(events_df: pd.DataFrame, user_id: str) -> Optional[GamingAlert]:
    """Detect rapid guessing (< 5s responses)."""
    user_events = events_df[events_df["user_id"] == user_id]
    
    rapid_responses = user_events[user_events["latency_ms"] < RAPID_RESPONSE_THRESHOLD_MS]
    rapid_ratio = len(rapid_responses) / len(user_events) if len(user_events) > 0 else 0
    
    if rapid_ratio < 0.1:
        return None  # Normal
    
    # Check if rapid responses are mostly incorrect
    rapid_incorrect = rapid_responses[rapid_responses["correct"] == 0]
    rapid_incorrect_ratio = len(rapid_incorrect) / len(rapid_responses) if len(rapid_responses) > 0 else 0
    
    if rapid_ratio >= 0.3 and rapid_incorrect_ratio >= 0.7:
        severity = "high"
        recommendation = "Requires educator review. Student may be frustrated or disengaged."
    elif rapid_ratio >= 0.2:
        severity = "medium"
        recommendation = "Monitor student. Consider reaching out to discuss study habits."
    else:
        severity = "low"
        recommendation = "Minor concern. Continue monitoring."
    
    return GamingAlert(
        user_id=user_id,
        alert_type="rapid_guessing",
        severity=severity,
        evidence={
            "rapid_response_count": len(rapid_responses),
            "total_responses": len(user_events),
            "rapid_ratio": round(rapid_ratio, 3),
            "rapid_incorrect_ratio": round(rapid_incorrect_ratio, 3),
            "avg_rapid_latency_ms": int(rapid_responses["latency_ms"].mean()),
        },
        recommendation=recommendation
    )

def detect_help_abuse(events_df: pd.DataFrame, user_id: str) -> Optional[GamingAlert]:
    """Detect excessive help requests before attempting answers."""
    user_events = events_df[events_df["user_id"] == user_id]
    
    if "help_requested" not in user_events.columns:
        return None
    
    help_requests = user_events[user_events["help_requested"] == True]
    help_ratio = len(help_requests) / len(user_events) if len(user_events) > 0 else 0
    
    if help_ratio < HELP_ABUSE_THRESHOLD:
        return None
    
    severity = "high" if help_ratio >= 0.5 else "medium"
    
    return GamingAlert(
        user_id=user_id,
        alert_type="help_abuse",
        severity=severity,
        evidence={
            "help_request_count": len(help_requests),
            "total_responses": len(user_events),
            "help_ratio": round(help_ratio, 3),
        },
        recommendation="Student frequently requests help before attempting. May need scaffolding or concept review."
    )

def detect_suspicious_patterns(events_df: pd.DataFrame, user_id: str) -> Optional[GamingAlert]:
    """Detect suspicious correctness patterns (many rapid wrong, then suddenly correct)."""
    user_events = events_df[events_df["user_id"] == user_id].sort_values("timestamp")
    
    if len(user_events) < 10:
        return None
    
    # Look for streaks of rapid incorrect followed by correct
    suspicious_streaks = 0
    current_rapid_incorrect = 0
    
    for _, row in user_events.iterrows():
        is_rapid = row["latency_ms"] < RAPID_RESPONSE_THRESHOLD_MS
        is_incorrect = row["correct"] == 0
        
        if is_rapid and is_incorrect:
            current_rapid_incorrect += 1
        elif current_rapid_incorrect >= RAPID_INCORRECT_STREAK and row["correct"] == 1:
            suspicious_streaks += 1
            current_rapid_incorrect = 0
        else:
            current_rapid_incorrect = 0
    
    if suspicious_streaks < 2:
        return None
    
    severity = "high" if suspicious_streaks >= 5 else "medium"
    
    return GamingAlert(
        user_id=user_id,
        alert_type="suspicious_pattern",
        severity=severity,
        evidence={
            "suspicious_streaks": suspicious_streaks,
            "pattern": "rapid_incorrect_then_correct",
        },
        recommendation="Pattern suggests possible answer-seeking behavior. Review session history."
    )

def analyze_student(events_df: pd.DataFrame, user_id: str) -> List[GamingAlert]:
    """Run all gaming detectors for a student."""
    alerts = []
    
    for detector in [detect_rapid_guessing, detect_help_abuse, detect_suspicious_patterns]:
        alert = detector(events_df, user_id)
        if alert:
            alerts.append(alert)
    
    return alerts

def generate_gaming_report(events_df: pd.DataFrame) -> pd.DataFrame:
    """Generate gaming report for all students."""
    all_alerts = []
    
    for user_id in events_df["user_id"].unique():
        alerts = analyze_student(events_df, user_id)
        for alert in alerts:
            all_alerts.append({
                "user_id": alert.user_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "evidence": str(alert.evidence),
                "recommendation": alert.recommendation,
            })
    
    return pd.DataFrame(all_alerts)
```

**Acceptance Criteria**:
- [ ] Rapid guessing detection works (< 5s threshold)
- [ ] Help abuse detection works (> 30% help requests)
- [ ] Suspicious pattern detection works
- [ ] Report generated for all students

---

### M2.2: Integrate with Demo CLI (2-3 days)

**Task**: Add gaming detection to `demo_trace.py`.

**Modify**: `scripts/demo_trace.py`

```python
@app.command()
def gaming_report(
    events_path: Path = typer.Option(
        "data/interim/edm_cup_2023_42_events.parquet",
        help="Path to events"
    ),
    output_path: Path = typer.Option(
        "reports/gaming_alerts.parquet",
        help="Output path for alerts"
    ),
    severity_filter: str = typer.Option(
        None,
        help="Filter by severity (low/medium/high)"
    ),
):
    """Generate gaming detection report for all students."""
    from src.common.gaming_detection import generate_gaming_report
    
    events_df = pd.read_parquet(events_path)
    
    console.print("Analyzing student behavior patterns...")
    alerts_df = generate_gaming_report(events_df)
    
    if severity_filter:
        alerts_df = alerts_df[alerts_df["severity"] == severity_filter]
    
    # Save report
    alerts_df.to_parquet(output_path)
    
    # Display summary
    console.print(f"\n[bold]Gaming Detection Report[/bold]")
    console.print(f"Total students analyzed: {events_df['user_id'].nunique():,}")
    console.print(f"Students flagged: {alerts_df['user_id'].nunique():,}")
    console.print(f"\nAlerts by type:")
    console.print(alerts_df.groupby("alert_type")["user_id"].count())
    console.print(f"\nAlerts by severity:")
    console.print(alerts_df.groupby("severity")["user_id"].count())
    console.print(f"\n✅ Saved to {output_path}")

@app.command()
def check_gaming(
    user_id: str = typer.Argument(..., help="Student ID to check"),
    events_path: Path = typer.Option(
        "data/interim/edm_cup_2023_42_events.parquet",
        help="Path to events"
    ),
):
    """Check gaming behavior for a specific student."""
    from src.common.gaming_detection import analyze_student
    
    events_df = pd.read_parquet(events_path)
    alerts = analyze_student(events_df, user_id)
    
    if not alerts:
        console.print(f"[green]✅ No gaming alerts for {user_id}[/green]")
        return
    
    console.print(f"[yellow]⚠️ Gaming Alerts for {user_id}[/yellow]\n")
    
    for alert in alerts:
        color = {"low": "yellow", "medium": "orange3", "high": "red"}[alert.severity]
        console.print(f"[{color}][{alert.severity.upper()}] {alert.alert_type}[/{color}]")
        console.print(f"  Evidence: {alert.evidence}")
        console.print(f"  Recommendation: {alert.recommendation}\n")
```

---

### M2.3: Tests (2 days)

**New File**: `tests/test_gaming_detection.py`

```python
# ABOUTME: Tests for gaming detection module.
# ABOUTME: Validates rapid guessing, help abuse, and suspicious pattern detection.

import pytest
import pandas as pd
import numpy as np
from src.common.gaming_detection import (
    detect_rapid_guessing,
    detect_help_abuse,
    detect_suspicious_patterns,
    analyze_student,
)

@pytest.fixture
def normal_student_events():
    """Events for a normal student."""
    return pd.DataFrame({
        "user_id": ["student_normal"] * 20,
        "item_id": [f"item_{i}" for i in range(20)],
        "correct": np.random.choice([0, 1], 20, p=[0.3, 0.7]),
        "latency_ms": np.random.randint(10000, 60000, 20),  # 10-60 seconds
        "help_requested": [False] * 18 + [True, True],  # 10% help
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
    })

@pytest.fixture
def rapid_guesser_events():
    """Events for a rapid guessing student."""
    return pd.DataFrame({
        "user_id": ["student_rapid"] * 20,
        "item_id": [f"item_{i}" for i in range(20)],
        "correct": [0] * 15 + [1] * 5,  # Mostly incorrect
        "latency_ms": [2000] * 15 + [30000] * 5,  # 75% rapid
        "help_requested": [False] * 20,
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
    })

@pytest.fixture
def help_abuser_events():
    """Events for a help-abusing student."""
    return pd.DataFrame({
        "user_id": ["student_help"] * 20,
        "item_id": [f"item_{i}" for i in range(20)],
        "correct": np.random.choice([0, 1], 20),
        "latency_ms": np.random.randint(10000, 60000, 20),
        "help_requested": [True] * 12 + [False] * 8,  # 60% help
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
    })

class TestRapidGuessing:
    def test_no_alert_for_normal_student(self, normal_student_events):
        alert = detect_rapid_guessing(normal_student_events, "student_normal")
        assert alert is None or alert.severity == "low"
    
    def test_high_alert_for_rapid_guesser(self, rapid_guesser_events):
        alert = detect_rapid_guessing(rapid_guesser_events, "student_rapid")
        assert alert is not None
        assert alert.severity in ["medium", "high"]
        assert alert.alert_type == "rapid_guessing"

class TestHelpAbuse:
    def test_no_alert_for_normal_student(self, normal_student_events):
        alert = detect_help_abuse(normal_student_events, "student_normal")
        assert alert is None
    
    def test_alert_for_help_abuser(self, help_abuser_events):
        alert = detect_help_abuse(help_abuser_events, "student_help")
        assert alert is not None
        assert alert.alert_type == "help_abuse"
        assert alert.evidence["help_ratio"] >= 0.3

class TestAnalyzeStudent:
    def test_returns_all_alerts(self, rapid_guesser_events):
        alerts = analyze_student(rapid_guesser_events, "student_rapid")
        assert isinstance(alerts, list)
```

---

## Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | M1.1: Extract attention weights | `sakt_attention.parquet` |
| 1-2 | M1.2: Explanation generator | `src/common/explainability.py` |
| 2 | M1.3: Demo CLI integration | `demo explain` command |
| 3 | M2.1: Gaming detector | `src/common/gaming_detection.py` |
| 3-4 | M2.2: Demo CLI + M2.3: Tests | `demo gaming-report`, tests |

---

## Acceptance Criteria (Phase 5A Complete)

- [ ] `demo explain <user> <skill>` shows attention-based explanation
- [ ] `demo gaming-report` generates alerts for all students
- [ ] `demo check-gaming <user>` shows alerts for specific student
- [ ] All tests pass
- [ ] README updated with new commands

---

## Revision Log

- 2025-11-27: Initial implementation plan

