# Phase 5A Implementation Plan: Explainable Analytics

This ExecPlan is a living document for implementing explainable features on top of the twin-engine system.

## Purpose / Big Picture

**For UWorld Demo**: Add "why" to our predictions. Current demo shows *what* (mastery=0.45), Phase 5A adds *why* (because you struggled with Q8, Q10 on ratios).

**Value Proposition:**
- **Trust**: Students/educators understand predictions, not just accept them
- **Actionable**: "Focus on these specific problems" vs. generic "practice more"
- **Differentiation**: Competitors (TrueLearn, AMBOSS) don't offer attention-based explanations
- **Research-backed**: Based on XKT literature (He et al. 2024, Pandey & Karypis 2019)

**Deliverables:**
1. `src/common/explainability.py` - Attention-based explanations from SAKT
2. `src/common/gaming_detection.py` - Behavior anomaly detection from clickstream
3. Enhanced `demo_trace.py` - Show "why" alongside recommendations
4. Tests and documentation

---

## How SAKT Attention Works (Background)

### The Mechanism

SAKT uses **scaled dot-product self-attention** to predict student performance. For each prediction, it computes attention weights over all past interactions:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

The **softmax output** gives weights [0-1] for each past interaction, showing which ones the model considers most relevant.

### Concrete Example

**Student history (10 questions):**
```
Position | Question   | Correct | Skill
---------|------------|---------|--------
   1     | Q_algebra  |    1    | 7.RP.A.1
   2     | Q_ratios   |    1    | 7.RP.A.1
   3     | Q_fracs    |    0    | 6.NS.A.1
   ...
   8     | Q_ratios2  |    0    | 7.RP.A.1  ← struggled
   9     | Q_graphs   |    1    | 8.EE.B.5
  10     | Q_ratios3  |    0    | 7.RP.A.1  ← struggled
```

**Predicting Q11 (7.RP.A.1 - Ratios):**

SAKT computes attention weights:
```
Position | Question   | Attention Weight | Interpretation
---------|------------|------------------|----------------
  10     | Q_ratios3  |      0.35       | Most recent, same skill, wrong
   8     | Q_ratios2  |      0.22       | Recent, same skill, wrong
   2     | Q_ratios   |      0.12       | Same skill, but correct + older
   5     | Q_props    |      0.11       | Related skill, correct
   ...   | ...        |      ...        | Lower weights for other items
```

**Result**: P(correct) = 0.35 (low mastery) because model attended heavily to recent failures on same skill.

### Translating to Explanation

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Student: ABC123
Skill: 7.RP.A.1 (Ratios & Proportional Relationships)
Predicted Mastery: 0.35

WHY THIS PREDICTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The model focused on these past interactions:

  #1  ❌ Q_ratios3 (35% influence) - Incorrect, most recent
  #2  ❌ Q_ratios2 (22% influence) - Incorrect
  #3  ✅ Q_ratios  (12% influence) - Correct, but older
  #4  ✅ Q_props   (11% influence) - Correct, related skill

INSIGHT:
Recent performance on ratio problems (2 wrong in a row)
heavily influenced the low mastery estimate.

RECOMMENDATION:
Review Q_ratios2 and Q_ratios3, then try similar items.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Progress

- [ ] Milestone 1: Extract attention weights from pyKT SAKT
- [ ] Milestone 2: Build explanation generator
- [ ] Milestone 3: Implement gaming detection
- [ ] Milestone 4: Integrate with demo CLI
- [ ] Milestone 5: Tests and documentation

---

## Milestone 1: Extract Attention Weights from SAKT (3-4 days)

### Challenge

pyKT's SAKT doesn't expose attention weights by default. The model returns predictions only.

### Solution: Forward Hooks

Use PyTorch forward hooks to capture attention weights during inference without modifying pyKT code.

### Implementation

**New file: `src/sakt_kt/attention_extractor.py`**

```python
# ABOUTME: Extracts attention weights from pyKT's SAKT model during inference.
# ABOUTME: Uses PyTorch forward hooks to capture intermediate attention computations.

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


class AttentionExtractor:
    """
    Captures attention weights from SAKT during forward pass.
    
    SAKT's attention mechanism:
    - Query: current exercise embedding
    - Key: past interaction embeddings
    - Value: past interaction embeddings
    
    Attention weights show which past interactions influence each prediction.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights: List[torch.Tensor] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
    def _attention_hook(self, module, input, output):
        """
        Hook to capture attention weights from scaled dot-product attention.
        
        In SAKT, attention is computed as:
            scores = Q @ K.T / sqrt(d_k)
            weights = softmax(scores, dim=-1)
        
        We capture the weights (post-softmax).
        """
        # pyKT's SAKT attention returns (output, attention_weights) when available
        # Otherwise, we compute weights from Q, K directly
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_weights.append(output[1].detach().cpu())
        else:
            # Fallback: compute attention from Q, K in input
            # This depends on pyKT's exact implementation
            pass
    
    def _find_attention_layers(self) -> List[nn.Module]:
        """Find attention layers in SAKT model."""
        attention_layers = []
        
        for name, module in self.model.named_modules():
            # pyKT's SAKT uses MultiheadAttention or custom attention
            if 'attn' in name.lower() or 'attention' in name.lower():
                attention_layers.append(module)
            elif isinstance(module, nn.MultiheadAttention):
                attention_layers.append(module)
        
        return attention_layers
    
    def register_hooks(self):
        """Register forward hooks on attention layers."""
        self.attention_weights = []
        attention_layers = self._find_attention_layers()
        
        for layer in attention_layers:
            hook = layer.register_forward_hook(self._attention_hook)
            self.hooks.append(hook)
        
        return len(self.hooks)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract(
        self,
        q_seq: torch.Tensor,
        r_seq: torch.Tensor,
        qry_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run forward pass and extract attention weights.
        
        Args:
            q_seq: Question sequence [batch, seq_len]
            r_seq: Response sequence [batch, seq_len]
            qry_seq: Query sequence (shifted questions) [batch, seq_len]
        
        Returns:
            predictions: Model output [batch, seq_len]
            attention_weights: List of attention tensors [batch, heads, seq_len, seq_len]
        """
        self.attention_weights = []
        self.register_hooks()
        
        try:
            with torch.no_grad():
                predictions = self.model(q_seq, r_seq, qry_seq)
        finally:
            self.remove_hooks()
        
        return predictions, self.attention_weights


def compute_attention_from_scratch(
    model: nn.Module,
    q_seq: torch.Tensor,
    r_seq: torch.Tensor,
    qry_seq: torch.Tensor,
    d_k: int = 64,
) -> torch.Tensor:
    """
    Manually compute attention weights if hooks don't capture them.
    
    This replicates SAKT's attention computation:
    1. Get exercise embeddings from q_seq
    2. Get interaction embeddings from (q_seq, r_seq) combined
    3. Compute scaled dot-product attention
    
    Args:
        model: SAKT model with embedding layers
        q_seq: Question sequence
        r_seq: Response sequence
        qry_seq: Query sequence
        d_k: Key dimension for scaling
    
    Returns:
        attention_weights: [batch, seq_len, seq_len]
    """
    with torch.no_grad():
        # Get embeddings
        # Note: Exact layer names depend on pyKT implementation
        exercise_emb = model.exercise_emb(qry_seq)  # Query: current question
        
        # Interaction embedding: combines question + response
        # In SAKT: interaction = question_emb + (response * 2 - 1) * some_embedding
        interaction_idx = q_seq + r_seq * model.num_c  # Standard SAKT encoding
        interaction_emb = model.interaction_emb(interaction_idx)  # Keys/Values
        
        # Scaled dot-product attention
        # Q: [batch, seq_len, d_model]
        # K: [batch, seq_len, d_model]
        scores = torch.matmul(exercise_emb, interaction_emb.transpose(-2, -1))
        scores = scores / np.sqrt(d_k)
        
        # Apply causal mask (can't attend to future)
        seq_len = q_seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask.unsqueeze(0), float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        return attention_weights


def extract_top_influences(
    attention_weights: torch.Tensor,
    item_ids: List[str],
    responses: List[int],
    position: int,
    k: int = 5,
) -> List[Dict]:
    """
    Get top-k most influential past interactions for a prediction.
    
    Args:
        attention_weights: [seq_len, seq_len] attention matrix
        item_ids: List of item IDs in sequence
        responses: List of responses (0/1) in sequence
        position: Position of prediction (which row to use)
        k: Number of top influences to return
    
    Returns:
        List of dicts with item_id, response, weight, position
    """
    # Get attention weights for this prediction (row = position)
    weights = attention_weights[position, :position]  # Can only attend to past
    
    if len(weights) == 0:
        return []
    
    # Get top-k indices
    top_k = min(k, len(weights))
    top_indices = weights.argsort(descending=True)[:top_k]
    
    influences = []
    for idx in top_indices:
        idx = int(idx)
        influences.append({
            "item_id": item_ids[idx],
            "correct": bool(responses[idx]),
            "weight": float(weights[idx]),
            "position": idx,
        })
    
    return influences
```

### Output Schema

```python
# sakt_attention.parquet
{
    "user_id": str,                    # Student ID
    "prediction_position": int,        # Position in sequence
    "predicted_item_id": str,          # Item being predicted
    "predicted_mastery": float,        # P(correct)
    "top_influences": List[Dict],      # Top-5 attended items
    # Each influence: {"item_id", "correct", "weight", "position"}
}
```

### Acceptance Criteria

- [ ] `AttentionExtractor` captures attention from pyKT SAKT
- [ ] Attention weights sum to ~1.0 per prediction
- [ ] Top-5 influences extracted correctly
- [ ] Works with existing trained model checkpoint

---

## Milestone 2: Explanation Generator (2-3 days)

### Implementation

**New file: `src/common/explainability.py`**

```python
# ABOUTME: Generates human-readable explanations from SAKT attention weights.
# ABOUTME: Maps attention patterns to interpretable factors affecting mastery.

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd


@dataclass
class MasteryExplanation:
    """Human-readable explanation for a mastery prediction."""
    user_id: str
    skill_id: str
    mastery_score: float
    key_factors: List[Dict]  # Top-k attended interactions
    insight: str             # Pattern-based insight
    recommendation: str      # Actionable next step
    confidence: str          # High/Medium/Low based on interaction count


def analyze_attention_pattern(
    key_factors: List[Dict],
    mastery_score: float,
) -> tuple[str, str]:
    """
    Analyze attention pattern to generate insight and recommendation.
    
    Patterns we detect:
    1. Recent failures dominate → "Recent struggles affecting score"
    2. Skill mismatch → "Model focused on related but different skills"
    3. Success history → "Strong foundation in this skill"
    4. Mixed signals → "Inconsistent performance"
    """
    if not key_factors:
        return "Insufficient history", "Practice more problems in this skill"
    
    # Count correct/incorrect in top factors
    correct_count = sum(1 for f in key_factors if f["correct"])
    incorrect_count = len(key_factors) - correct_count
    
    # Check recency (are recent items weighted higher?)
    positions = [f["position"] for f in key_factors]
    avg_position = sum(positions) / len(positions)
    max_position = max(positions)
    recency_bias = avg_position / max_position if max_position > 0 else 0.5
    
    # Generate insight
    if mastery_score < 0.4:
        if incorrect_count >= 3 and recency_bias > 0.7:
            insight = "Recent struggles heavily influenced this prediction. You missed several related problems recently."
            recommendation = "Review the problems you missed: " + ", ".join(
                f["item_id"][:8] for f in key_factors if not f["correct"]
            )[:50]
        elif incorrect_count >= 3:
            insight = "Multiple incorrect attempts across your history lowered the prediction."
            recommendation = "Start with easier problems in this skill to rebuild confidence."
        else:
            insight = "Model is uncertain due to limited correct attempts in this skill."
            recommendation = "Practice more problems to establish a track record."
    
    elif mastery_score < 0.7:
        if correct_count >= 2 and incorrect_count >= 2:
            insight = "Inconsistent performance - some successes but also some struggles."
            recommendation = "Focus on the specific problem types you missed."
        elif recency_bias < 0.4:
            insight = "Recent performance not reflected - older attempts weighted higher."
            recommendation = "Your recent attempts may not have been on similar problems. Try more."
        else:
            insight = "Moderate confidence based on mixed historical performance."
            recommendation = "Continue practicing to solidify understanding."
    
    else:  # mastery >= 0.7
        if correct_count >= 4:
            insight = "Strong track record on this skill. Model confident in your mastery."
            recommendation = "Ready for more challenging problems or move to related skills."
        else:
            insight = "Good performance, though based on limited interactions."
            recommendation = "A few more successful attempts will confirm mastery."
    
    return insight, recommendation


def generate_explanation(
    user_id: str,
    skill_id: str,
    mastery_score: float,
    attention_data: pd.DataFrame,
    events_df: pd.DataFrame,
    interaction_count: int,
) -> MasteryExplanation:
    """
    Generate complete explanation for a mastery prediction.
    
    Args:
        user_id: Student ID
        skill_id: Skill being explained
        mastery_score: Predicted mastery (0-1)
        attention_data: DataFrame with attention weights
        events_df: Original events (for skill lookup)
        interaction_count: Total interactions for this user
    """
    # Get attention data for this user's most recent prediction
    user_attention = attention_data[attention_data["user_id"] == user_id]
    
    if user_attention.empty:
        return MasteryExplanation(
            user_id=user_id,
            skill_id=skill_id,
            mastery_score=mastery_score,
            key_factors=[],
            insight="No attention data available for explanation.",
            recommendation="Practice more problems in this skill.",
            confidence="low",
        )
    
    # Get top influences from most recent prediction
    latest = user_attention.iloc[-1]
    key_factors = latest.get("top_influences", [])
    
    # Enrich factors with skill info from events
    for factor in key_factors:
        event = events_df[
            (events_df["user_id"] == user_id) &
            (events_df["item_id"] == factor["item_id"])
        ]
        if not event.empty:
            skills = event.iloc[0].get("skill_ids", [])
            factor["skill"] = skills[0] if skills else "unknown"
    
    # Generate insight and recommendation
    insight, recommendation = analyze_attention_pattern(key_factors, mastery_score)
    
    # Determine confidence based on interaction count
    if interaction_count < 5:
        confidence = "low"
    elif interaction_count < 20:
        confidence = "medium"
    else:
        confidence = "high"
    
    return MasteryExplanation(
        user_id=user_id,
        skill_id=skill_id,
        mastery_score=mastery_score,
        key_factors=key_factors,
        insight=insight,
        recommendation=recommendation,
        confidence=confidence,
    )


def format_explanation(explanation: MasteryExplanation) -> str:
    """Format explanation for CLI display."""
    lines = [
        "━" * 60,
        f"Student: {explanation.user_id}",
        f"Skill: {explanation.skill_id}",
        f"Mastery: {explanation.mastery_score:.2f} (confidence: {explanation.confidence})",
        "",
        "WHY THIS PREDICTION:",
        "━" * 60,
    ]
    
    if explanation.key_factors:
        lines.append("The model focused on these past interactions:")
        lines.append("")
        
        for i, factor in enumerate(explanation.key_factors, 1):
            emoji = "✅" if factor["correct"] else "❌"
            skill = factor.get("skill", "")[:15]
            lines.append(
                f"  #{i}  {emoji} {factor['item_id'][:12]} "
                f"({factor['weight']*100:.0f}% influence) - "
                f"{'Correct' if factor['correct'] else 'Incorrect'}"
                f"{f' [{skill}]' if skill else ''}"
            )
    else:
        lines.append("  No attention data available.")
    
    lines.extend([
        "",
        f"INSIGHT:",
        f"  {explanation.insight}",
        "",
        f"RECOMMENDATION:",
        f"  {explanation.recommendation}",
        "━" * 60,
    ])
    
    return "\n".join(lines)
```

### Acceptance Criteria

- [ ] `generate_explanation()` produces valid explanations
- [ ] `analyze_attention_pattern()` detects meaningful patterns
- [ ] Explanations are actionable (not just "practice more")
- [ ] Handles edge cases (no data, single interaction)

---

## Milestone 3: Gaming Detection (1-2 days)

### Background

Gaming detection uses clickstream data (latency, help requests) to identify:
1. **Rapid guessing**: < 5 second responses
2. **Help abuse**: > 30% help requests before attempting
3. **Suspicious patterns**: Rapid wrong streak → sudden correct

### Implementation

**New file: `src/common/gaming_detection.py`**

```python
# ABOUTME: Detects gaming/cheating behavior from student response patterns.
# ABOUTME: Uses latency, help requests, and correctness sequences.

from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import numpy as np


@dataclass
class GamingAlert:
    """Alert for potential gaming behavior."""
    user_id: str
    alert_type: str      # "rapid_guessing", "help_abuse", "suspicious_pattern"
    severity: str        # "low", "medium", "high"
    evidence: Dict       # Supporting metrics
    recommendation: str  # Action for educator


# Configurable thresholds
class GamingThresholds:
    RAPID_RESPONSE_MS = 5000      # < 5 seconds = rapid
    HELP_ABUSE_RATIO = 0.30       # > 30% help = concerning
    RAPID_INCORRECT_STREAK = 5   # 5+ rapid incorrect = suspicious
    MIN_INTERACTIONS = 10        # Need enough data to detect


def detect_rapid_guessing(
    events_df: pd.DataFrame,
    user_id: str,
    threshold_ms: int = GamingThresholds.RAPID_RESPONSE_MS,
) -> Optional[GamingAlert]:
    """Detect rapid guessing (< 5s responses, mostly incorrect)."""
    user_events = events_df[events_df["user_id"] == user_id]
    
    if len(user_events) < GamingThresholds.MIN_INTERACTIONS:
        return None
    
    rapid = user_events[user_events["latency_ms"] < threshold_ms]
    rapid_ratio = len(rapid) / len(user_events)
    
    if rapid_ratio < 0.15:
        return None  # Normal
    
    # Check if rapid responses are mostly incorrect
    rapid_incorrect = rapid[rapid["correct"] == 0]
    rapid_incorrect_ratio = len(rapid_incorrect) / len(rapid) if len(rapid) > 0 else 0
    
    # Determine severity
    if rapid_ratio >= 0.40 and rapid_incorrect_ratio >= 0.70:
        severity = "high"
        rec = "URGENT: Strong gaming pattern. Student may be frustrated. Educator review needed."
    elif rapid_ratio >= 0.25:
        severity = "medium"
        rec = "Student rushing through problems. Consider discussing study habits."
    else:
        severity = "low"
        rec = "Minor rapid response pattern. Continue monitoring."
    
    return GamingAlert(
        user_id=user_id,
        alert_type="rapid_guessing",
        severity=severity,
        evidence={
            "rapid_responses": len(rapid),
            "total_responses": len(user_events),
            "rapid_ratio_pct": round(rapid_ratio * 100, 1),
            "rapid_incorrect_pct": round(rapid_incorrect_ratio * 100, 1),
            "avg_rapid_time_ms": int(rapid["latency_ms"].mean()) if len(rapid) > 0 else 0,
        },
        recommendation=rec,
    )


def detect_help_abuse(
    events_df: pd.DataFrame,
    user_id: str,
    threshold: float = GamingThresholds.HELP_ABUSE_RATIO,
) -> Optional[GamingAlert]:
    """Detect excessive help requests."""
    user_events = events_df[events_df["user_id"] == user_id]
    
    if "help_requested" not in user_events.columns:
        return None
    
    if len(user_events) < GamingThresholds.MIN_INTERACTIONS:
        return None
    
    help_count = user_events["help_requested"].sum()
    help_ratio = help_count / len(user_events)
    
    if help_ratio < threshold:
        return None
    
    severity = "high" if help_ratio >= 0.50 else "medium"
    
    return GamingAlert(
        user_id=user_id,
        alert_type="help_abuse",
        severity=severity,
        evidence={
            "help_requests": int(help_count),
            "total_responses": len(user_events),
            "help_ratio_pct": round(help_ratio * 100, 1),
        },
        recommendation="Student frequently requests help. May need scaffolding or foundational review.",
    )


def detect_suspicious_patterns(
    events_df: pd.DataFrame,
    user_id: str,
) -> Optional[GamingAlert]:
    """Detect rapid wrong streak → sudden correct (answer-seeking)."""
    user_events = events_df[events_df["user_id"] == user_id].sort_values("timestamp")
    
    if len(user_events) < 15:
        return None
    
    suspicious_count = 0
    rapid_incorrect_streak = 0
    
    for _, row in user_events.iterrows():
        is_rapid = row["latency_ms"] < GamingThresholds.RAPID_RESPONSE_MS
        is_incorrect = row["correct"] == 0
        
        if is_rapid and is_incorrect:
            rapid_incorrect_streak += 1
        elif rapid_incorrect_streak >= GamingThresholds.RAPID_INCORRECT_STREAK and row["correct"] == 1:
            suspicious_count += 1
            rapid_incorrect_streak = 0
        else:
            rapid_incorrect_streak = 0
    
    if suspicious_count < 2:
        return None
    
    severity = "high" if suspicious_count >= 5 else "medium"
    
    return GamingAlert(
        user_id=user_id,
        alert_type="suspicious_pattern",
        severity=severity,
        evidence={
            "suspicious_sequences": suspicious_count,
            "pattern": "rapid_incorrect_streak_then_correct",
        },
        recommendation="Pattern suggests answer-seeking. Review session recordings if available.",
    )


def analyze_student(events_df: pd.DataFrame, user_id: str) -> List[GamingAlert]:
    """Run all detectors for a student."""
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

### Acceptance Criteria

- [ ] Rapid guessing detection works (< 5s, mostly incorrect)
- [ ] Help abuse detection works (> 30% help requests)
- [ ] Suspicious pattern detection works
- [ ] All thresholds are configurable

---

## Milestone 4: Demo CLI Integration (2 days)

### New Commands for `demo_trace.py`

```python
@app.command()
def explain(
    user_id: str = typer.Argument(..., help="Student ID"),
    skill: str = typer.Option(None, help="Specific skill to explain"),
):
    """
    Explain why a student has their current mastery level.
    
    Shows which past interactions influenced the prediction.
    """
    from src.common.explainability import generate_explanation, format_explanation
    
    # Load data
    events_df = pd.read_parquet("data/interim/edm_cup_2023_42_events.parquet")
    mastery_df = pd.read_parquet("reports/sakt_student_state.parquet")
    attention_df = pd.read_parquet("reports/sakt_attention.parquet")
    
    # Get user's skills
    user_events = events_df[events_df["user_id"] == user_id]
    if user_events.empty:
        console.print(f"[red]No data for user {user_id}[/red]")
        return
    
    # Determine skill to explain
    if skill is None:
        # Use skill with lowest mastery
        skills = [s for row in user_events["skill_ids"] for s in (row if row else [])]
        skill = skills[0] if skills else None
    
    # Get mastery score
    user_mastery = mastery_df[mastery_df["user_id"] == user_id]
    mastery_score = user_mastery["mastery"].mean() if not user_mastery.empty else 0.5
    
    # Generate explanation
    explanation = generate_explanation(
        user_id=user_id,
        skill_id=skill or "overall",
        mastery_score=mastery_score,
        attention_data=attention_df,
        events_df=events_df,
        interaction_count=len(user_events),
    )
    
    console.print(format_explanation(explanation))


@app.command()
def gaming_check(
    user_id: str = typer.Argument(None, help="Student ID (or all students)"),
    severity: str = typer.Option(None, help="Filter by severity"),
    output: Path = typer.Option("reports/gaming_alerts.parquet", help="Output path"),
):
    """
    Check for gaming/cheating behavior.
    
    Detects rapid guessing, help abuse, and suspicious patterns.
    """
    from src.common.gaming_detection import analyze_student, generate_gaming_report
    
    events_df = pd.read_parquet("data/interim/edm_cup_2023_42_events.parquet")
    
    if user_id:
        # Single student
        alerts = analyze_student(events_df, user_id)
        
        if not alerts:
            console.print(f"[green]✅ No gaming alerts for {user_id}[/green]")
            return
        
        console.print(f"[yellow]⚠️ Gaming Alerts for {user_id}[/yellow]\n")
        for alert in alerts:
            color = {"low": "yellow", "medium": "orange3", "high": "red"}[alert.severity]
            console.print(f"[{color}][{alert.severity.upper()}] {alert.alert_type}[/{color}]")
            for k, v in alert.evidence.items():
                console.print(f"    {k}: {v}")
            console.print(f"    → {alert.recommendation}\n")
    else:
        # All students
        console.print("Analyzing all students...")
        alerts_df = generate_gaming_report(events_df)
        
        if severity:
            alerts_df = alerts_df[alerts_df["severity"] == severity]
        
        alerts_df.to_parquet(output)
        
        console.print(f"\n[bold]Gaming Detection Summary[/bold]")
        console.print(f"Students analyzed: {events_df['user_id'].nunique():,}")
        console.print(f"Students flagged: {alerts_df['user_id'].nunique():,}")
        console.print(f"Total alerts: {len(alerts_df):,}")
        console.print(f"\nBy severity:")
        for sev in ["high", "medium", "low"]:
            count = len(alerts_df[alerts_df["severity"] == sev])
            if count > 0:
                console.print(f"  {sev}: {count:,}")
        console.print(f"\n✅ Saved to {output}")
```

### Sample Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Student: 1006OOQBE9
Skill: 7.RP.A.1
Mastery: 0.45 (confidence: high)

WHY THIS PREDICTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The model focused on these past interactions:

  #1  ❌ ZK1I2M76F (35% influence) - Incorrect [7.RP.A.1]
  #2  ❌ VPZ9SBLWG (22% influence) - Incorrect [7.RP.A.1]
  #3  ✅ 1IVTOQT7GK (12% influence) - Correct [7.RP.A.1]
  #4  ✅ 2FPPD9076Y (11% influence) - Correct [7.RP.A.2]
  #5  ❌ 2RNRZF1BSI (8% influence) - Incorrect [7.RP.A.1]

INSIGHT:
  Recent struggles heavily influenced this prediction. You missed 
  several related problems recently.

RECOMMENDATION:
  Review the problems you missed: ZK1I2M76, VPZ9SBLW, 2RNRZF1B
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Milestone 5: Tests and Documentation (1 day)

### Tests

**`tests/test_explainability.py`**

```python
import pytest
from src.common.explainability import (
    generate_explanation,
    analyze_attention_pattern,
    format_explanation,
)

def test_analyze_low_mastery_recent_failures():
    factors = [
        {"correct": False, "position": 9, "weight": 0.35},
        {"correct": False, "position": 8, "weight": 0.22},
        {"correct": True, "position": 2, "weight": 0.12},
    ]
    insight, rec = analyze_attention_pattern(factors, 0.35)
    assert "recent" in insight.lower() or "struggle" in insight.lower()

def test_analyze_high_mastery_success():
    factors = [
        {"correct": True, "position": 9, "weight": 0.30},
        {"correct": True, "position": 8, "weight": 0.25},
        {"correct": True, "position": 7, "weight": 0.20},
        {"correct": True, "position": 6, "weight": 0.15},
    ]
    insight, rec = analyze_attention_pattern(factors, 0.85)
    assert "strong" in insight.lower() or "confident" in insight.lower()

def test_format_explanation_structure():
    from src.common.explainability import MasteryExplanation
    exp = MasteryExplanation(
        user_id="test",
        skill_id="7.RP.A.1",
        mastery_score=0.5,
        key_factors=[],
        insight="Test insight",
        recommendation="Test recommendation",
        confidence="medium",
    )
    output = format_explanation(exp)
    assert "test" in output
    assert "7.RP.A.1" in output
    assert "Test insight" in output
```

**`tests/test_gaming_detection.py`**

```python
import pytest
import pandas as pd
import numpy as np
from src.common.gaming_detection import (
    detect_rapid_guessing,
    detect_help_abuse,
    analyze_student,
)

@pytest.fixture
def normal_student():
    return pd.DataFrame({
        "user_id": ["normal"] * 20,
        "correct": [1, 1, 0, 1, 1, 0, 1, 1, 1, 0] * 2,
        "latency_ms": np.random.randint(15000, 60000, 20),
        "help_requested": [False] * 18 + [True, True],
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
    })

@pytest.fixture
def gamer_student():
    return pd.DataFrame({
        "user_id": ["gamer"] * 20,
        "correct": [0] * 15 + [1] * 5,
        "latency_ms": [2000] * 15 + [30000] * 5,
        "help_requested": [False] * 20,
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
    })

def test_normal_student_no_alert(normal_student):
    alert = detect_rapid_guessing(normal_student, "normal")
    assert alert is None or alert.severity == "low"

def test_gamer_detected(gamer_student):
    alert = detect_rapid_guessing(gamer_student, "gamer")
    assert alert is not None
    assert alert.severity in ["medium", "high"]
```

### Documentation

Update README.md with new commands:

```markdown
## Explainability Features (Phase 5A)

### Explain a Student's Mastery

```bash
python scripts/demo_trace.py explain USER_ID --skill "7.RP.A.1"
```

Shows which past interactions influenced the mastery prediction.

### Gaming Detection

```bash
# Check single student
python scripts/demo_trace.py gaming-check USER_ID

# Generate report for all students
python scripts/demo_trace.py gaming-check --output reports/gaming_alerts.parquet
```

Detects:
- Rapid guessing (< 5s responses)
- Help abuse (> 30% help requests)
- Suspicious patterns (rapid wrong → sudden correct)
```

---

## Timeline

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | M1: Attention extraction | `src/sakt_kt/attention_extractor.py` |
| 1-2 | M2: Explanation generator | `src/common/explainability.py` |
| 2 | M3: Gaming detection | `src/common/gaming_detection.py` |
| 3 | M4: Demo CLI | `demo explain`, `demo gaming-check` |
| 3 | M5: Tests + docs | Tests pass, README updated |

**Total: 3 weeks**

---

## UWorld Demo Value

**Before Phase 5A:**
> "Student mastery on Ratios is 0.45"

**After Phase 5A:**
> "Student mastery on Ratios is 0.45 **because** they struggled with Q8 and Q10 recently. The model focused on these interactions (35% and 22% influence). **Recommendation:** Review Q8 and Q10, then try similar problems."

**Gaming Detection:**
> "Alert: Student XYZ789 shows rapid guessing pattern (40% of responses < 5s, 75% incorrect). **Recommendation:** Educator review needed."

This transforms the demo from "black box predictions" to "transparent, actionable analytics."

---

## Revision Log

- 2025-11-27: Initial implementation plan
- 2025-11-28: Revamped with detailed attention extraction mechanism and UWorld alignment
