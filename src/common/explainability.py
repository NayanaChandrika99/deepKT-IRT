# ABOUTME: Generates human-readable explanations from attention patterns.
# ABOUTME: Translates SAKT attention into insights and recommendations.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd


@dataclass
class MasteryExplanation:
    user_id: str
    skill_id: str
    mastery_score: float
    key_factors: List[Dict]
    insight: str
    recommendation: str
    confidence: str


def analyze_attention_pattern(key_factors: List[Dict], mastery_score: float) -> Tuple[str, str]:
    """
    Generate a short insight and recommendation from attention factors.
    """
    if not key_factors:
        return "Insufficient history for an explanation.", "Practice more problems in this skill."

    correct_count = sum(1 for f in key_factors if f.get("correct"))
    incorrect_count = len(key_factors) - correct_count
    positions = [f.get("position", 0) for f in key_factors if f.get("position") is not None]
    recency_bias = 0.5
    if positions:
        max_pos = max(positions)
        recency_bias = (sum(positions) / len(positions)) / max_pos if max_pos else 0.5

    if mastery_score < 0.4:
        if incorrect_count >= 2 and recency_bias > 0.6:
            return (
                "Recent struggles heavily influenced this prediction.",
                "Review the most recent incorrect problems before attempting new ones.",
            )
        return (
            "Limited correct attempts in this skill.",
            "Start with easier problems to rebuild confidence.",
        )

    if mastery_score < 0.7:
        if correct_count and incorrect_count:
            return (
                "Mixed performance across attempts.",
                "Focus on the specific problem types you missed to stabilize mastery.",
            )
        return (
            "Moderate confidence based on available history.",
            "Continue practicing similar problems to solidify understanding.",
        )

    # mastery high
    if correct_count >= 3:
        return (
            "Strong performance track record for this skill.",
            "Progress to more challenging or related skills.",
        )
    return (
        "Good performance with limited interactions.",
        "A few more successful attempts will confirm mastery.",
    )


def generate_explanation(
    user_id: str,
    skill_id: str,
    mastery_score: float,
    attention_data: pd.DataFrame,
    events_df: pd.DataFrame,
    interaction_count: int,
) -> MasteryExplanation:
    """
    Create a MasteryExplanation using attention data and event context.
    """
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

    latest = user_attention.iloc[-1]
    key_factors = latest.get("top_influences", []) or []

    # enrich with skill ids when possible
    for factor in key_factors:
        event = events_df[
            (events_df["user_id"] == user_id) & (events_df["item_id"] == factor.get("item_id"))
        ]
        if not event.empty:
            skills = event.iloc[0].get("skill_ids", [])
            factor["skill"] = skills[0] if skills else "unknown"

    insight, recommendation = analyze_attention_pattern(key_factors, mastery_score)

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
    """Render explanation as a human-readable block."""
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
        for idx, factor in enumerate(explanation.key_factors, 1):
            emoji = "✅" if factor.get("correct") else "❌"
            weight = factor.get("weight", 0.0) * 100
            item = str(factor.get("item_id", ""))[:12]
            skill = factor.get("skill", "")
            lines.append(
                f"  #{idx}  {emoji} {item} ({weight:.0f}% influence)"
                f"{f' [{skill}]' if skill else ''}"
            )
    else:
        lines.append("  No attention data available.")

    lines.extend(
        [
            "",
            "INSIGHT:",
            f"  {explanation.insight}",
            "",
            "RECOMMENDATION:",
            f"  {explanation.recommendation}",
            "━" * 60,
        ]
    )
    return "\n".join(lines)
