# ABOUTME: Generates human-readable explanations from attention patterns.
# ABOUTME: Translates SAKT attention into insights and recommendations.

from __future__ import annotations

import os
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
    Generate insight and specific recommendation from attention factors.
    Uses actual item IDs and skills from key_factors for actionable advice.
    
    This is the FALLBACK template-based method. Use LLM version if available.
    """
    if not key_factors:
        return "Insufficient history for an explanation.", "Practice more problems in this skill."

    correct_count = sum(1 for f in key_factors if f.get("correct"))
    incorrect_count = len(key_factors) - correct_count

    # Calculate recency bias: what fraction of attention is on recent interactions?
    # A score close to 1 means attention is on recent items, close to 0 means older items
    positions = [f.get("position", 0) for f in key_factors if f.get("position") is not None]
    recency_bias = 0.5  # Default neutral
    if positions:
        max_pos = max(positions)
        if max_pos > 0:
            # Normalize positions to [0, 1] and compute mean
            # Mean close to 1 = focus on recent, close to 0 = focus on old
            normalized_positions = [pos / max_pos for pos in positions]
            recency_bias = sum(normalized_positions) / len(normalized_positions)
        else:
            recency_bias = 0.5

    # Extract specific items and skills for recommendations
    missed_items = [f for f in key_factors if not f.get("correct")]
    missed_item_ids = [f.get("item_id", "")[:10] for f in missed_items[:3]]
    missed_skills = list({f.get("skill", "") for f in missed_items if f.get("skill")})[:2]

    if mastery_score < 0.4:
        if incorrect_count >= 2 and recency_bias > 0.6:
            insight = "Recent struggles heavily influenced this prediction."
            if missed_item_ids:
                rec = f"Review items: {', '.join(missed_item_ids)}. "
                if missed_skills:
                    rec += f"Focus on skills: {', '.join(missed_skills)}."
                else:
                    rec += "Then try similar problems."
            else:
                rec = "Review your most recent incorrect answers before attempting new ones."
            return insight, rec
        return (
            "Limited correct attempts in this skill.",
            "Start with easier problems to rebuild confidence.",
        )

    if mastery_score < 0.7:
        if correct_count and incorrect_count:
            insight = "Mixed performance across attempts."
            if missed_item_ids:
                rec = f"Review items you missed: {', '.join(missed_item_ids)}. "
                if missed_skills:
                    rec += f"Practice more problems in: {', '.join(missed_skills)}."
                else:
                    rec += "Then try similar problems to stabilize mastery."
            else:
                rec = "Practice more problems to stabilize mastery."
            return insight, rec
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
    # Handle numpy arrays and Series - convert to list of dicts
    raw_influences = latest["top_influences"] if "top_influences" in latest.index else []
    if hasattr(raw_influences, "tolist"):
        raw_influences = raw_influences.tolist()
    key_factors = list(raw_influences) if raw_influences is not None else []

    # enrich with skill ids when possible
    # Pre-filter events to user and create lookup dict for O(1) access
    user_events = events_df[events_df["user_id"] == user_id]
    if not user_events.empty:
        # Create item_id -> event lookup
        event_lookup = {row["item_id"]: row for _, row in user_events.iterrows()}

        for factor in key_factors:
            if not isinstance(factor, dict):
                continue
            item_id = factor.get("item_id")
            if not item_id or item_id not in event_lookup:
                continue

            event = event_lookup[item_id]
            skills = event.get("skill_ids", [])
            if hasattr(skills, "tolist"):
                skills = skills.tolist()
            factor["skill"] = skills[0] if skills else "unknown"

    # Try LLM first, fallback to templates
    use_llm = os.environ.get("USE_LLM_EXPLANATIONS", "false").lower() == "true"
    if use_llm:
        try:
            from .llm_explainability import (
                build_attention_context,
                generate_llm_insight_recommendation_sync,
            )
            context = build_attention_context(
                user_id=user_id,
                skill_id=skill_id,
                mastery_score=mastery_score,
                key_factors=key_factors,
                interaction_count=interaction_count,
            )
            insight, recommendation = generate_llm_insight_recommendation_sync(context)
        except Exception:
            # Fallback to template-based
            insight, recommendation = analyze_attention_pattern(key_factors, mastery_score)
    else:
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
