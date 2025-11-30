# ABOUTME: Generates item recommendations by combining mastery and item health.
# ABOUTME: Supports rule-based and RL (LinUCB bandit) recommendation modes.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .bandit import (
    LinUCBBandit,
    BanditRecommendation,
    build_student_context,
    items_to_arms,
    generate_rl_reason,
    is_exploring,
)


@dataclass
class ItemRecommendation:
    item_id: str
    topic: str
    difficulty: float
    discrimination: float
    reason: str


def recommend_items(
    user_id: str,
    target_skill: str,
    skill_mastery: pd.DataFrame,
    item_params: pd.DataFrame,
    max_items: int = 5,
    exclude_high_drift: bool = True,
) -> List[ItemRecommendation]:
    """
    Recommend items for a student on a given skill/topic (rule-based).
    """

    mastery_row = skill_mastery[
        (skill_mastery["user_id"] == user_id) & (skill_mastery["skill"] == target_skill)
    ]
    if mastery_row.empty:
        mastery_mean = 0.5
    else:
        mastery_mean = float(mastery_row.iloc[0]["mastery_mean"])

    candidates = item_params[item_params["topic"] == target_skill].copy()
    if exclude_high_drift and "drift_flag" in candidates.columns:
        candidates = candidates[~candidates["drift_flag"].fillna(False)]

    candidates = candidates.sort_values("difficulty")
    recs: List[ItemRecommendation] = []
    
    # Check if LLM should be used
    import os
    use_llm = os.environ.get("USE_LLM_EXPLANATIONS", "false").lower() == "true"
    
    for _, row in candidates.head(max_items).iterrows():
        item_id = str(row["item_id"])
        item_difficulty = float(row["difficulty"])
        
        if use_llm:
            try:
                from .llm_explainability import generate_llm_rule_based_reason_sync
                reason = generate_llm_rule_based_reason_sync(
                    user_id, target_skill, mastery_mean, item_id, item_difficulty
                )
            except Exception:
                # Fallback to template
                reason = f"Skill {target_skill} mastery {mastery_mean:.2f}, item difficulty {item_difficulty:.2f}"
        else:
            reason = f"Skill {target_skill} mastery {mastery_mean:.2f}, item difficulty {item_difficulty:.2f}"
        
        recs.append(
            ItemRecommendation(
                item_id=item_id,
                topic=str(row["topic"]),
                difficulty=item_difficulty,
                discrimination=float(row.get("discrimination", 1.0)),
                reason=reason,
            )
        )
    return recs


def recommend_items_rl(
    user_id: str,
    target_skill: str,
    item_params: pd.DataFrame,
    events_df: pd.DataFrame,
    bandit: LinUCBBandit,
    max_items: int = 5,
    exclude_high_drift: bool = True,
) -> List[BanditRecommendation]:
    """
    Recommend items using LinUCB contextual bandit.

    Args:
        user_id: Student identifier
        target_skill: Skill/topic to recommend for
        item_params: DataFrame with item_id, topic, difficulty columns
        events_df: Student events for building context
        bandit: Trained LinUCB bandit instance
        max_items: Maximum recommendations to return
        exclude_high_drift: Whether to exclude items with drift_flag=True

    Returns:
        List of BanditRecommendation sorted by UCB score
    """
    student = build_student_context(user_id, events_df, target_skill=target_skill)

    candidates = item_params[item_params["topic"] == target_skill].copy()
    if exclude_high_drift and "drift_flag" in candidates.columns:
        candidates = candidates[~candidates["drift_flag"].fillna(False)]

    items = items_to_arms(candidates)
    if not items:
        return []

    recommendations = []
    for item in items:
        expected, uncertainty = bandit.predict(student, item)
        ucb = expected + uncertainty
        # Use bandit's configurable exploration threshold
        exploration_flag = is_exploring(uncertainty, expected, bandit.exploration_threshold)

        # Use LLM-aware reason generator
        from .bandit import generate_rl_reason_with_llm
        reason = generate_rl_reason_with_llm(student, item, expected, uncertainty, exploration_flag)

        recommendations.append(
            BanditRecommendation(
                item=item,
                expected_reward=expected,
                uncertainty=uncertainty,
                ucb_score=ucb,
                reason=reason,
                is_exploration=exploration_flag,
            )
        )

    recommendations.sort(key=lambda r: r.ucb_score, reverse=True)
    return recommendations[:max_items]
