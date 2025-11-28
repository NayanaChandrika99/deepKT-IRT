# ABOUTME: Generates item recommendations by combining mastery and item health.
# ABOUTME: Filters items by skill/topic, drift status, and difficulty fit.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


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
    Recommend items for a student on a given skill/topic.
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
    for _, row in candidates.head(max_items).iterrows():
        reason = f"Skill {target_skill} mastery {mastery_mean:.2f}, item difficulty {float(row['difficulty']):.2f}"
        recs.append(
            ItemRecommendation(
                item_id=str(row["item_id"]),
                topic=str(row["topic"]),
                difficulty=float(row["difficulty"]),
                discrimination=float(row.get("discrimination", 1.0)),
                reason=reason,
            )
        )
    return recs
