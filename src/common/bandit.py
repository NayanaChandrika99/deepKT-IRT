# ABOUTME: Implements contextual multi-armed bandit for adaptive recommendations.
# ABOUTME: Uses LinUCB algorithm to balance exploration and exploitation.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class StudentContext:
    """Student features for contextual bandit."""

    user_id: str
    mastery: float  # Overall mastery (0-1)
    recent_accuracy: float  # Last N questions accuracy
    recent_speed: float  # Normalized response time (0=fast, 1=slow)
    help_tendency: float  # How often they request help (0-1)
    skill_gap: float  # Distance from target skill mastery


@dataclass
class ItemArm:
    """Item (arm) in the bandit."""

    item_id: str
    skill: str
    difficulty: float
    discrimination: float = 1.0


@dataclass
class BanditRecommendation:
    """Recommendation with bandit metadata."""

    item: ItemArm
    expected_reward: float  # Predicted P(success)
    uncertainty: float  # Confidence width
    ucb_score: float  # UCB = expected + uncertainty
    reason: str  # Human-readable explanation
    is_exploration: bool  # True if uncertainty dominated selection


class LinUCBBandit:
    """
    Linear Upper Confidence Bound (LinUCB) Contextual Bandit.

    Algorithm:
    1. Receive context (student features + item features)
    2. Predict expected reward using linear model: E[r] = theta @ x
    3. Compute uncertainty: sqrt(x @ A_inv @ x)
    4. Select item with highest UCB: E[r] + alpha * uncertainty
    5. Observe reward and update model
    """

    def __init__(self, n_features: int = 8, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha  # Exploration parameter
        self.A = np.eye(n_features)  # Design matrix (regularized covariance)
        self.b = np.zeros(n_features)  # Reward-weighted features
        self.theta = np.zeros(n_features)  # Learned weights
        self.n_updates = 0

    def get_context_vector(self, student: StudentContext, item: ItemArm) -> np.ndarray:
        """Create feature vector from student-item pair."""
        return np.array(
            [
                student.mastery,
                student.recent_accuracy,
                student.recent_speed,
                student.help_tendency,
                student.skill_gap,
                item.difficulty,
                student.mastery * item.difficulty,  # Challenge match
                abs(student.mastery - item.difficulty),  # Difficulty gap
            ]
        )

    def predict(self, student: StudentContext, item: ItemArm) -> Tuple[float, float]:
        """Predict expected reward and uncertainty."""
        x = self.get_context_vector(student, item)
        expected = np.clip(np.dot(self.theta, x), 0, 1)
        A_inv = np.linalg.inv(self.A)
        uncertainty = self.alpha * np.sqrt(np.dot(x, np.dot(A_inv, x)))
        return float(expected), float(uncertainty)

    def select_best(
        self,
        student: StudentContext,
        items: List[ItemArm],
    ) -> Tuple[Optional[ItemArm], float, float, float]:
        """Select best item using UCB."""
        if not items:
            return None, 0.0, 0.0, 0.0

        best: Optional[ItemArm] = None
        best_ucb = float("-inf")
        best_expected = 0.0
        best_uncertainty = 0.0

        for item in items:
            expected, uncertainty = self.predict(student, item)
            ucb = expected + uncertainty
            if ucb > best_ucb:
                best_ucb = ucb
                best = item
                best_expected = expected
                best_uncertainty = uncertainty

        return best, best_expected, best_uncertainty, best_ucb

    def update(self, student: StudentContext, item: ItemArm, reward: float) -> None:
        """Update model with observed reward (0 or 1)."""
        x = self.get_context_vector(student, item)
        self.A += np.outer(x, x)
        self.b += reward * x
        self.theta = np.linalg.solve(self.A, self.b)
        self.n_updates += 1

    def save(self, path: Path) -> None:
        """Save bandit state to file."""
        np.savez(
            path,
            A=self.A,
            b=self.b,
            theta=self.theta,
            n_features=self.n_features,
            alpha=self.alpha,
            n_updates=self.n_updates,
        )

    @classmethod
    def load(cls, path: Path) -> "LinUCBBandit":
        """Load bandit state from file."""
        data = np.load(path)
        bandit = cls(n_features=int(data["n_features"]), alpha=float(data["alpha"]))
        bandit.A = data["A"]
        bandit.b = data["b"]
        bandit.theta = data["theta"]
        bandit.n_updates = int(data["n_updates"])
        return bandit


def build_student_context(
    user_id: str,
    events_df: pd.DataFrame,
    mastery_df: Optional[pd.DataFrame] = None,
    target_skill: Optional[str] = None,
) -> StudentContext:
    """Build StudentContext from events data."""
    user_events = events_df[events_df["user_id"] == user_id].tail(20)

    if len(user_events) < 3:
        return StudentContext(
            user_id=user_id,
            mastery=0.5,
            recent_accuracy=0.5,
            recent_speed=0.5,
            help_tendency=0.1,
            skill_gap=0.3,
        )

    mastery = float(user_events["correct"].mean())
    recent_5 = user_events.tail(5)
    recent_accuracy = float(recent_5["correct"].mean())

    avg_latency = (
        recent_5["latency_ms"].mean() if "latency_ms" in recent_5.columns else 30000
    )
    recent_speed = min(avg_latency / 60000, 1.0)

    help_tendency = (
        float(user_events["help_requested"].mean())
        if "help_requested" in user_events.columns
        else 0.1
    )

    skill_gap = max(0, 0.8 - mastery)

    return StudentContext(
        user_id=user_id,
        mastery=mastery,
        recent_accuracy=recent_accuracy,
        recent_speed=float(recent_speed),
        help_tendency=help_tendency,
        skill_gap=skill_gap,
    )


def items_to_arms(
    item_params_df: pd.DataFrame, skill_filter: Optional[str] = None
) -> List[ItemArm]:
    """Convert item_params DataFrame to list of ItemArm."""
    df = item_params_df.copy()
    if skill_filter:
        df = df[df["topic"] == skill_filter]

    arms = []
    for _, row in df.iterrows():
        arms.append(
            ItemArm(
                item_id=str(row["item_id"]),
                skill=str(row.get("topic", "unknown")),
                difficulty=float(row["difficulty"]),
                discrimination=float(row.get("discrimination", 1.0)),
            )
        )
    return arms


def generate_rl_reason(
    student: StudentContext,
    item: ItemArm,
    expected: float,
    uncertainty: float,
    is_exploration: bool,
) -> str:
    """
    Generate human-readable reason for RL recommendation.
    
    This is the FALLBACK template-based method. Use LLM version if available.
    """
    parts = []

    if is_exploration:
        parts.append("Exploring to learn preferences")
    else:
        parts.append(f"High confidence: {expected:.0%} expected success")

    diff_gap = abs(student.mastery - item.difficulty)
    if diff_gap < 0.15:
        parts.append("good difficulty match")
    elif item.difficulty > student.mastery:
        parts.append("slightly challenging for growth")
    else:
        parts.append("builds confidence")

    return "; ".join(parts)


def generate_rl_reason_with_llm(
    student: StudentContext,
    item: ItemArm,
    expected: float,
    uncertainty: float,
    is_exploration: bool,
) -> str:
    """
    Generate human-readable reason for RL recommendation using LLM.
    Falls back to template-based if LLM unavailable.
    """
    import os
    use_llm = os.environ.get("USE_LLM_EXPLANATIONS", "false").lower() == "true"
    if use_llm:
        try:
            from .llm_explainability import generate_llm_rl_reason_sync
            return generate_llm_rl_reason_sync(student, item, expected, uncertainty, is_exploration)
        except Exception:
            # Fallback to template
            return generate_rl_reason(student, item, expected, uncertainty, is_exploration)
    return generate_rl_reason(student, item, expected, uncertainty, is_exploration)

