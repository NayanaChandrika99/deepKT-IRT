# ABOUTME: Defines canonical data structures shared by both learning engines.
# ABOUTME: Centralizes event, mastery, and item health schema definitions.

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Mapping, Optional


@dataclass(frozen=True)
class LearningEvent:
    """Canonical learning-event row produced by preprocessing."""

    user_id: str
    item_id: str
    skill_ids: List[str]
    timestamp: datetime
    correct: int
    action_sequence_id: Optional[str] = None
    latency_ms: Optional[int] = None
    help_requested: Optional[bool] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StudentMasterySnapshot:
    """Represents a student mastery vector at a specific timestamp."""

    user_id: str
    topic: str
    timestamp: datetime
    mastery: float
    prediction: Optional[float] = None


@dataclass(frozen=True)
class ItemHealthRecord:
    """Stores item-level health indicators exported by Wide & Deep IRT."""

    item_id: str
    topic: str
    difficulty: float
    discrimination: float
    guessing: float
    drift_flag: bool
    drift_score: Optional[float] = None
