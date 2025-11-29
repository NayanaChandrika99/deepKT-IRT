# ABOUTME: Aggregates per-interaction mastery into per-skill mastery summaries.
# ABOUTME: Joins SAKT outputs with events to enable skill-level recommendations.

from __future__ import annotations

from typing import Iterable

import pandas as pd


def aggregate_skill_mastery(mastery_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-interaction mastery scores to per-skill summaries.

    Steps:
    - Ensure events are ordered per user and annotated with positional index.
    - Join mastery predictions on (user_id, position).
    - Explode multi-skill events.
    - Compute mean/std/count per (user_id, skill).
    """

    if mastery_df is None or events_df is None or mastery_df.empty or events_df.empty:
        return pd.DataFrame(columns=["user_id", "skill", "mastery_mean", "mastery_std", "interaction_count"])

    events = events_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(events.get("timestamp")):
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.sort_values(["user_id", "timestamp"], kind="mergesort")
    events["position"] = events.groupby("user_id").cumcount()

    joined = mastery_df.merge(
        events[["user_id", "position", "skill_ids"]],
        on=["user_id", "position"],
        how="inner",
        validate="one_to_one",
    )

    # Normalize skill_ids to lists and explode.
    joined["skill_ids"] = joined["skill_ids"].apply(_normalize_skills)
    exploded = joined.explode("skill_ids")
    exploded = exploded.dropna(subset=["skill_ids"])

    if exploded.empty:
        return pd.DataFrame(columns=["user_id", "skill", "mastery_mean", "mastery_std", "interaction_count"])

    grouped = (
        exploded.groupby(["user_id", "skill_ids"])
        .agg(
            mastery_mean=("mastery", "mean"),
            mastery_std=("mastery", lambda s: float(s.std(ddof=0))),
            interaction_count=("mastery", "count"),
        )
        .reset_index()
        .rename(columns={"skill_ids": "skill"})
    )

    # Replace NaN std (single sample) with 0.0
    grouped["mastery_std"] = grouped["mastery_std"].fillna(0.0)
    return grouped


def _normalize_skills(value) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    # Handle numpy arrays
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return [str(v) for v in value if v]
    except ImportError:
        pass
    return []
