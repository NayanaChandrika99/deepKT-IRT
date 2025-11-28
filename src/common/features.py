# ABOUTME: Declares reusable feature builders operating on canonical events.
# ABOUTME: Provides placeholder functions that will feed both engines.

from typing import Iterable

import pandas as pd

from .schemas import LearningEvent


def build_clickstream_features(events: Iterable[LearningEvent]) -> pd.DataFrame:
    """
    Convert canonical learning events into aggregated clickstream features.

    The actual implementation will reproduce the feature groups defined in the
    Wide & Deep IRT paper (e.g., latency buckets, help request ratios).
    """

    rows = []
    for event in events:
        rows.append(
            {
                "user_id": event.user_id,
                "timestamp": event.timestamp,
                "correct": event.correct,
                "latency_ms": event.latency_ms,
                "help_requested": False if event.help_requested is None else bool(event.help_requested),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "user_id",
                "total_attempts",
                "correct_rate",
                "avg_latency_ms",
                "median_latency_ms",
                "help_rate",
                "attempts_last_24h",
                "first_ts",
                "last_ts",
            ]
        )

    df = pd.DataFrame(rows)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df["help_requested"] = df["help_requested"].fillna(False)

    # Ensure timestamp is datetime for window calculations.
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    grouped_rows = []
    for user_id, user_df in df.groupby("user_id", sort=False):
        total_attempts = len(user_df)
        correct_rate = float(user_df["correct"].mean()) if total_attempts else 0.0
        latencies = user_df["latency_ms"].dropna()
        avg_latency = float(latencies.mean()) if not latencies.empty else float("nan")
        median_latency = float(latencies.median()) if not latencies.empty else float("nan")
        help_rate = float(user_df["help_requested"].mean()) if total_attempts else 0.0
        first_ts = user_df["timestamp"].min()
        last_ts = user_df["timestamp"].max()

        window_start = last_ts - pd.Timedelta(hours=24)
        attempts_last_24h = int(user_df[user_df["timestamp"] >= window_start].shape[0])

        grouped_rows.append(
            {
                "user_id": user_id,
                "total_attempts": total_attempts,
                "correct_rate": correct_rate,
                "avg_latency_ms": avg_latency,
                "median_latency_ms": median_latency,
                "help_rate": help_rate,
                "attempts_last_24h": attempts_last_24h,
                "first_ts": first_ts,
                "last_ts": last_ts,
            }
        )

    grouped = pd.DataFrame(grouped_rows)

    return grouped
