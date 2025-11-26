# ABOUTME: Builds clickstream-derived feature tensors for the WD-IRT model.
# ABOUTME: Generates history features and metadata aligned with the paper's Student Action framing.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.common.schemas import LearningEvent

PAD_ACTION = 0
ACTION_CORRECT_NO_HELP = 1
ACTION_CORRECT_HELP = 2
ACTION_WRONG_NO_HELP = 3
ACTION_WRONG_HELP = 4


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for history features."""

    seq_len: int = 200
    latency_bins: int = 10
    bert_dim: int = 32


@dataclass(frozen=True)
class FeatureStats:
    """Global statistics used for feature normalization."""

    latency_edges: np.ndarray
    item_success_rates: Mapping[str, float]
    global_success_rate: float


def compute_feature_stats(events: pd.DataFrame, latency_bins: int = 10) -> FeatureStats:
    """Compute latency quantile edges and per-item success rates."""

    latencies = events["latency_ms"].dropna()
    latencies = latencies[latencies >= 0]
    if latencies.empty:
        edges = np.linspace(0.0, 1.0, latency_bins + 1)
    else:
        edges = np.quantile(latencies, np.linspace(0, 1, latency_bins + 1))
    edges[0] = 0.0
    edges[-1] = np.inf

    success_rates = events.groupby("item_id")["correct"].mean().to_dict()
    global_rate = float(events["correct"].mean())

    return FeatureStats(latency_edges=edges, item_success_rates=success_rates, global_success_rate=global_rate)


def load_problem_embeddings(problem_details_path: Path, bert_dim: int = 32) -> Dict[str, np.ndarray]:
    """Parse problem_text_bert_pca column into dense vectors."""

    df = pd.read_csv(problem_details_path, usecols=["problem_id", "problem_text_bert_pca", "problem_type"])
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        raw = row["problem_text_bert_pca"]
        vector = np.zeros(bert_dim, dtype=np.float32)
        if isinstance(raw, str) and raw.startswith("[") and raw.endswith("]"):
            parts = [p.strip() for p in raw.strip("[]").split(",") if p.strip()]
            floats = [float(p) for p in parts[:bert_dim]]
            vector[: len(floats)] = np.array(floats, dtype=np.float32)
        embeddings[row["problem_id"]] = vector
    return embeddings


def encode_history_sequences(
    events: Sequence[LearningEvent],
    reference_time: datetime,
    config: FeatureConfig,
    stats: FeatureStats,
    item_embeddings: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Encode the student's historical interactions into fixed-length sequences."""

    seq_len = config.seq_len
    # Convert to sorted dataframe for easier manipulation.
    rows = []
    for event in events:
        if event.timestamp >= reference_time:
            continue
        rows.append(
            {
                "timestamp": event.timestamp,
                "item_id": event.item_id,
                "correct": event.correct,
                "help_requested": bool(event.help_requested),
                "latency_ms": event.latency_ms if event.latency_ms is not None else -1,
            }
        )
    if not rows:
        rows_df = pd.DataFrame(columns=["timestamp", "item_id", "correct", "help_requested", "latency_ms"])
    else:
        rows_df = pd.DataFrame(rows)
        rows_df = rows_df.sort_values("timestamp")

    # Keep the most recent seq_len events.
    rows_df = rows_df.tail(seq_len)
    pad = seq_len - len(rows_df)

    action_codes = np.full(seq_len, PAD_ACTION, dtype=np.int64)
    recency_hours = np.zeros(seq_len, dtype=np.float32)
    latency_bins = np.zeros(seq_len, dtype=np.int64)
    success_rates = np.full(seq_len, stats.global_success_rate, dtype=np.float32)
    embeddings = np.zeros((seq_len, config.bert_dim), dtype=np.float32)
    missing = np.ones(seq_len, dtype=np.int64)

    latency_edges = stats.latency_edges

    for idx, (_, row) in enumerate(rows_df.iterrows()):
        slot = pad + idx
        action_codes[slot] = _encode_action(row["correct"], row["help_requested"])
        recency_hours[slot] = max(
            0.0,
            float((reference_time - row["timestamp"]).total_seconds()) / 3600.0,
        )
        latency_bins[slot] = _bucketize_latency(row["latency_ms"], latency_edges)
        success_rates[slot] = float(stats.item_success_rates.get(row["item_id"], stats.global_success_rate))
        embeddings[slot] = item_embeddings.get(row["item_id"], np.zeros(config.bert_dim, dtype=np.float32))
        missing[slot] = 0

    mask = missing.astype(np.float32)
    return {
        "history_actions": action_codes,
        "history_recency": recency_hours,
        "history_latency_bucket": latency_bins,
        "history_item_success_rates": success_rates,
        "history_item_embeddings": embeddings,
        "history_missing": missing,
        "mask": mask,
    }


def _encode_action(correct: int, help_requested: bool) -> int:
    """Map correctness/help flags to discrete action ids."""

    if correct:
        return ACTION_CORRECT_HELP if help_requested else ACTION_CORRECT_NO_HELP
    return ACTION_WRONG_HELP if help_requested else ACTION_WRONG_NO_HELP


def _bucketize_latency(latency_ms: float, edges: np.ndarray) -> int:
    """Bucket latency measurements based on precomputed quantile edges."""

    if latency_ms is None or latency_ms < 0:
        return 0
    return int(np.searchsorted(edges, latency_ms, side="right") - 1)


def parse_learning_events(events_iter: Iterable[LearningEvent]) -> pd.DataFrame:
    """Convert LearningEvent iterable to a pandas DataFrame for aggregation."""

    rows = []
    for event in events_iter:
        rows.append(
            {
                "user_id": event.user_id,
                "item_id": event.item_id,
                "timestamp": event.timestamp,
                "correct": event.correct,
                "help_requested": bool(event.help_requested),
                "latency_ms": event.latency_ms,
                "action_sequence_id": event.action_sequence_id,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def reference_timestamp_from_assignment(start_time_seconds: float | None, fallback: datetime) -> datetime:
    """Convert assignment start time into datetime with fallback."""

    if pd.notna(start_time_seconds):
        return datetime.fromtimestamp(float(start_time_seconds), tz=timezone.utc)
    return fallback
