# ABOUTME: Handles ingestion and preprocessing for clickstream datasets.
# ABOUTME: Provides adapters from EDM Cup data into the canonical schema.

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common.schemas import LearningEvent
from .features import (
    FeatureConfig,
    FeatureStats,
    compute_feature_stats,
    encode_history_sequences,
    load_problem_embeddings,
)

# Default fallback timestamp when no information is available.
FALLBACK_TIMESTAMP = datetime(2020, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class EdmDatasetPaths:
    """File paths required to build the EDM dataset."""

    events: Path
    assignment_details: Path
    assignment_relationships: Path
    unit_test_scores: Path
    problem_details: Path
    split_manifest: Path


@dataclass
class UnitTestSample:
    """Lightweight container describing a unit test record."""

    unit_assignment_id: str
    student_id: str
    future_item_id: str
    future_item_idx: int
    future_item_is_mc: float
    in_unit_assignment_ids: List[str]
    label: float


class EdmClickstreamDataset(Dataset):
    """PyTorch dataset producing Wide & Deep IRT training samples."""

    def __init__(
        self,
        split: str,
        paths: EdmDatasetPaths,
        feature_config: FeatureConfig,
        max_samples: Optional[int] = None,
    ) -> None:
        self.split = split
        self.paths = paths
        self.feature_config = feature_config
        self.max_samples = max_samples

        self.events_df = self._load_events(paths.events)
        self.event_indices = self.events_df.groupby("action_sequence_id").indices
        self.stats = compute_feature_stats(self.events_df, latency_bins=feature_config.latency_bins)
        self.item_embeddings = load_problem_embeddings(paths.problem_details, bert_dim=feature_config.bert_dim)

        self.assignment_students = self._load_assignment_students(paths.assignment_details)
        self.assignment_start_times = self._load_assignment_start_times(paths.assignment_details)
        self.unit_to_inunit = self._load_assignment_relationships(paths.assignment_relationships)
        self.problem_meta = self._load_problem_metadata(paths.problem_details)

        self.splits = self._load_split_manifest(paths.split_manifest)
        self.allowed_students = set(self.splits.get(split, []))

        self.scores_df = pd.read_csv(paths.unit_test_scores).dropna(subset=["score"])
        self.item_to_index = {pid: idx for idx, pid in enumerate(sorted(self.scores_df["problem_id"].unique()))}
        self.item_vocab_size = len(self.item_to_index)

        self.samples: List[UnitTestSample] = []
        self.history_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        features = self._get_history_features(sample)
        if features is None:
            features = self._empty_features()

        inputs = {
            "future_item": torch.tensor(sample.future_item_idx, dtype=torch.long),
            "future_item_is_mc": torch.tensor(sample.future_item_is_mc, dtype=torch.float32),
            "history_actions": torch.from_numpy(features["history_actions"]).long(),
            "history_recency": torch.from_numpy(features["history_recency"]).float(),
            "history_latency_bucket": torch.from_numpy(features["history_latency_bucket"]).long(),
            "history_item_success_rates": torch.from_numpy(features["history_item_success_rates"]).float(),
            "history_item_embeddings": torch.from_numpy(features["history_item_embeddings"]).float(),
            "history_missing": torch.from_numpy(features["history_missing"]).float(),
            "mask": torch.from_numpy(features["mask"]).float(),
        }
        label = torch.tensor(sample.label, dtype=torch.float32)
        metadata = {
            "student_id": sample.student_id,
            "unit_assignment_id": sample.unit_assignment_id,
            "item_id": sample.future_item_id,
        }
        return inputs, label, metadata

    def _build_samples(self) -> None:
        for _, row in self.scores_df.iterrows():
            unit_assignment_id = row["assignment_log_id"]
            future_item_id = row["problem_id"]
            student_id = self.assignment_students.get(unit_assignment_id)
            if not student_id or (self.allowed_students and student_id not in self.allowed_students):
                continue

            in_unit_ids = self.unit_to_inunit.get(unit_assignment_id, [])
            if not in_unit_ids:
                continue

            if future_item_id not in self.item_to_index:
                continue

            sample = UnitTestSample(
                unit_assignment_id=unit_assignment_id,
                student_id=student_id,
                future_item_id=future_item_id,
                future_item_idx=self.item_to_index[future_item_id],
                future_item_is_mc=self._is_multiple_choice(future_item_id),
                in_unit_assignment_ids=in_unit_ids,
                label=float(row["score"]),
            )
            self.samples.append(sample)

            if self.max_samples and len(self.samples) >= self.max_samples:
                break

    def _get_history_features(self, sample: UnitTestSample) -> Optional[Dict[str, np.ndarray]]:
        if sample.unit_assignment_id in self.history_cache:
            return self.history_cache[sample.unit_assignment_id]

        history_events: List[LearningEvent] = []
        for assignment_id in sample.in_unit_assignment_ids:
            events = self._events_for_assignment(assignment_id)
            if events:
                history_events.extend(events)

        if not history_events:
            self.history_cache[sample.unit_assignment_id] = self._empty_features()
            return self.history_cache[sample.unit_assignment_id]

        reference_time = self._reference_time(sample.unit_assignment_id, history_events)
        features = encode_history_sequences(
            history_events,
            reference_time,
            self.feature_config,
            self.stats,
            self.item_embeddings,
        )
        self.history_cache[sample.unit_assignment_id] = features
        return features

    def _reference_time(self, unit_assignment_id: str, history_events: List[LearningEvent]) -> datetime:
        start_time = self.assignment_start_times.get(unit_assignment_id)
        if start_time is not None:
            return start_time
        if history_events:
            return history_events[-1].timestamp + timedelta(seconds=1)
        return FALLBACK_TIMESTAMP

    def _events_for_assignment(self, assignment_id: str) -> List[LearningEvent]:
        idxs = self.event_indices.get(assignment_id)
        if idxs is None:
            return []
        subset = self.events_df.iloc[idxs]
        events: List[LearningEvent] = []
        for row in subset.itertuples(index=False):
            timestamp = row.timestamp if isinstance(row.timestamp, datetime) else pd.to_datetime(row.timestamp, utc=True)
            latency_ms = int(row.latency_ms) if row.latency_ms == row.latency_ms else None
            skill_ids = row.skill_ids if isinstance(row.skill_ids, list) else []
            events.append(
                LearningEvent(
                    user_id=row.user_id,
                    item_id=row.item_id,
                    skill_ids=skill_ids,
                    timestamp=timestamp,
                    correct=int(row.correct),
                    action_sequence_id=row.action_sequence_id,
                    latency_ms=latency_ms,
                    help_requested=bool(row.help_requested),
                )
            )
        return events

    def _empty_features(self) -> Dict[str, np.ndarray]:
        seq_len = self.feature_config.seq_len
        bert_dim = self.feature_config.bert_dim
        return {
            "history_actions": np.zeros(seq_len, dtype=np.int64),
            "history_recency": np.zeros(seq_len, dtype=np.float32),
            "history_latency_bucket": np.zeros(seq_len, dtype=np.int64),
            "history_item_success_rates": np.full(seq_len, self.stats.global_success_rate, dtype=np.float32),
            "history_item_embeddings": np.zeros((seq_len, bert_dim), dtype=np.float32),
            "history_missing": np.ones(seq_len, dtype=np.float32),
            "mask": np.ones(seq_len, dtype=np.float32),
        }

    @staticmethod
    def _load_events(events_path: Path) -> pd.DataFrame:
        df = pd.read_parquet(events_path)
        df = df.dropna(subset=["action_sequence_id"])
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _load_assignment_students(details_path: Path) -> Dict[str, str]:
        df = pd.read_csv(details_path, usecols=["assignment_log_id", "student_id"])
        return df.set_index("assignment_log_id")["student_id"].to_dict()

    @staticmethod
    def _load_assignment_start_times(details_path: Path) -> Dict[str, datetime]:
        df = pd.read_csv(details_path, usecols=["assignment_log_id", "assignment_start_time"])
        df["assignment_start_time"] = pd.to_numeric(df["assignment_start_time"], errors="coerce")
        df["assignment_start_time"] = pd.to_datetime(df["assignment_start_time"], unit="s", errors="coerce", utc=True)
        valid = df.dropna(subset=["assignment_start_time"])
        return {
            row["assignment_log_id"]: row["assignment_start_time"].to_pydatetime()
            for _, row in valid.iterrows()
        }

    @staticmethod
    def _load_assignment_relationships(rel_path: Path) -> Dict[str, List[str]]:
        df = pd.read_csv(rel_path)
        rels = df.groupby("unit_test_assignment_log_id")["in_unit_assignment_log_id"].apply(list)
        return rels.to_dict()

    @staticmethod
    def _load_problem_metadata(problem_details_path: Path) -> Dict[str, float]:
        df = pd.read_csv(problem_details_path, usecols=["problem_id", "problem_type"])
        df["is_mc"] = df["problem_type"].fillna("").str.contains("Multiple Choice", case=False).astype(float)
        return df.set_index("problem_id")["is_mc"].to_dict()

    @staticmethod
    def _load_split_manifest(split_path: Path) -> Dict[str, Sequence[str]]:
        return json.loads(Path(split_path).read_text())

    def _is_multiple_choice(self, problem_id: str) -> float:
        return float(self.problem_meta.get(problem_id, 1.0))


def load_edm_clickstream(events_path: Path) -> Iterable[LearningEvent]:
    """Backward-compatible helper retained for external callers."""
    df = EdmClickstreamDataset._load_events(events_path)
    for row in df.itertuples(index=False):
        yield LearningEvent(
            user_id=row.user_id,
            item_id=row.item_id,
            skill_ids=row.skill_ids if isinstance(row.skill_ids, list) else [],
            timestamp=row.timestamp if isinstance(row.timestamp, datetime) else pd.to_datetime(row.timestamp, utc=True),
            correct=int(row.correct),
            action_sequence_id=row.action_sequence_id,
            latency_ms=row.latency_ms if row.latency_ms == row.latency_ms else None,
            help_requested=bool(row.help_requested),
        )


def summarize_sessions(raw_path: Path) -> pd.DataFrame:
    """Placeholder retained for API compatibility."""
    raise NotImplementedError("Session summarization will be implemented alongside feature exports.")


def collate_wdirt_batch(batch):
    """Collate function that stacks dictionary inputs for WD-IRT."""

    inputs_list, labels_list, metadata_list = zip(*batch)
    collated_inputs = {}
    for key in inputs_list[0]:
        tensors = [sample[key] for sample in inputs_list]
        collated_inputs[key] = torch.stack(tensors, dim=0)
    labels = torch.stack(labels_list, dim=0).unsqueeze(-1)
    return collated_inputs, labels, metadata_list
