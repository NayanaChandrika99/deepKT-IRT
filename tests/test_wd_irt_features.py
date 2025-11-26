# ABOUTME: Tests clickstream history feature encoding for WD-IRT.
# ABOUTME: Ensures sequences are padded, bucketized, and embedded correctly.

from datetime import datetime, timedelta, timezone
import unittest

import numpy as np
import pandas as pd

from src.common.schemas import LearningEvent
from src.wd_irt.features import (
    FeatureConfig,
    FeatureStats,
    compute_feature_stats,
    encode_history_sequences,
)


def _event(ts_offset_hours: float, item_id: str, correct: int, help_flag: bool, latency_ms: int) -> LearningEvent:
    ts = datetime.now(timezone.utc) - timedelta(hours=ts_offset_hours)
    return LearningEvent(
        user_id="student",
        item_id=item_id,
        skill_ids=["s"],
        timestamp=ts,
        correct=correct,
        action_sequence_id="assign",
        latency_ms=latency_ms,
        help_requested=help_flag,
    )


class FeatureEncodingTest(unittest.TestCase):
    def test_compute_feature_stats_generates_edges(self):
        rng = np.random.default_rng(0).choice([1000, 2000, 3000], size=50)
        df = np.zeros((50,), dtype=[("item_id", "U5"), ("latency_ms", "f4"), ("correct", "i4")])
        df["item_id"] = ["item"] * 50
        df["latency_ms"] = rng
        df["correct"] = 1
        stats = compute_feature_stats(pd.DataFrame(df))
        self.assertEqual(stats.latency_edges.shape[0], 11)
        self.assertEqual(stats.global_success_rate, 1.0)

    def test_encode_history_sequences_padding_and_features(self):
        reference = datetime.now(timezone.utc)
        events = [
            _event(5, "A", correct=1, help_flag=False, latency_ms=5000),
            _event(3, "B", correct=0, help_flag=True, latency_ms=8000),
        ]

        config = FeatureConfig(seq_len=4, bert_dim=2)
        stats = FeatureStats(
            latency_edges=np.array([0.0, 1000.0, 10000.0, np.inf]),
            item_success_rates={"A": 0.7},
            global_success_rate=0.5,
        )
        embeddings = {
            "A": np.ones(2, dtype=np.float32),
            "B": np.full(2, 2.0, dtype=np.float32),
        }

        features = encode_history_sequences(events, reference, config, stats, embeddings)

        self.assertEqual(features["history_actions"].tolist(), [0, 0, 1, 4])
        self.assertEqual(features["history_missing"].tolist(), [1, 1, 0, 0])
        self.assertTrue(np.isclose(features["history_item_success_rates"][2], 0.7))
        self.assertTrue(np.allclose(features["history_item_embeddings"][2], np.ones(2)))
        self.assertEqual(features["history_latency_bucket"][3], 1)


if __name__ == "__main__":
    unittest.main()
