# ABOUTME: Validates shared feature and evaluation helpers.
# ABOUTME: Ensures clickstream aggregates and metrics computation work as expected.

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common.features import build_clickstream_features
from src.common.schemas import LearningEvent
from src.common.evaluation import evaluate_predictions


def _event(ts_hours_ago: float, user: str, item: str, correct: int, latency_ms=None, help_requested=None):
    return LearningEvent(
        user_id=user,
        item_id=item,
        skill_ids=["k"],
        timestamp=datetime.now(timezone.utc) - timedelta(hours=ts_hours_ago),
        correct=correct,
        action_sequence_id="seq",
        latency_ms=latency_ms,
        help_requested=help_requested,
    )


def test_build_clickstream_features_aggregates_per_user():
    events = [
        _event(3, "u1", "i1", 1, latency_ms=1000, help_requested=False),
        _event(2, "u1", "i2", 0, latency_ms=2000, help_requested=True),
        _event(1, "u2", "i3", 1, latency_ms=None, help_requested=None),
    ]

    df = build_clickstream_features(events)

    assert set(df.columns) == {
        "user_id",
        "total_attempts",
        "correct_rate",
        "avg_latency_ms",
        "median_latency_ms",
        "help_rate",
        "attempts_last_24h",
        "first_ts",
        "last_ts",
    }
    u1 = df.set_index("user_id").loc["u1"]
    assert u1["total_attempts"] == 2
    assert pytest.approx(u1["correct_rate"]) == 0.5
    assert pytest.approx(u1["avg_latency_ms"]) == 1500.0
    assert pytest.approx(u1["median_latency_ms"]) == 1500.0
    assert pytest.approx(u1["help_rate"]) == 0.5
    assert u1["attempts_last_24h"] >= 2
    assert u1["first_ts"] <= u1["last_ts"]


def test_evaluate_predictions_supports_common_metrics():
    preds = pd.DataFrame(
        {
            "y_true": [0, 1, 1, 0],
            "y_pred": [0.05, 0.92, 0.8, 0.4],
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["i1", "i2", "i3", "i4"],
        }
    )

    expected_auc = roc_auc_score(preds["y_true"], preds["y_pred"])
    expected_ap = average_precision_score(preds["y_true"], preds["y_pred"])
    results = evaluate_predictions(preds, metrics=["auc", "average_precision", "calibration_ece"])

    assert pytest.approx(results["auc"]) == expected_auc
    assert pytest.approx(results["average_precision"]) == expected_ap
    assert 0 <= results["calibration_ece"] <= 1


def test_evaluate_predictions_rejects_unknown_metric():
    preds = pd.DataFrame({"y_true": [0, 1], "y_pred": [0.2, 0.8]})
    with pytest.raises(ValueError):
        evaluate_predictions(preds, metrics=["auc", "unknown_metric"])
