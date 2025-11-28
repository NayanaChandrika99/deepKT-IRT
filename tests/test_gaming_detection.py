# ABOUTME: Tests gaming detection heuristics for rapid guessing and help abuse.
# ABOUTME: Uses synthetic clickstream data to trigger alerts.

import pandas as pd
import numpy as np

from src.common.gaming_detection import (
    detect_rapid_guessing,
    detect_help_abuse,
    analyze_student,
    GamingAlert,
)


def _mk_events(user_id: str, count: int, latency_ms: int, correct: int, help_flag: bool = False):
    return pd.DataFrame(
        {
            "user_id": [user_id] * count,
            "correct": [correct] * count,
            "latency_ms": [latency_ms] * count,
            "help_requested": [help_flag] * count,
            "timestamp": pd.date_range("2024-01-01", periods=count, freq="h"),
        }
    )


def test_detect_rapid_guessing_flags_high_ratio():
    rapid = _mk_events("gamer", 15, 2000, correct=0)
    slow = _mk_events("gamer", 5, 30000, correct=1)
    df = pd.concat([rapid, slow])

    alert = detect_rapid_guessing(df, "gamer")
    assert isinstance(alert, GamingAlert)
    assert alert.alert_type == "rapid_guessing"


def test_detect_help_abuse_flags_high_ratio():
    normal = _mk_events("helper", 8, 20000, correct=1, help_flag=False)
    helpers = _mk_events("helper", 12, 15000, correct=0, help_flag=True)
    df = pd.concat([normal, helpers])

    alert = detect_help_abuse(df, "helper", threshold=0.3)
    assert isinstance(alert, GamingAlert)
    assert alert.alert_type == "help_abuse"


def test_analyze_student_combines_detectors():
    rapid = _mk_events("combo", 12, 2000, correct=0)
    slow = _mk_events("combo", 8, 25000, correct=1)
    df = pd.concat([rapid, slow])

    alerts = analyze_student(df, "combo")
    assert any(a.alert_type == "rapid_guessing" for a in alerts)
