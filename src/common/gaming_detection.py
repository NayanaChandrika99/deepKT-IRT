# ABOUTME: Detects gaming or rushing behavior from clickstream signals.
# ABOUTME: Provides heuristics for rapid guessing, help abuse, and suspicious patterns.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class GamingAlert:
    user_id: str
    alert_type: str
    severity: str
    evidence: Dict
    recommendation: str


class GamingThresholds:
    RAPID_RESPONSE_MS = 5000
    HELP_ABUSE_RATIO = 0.30
    RAPID_INCORRECT_STREAK = 5
    MIN_INTERACTIONS = 10


def detect_rapid_guessing(
    events_df: pd.DataFrame,
    user_id: str,
    threshold_ms: int = GamingThresholds.RAPID_RESPONSE_MS,
) -> Optional[GamingAlert]:
    user_events = events_df[events_df["user_id"] == user_id]
    if len(user_events) < GamingThresholds.MIN_INTERACTIONS:
        return None

    rapid = user_events[user_events["latency_ms"] < threshold_ms]
    rapid_ratio = len(rapid) / len(user_events)
    if rapid_ratio < 0.15:
        return None

    rapid_incorrect = rapid[rapid["correct"] == 0]
    rapid_incorrect_ratio = (len(rapid_incorrect) / len(rapid)) if len(rapid) else 0.0

    if rapid_ratio >= 0.40 and rapid_incorrect_ratio >= 0.70:
        severity = "high"
        rec = "Strong rapid-guessing pattern. Encourage deliberate practice and review."
    elif rapid_ratio >= 0.25:
        severity = "medium"
        rec = "Student is rushing. Suggest slowing down and revisiting mistakes."
    else:
        severity = "low"
        rec = "Minor rapid attempts observed. Continue monitoring."

    return GamingAlert(
        user_id=user_id,
        alert_type="rapid_guessing",
        severity=severity,
        evidence={
            "rapid_responses": int(len(rapid)),
            "total_responses": int(len(user_events)),
            "rapid_ratio_pct": round(rapid_ratio * 100, 1),
            "rapid_incorrect_pct": round(rapid_incorrect_ratio * 100, 1),
        },
        recommendation=rec,
    )


def detect_help_abuse(
    events_df: pd.DataFrame,
    user_id: str,
    threshold: float = GamingThresholds.HELP_ABUSE_RATIO,
) -> Optional[GamingAlert]:
    user_events = events_df[events_df["user_id"] == user_id]
    if "help_requested" not in user_events.columns or len(user_events) < GamingThresholds.MIN_INTERACTIONS:
        return None

    help_count = user_events["help_requested"].sum()
    help_ratio = help_count / len(user_events)
    if help_ratio < threshold:
        return None

    severity = "high" if help_ratio >= 0.50 else "medium"
    rec = (
        "Frequent help requests. Consider scaffolding or guided practice."
        if severity == "medium"
        else "Heavy reliance on help. Educator review recommended."
    )

    return GamingAlert(
        user_id=user_id,
        alert_type="help_abuse",
        severity=severity,
        evidence={
            "help_requests": int(help_count),
            "total_responses": int(len(user_events)),
            "help_ratio_pct": round(help_ratio * 100, 1),
        },
        recommendation=rec,
    )


def detect_suspicious_patterns(events_df: pd.DataFrame, user_id: str) -> Optional[GamingAlert]:
    user_events = events_df[events_df["user_id"] == user_id].sort_values("timestamp")
    if len(user_events) < 15:
        return None

    suspicious_count = 0
    streak = 0
    for _, row in user_events.iterrows():
        is_rapid = row["latency_ms"] < GamingThresholds.RAPID_RESPONSE_MS
        is_incorrect = row["correct"] == 0
        if is_rapid and is_incorrect:
            streak += 1
        elif streak >= GamingThresholds.RAPID_INCORRECT_STREAK and row["correct"] == 1:
            suspicious_count += 1
            streak = 0
        else:
            streak = 0

    if suspicious_count < 2:
        return None

    severity = "high" if suspicious_count >= 5 else "medium"
    return GamingAlert(
        user_id=user_id,
        alert_type="suspicious_pattern",
        severity=severity,
        evidence={
            "suspicious_sequences": suspicious_count,
            "pattern": "rapid_incorrect_then_correct",
        },
        recommendation="Possible answer-seeking. Consider intervention or review logs.",
    )


def analyze_student(events_df: pd.DataFrame, user_id: str) -> List[GamingAlert]:
    alerts: List[GamingAlert] = []
    for detector in (detect_rapid_guessing, detect_help_abuse, detect_suspicious_patterns):
        alert = detector(events_df, user_id)
        if alert:
            alerts.append(alert)
    return alerts


def generate_gaming_report(events_df: pd.DataFrame) -> pd.DataFrame:
    alerts: List[Dict] = []
    for user_id in events_df["user_id"].unique():
        for alert in analyze_student(events_df, user_id):
            alerts.append(
                {
                    "user_id": alert.user_id,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "evidence": alert.evidence,
                    "recommendation": alert.recommendation,
                }
            )
    return pd.DataFrame(alerts)
