# ABOUTME: Proof-of-concept for gaming/cheating detection.
# ABOUTME: Demonstrates detection of rapid guessing and help abuse patterns.

"""
Proof-of-Concept: Gaming/Cheating Detection

This script demonstrates gaming detection using existing clickstream data:
- latency_ms: Response time in milliseconds
- help_requested: Whether student requested help
- correct: Response correctness

Usage:
    python scripts/poc_gaming_detection.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# Detection thresholds
RAPID_RESPONSE_THRESHOLD_MS = 5000   # < 5 seconds = rapid
HELP_ABUSE_THRESHOLD = 0.30          # > 30% help requests = concerning
SUSPICIOUS_STREAK_LENGTH = 5         # 5+ rapid incorrect = suspicious


@dataclass
class GamingAlert:
    """Alert for potential gaming behavior."""
    user_id: str
    alert_type: str
    severity: str
    evidence: Dict
    recommendation: str


def detect_rapid_guessing(events_df: pd.DataFrame, user_id: str) -> Optional[GamingAlert]:
    """Detect rapid guessing behavior (< 5 second responses)."""
    user_events = events_df[events_df["user_id"] == user_id]
    
    if len(user_events) < 10:
        return None  # Not enough data
    
    rapid_responses = user_events[user_events["latency_ms"] < RAPID_RESPONSE_THRESHOLD_MS]
    rapid_ratio = len(rapid_responses) / len(user_events)
    
    if rapid_ratio < 0.15:
        return None  # Normal behavior
    
    # Check if rapid responses are mostly incorrect
    rapid_incorrect = rapid_responses[rapid_responses["correct"] == 0]
    rapid_incorrect_ratio = len(rapid_incorrect) / max(len(rapid_responses), 1)
    
    # Determine severity
    if rapid_ratio >= 0.40 and rapid_incorrect_ratio >= 0.70:
        severity = "high"
        recommendation = "URGENT: Student shows strong gaming patterns. Immediate educator review recommended."
    elif rapid_ratio >= 0.25:
        severity = "medium"
        recommendation = "Student may be rushing. Consider reaching out to discuss study habits."
    else:
        severity = "low"
        recommendation = "Minor rapid response pattern. Continue monitoring."
    
    return GamingAlert(
        user_id=user_id,
        alert_type="rapid_guessing",
        severity=severity,
        evidence={
            "rapid_responses": len(rapid_responses),
            "total_responses": len(user_events),
            "rapid_ratio": round(rapid_ratio * 100, 1),
            "rapid_incorrect_ratio": round(rapid_incorrect_ratio * 100, 1),
            "avg_rapid_time_ms": int(rapid_responses["latency_ms"].mean()) if len(rapid_responses) > 0 else 0,
        },
        recommendation=recommendation
    )


def detect_help_abuse(events_df: pd.DataFrame, user_id: str) -> Optional[GamingAlert]:
    """Detect excessive help requests."""
    user_events = events_df[events_df["user_id"] == user_id]
    
    if "help_requested" not in user_events.columns:
        return None
    
    if len(user_events) < 10:
        return None
    
    help_requests = user_events[user_events["help_requested"] == True]
    help_ratio = len(help_requests) / len(user_events)
    
    if help_ratio < HELP_ABUSE_THRESHOLD:
        return None
    
    severity = "high" if help_ratio >= 0.50 else "medium"
    
    return GamingAlert(
        user_id=user_id,
        alert_type="help_abuse",
        severity=severity,
        evidence={
            "help_requests": len(help_requests),
            "total_responses": len(user_events),
            "help_ratio": round(help_ratio * 100, 1),
        },
        recommendation="Student frequently requests help. May need additional scaffolding or concept review."
    )


def detect_suspicious_patterns(events_df: pd.DataFrame, user_id: str) -> Optional[GamingAlert]:
    """Detect suspicious correctness patterns (rapid wrong streak → sudden correct)."""
    user_events = events_df[events_df["user_id"] == user_id].sort_values("timestamp")
    
    if len(user_events) < 15:
        return None
    
    suspicious_count = 0
    rapid_incorrect_streak = 0
    
    for _, row in user_events.iterrows():
        is_rapid = row["latency_ms"] < RAPID_RESPONSE_THRESHOLD_MS
        is_incorrect = row["correct"] == 0
        
        if is_rapid and is_incorrect:
            rapid_incorrect_streak += 1
        elif rapid_incorrect_streak >= SUSPICIOUS_STREAK_LENGTH and row["correct"] == 1:
            # Streak of rapid incorrect followed by correct = suspicious
            suspicious_count += 1
            rapid_incorrect_streak = 0
        else:
            rapid_incorrect_streak = 0
    
    if suspicious_count < 2:
        return None
    
    severity = "high" if suspicious_count >= 5 else "medium"
    
    return GamingAlert(
        user_id=user_id,
        alert_type="suspicious_pattern",
        severity=severity,
        evidence={
            "suspicious_sequences": suspicious_count,
            "pattern": "rapid_incorrect_streak_then_correct",
        },
        recommendation="Pattern suggests answer-seeking behavior. Review session recordings if available."
    )


def analyze_student(events_df: pd.DataFrame, user_id: str) -> List[GamingAlert]:
    """Run all detectors for a student."""
    alerts = []
    
    for detector in [detect_rapid_guessing, detect_help_abuse, detect_suspicious_patterns]:
        alert = detector(events_df, user_id)
        if alert:
            alerts.append(alert)
    
    return alerts


def display_alert(alert: GamingAlert):
    """Display a single alert."""
    severity_colors = {
        "low": "yellow",
        "medium": "orange3",
        "high": "red"
    }
    color = severity_colors.get(alert.severity, "white")
    
    console.print(Panel(
        f"[bold]Alert Type:[/bold] {alert.alert_type}\n"
        f"[bold]Severity:[/bold] [{color}]{alert.severity.upper()}[/{color}]\n\n"
        f"[bold]Evidence:[/bold]\n" + 
        "\n".join(f"  • {k}: {v}" for k, v in alert.evidence.items()) +
        f"\n\n[bold]Recommendation:[/bold]\n  {alert.recommendation}",
        title=f"⚠️ Gaming Alert: {alert.user_id}",
        border_style=color
    ))


def main():
    """Run POC demonstration."""
    
    console.rule("[bold blue]Proof-of-Concept: Gaming/Cheating Detection[/bold blue]")
    console.print()
    
    events_path = Path("data/interim/edm_cup_2023_42_events.parquet")
    
    if not events_path.exists():
        console.print("[yellow]Warning: Events data not found locally.[/yellow]")
        console.print("Creating synthetic data for demonstration.\n")
        
        # Create synthetic data with different student types
        np.random.seed(42)
        
        # Normal student
        normal_events = pd.DataFrame({
            "user_id": ["NORMAL_STUDENT"] * 30,
            "item_id": [f"Q{i:03d}" for i in range(30)],
            "correct": np.random.choice([0, 1], 30, p=[0.3, 0.7]),
            "latency_ms": np.random.randint(15000, 90000, 30),  # 15-90 seconds
            "help_requested": np.random.choice([True, False], 30, p=[0.1, 0.9]),
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="h"),
        })
        
        # Rapid guesser
        rapid_events = pd.DataFrame({
            "user_id": ["RAPID_GUESSER"] * 30,
            "item_id": [f"Q{i:03d}" for i in range(30)],
            "correct": [0] * 25 + [1] * 5,  # Mostly wrong
            "latency_ms": [2500] * 25 + [30000] * 5,  # 83% rapid
            "help_requested": [False] * 30,
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="h"),
        })
        
        # Help abuser
        help_events = pd.DataFrame({
            "user_id": ["HELP_ABUSER"] * 30,
            "item_id": [f"Q{i:03d}" for i in range(30)],
            "correct": np.random.choice([0, 1], 30),
            "latency_ms": np.random.randint(20000, 60000, 30),
            "help_requested": [True] * 18 + [False] * 12,  # 60% help
            "timestamp": pd.date_range("2024-01-01", periods=30, freq="h"),
        })
        
        events_df = pd.concat([normal_events, rapid_events, help_events], ignore_index=True)
    else:
        events_df = pd.read_parquet(events_path)
    
    console.print(f"[dim]Loaded {len(events_df):,} events from {events_df['user_id'].nunique():,} students[/dim]\n")
    
    # Analyze sample students
    if "NORMAL_STUDENT" in events_df["user_id"].values:
        sample_students = ["NORMAL_STUDENT", "RAPID_GUESSER", "HELP_ABUSER"]
    else:
        sample_students = events_df["user_id"].unique()[:5]
    
    all_alerts = []
    
    for user_id in sample_students:
        alerts = analyze_student(events_df, user_id)
        all_alerts.extend(alerts)
        
        if alerts:
            for alert in alerts:
                display_alert(alert)
        else:
            console.print(f"[green]✅ {user_id}: No gaming alerts[/green]\n")
    
    # Summary statistics
    console.rule("[bold]Summary[/bold]")
    
    if all_alerts:
        summary_table = Table(title="Alert Summary")
        summary_table.add_column("Alert Type")
        summary_table.add_column("Count", justify="right")
        summary_table.add_column("High Severity", justify="right", style="red")
        
        alert_types = {}
        for alert in all_alerts:
            if alert.alert_type not in alert_types:
                alert_types[alert.alert_type] = {"count": 0, "high": 0}
            alert_types[alert.alert_type]["count"] += 1
            if alert.severity == "high":
                alert_types[alert.alert_type]["high"] += 1
        
        for alert_type, counts in alert_types.items():
            summary_table.add_row(
                alert_type,
                str(counts["count"]),
                str(counts["high"])
            )
        
        console.print(summary_table)
    else:
        console.print("[green]No gaming alerts detected in sample.[/green]")
    
    # Show what full implementation would add
    console.print("\n" + "=" * 60)
    console.print("[bold]Full Implementation Would Add:[/bold]")
    console.print("  • Batch processing for all students")
    console.print("  • Export alerts to parquet for dashboards")
    console.print("  • Configurable thresholds via config file")
    console.print("  • Time-series analysis for pattern evolution")
    console.print("  • Integration with educator notification system")
    console.print("=" * 60)


if __name__ == "__main__":
    main()

