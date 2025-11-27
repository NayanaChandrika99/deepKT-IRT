# ABOUTME: Proof-of-concept for explainable knowledge tracing.
# ABOUTME: Demonstrates attention-based explanations without full SAKT integration.

"""
Proof-of-Concept: Explainable Knowledge Tracing

This script demonstrates how attention-based explanations would work
using existing SAKT predictions and simulated attention weights.

Full implementation would extract actual attention weights from the SAKT model.
This POC shows the output format and explanation generation logic.

Usage:
    python scripts/poc_explainability.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class MasteryExplanation:
    """Human-readable explanation for a mastery prediction."""
    user_id: str
    skill_id: str
    mastery_score: float
    key_factors: List[Dict]
    summary: str


def simulate_attention_weights(
    events_df: pd.DataFrame, 
    user_id: str, 
    n_interactions: int = 5
) -> pd.DataFrame:
    """
    Simulate attention weights for demonstration.
    
    In production, these would be extracted from SAKT's transformer attention.
    Here we use a simple heuristic: recent interactions get higher weight.
    """
    user_events = events_df[events_df["user_id"] == user_id].tail(20)
    
    if len(user_events) < n_interactions:
        return pd.DataFrame()
    
    # Simulate attention: exponential decay by recency
    n = len(user_events)
    raw_weights = np.exp(-0.1 * np.arange(n)[::-1])  # Recent = higher
    normalized_weights = raw_weights / raw_weights.sum()
    
    # Get top-N attended interactions
    top_indices = np.argsort(normalized_weights)[-n_interactions:][::-1]
    
    attention_data = []
    for idx in top_indices:
        row = user_events.iloc[idx]
        attention_data.append({
            "user_id": user_id,
            "item_id": row["item_id"],
            "attention_weight": normalized_weights[idx],
            "correct": row["correct"],
            "skill": row["skill_ids"][0] if isinstance(row["skill_ids"], list) and row["skill_ids"] else "unknown",
            "latency_ms": row["latency_ms"],
        })
    
    return pd.DataFrame(attention_data)


def generate_explanation(
    user_id: str,
    skill_id: str,
    mastery_score: float,
    attention_df: pd.DataFrame,
) -> MasteryExplanation:
    """Generate explanation from attention weights."""
    
    key_factors = []
    correct_count = 0
    incorrect_count = 0
    
    for _, row in attention_df.iterrows():
        outcome = "correct" if row["correct"] == 1 else "incorrect"
        if outcome == "correct":
            correct_count += 1
        else:
            incorrect_count += 1
        
        key_factors.append({
            "item_id": row["item_id"],
            "weight": round(row["attention_weight"], 3),
            "outcome": outcome,
            "skill": row["skill"],
            "latency_ms": int(row["latency_ms"]),
        })
    
    # Generate summary based on mastery level
    if mastery_score >= 0.7:
        summary = f"High mastery ({mastery_score:.2f}): "
        summary += f"Strong performance on {correct_count}/{len(key_factors)} key interactions. "
        summary += "Model confident in skill acquisition."
    elif mastery_score >= 0.4:
        summary = f"Moderate mastery ({mastery_score:.2f}): "
        summary += f"Mixed results - {correct_count} correct, {incorrect_count} incorrect. "
        summary += "Consider reviewing foundational concepts."
    else:
        summary = f"Low mastery ({mastery_score:.2f}): "
        summary += f"Struggled with {incorrect_count}/{len(key_factors)} key interactions. "
        summary += "Recommend focused practice on prerequisite skills."
    
    return MasteryExplanation(
        user_id=user_id,
        skill_id=skill_id,
        mastery_score=mastery_score,
        key_factors=key_factors,
        summary=summary
    )


def display_explanation(explanation: MasteryExplanation):
    """Rich display of explanation."""
    
    # Header
    console.print(Panel(
        f"[bold]Student:[/bold] {explanation.user_id}\n"
        f"[bold]Skill:[/bold] {explanation.skill_id}\n"
        f"[bold]Mastery:[/bold] {explanation.mastery_score:.2f}",
        title="Mastery Explanation",
        border_style="blue"
    ))
    
    # Key factors table
    table = Table(title="Key Factors (Attention-Weighted)")
    table.add_column("#", style="dim")
    table.add_column("Item", style="cyan")
    table.add_column("Outcome")
    table.add_column("Attention", justify="right")
    table.add_column("Skill", style="magenta")
    table.add_column("Response Time", justify="right")
    
    for i, factor in enumerate(explanation.key_factors, 1):
        outcome_style = "green" if factor["outcome"] == "correct" else "red"
        outcome_emoji = "✅" if factor["outcome"] == "correct" else "❌"
        
        latency_sec = factor["latency_ms"] / 1000
        latency_str = f"{latency_sec:.1f}s"
        
        table.add_row(
            str(i),
            factor["item_id"][:12] + "..." if len(factor["item_id"]) > 12 else factor["item_id"],
            f"[{outcome_style}]{outcome_emoji} {factor['outcome']}[/{outcome_style}]",
            f"{factor['weight']:.3f}",
            factor["skill"][:15] + "..." if len(factor["skill"]) > 15 else factor["skill"],
            latency_str,
        )
    
    console.print(table)
    
    # Summary
    if explanation.mastery_score >= 0.7:
        summary_style = "green"
    elif explanation.mastery_score >= 0.4:
        summary_style = "yellow"
    else:
        summary_style = "red"
    
    console.print(f"\n[{summary_style}]{explanation.summary}[/{summary_style}]")


def main():
    """Run POC demonstration."""
    
    console.rule("[bold blue]Proof-of-Concept: Explainable Knowledge Tracing[/bold blue]")
    console.print()
    
    # Check for data
    events_path = Path("data/interim/edm_cup_2023_42_events.parquet")
    mastery_path = Path("reports/sakt_student_state.parquet")
    
    if not events_path.exists() or not mastery_path.exists():
        console.print("[yellow]Warning: Required data files not found locally.[/yellow]")
        console.print("Using synthetic data for demonstration.\n")
        
        # Create synthetic data
        np.random.seed(42)
        events_df = pd.DataFrame({
            "user_id": ["DEMO_STUDENT"] * 20,
            "item_id": [f"Q{i:03d}" for i in range(20)],
            "skill_ids": [["7.RP.A.1"]] * 10 + [["7.RP.A.2"]] * 10,
            "correct": [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
            "latency_ms": np.random.randint(5000, 90000, 20),
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
        })
        
        sample_user = "DEMO_STUDENT"
        sample_skill = "7.RP.A.1"
        sample_mastery = 0.65
    else:
        events_df = pd.read_parquet(events_path)
        mastery_df = pd.read_parquet(mastery_path)
        
        # Pick a sample student
        sample_user = events_df["user_id"].iloc[0]
        
        # Get their mastery
        user_mastery = mastery_df[mastery_df["user_id"] == sample_user]
        if not user_mastery.empty:
            sample_skill = user_mastery.iloc[0].get("skill_id", "unknown")
            sample_mastery = user_mastery.iloc[0]["mastery"]
        else:
            sample_skill = "unknown"
            sample_mastery = 0.5
    
    console.print(f"[dim]Demonstrating explanation for student: {sample_user}[/dim]\n")
    
    # Generate simulated attention weights
    attention_df = simulate_attention_weights(events_df, sample_user)
    
    if attention_df.empty:
        console.print("[red]Not enough interactions to generate explanation.[/red]")
        return
    
    # Generate and display explanation
    explanation = generate_explanation(
        user_id=sample_user,
        skill_id=sample_skill,
        mastery_score=sample_mastery,
        attention_df=attention_df,
    )
    
    display_explanation(explanation)
    
    # Show what full implementation would add
    console.print("\n" + "=" * 60)
    console.print("[bold]Full Implementation Would Add:[/bold]")
    console.print("  • Extract actual attention weights from SAKT transformer")
    console.print("  • Map attention to specific skills, not just items")
    console.print("  • Generate recommendations based on weak attention areas")
    console.print("  • Export attention data to parquet for analysis")
    console.print("=" * 60)


if __name__ == "__main__":
    main()

