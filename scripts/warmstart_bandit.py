# ABOUTME: Warm-starts the LinUCB bandit using historical student events.
# ABOUTME: Simulates rewards from actual correctness data to pre-train the model.

"""
Warm-Start Bandit Script

Uses historical events to train the LinUCB bandit before deployment.
This gives the bandit initial knowledge about which items work for which students.

Usage:
    python scripts/warmstart_bandit.py
    python scripts/warmstart_bandit.py --events-path data/interim/edm_cup_2023_42_events.parquet
"""

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.common.bandit import (
    LinUCBBandit,
    build_student_context,
    ItemArm,
)

console = Console()
app = typer.Typer(help="Warm-start the LinUCB bandit from historical data.")


@app.command()
def warmstart(
    events_path: Path = typer.Option(
        Path("data/interim/edm_cup_2023_42_events.parquet"),
        "--events-path",
        help="Path to canonical events parquet.",
    ),
    item_params_path: Path = typer.Option(
        Path("reports/item_params.parquet"),
        "--item-params",
        help="Path to item parameters parquet.",
    ),
    output_path: Path = typer.Option(
        Path("reports/bandit_state.npz"),
        "--output",
        help="Path to save bandit state.",
    ),
    max_events: int = typer.Option(
        10000,
        "--max-events",
        help="Maximum events to use for training (for speed).",
    ),
    alpha: float = typer.Option(
        1.0,
        "--alpha",
        help="Exploration parameter for LinUCB.",
    ),
) -> None:
    """
    Train LinUCB bandit on historical events data.
    """
    console.rule("[bold blue]Warm-Starting LinUCB Bandit[/bold blue]")

    if not events_path.exists():
        console.print(f"[red]Events file not found: {events_path}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Loading events from {events_path}...[/dim]")
    events_df = pd.read_parquet(events_path)
    console.print(f"  Total events: {len(events_df):,}")
    console.print(f"  Unique users: {events_df['user_id'].nunique():,}")

    if item_params_path.exists():
        item_params_df = pd.read_parquet(item_params_path)
        item_difficulty = dict(zip(item_params_df["item_id"], item_params_df["difficulty"]))
        console.print(f"  Items with difficulty: {len(item_difficulty):,}")
    else:
        console.print("[yellow]Item params not found. Using default difficulty.[/yellow]")
        item_difficulty = {}

    bandit = LinUCBBandit(n_features=8, alpha=alpha)

    sample_events = events_df.sample(n=min(max_events, len(events_df)), random_state=42)
    console.print(f"\n[dim]Training on {len(sample_events):,} events...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training...", total=len(sample_events))

        for _, row in sample_events.iterrows():
            user_id = str(row["user_id"])
            item_id = str(row["item_id"])
            correct = int(row["correct"])

            user_events = events_df[events_df["user_id"] == user_id]
            student = build_student_context(user_id, user_events)

            difficulty = item_difficulty.get(item_id, 0.5)
            item = ItemArm(
                item_id=item_id,
                skill=str(row.get("skill_ids", ["unknown"])[0]) if row.get("skill_ids") else "unknown",
                difficulty=difficulty,
            )

            bandit.update(student, item, reward=float(correct))
            progress.advance(task)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bandit.save(output_path)

    console.print()
    console.print(f"[green]✅ Bandit trained and saved to {output_path}[/green]")
    console.print(f"   Updates: {bandit.n_updates:,}")
    console.print(f"   Alpha: {bandit.alpha}")
    console.print(f"   Features: {bandit.n_features}")

    console.print()
    console.print("[dim]Learned weights (feature importance):[/dim]")
    feature_names = [
        "mastery",
        "recent_accuracy",
        "recent_speed",
        "help_tendency",
        "skill_gap",
        "difficulty",
        "challenge_match",
        "difficulty_gap",
    ]
    for name, weight in zip(feature_names, bandit.theta):
        bar = "█" * int(abs(weight) * 10)
        sign = "+" if weight >= 0 else "-"
        console.print(f"   {name:18} {sign}{abs(weight):.3f} {bar}")


if __name__ == "__main__":
    app()

