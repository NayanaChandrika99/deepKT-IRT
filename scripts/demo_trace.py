# ABOUTME: Provides a CLI that narrates learning traces by joining mastery and item health data.
# ABOUTME: Describes which artifacts will be consumed so contributors can keep interfaces stable.

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

import pandas as pd

from src.common.mastery_aggregation import aggregate_skill_mastery
from src.common.recommendation import recommend_items

console = Console()
app = typer.Typer(help="Join student mastery (SAKT) with item health (Wide & Deep IRT).")


def _default_reports_dir() -> Path:
    return Path("reports")


@app.command()
def trace(
    student_id: str = typer.Option(..., "--student-id", help="Student identifier in the canonical schema."),
    topic: str = typer.Option(..., "--topic", help="Skill or curriculum node to focus on."),
    time_window: str = typer.Option(..., "--time-window", help="ISO8601 week or calendar window, e.g., 2023-W15."),
    wd_config: Path = typer.Option(Path("configs/wd_irt_edm.yaml"), "--wd-config", help="Wide & Deep IRT config path."),
    sakt_config: Path = typer.Option(Path("configs/sakt_assist2009.yaml"), "--sakt-config", help="SAKT config path."),
    reports_dir: Path = typer.Option(_default_reports_dir(), "--reports-dir", help="Directory containing model outputs."),
    events_path: Path = typer.Option(Path("data/interim/edm_cup_2023_42_events.parquet"), "--events-path", help="Canonical events parquet for skill_ids."),
    recommendation_count: int = typer.Option(5, "--recommendation-count", help="Number of items to surface."),
) -> None:
    """
    Emits a structured preview showing how outputs from both models will be interpreted once available.
    """
    console.rule("[bold blue]DeepKT + Wide&Deep IRT Trace[/bold blue]")
    console.print(f"[bold]Student:[/] {student_id}")
    console.print(f"[bold]Topic:[/] {topic}")
    console.print(f"[bold]Time Window:[/] {time_window}")
    console.print()

    student_state_path = reports_dir / "sakt_student_state.parquet"
    skill_mastery_path = reports_dir / "skill_mastery.parquet"
    item_params_path = reports_dir / "item_params.parquet"
    item_drift_path = reports_dir / "item_drift.parquet"

    if not skill_mastery_path.exists():
        if not student_state_path.exists():
            raise typer.Exit(f"Missing student mastery parquet at {student_state_path}")
        if not events_path.exists():
            raise typer.Exit(f"Missing events parquet at {events_path}")
        console.print("[yellow]Generating skill mastery from student state + events...[/yellow]")
        mastery_df = pd.read_parquet(student_state_path)
        events_df = pd.read_parquet(events_path)
        skill_mastery_df = aggregate_skill_mastery(mastery_df, events_df)
        skill_mastery_path.parent.mkdir(parents=True, exist_ok=True)
        skill_mastery_df.to_parquet(skill_mastery_path, index=False)
    else:
        skill_mastery_df = pd.read_parquet(skill_mastery_path)

    if not item_params_path.exists():
        raise typer.Exit(f"Missing item parameters parquet at {item_params_path}")
    item_params_df = pd.read_parquet(item_params_path)
    if item_drift_path.exists():
        drift_df = pd.read_parquet(item_drift_path)
        item_params_df = item_params_df.merge(drift_df, on="item_id", how="left")

    user_mastery = skill_mastery_df[
        (skill_mastery_df["user_id"] == student_id) & (skill_mastery_df["skill"] == topic)
    ]
    mastery_mean = float(user_mastery["mastery_mean"].iloc[0]) if not user_mastery.empty else 0.5
    interactions = int(user_mastery["interaction_count"].iloc[0]) if not user_mastery.empty else 0

    recs = recommend_items(
        user_id=student_id,
        target_skill=topic,
        skill_mastery=skill_mastery_df,
        item_params=item_params_df,
        max_items=recommendation_count,
        exclude_high_drift=True,
    )

    console.print()
    console.print("[bold green]Mastery[/bold green]")
    mastery_table = Table(show_header=True, header_style="bold magenta")
    mastery_table.add_column("Skill")
    mastery_table.add_column("Mastery")
    mastery_table.add_column("Interactions")
    mastery_table.add_row(topic, f"{mastery_mean:.2f}", str(interactions))
    console.print(mastery_table)

    console.print()
    console.print("[bold yellow]Recommendations[/bold yellow]")
    rec_table = Table(show_header=True, header_style="bold magenta")
    rec_table.add_column("Item ID")
    rec_table.add_column("Difficulty")
    rec_table.add_column("Reason")
    for rec in recs:
        rec_table.add_row(rec.item_id, f"{rec.difficulty:.2f}", rec.reason)
    console.print(rec_table)


if __name__ == "__main__":
    app()
