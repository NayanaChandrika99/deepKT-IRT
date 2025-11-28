# ABOUTME: Provides a CLI that narrates learning traces by joining mastery and item health data.
# ABOUTME: Describes which artifacts will be consumed so contributors can keep interfaces stable.

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.common.explainability import generate_explanation, format_explanation
from src.common.gaming_detection import analyze_student, generate_gaming_report
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


@app.command()
def explain(
    user_id: str = typer.Option(..., "--user-id", help="Student identifier to explain."),
    skill: str = typer.Option(None, "--skill", help="Skill to explain; defaults to user's weakest skill."),
    reports_dir: Path = typer.Option(_default_reports_dir(), "--reports-dir", help="Directory containing model outputs."),
    events_path: Path = typer.Option(Path("data/interim/edm_cup_2023_42_events.parquet"), "--events-path", help="Canonical events parquet."),
    attention_path: Path = typer.Option(Path("reports/sakt_attention.parquet"), "--attention-path", help="Parquet containing attention weights (optional)."),
) -> None:
    """
    Explain why a student's mastery sits where it does using attention patterns.
    """
    events_df = pd.read_parquet(events_path)
    student_state_path = reports_dir / "sakt_student_state.parquet"
    if not student_state_path.exists():
        console.print(f"[red]Missing student state at {student_state_path}[/red]")
        raise typer.Exit(code=1)
    mastery_df = pd.read_parquet(student_state_path)

    skill_mastery_path = reports_dir / "skill_mastery.parquet"
    if skill_mastery_path.exists():
        skill_mastery_df = pd.read_parquet(skill_mastery_path)
    else:
        skill_mastery_df = aggregate_skill_mastery(mastery_df, events_df)
        skill_mastery_path.parent.mkdir(parents=True, exist_ok=True)
        skill_mastery_df.to_parquet(skill_mastery_path, index=False)

    user_mastery = skill_mastery_df[skill_mastery_df["user_id"] == user_id]
    if user_mastery.empty:
        console.print(f"[yellow]No mastery data for {user_id}[/yellow]")
        raise typer.Exit(code=1)

    if skill is None:
        weakest = user_mastery.sort_values("mastery_mean").head(1)
        skill = weakest.iloc[0]["skill"]
    mastery_score = float(user_mastery[user_mastery["skill"] == skill]["mastery_mean"].iloc[0])
    interaction_count = int(user_mastery[user_mastery["skill"] == skill]["interaction_count"].iloc[0])

    attention_df = pd.read_parquet(attention_path) if attention_path.exists() else pd.DataFrame(
        {"user_id": [], "top_influences": []}
    )

    explanation = generate_explanation(
        user_id=user_id,
        skill_id=skill,
        mastery_score=mastery_score,
        attention_data=attention_df,
        events_df=events_df,
        interaction_count=interaction_count,
    )

    console.print(format_explanation(explanation))


@app.command("gaming-check")
def gaming_check(
    user_id: str = typer.Option(None, "--user-id", help="Student identifier to check; leave empty to scan all."),
    events_path: Path = typer.Option(Path("data/interim/edm_cup_2023_42_events.parquet"), "--events-path", help="Canonical events parquet."),
    output: Path = typer.Option(Path("reports/gaming_alerts.parquet"), "--output", help="Output parquet for alerts when scanning all."),
    severity: str = typer.Option(None, "--severity", help="Optional severity filter for reports."),
) -> None:
    """
    Detect rapid guessing, help abuse, and suspicious patterns.
    """
    events_df = pd.read_parquet(events_path)

    if user_id:
        alerts = analyze_student(events_df, user_id)
        if not alerts:
            console.print(f"[green]✅ No gaming alerts for {user_id}[/green]")
            return
        for alert in alerts:
            color = {"low": "yellow", "medium": "orange3", "high": "red"}.get(alert.severity, "white")
            console.print(f"[{color}]{alert.alert_type} ({alert.severity})[/{color}]")
            for k, v in alert.evidence.items():
                console.print(f"  {k}: {v}")
            console.print(f"  → {alert.recommendation}")
        return

    alerts_df = generate_gaming_report(events_df)
    if severity:
        alerts_df = alerts_df[alerts_df["severity"] == severity]
    alerts_df.to_parquet(output, index=False)
    console.print(f"[bold]Analyzed {events_df['user_id'].nunique():,} students; alerts saved to {output}[/bold]")


if __name__ == "__main__":
    app()
