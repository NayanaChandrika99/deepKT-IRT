# ABOUTME: Provides a CLI that narrates learning traces by joining mastery and item health data.
# ABOUTME: Describes which artifacts will be consumed so contributors can keep interfaces stable.

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

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

    console.print("[bold green]Inputs[/bold green]")
    inputs = Table(show_header=True, header_style="bold magenta")
    inputs.add_column("Artifact")
    inputs.add_column("Path / Description")
    inputs.add_row("SAKT config", str(sakt_config))
    inputs.add_row("WD-IRT config", str(wd_config))
    inputs.add_row("Student mastery", str(reports_dir / "student_state.parquet"))
    inputs.add_row("Next-step predictions", str(reports_dir / "next_correct_predictions.parquet"))
    inputs.add_row("Item parameters", str(reports_dir / "item_params.parquet"))
    inputs.add_row("Item drift flags", str(reports_dir / "item_drift.parquet"))
    console.print(inputs)

    console.print()
    console.print("[bold yellow]Planned Output Structure[/bold yellow]")
    console.print(
        "- Weak topics sourced from mastery vectors (threshold configurable via configs).\n"
        "- Recommended remediation items selected from WD-IRT parameters filtered by topic.\n"
        "- Behavior insights summarizing clickstream segments for the requested window."
    )

    console.print()
    console.print("[bold cyan]Sample Narrative[/bold cyan]")
    console.print(
        f"Student {student_id} shows latent mastery below threshold on {topic}. "
        f"Recommend reviewing high-quality items with low drift risk inside {time_window}. "
        "Actual values will populate after both training pipelines export their parquet artifacts."
    )

    console.print()
    console.print(f"[bold]Requested {recommendation_count} items; CLI will emit exact IDs once exports exist.[/bold]")


if __name__ == "__main__":
    app()
