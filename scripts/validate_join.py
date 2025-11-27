# ABOUTME: Validates that SAKT and WD-IRT outputs can be meaningfully joined.
# ABOUTME: Reports join coverage and identifies any schema mismatches.

import json
from pathlib import Path
from typing import Dict, Set

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Validate joinability between SAKT and WD-IRT outputs.")


@app.command()
def validate(
    reports_dir: Path = typer.Option(Path("reports"), "--reports-dir", help="Directory containing model outputs."),
    sample_students: int = typer.Option(5, "--sample-students", help="Number of students to sample for detailed output."),
) -> None:
    """
    Validates that SAKT and WD-IRT outputs can be joined.
    
    Reports:
    - Item ID overlap between models
    - User ID overlap
    - Join coverage statistics
    - Sample joined records
    """
    console.rule("[bold blue]Joinability Validation[/bold blue]")
    
    # Load outputs
    sakt_preds_path = reports_dir / "sakt_predictions.parquet"
    sakt_state_path = reports_dir / "sakt_student_state.parquet"
    wdirt_params_path = reports_dir / "item_params.parquet"
    wdirt_drift_path = reports_dir / "item_drift.parquet"
    
    if not sakt_preds_path.exists():
        console.print(f"[red]Error:[/] {sakt_preds_path} not found")
        raise typer.Exit(1)
    if not wdirt_params_path.exists():
        console.print(f"[red]Error:[/] {wdirt_params_path} not found")
        raise typer.Exit(1)
    
    console.print("[green]Loading outputs...[/green]")
    sakt_preds = pd.read_parquet(sakt_preds_path)
    sakt_state = pd.read_parquet(sakt_state_path) if sakt_state_path.exists() else None
    wdirt_params = pd.read_parquet(wdirt_params_path)
    wdirt_drift = pd.read_parquet(wdirt_drift_path) if wdirt_drift_path.exists() else None
    
    # Compute overlaps
    sakt_items: Set[str] = set(sakt_preds["item_id"].unique())
    wdirt_items: Set[str] = set(wdirt_params["item_id"].unique())
    overlap_items = sakt_items & wdirt_items
    
    sakt_users: Set[str] = set(sakt_preds["user_id"].unique())
    
    # Filter SAKT to joinable items
    sakt_joinable = sakt_preds[sakt_preds["item_id"].isin(overlap_items)]
    
    # Summary table
    table = Table(title="Joinability Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    
    table.add_row("SAKT unique items", f"{len(sakt_items):,}")
    table.add_row("WD-IRT unique items", f"{len(wdirt_items):,}")
    table.add_row("Overlapping items", f"{len(overlap_items):,}")
    table.add_row("Item coverage", f"{len(overlap_items) / len(sakt_items) * 100:.2f}%")
    table.add_row("", "")
    table.add_row("SAKT total predictions", f"{len(sakt_preds):,}")
    table.add_row("SAKT joinable predictions", f"{len(sakt_joinable):,}")
    table.add_row("Prediction coverage", f"{len(sakt_joinable) / len(sakt_preds) * 100:.2f}%")
    table.add_row("", "")
    table.add_row("SAKT unique users", f"{len(sakt_users):,}")
    
    console.print(table)
    
    # Explanation
    console.print()
    console.print("[yellow]Note:[/yellow] WD-IRT only models unit test items (from `unit_test_scores.csv`),")
    console.print("while SAKT models all practice items. Joinability is limited to unit test scenarios.")
    
    # Sample joined records
    if len(sakt_joinable) > 0:
        console.print()
        console.print(f"[green]Sampling {min(sample_students, len(sakt_users))} students...[/green]")
        
        sample_users = list(sakt_users)[:sample_students]
        for user_id in sample_users:
            user_preds = sakt_joinable[sakt_joinable["user_id"] == user_id]
            if len(user_preds) == 0:
                continue
            
            console.print(f"\n[bold]Student:[/bold] {user_id}")
            console.print(f"  Joinable predictions: {len(user_preds)}")
            
            # Join with WD-IRT params
            joined = user_preds.merge(
                wdirt_params[["item_id", "difficulty", "guessing"]],
                on="item_id",
                how="left"
            )
            
            if len(joined) > 0:
                console.print(f"  Sample items:")
                for _, row in joined.head(3).iterrows():
                    console.print(
                        f"    {row['item_id']}: predicted={row['predicted']:.3f}, "
                        f"actual={row['actual']}, difficulty={row['difficulty']:.2f}"
                    )
    
    # Schema validation
    console.print()
    console.print("[green]Schema Validation:[/green]")
    
    required_cols = {
        "sakt_predictions": ["user_id", "item_id", "predicted", "actual"],
        "wdirt_params": ["item_id", "difficulty", "guessing"],
    }
    
    schemas_ok = True
    for name, df in [("sakt_predictions", sakt_preds), ("wdirt_params", wdirt_params)]:
        missing = set(required_cols[name]) - set(df.columns)
        if missing:
            console.print(f"[red]  {name}: Missing columns {missing}[/red]")
            schemas_ok = False
        else:
            console.print(f"[green]  {name}: Schema OK[/green]")
    
    if schemas_ok and len(overlap_items) > 0:
        console.print()
        console.print("[bold green]✅ Validation passed: Outputs are joinable[/bold green]")
        console.print(f"   Focus on {len(overlap_items):,} unit test items for demo scenarios.")
    elif len(overlap_items) == 0:
        console.print()
        console.print("[yellow]⚠️  Warning: No overlapping items found[/yellow]")
        console.print("   This may indicate a data mismatch or different datasets.")
    else:
        console.print()
        console.print("[red]❌ Validation failed: Schema issues detected[/red]")


if __name__ == "__main__":
    app()

