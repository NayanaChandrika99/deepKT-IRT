#!/usr/bin/env python3
# ABOUTME: Generates bandit_decisions.parquet by replaying historical recommendation decisions.
# ABOUTME: Logs student context, candidate scoring (mu/sigma/UCB), chosen items, and outcomes.

"""
Generate Bandit Decisions Artifact

Creates reports/bandit_decisions.parquet with one row per recommendation decision.
This artifact is used in the system demo notebook to visualize the RL decision process.

Usage:
    python scripts/generate_bandit_decisions.py
    python scripts/generate_bandit_decisions.py --max-decisions 1000
"""

from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer

from src.common.bandit import LinUCBBandit, build_student_context, ItemArm

console = Console()
app = typer.Typer(help="Generate bandit decision log for visualization.")


def compute_decision_event(
    bandit: LinUCBBandit,
    student_id: str,
    events_df: pd.DataFrame,
    item_params_df: pd.DataFrame,
    skill_mastery_df: pd.DataFrame,
    target_skill: str,
    timestamp: int,
) -> List[Dict]:
    """Compute a single decision event with full candidate scoring."""
    
    # Build student context
    student_events = events_df[
        (events_df['user_id'] == student_id) & 
        (events_df['timestamp'] <= timestamp)
    ].tail(20)
    
    if len(student_events) < 3:
        return []
    
    student = build_student_context(student_id, student_events, target_skill=target_skill)
    
    # Get candidate items for this skill
    candidates = item_params_df[item_params_df['topic'] == target_skill].copy()
    if candidates.empty:
        return []
    
    # Filter out drifted items
    if 'drift_flag' in candidates.columns:
        candidates = candidates[~candidates['drift_flag'].fillna(False)]
    
    # Get SAKT predictions if available
    sakt_predictions = {}
    
    decision_rows = []
    
    for _, item_row in candidates.iterrows():
        item_id = str(item_row['item_id'])
        difficulty = float(item_row['difficulty'])
        discrimination = float(item_row.get('discrimination', 1.0))
        
        # Create ItemArm
        item = ItemArm(
            item_id=item_id,
            skill=target_skill,
            difficulty=difficulty,
            discrimination=discrimination,
        )
        
        # Compute bandit scores
        x = bandit.get_context_vector(student, item)
        mu = np.clip(np.dot(bandit.theta, x), 0, 1)
        y = np.linalg.solve(bandit.A, x)
        sigma = bandit.alpha * np.sqrt(np.dot(x, y))
        ucb = mu + sigma
        
        # Determine mode
        is_explore = sigma > mu * bandit.exploration_threshold
        
        # Get SAKT prediction (mock for now, would come from sakt_predictions.parquet)
        sakt_p = sakt_predictions.get(item_id, student.mastery)
        
        decision_rows.append({
            't': timestamp,
            'student_id': student_id,
            'skill_id': target_skill,
            'candidate_item_id': item_id,
            # Student features
            'x_mastery': student.mastery,
            'x_recent_accuracy': student.recent_accuracy,
            'x_recent_speed': student.recent_speed,
            'x_help_tendency': student.help_tendency,
            'x_skill_gap': student.skill_gap,
            # Item features
            'irt_difficulty': difficulty,
            'irt_discrimination': discrimination,
            'drift_flag': item_row.get('drift_flag', False),
            'sakt_p_correct': sakt_p,
            # Bandit math
            'mu': float(mu),
            'sigma': float(sigma),
            'ucb': float(ucb),
            'alpha': bandit.alpha,
            'mode': 'explore' if is_explore else 'exploit',
        })
    
    # Rank by UCB
    decision_df = pd.DataFrame(decision_rows)
    decision_df['rank'] = decision_df['ucb'].rank(ascending=False, method='first').astype(int)
    
    # Mark chosen item (top UCB)
    decision_df['chosen'] = decision_df['rank'] == 1
    chosen_item_id = decision_df[decision_df['chosen']]['candidate_item_id'].iloc[0]
    
    # Get observed reward (if this item was actually attempted)
    future_events = events_df[
        (events_df['user_id'] == student_id) &
        (events_df['timestamp'] > timestamp) &
        (events_df['item_id'] == chosen_item_id)
    ]
    
    if not future_events.empty:
        reward_observed = int(future_events.iloc[0]['correct'])
    else:
        reward_observed = -1  # Not observed
    
    decision_df['reward_observed'] = reward_observed
    decision_df['topk_ucb_gap'] = decision_df['ucb'].max() - decision_df['ucb']
    
    return decision_df.to_dict('records')


@app.command()
def generate(
    events_path: Path = typer.Option(
        Path("data/interim/edm_cup_2023_42_events.parquet"),
        help="Path to events parquet",
    ),
    item_params_path: Path = typer.Option(
        Path("reports/item_params.parquet"),
        help="Path to item parameters",
    ),
    skill_mastery_path: Path = typer.Option(
        Path("reports/skill_mastery.parquet"),
        help="Path to skill mastery",
    ),
    bandit_path: Path = typer.Option(
        Path("reports/bandit_state.npz"),
        help="Path to bandit state",
    ),
    output_path: Path = typer.Option(
        Path("reports/bandit_decisions.parquet"),
        help="Output path for decisions",
    ),
    max_decisions: int = typer.Option(
        500,
        help="Maximum decision events to generate",
    ),
) -> None:
    """Generate bandit decision log for visualization."""
    
    console.rule("[bold blue]Generating Bandit Decisions[/bold blue]")
    
    # Load data
    console.print("[dim]Loading data...[/dim]")
    events_df = pd.read_parquet(events_path)
    item_params_df = pd.read_parquet(item_params_path)
    skill_mastery_df = pd.read_parquet(skill_mastery_path)
    
    console.print(f"  Events: {len(events_df):,}")
    console.print(f"  Items: {len(item_params_df):,}")
    console.print(f"  Skills: {skill_mastery_df['skill'].nunique():,}")
    
    # Load or create bandit
    if bandit_path.exists():
        console.print(f"[dim]Loading bandit from {bandit_path}[/dim]")
        bandit = LinUCBBandit.load(bandit_path)
        console.print(f"  Bandit updates: {bandit.n_updates:,}")
    else:
        console.print("[yellow]No bandit state found. Using fresh bandit.[/yellow]")
        bandit = LinUCBBandit(n_features=8, alpha=1.0)
    
    # Sample decision events
    # Strategy: Sample students, pick their interaction points, simulate recommendations
    console.print(f"\n[dim]Sampling {max_decisions} decision events...[/dim]")
    
    students = events_df['user_id'].unique()
    sampled_students = np.random.choice(students, size=min(100, len(students)), replace=False)
    
    all_decisions = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating decisions...", total=max_decisions)
        
        for student_id in sampled_students:
            if len(all_decisions) >= max_decisions:
                break
            
            student_events = events_df[events_df['user_id'] == student_id].sort_values('timestamp')
            
            # Sample decision points (every 5th event)
            decision_points = student_events.iloc[::5]
            
            for _, event in decision_points.iterrows():
                if len(all_decisions) >= max_decisions:
                    break
                
                # Get student's active skills
                student_skills = skill_mastery_df[skill_mastery_df['user_id'] == student_id]['skill'].unique()
                if len(student_skills) == 0:
                    continue
                
                target_skill = np.random.choice(student_skills)
                
                decision_rows = compute_decision_event(
                    bandit=bandit,
                    student_id=student_id,
                    events_df=events_df,
                    item_params_df=item_params_df,
                    skill_mastery_df=skill_mastery_df,
                    target_skill=target_skill,
                    timestamp=event['timestamp'],
                )
                
                all_decisions.extend(decision_rows)
                progress.advance(task, advance=len(decision_rows))
    
    # Save to parquet
    decisions_df = pd.DataFrame(all_decisions)
    
    console.print(f"\n[dim]Saving to {output_path}...[/dim]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_df.to_parquet(output_path, index=False)
    
    console.print(f"\n[green]✓ Generated {len(decisions_df):,} decision rows[/green]")
    console.print(f"  Unique decisions: {decisions_df.groupby(['t', 'student_id']).ngroups:,}")
    console.print(f"  Explore ratio: {(decisions_df['mode'] == 'explore').mean():.1%}")
    console.print(f"  Chosen items: {decisions_df['chosen'].sum():,}")


if __name__ == "__main__":
    app()
