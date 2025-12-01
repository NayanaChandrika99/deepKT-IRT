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
    max_candidates: int = 200,
) -> List[Dict]:
    """Compute a single decision event with sampled candidate scoring.

    Strategy: Sample candidates to balance:
    1. All items from target skill (pedagogically relevant)
    2. Stratified sample across difficulty ranges (show diversity)
    3. Limit total to max_candidates for performance
    """

    # Build student context
    student_events = events_df[
        (events_df['user_id'] == student_id) &
        (events_df['timestamp'] <= timestamp)
    ].tail(20)

    if len(student_events) < 3:
        return []

    student = build_student_context(student_id, student_events, target_skill=target_skill)

    # Filter out drifted items (quality filter)
    all_items = item_params_df.copy()
    if 'drift_flag' in all_items.columns:
        all_items = all_items[~all_items['drift_flag'].fillna(False)]

    if all_items.empty:
        return []

    # Smart sampling strategy
    # 1. Include ALL items from target skill (these are pedagogically relevant)
    target_skill_items = all_items[all_items['topic'] == target_skill]

    # 2. Sample from other skills across difficulty quantiles
    other_items = all_items[all_items['topic'] != target_skill]

    n_target = len(target_skill_items)
    n_remaining = max(0, max_candidates - n_target)

    if n_remaining > 0 and len(other_items) > 0:
        # Stratified sampling by difficulty quartiles
        other_items = other_items.copy()
        other_items['difficulty_bin'] = pd.qcut(
            other_items['difficulty'],
            q=min(4, len(other_items)),
            labels=False,
            duplicates='drop'
        )
        sampled_other = other_items.groupby('difficulty_bin', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1, n_remaining // 4)), random_state=42),
            include_groups=False
        ).head(n_remaining)

        candidates = pd.concat([target_skill_items, sampled_other]).reset_index(drop=True)
    else:
        candidates = target_skill_items.reset_index(drop=True)

    # Vectorized UCB computation (much faster)
    decision_rows = []
    context_vectors = []

    # First pass: build context vectors
    for _, item_row in candidates.iterrows():
        item = ItemArm(
            item_id=str(item_row['item_id']),
            skill=target_skill,
            difficulty=float(item_row['difficulty']),
            discrimination=float(item_row.get('discrimination', 1.0)),
        )
        x = bandit.get_context_vector(student, item)
        context_vectors.append(x)

    # Vectorized computation
    X = np.array(context_vectors)  # Shape: (n_candidates, n_features)
    mu_vec = np.clip(X @ bandit.theta, 0, 1)  # Vectorized dot product

    # Compute sigma for each context (still needs loop for solve, but faster)
    sigma_vec = np.zeros(len(candidates))
    for i, x in enumerate(context_vectors):
        y = np.linalg.solve(bandit.A, x)
        sigma_vec[i] = bandit.alpha * np.sqrt(np.dot(x, y))

    ucb_vec = mu_vec + sigma_vec
    is_explore_vec = sigma_vec > mu_vec * bandit.exploration_threshold

    # Build decision rows with vectorized results
    sakt_predictions = {}
    for i, (_, item_row) in enumerate(candidates.iterrows()):
        item_id = str(item_row['item_id'])
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
            'irt_difficulty': float(item_row['difficulty']),
            'irt_discrimination': float(item_row.get('discrimination', 1.0)),
            'drift_flag': item_row.get('drift_flag', False),
            'sakt_p_correct': sakt_p,
            # Bandit math (from vectorized computation)
            'mu': float(mu_vec[i]),
            'sigma': float(sigma_vec[i]),
            'ucb': float(ucb_vec[i]),
            'alpha': bandit.alpha,
            'mode': 'explore' if is_explore_vec[i] else 'exploit',
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
    # Strategy: Pre-filter viable students, then sample decision points
    console.print(f"\n[dim]Pre-filtering viable students...[/dim]")

    # Get students with sufficient event history and skill mastery data
    event_counts = events_df['user_id'].value_counts()
    students_with_history = event_counts[event_counts >= 20].index

    students_with_skills = skill_mastery_df['user_id'].unique()
    viable_students = list(set(students_with_history) & set(students_with_skills))

    console.print(f"  Viable students: {len(viable_students):,} (of {events_df['user_id'].nunique():,} total)")

    if len(viable_students) == 0:
        console.print("[red]✗ No viable students found![/red]")
        return

    # Sample more students to ensure we hit max_decisions
    # Estimate: ~0.12 success rate, so sample 10x to be safe
    n_students_to_sample = min(len(viable_students), max(500, max_decisions // 10))
    sampled_students = np.random.choice(viable_students, size=n_students_to_sample, replace=False)

    console.print(f"  Sampling from {len(sampled_students):,} students")
    console.print(f"\n[dim]Generating {max_decisions} decision events...[/dim]")

    all_decisions = []
    decision_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating decisions...", total=max_decisions)

        for student_id in sampled_students:
            if decision_count >= max_decisions:
                break

            student_events = events_df[events_df['user_id'] == student_id].sort_values('timestamp')

            # Sample decision points (every 5th event)
            decision_points = student_events.iloc[::5]

            # Get student's active skills (pre-filtered, so this should always succeed)
            student_skills = skill_mastery_df[skill_mastery_df['user_id'] == student_id]['skill'].unique()
            if len(student_skills) == 0:
                continue

            for _, event in decision_points.iterrows():
                if decision_count >= max_decisions:
                    break

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

                if decision_rows:  # Only count successful generations
                    all_decisions.extend(decision_rows)
                    decision_count += 1
                    progress.advance(task, advance=1)
    
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
