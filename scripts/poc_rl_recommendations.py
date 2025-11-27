# ABOUTME: Proof-of-concept for reinforcement learning recommendations.
# ABOUTME: Demonstrates contextual bandit approach for adaptive item selection.

"""
Proof-of-Concept: RL-Based Recommendations (Contextual Multi-Armed Bandit)

This script demonstrates how RL recommendations would work using a simple
Contextual Bandit (LinUCB) for adaptive item selection.

Instead of rule-based recommendations, this learns which items work best
for which student profiles, balancing exploration and exploitation.

Usage:
    python scripts/poc_rl_recommendations.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class StudentContext:
    """Student features for contextual bandit."""
    user_id: str
    mastery: float              # Overall mastery (0-1)
    recent_accuracy: float      # Last 5 questions accuracy
    recent_speed: float         # Normalized recent response time
    help_tendency: float        # How often they request help
    skill_gap: float            # Distance from target skill mastery


@dataclass
class ItemArm:
    """Item (arm) in the bandit."""
    item_id: str
    skill: str
    difficulty: float
    expected_time_sec: float


@dataclass
class Recommendation:
    """Recommended item with explanation."""
    item: ItemArm
    expected_reward: float
    uncertainty: float
    reason: str


class LinUCBBandit:
    """
    Linear Upper Confidence Bound (LinUCB) Contextual Bandit.
    
    This is a simplified implementation for POC purposes.
    Production would use optimized libraries like Vowpal Wabbit.
    """
    
    def __init__(self, n_features: int, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha  # Exploration parameter
        
        # Per-arm parameters (we'll use a shared model for simplicity)
        self.A = np.eye(n_features)  # Design matrix
        self.b = np.zeros(n_features)  # Reward vector
        self.theta = np.zeros(n_features)  # Learned weights
        
    def get_context_vector(self, student: StudentContext, item: ItemArm) -> np.ndarray:
        """Create feature vector from student-item pair."""
        return np.array([
            student.mastery,
            student.recent_accuracy,
            student.recent_speed,
            student.help_tendency,
            student.skill_gap,
            item.difficulty,
            # Interaction features
            student.mastery * item.difficulty,  # Challenge match
            abs(student.mastery - item.difficulty),  # Difficulty gap
        ])
    
    def predict(self, student: StudentContext, item: ItemArm) -> Tuple[float, float]:
        """Predict expected reward and uncertainty."""
        x = self.get_context_vector(student, item)
        
        # Expected reward
        expected = np.dot(self.theta, x)
        
        # Uncertainty (confidence bound)
        A_inv = np.linalg.inv(self.A)
        uncertainty = self.alpha * np.sqrt(np.dot(x, np.dot(A_inv, x)))
        
        return expected, uncertainty
    
    def select_arm(
        self, 
        student: StudentContext, 
        items: List[ItemArm]
    ) -> Tuple[ItemArm, float, float]:
        """Select best item using UCB."""
        best_item = None
        best_ucb = float('-inf')
        best_expected = 0
        best_uncertainty = 0
        
        for item in items:
            expected, uncertainty = self.predict(student, item)
            ucb = expected + uncertainty  # Upper confidence bound
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_item = item
                best_expected = expected
                best_uncertainty = uncertainty
        
        return best_item, best_expected, best_uncertainty
    
    def update(self, student: StudentContext, item: ItemArm, reward: float):
        """Update model with observed reward."""
        x = self.get_context_vector(student, item)
        
        self.A += np.outer(x, x)
        self.b += reward * x
        self.theta = np.linalg.solve(self.A, self.b)


def create_student_context(
    user_id: str,
    events_df: pd.DataFrame,
    mastery_df: Optional[pd.DataFrame] = None,
) -> StudentContext:
    """Create student context from events data."""
    user_events = events_df[events_df["user_id"] == user_id].tail(20)
    
    if len(user_events) < 5:
        # Default context for cold-start
        return StudentContext(
            user_id=user_id,
            mastery=0.5,
            recent_accuracy=0.5,
            recent_speed=0.5,
            help_tendency=0.1,
            skill_gap=0.3,
        )
    
    recent_5 = user_events.tail(5)
    
    mastery = user_events["correct"].mean()
    recent_accuracy = recent_5["correct"].mean()
    
    # Normalize speed (lower is faster)
    avg_latency = recent_5["latency_ms"].mean()
    recent_speed = min(avg_latency / 60000, 1.0)  # Cap at 1 minute
    
    help_tendency = user_events["help_requested"].mean() if "help_requested" in user_events else 0.1
    
    # Skill gap (how far from target 0.8 mastery)
    skill_gap = max(0, 0.8 - mastery)
    
    return StudentContext(
        user_id=user_id,
        mastery=mastery,
        recent_accuracy=recent_accuracy,
        recent_speed=recent_speed,
        help_tendency=help_tendency,
        skill_gap=skill_gap,
    )


def create_item_pool(wdirt_df: Optional[pd.DataFrame] = None) -> List[ItemArm]:
    """Create pool of candidate items."""
    if wdirt_df is not None and len(wdirt_df) > 0:
        items = []
        for _, row in wdirt_df.head(20).iterrows():
            items.append(ItemArm(
                item_id=row["item_id"],
                skill=row.get("topic", "unknown"),
                difficulty=row["difficulty"],
                expected_time_sec=30,  # Default
            ))
        return items
    
    # Synthetic items for demo
    return [
        ItemArm("Q001", "7.RP.A.1", 0.3, 25),
        ItemArm("Q002", "7.RP.A.1", 0.5, 35),
        ItemArm("Q003", "7.RP.A.1", 0.7, 45),
        ItemArm("Q004", "7.RP.A.2", 0.4, 30),
        ItemArm("Q005", "7.RP.A.2", 0.6, 40),
        ItemArm("Q006", "7.RP.A.2", 0.8, 50),
        ItemArm("Q007", "7.NS.A.1", 0.35, 28),
        ItemArm("Q008", "7.NS.A.1", 0.55, 38),
        ItemArm("Q009", "7.EE.A.1", 0.45, 32),
        ItemArm("Q010", "7.EE.A.1", 0.65, 42),
    ]


def get_recommendation_reason(
    student: StudentContext, 
    item: ItemArm, 
    uncertainty: float
) -> str:
    """Generate human-readable reason for recommendation."""
    reasons = []
    
    # Challenge match
    diff_gap = abs(student.mastery - item.difficulty)
    if diff_gap < 0.15:
        reasons.append("Good difficulty match")
    elif item.difficulty > student.mastery:
        reasons.append("Slightly challenging to promote growth")
    else:
        reasons.append("Builds confidence with achievable difficulty")
    
    # Exploration vs exploitation
    if uncertainty > 0.3:
        reasons.append("Exploring to learn student preferences")
    else:
        reasons.append("High confidence based on similar students")
    
    # Speed consideration
    if student.recent_speed > 0.7 and item.expected_time_sec < 35:
        reasons.append("Quick item for student's current pace")
    
    return "; ".join(reasons)


def recommend_items(
    bandit: LinUCBBandit,
    student: StudentContext,
    items: List[ItemArm],
    n_recommendations: int = 3,
) -> List[Recommendation]:
    """Get top-N recommendations for a student."""
    
    scored_items = []
    for item in items:
        expected, uncertainty = bandit.predict(student, item)
        ucb = expected + uncertainty
        scored_items.append((item, expected, uncertainty, ucb))
    
    # Sort by UCB (exploration-exploitation balance)
    scored_items.sort(key=lambda x: x[3], reverse=True)
    
    recommendations = []
    for item, expected, uncertainty, _ in scored_items[:n_recommendations]:
        reason = get_recommendation_reason(student, item, uncertainty)
        recommendations.append(Recommendation(
            item=item,
            expected_reward=expected,
            uncertainty=uncertainty,
            reason=reason,
        ))
    
    return recommendations


def display_student_context(student: StudentContext):
    """Display student context."""
    console.print(Panel(
        f"[bold]User ID:[/bold] {student.user_id}\n"
        f"[bold]Overall Mastery:[/bold] {student.mastery:.2f}\n"
        f"[bold]Recent Accuracy:[/bold] {student.recent_accuracy:.2f}\n"
        f"[bold]Recent Speed:[/bold] {student.recent_speed:.2f} (0=fast, 1=slow)\n"
        f"[bold]Help Tendency:[/bold] {student.help_tendency:.2f}\n"
        f"[bold]Skill Gap:[/bold] {student.skill_gap:.2f}",
        title="Student Context",
        border_style="cyan"
    ))


def display_recommendations(recommendations: List[Recommendation]):
    """Display recommendations in a table."""
    table = Table(title="RL-Powered Recommendations")
    table.add_column("Rank", style="dim")
    table.add_column("Item", style="cyan")
    table.add_column("Skill", style="magenta")
    table.add_column("Difficulty", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Uncertainty", justify="right")
    table.add_column("Reason")
    
    for i, rec in enumerate(recommendations, 1):
        table.add_row(
            str(i),
            rec.item.item_id,
            rec.item.skill[:12] + "..." if len(rec.item.skill) > 12 else rec.item.skill,
            f"{rec.item.difficulty:.2f}",
            f"{rec.expected_reward:.3f}",
            f"±{rec.uncertainty:.3f}",
            rec.reason[:40] + "..." if len(rec.reason) > 40 else rec.reason,
        )
    
    console.print(table)


def simulate_learning_episode(bandit: LinUCBBandit, items: List[ItemArm], n_rounds: int = 50):
    """Simulate learning from interactions."""
    
    console.print("\n[dim]Simulating 50 student interactions to train the bandit...[/dim]")
    
    for _ in range(n_rounds):
        # Random student context
        student = StudentContext(
            user_id=f"sim_{np.random.randint(1000)}",
            mastery=np.random.uniform(0.3, 0.8),
            recent_accuracy=np.random.uniform(0.2, 0.9),
            recent_speed=np.random.uniform(0.2, 0.8),
            help_tendency=np.random.uniform(0, 0.3),
            skill_gap=np.random.uniform(0, 0.5),
        )
        
        # Select item
        item, _, _ = bandit.select_arm(student, items)
        
        # Simulate reward (higher if difficulty matches mastery)
        diff_match = 1 - abs(student.mastery - item.difficulty)
        noise = np.random.normal(0, 0.1)
        reward = np.clip(diff_match + noise, 0, 1)
        
        # Update bandit
        bandit.update(student, item, reward)
    
    console.print("[green]✅ Bandit trained on 50 simulated interactions[/green]\n")


def main():
    """Run POC demonstration."""
    
    console.rule("[bold blue]Proof-of-Concept: RL-Based Recommendations[/bold blue]")
    console.print()
    
    # Load data if available
    events_path = Path("data/interim/edm_cup_2023_42_events.parquet")
    wdirt_path = Path("reports/item_params.parquet")
    
    if events_path.exists():
        events_df = pd.read_parquet(events_path)
        sample_user = events_df["user_id"].iloc[0]
    else:
        console.print("[yellow]Events data not found. Using synthetic student.[/yellow]\n")
        events_df = None
        sample_user = "DEMO_STUDENT"
    
    if wdirt_path.exists():
        wdirt_df = pd.read_parquet(wdirt_path)
        items = create_item_pool(wdirt_df)
    else:
        console.print("[yellow]Item params not found. Using synthetic items.[/yellow]\n")
        items = create_item_pool()
    
    # Initialize bandit
    n_features = 8  # Matches get_context_vector output
    bandit = LinUCBBandit(n_features=n_features, alpha=1.0)
    
    # Simulate training
    simulate_learning_episode(bandit, items, n_rounds=50)
    
    # Create student context
    if events_df is not None:
        student = create_student_context(sample_user, events_df)
    else:
        student = StudentContext(
            user_id="DEMO_STUDENT",
            mastery=0.55,
            recent_accuracy=0.60,
            recent_speed=0.45,
            help_tendency=0.15,
            skill_gap=0.25,
        )
    
    display_student_context(student)
    
    # Get recommendations
    recommendations = recommend_items(bandit, student, items, n_recommendations=5)
    display_recommendations(recommendations)
    
    # Compare with rule-based
    console.rule("[bold]Comparison: RL vs Rule-Based[/bold]")
    
    console.print("[bold cyan]RL Approach (LinUCB):[/bold cyan]")
    console.print("  • Learns from student interactions over time")
    console.print("  • Balances exploration (try new items) and exploitation (use best known)")
    console.print("  • Adapts to individual student patterns")
    console.print("  • Gets better as more data is collected")
    
    console.print("\n[bold yellow]Rule-Based Approach (Current):[/bold yellow]")
    console.print("  • Fixed rules: if mastery < X, recommend Y")
    console.print("  • No exploration: always recommends same items")
    console.print("  • One-size-fits-all recommendations")
    console.print("  • Doesn't improve over time")
    
    # Show what full implementation would add
    console.print("\n" + "=" * 60)
    console.print("[bold]Full Implementation Would Add:[/bold]")
    console.print("  • Persistent bandit model (save/load)")
    console.print("  • Online learning from real interactions")
    console.print("  • Multi-objective rewards (correctness + retention)")
    console.print("  • Thompson Sampling or Neural Bandit alternatives")
    console.print("  • A/B testing framework for RL vs rule-based")
    console.print("  • Expected 14%+ improvement over rule-based (from literature)")
    console.print("=" * 60)


if __name__ == "__main__":
    main()

