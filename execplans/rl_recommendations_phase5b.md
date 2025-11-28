# RL-Based Recommendations (Phase 5B)

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `PLANS.md` at repository root.

## Purpose / Big Picture

After this change, the demo CLI will use a **contextual multi-armed bandit (LinUCB)** to recommend items instead of simple rule-based sorting. This enables the system to:

1. **Learn which items work best** for which student profiles (exploration-exploitation balance)
2. **Improve over time** as more student interactions are observed
3. **Provide uncertainty estimates** ("we're 80% confident this is the right item" vs "we're exploring")

A user can run:

    python scripts/demo_trace.py trace --student-id ABC123 --topic 7.RP.A.1 --time-window 2024-W01 --use-rl

And see recommendations with expected success probability and exploration status. Educators can also view A/B comparison:

    python scripts/demo_trace.py compare-recs --student-id ABC123 --topic 7.RP.A.1

This shows side-by-side: rule-based vs RL recommendations, demonstrating the RL advantage.

**Value for UWorld Demo:**
- Research-backed approach (Schmucker et al. 2025, Meng et al. 2025)
- Self-improving system (learns from usage)
- Industry-standard (Google, Duolingo use contextual bandits)
- Quantifiable: literature shows 14%+ improvement over rule-based

---

## Progress

- [ ] Milestone 1: Create production-ready bandit module
- [ ] Milestone 2: Add persistence (save/load trained models)
- [ ] Milestone 3: Integrate into recommendation system
- [ ] Milestone 4: Add demo CLI support
- [ ] Milestone 5: Create A/B comparison and tests
- [ ] Milestone 6: Documentation and validation

---

## Surprises & Discoveries

(To be populated during implementation)

---

## Decision Log

- Decision: Use LinUCB over Thompson Sampling
  Rationale: LinUCB is simpler, well-understood, and has strong theoretical guarantees. Thompson Sampling requires Bayesian sampling which adds complexity. Can upgrade later if needed.
  Date/Author: 2025-11-28 / ExecPlan

- Decision: Use shared model (not per-arm parameters)
  Rationale: With many items (arms), per-arm parameters lead to cold-start issues. A shared linear model with student-item features generalizes better and learns faster.
  Date/Author: 2025-11-28 / ExecPlan

- Decision: Store bandit state as numpy arrays in .npz file
  Rationale: Simple, portable, no extra dependencies. The state is small (design matrix A, reward vector b, weights theta). Can upgrade to database later for production.
  Date/Author: 2025-11-28 / ExecPlan

---

## Outcomes & Retrospective

(To be populated upon completion)

---

## Context and Orientation

### Current State

The demo CLI (`scripts/demo_trace.py`) calls `recommend_items()` from `src/common/recommendation.py`. This function is **rule-based**: it filters items by topic, excludes high-drift items, and sorts by difficulty. It does not learn from outcomes.

A **proof-of-concept** exists at `scripts/poc_rl_recommendations.py` that demonstrates:
- `LinUCBBandit` class implementing contextual bandit
- `StudentContext` dataclass for student features
- `ItemArm` dataclass for item features
- Feature engineering: mastery, recent_accuracy, help_tendency, difficulty, etc.

The POC is not integrated into the main codebase.

### Key Files

| Path | Purpose |
|------|---------|
| `src/common/recommendation.py` | Current rule-based recommendation (to be extended) |
| `src/common/mastery_aggregation.py` | Aggregates SAKT mastery by skill |
| `scripts/poc_rl_recommendations.py` | POC with LinUCB implementation |
| `scripts/demo_trace.py` | Demo CLI that calls recommendation system |
| `reports/item_params.parquet` | WD-IRT item parameters (difficulty, discrimination) |
| `reports/sakt_student_state.parquet` | SAKT per-interaction mastery |

### Key Terms

**Contextual Bandit**: A reinforcement learning framework where an agent chooses actions (arms) based on context (features), observes a reward, and updates its policy. Unlike full RL, there are no state transitions—each decision is independent.

**LinUCB (Linear Upper Confidence Bound)**: An algorithm that models expected reward as a linear function of features and selects items by balancing expected reward with uncertainty (exploration bonus).

**Arm**: In bandit terminology, an "arm" is a choice. Here, each item is an arm.

**Context**: Features describing the current situation. Here: student mastery, recent accuracy, item difficulty, etc.

**UCB (Upper Confidence Bound)**: Selection criterion: `UCB = expected_reward + alpha * uncertainty`. Higher uncertainty means more exploration.

---

## Plan of Work

### Milestone 1: Create Production-Ready Bandit Module

Create `src/common/bandit.py` with:

1. `StudentContext` dataclass — student features for bandit
2. `ItemArm` dataclass — item features for bandit  
3. `LinUCBBandit` class — the bandit algorithm
4. `BanditRecommendation` dataclass — recommendation with uncertainty

This is mostly extracting and refining the POC code, but with proper typing, docstrings, and edge case handling.

**New file: `src/common/bandit.py`**

    # ABOUTME: Implements contextual multi-armed bandit for adaptive recommendations.
    # ABOUTME: Uses LinUCB algorithm to balance exploration and exploitation.

    from __future__ import annotations
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import List, Tuple, Optional, Dict
    import numpy as np
    import pandas as pd


    @dataclass
    class StudentContext:
        """Student features for contextual bandit."""
        user_id: str
        mastery: float           # Overall mastery (0-1)
        recent_accuracy: float   # Last N questions accuracy
        recent_speed: float      # Normalized response time (0=fast, 1=slow)
        help_tendency: float     # How often they request help (0-1)
        skill_gap: float         # Distance from target skill mastery


    @dataclass
    class ItemArm:
        """Item (arm) in the bandit."""
        item_id: str
        skill: str
        difficulty: float
        discrimination: float = 1.0


    @dataclass
    class BanditRecommendation:
        """Recommendation with bandit metadata."""
        item: ItemArm
        expected_reward: float    # Predicted P(success)
        uncertainty: float        # Confidence width
        ucb_score: float          # UCB = expected + uncertainty
        reason: str               # Human-readable explanation
        is_exploration: bool      # True if uncertainty dominated selection


    class LinUCBBandit:
        """
        Linear Upper Confidence Bound (LinUCB) Contextual Bandit.
        
        Algorithm:
        1. Receive context (student features + item features)
        2. Predict expected reward using linear model: E[r] = theta @ x
        3. Compute uncertainty: sqrt(x @ A_inv @ x)
        4. Select item with highest UCB: E[r] + alpha * uncertainty
        5. Observe reward and update model
        """
        
        def __init__(self, n_features: int = 8, alpha: float = 1.0):
            self.n_features = n_features
            self.alpha = alpha  # Exploration parameter
            self.A = np.eye(n_features)  # Design matrix (regularized covariance)
            self.b = np.zeros(n_features)  # Reward-weighted features
            self.theta = np.zeros(n_features)  # Learned weights
            self.n_updates = 0
        
        def get_context_vector(self, student: StudentContext, item: ItemArm) -> np.ndarray:
            """Create feature vector from student-item pair."""
            return np.array([
                student.mastery,
                student.recent_accuracy,
                student.recent_speed,
                student.help_tendency,
                student.skill_gap,
                item.difficulty,
                student.mastery * item.difficulty,  # Challenge match
                abs(student.mastery - item.difficulty),  # Difficulty gap
            ])
        
        def predict(self, student: StudentContext, item: ItemArm) -> Tuple[float, float]:
            """Predict expected reward and uncertainty."""
            x = self.get_context_vector(student, item)
            expected = np.clip(np.dot(self.theta, x), 0, 1)
            A_inv = np.linalg.inv(self.A)
            uncertainty = self.alpha * np.sqrt(np.dot(x, np.dot(A_inv, x)))
            return expected, uncertainty
        
        def select_best(
            self,
            student: StudentContext,
            items: List[ItemArm],
        ) -> Tuple[ItemArm, float, float, float]:
            """Select best item using UCB."""
            best = None
            best_ucb = float('-inf')
            best_expected = 0.0
            best_uncertainty = 0.0
            
            for item in items:
                expected, uncertainty = self.predict(student, item)
                ucb = expected + uncertainty
                if ucb > best_ucb:
                    best_ucb = ucb
                    best = item
                    best_expected = expected
                    best_uncertainty = uncertainty
            
            return best, best_expected, best_uncertainty, best_ucb
        
        def update(self, student: StudentContext, item: ItemArm, reward: float) -> None:
            """Update model with observed reward (0 or 1)."""
            x = self.get_context_vector(student, item)
            self.A += np.outer(x, x)
            self.b += reward * x
            self.theta = np.linalg.solve(self.A, self.b)
            self.n_updates += 1
        
        def save(self, path: Path) -> None:
            """Save bandit state to file."""
            np.savez(
                path,
                A=self.A,
                b=self.b,
                theta=self.theta,
                n_features=self.n_features,
                alpha=self.alpha,
                n_updates=self.n_updates,
            )
        
        @classmethod
        def load(cls, path: Path) -> "LinUCBBandit":
            """Load bandit state from file."""
            data = np.load(path)
            bandit = cls(n_features=int(data["n_features"]), alpha=float(data["alpha"]))
            bandit.A = data["A"]
            bandit.b = data["b"]
            bandit.theta = data["theta"]
            bandit.n_updates = int(data["n_updates"])
            return bandit


    def build_student_context(
        user_id: str,
        events_df: pd.DataFrame,
        mastery_df: Optional[pd.DataFrame] = None,
        target_skill: Optional[str] = None,
    ) -> StudentContext:
        """Build StudentContext from events data."""
        user_events = events_df[events_df["user_id"] == user_id].tail(20)
        
        if len(user_events) < 3:
            return StudentContext(
                user_id=user_id,
                mastery=0.5,
                recent_accuracy=0.5,
                recent_speed=0.5,
                help_tendency=0.1,
                skill_gap=0.3,
            )
        
        mastery = user_events["correct"].mean()
        recent_5 = user_events.tail(5)
        recent_accuracy = recent_5["correct"].mean()
        
        avg_latency = recent_5["latency_ms"].mean() if "latency_ms" in recent_5 else 30000
        recent_speed = min(avg_latency / 60000, 1.0)
        
        help_tendency = user_events["help_requested"].mean() if "help_requested" in user_events else 0.1
        
        skill_gap = max(0, 0.8 - mastery)
        
        return StudentContext(
            user_id=user_id,
            mastery=mastery,
            recent_accuracy=recent_accuracy,
            recent_speed=recent_speed,
            help_tendency=help_tendency,
            skill_gap=skill_gap,
        )


    def items_to_arms(item_params_df: pd.DataFrame, skill_filter: Optional[str] = None) -> List[ItemArm]:
        """Convert item_params DataFrame to list of ItemArm."""
        df = item_params_df.copy()
        if skill_filter:
            df = df[df["topic"] == skill_filter]
        
        arms = []
        for _, row in df.iterrows():
            arms.append(ItemArm(
                item_id=str(row["item_id"]),
                skill=str(row.get("topic", "unknown")),
                difficulty=float(row["difficulty"]),
                discrimination=float(row.get("discrimination", 1.0)),
            ))
        return arms

**Tests: `tests/test_bandit.py`**

    # ABOUTME: Unit tests for contextual bandit recommendation system.
    # ABOUTME: Verifies LinUCB learns, persists, and recommends correctly.

    import tempfile
    import unittest
    from pathlib import Path
    import numpy as np
    import pandas as pd

    from src.common.bandit import (
        LinUCBBandit,
        StudentContext,
        ItemArm,
        build_student_context,
        items_to_arms,
    )


    class TestLinUCBBandit(unittest.TestCase):
        def setUp(self):
            self.bandit = LinUCBBandit(n_features=8, alpha=1.0)
            self.student = StudentContext(
                user_id="test",
                mastery=0.5,
                recent_accuracy=0.6,
                recent_speed=0.4,
                help_tendency=0.1,
                skill_gap=0.3,
            )
            self.items = [
                ItemArm("Q1", "7.RP.A.1", 0.3),
                ItemArm("Q2", "7.RP.A.1", 0.5),
                ItemArm("Q3", "7.RP.A.1", 0.7),
            ]
        
        def test_initial_predictions_have_uncertainty(self):
            """Untrained bandit should have high uncertainty."""
            for item in self.items:
                expected, uncertainty = self.bandit.predict(self.student, item)
                self.assertGreater(uncertainty, 0.5, "Initial uncertainty should be high")
        
        def test_update_reduces_uncertainty(self):
            """Training should reduce uncertainty."""
            item = self.items[1]
            _, uncertainty_before = self.bandit.predict(self.student, item)
            
            for _ in range(10):
                self.bandit.update(self.student, item, reward=0.8)
            
            _, uncertainty_after = self.bandit.predict(self.student, item)
            self.assertLess(uncertainty_after, uncertainty_before)
        
        def test_select_best_returns_item(self):
            """select_best should return an item."""
            item, expected, uncertainty, ucb = self.bandit.select_best(self.student, self.items)
            self.assertIsNotNone(item)
            self.assertIn(item, self.items)
        
        def test_save_load_roundtrip(self):
            """Bandit state should survive save/load."""
            for _ in range(5):
                self.bandit.update(self.student, self.items[0], 0.7)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "bandit.npz"
                self.bandit.save(path)
                loaded = LinUCBBandit.load(path)
            
            self.assertEqual(loaded.n_updates, self.bandit.n_updates)
            np.testing.assert_array_almost_equal(loaded.theta, self.bandit.theta)


    class TestStudentContextBuilder(unittest.TestCase):
        def test_build_from_events(self):
            events_df = pd.DataFrame({
                "user_id": ["U1"] * 10,
                "correct": [1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
                "latency_ms": [20000] * 10,
                "help_requested": [False] * 10,
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            })
            
            ctx = build_student_context("U1", events_df)
            self.assertEqual(ctx.user_id, "U1")
            self.assertAlmostEqual(ctx.mastery, 0.7, places=1)
        
        def test_cold_start_student(self):
            events_df = pd.DataFrame({
                "user_id": ["U1"],
                "correct": [1],
                "latency_ms": [20000],
                "help_requested": [False],
                "timestamp": pd.date_range("2024-01-01", periods=1, freq="h"),
            })
            
            ctx = build_student_context("U1", events_df)
            self.assertEqual(ctx.mastery, 0.5)  # Default for cold start

**Acceptance Criteria:**
- `LinUCBBandit` can predict, select, and update
- State persists correctly via save/load
- Tests pass: `PYTHONPATH=. uv run pytest tests/test_bandit.py -v`

---

### Milestone 2: Integrate into Recommendation System

Extend `src/common/recommendation.py` to support both rule-based and RL modes.

**Changes to `src/common/recommendation.py`:**

Add a new function `recommend_items_rl()` that uses the bandit:

    def recommend_items_rl(
        user_id: str,
        target_skill: str,
        skill_mastery: pd.DataFrame,
        item_params: pd.DataFrame,
        events_df: pd.DataFrame,
        bandit: LinUCBBandit,
        max_items: int = 5,
        exclude_high_drift: bool = True,
    ) -> List[BanditRecommendation]:
        """
        Recommend items using LinUCB contextual bandit.
        """
        from .bandit import build_student_context, items_to_arms, BanditRecommendation
        
        student = build_student_context(user_id, events_df, target_skill=target_skill)
        
        candidates = item_params[item_params["topic"] == target_skill].copy()
        if exclude_high_drift and "drift_flag" in candidates.columns:
            candidates = candidates[~candidates["drift_flag"].fillna(False)]
        
        items = items_to_arms(candidates)
        if not items:
            return []
        
        recommendations = []
        for item in items:
            expected, uncertainty = bandit.predict(student, item)
            ucb = expected + uncertainty
            is_exploration = uncertainty > expected * 0.5
            
            reason = _generate_rl_reason(student, item, expected, uncertainty, is_exploration)
            
            recommendations.append(BanditRecommendation(
                item=item,
                expected_reward=expected,
                uncertainty=uncertainty,
                ucb_score=ucb,
                reason=reason,
                is_exploration=is_exploration,
            ))
        
        recommendations.sort(key=lambda r: r.ucb_score, reverse=True)
        return recommendations[:max_items]


    def _generate_rl_reason(
        student: StudentContext,
        item: ItemArm,
        expected: float,
        uncertainty: float,
        is_exploration: bool,
    ) -> str:
        """Generate human-readable reason for RL recommendation."""
        parts = []
        
        if is_exploration:
            parts.append("Exploring to learn your preferences")
        else:
            parts.append(f"High confidence: {expected:.0%} expected success")
        
        diff_gap = abs(student.mastery - item.difficulty)
        if diff_gap < 0.15:
            parts.append("good difficulty match")
        elif item.difficulty > student.mastery:
            parts.append("slightly challenging for growth")
        else:
            parts.append("builds confidence")
        
        return "; ".join(parts)

**Acceptance Criteria:**
- `recommend_items_rl()` returns `BanditRecommendation` objects
- Recommendations are sorted by UCB score
- `is_exploration` flag correctly identifies exploration vs exploitation

---

### Milestone 3: Add Demo CLI Support

Update `scripts/demo_trace.py` to support RL recommendations.

**Changes:**

1. Add `--use-rl` flag to `trace` command
2. Add `compare-recs` command for A/B comparison
3. Load/save bandit state from `reports/bandit_state.npz`

**New `trace` command signature:**

    @app.command()
    def trace(
        student_id: str = ...,
        topic: str = ...,
        time_window: str = ...,
        use_rl: bool = typer.Option(False, "--use-rl", help="Use RL bandit recommendations"),
        ...
    ):

**New `compare-recs` command:**

    @app.command()
    def compare_recs(
        student_id: str = ...,
        topic: str = ...,
    ):
        """Compare rule-based vs RL recommendations side-by-side."""
        # Show both recommendation types in a rich table

**Acceptance Criteria:**
- `python scripts/demo_trace.py trace --student-id X --topic Y --time-window Z --use-rl` works
- `python scripts/demo_trace.py compare-recs --student-id X --topic Y` shows comparison table
- Bandit state is saved/loaded from `reports/bandit_state.npz`

---

### Milestone 4: Simulation and Warm-Start

Create a script to simulate training data for the bandit (since we don't have real interaction outcomes).

**New file: `scripts/warmstart_bandit.py`**

    # ABOUTME: Simulates training data to warm-start the LinUCB bandit.
    # ABOUTME: Uses historical events to generate synthetic rewards.

This script:
1. Loads historical events
2. For each event, simulates a "reward" based on actual correctness
3. Updates the bandit
4. Saves warm-started state to `reports/bandit_state.npz`

**Acceptance Criteria:**
- Running `python scripts/warmstart_bandit.py` creates `reports/bandit_state.npz`
- Warm-started bandit has `n_updates > 0`
- Bandit shows reasonable predictions after warm-start

---

### Milestone 5: Tests and Validation

Add comprehensive tests and validate end-to-end.

**Tests:**
1. Unit tests for bandit (Milestone 1)
2. Integration tests for recommendation flow
3. CLI smoke tests

**Validation:**
- Compare RL recommendations to rule-based
- Verify uncertainty decreases with training
- Verify exploration/exploitation balance

---

### Milestone 6: Documentation

Update README.md with RL recommendation usage.

---

## Concrete Steps

**Milestone 1:**

    cd ~/Documents/Personal/deepKT+IRT
    # Create bandit module
    # Create tests
    PYTHONPATH=. uv run pytest tests/test_bandit.py -v
    # Expected: 4+ tests pass

**Milestone 2:**

    # Extend recommendation.py
    PYTHONPATH=. uv run pytest tests/test_recommendation.py tests/test_bandit.py -v
    # Expected: All tests pass

**Milestone 3:**

    # Update demo_trace.py
    python scripts/demo_trace.py trace --student-id <user> --topic 7.RP.A.1 --time-window 2024-W01 --use-rl
    # Expected: RL recommendations with uncertainty shown

**Milestone 4:**

    python scripts/warmstart_bandit.py
    # Expected: reports/bandit_state.npz created
    python scripts/demo_trace.py compare-recs --student-id <user> --topic 7.RP.A.1
    # Expected: Side-by-side comparison table

---

## Validation and Acceptance

The feature is complete when:

1. `python scripts/demo_trace.py trace ... --use-rl` shows RL recommendations
2. Recommendations include expected success probability and uncertainty
3. `compare-recs` command shows rule-based vs RL side-by-side
4. Bandit state persists across runs
5. All tests pass: `PYTHONPATH=. uv run pytest tests/test_bandit.py tests/test_recommendation.py -v`

**Expected Output:**

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RL-Powered Recommendations (LinUCB Bandit)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    | Rank | Item     | Expected | Uncertainty | Mode        | Reason                          |
    |------|----------|----------|-------------|-------------|----------------------------------|
    | 1    | ZK1I2M76 | 72%      | ±12%        | Exploit     | High confidence; good difficulty |
    | 2    | VPZ9SBLW | 65%      | ±18%        | Exploit     | Slightly challenging for growth  |
    | 3    | 2FPPD9A7 | 58%      | ±35%        | Explore     | Exploring to learn preferences   |
    
    Bandit: 847 updates | Alpha: 1.0 | Model: LinUCB
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

## Idempotence and Recovery

- All steps can be re-run safely
- Bandit state is additive (more updates don't hurt)
- If warm-start fails, delete `reports/bandit_state.npz` and retry
- Tests are deterministic

---

## Artifacts and Notes

**POC Reference:** `scripts/poc_rl_recommendations.py` contains the working prototype.

**Key Design Decisions:**
- Shared linear model (not per-arm) for cold-start robustness
- 8-feature context vector: mastery, recent_accuracy, recent_speed, help_tendency, skill_gap, difficulty, challenge_match, difficulty_gap
- Alpha=1.0 for initial exploration; can tune later

**Research Citations:**
- Schmucker et al. (2025). "Learning to Optimize Feedback for One Million Students". arXiv:2508.00270
- Meng et al. (2025). "AVAR-RL: adaptive reinforcement learning approach". 14.2% higher precision
- Li et al. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation". WWW 2010

---

## Interfaces and Dependencies

**New Module: `src/common/bandit.py`**

    class LinUCBBandit:
        def __init__(self, n_features: int = 8, alpha: float = 1.0): ...
        def predict(self, student: StudentContext, item: ItemArm) -> Tuple[float, float]: ...
        def select_best(self, student: StudentContext, items: List[ItemArm]) -> Tuple[ItemArm, float, float, float]: ...
        def update(self, student: StudentContext, item: ItemArm, reward: float) -> None: ...
        def save(self, path: Path) -> None: ...
        @classmethod
        def load(cls, path: Path) -> "LinUCBBandit": ...

    @dataclass
    class StudentContext:
        user_id: str
        mastery: float
        recent_accuracy: float
        recent_speed: float
        help_tendency: float
        skill_gap: float

    @dataclass
    class ItemArm:
        item_id: str
        skill: str
        difficulty: float
        discrimination: float = 1.0

    @dataclass
    class BanditRecommendation:
        item: ItemArm
        expected_reward: float
        uncertainty: float
        ucb_score: float
        reason: str
        is_exploration: bool

    def build_student_context(user_id: str, events_df: pd.DataFrame, ...) -> StudentContext: ...
    def items_to_arms(item_params_df: pd.DataFrame, skill_filter: Optional[str] = None) -> List[ItemArm]: ...

**Extended: `src/common/recommendation.py`**

    def recommend_items_rl(
        user_id: str,
        target_skill: str,
        skill_mastery: pd.DataFrame,
        item_params: pd.DataFrame,
        events_df: pd.DataFrame,
        bandit: LinUCBBandit,
        max_items: int = 5,
        exclude_high_drift: bool = True,
    ) -> List[BanditRecommendation]: ...

---

## Timeline Estimate

| Milestone | Effort | Deliverable |
|-----------|--------|-------------|
| M1: Bandit module | 2-3 hours | `src/common/bandit.py`, tests |
| M2: Integration | 1-2 hours | Extended recommendation.py |
| M3: CLI support | 2-3 hours | --use-rl, compare-recs |
| M4: Warm-start | 1-2 hours | warmstart_bandit.py |
| M5: Validation | 1-2 hours | Tests, validation |
| M6: Docs | 30 min | README update |

**Total: 8-12 hours (2-3 days)**

---

## Revision Log

- 2025-11-28: Initial ExecPlan drafted based on POC and enhancements_phase5.md requirements

