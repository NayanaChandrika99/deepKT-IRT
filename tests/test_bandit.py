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
    BanditRecommendation,
    build_student_context,
    items_to_arms,
    generate_rl_reason,
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
        item, expected, uncertainty, ucb = self.bandit.select_best(
            self.student, self.items
        )
        self.assertIsNotNone(item)
        self.assertIn(item, self.items)

    def test_select_best_empty_list(self):
        """select_best should handle empty item list."""
        item, expected, uncertainty, ucb = self.bandit.select_best(self.student, [])
        self.assertIsNone(item)

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

    def test_context_vector_shape(self):
        """Context vector should have correct shape."""
        x = self.bandit.get_context_vector(self.student, self.items[0])
        self.assertEqual(len(x), 8)

    def test_predictions_bounded(self):
        """Predictions should be clipped to [0, 1]."""
        # Train with extreme values
        for _ in range(20):
            self.bandit.update(self.student, self.items[0], 1.0)

        expected, _ = self.bandit.predict(self.student, self.items[0])
        self.assertGreaterEqual(expected, 0.0)
        self.assertLessEqual(expected, 1.0)


class TestStudentContextBuilder(unittest.TestCase):
    def test_build_from_events(self):
        events_df = pd.DataFrame(
            {
                "user_id": ["U1"] * 10,
                "correct": [1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
                "latency_ms": [20000] * 10,
                "help_requested": [False] * 10,
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            }
        )

        ctx = build_student_context("U1", events_df)
        self.assertEqual(ctx.user_id, "U1")
        self.assertAlmostEqual(ctx.mastery, 0.7, places=1)

    def test_cold_start_student(self):
        events_df = pd.DataFrame(
            {
                "user_id": ["U1"],
                "correct": [1],
                "latency_ms": [20000],
                "help_requested": [False],
                "timestamp": pd.date_range("2024-01-01", periods=1, freq="h"),
            }
        )

        ctx = build_student_context("U1", events_df)
        self.assertEqual(ctx.mastery, 0.5)  # Default for cold start

    def test_missing_user(self):
        events_df = pd.DataFrame(
            {
                "user_id": ["U1"] * 5,
                "correct": [1, 0, 1, 1, 0],
                "latency_ms": [20000] * 5,
                "help_requested": [False] * 5,
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            }
        )

        ctx = build_student_context("UNKNOWN_USER", events_df)
        self.assertEqual(ctx.mastery, 0.5)  # Default for missing user


class TestItemsToArms(unittest.TestCase):
    def test_convert_items(self):
        df = pd.DataFrame(
            {
                "item_id": ["Q1", "Q2", "Q3"],
                "topic": ["7.RP.A.1", "7.RP.A.1", "7.RP.A.2"],
                "difficulty": [0.3, 0.5, 0.7],
                "discrimination": [1.0, 1.2, 0.9],
            }
        )

        arms = items_to_arms(df)
        self.assertEqual(len(arms), 3)
        self.assertEqual(arms[0].item_id, "Q1")
        self.assertEqual(arms[0].difficulty, 0.3)

    def test_filter_by_skill(self):
        df = pd.DataFrame(
            {
                "item_id": ["Q1", "Q2", "Q3"],
                "topic": ["7.RP.A.1", "7.RP.A.1", "7.RP.A.2"],
                "difficulty": [0.3, 0.5, 0.7],
            }
        )

        arms = items_to_arms(df, skill_filter="7.RP.A.1")
        self.assertEqual(len(arms), 2)


class TestGenerateReason(unittest.TestCase):
    def test_exploration_reason(self):
        student = StudentContext("U1", 0.5, 0.5, 0.5, 0.1, 0.3)
        item = ItemArm("Q1", "7.RP.A.1", 0.5)

        reason = generate_rl_reason(student, item, 0.5, 0.4, is_exploration=True)
        self.assertIn("Exploring", reason)

    def test_exploitation_reason(self):
        student = StudentContext("U1", 0.5, 0.5, 0.5, 0.1, 0.3)
        item = ItemArm("Q1", "7.RP.A.1", 0.5)

        reason = generate_rl_reason(student, item, 0.7, 0.1, is_exploration=False)
        self.assertIn("confidence", reason)


if __name__ == "__main__":
    unittest.main()

