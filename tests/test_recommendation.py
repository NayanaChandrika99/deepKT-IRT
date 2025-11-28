# ABOUTME: Tests recommendation logic combining mastery and item health.
# ABOUTME: Ensures items are filtered by skill and drift, sorted by difficulty.

from src.common.recommendation import recommend_items
import pandas as pd


def test_recommend_items_filters_and_sorts():
    skill_mastery = pd.DataFrame(
        {
            "user_id": ["u1"],
            "skill": ["s1"],
            "mastery_mean": [0.45],
            "mastery_std": [0.05],
            "interaction_count": [3],
        }
    )
    item_params = pd.DataFrame(
        {
            "item_id": ["easy", "mid", "hard", "drifty"],
            "topic": ["s1", "s1", "s1", "s1"],
            "difficulty": [0.2, 0.5, 0.8, 0.3],
            "discrimination": [1.0, 1.0, 1.0, 1.0],
            "drift_flag": [False, False, False, True],
        }
    )

    recs = recommend_items("u1", "s1", skill_mastery, item_params, max_items=3, exclude_high_drift=True)

    assert [r.item_id for r in recs] == ["easy", "mid", "hard"]
    for rec in recs:
        assert "s1" in rec.reason
        assert "0.45" in rec.reason


def test_recommend_items_handles_missing_user():
    skill_mastery = pd.DataFrame(
        {"user_id": ["other"], "skill": ["s1"], "mastery_mean": [0.6], "mastery_std": [0.0], "interaction_count": [1]}
    )
    item_params = pd.DataFrame(
        {
            "item_id": ["x"],
            "topic": ["s1"],
            "difficulty": [0.4],
            "discrimination": [1.0],
            "drift_flag": [False],
        }
    )

    recs = recommend_items("u1", "s1", skill_mastery, item_params, max_items=1)
    assert recs[0].item_id == "x"
