# ABOUTME: End-to-end demo pipeline test for mastery aggregation and recommendations.
# ABOUTME: Ensures skill mastery parquet can be built and recommendations generated from item params.

import pandas as pd

from src.common.mastery_aggregation import aggregate_skill_mastery
from src.common.recommendation import recommend_items


def test_mastery_and_recommendation_pipeline():
    events = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "skill_ids": [["s1"], ["s1", "s2"], ["s2"]],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    mastery = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "position": [0, 1, 2],
            "mastery": [0.3, 0.5, 0.7],
        }
    )
    item_params = pd.DataFrame(
        {
            "item_id": ["easy_s1", "hard_s1", "s2_item"],
            "topic": ["s1", "s1", "s2"],
            "difficulty": [0.2, 0.8, 0.4],
            "discrimination": [1.0, 1.0, 1.0],
            "drift_flag": [False, False, False],
        }
    )

    skill_mastery = aggregate_skill_mastery(mastery, events)
    assert not skill_mastery.empty
    assert set(skill_mastery["skill"]) == {"s1", "s2"}

    recs = recommend_items("u1", "s1", skill_mastery, item_params, max_items=2)
    assert [r.item_id for r in recs] == ["easy_s1", "hard_s1"]
    for rec in recs:
        assert "s1" in rec.reason
