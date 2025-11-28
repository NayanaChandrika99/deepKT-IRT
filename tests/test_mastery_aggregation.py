# ABOUTME: Tests skill-level mastery aggregation logic.
# ABOUTME: Ensures mastery joins events by position and explodes multi-skill events.

import pandas as pd
import pytest

from src.common.mastery_aggregation import aggregate_skill_mastery


def test_aggregate_skill_mastery_by_position_and_skill():
    events = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["i1", "i2", "i3"],
            "skill_ids": [["s1", "s2"], ["s1"], ["s3"]],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    mastery = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "position": [0, 1, 0],
            "mastery": [0.4, 0.6, 0.9],
        }
    )

    aggregated = aggregate_skill_mastery(mastery, events)

    # u1 has mastery on s1 from two events (0.4, 0.6) and s2 from one event (0.4)
    row_s1 = aggregated[(aggregated["user_id"] == "u1") & (aggregated["skill"] == "s1")].iloc[0]
    assert row_s1["interaction_count"] == 2
    assert row_s1["mastery_mean"] == 0.5
    assert row_s1["mastery_std"] == pytest.approx(0.1)

    row_s2 = aggregated[(aggregated["user_id"] == "u1") & (aggregated["skill"] == "s2")].iloc[0]
    assert row_s2["interaction_count"] == 1
    assert row_s2["mastery_mean"] == 0.4
    assert row_s2["mastery_std"] == 0.0

    row_s3 = aggregated[(aggregated["user_id"] == "u2") & (aggregated["skill"] == "s3")].iloc[0]
    assert row_s3["interaction_count"] == 1
    assert row_s3["mastery_mean"] == 0.9
