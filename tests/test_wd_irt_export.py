# ABOUTME: Tests drift scoring for WD-IRT item export.
# ABOUTME: Ensures drift flags are derived from difficulty z-scores.

import pandas as pd

from src.wd_irt.export import compute_drift_flags


def test_compute_drift_flags_marks_outliers():
    df = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "difficulty": [0.0, 0.1, 2.5, -2.2],
        }
    )

    drifted = compute_drift_flags(df, threshold=1.0)

    assert set(drifted.columns) == {"item_id", "drift_flag", "drift_score"}
    flags = drifted.set_index("item_id")["drift_flag"].to_dict()
    assert flags["c"] is True
    assert flags["d"] is True
    assert flags["a"] is False
    assert flags["b"] is False


def test_compute_drift_flags_handles_small_samples():
    df = pd.DataFrame({"item_id": ["solo"], "difficulty": [0.3]})
    drifted = compute_drift_flags(df)
    row = drifted.iloc[0]
    assert row["drift_flag"] is False
    assert row["drift_score"] == 0.0
