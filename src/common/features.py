# ABOUTME: Declares reusable feature builders operating on canonical events.
# ABOUTME: Provides placeholder functions that will feed both engines.

from typing import Iterable

import pandas as pd

from .schemas import LearningEvent


def build_clickstream_features(events: Iterable[LearningEvent]) -> pd.DataFrame:
    """
    Convert canonical learning events into aggregated clickstream features.

    The actual implementation will reproduce the feature groups defined in the
    Wide & Deep IRT paper (e.g., latency buckets, help request ratios).
    """

    raise NotImplementedError("Clickstream feature builder pending implementation.")
