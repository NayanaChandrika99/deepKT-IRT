# ABOUTME: Defines evaluation helpers shared by both engines.
# ABOUTME: Will compute metrics like AUC, calibration, and subgroup slices.

from typing import Iterable, Mapping

import pandas as pd


def evaluate_predictions(predictions: pd.DataFrame, metrics: Iterable[str]) -> Mapping[str, float]:
    """
    Evaluate predictions dataframe using the requested metric names.

    Parameters
    ----------
    predictions : pd.DataFrame
        Expected columns: ['y_true', 'y_pred', 'user_id', 'item_id'] plus optional metadata.
    metrics : Iterable[str]
        Metric identifiers such as 'auc', 'average_precision', 'calibration_ece'.
    """

    raise NotImplementedError("Evaluation helpers will be implemented alongside the training code.")
