# ABOUTME: Defines evaluation helpers shared by both engines.
# ABOUTME: Will compute metrics like AUC, calibration, and subgroup slices.

from typing import Iterable, Mapping

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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

    if predictions is None or len(predictions) == 0:
        return {metric: np.nan for metric in metrics}

    y_true = predictions["y_true"].astype(float)
    y_pred = predictions["y_pred"].astype(float).clip(0.0, 1.0)

    results = {}
    for metric in metrics:
        if metric == "auc":
            # roc_auc_score requires both classes; return 0.0 when degenerate.
            if len(np.unique(y_true)) < 2:
                results[metric] = 0.0
            else:
                results[metric] = float(roc_auc_score(y_true, y_pred))
        elif metric == "average_precision":
            results[metric] = float(average_precision_score(y_true, y_pred))
        elif metric == "calibration_ece":
            results[metric] = float(_expected_calibration_error(y_true, y_pred))
        else:
            raise ValueError(f"Unsupported metric '{metric}'.")

    return results


def _expected_calibration_error(y_true: pd.Series, y_pred: pd.Series, num_bins: int = 10) -> float:
    """
    Compute expected calibration error using equal-width bins between 0 and 1.
    """

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    digitized = np.digitize(y_pred, bins) - 1  # bins start at 0
    total = len(y_true)
    if total == 0:
        return np.nan

    ece = 0.0
    for b in range(num_bins):
        mask = digitized == b
        count = mask.sum()
        if count == 0:
            continue
        bin_true = y_true[mask]
        bin_pred = y_pred[mask]
        acc = bin_true.mean()
        conf = bin_pred.mean()
        ece += (count / total) * abs(acc - conf)
    return ece
