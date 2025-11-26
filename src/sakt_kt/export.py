# ABOUTME: Exports student mastery states and predictions from trained SAKT models.
# ABOUTME: Produces parquet artifacts consumed by demos and downstream analytics.

from pathlib import Path


def export_student_state(checkpoint_path: Path, output_dir: Path) -> None:
    """
    Write student_state.parquet and next_correct_predictions.parquet.
    """

    raise NotImplementedError("Export logic will run once pyKT training is finished.")
