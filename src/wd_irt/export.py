# ABOUTME: Handles exporting item health artifacts from trained WD-IRT models.
# ABOUTME: Writes parquet and markdown outputs consumed by reports and demos.

from pathlib import Path


def export_item_health(model_checkpoint: Path, output_dir: Path) -> None:
    """
    Export item parameters, drift scores, and behavior slices to the reports directory.
    """

    raise NotImplementedError("Exporter will run after model training is wired.")
