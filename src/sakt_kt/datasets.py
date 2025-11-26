# ABOUTME: Prepares sequences and metadata for SAKT training via pyKT.
# ABOUTME: Converts canonical events into the format expected by pyKT toolkit.

from pathlib import Path
from typing import Iterable

from src.common.schemas import LearningEvent


def load_assistments_sequences(events_path: Path) -> Iterable[LearningEvent]:
    """
    Load ASSISTments2009 events normalized to the canonical schema.
    """

    raise NotImplementedError("ASSISTments adapter will integrate with preprocessing scripts.")
