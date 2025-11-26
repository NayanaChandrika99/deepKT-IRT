# ABOUTME: Bridges canonical events to pyKT's dataset and trainer interfaces.
# ABOUTME: Outlines hooks for injecting custom preprocessing into pyKT.

from pathlib import Path
from typing import Any, Dict


def build_pykt_config(config_path: Path) -> Dict[str, Any]:
    """
    Parse the YAML config and shape a dictionary consumable by pyKT.
    """

    raise NotImplementedError("pyKT config adapter pending integration work.")
