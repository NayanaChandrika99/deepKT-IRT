# ABOUTME: Provides the Typer CLI entrypoint for training SAKT via pyKT.
# ABOUTME: Loads configs, builds datasets, and kicks off the pyKT trainer.

from pathlib import Path

import typer

from .adapters import build_pykt_config

app = typer.Typer(help="Train the SAKT engine using pyKT.")


@app.command()
def train(config: Path = typer.Option(..., "--config", help="Path to sakt config YAML.")) -> None:
    raise NotImplementedError("SAKT trainer will load pyKT modules.")


def train_sakt(config_path: Path) -> None:
    """
    Programmatic entrypoint mirrored by the CLI command.
    """

    raise NotImplementedError("Programmatic trainer pending pyKT wiring.")


if __name__ == "__main__":
    app()
