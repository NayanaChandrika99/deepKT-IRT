# ABOUTME: Provides the CLI entrypoint for training the Wide & Deep IRT pipeline.
# ABOUTME: Loads configs, prepares data, and invokes the model fit routine.

from pathlib import Path
import json

import pytorch_lightning as pl
import torch
import typer
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .datasets import EdmClickstreamDataset, EdmDatasetPaths, collate_wdirt_batch
from .export import export_item_health
from .features import FeatureConfig
from .model import WideDeepConfig, WideDeepIrtModule

app = typer.Typer(help="Train the Wide & Deep IRT engine.")


@app.command()
def train(config: Path = typer.Option(..., "--config", help="Path to wd_irt config YAML.")) -> None:
    train_model(config)


@app.command()
def export(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to model checkpoint (.ckpt file)."),
    config: Path = typer.Option(..., "--config", help="Path to wd_irt config YAML."),
    output_dir: Path = typer.Option(Path("reports"), "--output-dir", help="Directory to write exports."),
) -> None:
    """Export item parameters and health artifacts from a trained model."""
    export_item_health(checkpoint, config, output_dir)


def train_model(config_path: Path) -> None:
    """Programmatic entrypoint mirrored by the Typer CLI."""

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    feature_config = FeatureConfig(**data_cfg["feature_config"])
    paths = EdmDatasetPaths(
        events=Path(data_cfg["events_path"]),
        assignment_details=Path(data_cfg["assignment_details"]),
        assignment_relationships=Path(data_cfg["assignment_relationships"]),
        unit_test_scores=Path(data_cfg["unit_test_scores"]),
        problem_details=Path(data_cfg["problem_details"]),
        split_manifest=Path(data_cfg["split_manifest"]),
    )

    train_dataset = EdmClickstreamDataset(
        split="train",
        paths=paths,
        feature_config=feature_config,
        max_samples=data_cfg.get("max_train_samples"),
    )
    val_dataset = EdmClickstreamDataset(
        split="val",
        paths=paths,
        feature_config=feature_config,
        max_samples=data_cfg.get("max_val_samples"),
    )

    trainer_cfg = cfg.get("trainer", {})
    batch_size = trainer_cfg.get("batch_size", 256)
    num_workers = trainer_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_wdirt_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_wdirt_batch,
        pin_memory=True,
    )

    model_config = WideDeepConfig(**cfg["model"])
    model = WideDeepIrtModule(
        config=model_config,
        feature_config=feature_config,
        item_vocab_size=train_dataset.item_vocab_size,
        action_vocab_size=5,
        latency_bucket_count=feature_config.latency_bins + 1,
    )

    outputs_cfg = cfg.get("outputs", {})
    checkpoint_dir = Path(outputs_cfg.get("checkpoints_dir", "reports/checkpoints/wd_irt"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(outputs_cfg.get("metrics_dir", "reports/metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{cfg['run_name']}" + "-{epoch:02d}-{val_auc:.4f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 5),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", 32),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
        callbacks=[checkpoint_callback],
        default_root_dir=str(checkpoint_dir),
    )

    trainer.fit(model, train_loader, val_loader)

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if isinstance(v, (float, torch.Tensor))}
    metrics_path = metrics_dir / f"{cfg['run_name']}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()
