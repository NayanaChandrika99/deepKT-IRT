# ABOUTME: Provides the Typer CLI entrypoint for training SAKT via pyKT.
# ABOUTME: Loads configs, builds datasets, and runs the training loop.

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import typer
import yaml
from sklearn.metrics import roc_auc_score

from .adapters import canonical_to_pykt_csv
from .datasets import prepare_dataloaders, load_data_config

app = typer.Typer(help="Train the SAKT engine using pyKT.")


@app.command()
def train(
    config: Path = typer.Option(..., "--config", help="Path to sakt config YAML."),
) -> None:
    """Train SAKT model from config file."""
    train_sakt(config)


def train_sakt(config_path: Path) -> Dict[str, float]:
    """
    Train SAKT model from config YAML.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dict with final metrics (train_loss, val_auc)
    """
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    print(f"[sakt] Loading config from {config_path}")
    
    # Set seed for reproducibility
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    
    # Prepare data
    data_cfg = cfg["data"]
    events_path = Path(data_cfg["events_path"])
    pykt_dir = Path(data_cfg["pykt_dir"])
    
    print(f"[sakt] Preparing dataset from {events_path}")
    events_df = pd.read_parquet(events_path)
    
    model_cfg = cfg["model"]
    max_seq_len = model_cfg.get("seq_len", 200)
    
    # Convert to pyKT format
    csv_path, data_config = canonical_to_pykt_csv(
        events_df, pykt_dir, max_seq_len=max_seq_len
    )
    print(f"[sakt] Data prepared: {data_config['num_q']} questions, {data_config['num_c']} concepts")
    
    # Create dataloaders
    training_cfg = cfg["training"]
    batch_size = training_cfg.get("batch_size", 64)
    val_fold = cfg.get("evaluation", {}).get("val_fold", 0)
    
    train_loader, val_loader = prepare_dataloaders(
        csv_path, batch_size=batch_size, fold=val_fold
    )
    print(f"[sakt] Train: {len(train_loader.dataset)} sequences, Val: {len(val_loader.dataset)} sequences")
    
    # Initialize model
    print("[sakt] Initializing SAKT model...")
    model = _init_sakt_model(model_cfg, data_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[sakt] Model initialized: {param_count:,} parameters")
    
    # Setup device
    accelerator = training_cfg.get("accelerator", "auto")
    if accelerator == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif accelerator == "gpu" or accelerator == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[sakt] Using device: {device}")
    model = model.to(device)
    
    # Setup optimizer
    lr = training_cfg.get("learning_rate", 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Training loop
    max_epochs = training_cfg.get("max_epochs", 30)
    early_stopping_patience = training_cfg.get("early_stopping_patience", 5)
    
    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    
    outputs_cfg = cfg.get("outputs", {})
    checkpoint_dir = Path(outputs_cfg.get("checkpoints_dir", "reports/checkpoints/sakt"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(outputs_cfg.get("metrics_dir", "reports/metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[sakt] Training for up to {max_epochs} epochs...")
    
    for epoch in range(max_epochs):
        # Training phase
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation phase
        val_auc = _validate_epoch(model, val_loader, device)
        
        print(f"[sakt] Epoch {epoch+1}/{max_epochs}: train_loss={train_loss:.4f}, val_auc={val_auc:.4f}")
        
        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            # Save best checkpoint
            checkpoint_path = checkpoint_dir / f"{cfg['run_name']}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "data_config": data_config,
                "model_config": model_cfg,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"[sakt] Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                break
    
    print(f"[sakt] Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    print(f"[sakt] Checkpoint saved to {checkpoint_dir}")
    
    # Save final metrics
    metrics = {
        "train_loss": train_loss,
        "val_auc": val_auc,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
    }
    metrics_path = metrics_dir / f"{cfg['run_name']}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[sakt] Metrics saved to {metrics_path}")
    
    return metrics


def _init_sakt_model(model_cfg: Dict[str, Any], data_config: Dict[str, Any]):
    """Initialize SAKT model using pyKT's init_model."""
    from pykt.models import init_model
    
    # Build model config for pyKT
    pykt_model_config = {
        "seq_len": model_cfg.get("seq_len", 200),
        "emb_size": model_cfg.get("emb_size", 64),
        "num_attn_heads": model_cfg.get("num_attn_heads", 4),
        "dropout": model_cfg.get("dropout", 0.2),
    }
    
    emb_type = model_cfg.get("emb_type", "qid")
    
    model = init_model(
        model_name="sakt",
        model_config=pykt_model_config,
        data_config=data_config,
        emb_type=emb_type,
    )
    
    return model


def _build_shifted_query(qseqs: torch.Tensor) -> torch.Tensor:
    """
    Shift question sequences right by one with 0 padding.
    
    SAKT attention uses shifted questions as the query.
    """
    batch_size, seq_len = qseqs.shape
    qry = torch.zeros_like(qseqs)
    qry[:, 1:] = qseqs[:, :-1]
    return qry


def _train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        qseqs = batch["qseqs"].to(device)
        rseqs = batch["rseqs"].to(device)
        masks = batch["masks"].to(device)
        qryseqs = _build_shifted_query(qseqs).to(device)
        
        optimizer.zero_grad()
        
        # SAKT forward: (questions, responses, query)
        y_pred = model(qseqs, rseqs, qryseqs)
        
        # Mask out padding and align predictions with targets
        valid_mask = masks[:, 1:].float()
        if valid_mask.sum() == 0:
            continue
        
        y_true = rseqs[:, 1:].float()
        y_pred = y_pred[:, :-1]
        
        y_pred_masked = y_pred[valid_mask.bool()]
        y_true_masked = y_true[valid_mask.bool()]
        
        if len(y_pred_masked) == 0:
            continue
        
        loss = criterion(y_pred_masked, y_true_masked)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def _validate_epoch(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> float:
    """Run validation, return AUC."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            qseqs = batch["qseqs"].to(device)
            rseqs = batch["rseqs"].to(device)
            masks = batch["masks"].to(device)
            qryseqs = _build_shifted_query(qseqs).to(device)
            
            y_pred = model(qseqs, rseqs, qryseqs)
            
            valid_mask = masks[:, 1:].float()
            y_true = rseqs[:, 1:].float()
            y_pred = y_pred[:, :-1]
            
            y_pred_masked = y_pred[valid_mask.bool()]
            y_true_masked = y_true[valid_mask.bool()]
            
            all_preds.extend(y_pred_masked.cpu().numpy().tolist())
            all_labels.extend(y_true_masked.cpu().numpy().tolist())
    
    if len(all_preds) > 0 and len(set(all_labels)) > 1:
        return roc_auc_score(all_labels, all_preds)
    return 0.0


if __name__ == "__main__":
    app()
