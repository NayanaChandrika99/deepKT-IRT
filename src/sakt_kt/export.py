# ABOUTME: Exports student mastery states and predictions from trained SAKT models.
# ABOUTME: Produces parquet artifacts consumed by demos and downstream analytics.

from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import torch
import yaml

from .adapters import canonical_to_pykt_csv
from .attention_extractor import (
    AttentionExtractor,
    compute_attention_from_scratch,
    extract_top_influences,
)
from .datasets import PyKTDataset, load_data_config


def export_student_mastery(
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    extract_attention: bool = True,
) -> None:
    """
    Export student mastery and predictions from a trained SAKT model.
    
    Generates artifacts:
    1. sakt_student_state.parquet - Per-interaction mastery estimates
    2. sakt_predictions.parquet - Predicted vs actual correctness
    3. sakt_attention.parquet - Attention weights for explainability (optional)
    
    Args:
        checkpoint_path: Path to saved model checkpoint (.pt file)
        config_path: Path to training config YAML
        output_dir: Directory to write output files
        extract_attention: Whether to extract and save attention weights
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    print(f"[sakt-export] Loading config from {config_path}")
    
    # Load checkpoint
    print(f"[sakt-export] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    data_config = checkpoint["data_config"]
    model_cfg = checkpoint["model_config"]
    
    # Reconstruct model
    model = _load_model_from_checkpoint(checkpoint, model_cfg, data_config)
    model.eval()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sakt-export] Using device: {device}")
    model = model.to(device)
    
    # Load events for user/item mapping
    data_cfg = cfg["data"]
    events_path = Path(data_cfg["events_path"])
    pykt_dir = Path(data_cfg["pykt_dir"])
    
    print(f"[sakt-export] Loading events from {events_path}")
    events_df = pd.read_parquet(events_path)
    
    # Build vocabulary mappings (same as during training)
    unique_items = sorted(events_df["item_id"].unique())
    item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    # Load pyKT formatted data
    csv_path = pykt_dir / "train_valid_sequences.csv"
    if not csv_path.exists():
        print(f"[sakt-export] Regenerating pyKT data...")
        max_seq_len = model_cfg.get("seq_len", 200)
        csv_path, _ = canonical_to_pykt_csv(events_df, pykt_dir, max_seq_len=max_seq_len)
    
    # Run inference on all data (with optional attention extraction)
    print("[sakt-export] Running inference...")
    predictions_rows, mastery_rows, attention_rows = _run_inference_with_attention(
        model, csv_path, device, idx_to_item, events_df,
        data_config=data_config,
        extract_attention=extract_attention,
    )
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions_rows)
    predictions_path = output_dir / "sakt_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    print(f"✅ Exported {len(predictions_df)} predictions to {predictions_path}")
    
    # Compute and log AUC
    if len(predictions_df) > 0:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(predictions_df["actual"], predictions_df["predicted"])
        print(f"   Prediction AUC: {auc:.4f}")
    
    # Save student mastery
    mastery_df = pd.DataFrame(mastery_rows)
    mastery_path = output_dir / "sakt_student_state.parquet"
    mastery_df.to_parquet(mastery_path, index=False)
    print(f"✅ Exported {len(mastery_df)} mastery records to {mastery_path}")
    print(f"   Unique students: {mastery_df['user_id'].nunique()}")
    
    # Save attention data if extracted
    if attention_rows:
        attention_df = pd.DataFrame(attention_rows)
        attention_path = output_dir / "sakt_attention.parquet"
        attention_df.to_parquet(attention_path, index=False)
        print(f"✅ Exported {len(attention_df)} attention records to {attention_path}")
    elif extract_attention:
        print("⚠️ No attention weights captured (model may not expose attention)")
    
    # Generate summary report
    _generate_mastery_report(mastery_df, predictions_df, output_dir)


def _load_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    model_cfg: Dict[str, Any],
    data_config: Dict[str, Any],
):
    """Reconstruct SAKT model from checkpoint."""
    # Import SAKT directly to avoid pykt's init_model which has tkinter dependency issues
    from pykt.models.sakt import SAKT
    
    model = SAKT(
        num_c=data_config["num_c"],
        seq_len=model_cfg.get("seq_len", 200),
        emb_size=model_cfg.get("emb_size", 64),
        num_attn_heads=model_cfg.get("num_attn_heads", 4),
        dropout=model_cfg.get("dropout", 0.2),
        num_en=model_cfg.get("num_encoder_layers", 2),
        emb_type=model_cfg.get("emb_type", "qid"),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _build_shifted_query(qseqs: torch.Tensor) -> torch.Tensor:
    """Shift question sequences right by one with 0 padding."""
    batch_size, seq_len = qseqs.shape
    qry = torch.zeros_like(qseqs)
    qry[:, 1:] = qseqs[:, :-1]
    return qry


def _run_inference_with_attention(
    model,
    csv_path: Path,
    device: torch.device,
    idx_to_item: Dict[int, str],
    events_df: pd.DataFrame,
    data_config: Optional[Dict[str, Any]] = None,
    extract_attention: bool = True,
) -> tuple:
    """Run inference on all data, collect predictions, mastery, and attention."""
    predictions_rows = []
    mastery_rows = []
    attention_rows = []
    
    # Load full dataset (all folds)
    df = pd.read_csv(csv_path)
    
    # Setup attention extractor if requested
    extractor = AttentionExtractor(model) if extract_attention else None
    num_c = data_config.get("num_c", 1) if data_config else 1
    
    # Process each user
    with torch.no_grad():
        for idx, row in df.iterrows():
            user_id = row["uid"]
            questions = [int(x) for x in row["questions"].split(",")]
            responses = [int(x) for x in row["responses"].split(",")]
            
            # Convert to tensors
            qseqs = torch.tensor([questions], dtype=torch.long, device=device)
            rseqs = torch.tensor([responses], dtype=torch.long, device=device)
            qryseqs = _build_shifted_query(qseqs)
            
            # Get predictions (with or without attention extraction)
            attn_weights = None
            if extractor:
                y_pred, captured_attn = extractor.extract(qseqs, rseqs, qryseqs)
                y_pred = y_pred.squeeze(0).cpu().numpy()
                
                # Use captured attention or compute from scratch
                if captured_attn:
                    # Take first layer's attention, average over heads if multi-head
                    attn = captured_attn[0]
                    if attn.dim() == 4:  # [batch, heads, seq, seq]
                        attn = attn.mean(dim=1)  # Average over heads
                    attn_weights = attn.squeeze(0)  # [seq, seq]
                else:
                    # Try computing attention manually
                    computed = compute_attention_from_scratch(
                        model, qseqs.cpu(), rseqs.cpu(), num_c
                    )
                    if computed is not None:
                        attn_weights = computed.squeeze(0)
            else:
                y_pred = model(qseqs, rseqs, qryseqs)
                y_pred = y_pred.squeeze(0).cpu().numpy()
            
            # Build item_ids list for this sequence
            item_ids = [idx_to_item.get(q, f"item_{q}") for q in questions]
            
            # Collect predictions and mastery for non-padding positions
            for pos in range(len(questions)):
                q_idx = questions[pos]
                r = responses[pos]
                
                # Skip padding
                if q_idx == 0:
                    continue
                
                item_id = idx_to_item.get(q_idx, f"item_{q_idx}")
                
                # Prediction for position pos+1 (next response)
                if pos < len(questions) - 1 and questions[pos + 1] != 0:
                    next_r = responses[pos + 1]
                    pred = float(y_pred[pos])
                    
                    predictions_rows.append({
                        "user_id": user_id,
                        "item_id": idx_to_item.get(questions[pos + 1], f"item_{questions[pos + 1]}"),
                        "position": pos + 1,
                        "actual": next_r,
                        "predicted": pred,
                    })
                
                # Mastery estimate (prediction at this position)
                mastery_rows.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "position": pos,
                    "response": r,
                    "mastery": float(y_pred[pos]) if pos > 0 else 0.5,
                })
            
            # Extract top influences for the last valid position
            if attn_weights is not None and extract_attention:
                # Find last non-padding position
                last_pos = len(questions) - 1
                while last_pos > 0 and questions[last_pos] == 0:
                    last_pos -= 1
                
                if last_pos > 0:
                    top_influences = extract_top_influences(
                        attn_weights, item_ids, responses, last_pos, k=5
                    )
                    
                    attention_rows.append({
                        "user_id": user_id,
                        "position": last_pos,
                        "mastery": float(y_pred[last_pos]) if last_pos > 0 else 0.5,
                        "top_influences": top_influences,
                    })
    
    return predictions_rows, mastery_rows, attention_rows


def _generate_mastery_report(
    mastery_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate markdown summary of student mastery analysis."""
    report_path = output_dir / "sakt_mastery_report.md"
    
    with open(report_path, "w") as f:
        f.write("# SAKT Student Mastery Report\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- Total students: {mastery_df['user_id'].nunique()}\n")
        f.write(f"- Total interactions: {len(mastery_df)}\n")
        f.write(f"- Total predictions: {len(predictions_df)}\n\n")
        
        if len(predictions_df) > 0:
            from sklearn.metrics import roc_auc_score, accuracy_score
            
            auc = roc_auc_score(predictions_df["actual"], predictions_df["predicted"])
            acc = accuracy_score(predictions_df["actual"], (predictions_df["predicted"] > 0.5).astype(int))
            
            f.write("## Prediction Performance\n\n")
            f.write(f"- AUC: {auc:.4f}\n")
            f.write(f"- Accuracy: {acc:.4f}\n\n")
        
        f.write("## Mastery Distribution\n\n")
        f.write(f"- Mean mastery: {mastery_df['mastery'].mean():.4f}\n")
        f.write(f"- Std mastery: {mastery_df['mastery'].std():.4f}\n")
        f.write(f"- Min mastery: {mastery_df['mastery'].min():.4f}\n")
        f.write(f"- Max mastery: {mastery_df['mastery'].max():.4f}\n\n")
        
        f.write("## Response Distribution\n\n")
        response_counts = mastery_df["response"].value_counts()
        total = len(mastery_df)
        for resp, count in response_counts.items():
            pct = 100 * count / total
            label = "Correct" if resp == 1 else "Incorrect"
            f.write(f"- {label}: {count} ({pct:.1f}%)\n")
        f.write("\n")
        
        # Top students by average mastery
        f.write("## Top 10 Students by Average Mastery\n\n")
        student_mastery = mastery_df.groupby("user_id")["mastery"].mean().sort_values(ascending=False)
        f.write("| User ID | Avg Mastery | Interactions |\n")
        f.write("|---------|-------------|---------------|\n")
        for user_id in student_mastery.head(10).index:
            avg = student_mastery[user_id]
            count = len(mastery_df[mastery_df["user_id"] == user_id])
            f.write(f"| {user_id} | {avg:.4f} | {count} |\n")
        f.write("\n")
        
        # Students needing support (low mastery)
        f.write("## Students with Low Average Mastery (Bottom 10)\n\n")
        f.write("| User ID | Avg Mastery | Interactions |\n")
        f.write("|---------|-------------|---------------|\n")
        for user_id in student_mastery.tail(10).index:
            avg = student_mastery[user_id]
            count = len(mastery_df[mastery_df["user_id"] == user_id])
            f.write(f"| {user_id} | {avg:.4f} | {count} |\n")
    
    print(f"✅ Generated mastery report: {report_path}")
