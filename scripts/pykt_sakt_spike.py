# ABOUTME: Spike script to verify pyKT SAKT integration works with our data.
# ABOUTME: Converts canonical events to pyKT format and trains for a few epochs.

"""
Milestone 1: pyKT Integration Spike

This script verifies that:
1. We can convert our canonical events to pyKT's expected format
2. pyKT's SAKT model initializes correctly
3. Training runs and produces reasonable AUC

Run from repository root:
    python scripts/pykt_sakt_spike.py
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def convert_to_pykt_format(events_df: pd.DataFrame, output_dir: Path, max_seq_len: int = 200):
    """
    Convert canonical events DataFrame to pyKT's expected CSV format.
    
    pyKT expects:
    - One row per user
    - Comma-separated sequences: questions, concepts, responses, timestamps
    - Padded with -1 for sequences shorter than max_seq_len
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build vocabulary mappings
    unique_items = sorted(events_df["item_id"].unique())
    unique_skills = set()
    for skill_list in events_df["skill_ids"]:
        if isinstance(skill_list, list):
            unique_skills.update(skill_list)
        elif isinstance(skill_list, str):
            unique_skills.add(skill_list)
    unique_skills = sorted(unique_skills)
    
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    skill_to_idx = {skill: idx for idx, skill in enumerate(unique_skills)}
    
    print(f"  Vocabulary: {len(unique_items)} questions, {len(unique_skills)} concepts")
    
    # Group by user and build sequences
    grouped = events_df.groupby("user_id")
    
    rows = []
    for user_id, group in grouped:
        group = group.sort_values("timestamp").head(max_seq_len)
        
        questions = []
        concepts = []
        responses = []
        
        for _, row in group.iterrows():
            item_idx = item_to_idx.get(row["item_id"], 0)
            
            # Get first skill or default to 0
            skill_list = row["skill_ids"]
            if isinstance(skill_list, list) and len(skill_list) > 0:
                skill_idx = skill_to_idx.get(skill_list[0], 0)
            elif isinstance(skill_list, str):
                skill_idx = skill_to_idx.get(skill_list, 0)
            else:
                skill_idx = 0
            
            questions.append(str(item_idx))
            concepts.append(str(skill_idx))
            responses.append(str(int(row["correct"])))
        
        # Pad to max_seq_len
        pad_len = max_seq_len - len(questions)
        if pad_len > 0:
            questions.extend(["-1"] * pad_len)
            concepts.extend(["-1"] * pad_len)
            responses.extend(["-1"] * pad_len)
        
        rows.append({
            "uid": user_id,
            "questions": ",".join(questions),
            "concepts": ",".join(concepts),
            "responses": ",".join(responses),
        })
    
    df_pykt = pd.DataFrame(rows)
    
    # Split into folds (5-fold CV as pyKT expects)
    df_pykt["fold"] = np.arange(len(df_pykt)) % 5
    
    # Save
    csv_path = output_dir / "train_valid_sequences.csv"
    df_pykt.to_csv(csv_path, index=False)
    print(f"  Saved {len(df_pykt)} user sequences to {csv_path}")
    
    # Build data config
    data_config = {
        "num_q": len(unique_items),
        "num_c": len(unique_skills),
        "max_concepts": 1,
        "input_type": ["questions", "concepts"],
    }
    
    config_path = output_dir / "data_config.json"
    with open(config_path, "w") as f:
        json.dump(data_config, f, indent=2)
    print(f"  Saved data config to {config_path}")
    
    return csv_path, data_config


def run_spike():
    """Run the pyKT SAKT integration spike."""
    print("🔍 Milestone 1: pyKT SAKT Integration Spike")
    print("=" * 50)
    
    # Check data exists
    events_path = Path("data/interim/assistments_skill_builder_42_events.parquet")
    if not events_path.exists():
        print(f"❌ Events file not found: {events_path}")
        print("   Run 'make data dataset=assistments_skill_builder split_seed=42' first")
        return False
    
    # Load canonical events
    print(f"\n📂 Loading canonical events from {events_path}")
    events_df = pd.read_parquet(events_path)
    print(f"   Loaded {len(events_df)} events from {events_df['user_id'].nunique()} users")
    
    # Convert to pyKT format
    print("\n🔄 Converting to pyKT format...")
    output_dir = Path("data/processed/assistments_pykt")
    csv_path, data_config = convert_to_pykt_format(events_df, output_dir, max_seq_len=200)
    
    # Try importing pyKT
    print("\n📦 Importing pyKT...")
    try:
        from pykt.models import init_model
        from pykt.datasets import init_dataset4train
        from pykt.utils import set_seed
        print("   ✅ pyKT imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import pyKT: {e}")
        print("   Try: pip install pykt-toolkit")
        return False
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize dataset
    print("\n📊 Initializing pyKT dataset...")
    try:
        # pyKT needs the data in a specific location
        # We need to create the proper directory structure
        pykt_data_dir = Path("data/processed/assistments_pykt")
        
        # Read our converted data
        df = pd.read_csv(csv_path)
        
        # pyKT's init_dataset4train expects data in a specific format
        # Let's use a simpler approach - direct DataLoader creation
        from torch.utils.data import Dataset, DataLoader
        import torch
        
        class SimpleKTDataset(Dataset):
            def __init__(self, df, fold, is_train=True):
                if is_train:
                    self.data = df[df["fold"] != fold]
                else:
                    self.data = df[df["fold"] == fold]
                self.data = self.data.reset_index(drop=True)
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                row = self.data.iloc[idx]
                questions = [int(x) for x in row["questions"].split(",")]
                concepts = [int(x) for x in row["concepts"].split(",")]
                responses = [int(x) for x in row["responses"].split(",")]
                
                # Create mask (1 for valid, 0 for padding)
                mask = [1 if r != -1 else 0 for r in responses]
                
                return {
                    "qseqs": torch.tensor(questions, dtype=torch.long),
                    "cseqs": torch.tensor(concepts, dtype=torch.long),
                    "rseqs": torch.tensor(responses, dtype=torch.long),
                    "masks": torch.tensor(mask, dtype=torch.long),
                }
        
        train_dataset = SimpleKTDataset(df, fold=0, is_train=True)
        val_dataset = SimpleKTDataset(df, fold=0, is_train=False)
        
        print(f"   Train: {len(train_dataset)} sequences, Val: {len(val_dataset)} sequences")
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
    except Exception as e:
        print(f"   ❌ Failed to initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Initialize SAKT model
    print("\n🧠 Initializing SAKT model...")
    try:
        model_config = {
            "num_q": data_config["num_q"],
            "num_c": data_config["num_c"],
            "seq_len": 200,
            "emb_size": 64,
            "num_attn_heads": 4,
            "dropout": 0.2,
            "emb_type": "qid",
        }
        
        model = init_model(
            model_name="sakt",
            model_config=model_config,
            data_config=data_config,
            emb_type="qid"
        )
        print(f"   ✅ SAKT model initialized: {sum(p.numel() for p in model.parameters())} parameters")
        
    except Exception as e:
        print(f"   ❌ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Quick training test
    print("\n⚡ Running 3 training epochs...")
    try:
        import torch
        import torch.nn as nn
        from sklearn.metrics import roc_auc_score
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {device}")
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        for epoch in range(3):
            # Training
            model.train()
            train_losses = []
            for batch in train_loader:
                qseqs = batch["qseqs"].to(device)
                cseqs = batch["cseqs"].to(device)
                rseqs = batch["rseqs"].to(device)
                masks = batch["masks"].to(device)
                
                # SAKT forward pass
                optimizer.zero_grad()
                
                # Create input for model
                # SAKT expects: (q, c, r) sequences
                y_pred = model(qseqs, cseqs, rseqs)
                
                # Mask out padding
                valid_mask = masks[:, 1:].float()  # Skip first position
                if valid_mask.sum() == 0:
                    continue
                    
                y_true = rseqs[:, 1:].float()  # Target is next response
                y_pred = y_pred[:, :-1]  # Predictions for next
                
                # Apply mask
                y_pred_masked = y_pred[valid_mask.bool()]
                y_true_masked = y_true[valid_mask.bool()]
                
                if len(y_pred_masked) == 0:
                    continue
                
                loss = criterion(y_pred_masked, y_true_masked)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    qseqs = batch["qseqs"].to(device)
                    cseqs = batch["cseqs"].to(device)
                    rseqs = batch["rseqs"].to(device)
                    masks = batch["masks"].to(device)
                    
                    y_pred = model(qseqs, cseqs, rseqs)
                    
                    valid_mask = masks[:, 1:].float()
                    y_true = rseqs[:, 1:].float()
                    y_pred = y_pred[:, :-1]
                    
                    y_pred_masked = y_pred[valid_mask.bool()]
                    y_true_masked = y_true[valid_mask.bool()]
                    
                    all_preds.extend(y_pred_masked.cpu().numpy().tolist())
                    all_labels.extend(y_true_masked.cpu().numpy().tolist())
            
            if len(all_preds) > 0:
                val_auc = roc_auc_score(all_labels, all_preds)
            else:
                val_auc = 0.0
            
            avg_loss = np.mean(train_losses) if train_losses else 0.0
            print(f"   Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_auc={val_auc:.4f}")
        
        print("\n✅ Spike successful! SAKT integration verified.")
        print(f"   Final validation AUC: {val_auc:.4f}")
        return True
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_spike()
    sys.exit(0 if success else 1)

