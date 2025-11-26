# ABOUTME: PyTorch Dataset and DataLoader utilities for SAKT training via pyKT.
# ABOUTME: Loads pyKT-formatted CSV and produces tensors for (questions, responses, query, mask).

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class PyKTDataset(Dataset):
    """
    Dataset that loads pyKT-formatted CSV and returns tensors for SAKT.
    
    SAKT expects three inputs:
    - qseqs: Question IDs (1-indexed, 0 = padding)
    - rseqs: Response values (0/1, with 0 also used for padding)
    - qryseqs: Shifted question sequence (prepend 0, drop last) for attention query
    
    The mask indicates valid (non-padding) positions.
    """
    
    def __init__(
        self,
        csv_path: Path,
        fold: int = 0,
        is_train: bool = True,
    ):
        """
        Initialize dataset from pyKT CSV.
        
        Args:
            csv_path: Path to train_valid_sequences.csv
            fold: Which fold to hold out for validation
            is_train: If True, use all folds except `fold`; if False, use only `fold`
        """
        df = pd.read_csv(csv_path)
        
        if is_train:
            self.data = df[df["fold"] != fold].reset_index(drop=True)
        else:
            self.data = df[df["fold"] == fold].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        questions = [int(x) for x in row["questions"].split(",")]
        responses = [int(x) for x in row["responses"].split(",")]
        
        qseqs = torch.tensor(questions, dtype=torch.long)
        rseqs = torch.tensor(responses, dtype=torch.long)
        
        # Build shifted query sequence for SAKT attention
        qryseqs = build_shifted_query(qseqs)
        
        # Mask: 1 for valid positions, 0 for padding
        # Questions are 1-indexed, so q=0 means padding
        masks = (qseqs != 0).long()
        
        return {
            "qseqs": qseqs,
            "rseqs": rseqs,
            "qryseqs": qryseqs,
            "masks": masks,
        }


def build_shifted_query(qseqs: torch.Tensor) -> torch.Tensor:
    """
    Create shifted query sequence for SAKT attention mechanism.
    
    SAKT's attention uses a shifted version of the question sequence:
    - Prepend 0 (padding) at the start
    - Drop the last element
    
    This allows the model to attend to past interactions when predicting
    the current question's outcome.
    
    Args:
        qseqs: Question sequence tensor of shape (seq_len,)
    
    Returns:
        Shifted tensor of same shape with 0 prepended and last element dropped
    """
    # Prepend 0 and remove last element
    shifted = torch.cat([torch.tensor([0], dtype=qseqs.dtype), qseqs[:-1]])
    return shifted


def prepare_dataloaders(
    csv_path: Path,
    batch_size: int = 64,
    fold: int = 0,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from pyKT CSV.
    
    Args:
        csv_path: Path to train_valid_sequences.csv
        batch_size: Batch size for both loaders
        fold: Which fold to use for validation
        num_workers: Number of data loading workers
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    train_dataset = PyKTDataset(csv_path, fold=fold, is_train=True)
    val_dataset = PyKTDataset(csv_path, fold=fold, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader


def load_data_config(config_path: Path) -> Dict[str, Any]:
    """
    Load the data_config.json file.
    
    Args:
        config_path: Path to data_config.json
    
    Returns:
        Dict with num_q, num_c, emb_path, etc.
    """
    import json
    with open(config_path) as f:
        return json.load(f)
