# docs/scripts/exporters/export_feature_importance.py
# ABOUTME: Feature importance exporter for Wide & Deep IRT model
# ABOUTME: Extracts and normalizes feature weights from PyTorch checkpoints

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

def compute_l1_norm(tensor: torch.Tensor) -> float:
    """Compute L1 norm as importance metric"""
    return float(torch.abs(tensor).sum())

def compute_feature_importance_from_checkpoint(checkpoint_path: Path) -> Dict:
    """
    Load PyTorch checkpoint and compute feature importance.

    Args:
        checkpoint_path: Path to .ckpt file

    Returns:
        Dict with feature importance data
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    return export_feature_importance_for_testing(state_dict)

def export_feature_importance_for_testing(state_dict: Dict[str, torch.Tensor]) -> Dict:
    """
    Compute feature importance from model weights.

    Args:
        state_dict: PyTorch state dict with model weights

    Returns:
        Dict with normalized feature importance per group
    """
    importance_map = {}

    # Group 1: Wide features (linear IRT parameters)
    wide_keys = [k for k in state_dict.keys() if 'wide' in k.lower()]
    if wide_keys:
        wide_importance = sum(compute_l1_norm(state_dict[k]) for k in wide_keys)
        importance_map['Wide Features (IRT)'] = {
            'importance': wide_importance,
            'features': ['difficulty', 'discrimination', 'user_id', 'item_id']
        }

    # Group 2: Deep embeddings
    embedding_keys = [k for k in state_dict.keys() if 'embedding' in k.lower()]
    if embedding_keys:
        embedding_importance = sum(compute_l1_norm(state_dict[k]) for k in embedding_keys)

        # Break down by embedding type
        user_emb_keys = [k for k in embedding_keys if 'user' in k.lower()]
        item_emb_keys = [k for k in embedding_keys if 'item' in k.lower()]
        skill_emb_keys = [k for k in embedding_keys if 'skill' in k.lower()]

        features = []
        if user_emb_keys:
            features.append('user_embeddings')
        if item_emb_keys:
            features.append('item_embeddings')
        if skill_emb_keys:
            features.append('skill_embeddings')

        importance_map['Deep Features (Embeddings)'] = {
            'importance': embedding_importance,
            'features': features or ['sequence_embeddings']
        }

    # Group 3: Deep network layers
    fc_keys = [k for k in state_dict.keys() if any(x in k.lower() for x in ['fc', 'dense', 'linear']) and 'wide' not in k.lower()]
    if fc_keys:
        fc_importance = sum(compute_l1_norm(state_dict[k]) for k in fc_keys)
        importance_map['Deep Features (Network)'] = {
            'importance': fc_importance,
            'features': ['hidden_layers', 'temporal_patterns', 'skill_interactions']
        }

    # Normalize to sum to 1.0
    total_importance = sum(v['importance'] for v in importance_map.values())

    data = []
    for group_name, group_data in importance_map.items():
        normalized_importance = group_data['importance'] / total_importance if total_importance > 0 else 0
        data.append({
            'feature_group': group_name,
            'importance': float(normalized_importance),
            'features': group_data['features']
        })

    # Sort by importance descending
    data.sort(key=lambda x: x['importance'], reverse=True)

    return {
        'data': data,
        'metadata': {
            'model': 'WD-IRT',
            'mock': False,
            'total_params': sum(p.numel() for p in state_dict.values()),
            'num_groups': len(data)
        }
    }

def export_feature_importance(reports_dir: Path) -> Dict:
    """
    Export feature importance from latest checkpoint.

    Args:
        reports_dir: Path to reports directory

    Returns:
        Dict for JSON export
    """
    checkpoint_dir = reports_dir / 'checkpoints' / 'wd_irt_edm'
    checkpoint_path = checkpoint_dir / 'latest.ckpt'

    if not checkpoint_path.exists():
        # Try to find any checkpoint
        checkpoint_files = list(checkpoint_dir.glob('*.ckpt'))
        if checkpoint_files:
            checkpoint_path = checkpoint_files[0]
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    return compute_feature_importance_from_checkpoint(checkpoint_path)
