# docs/scripts/exporters/export_attention_network.py
# ABOUTME: Attention network exporter for force-directed graph
# ABOUTME: Extracts attention weights and formats as graph nodes/links

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

def export_attention_network_for_testing(
    attn_weights: torch.Tensor,
    tokens: List[str],
    threshold: float = 0.0
) -> Dict:
    """
    Format attention weights for network visualization.
    
    Args:
        attn_weights: Tensor of shape (batch, heads, seq, seq)
        tokens: List of token strings corresponding to sequence
        threshold: Minimum attention weight to include as link
        
    Returns:
        Dict with nodes and links
    """
    # Average over heads and batch
    if attn_weights.dim() == 4:
        # (batch, heads, seq, seq) -> (seq, seq)
        avg_attn = attn_weights.mean(dim=(0, 1))
    elif attn_weights.dim() == 3:
        # (heads, seq, seq) -> (seq, seq)
        avg_attn = attn_weights.mean(dim=0)
    else:
        avg_attn = attn_weights
        
    # Create nodes
    nodes = []
    for i, token in enumerate(tokens):
        nodes.append({
            'id': f"node_{i}",
            'label': token,
            'index': i,
            'group': 'query' if i == len(tokens)-1 else 'context'
        })
        
    # Create links
    links = []
    seq_len = len(tokens)
    
    # Ensure we don't go out of bounds if weights are smaller than tokens
    limit = min(seq_len, avg_attn.shape[0])
    
    for i in range(limit): # Target (Query)
        for j in range(limit): # Source (Key)
            weight = float(avg_attn[i, j])
            
            if weight > threshold:
                links.append({
                    'source': f"node_{j}",
                    'target': f"node_{i}",
                    'weight': weight,
                    'value': weight * 10 # Visual thickness
                })
                
    return {
        'nodes': nodes,
        'links': links,
        'metadata': {
            'num_tokens': len(tokens),
            'threshold': threshold
        }
    }

def export_attention_network(reports_dir: Path) -> Dict:
    """
    Export attention network from attention data.
    
    Args:
        reports_dir: Path to reports directory
        
    Returns:
        Dict for JSON export
    """
    # Try to find attention data
    attn_path = reports_dir / "attention_sample.json"
    
    if attn_path.exists():
        import json
        with open(attn_path) as f:
            data = json.load(f)
            # If already formatted, return
            if 'nodes' in data and 'links' in data:
                return data
            # If raw weights, process them (mocking this path for now)
            
    # Fallback: Mock data
    tokens = ['q1', 'q2', 'q3', 'q4', 'q5']
    weights = torch.rand(1, 1, 5, 5)
    # Make it sparse
    weights = weights * (weights > 0.7).float()
    # Normalize
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
    
    return export_attention_network_for_testing(weights, tokens, threshold=0.1)
