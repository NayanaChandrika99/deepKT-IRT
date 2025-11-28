# ABOUTME: Captures attention weights from SAKT or compatible attention models.
# ABOUTME: Provides hook-based extraction and utilities for interpretability.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


class AttentionExtractor:
    """
    Capture attention weights during a forward pass using forward hooks.

    Works with modules that either return (output, attn_weights) tuples (e.g.,
    nn.MultiheadAttention) or expose "attn"/"attention" in their names.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights: List[torch.Tensor] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _attention_hook(self, module, inputs, output):
        # Many attention layers return (out, weights); capture weights when present.
        if isinstance(output, tuple) and len(output) > 1:
            weights = output[1]
            if isinstance(weights, torch.Tensor):
                self.attention_weights.append(weights.detach().cpu())
        elif isinstance(output, torch.Tensor):
            # No weights provided; skip.
            return

    def _find_attention_layers(self) -> List[nn.Module]:
        layers: List[nn.Module] = []
        for name, module in self.model.named_modules():
            lower = name.lower()
            if "attn" in lower or "attention" in lower or isinstance(module, nn.MultiheadAttention):
                layers.append(module)
        return layers

    def register_hooks(self) -> int:
        self.attention_weights = []
        for layer in self._find_attention_layers():
            self._hooks.append(layer.register_forward_hook(self._attention_hook))
        return len(self._hooks)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def extract(
        self, q_seq: torch.Tensor, r_seq: torch.Tensor, qry_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run the model while capturing attention weights.
        """
        self.register_hooks()
        try:
            with torch.no_grad():
                output = self.model(q_seq, r_seq, qry_seq)
        finally:
            self.remove_hooks()

        predictions = output[0] if isinstance(output, tuple) else output
        return predictions, self.attention_weights


def compute_attention_from_scratch(
    model: nn.Module,
    q_seq: torch.Tensor,
    r_seq: torch.Tensor,
    num_c: int,
    d_k: int = 64,
) -> Optional[torch.Tensor]:
    """
    Manually compute attention weights when hooks don't capture them.

    Replicates SAKT's scaled dot-product attention:
    1. Query: current exercise embedding
    2. Key/Value: past interaction embeddings

    Args:
        model: SAKT model with embedding layers
        q_seq: Question sequence [batch, seq_len]
        r_seq: Response sequence [batch, seq_len]
        num_c: Number of concepts (for interaction encoding)
        d_k: Key dimension for scaling

    Returns:
        attention_weights: [batch, seq_len, seq_len] or None if embeddings not found
    """
    with torch.no_grad():
        # Try to find embedding layers
        exercise_emb = None
        interaction_emb = None

        for name, module in model.named_modules():
            lower = name.lower()
            if "exercise" in lower and "emb" in lower:
                exercise_emb = module
            elif "interaction" in lower and "emb" in lower:
                interaction_emb = module

        if exercise_emb is None or interaction_emb is None:
            return None

        # Query: current question embeddings
        qry_seq = torch.zeros_like(q_seq)
        qry_seq[:, 1:] = q_seq[:, :-1]
        q_emb = exercise_emb(qry_seq)

        # Key/Value: interaction embeddings (question + response encoding)
        interaction_idx = q_seq + r_seq * num_c
        k_emb = interaction_emb(interaction_idx)

        # Scaled dot-product attention
        scores = torch.matmul(q_emb, k_emb.transpose(-2, -1))
        scores = scores / np.sqrt(d_k)

        # Apply causal mask (can't attend to future)
        seq_len = q_seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q_seq.device), diagonal=1).bool()
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))

        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        return attention_weights


def extract_top_influences(
    attention_weights: torch.Tensor,
    item_ids: List[str],
    responses: List[int],
    position: int,
    k: int = 5,
) -> List[Dict]:
    """
    Get top-k most influential past interactions for a prediction.

    Args:
        attention_weights: [seq_len, seq_len] attention matrix (single sample)
        item_ids: List of item IDs in sequence
        responses: List of responses (0/1) in sequence
        position: Position of prediction (which row to use)
        k: Number of top influences to return

    Returns:
        List of dicts with item_id, correct, weight, position
    """
    if position <= 0:
        return []

    # Get attention weights for this prediction (row = position)
    # Can only attend to positions 0..position-1
    weights = attention_weights[position, :position]

    if len(weights) == 0:
        return []

    # Convert to numpy for easier processing
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()

    # Get top-k indices by weight
    top_k = min(k, len(weights))
    top_indices = np.argsort(weights)[-top_k:][::-1]

    influences = []
    for idx in top_indices:
        if idx < len(item_ids) and idx < len(responses):
            influences.append({
                "item_id": item_ids[idx],
                "correct": bool(responses[idx]),
                "weight": float(weights[idx]),
                "position": int(idx),
            })

    return influences


def aggregate_attention_for_user(
    attention_weights_list: List[torch.Tensor],
    item_ids_list: List[List[str]],
    responses_list: List[List[int]],
    top_k: int = 5,
) -> List[Dict]:
    """
    Aggregate attention data across multiple sequences for a user.

    For each sequence, extracts top influences for the final prediction.

    Args:
        attention_weights_list: List of [seq_len, seq_len] attention matrices
        item_ids_list: List of item ID sequences
        responses_list: List of response sequences
        top_k: Number of top influences per sequence

    Returns:
        List of dicts with user-level attention summary
    """
    all_influences = []

    for attn, items, responses in zip(
        attention_weights_list, item_ids_list, responses_list
    ):
        if attn is None or len(items) == 0:
            continue

        # Get attention for the last valid position
        seq_len = len(items)
        last_pos = seq_len - 1

        # Find last non-padding position
        while last_pos > 0 and items[last_pos] == "":
            last_pos -= 1

        if last_pos > 0:
            influences = extract_top_influences(
                attn, items, responses, last_pos, k=top_k
            )
            all_influences.extend(influences)

    return all_influences
