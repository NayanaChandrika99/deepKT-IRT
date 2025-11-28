# ABOUTME: Captures attention weights from SAKT or compatible attention models.
# ABOUTME: Provides hook-based extraction and utilities for interpretability.

from __future__ import annotations

from typing import List, Tuple

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
