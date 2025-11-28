# ABOUTME: Tests capturing attention weights during forward pass.
# ABOUTME: Uses a tiny model with MultiheadAttention to verify hook behavior.

import torch
from torch import nn

from src.sakt_kt.attention_extractor import AttentionExtractor


class TinyAttentionModel(nn.Module):
    def __init__(self, d_model=8, nhead=2):
        super().__init__()
        self.emb = nn.Embedding(10, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

    def forward(self, q_seq, r_seq, qry_seq):
        # Ignore r_seq for this dummy; use embeddings as both query and key/value.
        x = self.emb(q_seq)
        out, weights = self.attn(x, x, x, need_weights=True)
        return out, weights


def test_attention_extractor_captures_weights():
    model = TinyAttentionModel()
    extractor = AttentionExtractor(model)

    q = torch.tensor([[1, 2, 3]], dtype=torch.long)
    r = torch.zeros_like(q)
    qry = q.clone()

    preds, weights = extractor.extract(q, r, qry)

    # Expect one captured tensor matching MultiheadAttention output weights.
    assert len(weights) == 1
    captured = weights[0]
    # Support shapes with or without heads dimension.
    if captured.ndim == 4:
        assert captured.shape[0] == 1
        assert captured.shape[2] == q.size(1)
    elif captured.ndim == 3:
        assert captured.shape[0] == 1
        assert captured.shape[1] == q.size(1)
    # Weights should sum to ~1 along the source dimension.
    row_sums = captured.reshape(captured.shape[0], -1, captured.shape[-1])[0, 0].sum().item()
    assert 0.9 <= row_sums <= 1.1
