import pytest
import numpy as np
import torch
from docs.scripts.exporters.export_attention_network import export_attention_network_for_testing

def test_attention_network_schema():
    """Test attention network export has nodes and links"""
    # Mock attention weights (batch=1, heads=1, seq=3, seq=3)
    attn_weights = torch.tensor([[[
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.2, 0.3, 0.5]
    ]]])
    
    # Mock tokens
    tokens = ['q1', 'q2', 'q3']
    
    result = export_attention_network_for_testing(attn_weights, tokens)
    
    assert 'nodes' in result
    assert 'links' in result
    assert len(result['nodes']) == 3
    
    # Check links (should only have non-zero weights)
    # q2 -> q1 (0.5), q2 -> q2 (0.5)
    # q3 -> q1 (0.2), q3 -> q2 (0.3), q3 -> q3 (0.5)
    # Total 5 links (excluding self-loops if filtered, but here we keep them)
    assert len(result['links']) >= 3

def test_attention_thresholding():
    """Test that low attention weights are filtered out"""
    attn_weights = torch.tensor([[[
        [1.0, 0.0],
        [0.01, 0.99]  # 0.01 should be filtered
    ]]])
    tokens = ['a', 'b']
    
    result = export_attention_network_for_testing(attn_weights, tokens, threshold=0.1)
    
    # Should only have a->a and b->b links
    links = result['links']
    assert len(links) == 2
    for link in links:
        assert link['weight'] >= 0.1
