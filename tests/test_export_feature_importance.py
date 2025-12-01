import pytest
import torch
import numpy as np
from pathlib import Path
from docs.scripts.exporters.export_feature_importance import (
    export_feature_importance_for_testing,
    compute_feature_importance_from_checkpoint
)

def test_feature_importance_schema():
    """Test feature importance export has required structure"""
    # Create mock model weights
    mock_weights = {
        'wide_linear.weight': torch.randn(1, 10),  # 10 wide features
        'deep_embeddings.user.weight': torch.randn(100, 32),
        'deep_embeddings.item.weight': torch.randn(500, 32),
        'deep_fc.0.weight': torch.randn(64, 96),  # 32+32+32 concat
    }

    result = export_feature_importance_for_testing(mock_weights)

    # Schema validation
    assert 'data' in result
    assert len(result['data']) >= 2  # At least Wide and Deep

    for group in result['data']:
        assert 'feature_group' in group
        assert 'importance' in group
        assert 'features' in group
        assert isinstance(group['features'], list)
        assert 0 <= group['importance'] <= 1

    # Sum of importance should be ~1.0
    total_importance = sum(g['importance'] for g in result['data'])
    assert 0.95 <= total_importance <= 1.05

def test_feature_importance_ordering():
    """Test groups are ordered by importance descending"""
    mock_weights = {
        'wide_linear.weight': torch.randn(1, 5),
        'deep_embeddings.user.weight': torch.randn(50, 16),
    }

    result = export_feature_importance_for_testing(mock_weights)

    importances = [g['importance'] for g in result['data']]
    assert importances == sorted(importances, reverse=True)

def test_checkpoint_loading():
    """Test actual PyTorch checkpoint loading"""
    # Create a minimal checkpoint
    checkpoint_dir = Path('tests/fixtures/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'test_model.ckpt'

    # Save a simple state dict
    state_dict = {
        'wide_linear.weight': torch.randn(1, 8),
        'deep_embeddings.user.weight': torch.randn(20, 16),
    }
    torch.save({'state_dict': state_dict}, checkpoint_path)

    try:
        result = compute_feature_importance_from_checkpoint(checkpoint_path)
        assert 'data' in result
        assert len(result['data']) >= 1
    finally:
        # Cleanup
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if checkpoint_dir.exists():
            checkpoint_dir.rmdir()

def test_metadata_includes_checkpoint_info():
    """Test metadata records checkpoint source"""
    mock_weights = {
        'wide_linear.weight': torch.randn(1, 5),
    }

    result = export_feature_importance_for_testing(mock_weights)

    assert 'metadata' in result
    assert 'model' in result['metadata']
    assert 'mock' in result['metadata']
