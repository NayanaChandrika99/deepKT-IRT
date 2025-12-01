import pytest
import pandas as pd
from docs.scripts.exporters.export_mastery_timeline import export_mastery_timeline_for_testing

def test_mastery_timeline_schema():
    """Test mastery timeline export has required structure"""
    # Create mock mastery data
    mock_df = pd.DataFrame({
        'user_id': ['u1', 'u1', 'u1'],
        'item_id': ['i1', 'i2', 'i3'],
        'skill_id': ['s1', 's1', 's2'],
        'position': [1, 2, 3],
        'mastery': [0.5, 0.6, 0.4],
        'response': [1, 1, 0]
    })

    result = export_mastery_timeline_for_testing(mock_df)

    assert 'data' in result
    assert 'metadata' in result
    assert len(result['data']) == 3

    item = result['data'][0]
    assert 'user_id' in item
    assert 'sequence_position' in item
    assert 'mastery_score' in item
    assert 'skill_id' in item

def test_mastery_timeline_metadata():
    """Test metadata includes animation info"""
    mock_df = pd.DataFrame({
        'user_id': ['u1'],
        'item_id': ['i1'],
        'skill_id': ['s1'],
        'position': [1],
        'mastery': [0.5],
        'response': [1]
    })

    result = export_mastery_timeline_for_testing(mock_df)

    meta = result['metadata']
    assert 'animation' in meta
    assert 'frame_duration_ms' in meta['animation']
    assert 'max_frames' in meta['animation']
    assert meta['total_sequences'] == 1
    assert meta['max_position'] == 1
