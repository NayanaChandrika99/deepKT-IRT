import pytest
import pandas as pd
from docs.scripts.exporters.export_sankey import export_sankey_for_testing

def test_sankey_schema():
    """Test Sankey export has nodes and links"""
    # Create mock event flow
    # Source -> Action -> Result
    mock_df = pd.DataFrame({
        'user_id': ['u1', 'u1', 'u2'],
        'source': ['Home', 'Home', 'Quiz'],
        'target': ['Quiz', 'Profile', 'Result'],
        'value': [1, 1, 1]
    })
    
    result = export_sankey_for_testing(mock_df)
    
    assert 'nodes' in result
    assert 'links' in result
    
    # Check nodes
    node_labels = [n['label'] for n in result['nodes']]
    assert 'Home' in node_labels
    assert 'Quiz' in node_labels
    
    # Check links
    assert len(result['links']) >= 2
    link = result['links'][0]
    assert 'source' in link
    assert 'target' in link
    assert 'value' in link

def test_sankey_aggregation():
    """Test that identical flows are aggregated"""
    mock_df = pd.DataFrame({
        'source': ['A', 'A'],
        'target': ['B', 'B'],
        'value': [1, 1]
    })
    
    result = export_sankey_for_testing(mock_df)
    
    # Should be one link with value 2
    assert len(result['links']) == 1
    assert result['links'][0]['value'] == 2
