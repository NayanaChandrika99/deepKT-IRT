import pytest
from pathlib import Path
from docs.scripts.exporters.export_lineage import export_lineage_for_testing

def test_lineage_schema():
    """Test lineage export has nodes and links"""
    # Create mock file structure
    mock_files = [
        Path('data/raw/events.csv'),
        Path('data/processed/train.parquet'),
        Path('reports/model.ckpt')
    ]
    
    result = export_lineage_for_testing(mock_files)
    
    assert 'nodes' in result
    assert 'links' in result
    assert len(result['nodes']) == 3
    
    node = result['nodes'][0]
    assert 'id' in node
    assert 'group' in node
    assert 'size' in node

def test_lineage_grouping():
    """Test files are grouped by directory"""
    mock_files = [
        Path('data/raw/events.csv'),
        Path('reports/model.ckpt')
    ]
    
    result = export_lineage_for_testing(mock_files)
    
    groups = {n['id']: n['group'] for n in result['nodes']}
    assert groups['events.csv'] == 'raw'
    assert groups['model.ckpt'] == 'reports'

def test_lineage_linking():
    """Test links are created based on naming conventions"""
    # events.csv -> events.parquet
    mock_files = [
        Path('data/raw/events.csv'),
        Path('data/processed/events.parquet')
    ]
    
    result = export_lineage_for_testing(mock_files)
    
    assert len(result['links']) >= 1
    link = result['links'][0]
    assert link['source'] == 'events.csv'
    assert link['target'] == 'events.parquet'
