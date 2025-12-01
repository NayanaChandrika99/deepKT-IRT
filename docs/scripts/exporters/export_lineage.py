# docs/scripts/exporters/export_lineage.py
# ABOUTME: Data lineage exporter for force-directed graph visualization
# ABOUTME: Scans file system to map dependencies between data artifacts

from pathlib import Path
from typing import Dict, List, Optional

def get_file_group(filename: str, parent_dir: str) -> str:
    """Determine group based on directory or filename"""
    if 'raw' in parent_dir:
        return 'raw'
    if 'processed' in parent_dir:
        return 'processed'
    if 'reports' in parent_dir:
        return 'reports'
    if 'checkpoints' in parent_dir:
        return 'models'
    return 'other'

def infer_links(files: List[Path]) -> List[Dict]:
    """
    Infer links between files based on naming conventions.
    
    Rules:
    - .csv -> .parquet (same name)
    - .parquet -> .ckpt (training data to model)
    - .ckpt -> .json (model to metrics/reports)
    """
    links = []
    filenames = {f.name: f for f in files}
    stems = {f.stem: f for f in files}
    
    for f in files:
        # Rule 1: CSV to Parquet conversion
        if f.suffix == '.parquet':
            csv_name = f.stem + '.csv'
            if csv_name in filenames:
                links.append({'source': csv_name, 'target': f.name, 'type': 'transform'})
        
        # Rule 2: Parquet to Model (loose association)
        if f.suffix == '.ckpt':
            # Link all processed parquet files to model
            for other in files:
                if other.suffix == '.parquet' and 'processed' in str(other):
                    links.append({'source': other.name, 'target': f.name, 'type': 'train'})
                    
        # Rule 3: Model to Reports
        if f.suffix == '.json' or f.suffix == '.html':
            # Link models to reports
            for other in files:
                if other.suffix == '.ckpt':
                    links.append({'source': other.name, 'target': f.name, 'type': 'evaluate'})

    return links

def export_lineage_for_testing(files: List[Path]) -> Dict:
    """
    Generate lineage graph from list of files.
    
    Args:
        files: List of file paths
        
    Returns:
        Dict with nodes and links
    """
    nodes = []
    for f in files:
        group = get_file_group(f.name, str(f.parent))
        nodes.append({
            'id': f.name,
            'group': group,
            'path': str(f),
            'size': 10 + (5 if group == 'models' else 0)
        })
        
    links = infer_links(files)
    
    return {
        'nodes': nodes,
        'links': links,
        'metadata': {
            'total_files': len(files),
            'total_links': len(links)
        }
    }

def export_lineage(project_root: Path) -> Dict:
    """
    Scan project directories and export lineage graph.
    
    Args:
        project_root: Root directory of project
        
    Returns:
        Dict for JSON export
    """
    # Directories to scan
    dirs_to_scan = [
        project_root / 'data/raw',
        project_root / 'data/processed',
        project_root / 'reports',
        project_root / 'reports/checkpoints'
    ]
    
    files = []
    for d in dirs_to_scan:
        if d.exists():
            # Add all relevant files
            files.extend([f for f in d.rglob('*') if f.is_file() and f.suffix in ['.csv', '.parquet', '.ckpt', '.json', '.html']])
            
    return export_lineage_for_testing(files)
