# docs/scripts/exporters/export_mastery_timeline.py
# ABOUTME: Student mastery timeline exporter for animated visualization
# ABOUTME: Formats mastery data for frame-by-frame animation

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

def export_mastery_timeline_for_testing(mastery_df: pd.DataFrame) -> Dict:
    """
    Format mastery data for timeline visualization.

    Args:
        mastery_df: DataFrame with columns:
            - user_id
            - item_id
            - skill_id
            - position (sequence position)
            - mastery (score 0-1)
            - response (0/1)

    Returns:
        Dict with data and animation metadata
    """
    # Ensure required columns
    required_cols = ['user_id', 'skill_id', 'position', 'mastery']
    for col in required_cols:
        if col not in mastery_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert to list of dicts
    data = []
    for _, row in mastery_df.iterrows():
        data.append({
            'user_id': str(row['user_id']),
            'skill_id': str(row['skill_id']),
            'sequence_position': int(row['position']),
            'mastery_score': float(row['mastery']),
            'item_id': str(row.get('item_id', '')),
            'response': int(row.get('response', 0))
        })

    # Compute metadata
    total_sequences = len(mastery_df['user_id'].unique())
    max_position = int(mastery_df['position'].max()) if not mastery_df.empty else 0
    skills = list(mastery_df['skill_id'].unique()) if not mastery_df.empty else []

    return {
        'data': data,
        'metadata': {
            'total_sequences': total_sequences,
            'max_position': max_position,
            'skills': skills,
            'animation': {
                'frame_duration_ms': 300,
                'transition_duration_ms': 200,
                'max_frames': 200
            }
        }
    }

def export_mastery_timeline(reports_dir: Path) -> Dict:
    """
    Export mastery timeline from parquet file.

    Args:
        reports_dir: Path to reports directory

    Returns:
        Dict for JSON export
    """
    mastery_path = reports_dir / "sakt_mastery.parquet"

    if not mastery_path.exists():
        raise FileNotFoundError(f"Mastery data not found: {mastery_path}")

    mastery_df = pd.read_parquet(mastery_path)

    # Rename columns to match expected schema if needed
    # Assuming parquet has: user_id, item_id, skill_id, timestamp, correct, mastery_prob
    if 'mastery_prob' in mastery_df.columns:
        mastery_df = mastery_df.rename(columns={'mastery_prob': 'mastery'})

    # Add position if missing (rank by timestamp per user)
    if 'position' not in mastery_df.columns and 'timestamp' in mastery_df.columns:
        mastery_df = mastery_df.sort_values(['user_id', 'timestamp'])
        mastery_df['position'] = mastery_df.groupby('user_id').cumcount() + 1

    return export_mastery_timeline_for_testing(mastery_df)
