# ABOUTME: Converts canonical learning events to pyKT's expected data format.
# ABOUTME: Handles 1-indexing, padding, vocabulary building, and config generation.

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def canonical_to_pykt_csv(
    events_df: pd.DataFrame,
    output_dir: Path,
    max_seq_len: int = 200,
    n_folds: int = 5,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Convert canonical events DataFrame to pyKT's expected CSV format.
    
    pyKT expects:
    - One row per user
    - Comma-separated sequences: questions, concepts, responses
    - IDs are 1-indexed with 0 reserved for padding
    - Sequences padded to max_seq_len with 0s
    
    Args:
        events_df: DataFrame with columns: user_id, item_id, skill_ids, timestamp, correct
        output_dir: Directory to write output files
        max_seq_len: Maximum sequence length (longer sequences are truncated)
        n_folds: Number of cross-validation folds
    
    Returns:
        csv_path: Path to the generated train_valid_sequences.csv
        data_config: Dict with num_q, num_c, emb_path, etc.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build vocabulary mappings
    unique_items = sorted(events_df["item_id"].unique())
    unique_skills = _extract_unique_skills(events_df)
    
    # If no skills found, use items as concepts (fallback)
    if len(unique_skills) == 0:
        unique_skills = unique_items.copy()
    
    unique_skills = sorted(unique_skills)
    
    # pyKT expects 1-indexed IDs with 0 reserved for padding
    item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}
    skill_to_idx = {skill: idx + 1 for idx, skill in enumerate(unique_skills)}
    
    # Group by user and build sequences
    rows = []
    for user_id, group in events_df.groupby("user_id"):
        group = group.sort_values("timestamp").head(max_seq_len)
        
        questions, concepts, responses = _encode_user_sequence(
            group, item_to_idx, skill_to_idx
        )
        
        # Pad to max_seq_len (use 0 for padding, as IDs are 1-indexed)
        pad_len = max_seq_len - len(questions)
        if pad_len > 0:
            questions.extend([0] * pad_len)
            concepts.extend([0] * pad_len)
            responses.extend([0] * pad_len)
        
        rows.append({
            "uid": user_id,
            "questions": ",".join(str(x) for x in questions),
            "concepts": ",".join(str(x) for x in concepts),
            "responses": ",".join(str(x) for x in responses),
        })
    
    df_pykt = pd.DataFrame(rows)
    
    # Assign folds for cross-validation
    df_pykt["fold"] = np.arange(len(df_pykt)) % n_folds
    
    # Save CSV
    csv_path = output_dir / "train_valid_sequences.csv"
    df_pykt.to_csv(csv_path, index=False)
    
    # Build and save data config
    # Add +1 to account for padding index 0 (IDs are 1-indexed)
    data_config = build_data_config(
        num_questions=len(unique_items) + 1,
        num_concepts=len(unique_skills) + 1,
    )
    
    config_path = output_dir / "data_config.json"
    with open(config_path, "w") as f:
        json.dump(data_config, f, indent=2)
    
    return csv_path, data_config


def build_data_config(
    num_questions: int,
    num_concepts: int,
    max_concepts: int = 1,
) -> Dict[str, Any]:
    """
    Build the data_config dict that pyKT expects.
    
    Args:
        num_questions: Total unique questions including padding index 0
        num_concepts: Total unique concepts including padding index 0
        max_concepts: Maximum concepts per question (1 for most datasets)
    
    Returns:
        Dict with keys: num_q, num_c, max_concepts, input_type, emb_path
    """
    return {
        "num_q": num_questions,
        "num_c": num_concepts,
        "max_concepts": max_concepts,
        "input_type": ["questions", "concepts"],
        "emb_path": "",  # Empty string if not using pre-trained embeddings
    }


def _extract_unique_skills(events_df: pd.DataFrame) -> List[str]:
    """Extract unique skills from events, handling various skill_ids formats."""
    unique_skills = set()
    
    for skill_list in events_df["skill_ids"]:
        if isinstance(skill_list, list):
            unique_skills.update([s for s in skill_list if s])
        elif isinstance(skill_list, str) and skill_list:
            unique_skills.add(skill_list)
    
    return list(unique_skills)


def _encode_user_sequence(
    group: pd.DataFrame,
    item_to_idx: Dict[str, int],
    skill_to_idx: Dict[str, int],
) -> Tuple[List[int], List[int], List[int]]:
    """
    Encode a single user's interaction sequence.
    
    Returns:
        questions: List of 1-indexed question IDs
        concepts: List of 1-indexed concept IDs
        responses: List of 0/1 responses
    """
    questions = []
    concepts = []
    responses = []
    
    for _, row in group.iterrows():
        # Questions are 1-indexed, default to 1 if not found
        item_idx = item_to_idx.get(row["item_id"], 1)
        
        # Get first skill or use item_id as fallback concept
        skill_idx = _get_skill_index(row, skill_to_idx, item_to_idx)
        
        questions.append(item_idx)
        concepts.append(skill_idx)
        responses.append(int(row["correct"]))
    
    return questions, concepts, responses


def _get_skill_index(
    row: pd.Series,
    skill_to_idx: Dict[str, int],
    item_to_idx: Dict[str, int],
) -> int:
    """
    Get the skill/concept index for an interaction.
    Falls back to item_id if no skill is available.
    """
    skill_list = row["skill_ids"]
    
    if isinstance(skill_list, list) and len(skill_list) > 0:
        first_skill = skill_list[0]
        if first_skill:
            return skill_to_idx.get(first_skill, 1)
        else:
            return skill_to_idx.get(row["item_id"], 1)
    elif isinstance(skill_list, str) and skill_list:
        return skill_to_idx.get(skill_list, 1)
    else:
        return skill_to_idx.get(row["item_id"], 1)
