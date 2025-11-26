# ABOUTME: Handles exporting item health artifacts from trained WD-IRT models.
# ABOUTME: Writes parquet and markdown outputs consumed by reports and demos.

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import torch

from src.wd_irt.datasets import EdmClickstreamDataset, EdmDatasetPaths
from src.wd_irt.features import FeatureConfig
from src.wd_irt.model import WideDeepConfig, WideDeepIrtModule


def export_item_health(
    model_checkpoint: Path,
    config_path: Path,
    output_dir: Path,
    problem_details_path: Optional[Path] = None,
) -> None:
    """
    Export item parameters, drift scores, and behavior slices from a trained model.

    Args:
        model_checkpoint: Path to the trained model checkpoint (.ckpt file)
        config_path: Path to the training config YAML
        output_dir: Directory to write output artifacts
        problem_details_path: Optional path to problem_details.csv for topic mapping
    """
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    feature_config = FeatureConfig(**data_cfg["feature_config"])
    paths = EdmDatasetPaths(
        events=Path(data_cfg["events_path"]),
        assignment_details=Path(data_cfg["assignment_details"]),
        assignment_relationships=Path(data_cfg["assignment_relationships"]),
        unit_test_scores=Path(data_cfg["unit_test_scores"]),
        problem_details=Path(data_cfg["problem_details"]),
        split_manifest=Path(data_cfg["split_manifest"]),
    )

    # Create a minimal dataset to get item vocabulary mapping
    dataset = EdmClickstreamDataset(
        split="train",
        paths=paths,
        feature_config=feature_config,
        max_samples=1,  # Just need the vocab mapping
    )

    # Load model from checkpoint
    model_config = WideDeepConfig(**cfg["model"])
    model = WideDeepIrtModule.load_from_checkpoint(
        str(model_checkpoint),
        config=model_config,
        feature_config=feature_config,
        item_vocab_size=dataset.item_vocab_size,
        action_vocab_size=5,
        latency_bucket_count=feature_config.latency_bins + 1,
    )
    model.eval()

    # Extract item parameters
    with torch.no_grad():
        item_beta = model.item_beta.cpu().numpy()  # Difficulty parameters
        item_guess_logits = model.item_guess.cpu().numpy()  # Guessing logits
        item_guess = torch.sigmoid(model.item_guess).cpu().numpy()  # Guessing probabilities

    # Create reverse mapping: index -> item_id
    index_to_item = {idx: item_id for item_id, idx in dataset.item_to_index.items()}

    # Load problem details for topic/skill mapping if available
    topic_map: Dict[str, str] = {}
    if problem_details_path and problem_details_path.exists():
        problem_df = pd.read_csv(problem_details_path, usecols=["problem_id", "problem_skill_code"])
        for _, row in problem_df.iterrows():
            skill_code = str(row["problem_skill_code"]) if pd.notna(row["problem_skill_code"]) else "unknown"
            topic_map[row["problem_id"]] = skill_code.split(",")[0] if "," in skill_code else skill_code

    # Build item parameters DataFrame
    item_params_rows = []
    for idx in range(len(item_beta)):
        item_id = index_to_item.get(idx)
        if item_id is None:
            continue
        item_params_rows.append(
            {
                "item_id": item_id,
                "topic": topic_map.get(item_id, "unknown"),
                "difficulty": float(item_beta[idx]),
                "discrimination": 1.0,  # Wide & Deep IRT uses fixed discrimination
                "guessing": float(item_guess[idx]),
            }
        )

    item_params_df = pd.DataFrame(item_params_rows)
    item_params_path = output_dir / "item_params.parquet"
    item_params_df.to_parquet(item_params_path, index=False)
    print(f"✅ Exported {len(item_params_df)} item parameters to {item_params_path}")

    # Export item drift (placeholder - requires temporal analysis)
    item_drift_df = pd.DataFrame(
        {
            "item_id": item_params_df["item_id"],
            "drift_flag": [False] * len(item_params_df),
            "drift_score": [0.0] * len(item_params_df),
        }
    )
    item_drift_path = output_dir / "item_drift.parquet"
    item_drift_df.to_parquet(item_drift_path, index=False)
    print(f"✅ Exported drift flags to {item_drift_path} (placeholder - drift analysis not yet implemented)")

    # Generate behavior slices markdown
    behavior_slices_path = output_dir / "behavior_slices.md"
    _generate_behavior_slices(item_params_df, behavior_slices_path)
    print(f"✅ Exported behavior slices to {behavior_slices_path}")


def _generate_behavior_slices(item_params_df: pd.DataFrame, output_path: Path) -> None:
    """Generate markdown report summarizing item health by topic/ability groups."""
    with open(output_path, "w") as f:
        f.write("# Item Health Behavior Slices\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(f"Total items analyzed: {len(item_params_df)}\n\n")

        f.write("## Difficulty Distribution\n\n")
        f.write(f"- Mean difficulty: {item_params_df['difficulty'].mean():.3f}\n")
        f.write(f"- Std difficulty: {item_params_df['difficulty'].std():.3f}\n")
        f.write(f"- Min difficulty: {item_params_df['difficulty'].min():.3f}\n")
        f.write(f"- Max difficulty: {item_params_df['difficulty'].max():.3f}\n\n")

        f.write("## Guessing Parameter Distribution\n\n")
        f.write(f"- Mean guessing: {item_params_df['guessing'].mean():.3f}\n")
        f.write(f"- Std guessing: {item_params_df['guessing'].std():.3f}\n\n")

        f.write("## Items by Topic\n\n")
        topic_counts = item_params_df["topic"].value_counts()
        for topic, count in topic_counts.head(20).items():
            f.write(f"- {topic}: {count} items\n")
        if len(topic_counts) > 20:
            f.write(f"\n... and {len(topic_counts) - 20} more topics\n")

        f.write("\n## High-Difficulty Items (Top 10)\n\n")
        top_difficult = item_params_df.nlargest(10, "difficulty")[["item_id", "topic", "difficulty", "guessing"]]
        f.write("| item_id | topic | difficulty | guessing |\n")
        f.write("|---------|-------|------------|----------|\n")
        for _, row in top_difficult.iterrows():
            f.write(f"| {row['item_id']} | {row['topic']} | {row['difficulty']:.3f} | {row['guessing']:.3f} |\n")
        f.write("\n")

        f.write("## High-Guessing Items (Top 10)\n\n")
        top_guessing = item_params_df.nlargest(10, "guessing")[["item_id", "topic", "difficulty", "guessing"]]
        f.write("| item_id | topic | difficulty | guessing |\n")
        f.write("|---------|-------|------------|----------|\n")
        for _, row in top_guessing.iterrows():
            f.write(f"| {row['item_id']} | {row['topic']} | {row['difficulty']:.3f} | {row['guessing']:.3f} |\n")
        f.write("\n")
