# ABOUTME: Quick smoke test to validate training pipeline works without GPU.
# ABOUTME: Runs a single batch through the model to catch shape/import errors.

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.wd_irt.datasets import EdmClickstreamDataset, EdmDatasetPaths, collate_wdirt_batch
from src.wd_irt.features import FeatureConfig
from src.wd_irt.model import WideDeepConfig, WideDeepIrtModule


def test_smoke():
    """Run a single batch through the model to validate shapes and forward pass."""
    
    print("🔍 Smoke test: Validating training pipeline...")
    
    # Check if data exists
    data_dir = Path("data")
    if not (data_dir / "interim" / "edm_cup_2023_42_events.parquet").exists():
        print("❌ Preprocessed data not found. Run 'make data' first.")
        return False
    
    if not (data_dir / "splits" / "edm_cup_2023_42.json").exists():
        print("❌ Split manifest not found. Run 'make data' first.")
        return False
    
    # Use minimal config
    feature_config = FeatureConfig(seq_len=10, latency_bins=5, bert_dim=16)  # Tiny for speed
    
    paths = EdmDatasetPaths(
        events=Path("data/interim/edm_cup_2023_42_events.parquet"),
        assignment_details=Path("data/raw/edm_cup_2023/assignment_details.csv"),
        assignment_relationships=Path("data/raw/edm_cup_2023/assignment_relationships.csv"),
        unit_test_scores=Path("data/raw/edm_cup_2023/training_unit_test_scores.csv"),
        problem_details=Path("data/raw/edm_cup_2023/problem_details.csv"),
        split_manifest=Path("data/splits/edm_cup_2023_42.json"),
    )
    
    print("📦 Loading dataset (this may take a minute)...")
    try:
        dataset = EdmClickstreamDataset(
            split="train",
            paths=paths,
            feature_config=feature_config,
            max_samples=10,  # Just 10 samples for smoke test
        )
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return False
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_wdirt_batch,
    )
    
    # Get one batch
    print("📊 Fetching a batch...")
    try:
        batch = next(iter(dataloader))
        inputs, labels, metadata = batch
        print(f"✅ Batch shape - labels: {labels.shape}")
        print(f"   Input keys: {list(inputs.keys())}")
        for key, val in inputs.items():
            print(f"   {key}: {val.shape}")
    except Exception as e:
        print(f"❌ Batch fetching failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create model
    print("🧠 Creating model...")
    try:
        model_config = WideDeepConfig(
            wide_units=64,
            deep_units=[128, 64],
            embedding_dim=32,
            dropout=0.1,
            activation="relu",
            learning_rate=0.001,
            weight_decay=0.01,
            ability_regularizer=0.001,
        )
        
        model = WideDeepIrtModule(
            config=model_config,
            feature_config=feature_config,
            item_vocab_size=dataset.item_vocab_size,
            action_vocab_size=5,
            latency_bucket_count=feature_config.latency_bins + 1,
        )
        print(f"✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Forward pass
    print("⚡ Running forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            probs, ability = model(inputs)
        print(f"✅ Forward pass successful!")
        print(f"   probs shape: {probs.shape}")
        print(f"   ability shape: {ability.shape}")
        
        # Check shapes match
        if probs.shape != labels.shape:
            print(f"❌ Shape mismatch: probs {probs.shape} vs labels {labels.shape}")
            return False
        
        print(f"✅ Shapes match: {probs.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Training step
    print("🎯 Testing training step...")
    try:
        model.train()
        loss = model.training_step((inputs, labels, metadata), 0)
        print(f"✅ Training step successful! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validation step
    print("📈 Testing validation step...")
    try:
        model.eval()
        loss = model.validation_step((inputs, labels, metadata), 0)
        print(f"✅ Validation step successful! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Validation step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All smoke tests passed! Pipeline is ready for GPU training.")
    return True


if __name__ == "__main__":
    success = test_smoke()
    sys.exit(0 if success else 1)

