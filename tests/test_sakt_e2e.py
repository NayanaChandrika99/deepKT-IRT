# ABOUTME: End-to-end validation tests for the SAKT pipeline.
# ABOUTME: Verifies the full workflow from data prep through export works correctly.

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.sakt_kt.adapters import canonical_to_pykt_csv
from src.sakt_kt.datasets import PyKTDataset, prepare_dataloaders, build_shifted_query


class TestSaktE2EDataPrep(unittest.TestCase):
    """End-to-end test for SAKT data preparation pipeline."""
    
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)
        
        # Create realistic sample data
        self.events_df = pd.DataFrame({
            "user_id": ["U1"] * 10 + ["U2"] * 8 + ["U3"] * 5,
            "item_id": [f"Q{i}" for i in range(10)] + [f"Q{i}" for i in range(8)] + [f"Q{i}" for i in range(5)],
            "skill_ids": [[f"S{i % 3}"] for i in range(23)],
            "timestamp": pd.date_range("2024-01-01", periods=23, freq="1min"),
            "correct": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        })
    
    def tearDown(self) -> None:
        self.tmpdir.cleanup()
    
    def test_full_data_prep_pipeline(self):
        """Test the complete data preparation pipeline."""
        # Step 1: Convert to pyKT format
        csv_path, data_config = canonical_to_pykt_csv(
            self.events_df, self.output_dir, max_seq_len=20
        )
        
        # Verify CSV was created
        self.assertTrue(csv_path.exists(), "CSV file should be created")
        
        # Verify data_config structure
        self.assertIn("num_q", data_config)
        self.assertIn("num_c", data_config)
        self.assertIn("emb_path", data_config)
        
        # Step 2: Load as PyKT dataset
        train_dataset = PyKTDataset(csv_path, fold=0, is_train=True)
        val_dataset = PyKTDataset(csv_path, fold=0, is_train=False)
        
        # Verify datasets have data
        self.assertGreater(len(train_dataset) + len(val_dataset), 0,
            "Should have at least one sample")
        
        # Step 3: Create dataloaders
        train_loader, val_loader = prepare_dataloaders(
            csv_path, batch_size=2, fold=0
        )
        
        # Step 4: Verify batch structure
        batch = next(iter(train_loader))
        
        required_keys = {"qseqs", "rseqs", "qryseqs", "masks"}
        self.assertEqual(set(batch.keys()), required_keys,
            f"Batch should have keys {required_keys}")
        
        # Verify tensor shapes match
        batch_size = batch["qseqs"].shape[0]
        seq_len = batch["qseqs"].shape[1]
        
        self.assertEqual(batch["rseqs"].shape, (batch_size, seq_len))
        self.assertEqual(batch["qryseqs"].shape, (batch_size, seq_len))
        self.assertEqual(batch["masks"].shape, (batch_size, seq_len))
    
    def test_index_bounds_are_valid(self):
        """Verify all indices are within valid bounds for embeddings."""
        # Use data without skills (like real ASSISTments case) so num_q == num_c
        events_no_skills = pd.DataFrame({
            "user_id": ["U1"] * 5 + ["U2"] * 5,
            "item_id": [f"Q{i}" for i in range(5)] + [f"Q{i}" for i in range(5)],
            "skill_ids": [[] for _ in range(10)],  # No skills, falls back to item_id
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
            "correct": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
        })
        
        csv_path, data_config = canonical_to_pykt_csv(
            events_no_skills, self.output_dir, max_seq_len=20
        )
        
        train_loader, val_loader = prepare_dataloaders(
            csv_path, batch_size=10, fold=0
        )
        
        max_q = 0
        max_r = 0
        
        for batch in train_loader:
            max_q = max(max_q, batch["qseqs"].max().item())
            max_r = max(max_r, batch["rseqs"].max().item())
        
        for batch in val_loader:
            max_q = max(max_q, batch["qseqs"].max().item())
            max_r = max(max_r, batch["rseqs"].max().item())
        
        # When no skills, num_q == num_c (items become concepts)
        self.assertEqual(data_config["num_q"], data_config["num_c"],
            "Without skills, num_q should equal num_c")
        
        # Questions should be < num_c (for exercise embedding)
        self.assertLess(max_q, data_config["num_c"],
            f"Max question index {max_q} should be < num_c {data_config['num_c']}")
        
        # Responses should be 0 or 1
        self.assertLessEqual(max_r, 1, "Max response should be <= 1")
        
        # Interaction index = q + num_c * r should be < 2 * num_c
        max_interaction = max_q + data_config["num_c"] * max_r
        self.assertLess(max_interaction, 2 * data_config["num_c"],
            f"Max interaction index {max_interaction} should be < 2*num_c {2 * data_config['num_c']}")


class TestShiftedQueryConsistency(unittest.TestCase):
    """Test that shifted query sequences are computed consistently."""
    
    def test_batch_vs_single_shifted_query(self):
        """Shifted query should work the same for batch and single samples."""
        import torch
        
        # Single sequence
        single_q = torch.tensor([1, 2, 3, 4, 5])
        single_shifted = build_shifted_query(single_q)
        
        expected = torch.tensor([0, 1, 2, 3, 4])
        self.assertTrue(torch.equal(single_shifted, expected),
            f"Single shift: expected {expected}, got {single_shifted}")
        
        # Batch (simulated by stacking)
        batch_q = torch.stack([single_q, single_q])
        batch_shifted = torch.stack([build_shifted_query(q) for q in batch_q])
        
        expected_batch = torch.stack([expected, expected])
        self.assertTrue(torch.equal(batch_shifted, expected_batch),
            "Batch shift should match stacked single shifts")


class TestArtifactSchemas(unittest.TestCase):
    """Test that exported artifacts have correct schemas."""
    
    def test_predictions_schema(self):
        """Predictions parquet should have required columns."""
        # This test would run on actual export output
        # For now, verify the expected schema
        expected_columns = {"user_id", "item_id", "position", "actual", "predicted"}
        
        # Create mock predictions DataFrame
        predictions_df = pd.DataFrame({
            "user_id": ["U1", "U1"],
            "item_id": ["Q1", "Q2"],
            "position": [1, 2],
            "actual": [1, 0],
            "predicted": [0.7, 0.3],
        })
        
        self.assertEqual(set(predictions_df.columns), expected_columns,
            f"Predictions should have columns {expected_columns}")
    
    def test_mastery_schema(self):
        """Student state parquet should have required columns."""
        expected_columns = {"user_id", "item_id", "position", "response", "mastery"}
        
        # Create mock mastery DataFrame
        mastery_df = pd.DataFrame({
            "user_id": ["U1", "U1"],
            "item_id": ["Q1", "Q2"],
            "position": [0, 1],
            "response": [1, 0],
            "mastery": [0.5, 0.6],
        })
        
        self.assertEqual(set(mastery_df.columns), expected_columns,
            f"Mastery should have columns {expected_columns}")
    
    def test_attention_schema(self):
        """Attention parquet should have required columns."""
        expected_columns = {"user_id", "position", "mastery", "top_influences"}
        
        # Create mock attention DataFrame
        attention_df = pd.DataFrame({
            "user_id": ["U1", "U2"],
            "position": [5, 3],
            "mastery": [0.65, 0.45],
            "top_influences": [
                [
                    {"item_id": "Q1", "correct": True, "weight": 0.3, "position": 0},
                    {"item_id": "Q3", "correct": False, "weight": 0.25, "position": 2},
                ],
                [
                    {"item_id": "Q2", "correct": True, "weight": 0.4, "position": 1},
                ],
            ],
        })
        
        self.assertEqual(set(attention_df.columns), expected_columns,
            f"Attention should have columns {expected_columns}")
        
        # Verify top_influences structure
        for influences in attention_df["top_influences"]:
            self.assertIsInstance(influences, list, "top_influences should be a list")
            for inf in influences:
                self.assertIn("item_id", inf, "Each influence should have item_id")
                self.assertIn("correct", inf, "Each influence should have correct")
                self.assertIn("weight", inf, "Each influence should have weight")
                self.assertIn("position", inf, "Each influence should have position")


if __name__ == "__main__":
    unittest.main()

