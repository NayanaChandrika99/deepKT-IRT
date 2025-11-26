# ABOUTME: Tests for SAKT data adapter and dataset utilities.
# ABOUTME: Validates 1-indexing, padding, sequence length, and data config.

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.sakt_kt.adapters import (
    canonical_to_pykt_csv,
    build_data_config,
    _extract_unique_skills,
)
from src.sakt_kt.datasets import (
    PyKTDataset,
    build_shifted_query,
    prepare_dataloaders,
)


class TestCanonicalToPyktCsv(unittest.TestCase):
    """Tests for canonical_to_pykt_csv conversion."""
    
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)
        
        # Create sample canonical events
        self.events_df = pd.DataFrame({
            "user_id": ["U1", "U1", "U1", "U2", "U2"],
            "item_id": ["Q1", "Q2", "Q3", "Q1", "Q4"],
            "skill_ids": [["S1"], ["S2"], ["S1"], ["S1"], ["S3"]],
            "timestamp": pd.to_datetime([
                "2024-01-01 10:00",
                "2024-01-01 10:01",
                "2024-01-01 10:02",
                "2024-01-01 11:00",
                "2024-01-01 11:01",
            ]),
            "correct": [1, 0, 1, 1, 0],
        })
    
    def tearDown(self) -> None:
        self.tmpdir.cleanup()
    
    def test_one_indexing(self):
        """Question and concept IDs should be 1-indexed (min = 1, not 0)."""
        csv_path, config = canonical_to_pykt_csv(
            self.events_df, self.output_dir, max_seq_len=10
        )
        
        df = pd.read_csv(csv_path)
        
        # Check all question sequences
        for _, row in df.iterrows():
            qs = [int(x) for x in row["questions"].split(",")]
            non_padding_qs = [q for q in qs if q > 0]
            if non_padding_qs:
                self.assertGreaterEqual(min(non_padding_qs), 1,
                    "Question IDs should be 1-indexed (min >= 1)")
    
    def test_padding_with_zero(self):
        """Padding should use 0."""
        csv_path, config = canonical_to_pykt_csv(
            self.events_df, self.output_dir, max_seq_len=10
        )
        
        df = pd.read_csv(csv_path)
        
        # U1 has 3 events, so should have 7 padding zeros
        u1_row = df[df["uid"] == "U1"].iloc[0]
        qs = [int(x) for x in u1_row["questions"].split(",")]
        
        # Last 7 should be padding (0)
        self.assertEqual(qs[3:], [0] * 7,
            "Padding positions should be 0")
    
    def test_max_seq_len_enforced(self):
        """Sequences should not exceed max_seq_len."""
        csv_path, config = canonical_to_pykt_csv(
            self.events_df, self.output_dir, max_seq_len=5
        )
        
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            qs = row["questions"].split(",")
            self.assertEqual(len(qs), 5,
                f"Sequence length should be exactly max_seq_len (5), got {len(qs)}")
    
    def test_data_config_keys(self):
        """data_config should have required pyKT keys."""
        csv_path, config = canonical_to_pykt_csv(
            self.events_df, self.output_dir, max_seq_len=10
        )
        
        required_keys = {"num_q", "num_c", "max_concepts", "input_type", "emb_path"}
        self.assertEqual(set(config.keys()), required_keys,
            f"Config should have keys {required_keys}")
    
    def test_num_q_includes_padding_index(self):
        """num_q should be len(unique_items) + 1 for padding index 0."""
        csv_path, config = canonical_to_pykt_csv(
            self.events_df, self.output_dir, max_seq_len=10
        )
        
        # We have 4 unique items: Q1, Q2, Q3, Q4
        # num_q should be 4 + 1 = 5
        self.assertEqual(config["num_q"], 5,
            "num_q should include +1 for padding index 0")
    
    def test_num_c_includes_padding_index(self):
        """num_c should be len(unique_skills) + 1 for padding index 0."""
        csv_path, config = canonical_to_pykt_csv(
            self.events_df, self.output_dir, max_seq_len=10
        )
        
        # We have 3 unique skills: S1, S2, S3
        # num_c should be 3 + 1 = 4
        self.assertEqual(config["num_c"], 4,
            "num_c should include +1 for padding index 0")


class TestFallbackToItemId(unittest.TestCase):
    """Tests for fallback behavior when skill_ids are missing."""
    
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)
    
    def tearDown(self) -> None:
        self.tmpdir.cleanup()
    
    def test_empty_skills_uses_item_id(self):
        """When skill_ids are empty, should fallback to item_id as concept."""
        events_df = pd.DataFrame({
            "user_id": ["U1", "U1"],
            "item_id": ["Q1", "Q2"],
            "skill_ids": [[], []],  # Empty skills
            "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:01"]),
            "correct": [1, 0],
        })
        
        csv_path, config = canonical_to_pykt_csv(
            events_df, self.output_dir, max_seq_len=10
        )
        
        # num_c should equal num_q since items become concepts
        self.assertEqual(config["num_c"], config["num_q"],
            "When no skills, num_c should equal num_q (items used as concepts)")


class TestBuildDataConfig(unittest.TestCase):
    """Tests for build_data_config function."""
    
    def test_returns_expected_structure(self):
        """Should return dict with all required pyKT keys."""
        config = build_data_config(num_questions=100, num_concepts=50)
        
        self.assertEqual(config["num_q"], 100)
        self.assertEqual(config["num_c"], 50)
        self.assertEqual(config["max_concepts"], 1)
        self.assertEqual(config["input_type"], ["questions", "concepts"])
        self.assertEqual(config["emb_path"], "")


class TestBuildShiftedQuery(unittest.TestCase):
    """Tests for build_shifted_query function."""
    
    def test_shifts_correctly(self):
        """Should prepend 0 and drop last element."""
        qseqs = torch.tensor([1, 2, 3, 4, 5])
        shifted = build_shifted_query(qseqs)
        
        expected = torch.tensor([0, 1, 2, 3, 4])
        self.assertTrue(torch.equal(shifted, expected),
            f"Expected {expected}, got {shifted}")
    
    def test_preserves_length(self):
        """Output should have same length as input."""
        qseqs = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        shifted = build_shifted_query(qseqs)
        
        self.assertEqual(len(shifted), len(qseqs),
            "Shifted sequence should have same length as input")


class TestPyKTDataset(unittest.TestCase):
    """Tests for PyKTDataset class."""
    
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmpdir.name) / "train_valid_sequences.csv"
        
        # Create sample CSV
        df = pd.DataFrame({
            "uid": ["U1", "U2", "U3", "U4", "U5"],
            "questions": ["1,2,0,0", "3,4,5,0", "1,0,0,0", "2,3,0,0", "4,5,1,2"],
            "concepts": ["1,2,0,0", "3,4,5,0", "1,0,0,0", "2,3,0,0", "4,5,1,2"],
            "responses": ["1,0,0,0", "1,1,0,0", "0,0,0,0", "1,0,0,0", "1,1,1,0"],
            "fold": [0, 1, 2, 3, 4],
        })
        df.to_csv(self.csv_path, index=False)
    
    def tearDown(self) -> None:
        self.tmpdir.cleanup()
    
    def test_train_excludes_fold(self):
        """Training set should exclude the specified fold."""
        dataset = PyKTDataset(self.csv_path, fold=0, is_train=True)
        self.assertEqual(len(dataset), 4,
            "Training set should have 4 samples (excluding fold 0)")
    
    def test_val_includes_only_fold(self):
        """Validation set should include only the specified fold."""
        dataset = PyKTDataset(self.csv_path, fold=0, is_train=False)
        self.assertEqual(len(dataset), 1,
            "Validation set should have 1 sample (only fold 0)")
    
    def test_returns_expected_keys(self):
        """__getitem__ should return dict with expected tensor keys."""
        dataset = PyKTDataset(self.csv_path, fold=0, is_train=True)
        sample = dataset[0]
        
        expected_keys = {"qseqs", "rseqs", "qryseqs", "masks"}
        self.assertEqual(set(sample.keys()), expected_keys,
            f"Sample should have keys {expected_keys}")
    
    def test_tensors_have_correct_dtype(self):
        """All tensors should be torch.long."""
        dataset = PyKTDataset(self.csv_path, fold=0, is_train=True)
        sample = dataset[0]
        
        for key, tensor in sample.items():
            self.assertEqual(tensor.dtype, torch.long,
                f"{key} should be torch.long, got {tensor.dtype}")
    
    def test_mask_identifies_padding(self):
        """Mask should be 1 for valid positions, 0 for padding (q=0)."""
        dataset = PyKTDataset(self.csv_path, fold=2, is_train=False)
        sample = dataset[0]  # U3 with questions "1,0,0,0"
        
        expected_mask = torch.tensor([1, 0, 0, 0])
        self.assertTrue(torch.equal(sample["masks"], expected_mask),
            f"Expected mask {expected_mask}, got {sample['masks']}")


class TestPrepareDataloaders(unittest.TestCase):
    """Tests for prepare_dataloaders function."""
    
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmpdir.name) / "train_valid_sequences.csv"
        
        # Create sample CSV with 10 users
        df = pd.DataFrame({
            "uid": [f"U{i}" for i in range(10)],
            "questions": ["1,2,3,0"] * 10,
            "concepts": ["1,2,3,0"] * 10,
            "responses": ["1,0,1,0"] * 10,
            "fold": [i % 5 for i in range(10)],
        })
        df.to_csv(self.csv_path, index=False)
    
    def tearDown(self) -> None:
        self.tmpdir.cleanup()
    
    def test_returns_two_dataloaders(self):
        """Should return train and val DataLoaders."""
        train_loader, val_loader = prepare_dataloaders(
            self.csv_path, batch_size=2, fold=0
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
    
    def test_batch_size_respected(self):
        """DataLoaders should use specified batch size."""
        train_loader, val_loader = prepare_dataloaders(
            self.csv_path, batch_size=3, fold=0
        )
        
        batch = next(iter(train_loader))
        # Batch size should be 3 (or less if dataset is smaller)
        self.assertLessEqual(batch["qseqs"].shape[0], 3)


if __name__ == "__main__":
    unittest.main()

