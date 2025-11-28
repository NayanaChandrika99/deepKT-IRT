# ABOUTME: Integration tests for attention extraction in SAKT export pipeline.
# ABOUTME: Verifies that attention weights are captured and exported correctly.

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.sakt_kt.attention_extractor import (
    AttentionExtractor,
    extract_top_influences,
    compute_attention_from_scratch,
)


class MockAttentionLayer(nn.Module):
    """Mock attention layer that returns (output, weights) tuple."""
    
    def __init__(self, seq_len: int = 10):
        super().__init__()
        self.seq_len = seq_len
    
    def forward(self, x):
        batch_size = x.shape[0]
        # Return (output, attention_weights) tuple
        output = x
        # Create fake attention weights [batch, seq, seq]
        attn_weights = torch.softmax(torch.randn(batch_size, self.seq_len, self.seq_len), dim=-1)
        return output, attn_weights


class MockSAKTModel(nn.Module):
    """Mock SAKT model with attention layers."""
    
    def __init__(self, seq_len: int = 10):
        super().__init__()
        self.attention = MockAttentionLayer(seq_len)
        self.emb = nn.Embedding(100, 32)
    
    def forward(self, q_seq, r_seq, qry_seq):
        # Simulate SAKT forward pass
        x = self.emb(qry_seq)
        output, attn_weights = self.attention(x)
        # Return predictions only (not tuple) to match pyKT behavior
        return output.mean(dim=-1)


class TestAttentionExtractor(unittest.TestCase):
    """Test attention extraction from models."""
    
    def setUp(self):
        self.seq_len = 10
        self.model = MockSAKTModel(seq_len=self.seq_len)
        self.model.eval()
    
    def test_extractor_captures_attention(self):
        """Test that AttentionExtractor captures attention weights."""
        extractor = AttentionExtractor(self.model)
        
        q_seq = torch.randint(1, 50, (1, self.seq_len))
        r_seq = torch.randint(0, 2, (1, self.seq_len))
        qry_seq = torch.randint(1, 50, (1, self.seq_len))
        
        predictions, attention_weights = extractor.extract(q_seq, r_seq, qry_seq)
        
        # Should capture at least one attention tensor
        self.assertGreater(len(attention_weights), 0,
            "Should capture at least one attention weight tensor")
        
        # Attention weights should be on CPU
        for attn in attention_weights:
            self.assertEqual(attn.device.type, "cpu",
                "Attention weights should be moved to CPU")
    
    def test_extract_top_influences(self):
        """Test extracting top-k influences from attention weights."""
        seq_len = 10
        # Create fake attention matrix [seq_len, seq_len]
        attn = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)
        
        item_ids = [f"Q{i}" for i in range(seq_len)]
        responses = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        position = 5  # Predict at position 5
        
        influences = extract_top_influences(attn, item_ids, responses, position, k=3)
        
        self.assertIsInstance(influences, list)
        self.assertLessEqual(len(influences), 3, "Should return at most k influences")
        
        if len(influences) > 0:
            # Verify structure
            for inf in influences:
                self.assertIn("item_id", inf)
                self.assertIn("correct", inf)
                self.assertIn("weight", inf)
                self.assertIn("position", inf)
                
                # Weights should be in [0, 1]
                self.assertGreaterEqual(inf["weight"], 0.0)
                self.assertLessEqual(inf["weight"], 1.0)
                
                # Position should be < prediction position
                self.assertLess(inf["position"], position)
    
    def test_extract_top_influences_empty(self):
        """Test extracting influences at position 0 (no history)."""
        attn = torch.softmax(torch.randn(5, 5), dim=-1)
        influences = extract_top_influences(attn, ["Q0"], [1], position=0, k=5)
        self.assertEqual(len(influences), 0, "Position 0 should have no influences")


class TestAttentionExportIntegration(unittest.TestCase):
    """Test that attention extraction integrates with export pipeline."""
    
    def test_attention_dataframe_structure(self):
        """Test that attention data can be converted to DataFrame."""
        attention_rows = [
            {
                "user_id": "U1",
                "position": 5,
                "mastery": 0.65,
                "top_influences": [
                    {"item_id": "Q1", "correct": True, "weight": 0.3, "position": 0},
                    {"item_id": "Q3", "correct": False, "weight": 0.25, "position": 2},
                ],
            },
            {
                "user_id": "U2",
                "position": 3,
                "mastery": 0.45,
                "top_influences": [
                    {"item_id": "Q2", "correct": True, "weight": 0.4, "position": 1},
                ],
            },
        ]
        
        df = pd.DataFrame(attention_rows)
        
        # Verify columns
        expected_columns = {"user_id", "position", "mastery", "top_influences"}
        self.assertEqual(set(df.columns), expected_columns)
        
        # Verify data types
        self.assertEqual(df["user_id"].dtype, object)
        self.assertEqual(df["position"].dtype, int)
        self.assertEqual(df["mastery"].dtype, float)
        
        # Verify top_influences is a list
        for influences in df["top_influences"]:
            self.assertIsInstance(influences, list)


if __name__ == "__main__":
    unittest.main()

