# ABOUTME: Validates EDM dataset construction for Wide & Deep IRT.
# ABOUTME: Ensures samples are created with expected tensor shapes.

from datetime import datetime, timezone
from pathlib import Path
import json
import unittest

import pandas as pd
import torch

from src.wd_irt.datasets import EdmClickstreamDataset, EdmDatasetPaths
from src.wd_irt.features import FeatureConfig


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class EdmDatasetTest(unittest.TestCase):
    def test_edm_dataset_single_sample(self):
        tmp_path = Path(self._tmp_dir.name)
        events = pd.DataFrame(
            [
                {
                    "user_id": "stu1",
                    "item_id": "hist1",
                    "skill_ids": ["k1"],
                    "timestamp": _ts("2021-01-01T00:00:00"),
                    "correct": 1,
                    "action_sequence_id": "A1",
                    "latency_ms": 1200,
                    "help_requested": False,
                },
                {
                    "user_id": "stu1",
                    "item_id": "hist2",
                    "skill_ids": ["k2"],
                    "timestamp": _ts("2021-01-01T01:00:00"),
                    "correct": 0,
                    "action_sequence_id": "A2",
                    "latency_ms": 2200,
                    "help_requested": True,
                },
            ]
        )
        events_path = tmp_path / "events.parquet"
        events.to_parquet(events_path)

        assignment_details = pd.DataFrame(
            [
                {
                    "assignment_log_id": "UT1",
                    "student_id": "stu1",
                    "assignment_start_time": _ts("2021-01-02T00:00:00").timestamp(),
                },
                {
                    "assignment_log_id": "A1",
                    "student_id": "stu1",
                    "assignment_start_time": _ts("2021-01-01T00:00:00").timestamp(),
                },
                {
                    "assignment_log_id": "A2",
                    "student_id": "stu1",
                    "assignment_start_time": _ts("2021-01-01T01:00:00").timestamp(),
                },
            ]
        )
        assignment_details_path = tmp_path / "assignment_details.csv"
        assignment_details.to_csv(assignment_details_path, index=False)

        relationships = pd.DataFrame(
            [
                {"unit_test_assignment_log_id": "UT1", "in_unit_assignment_log_id": "A1"},
                {"unit_test_assignment_log_id": "UT1", "in_unit_assignment_log_id": "A2"},
            ]
        )
        relationships_path = tmp_path / "assignment_relationships.csv"
        relationships.to_csv(relationships_path, index=False)

        unit_test_scores = pd.DataFrame(
            [
                {"assignment_log_id": "UT1", "problem_id": "PFUT", "score": 1},
            ]
        )
        scores_path = tmp_path / "training_unit_test_scores.csv"
        unit_test_scores.to_csv(scores_path, index=False)

        problem_details = pd.DataFrame(
            [
                {
                    "problem_id": "hist1",
                    "problem_type": "Multiple Choice",
                    "problem_text_bert_pca": "[0.1,0.2]",
                },
                {
                    "problem_id": "hist2",
                    "problem_type": "Open Response",
                    "problem_text_bert_pca": "[0.3,0.4]",
                },
                {
                    "problem_id": "PFUT",
                    "problem_type": "Multiple Choice",
                    "problem_text_bert_pca": "[0.5,0.6]",
                },
            ]
        )
        problem_details_path = tmp_path / "problem_details.csv"
        problem_details.to_csv(problem_details_path, index=False)

        splits = {"train": ["stu1"], "val": [], "test": []}
        split_manifest_path = tmp_path / "splits.json"
        split_manifest_path.write_text(json.dumps(splits))

        paths = EdmDatasetPaths(
            events=events_path,
            assignment_details=assignment_details_path,
            assignment_relationships=relationships_path,
            unit_test_scores=scores_path,
            problem_details=problem_details_path,
            split_manifest=split_manifest_path,
        )

        dataset = EdmClickstreamDataset(
            split="train",
            paths=paths,
            feature_config=FeatureConfig(seq_len=4, bert_dim=2),
            max_samples=4,
        )

        self.assertEqual(len(dataset), 1)
        inputs, label, meta = dataset[0]
        self.assertEqual(inputs["history_actions"].shape, (4,))
        self.assertEqual(inputs["history_item_embeddings"].shape, (4, 2))
        self.assertTrue(torch.isclose(label, torch.tensor(1.0)))
        self.assertEqual(meta["student_id"], "stu1")

    def setUp(self) -> None:
        from tempfile import TemporaryDirectory

        self._tmp_dir = TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
