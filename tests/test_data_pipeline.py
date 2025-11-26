# ABOUTME: Validates the data pipeline for canonical learning events.
# ABOUTME: Ensures deterministic splits and schema normalization.

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.common.data_pipeline import generate_user_splits, prepare_learning_events


class EdmDataPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.tmpdir.name)
        self._write_csv(
            "action_logs.csv",
            """assignment_log_id,timestamp,problem_id,max_attempts,available_core_tutoring,score_viewable,continuous_score_viewable,action,hint_id,explanation_id
A1,1600000000,P1,3,answer,1,1,assignment_started,,
A1,1600000010,P1,3,answer,1,1,problem_started,,
A1,1600000020,P1,3,answer,1,1,hint_requested,H1,
A1,1600000030,P1,3,answer,1,1,wrong_response,,
A1,1600000040,P1,3,answer,1,1,correct_response,,
A2,1600000100,P2,3,answer,1,1,assignment_started,,
A2,1600000110,P2,3,answer,1,1,problem_started,,
A2,1600000120,P2,3,answer,1,1,correct_response,,
""",
        )
        self._write_csv(
            "assignment_details.csv",
            """assignment_log_id,teacher_id,class_id,student_id,sequence_id,assignment_release_date,assignment_due_date,assignment_start_time,assignment_end_time
A1,T1,C1,S1,SEQ1,1599999000,1600009000,1599999900,1600010000
A2,T1,C1,S2,SEQ2,1599999000,1600009000,1599999900,1600010000
""",
        )
        self._write_csv(
            "problem_details.csv",
            """problem_id,problem_multipart_id,problem_multipart_position,problem_type,problem_skill_code,problem_skill_description,problem_contains_image,problem_contains_equation,problem_contains_video,problem_text_bert_pca
P1,PM1,1,Multiple Choice,SK1,Skill One,0,0,0,[0,0]
P2,PM2,1,Multiple Choice,SK2,Skill Two,0,0,0,[0,0]
""",
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_csv(self, name: str, data: str) -> None:
        (self.raw_dir / name).write_text(data.strip() + "\n", encoding="utf-8")

    def test_prepare_learning_events_schema(self) -> None:
        df = prepare_learning_events(self.raw_dir, dataset="edm_cup_2023")
        expected_columns = [
            "user_id",
            "item_id",
            "skill_ids",
            "timestamp",
            "correct",
            "action_sequence_id",
            "latency_ms",
            "help_requested",
        ]
        self.assertEqual(expected_columns, list(df.columns))
        self.assertEqual(3, len(df))
        self.assertTrue(all(isinstance(v, list) for v in df["skill_ids"]))
        s1_rows = df[df["user_id"] == "S1"].sort_values("timestamp")
        self.assertTrue((s1_rows["timestamp"].diff().fillna(pd.Timedelta(seconds=0)) >= pd.Timedelta(seconds=0)).all())
        wrong_event = s1_rows.iloc[0]
        self.assertEqual(0, wrong_event["correct"])
        self.assertTrue(wrong_event["help_requested"])
        self.assertGreaterEqual(wrong_event["latency_ms"], 0)

    def test_generate_user_splits_is_deterministic(self) -> None:
        users = ["S1", "S2", "S3", "S4", "S5"]
        splits_a = generate_user_splits(users, train_ratio=0.6, val_ratio=0.2, seed=7)
        splits_b = generate_user_splits(users, train_ratio=0.6, val_ratio=0.2, seed=7)
        self.assertEqual(splits_a, splits_b)
        all_users = set(users)
        self.assertEqual(all_users, set(splits_a["train"]) | set(splits_a["val"]) | set(splits_a["test"]))
        self.assertTrue(set(splits_a["train"]).isdisjoint(splits_a["val"]))
        self.assertTrue(set(splits_a["train"]).isdisjoint(splits_a["test"]))
        self.assertTrue(set(splits_a["val"]).isdisjoint(splits_a["test"]))


class AssistmentsDataPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.tmpdir.name)
        self._write_csv(
            "skill_builder_data.csv",
            """order_id,assignment_id,user_id,assistment_id,problem_id,original,correct,attempt_count,ms_first_response,tutor_mode,answer_type,sequence_id,student_class_id,position,type,base_sequence_id,skill_id,skill_name,teacher_id,school_id,hint_count,hint_total,overlap_time,template_id,answer_id,answer_text,first_action,bottom_hint,opportunity,opportunity_original
100,500,student_a,2001,3001,1,0,2,1000,tutor,algebra,1,10,1,MasterySection,1,7,Decimals,111,222,1,0,0,1,2,,first_action,0,1,1
105,500,student_a,2002,3002,1,1,1,500,tutor,algebra,1,10,1,MasterySection,1,,Decimals,111,222,0,0,0,1,2,,first_action,1,2,2
200,501,student_b,2003,3003,1,1,1,200,tutor,algebra,1,11,2,MasterySection,1,8,Fractions,111,222,0,0,0,1,2,,first_action,0,1,1
""",
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_csv(self, name: str, data: str) -> None:
        (self.raw_dir / name).write_text(data.strip() + "\n", encoding="utf-8")

    def test_prepare_learning_events_assistments(self) -> None:
        df = prepare_learning_events(self.raw_dir, dataset="assistments_skill_builder")
        self.assertEqual(3, len(df))
        self.assertTrue(df["timestamp"].is_monotonic_increasing)
        student_a = df[df["user_id"] == "student_a"]
        self.assertEqual(2, len(student_a))
        self.assertEqual(["3001", "3002"], student_a["item_id"].tolist())
        # First record correct=0 with help requested
        self.assertEqual(0, int(student_a.iloc[0]["correct"]))
        self.assertTrue(student_a.iloc[0]["help_requested"])
        # Skill ids should fallback to skill_name when skill_id missing
        skills_second = student_a.iloc[1]["skill_ids"]
        self.assertIn("Decimals", skills_second)
        # Latency should match ms_first_response
        self.assertEqual(1000, int(student_a.iloc[0]["latency_ms"]))


if __name__ == "__main__":
    unittest.main()
