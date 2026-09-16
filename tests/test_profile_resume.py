"""Completed profiling variants can be safely reused during aggregation."""
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.profiling import PROFILE_SCHEMA_VERSION, TIMING_SCHEMA, TIMING_SCHEMA_VERSION
from scripts.profile_models import load_existing_outcome


class ProfileResumeTests(unittest.TestCase):
    def test_load_existing_outcome_reads_structured_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcome.json"
            path.write_text(json.dumps({
                "model_id": "yolo11s", "batch": 32, "status": "success",
                "images_per_second": 90.0,
                "measured_elapsed_seconds": 10.0,
                "measured_steps": 30,
                "profile_schema_version": PROFILE_SCHEMA_VERSION,
                "timing_schema": TIMING_SCHEMA,
                "timing_schema_version": TIMING_SCHEMA_VERSION,
            }))
            outcome = load_existing_outcome(path)
            self.assertIsNotNone(outcome)
            self.assertEqual(outcome.model_id, "yolo11s")
            self.assertEqual(outcome.images_per_second, 90.0)

    def test_load_existing_outcome_rejects_legacy_timing_without_mutating_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcome.json"
            original = json.dumps({
                "model_id": "yolo11s", "batch": 32, "status": "success",
                "images_per_second": 900.0,
            })
            path.write_text(original, encoding="utf-8")
            self.assertIsNone(load_existing_outcome(path))
            self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
