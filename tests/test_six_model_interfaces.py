"""CPU-only regression tests for the six-model implementation contracts."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bddcv.checkpointing import reconcile_results_csv  # noqa: E402
from bddcv.constants import DET_CLASSES  # noqa: E402
from bddcv.evaluation import evaluate  # noqa: E402
from bddcv.prediction import (  # noqa: E402
    rfdetr_result_to_coco, ultralytics_result_to_coco, write_coco_predictions,
)
from bddcv.profiling import (  # noqa: E402
    PROFILE_SCHEMA_VERSION, TIMING_SCHEMA, TIMING_SCHEMA_VERSION,
    ProfileController, ProfileOutcome, ProfileStatus, classify_profile_output, select_profile,
)
from bddcv.registry import ModelResolutionError, resolve_model_id, list_models  # noqa: E402
from bddcv.rfdetr import build_adapter, validate_adapter  # noqa: E402


class RegistryTests(unittest.TestCase):
    def test_approved_roster_and_generic_checkpoint_guard(self):
        self.assertEqual(
            {spec.model_id for spec in list_models()},
            {"yolo11s", "yolo11m", "yolo26s", "frcnn-r50-fpn-v2", "rtdetr-l", "rfdetr-small"},
        )
        with self.assertRaises(ModelResolutionError):
            resolve_model_id(checkpoint="best.pt")
        self.assertEqual(resolve_model_id("rtdetr-l", "best.pt"), "rtdetr-l")


class AdapterTests(unittest.TestCase):
    def test_adapter_links_bytes_and_preserves_categories(self):
        with tempfile.TemporaryDirectory(prefix="bdd rfdetr ") as raw:
            root = Path(raw) / "source"
            output = Path(raw) / "adapter"
            for split, name in (("train", "a.jpg"), ("val", "b.jpg")):
                (root / "images" / split).mkdir(parents=True, exist_ok=True)
                (root / "annotations").mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4), (split == "train", 0, 0)).save(root / "images" / split / name)
                (root / f"{split}_images.txt").write_text(name + "\n", encoding="utf-8")
                (root / "annotations" / f"instances_{split}.json").write_text(json.dumps({
                    "images": [{"id": 1, "file_name": name, "width": 4, "height": 4}],
                    "annotations": [],
                    "categories": [{"id": i + 1, "name": value} for i, value in enumerate(DET_CLASSES)],
                }), encoding="utf-8")
            report = build_adapter(root, output)
            self.assertTrue(report["valid"])
            self.assertTrue(validate_adapter(root, output).valid)
            self.assertEqual((root / "images/train/a.jpg").read_bytes(),
                             (output / "train/a.jpg").read_bytes())


class PredictionAndEvaluationTests(unittest.TestCase):
    def test_backends_map_ids_and_accept_empty_outputs(self):
        ids = {"frame.jpg": 17}

        class Boxes:
            xyxy = [[1, 2, 5, 7]]
            conf = [0.8]
            cls = [2]

        class Result:
            path = "/tmp/frame.jpg"
            boxes = Boxes()

        yolo = ultralytics_result_to_coco(Result(), ids)
        self.assertEqual(yolo[0]["image_id"], 17)
        self.assertEqual(yolo[0]["category_id"], 3)
        self.assertEqual(rfdetr_result_to_coco({"path": "frame.jpg", "xyxy": [],
                                                 "confidence": [], "class_id": []}, ids), [])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = {
                "images": [{"id": 17, "file_name": "frame.jpg", "width": 10, "height": 10}],
                "annotations": [{"id": 1, "image_id": 17, "category_id": 3,
                                 "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0}],
                "categories": [{"id": i + 1, "name": name} for i, name in enumerate(DET_CLASSES)],
            }
            (root / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
            pred = root / "pred.json"
            write_coco_predictions([], pred)
            result = evaluate(root / "gt.json", pred)
            self.assertEqual(result["overall"]["mAP50_95"], 0.0)


class ProfilingAndResumeTests(unittest.TestCase):
    def test_profile_variants_use_distinct_output_directories(self):
        controller = ProfileController(Path("runs/profile"))
        self.assertEqual(
            controller.candidate_output("yolo11s", 8, "workers-4-prefetch-2"),
            Path("runs/profile/yolo11s/workers-4-prefetch-2/batch-8"),
        )

    def test_statuses_and_memory_selection(self):
        self.assertEqual(classify_profile_output("x", 1, returncode=1,
                                                 stdout="loss nan").status,
                         ProfileStatus.NONFINITE)
        self.assertEqual(classify_profile_output("x", 1, returncode=1,
                                                 stdout="validation OOM").status,
                         ProfileStatus.VALIDATION_FAILURE)
        selected = select_profile([
            ProfileOutcome("x", 8, "success", images_per_second=100,
                           peak_memory_bytes=850, total_memory_bytes=1000,
                           measured_steps=1, measured_elapsed_seconds=1.0,
                           profile_schema_version=PROFILE_SCHEMA_VERSION,
                           timing_schema=TIMING_SCHEMA,
                           timing_schema_version=TIMING_SCHEMA_VERSION),
            ProfileOutcome("x", 16, "success", images_per_second=99,
                           peak_memory_bytes=600, total_memory_bytes=1000,
                           measured_steps=1, measured_elapsed_seconds=1.0,
                           profile_schema_version=PROFILE_SCHEMA_VERSION,
                           timing_schema=TIMING_SCHEMA,
                           timing_schema_version=TIMING_SCHEMA_VERSION),
            ProfileOutcome("x", 32, "success", images_per_second=120,
                           peak_memory_bytes=950, total_memory_bytes=1000,
                           measured_steps=1, measured_elapsed_seconds=1.0,
                           profile_schema_version=PROFILE_SCHEMA_VERSION,
                           timing_schema=TIMING_SCHEMA,
                           timing_schema_version=TIMING_SCHEMA_VERSION),
        ])
        self.assertEqual(selected.batch, 16)

    def test_selection_ignores_outcomes_from_incompatible_timing_schema(self):
        selected = select_profile([
            ProfileOutcome(
                "x", 32, "success", images_per_second=900,
                peak_memory_bytes=600, total_memory_bytes=1000,
                profile_schema_version=PROFILE_SCHEMA_VERSION - 1,
                timing_schema=TIMING_SCHEMA,
                timing_schema_version=TIMING_SCHEMA_VERSION - 1,
                measured_steps=1, measured_elapsed_seconds=1.0,
            ),
            ProfileOutcome(
                "x", 8, "success", images_per_second=100,
                peak_memory_bytes=600, total_memory_bytes=1000,
                profile_schema_version=PROFILE_SCHEMA_VERSION,
                timing_schema=TIMING_SCHEMA,
                timing_schema_version=TIMING_SCHEMA_VERSION,
                measured_steps=1, measured_elapsed_seconds=1.0,
            ),
        ])
        self.assertEqual(selected.batch, 8)

    def test_csv_ahead_of_checkpoint_is_preserved_as_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            path.write_text("epoch,loss\n1,0.5\n3,0.1\n", encoding="utf-8")
            report = reconcile_results_csv(path, 1)
            self.assertEqual(report["stale_rows"], 1)
            self.assertIn("3,0.1", Path(str(path) + ".stale").read_text(encoding="utf-8"))
            self.assertNotIn("3,0.1", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
