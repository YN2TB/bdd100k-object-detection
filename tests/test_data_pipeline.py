"""Small regression checks for the dataset preparation pipeline."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import build_labels, prepare_data, verify_labels


class DataPipelineTests(unittest.TestCase):
    def test_manifest_drift_stops_before_subset_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            originals = {"train": "train.jpg\n", "val": "val.jpg\n", "test": "test.jpg\n"}
            for split, content in originals.items():
                (dataset / f"{split}_images.txt").write_text(content)

            candidates = {"train": ["different.jpg"], "val": ["val.jpg"], "test": ["test.jpg"]}
            with patch.object(prepare_data, "DATASET_DIR", dataset):
                with self.assertRaises(SystemExit):
                    prepare_data.reject_manifest_drift(candidates)

            for split, content in originals.items():
                self.assertEqual((dataset / f"{split}_images.txt").read_text(), content)

    def test_build_rejects_duplicate_or_missing_raw_records(self):
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory) / "images"
            (images / "train").mkdir(parents=True)
            for name in ("a.jpg", "b.jpg"):
                (images / "train" / name).touch()

            with patch.object(build_labels, "IMAGES_DIR", images):
                with self.assertRaisesRegex(SystemExit, "duplicate filename"):
                    build_labels.validate_inputs("train", ["a.jpg", "a.jpg"])

                records = iter([{"name": "a.jpg"}])
                with patch.object(build_labels, "stream_records", return_value=records):
                    with self.assertRaisesRegex(SystemExit, "coverage mismatch"):
                        build_labels.validate_inputs("train", ["a.jpg", "b.jpg"])

                records = iter([{"name": "a.jpg"}, {"name": "a.jpg"}, {"name": "b.jpg"}])
                with patch.object(build_labels, "stream_records", return_value=records):
                    with self.assertRaisesRegex(SystemExit, "coverage mismatch"):
                        build_labels.validate_inputs("train", ["a.jpg", "b.jpg"])

    def test_build_removes_identical_boxes_from_both_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            subset = Path(directory)
            image_dir = subset / "images" / "train"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (100, 50)).save(image_dir / "a.jpg")
            (subset / "train_images.txt").write_text("a.jpg\n", encoding="utf-8")
            record = {"name": "a.jpg"}
            boxes = [(2, 10.0, 10.0, 30.0, 30.0)] * 2

            with (
                patch.object(build_labels, "SUBSET_DIR", subset),
                patch.object(build_labels, "IMAGES_DIR", subset / "images"),
                patch.object(build_labels, "LABELS_DIR", subset / "labels"),
                patch.object(build_labels, "ANN_DIR", subset / "annotations"),
                patch.object(build_labels, "all_records", side_effect=lambda: iter([record])),
                patch.object(build_labels, "detection_boxes", return_value=boxes),
            ):
                stats = build_labels.build("train")

            yolo_lines = (subset / "labels" / "train" / "a.txt").read_text().splitlines()
            coco = json.loads(
                (subset / "annotations" / "instances_train.json").read_text()
            )
            self.assertEqual(len(yolo_lines), 1)
            self.assertEqual(len(coco["annotations"]), 1)
            self.assertEqual(stats["duplicates"], 1)

    def test_empty_coco_image_requires_an_empty_yolo_label(self):
        with tempfile.TemporaryDirectory() as directory:
            subset = Path(directory)
            (subset / "annotations").mkdir()
            labels = subset / "labels" / "val"
            labels.mkdir(parents=True)
            (subset / "annotations" / "instances_val.json").write_text(json.dumps({
                "images": [{"id": 1, "file_name": "empty.jpg", "width": 100, "height": 50}],
                "annotations": [],
            }), encoding="utf-8")
            label = labels / "empty.txt"
            label.write_text("", encoding="utf-8")

            with patch.object(verify_labels, "SUBSET_DIR", subset):
                self.assertTrue(verify_labels.numeric_check("val"))
                label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
                self.assertFalse(verify_labels.numeric_check("val"))

    def test_verify_main_returns_failure_status(self):
        with (
            patch.object(verify_labels, "numeric_check", side_effect=[True, False, True]),
            patch.object(verify_labels, "montage", return_value=Path("montage.png")),
        ):
            self.assertEqual(verify_labels.main(), 1)


if __name__ == "__main__":
    unittest.main()
