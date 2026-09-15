"""Focused CPU tests for the two hand-written CNN detectors."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bddcv.cnn_detector import (
    BOXES_PER_CELL, GRID_SIZE, build_cnn_detector, decode_detections,
    detection_loss,
)
from bddcv.constants import DET_CLASSES
from bddcv.registry import PRIMARY_MODEL_IDS


class CustomCNNTests(unittest.TestCase):
    def test_primary_roster_has_two_custom_cnns_and_yolo(self):
        self.assertEqual(PRIMARY_MODEL_IDS, ("simple-cnn", "complex-cnn", "yolo11s"))

    def test_models_share_output_shape_but_differ_in_size(self):
        outputs = {}
        parameters = {}
        for model_id in ("simple-cnn", "complex-cnn"):
            model = build_cnn_detector(model_id).eval()
            with torch.no_grad():
                outputs[model_id] = model(torch.zeros(1, 3, 128, 128)).shape
            parameters[model_id] = sum(value.numel() for value in model.parameters())
        expected = (1, BOXES_PER_CELL, 5 + len(DET_CLASSES), *GRID_SIZE)
        self.assertEqual(outputs["simple-cnn"], expected)
        self.assertEqual(outputs["complex-cnn"], expected)
        self.assertLess(parameters["simple-cnn"], parameters["complex-cnn"])

    def test_detection_loss_is_finite(self):
        output = torch.zeros(1, BOXES_PER_CELL, 5 + len(DET_CLASSES), *GRID_SIZE)
        target = {
            "objectness": torch.zeros(1, BOXES_PER_CELL, *GRID_SIZE),
            "boxes": torch.zeros(1, BOXES_PER_CELL, 4, *GRID_SIZE),
            "classes": torch.full((1, BOXES_PER_CELL, *GRID_SIZE), -1, dtype=torch.long),
        }
        target["objectness"][0, 0, 2, 3] = 1
        target["boxes"][0, 0, :, 2, 3] = torch.tensor([0.5, 0.5, 0.2, 0.1])
        target["classes"][0, 0, 2, 3] = 2
        self.assertTrue(torch.isfinite(detection_loss(output, target)["total"]))

    def test_decoder_returns_coco_box(self):
        output = torch.full((BOXES_PER_CELL, 5 + len(DET_CLASSES), *GRID_SIZE), -10.0)
        output[0, 0, 2, 3] = 10.0
        output[0, 1:5, 2, 3] = 0.0
        output[0, 5 + 2, 2, 3] = 10.0
        records = decode_detections(output, image_id=7, width=1280, height=720)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["image_id"], 7)
        self.assertEqual(records[0]["category_id"], 3)
        self.assertTrue(all(value >= 0 for value in records[0]["bbox"]))


if __name__ == "__main__":
    unittest.main()
