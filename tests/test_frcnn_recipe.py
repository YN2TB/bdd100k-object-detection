"""Faster R-CNN CLI metadata cannot diverge from its implemented recipe."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_frcnn.py"
SPEC = importlib.util.spec_from_file_location("train_frcnn", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class FasterRcnnRecipeTests(unittest.TestCase):
    def test_implemented_recipe_is_accepted(self):
        MODULE.validate_recipe(SimpleNamespace(
            model_id="frcnn-r50-fpn-v2", optimizer="SGD",
            schedule="warmup+cosine", imgsz=640,
        ))

    def test_unimplemented_optimizer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "optimizer"):
            MODULE.validate_recipe(SimpleNamespace(
                model_id="frcnn-r50-fpn-v2", optimizer="AdamW",
                schedule="warmup+cosine", imgsz=640,
            ))

    def test_unimplemented_resolution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "imgsz"):
            MODULE.validate_recipe(SimpleNamespace(
                model_id="frcnn-r50-fpn-v2", optimizer="SGD",
                schedule="warmup+cosine", imgsz=512,
            ))

    def test_training_config_records_supervisor_compatibility_fields(self):
        config = MODULE.training_config(SimpleNamespace(
            model_id="frcnn-r50-fpn-v2", batch=8, accumulate=1, imgsz=640,
            optimizer="SGD", lr=0.01, schedule="warmup+cosine", epochs=50,
            warmup_iters=500, limit_train=0, workers=4, seed=0,
            validation_batch=4,
        ), "fingerprint")
        self.assertEqual(config["epochs"], 50)
        self.assertEqual(config["seed"], 0)
        self.assertEqual(config["batch"], 8)
        self.assertEqual(config["imgsz"], 640)
        self.assertEqual(config["validation_batch"], 4)


if __name__ == "__main__":
    unittest.main()
