"""RF-DETR isolation, model selection, and adapter integrity contracts."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bddcv.constants import DET_CLASSES
from bddcv.rfdetr import RFDETRAdapter, RFDETRAdapterError, isolated_python
from bddcv.paths import CACHE_DIR, prepare_rfdetr_runtime
from scripts.train_rfdetr import build_train_kwargs, validate_full_checkpoint


class RFDETRContractTests(unittest.TestCase):
    def test_runtime_uses_project_cache(self):
        with patch.dict(os.environ, {}, clear=True):
            prepare_rfdetr_runtime()
            self.assertEqual(os.environ["RF_HOME"], str(CACHE_DIR / "rfdetr"))
            self.assertEqual(os.environ["HF_HOME"], str(CACHE_DIR / "rfdetr/huggingface"))
            self.assertEqual(os.environ["MPLCONFIGDIR"], str(CACHE_DIR / "matplotlib"))

    def test_explicit_venv_interpreter_keeps_venv_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "python-real"
            real.write_text("")
            venv = root / "venv-python"
            venv.symlink_to(real)
            self.assertEqual(isolated_python(python=venv), venv.absolute())

    def test_adapter_constructs_small_model(self):
        class Small:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        adapter = object.__new__(RFDETRAdapter)
        adapter.model_id = "rfdetr-small"
        adapter.checkpoint = Path("weights/rfdetr/rf-detr-small.pth")
        with patch.object(RFDETRAdapter, "_load_class", return_value=Small):
            model = adapter.construct()
        self.assertEqual(model.kwargs["pretrain_weights"], str(adapter.checkpoint))

    def test_prediction_loads_trained_checkpoint_through_native_loader(self):
        class Small:
            @classmethod
            def from_checkpoint(cls, path):
                return ("loaded", path)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_best_total.pth"
            checkpoint.write_bytes(b"fixture")
            adapter = object.__new__(RFDETRAdapter)
            adapter.model_id = "rfdetr-small"
            adapter.checkpoint = checkpoint
            with patch.object(RFDETRAdapter, "_load_class", return_value=Small):
                self.assertEqual(
                    adapter.construct(for_training=False),
                    ("loaded", str(checkpoint)),
                )

    def test_duplicate_coco_image_ids_are_rejected_before_adapter_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            out = Path(directory) / "adapter"
            (root / "annotations").mkdir(parents=True)
            for split, names in (("train", ["a.jpg", "b.jpg"]), ("val", ["c.jpg"])):
                (root / "images" / split).mkdir(parents=True)
                (root / f"{split}_images.txt").write_text("\n".join(names) + "\n")
                for name in names:
                    (root / "images" / split / name).write_bytes(name.encode())
                images = [{"id": 1, "file_name": name} for name in names]
                (root / "annotations" / f"instances_{split}.json").write_text(json.dumps({
                    "images": images, "annotations": [],
                    "categories": [{"id": i + 1, "name": name}
                                   for i, name in enumerate(DET_CLASSES)],
                }))
            from bddcv.rfdetr import build_adapter
            with self.assertRaises(RFDETRAdapterError):
                build_adapter(root, out)
            self.assertFalse(out.exists())

    def test_training_uses_native_small_recipe_and_full_resume_path(self):
        out = Path("runs/smoke/rfdetr-small")
        values = build_train_kwargs(
            adapter=Path("data/adapters/rfdetr"), out=out, epochs=50,
            batch=2, workers=4, accumulation=3, resume=out / "last.ckpt",
        )
        self.assertEqual(values["output_dir"], str(out))
        self.assertEqual(values["resume"], str(out / "last.ckpt"))
        self.assertEqual(values["epochs"], 50)
        self.assertEqual(values["batch_size"], 2)
        self.assertEqual(values["grad_accum_steps"], 3)
        self.assertEqual(values["eval_batch_size"], 1)
        self.assertEqual(values["seed"], 0)
        self.assertFalse(values["run_test"])
        self.assertFalse(values["early_stopping"])
        self.assertEqual(values["resolution"], 512)
        self.assertEqual(values["checkpoint_interval"], 1)
        self.assertEqual(values["device"], "cuda")

    def test_lightweight_or_incomplete_checkpoint_cannot_resume(self):
        with self.assertRaises(RFDETRAdapterError):
            validate_full_checkpoint(Path("checkpoint_best_total.pth"), {})
        with self.assertRaises(RFDETRAdapterError):
            validate_full_checkpoint(Path("last.ckpt"), {"epoch": 0})


if __name__ == "__main__":
    unittest.main()
