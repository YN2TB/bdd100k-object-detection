"""Supervisor progress comes only from compatible checkpoints."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scripts.train_with_resume import (
    checkpoint_epoch,
    epochs_done,
    run_checkpoint_path,
    validate_ultra_checkpoint,
    write_or_validate_model_metadata,
)
from bddcv.registry import get_model_spec


class SupervisorProgressTests(unittest.TestCase):
    def test_repeated_ultra_resume_requires_matching_checkpoint_provenance(self):
        config = {"model_id": "yolo11s", "epochs": 50, "batch": 8,
                  "imgsz": 640, "workers": 4, "seed": 0,
                  "data_config": "/tmp/data.yaml", "dataset_fingerprint": "a"}
        checkpoint = {"epoch": 1, "optimizer": {}, "scaler": {},
                      "train_args": {**config, "model": "/tmp/weights/last.pt",
                                     "data": "/tmp/data.yaml"},
                      "bddcv_model_id": "yolo11s", "bddcv_config": config}
        self.assertEqual(validate_ultra_checkpoint(checkpoint, config), 2)
        checkpoint["bddcv_config"] = {**config, "dataset_fingerprint": "b"}
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_ultra_checkpoint(checkpoint, config)

    def test_csv_without_checkpoint_never_proves_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results.csv").write_text("epoch,loss\n50,0.1\n")
            self.assertEqual(
                epochs_done(root / "results.csv", root / "last.pt", "frcnn"), 0
            )

    def test_rfdetr_uses_full_lightning_checkpoint(self):
        root = Path("runs/smoke/rfdetr-small")
        self.assertEqual(run_checkpoint_path(root, "rfdetr"), root / "last.ckpt")
        self.assertEqual(run_checkpoint_path(root, "ultra"), root / "weights/last.pt")
        self.assertEqual(run_checkpoint_path(root, "frcnn"), root / "last.pt")

    def test_incomplete_rfdetr_checkpoint_never_proves_progress(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"epoch": 5}, root / "last.ckpt")
            self.assertEqual(
                epochs_done(root / "results.csv", root / "last.ckpt", "rfdetr"), 0
            )

    def test_existing_compatible_model_metadata_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            payload = {"model_id": "yolo11s", "batch": 8, "epochs": 50,
                       "imgsz": 640, "workers": 4, "seed": 0, "marker": "keep"}
            path.write_text(json.dumps(payload))
            created = write_or_validate_model_metadata(
                path, get_model_spec("yolo11s"),
                batch=8, epochs=50, imgsz=640, workers=4, seed=0,
            )
            self.assertFalse(created)
            self.assertEqual(json.loads(path.read_text())["marker"], "keep")

    def test_trainer_canonical_metadata_matches_supervisor_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps({
                "model_id": "rfdetr-small", "physical_batch": 8,
                "epochs": 50, "resolution": 512, "workers": 4, "seed": 0,
            }))
            self.assertFalse(write_or_validate_model_metadata(
                path, get_model_spec("rfdetr-small"),
                batch=8, epochs=50, imgsz=512, workers=4, seed=0,
            ))

    def test_stripped_ultralytics_checkpoint_cannot_claim_target_without_csv_rows(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "last.pt"
            torch.save({
                "epoch": -1, "optimizer": None, "scaler": None,
                "train_args": {"epochs": 2},
            }, checkpoint)
            (root / "results.csv").write_text(
                "epoch,metrics/mAP50-95\n1,0.1\n", encoding="utf-8"
            )
            self.assertIsNone(checkpoint_epoch(checkpoint, "ultra"))
            self.assertEqual(
                epochs_done(root / "results.csv", checkpoint, "ultra"), 1
            )

    def test_ultralytics_resume_rejects_checkpoint_train_args_drift(self):
        with self.assertRaisesRegex(ValueError, "batch"):
            validate_ultra_checkpoint({
                "epoch": 0, "optimizer": {"state": {}}, "scaler": {},
                "train_args": {
                    "model": "yolo11s.pt", "epochs": 50, "imgsz": 640,
                    "batch": 8, "seed": 0,
                },
            }, {
                "model_id": "yolo11s", "model": "yolo11s.pt", "epochs": 50,
                "imgsz": 640, "batch": 16, "seed": 0,
            })

    def test_terminal_ultralytics_checkpoint_still_checks_train_args(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "last.pt"
            torch.save({
                "epoch": -1, "optimizer": None, "scaler": None,
                "train_args": {
                    "model": "yolo11s.pt", "epochs": 2, "imgsz": 640,
                    "batch": 8, "seed": 0,
                },
            }, checkpoint)
            (root / "results.csv").write_text(
                "epoch,metrics/mAP50-95\n1,0.1\n2,0.2\n", encoding="utf-8"
            )
            from scripts.train_with_resume import _validate_ultra_artifacts
            with self.assertRaisesRegex(ValueError, "batch"):
                _validate_ultra_artifacts(
                    checkpoint, root / "results.csv",
                    {"model_id": "yolo11s", "model": "yolo11s.pt",
                     "epochs": 2, "imgsz": 640, "batch": 16, "seed": 0},
                    2,
                )


if __name__ == "__main__":
    unittest.main()
