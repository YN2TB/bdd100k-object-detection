"""Ultralytics smoke stops retain complete resume state."""
import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.train_ultralytics import (
    dataset_fingerprint_from_config,
    ensure_fresh_output,
    inspect_run_progress,
    reconcile_resume_progress,
    validate_resume_checkpoint,
    validate_resume_configuration,
)


class UltralyticsCheckpointTests(unittest.TestCase):
    def test_every_native_save_publishes_resume_provenance(self):
        import torch
        from types import SimpleNamespace
        from scripts import train_ultralytics as entry
        with TemporaryDirectory() as directory:
            last = Path(directory) / "last.pt"
            config = {"model_id": "yolo11s", "dataset_fingerprint": "a"}
            callback = entry.checkpoint_provenance_callback(config)
            for epoch in (0, 1):
                torch.save({"epoch": epoch, "optimizer": {}, "scaler": {},
                            "train_args": {"epochs": 50}}, last)
                callback(SimpleNamespace(last=last))
                saved = torch.load(last, weights_only=False)
                self.assertEqual(saved["bddcv_config"], config)
                self.assertEqual(saved["epoch"], epoch)

    def test_resume_rejects_checkpoint_without_dataset_provenance(self):
        config = {"model_id": "yolo11s", "batch": 8, "epochs": 50,
                  "imgsz": 640, "workers": 4, "seed": 0,
                  "dataset_fingerprint": "a"}
        checkpoint = {"train_args": {**config, "model": "yolo11s.pt"}}
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_resume_configuration(config, checkpoint, config)

    def test_complete_resume_state_reports_one_based_progress(self):
        checkpoint = {
            "epoch": 0, "optimizer": {"state": {}}, "scaler": {},
            "train_args": {"epochs": 50},
        }
        self.assertEqual(validate_resume_checkpoint(Path("last.pt"), checkpoint), 1)

    def test_stripped_checkpoint_is_rejected_for_resume(self):
        with self.assertRaisesRegex(ValueError, "optimizer"):
            validate_resume_checkpoint(Path("last.pt"), {
                "epoch": 0, "optimizer": None, "scaler": {},
                "train_args": {"epochs": 50},
            })

    def test_dataset_fingerprint_tracks_config_manifests_and_annotations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subset = root / "subset"
            annotations = subset / "annotations"
            annotations.mkdir(parents=True)
            (subset / "train_images.txt").write_text("train.jpg\n", encoding="utf-8")
            (subset / "val_images.txt").write_text("val.jpg\n", encoding="utf-8")
            (annotations / "instances_train.json").write_text("{}\n", encoding="utf-8")
            (annotations / "instances_val.json").write_text("{}\n", encoding="utf-8")
            config = root / "dataset.yaml"
            config.write_text(
                f"path: {subset}\ntrain: images/train\nval: images/val\n",
                encoding="utf-8",
            )

            original = dataset_fingerprint_from_config(config)
            (subset / "train_images.txt").write_text("changed.jpg\n", encoding="utf-8")
            self.assertNotEqual(original, dataset_fingerprint_from_config(config))

            (subset / "train_images.txt").write_text("train.jpg\n", encoding="utf-8")
            config.write_text(
                f"path: {subset}\ntrain: images/other\nval: images/val\n",
                encoding="utf-8",
            )
            self.assertNotEqual(original, dataset_fingerprint_from_config(config))

            config.write_text(
                f"path: {subset}\ntrain: images/train\nval: images/val\n",
                encoding="utf-8",
            )
            (annotations / "instances_val.json").write_text("changed\n", encoding="utf-8")
            self.assertNotEqual(original, dataset_fingerprint_from_config(config))

    def test_resume_requires_checkpoint_and_metadata_to_match(self):
        metadata = {
            "model_id": "yolo11s", "batch": 8, "epochs": 50,
            "imgsz": 640, "workers": 4, "seed": 0,
            "dataset_fingerprint": "dataset-a",
        }
        checkpoint = {
            "epoch": 0, "optimizer": {"state": {}}, "scaler": {},
            "train_args": {
                "epochs": 50, "batch": 16, "imgsz": 640,
                "workers": 4, "seed": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_resume_configuration(
                metadata,
                checkpoint,
                {**metadata},
            )

    def test_repeated_resume_accepts_marked_last_checkpoint_only_for_same_model(self):
        config = {
            "model_id": "yolo11s", "batch": 8, "epochs": 50,
            "imgsz": 640, "workers": 4, "seed": 0,
            "dataset_fingerprint": "dataset-a",
            "data_config": "/tmp/dataset.yaml",
        }
        checkpoint = {
            "epoch": 0, "optimizer": {"state": {}}, "scaler": {},
            "train_args": {
                "model": "/tmp/run/weights/last.pt", "data": "/tmp/dataset.yaml",
                "epochs": 50, "batch": 8, "imgsz": 640,
                "workers": 4, "seed": 0,
            },
            "bddcv_model_id": "yolo11s", "bddcv_config": config,
        }
        validate_resume_configuration(config, checkpoint, config)

        checkpoint["bddcv_model_id"] = "yolo11m"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_resume_configuration(config, checkpoint, config)

    def test_clean_return_uses_backend_progress_and_marks_early_exit(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["epoch", "loss"])
                writer.writerow([1, 0.5])
            checkpoint_path = out / "weights" / "last.pt"
            checkpoint_path.parent.mkdir()
            checkpoint_path.write_text("checkpoint", encoding="utf-8")

            evidence = inspect_run_progress(out, checkpoint_path, target_epochs=2)
            self.assertEqual(evidence["completed_epochs"], 0)
            self.assertEqual(evidence["status"], "incomplete")
            self.assertFalse(evidence["checkpoint_valid"])

    def test_clean_return_can_complete_from_csv_when_backend_strips_checkpoint(self):
        import torch

        with TemporaryDirectory() as directory:
            out = Path(directory)
            with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["epoch", "loss"])
                writer.writerow([1, 0.5])
                writer.writerow([2, 0.2])
            checkpoint_path = out / "weights" / "last.pt"
            checkpoint_path.parent.mkdir()
            torch.save({
                "epoch": -1, "optimizer": None, "scaler": None,
                "train_args": {"epochs": 2},
            }, checkpoint_path)

            evidence = inspect_run_progress(out, checkpoint_path, target_epochs=2)
            self.assertEqual(evidence["completed_epochs"], 2)
            self.assertEqual(evidence["status"], "complete")
            self.assertEqual(evidence["evidence"], "results.csv")

    def test_resume_reconciles_csv_to_one_based_checkpoint_progress(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "results.csv").write_text(
                "epoch,loss\n1,0.5\n3,0.1\n", encoding="utf-8"
            )
            report = reconcile_resume_progress(out, 1)
            self.assertEqual(report["kept_rows"], 1)
            self.assertEqual(report["stale_rows"], 1)
            self.assertIn("1,0.5", (out / "results.csv").read_text(encoding="utf-8"))
            self.assertNotIn("3,0.1", (out / "results.csv").read_text(encoding="utf-8"))

    def test_fresh_run_rejects_existing_checkpoint_or_log_artifacts(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "weights").mkdir()
            with self.assertRaisesRegex(FileExistsError, "existing"):
                ensure_fresh_output(out)

            clean = out / "clean"
            (clean / "logs").mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "existing"):
                ensure_fresh_output(clean)


if __name__ == "__main__":
    unittest.main()
