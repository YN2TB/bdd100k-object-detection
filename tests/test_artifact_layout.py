"""Regression checks for output routing; no GPU or downloads required."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("wrapper", ROOT / "scripts/train_with_resume.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class OutputRoutingTests(unittest.TestCase):
    def test_fresh_and_resume_route_to_same_explicit_run(self):
        with tempfile.TemporaryDirectory(prefix="bdd space '") as tmp:
            run = Path(tmp) / "runs/train/example"
            args = argparse.Namespace(mode="ultra", model="rtdetr-l.pt", run_dir=run,
                                      out=run, epochs=50, imgsz=640, batch=8,
                                      workers=4, name="example")
            captured = []
            oom_retries = []

            class FakeTrainer:
                def __init__(self, model):
                    self.model = model
                    self.callbacks = {}

                def add_callback(self, name, callback):
                    self.callbacks[name] = callback

                def train(self, **kwargs):
                    state = argparse.Namespace(_oom_retries=0)
                    self.callbacks["on_train_epoch_start"](state)
                    oom_retries.append(state._oom_retries)
                    captured.append((self.model, kwargs))

            fake = types.ModuleType("ultralytics")
            fake.RTDETR = FakeTrainer
            with patch.dict(sys.modules, ultralytics=fake), patch("bddcv.paths.prepare_ultralytics"):
                # Model execution is replaced; generated command and its arguments are real.
                for resume in (False, True):
                    command = wrapper.build_command(args, resume)
                    exec(command[-1], {})
            self.assertEqual(captured[0][1]["project"], str(run.parent))
            self.assertEqual(captured[0][1]["name"], "example")
            self.assertEqual(captured[1][1]["save_dir"], str(run))
            self.assertEqual(captured[1][0], str(run / "weights/last.pt"))
            self.assertEqual(Path(captured[0][0]), ROOT / "weights/rtdetr-l.pt")
            self.assertEqual(oom_retries, [3, 3])

    def test_relative_output_uses_callers_directory(self):
        from bddcv.paths import resolve_output, TRAIN_DIR
        with tempfile.TemporaryDirectory(prefix="bdd cwd ") as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                self.assertEqual(resolve_output(None, TRAIN_DIR / "frcnn"),
                                 ROOT / "runs/train/frcnn")
                self.assertEqual(resolve_output(Path("custom run"), TRAIN_DIR),
                                 Path(tmp) / "custom run")
            finally:
                os.chdir(previous)

    def test_bare_weight_goes_to_weights_but_explicit_path_is_preserved(self):
        from bddcv.paths import resolve_weights
        self.assertEqual(resolve_weights("yolo11s.pt"), ROOT / "weights/yolo11s.pt")
        self.assertEqual(resolve_weights("./custom.pt"), ROOT / "custom.pt")


if __name__ == "__main__":
    unittest.main()
