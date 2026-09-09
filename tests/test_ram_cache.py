"""RAM-cache trials must improve throughput without exceeding memory budgets."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bddcv import profiling


def outcome(rate=100, ram=40, **metadata):
    return profiling.ProfileOutcome(
        'yolo11s', 8, 'success', measured_steps=30,
        measured_elapsed_seconds=240 / rate,
        profile_schema_version=profiling.PROFILE_SCHEMA_VERSION,
        timing_schema=profiling.TIMING_SCHEMA,
        timing_schema_version=profiling.TIMING_SCHEMA_VERSION,
        total_memory_bytes=1000, peak_total_memory_bytes=500,
        metadata={'host_ram_peak_percent': ram, **metadata},
    )


class RamCacheTests(unittest.TestCase):
    def test_trial_requires_both_low_utilization_and_large_measured_wait(self):
        base = outcome(median_utilization_percent=79, data_wait_fraction=0.21)
        self.assertTrue(profiling.should_try_ram_cache(base))
        for util, wait in [(80, 0.3), (70, 0.2), (None, 0.3)]:
            self.assertFalse(profiling.should_try_ram_cache(outcome(
                median_utilization_percent=util, data_wait_fraction=wait)))

    def test_cache_requires_five_percent_gain_and_less_than_75_percent_ram(self):
        base = outcome()
        self.assertTrue(profiling.accept_ram_cache(base, outcome(rate=105, cache='ram')))
        for rate, ram in [(104, 40), (110, 75), (110, None)]:
            self.assertFalse(profiling.accept_ram_cache(base, outcome(rate, ram, cache='ram')))
        high_vram = outcome(rate=110, cache='ram')
        high_vram.peak_total_memory_bytes = 950
        self.assertFalse(profiling.accept_ram_cache(base, high_vram))

    def test_decoded_cache_preserves_fresh_augmentation_inputs(self):
        from bddcv.cache import DecodedImageCache
        cache = DecodedImageCache([1, 1, 2], lambda key: [key], memory_percent=lambda: 10)
        cached = cache(1)
        cached.append('augmentation')
        self.assertEqual(cache(1), [1])
        self.assertEqual(len(cache), 2)

    def test_controller_runs_and_retains_conditional_cache_variant(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts import profile_models

        def candidate(_controller, model_id, batch, **kwargs):
            cached = kwargs.get("cache") == "ram"
            value = outcome(rate=110 if cached else 100,
                            cache="ram" if cached else "none",
                            median_utilization_percent=60, data_wait_fraction=0.3,
                            workers=kwargs.get("workers", 0), prefetch=kwargs.get("prefetch", 2))
            value.batch = batch
            value.measured_elapsed_seconds = batch * 30 / (110 if cached else 100)
            return value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("sys.argv", ["profile_models", "--model-id", "yolo11s", "--out", directory]), patch.object(profile_models, "make_probe_manifest"), patch.object(profiling.ProfileController, "run_candidate", candidate), patch("builtins.print"):
                self.assertEqual(profile_models.main(), 0)
            summary = json.loads((root / "summary.json").read_text())
            self.assertTrue(summary["cache_trials"]["yolo11s"]["retained"])
            self.assertEqual(summary["selected"]["yolo11s"]["metadata"]["cache"], "ram")

    def test_frcnn_cache_keeps_images_and_targets_identical(self):
        import json
        import tempfile
        from pathlib import Path
        import torch
        from PIL import Image
        from bddcv.frcnn import CocoDetectionDataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "one.jpg"
            Image.new("RGB", (8, 4), color=(10, 20, 30)).save(image)
            annotation = root / "gt.json"
            annotation.write_text(json.dumps({
                "images": [{"id": 3, "file_name": "one.jpg"}],
                "annotations": [{"image_id": 3, "category_id": 1, "bbox": [1, 1, 2, 2]}],
            }))
            dataset = CocoDetectionDataset(root, annotation)
            before_image, before_target = dataset[0]
            dataset.cache_images()
            image.unlink()
            after_image, after_target = dataset[0]
            self.assertTrue(torch.equal(before_image, after_image))
            for key in before_target:
                self.assertTrue(torch.equal(before_target[key], after_target[key]))

    def test_selection_requires_total_device_memory_evidence(self):
        value = outcome()
        value.total_memory_bytes = None
        self.assertIsNone(profiling.select_profile([value]))
        value.total_memory_bytes = 1000
        value.peak_total_memory_bytes = None
        self.assertIsNone(profiling.select_profile([value]))

    def test_ultra_resume_rejects_cache_setting_drift(self):
        from scripts.train_ultralytics import validate_resume_configuration
        config = {"model_id": "yolo11s", "cache": "none", "prefetch": 2}
        checkpoint = {"bddcv_config": {**config, "cache": "ram"},
                      "train_args": {"model": "yolo11s.pt"}}
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_resume_configuration(config, checkpoint, config)

    def test_cache_refuses_host_ram_limit(self):
        from bddcv.cache import DecodedImageCache
        with self.assertRaisesRegex(RuntimeError, 'RAM'):
            DecodedImageCache([1], lambda key: [key], memory_percent=lambda: 75)


if __name__ == '__main__':
    unittest.main()
