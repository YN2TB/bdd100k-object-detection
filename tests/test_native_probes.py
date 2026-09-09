"""Native profiling helpers preserve the requested sample/step contract."""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bddcv.native_probes import (
    build_ultralytics_train_kwargs,
    build_rfdetr_train_kwargs,
    configure_loader_prefetch,
    cycle_names,
    disable_ultralytics_oom_retry,
    loader_options,
    parse_nvidia_sample,
    RepeatDataset,
    summarize_steps,
    TimedLoader,
)


class NativeProbeTests(unittest.TestCase):
    def test_cycle_names_repeats_deterministically_to_exact_batches(self):
        self.assertEqual(
            cycle_names(["a.jpg", "b.jpg", "c.jpg"], 8),
            ["a.jpg", "b.jpg", "c.jpg", "a.jpg", "b.jpg", "c.jpg", "a.jpg", "b.jpg"],
        )

    def test_monitor_sample_parses_all_required_telemetry(self):
        self.assertEqual(
            parse_nvidia_sample("42, 6144, 12288, 71, 132.5"),
            {"utilization_percent": 42.0, "memory_used_mib": 6144.0,
             "memory_total_mib": 12288.0, "temperature_c": 71.0,
             "power_w": 132.5},
        )

    def test_step_summary_separates_warmup_and_measured_steps(self):
        result = summarize_steps([1.0] * 10 + [2.0] * 30, batch=4, warmup=10)
        self.assertEqual(result["warmup_steps"], 10)
        self.assertEqual(result["measured_steps"], 30)
        self.assertEqual(result["median_step_seconds"], 2.0)
        self.assertEqual(result["p95_step_seconds"], 2.0)
        self.assertEqual(result["images_per_second"], 2.0)

    def test_step_summary_uses_aggregate_elapsed_after_warmup(self):
        result = summarize_steps([100.0, 100.0, 1.0, 3.0], batch=4, warmup=2)
        self.assertEqual(result["measured_elapsed_seconds"], 4.0)
        self.assertEqual(result["images_per_second"], 2.0)
        from bddcv.profiling import TIMING_SCHEMA_VERSION
        self.assertEqual(result["timing_schema_version"], TIMING_SCHEMA_VERSION)

    def test_timed_loader_records_fetch_wait_separately(self):
        from unittest.mock import patch
        waits = []
        with patch("bddcv.native_probes.time.perf_counter", side_effect=[1.0, 1.25, 2.0]):
            timed = TimedLoader(["batch"], lambda _start: None, waits.append)
            self.assertEqual(list(timed), ["batch"])
        self.assertEqual(waits, [0.25])

    def test_timed_loader_marks_before_fetch(self):
        events = []

        class Loader:
            def __iter__(self):
                events.append(("fetch", time.perf_counter()))
                yield "batch"

        timed = TimedLoader(Loader(), lambda started: events.append(("mark", started)))
        self.assertEqual(list(timed), ["batch"])
        self.assertEqual([event[0] for event in events], ["fetch", "mark"])
        self.assertLessEqual(events[1][1], events[0][1])

    def test_timed_loader_delegates_framework_loader_attributes(self):
        class Loader:
            dataset = object()
            sampler = object()
            batch_size = 8

            def __iter__(self):
                return iter(())

            def __len__(self):
                return 0

            def reset(self):
                return "reset"

        timed = TimedLoader(Loader(), lambda _started: None)
        self.assertIsNotNone(timed.dataset)
        self.assertIsNotNone(timed.sampler)
        self.assertEqual(timed.batch_size, 8)
        self.assertEqual(timed.reset(), "reset")

    def test_ultralytics_probe_saves_checkpoint_required_by_model_api(self):
        kwargs = build_ultralytics_train_kwargs(
            dataset=Path("dataset.yaml"), out=Path("profile"), batch=8,
            workers=0, imgsz=640,
        )
        self.assertTrue(kwargs["save"])
        self.assertEqual(kwargs["epochs"], 50)
        self.assertEqual(kwargs["batch"], 8)

    def test_ultralytics_probe_disables_native_batch_reduction(self):
        class Trainer:
            _oom_retries = 0

        trainer = Trainer()
        disable_ultralytics_oom_retry(trainer)
        self.assertEqual(trainer._oom_retries, 3)

    def test_rfdetr_probe_pins_seed_zero(self):
        kwargs = build_rfdetr_train_kwargs(
            dataset_dir=Path("dataset"), output_dir=Path("profile"), batch=2,
            workers=4, prefetch=2,
        )
        self.assertEqual(kwargs["seed"], 0)

    def test_loader_options_only_set_prefetch_for_worker_processes(self):
        self.assertEqual(loader_options(0, 4), {"num_workers": 0})
        self.assertEqual(
            loader_options(2, 4),
            {"num_workers": 2, "prefetch_factor": 4, "persistent_workers": True},
        )

    def test_configure_prefetch_changes_live_worker_loader(self):
        class Loader:
            num_workers = 4
            prefetch_factor = 2

        loader = Loader()
        configure_loader_prefetch(loader, 4)
        self.assertEqual(loader.prefetch_factor, 4)

    def test_repeat_dataset_has_exact_requested_length(self):
        dataset = RepeatDataset(["a", "b", "c"], 8)
        self.assertEqual(len(dataset), 8)
        self.assertEqual([dataset[index] for index in range(8)],
                         ["a", "b", "c", "a", "b", "c", "a", "b"])


if __name__ == "__main__":
    unittest.main()
