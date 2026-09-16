"""Native backend probes used by fresh profiling subprocesses."""
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .profiling import PROFILE_SCHEMA_VERSION, TIMING_SCHEMA, TIMING_SCHEMA_VERSION


class ValidationProbeError(RuntimeError):
    """Raised when the validation portion of an otherwise valid probe fails."""


class RepeatDataset:
    """Repeat a map-style dataset deterministically to an exact length."""

    def __init__(self, dataset: Any, length: int) -> None:
        if len(dataset) < 1 or length < 1:
            raise ValueError("dataset and repeated length must be non-empty")
        self.dataset = dataset
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]


class TimedLoader:
    """Delegate a loader while marking each batch before its fetch begins."""

    def __init__(self, loader: Any, on_batch_start: Any, on_data_wait: Any = None) -> None:
        self.loader = loader
        self.on_batch_start = on_batch_start
        self.on_data_wait = on_data_wait

    def __iter__(self):
        iterator = iter(self.loader)
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            if self.on_data_wait is not None:
                self.on_data_wait(time.perf_counter() - started)
            self.on_batch_start(started)
            yield batch

    def __len__(self) -> int:
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


def cycle_names(names: list[str], count: int) -> list[str]:
    """Repeat a non-empty deterministic pool to exactly ``count`` entries."""
    if not names or count < 0:
        raise ValueError("names must be non-empty and count non-negative")
    return [names[index % len(names)] for index in range(count)]


def loader_options(workers: int, prefetch: int) -> dict[str, int | bool]:
    """Return valid DataLoader worker options for a profiling candidate."""
    if workers < 0 or prefetch < 1:
        raise ValueError("workers must be non-negative and prefetch must be positive")
    options: dict[str, int | bool] = {"num_workers": workers}
    if workers:
        options.update({"prefetch_factor": prefetch, "persistent_workers": True})
    return options


def configure_loader_prefetch(loader: Any, prefetch: int) -> Any:
    """Set the documented DataLoader prefetch value before iteration starts."""
    if getattr(loader, "num_workers", 0):
        if loader.prefetch_factor != prefetch:
            loader.prefetch_factor = prefetch
            if hasattr(loader, "reset"):
                loader.reset()
    return loader


def parse_nvidia_sample(line: str) -> dict[str, float]:
    """Parse the stable five-column nvidia-smi telemetry schema."""
    values = [float(value.strip()) for value in line.split(",")]
    if len(values) != 5:
        raise ValueError(f"expected five nvidia-smi columns, got {line!r}")
    return dict(zip((
        "utilization_percent", "memory_used_mib", "memory_total_mib",
        "temperature_c", "power_w",
    ), values))


def summarize_steps(durations: list[float], *, batch: int, warmup: int) -> dict[str, float | int]:
    """Summarize synchronized step durations after the warm-up prefix."""
    if batch < 1 or warmup < 0:
        raise ValueError("batch must be positive and warmup must be non-negative")
    measured = durations[warmup:]
    if len(measured) == 0 or any(value <= 0 or not math.isfinite(value) for value in measured):
        raise ValueError("no finite measured step durations")
    ordered = sorted(measured)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    median = statistics.median(measured)
    elapsed = math.fsum(measured)
    return {
        "warmup_steps": min(warmup, len(durations)),
        "measured_steps": len(measured),
        "median_step_seconds": median,
        "p95_step_seconds": p95,
        "measured_elapsed_seconds": elapsed,
        "images_per_second": batch * len(measured) / elapsed,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "timing_schema": TIMING_SCHEMA,
        "timing_schema_version": TIMING_SCHEMA_VERSION,
    }


class NvidiaMonitor:
    """Collect total-device telemetry from one persistent nvidia-smi process."""

    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []
        self.host_ram_peak_percent = 0.0
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        import psutil
        self.host_ram_peak_percent = psutil.virtual_memory().percent
        command = [
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits", "--loop-ms=200",
        ]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        )

        def consume() -> None:
            assert self.process and self.process.stdout
            for line in self.process.stdout:
                try:
                    self.samples.append(parse_nvidia_sample(line))
                    self.host_ram_peak_percent = max(self.host_ram_peak_percent, psutil.virtual_memory().percent)
                except ValueError:
                    continue

        self.thread = threading.Thread(target=consume, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, float | None]:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.thread:
            self.thread.join(timeout=3)
        result: dict[str, float | None] = {}
        for key in ("utilization_percent", "memory_used_mib", "temperature_c", "power_w"):
            values = [sample[key] for sample in self.samples]
            result[f"median_{key}"] = statistics.median(values) if values else None
            result[f"peak_{key}"] = max(values) if values else None
        result["host_ram_peak_percent"] = self.host_ram_peak_percent
        result["memory_total_mib"] = self.samples[-1]["memory_total_mib"] if self.samples else None
        return result


def _write_profile_dataset(source: Path, manifest: Path, out: Path,
                           batch: int, train_steps: int, val_steps: int) -> Path:
    import yaml

    pool = manifest.read_text(encoding="utf-8").splitlines()
    if len(pool) != 512 or len(set(pool)) != 512:
        raise ValueError("profile manifest must contain 512 unique filenames")
    train = cycle_names(pool, batch * train_steps)
    val_pool = (source / "val_images.txt").read_text(encoding="utf-8").splitlines()
    val = cycle_names(val_pool, batch * val_steps)
    train_file = out / "train-cycle.txt"
    val_file = out / "val-cycle.txt"
    train_file.write_text("\n".join(str(source / "images/train" / name) for name in train) + "\n")
    val_file.write_text("\n".join(str(source / "images/val" / name) for name in val) + "\n")
    config = out / "dataset.yaml"
    names = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/bdd_source.yaml").read_text())["names"]
    config.write_text(yaml.safe_dump({"path": str(source), "train": str(train_file),
                                      "val": str(val_file), "names": names}, sort_keys=False))
    return config


def build_ultralytics_train_kwargs(*, dataset: Path, out: Path, batch: int,
                                    workers: int, imgsz: int, cache: str = "none") -> dict[str, Any]:
    """Build the fixed native recipe, including the API-required checkpoint."""
    return {
        "data": str(dataset), "epochs": 50, "imgsz": imgsz, "batch": batch,
        "amp": True, "device": 0, "workers": workers, "project": str(out),
        "name": "run", "exist_ok": True, "val": True, "plots": False,
        # YOLO.train() reloads best.pt/last.pt after trainer.train().  The
        # controlled callback stops before the configured 50-epoch horizon, so
        # save=False violates that public API postcondition.
        "save": True,
        "seed": 0, "deterministic": True, "cache": "ram" if cache == "ram" else False, "verbose": False,
    }


def build_rfdetr_train_kwargs(*, dataset_dir: Path, output_dir: Path, batch: int,
                              workers: int, prefetch: int,
                              class_names: Iterable[str] | None = None) -> dict[str, Any]:
    """Build the fixed RF-DETR probe recipe with the shared seed contract."""
    kwargs: dict[str, Any] = {
        "dataset_dir": str(dataset_dir), "output_dir": str(output_dir),
        "epochs": 50, "batch_size": batch, "grad_accum_steps": 1,
        "eval_batch_size": 1, "num_workers": workers, "prefetch_factor": prefetch,
        "persistent_workers": workers > 0, "resolution": 512,
        "checkpoint_interval": 50, "run_test": False, "early_stopping": False,
        "seed": 0, "device": "cuda", "tensorboard": False,
        "progress_bar": None, "save_dataset_grids": False,
    }
    if class_names is not None:
        kwargs["class_names"] = list(class_names)
    return kwargs


def disable_ultralytics_oom_retry(trainer: Any) -> None:
    """Force the pinned trainer to report OOM instead of silently shrinking batch."""
    trainer._oom_retries = 3


def run_ultralytics_probe(*, model_id: str, checkpoint: Path, source: Path,
                          manifest: Path, out: Path, batch: int, workers: int,
                          prefetch: int, warmup: int, steps: int,
                          validation_steps: int, imgsz: int, cache: str = "none") -> dict[str, Any]:
    """Run YOLO's native trainer for one controlled 50-horizon epoch."""
    import psutil
    import torch
    import ultralytics
    from ultralytics import YOLO
    import ultralytics.models.yolo.detect.train as yolo_train
    import ultralytics.models.yolo.detect.val as yolo_val

    from .paths import prepare_ultralytics

    prepare_ultralytics()
    total_steps = warmup + steps
    dataset = _write_profile_dataset(source, manifest, out, batch, total_steps, validation_steps)
    durations: list[float] = []
    data_wait: list[float] = []
    batch_starts: list[float] = []
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    started: float | None = None
    ended: float | None = None
    hook = None
    validation_batches = 0

    def on_train_start(trainer) -> None:
        nonlocal hook
        if cache == "ram":
            images = getattr(trainer.train_loader.dataset, "ims", [])
            if not images or any(image is None for image in images):
                raise RuntimeError("native RAM cache was requested but not populated")
            if psutil.virtual_memory().percent >= 75:
                raise RuntimeError("RAM cache reached the 75% host memory ceiling")
        def capture(_module, inputs) -> None:
            value = inputs[0] if inputs else None
            if isinstance(value, dict):
                value = value.get("img")
            if isinstance(value, torch.Tensor):
                shapes.add(tuple(value.shape))
                dtypes.add(str(value.dtype))
        hook = trainer.model.register_forward_pre_hook(capture)

    def on_train_epoch_start(trainer) -> None:
        # The pinned trainer resets this counter immediately after
        # on_train_start, so enforce fixed-batch behavior at the epoch boundary.
        disable_ultralytics_oom_retry(trainer)

    def on_train_batch_start(trainer) -> None:
        nonlocal started
        if int(trainer.batch_size) != batch:
            raise RuntimeError(
                f"fixed batch violation: requested {batch}, trainer changed to {trainer.batch_size}"
            )
        if not batch_starts:
            raise RuntimeError("timed training loader did not mark batch start")
        started = batch_starts.pop(0)
        torch.cuda.synchronize()

    def on_train_batch_end(trainer) -> None:
        nonlocal ended
        torch.cuda.synchronize()
        finished = time.perf_counter()
        assert started is not None
        # Consecutive completion boundaries include fetch/bookkeeping and avoid
        # counting prefetched batches as overlapping training intervals.
        durations.append(finished - (ended if ended is not None else started))
        ended = finished
        loss = float(trainer.loss.detach().sum().cpu())
        if not math.isfinite(loss):
            raise FloatingPointError("non-finite training loss")

    def on_train_epoch_end(trainer) -> None:
        trainer.stop = True

    def on_val_batch_end(_validator) -> None:
        nonlocal validation_batches
        validation_batches += 1

    model = YOLO(str(checkpoint))
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_start", on_train_batch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_batch_end", on_val_batch_end)
    monitor = NvidiaMonitor()
    original_train_loader = yolo_train.build_dataloader
    original_val_loader = yolo_val.build_dataloader

    def configured_train_loader(*args: Any, **kwargs: Any):
        loader = configure_loader_prefetch(original_train_loader(*args, **kwargs), prefetch)
        return TimedLoader(loader, batch_starts.append, data_wait.append)

    def configured_val_loader(*args: Any, **kwargs: Any):
        return configure_loader_prefetch(original_val_loader(*args, **kwargs), prefetch)

    yolo_train.build_dataloader = configured_train_loader
    yolo_val.build_dataloader = configured_val_loader
    monitor.start()
    torch.cuda.reset_peak_memory_stats()
    try:
        model.train(**build_ultralytics_train_kwargs(
            dataset=dataset, out=out, batch=batch, workers=workers, imgsz=imgsz, cache=cache,
        ))
    finally:
        yolo_train.build_dataloader = original_train_loader
        yolo_val.build_dataloader = original_val_loader
        telemetry = monitor.stop()
        if hook:
            hook.remove()
    if len(durations) != total_steps:
        raise RuntimeError(f"expected {total_steps} training steps, observed {len(durations)}")
    if validation_batches != validation_steps:
        raise RuntimeError(
            f"expected {validation_steps} validation steps, observed {validation_batches}"
        )
    summary = summarize_steps(durations, batch=batch, warmup=warmup)
    total_bytes = int((telemetry["memory_total_mib"] or 0) * 1024 * 1024)
    used_bytes = int((telemetry["peak_memory_used_mib"] or 0) * 1024 * 1024)
    return {
        "status": "success", "model_id": model_id, "batch": batch,
        **summary, "validation_steps": validation_steps,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_total_memory_bytes": used_bytes, "total_memory_bytes": total_bytes,
        "metadata": {
            "backend": "ultralytics", "ultralytics": ultralytics.__version__,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "tensor_shapes": [list(value) for value in sorted(shapes)],
            "dtypes": sorted(dtypes), "workers": workers, "prefetch": prefetch,
            "data_wait_median_seconds": statistics.median(data_wait[warmup:]) if data_wait[warmup:] else None,
            "data_wait_fraction": math.fsum(data_wait[warmup:]) / summary["measured_elapsed_seconds"],
            "cache": cache,
            "host_ram_peak_percent": psutil.virtual_memory().percent,
            **telemetry,
        },
    }


def run_frcnn_probe(*, model_id: str, source: Path, manifest: Path, out: Path,
                    batch: int, validation_batch: int, workers: int, prefetch: int,
                    warmup: int, steps: int, validation_steps: int,
                    imgsz: int, cache: str = "none") -> dict[str, Any]:
    """Profile the repository's production Faster R-CNN training recipe."""
    import psutil
    import torch
    from torch.utils.data import DataLoader

    from .frcnn import CocoDetectionDataset, build_model, collate
    from .paths import prepare_runtime

    if imgsz != 640:
        raise ValueError("Faster R-CNN comparison contract requires imgsz=640")
    prepare_runtime()
    device = torch.device("cuda:0")
    total_steps = warmup + steps
    train_ds = CocoDetectionDataset(
        source / "images/train", source / "annotations/instances_train.json", train=True,
    )
    train_by_name = {item["file_name"]: item for item in train_ds.images}
    pool = manifest.read_text(encoding="utf-8").splitlines()
    missing = sorted(set(pool) - set(train_by_name))
    if len(pool) != 512 or len(set(pool)) != 512 or missing:
        raise ValueError(f"invalid 512-image probe manifest; missing={missing[:3]}")
    train_ds.images = [
        train_by_name[name] for name in cycle_names(pool, batch * total_steps)
    ]

    val_ds = CocoDetectionDataset(
        source / "images/val", source / "annotations/instances_val.json", train=False,
    )
    val_by_name = {item["file_name"]: item for item in val_ds.images}
    val_pool = (source / "val_images.txt").read_text(encoding="utf-8").splitlines()
    val_ds.images = [
        val_by_name[name]
        for name in cycle_names(val_pool, validation_batch * validation_steps)
    ]
    if cache == "ram":
        train_ds.cache_images()
        val_ds.cache_images()
    common = {"collate_fn": collate, "pin_memory": True,
              **loader_options(workers, prefetch)}
    batch_starts: list[float] = []
    data_wait: list[float] = []
    train_loader = TimedLoader(
        DataLoader(train_ds, batch_size=batch, shuffle=False, **common),
        batch_starts.append, data_wait.append,
    )
    val_loader = DataLoader(val_ds, batch_size=validation_batch, shuffle=False, **common)

    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.SGD(
        [value for value in model.parameters() if value.requires_grad],
        lr=0.01, momentum=0.9, weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    full_train_images = len(train_by_name)
    total_schedule_steps = 50 * math.ceil(full_train_images / batch)
    durations: list[float] = []
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    previous_end: float | None = None

    def capture_transform(_module, _inputs, output) -> None:
        image_list = output[0] if isinstance(output, tuple) else output
        tensors = getattr(image_list, "tensors", None)
        if isinstance(tensors, torch.Tensor):
            shapes.add(tuple(tensors.shape))
            dtypes.add(str(tensors.dtype))

    hook = model.transform.register_forward_hook(capture_transform)
    monitor = NvidiaMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats()
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for iteration, (images, targets) in enumerate(train_loader):
            if not batch_starts:
                raise RuntimeError("timed training loader did not mark batch start")
            started = batch_starts.pop(0)
            torch.cuda.synchronize()
            lr = 0.01 * (iteration + 1) / 500 if iteration < 500 else (
                0.01 * 0.5 * (
                    1 + math.cos(math.pi * (iteration - 500) / max(1, total_schedule_steps - 500))
                )
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            with torch.autocast("cuda", enabled=True):
                loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Faster R-CNN loss at step {iteration}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            finished = time.perf_counter()
            durations.append(finished - (previous_end if previous_end is not None else started))
            previous_end = finished

        model.eval()
        observed_validation = 0
        with torch.no_grad():
            for images, _targets in val_loader:
                images = [image.to(device, non_blocking=True) for image in images]
                with torch.autocast("cuda", enabled=True):
                    outputs = model(images)
                if any(not torch.isfinite(output["scores"]).all() for output in outputs):
                    raise FloatingPointError("non-finite Faster R-CNN validation score")
                observed_validation += 1
        torch.cuda.synchronize()
    finally:
        telemetry = monitor.stop()
        hook.remove()

    if len(durations) != total_steps:
        raise RuntimeError(f"expected {total_steps} training steps, observed {len(durations)}")
    if observed_validation != validation_steps:
        raise RuntimeError(
            f"expected {validation_steps} validation steps, observed {observed_validation}"
        )
    summary = summarize_steps(durations, batch=batch, warmup=warmup)
    return {
        "status": "success", "model_id": model_id, "batch": batch, **summary,
        "validation_steps": observed_validation,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_total_memory_bytes": int((telemetry["peak_memory_used_mib"] or 0) * 1024 * 1024),
        "total_memory_bytes": int((telemetry["memory_total_mib"] or 0) * 1024 * 1024),
        "metadata": {
            "backend": "frcnn", "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "tensor_shapes": [list(v) for v in sorted(shapes)],
            "dtypes": sorted(dtypes), "workers": workers, "prefetch": prefetch,
            "validation_batch": validation_batch,
            "data_wait_median_seconds": statistics.median(data_wait[warmup:]) if data_wait[warmup:] else None,
            "data_wait_fraction": math.fsum(data_wait[warmup:]) / summary["measured_elapsed_seconds"],
            "cache": cache,
            "host_ram_percent": psutil.virtual_memory().percent, **telemetry,
        },
    }


def run_rfdetr_probe(*, model_id: str, checkpoint: Path, source: Path,
                     manifest: Path, out: Path, batch: int, workers: int,
                     prefetch: int, warmup: int, steps: int,
                     validation_steps: int, imgsz: int, cache: str = "none") -> dict[str, Any]:
    """Profile RF-DETR through its pinned public trainer and native callbacks."""
    import psutil
    import torch
    import rfdetr.training as training
    from pytorch_lightning import Callback

    from .constants import DET_CLASSES
    from .paths import prepare_rfdetr_runtime
    from .rfdetr import RFDETRAdapter, build_probe_adapter

    if imgsz != 512:
        raise ValueError("RF-DETR Small comparison contract requires imgsz=512")
    prepare_rfdetr_runtime()
    pool = manifest.read_text(encoding="utf-8").splitlines()
    if len(pool) != 512 or len(set(pool)) != 512:
        raise ValueError("profile manifest must contain 512 unique filenames")
    val_pool = (source / "val_images.txt").read_text(encoding="utf-8").splitlines()
    adapter = build_probe_adapter(
        source, out / "dataset", pool, val_pool[:validation_steps],
    )
    total_steps = warmup + steps
    durations: list[float] = []
    data_wait: list[float] = []
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    phase = "setup"
    batch_starts: list[float] = []

    original_datamodule = training.RFDETRDataModule
    original_build_trainer = training.build_trainer

    class ProfileDataModule(original_datamodule):
        _profile_wrapped = False

        def setup(self, stage: str) -> None:
            super().setup(stage)
            if stage == "fit" and not self._profile_wrapped:
                if cache == "ram":
                    from .cache import DecodedImageCache
                    for dataset in (self._dataset_train, self._dataset_val):
                        dataset._decode_image = DecodedImageCache(dataset.ids, dataset._decode_image)
                self._dataset_train = RepeatDataset(self._dataset_train, batch * total_steps)
                self._dataset_val = RepeatDataset(self._dataset_val, validation_steps)
                self._profile_wrapped = True

        def train_dataloader(self):
            return TimedLoader(super().train_dataloader(), batch_starts.append, data_wait.append)

    class ProfileCallback(Callback):
        def __init__(self) -> None:
            self.started: float | None = None
            self.ended: float | None = None
            self.validation_batches = 0
            self.hook = None

        def on_fit_start(self, _trainer, pl_module) -> None:
            def capture(_module, inputs) -> None:
                samples = inputs[0] if inputs else None
                tensors = getattr(samples, "tensors", None)
                if isinstance(tensors, torch.Tensor):
                    shapes.add(tuple(tensors.shape))
                    dtypes.add(str(tensors.dtype))
            self.hook = pl_module.model.register_forward_pre_hook(capture)

        def on_train_batch_start(self, _trainer, _module, batch_value, _batch_idx) -> None:
            nonlocal phase
            phase = "train"
            if not batch_starts:
                raise RuntimeError("timed training loader did not mark batch start")
            self.started = batch_starts.pop(0)
            torch.cuda.synchronize()

        def on_train_batch_end(self, _trainer, _module, outputs, _batch_value, _batch_idx) -> None:
            torch.cuda.synchronize()
            finished = time.perf_counter()
            if self.started is None:
                raise RuntimeError("RF-DETR timing callback missed batch start")
            durations.append(finished - (self.ended if self.ended is not None else self.started))
            self.ended = finished
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
                raise FloatingPointError("non-finite RF-DETR training loss")

        def on_validation_start(self, _trainer, _module) -> None:
            nonlocal phase
            phase = "validation"

        def on_validation_batch_end(self, _trainer, _module, outputs,
                                    _batch_value, _batch_idx, _dataloader_idx=0) -> None:
            self.validation_batches += 1
            for result in outputs.get("results", []) if isinstance(outputs, dict) else []:
                scores = result.get("scores") if isinstance(result, dict) else None
                if isinstance(scores, torch.Tensor) and not torch.isfinite(scores).all():
                    raise FloatingPointError("non-finite RF-DETR validation score")

        def on_validation_end(self, trainer, _module) -> None:
            trainer.should_stop = True

        def on_fit_end(self, _trainer, _module) -> None:
            if self.hook:
                self.hook.remove()

    callback = ProfileCallback()

    def build_profile_trainer(*args: Any, **kwargs: Any):
        kwargs.update({
            "limit_train_batches": total_steps,
            "limit_val_batches": validation_steps,
            "num_sanity_val_steps": 0,
        })
        trainer = original_build_trainer(*args, **kwargs)
        trainer.callbacks.append(callback)
        return trainer

    training.RFDETRDataModule = ProfileDataModule
    training.build_trainer = build_profile_trainer
    monitor = NvidiaMonitor()
    monitor.start()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = RFDETRAdapter(
            model_id=model_id, checkpoint=checkpoint, python=os.sys.executable,
        ).construct()
        try:
            model.train(**build_rfdetr_train_kwargs(
                dataset_dir=adapter, output_dir=out / "run", batch=batch,
                workers=workers, prefetch=prefetch, class_names=DET_CLASSES,
            ))
        except torch.cuda.OutOfMemoryError as exc:
            if phase == "validation":
                raise ValidationProbeError(f"RF-DETR validation OOM: {exc}") from exc
            raise
    finally:
        training.RFDETRDataModule = original_datamodule
        training.build_trainer = original_build_trainer
        telemetry = monitor.stop()

    if len(durations) != total_steps:
        raise RuntimeError(f"expected {total_steps} training steps, observed {len(durations)}")
    if callback.validation_batches != validation_steps:
        raise ValidationProbeError(
            f"expected {validation_steps} validation steps, observed {callback.validation_batches}"
        )
    summary = summarize_steps(durations, batch=batch, warmup=warmup)
    return {
        "status": "success", "model_id": model_id, "batch": batch, **summary,
        "validation_steps": callback.validation_batches,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_total_memory_bytes": int((telemetry["peak_memory_used_mib"] or 0) * 1024 * 1024),
        "total_memory_bytes": int((telemetry["memory_total_mib"] or 0) * 1024 * 1024),
        "metadata": {
            "backend": "rfdetr", "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "tensor_shapes": [list(v) for v in sorted(shapes)],
            "dtypes": sorted(dtypes), "workers": workers, "prefetch": prefetch,
            "validation_batch": 1,
            "data_wait_median_seconds": statistics.median(data_wait[warmup:]) if data_wait[warmup:] else None,
            "data_wait_fraction": math.fsum(data_wait[warmup:]) / summary["measured_elapsed_seconds"],
            "cache": cache,
            "host_ram_percent": psutil.virtual_memory().percent, **telemetry,
        },
    }


__all__ = [
    "NvidiaMonitor", "RepeatDataset", "ValidationProbeError",
    "build_rfdetr_train_kwargs", "build_ultralytics_train_kwargs",
    "configure_loader_prefetch", "cycle_names",
    "disable_ultralytics_oom_retry", "loader_options", "parse_nvidia_sample",
    "run_frcnn_probe", "run_rfdetr_probe", "run_ultralytics_probe", "summarize_steps",
    "TimedLoader",
]
