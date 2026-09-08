"""Run one isolated Ultralytics smoke epoch and record time/peak CUDA memory."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bddcv.paths import DATA_CONFIG, SMOKE_DIR, prepare_ultralytics, resolve_output, resolve_weights


class Tee:
    """Mirror trainer output to console and the run's log."""

    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="rtdetr-l.pt")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="smoke-only train fraction; validation still uses the full split")
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    name = args.name or f"smoke_{Path(args.model).stem}"
    if Path(name).name != name or name in (".", ".."):
        parser.error("--name must be one directory name")
    out = resolve_output(args.out, SMOKE_DIR / name)
    if (out / "results.csv").exists() or (out / "weights").exists():
        parser.error("smoke output already contains training artifacts; choose a new --out")
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    model = resolve_weights(args.model)
    started, success = time.time(), False
    with (logs / "smoke.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            prepare_ultralytics()
            import torch
            from ultralytics import RTDETR, YOLO

            cuda = torch.cuda.is_available() and args.device != "cpu"
            device = int(args.device.split(",")[0]) if cuda else None
            if cuda:
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats(device)
            try:
                cls = RTDETR if "rtdetr" in model.stem.lower() else YOLO
                cls(str(model)).train(data=str(DATA_CONFIG), epochs=1, imgsz=args.imgsz,
                                     batch=args.batch, amp=True, device=args.device,
                                     workers=args.workers, project=str(out.parent), name=out.name,
                                     exist_ok=True, seed=0, fraction=args.fraction)
                success = True
            finally:
                result = {"success": success, "batch": args.batch, "fraction": args.fraction,
                          "model": str(model), "run_dir": str(out),
                          "wall_seconds": time.time() - started,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else 0}
                (logs / "smoke_status.json").write_text(json.dumps(result, indent=2) + "\n")
                print("SMOKE_STATUS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
