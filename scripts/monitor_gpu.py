"""Record nvidia-smi samples while a specific process is alive."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bddcv.paths import resolve_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int, help="training/supervisor PID")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="default: <run-dir>/logs/gpu.csv")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    import psutil

    try:
        process = psutil.Process(args.pid)
        created = process.create_time()
    except psutil.NoSuchProcess:
        parser.error("the target process has already exited")
    out = resolve_output(args.out, args.run_dir.resolve() / "logs/gpu.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="", buffering=1) as output:
        csv.writer(output).writerow(["timestamp", "gpu_util_percent", "device_memory_used_mib", "temperature_c"])
        while True:
            try:
                if (not process.is_running() or process.create_time() != created
                        or process.status() == psutil.STATUS_ZOMBIE):
                    break
            except psutil.NoSuchProcess:
                break
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stderr.strip() or "nvidia-smi failed")
            output.write(result.stdout)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
