"""Move legacy artifacts without rewriting their contents; dry-run by default.

Run on the host, with no trainer/monitor using this project. Receipts record every
file and permit rollback even after an interrupted move. No checkpoint is loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_moves(root: Path) -> list[tuple[Path, Path]]:
    """Discover only known legacy outputs; leave unknown data in place."""
    moves = []

    def add(source: Path, target: Path) -> None:
        if source.is_symlink():
            raise ValueError(f"Refusing symlink: {source}")
        if source.is_dir():
            for child in sorted(source.rglob("*")):
                if child.is_symlink():
                    raise ValueError(f"Refusing symlink: {child}")
                if child.is_file():
                    moves.append((child, target / child.relative_to(source)))
        elif source.is_file():
            moves.append((source, target))

    for weight in sorted(root.glob("*.pt")):
        add(weight, root / "weights" / weight.name)
    add(root / "archive.zip", root / "data/archives/archive.zip")
    for run in sorted((root / "runs/detect/runs").glob("*")):
        kind = "smoke" if run.name.startswith("smoke") else "train"
        add(run, root / "runs" / kind / run.name)
    add(root / "runs/frcnn", root / "runs/train/frcnn")
    for run in sorted((root / "runs").glob("smoke_*")):
        add(run, root / "runs/smoke" / run.name)
    for source in sorted((root / "runs/logs").glob("*")):
        name = source.name
        if source.suffix == ".py":
            target = root / "runs/maintenance/legacy_scripts" / name
        elif name in ("smoke_args.yaml", "smoke_results.csv"):
            target = root / "runs/smoke/smoke_rtdetr" / name.removeprefix("smoke_")
        elif name.startswith("smoke_"):
            target = root / "runs/smoke/smoke_rtdetr/logs" / name
        elif name.startswith("rtdetr"):
            target = root / "runs/train/rtdetr-l/logs" / name
        else:
            target = root / "runs/maintenance/legacy_logs" / name
        add(source, target)
    for image in sorted((root / "data/source_daytime_clear").glob("verify_*.png")):
        add(image, root / "runs/verification" / image.name)
    for pred in sorted((root / "runs").glob("preds_*.json")):
        add(pred, root / "runs/predictions" / pred.name)
    return moves


def assert_idle(root: Path) -> None:
    """Require host process visibility; reject active project processes/open files."""
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("Process inspection requires psutil; use the project .venv.") from exc
    if sys.platform.startswith("linux"):
        # A PID namespace can conceal the host's trainer even when no process matches.
        if psutil.Process(1).name() not in ("systemd", "init"):
            raise RuntimeError("Cannot verify host processes in this PID namespace; run on the host.")
    ancestors = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    blockers = []
    root_text = str(root.resolve())
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid in ancestors:
            continue
        try:
            command = " ".join(process.info["cmdline"] or [])
            name = (process.info["name"] or "").lower()
            if not any(word in name for word in ("python", "yolo", "torch")):
                continue
            cwd = process.cwd()
            relevant = root_text in command or cwd == root_text or cwd.startswith(root_text + os.sep)
            files = process.open_files()
            has_artifact = any(f.path.startswith(root_text + os.sep + prefix)
                               for f in files for prefix in ("runs/", "weights/", "rtdetr-l.pt"))
            if relevant or has_artifact:
                blockers.append(f"PID {process.pid}: {process.info['name']}")
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError(f"Cannot inspect potential worker PID {process.pid}") from exc
    if blockers:
        raise RuntimeError("Stop project workers before migration: " + "; ".join(blockers))


def checked_path(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository or is a symlink: {name}")
    return path


def write_receipt(path: Path, receipt: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def remove_empty_parents(path: Path, root: Path) -> None:
    while path != root and path.is_relative_to(root):
        try:
            path.rmdir()
        except OSError:
            break
        path = path.parent


def apply_moves(root: Path, moves: list[tuple[Path, Path]]) -> Path | None:
    """Preflight all destinations and journal hashes before the first move."""
    if not moves:
        return None
    assert_idle(root)
    entries, targets = [], set()
    for source, target in moves:
        source = checked_path(root, str(source.relative_to(root)))
        target = checked_path(root, str(target.relative_to(root)))
        if target.exists() or target in targets:
            raise FileExistsError(target)
        targets.add(target)
        entries.append({"source": str(source.relative_to(root)),
                        "target": str(target.relative_to(root)),
                        "size": source.stat().st_size, "sha256": sha256(source)})
    folder = root / "runs/maintenance"
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / f"migration-{uuid.uuid4().hex}.json"
    receipt = {"version": 1, "root": str(root.resolve()), "status": "moving",
               "created_at": datetime.now(timezone.utc).isoformat(), "files": entries}
    write_receipt(receipt_path, receipt)
    for entry in entries:
        source, target = root / entry["source"], root / entry["target"]
        if sha256(source) != entry["sha256"]:
            raise ValueError(f"Source changed during migration: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids os.rename overwriting a concurrently created target.
        # Hard-link + unlink preserves file bytes/metadata on this same-filesystem layout.
        os.link(source, target)
        if sha256(target) != entry["sha256"]:
            raise ValueError(f"Hash mismatch: {target}")
        source.unlink()
        remove_empty_parents(source.parent, root)
    receipt["status"] = "complete"
    write_receipt(receipt_path, receipt)
    return receipt_path


def rollback(root: Path, receipt_path: Path, apply: bool = True) -> None:
    """Preflight everything; tolerate an interrupted link/unlink or repeated rollback."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("version") != 1 or Path(receipt["root"]).resolve() != root.resolve():
        raise ValueError("Receipt belongs to a different repository or schema")
    if apply:
        assert_idle(root)
    operations = []
    for entry in receipt["files"]:
        source = checked_path(root, entry["source"])
        target = checked_path(root, entry["target"])
        present = [p for p in (source, target) if p.exists()]
        if not present or any(sha256(p) != entry["sha256"] for p in present):
            raise ValueError(f"Missing or modified artifact: {entry['source']}")
        if source.exists() and target.exists() and not os.path.samefile(source, target):
            raise FileExistsError(source)
        if target.exists():
            operations.append((source, target))
    if not apply:
        for source, target in operations:
            print(f"{target.relative_to(root)} -> {source.relative_to(root)}")
        return
    for source, target in reversed(operations):
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            os.link(target, source)
        target.unlink()
        remove_empty_parents(target.parent, root)
    receipt["status"] = "rolled_back"
    write_receipt(receipt_path, receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--apply", action="store_true", help="move files; otherwise only list")
    parser.add_argument("--rollback", type=Path, help="receipt to reverse (also needs --apply)")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.rollback:
            rollback(root, args.rollback.resolve(), apply=args.apply)
        else:
            moves = plan_moves(root)
            for source, target in moves:
                print(f"{source.relative_to(root)} -> {target.relative_to(root)}")
            print(f"{len(moves)} file(s); {'applying' if args.apply else 'dry-run, no writes'}")
            if args.apply:
                receipt = apply_moves(root, moves)
                if receipt:
                    print(f"Receipt: {receipt}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Migration stopped: {exc}\n")


if __name__ == "__main__":
    main()
