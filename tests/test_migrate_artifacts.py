"""Exercise real file moves and rollback on isolated fixture repositories."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class MigrationTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "scripts/maintenance/migrate_artifacts.py"
        self.assertTrue(path.exists(), "migration entrypoint is not implemented")
        spec = importlib.util.spec_from_file_location("migration", path)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.tmp = tempfile.TemporaryDirectory(prefix="bdd migration ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Fixture files have no host workers; exercise actual migration, not process enumeration.
        guard = patch.object(self.m, "assert_idle")
        guard.start()
        self.addCleanup(guard.stop)
        self.put("rtdetr-l.pt", b"original pretrained")
        self.put("runs/detect/runs/rtdetr-l/weights/last.pt", b"checkpoint")
        self.put("runs/detect/runs/rtdetr-l/results.csv", b"epoch\r\n50\r\n")
        self.put("runs/logs/rtdetr.log", b"COMPLETE\n")
        self.put("runs/logs/smoke_args.yaml", b"epochs: 1\n")
        self.put("runs/logs/smoke_rtdetr.py", b"# historical script\n")
        self.put("data/source_daytime_clear/verify_val.png", b"image")

    def put(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value)

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file()}

    def test_dry_run_changes_nothing(self):
        before = self.snapshot()
        moves = self.m.plan_moves(self.root)
        self.assertTrue(moves)
        self.assertEqual(before, self.snapshot())

    def test_apply_verify_rerun_and_rollback(self):
        before = self.snapshot()
        receipt = self.m.apply_moves(self.root, self.m.plan_moves(self.root))
        self.assertEqual((self.root / "weights/rtdetr-l.pt").read_bytes(), b"original pretrained")
        self.assertEqual((self.root / "runs/train/rtdetr-l/weights/last.pt").read_bytes(), b"checkpoint")
        self.assertEqual((self.root / "runs/smoke/smoke_rtdetr/args.yaml").read_bytes(), b"epochs: 1\n")
        self.assertEqual(self.m.plan_moves(self.root), [])
        self.m.rollback(self.root, receipt)
        after = self.snapshot()
        for name, value in before.items():
            self.assertEqual(after[name], value)
        self.assertFalse((self.root / "weights/rtdetr-l.pt").exists())

    def test_conflict_rejected_before_any_move(self):
        self.put("weights/rtdetr-l.pt", b"another weight")
        before = self.snapshot()
        with self.assertRaises(FileExistsError):
            self.m.apply_moves(self.root, self.m.plan_moves(self.root))
        self.assertEqual(before, self.snapshot())

    def test_rollback_refuses_modified_artifact(self):
        receipt = self.m.apply_moves(self.root, self.m.plan_moves(self.root))
        self.put("weights/rtdetr-l.pt", b"changed after migration")
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.m.rollback(self.root, receipt)
        self.assertEqual(before, self.snapshot())

    def test_process_guard_failure_prevents_writes(self):
        before = self.snapshot()
        with patch.object(self.m, "assert_idle", side_effect=RuntimeError("active worker")):
            with self.assertRaises(RuntimeError):
                self.m.apply_moves(self.root, self.m.plan_moves(self.root))
        self.assertEqual(before, self.snapshot())

    def test_interrupted_move_can_be_rolled_back(self):
        before = self.snapshot()
        real_link = self.m.os.link
        calls = 0

        def interrupt_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated interruption")
            return real_link(source, target)

        with patch.object(self.m.os, "link", side_effect=interrupt_second):
            with self.assertRaises(OSError):
                self.m.apply_moves(self.root, self.m.plan_moves(self.root))
        receipt = next((self.root / "runs/maintenance").glob("migration-*.json"))
        self.m.rollback(self.root, receipt)
        for name, value in before.items():
            self.assertEqual((self.root / name).read_bytes(), value)


if __name__ == "__main__":
    unittest.main()
