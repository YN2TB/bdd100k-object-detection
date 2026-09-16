"""Reproduce unsafe checkpoint acceptance and incomplete RNG restoration."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bddcv.checkpointing import UnsafeResumeError, require_resume_state, restore_rng_state


class ResumeIntegrityTests(unittest.TestCase):
    def test_none_optimizer_is_not_resume_state(self):
        with self.assertRaises(UnsafeResumeError):
            require_resume_state({"optimizer": None}, ["optimizer"])

    def test_empty_rng_cannot_silently_resume(self):
        with self.assertRaises(UnsafeResumeError):
            restore_rng_state({})

    def test_partial_rng_cannot_silently_resume(self):
        with self.assertRaises(UnsafeResumeError):
            restore_rng_state({"python": None})


if __name__ == "__main__":
    unittest.main()
