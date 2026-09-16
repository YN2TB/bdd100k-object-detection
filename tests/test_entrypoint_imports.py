"""Entry points must resolve sibling helpers without an inherited PYTHONPATH."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class EntrypointImportTests(unittest.TestCase):
    def test_supervisor_resolves_checkpoint_validator_from_other_directory(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/train_with_resume.py'
        code = f'''import runpy
entry = runpy.run_path({str(script)!r}, run_name="preflight")
try:
    entry["validate_ultra_checkpoint"]({{"epoch": 0}}, {{}})
except ValueError as error:
    assert "optimizer" in str(error), str(error)
else:
    raise AssertionError("invalid checkpoint accepted")
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-I', '-c', code], cwd=directory,
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
