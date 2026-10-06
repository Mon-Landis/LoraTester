from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for frontend runtime tests")
    def test_frontend_runtime_invariants(self):
        result = subprocess.run(
            [shutil.which("node"), "--test", "tests/frontend_runtime.test.mjs"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
