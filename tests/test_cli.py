"""核心 CLI 的命令边界、JSON 分析及比较行为，不依赖 ROS2。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from robottrace.io import load_and_analyze
from test_comparison import fixture, persist

ROOT = Path(__file__).resolve().parents[1]


class CoreCliTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, "-B", "-m", "robottrace", *map(str, args)],
            cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_core_command_help_and_removed_command_rejection(self):
        for command in ((), ("run",), ("analyze",), ("compare",)):
            result = self.invoke(*command, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("report", result.stdout)
        result = self.invoke("report", "--help")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_analysis_still_writes_authoritative_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            run, events = fixture()
            persist(path, run, events)
            result = self.invoke("analyze", path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads((path / "analysis.json").read_text()),
                load_and_analyze(path),
            )
            self.assertIn("observations=complete", result.stdout)

    def test_incomplete_observations_do_not_become_zero_or_cli_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            run, events = fixture(omit=((1, "consumer", "callback_entry"),))
            persist(path, run, events)
            result = self.invoke("analyze", path)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads((path / "analysis.json").read_text())
            self.assertEqual(saved["observations"]["status"], "incomplete")
            metric = saved["messages"][0]["metrics"]["consumer_age_ns"]
            self.assertIsNone(metric["value_ns"])
            self.assertEqual(metric["reason"], "MISSING_ENDPOINT")

    def test_comparison_keeps_json_and_no_implicit_input_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, events = fixture()
            for name in ("baseline", "candidate"):
                persist(root / name, run, events)
            result = self.invoke("compare", root / "baseline", root / "candidate", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            comparison = json.loads(result.stdout)
            self.assertEqual(comparison["status"], "COMPARABLE")
            for value in comparison["metrics"].values():
                self.assertEqual(value["verdict"], "NO_OBSERVED_DIFFERENCE")
            self.assertFalse(list(root.rglob("analysis.json")))


if __name__ == "__main__":
    unittest.main()
