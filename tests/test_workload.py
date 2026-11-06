import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict, replace

from robottrace import workload
from test_analysis import message


class FakeProcess:
    """模拟进程操作及有限等待，不启动 ROS2，也不实际发送信号。"""

    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.signals = []
        self.waits = []
        self.kills = 0
        self.signal_error = None
        self.wait_errors = {}
        self.kill_error = None

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)
        if self.signal_error:
            raise self.signal_error

    def wait(self, timeout):
        self.waits.append(timeout)
        if timeout in self.wait_errors:
            raise self.wait_errors[timeout]
        self.returncode = -signal.SIGKILL if self.kills else 0
        return self.returncode

    def kill(self):
        self.kills += 1
        if self.kill_error:
            raise self.kill_error


class WorkloadLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "evidence").mkdir()
        (self.root / "bin").mkdir()
        (self.root / "bin/pipeline").write_bytes(b"test-only binary")
        self.boot = "test-boot"
        self.offsets = "monotonic 0 0\nboottime 0 0\n"
        self.domain = "host"
        self.processes = {}
        self.status_change = None
        self.process_change = None
        self.primary = RuntimeError("primary pipeline failure")
        self.patch(patch.object(workload, "ROOT", self.root))
        self.patch(
            patch.object(
                workload,
                "clock_evidence",
                return_value=(self.boot, self.offsets, self.domain),
            )
        )
        self.patch(
            patch.object(workload.time, "monotonic_ns", return_value=10_000_000_000)
        )
        self.patch(
            patch.dict(
                workload.os.environ,
                {
                    "ROBOTTRACE_HOST_CLOCK": json.dumps(
                        {"boot_id": self.boot, "offsets": self.offsets}
                    )
                },
            )
        )
        self.patch(patch.object(workload.subprocess, "Popen", side_effect=self.spawn))
        self.patch(
            patch.object(
                workload.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            )
        )
        self.versions = self.patch(
            patch.object(
                workload.subprocess, "check_output", return_value="test-package\t1\n"
            )
        )

    def patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def spawn(self, args, **kwargs):
        stage, run_id, directory = args[1:4]
        out = Path(directory)
        process = FakeProcess({"source": 1, "processor": 2, "consumer": 3}[stage])
        self.processes[stage] = process
        if self.process_change:
            self.process_change(stage, process)
        events = [
            replace(e, run_id=run_id)
            for e in message(source="source:1")
            if e.stage == stage
        ]
        (out / (stage + ".jsonl")).write_text(
            "".join(json.dumps(asdict(e)) + "\n" for e in events)
        )
        node = {
            "stage": stage,
            "process_id": process.pid,
            "stored": len(events),
            "dropped": 0,
            "flushed": True,
            "clock_domain": self.domain,
            "boot_id": self.boot,
            "time_offsets": self.offsets,
            "safety_deadline": False,
        }
        text = json.dumps(node)
        if self.status_change:
            text = self.status_change(stage, text)
        if text is not None:
            (out / (stage + ".status.json")).write_text(text)
        (out / (stage + ".started")).write_text("1")
        if stage == "source":
            (out / "matched.json").write_text("{}")
            # 只绕过测试的 drain 等待，不依赖真实调度或运行时长。
            (out / "published.json").write_text('{"published":1,"completed_ns":0}')
        return process

    def run_abnormal(self):
        self.patch(patch.object(workload, "wait_file", side_effect=self.primary))
        with self.assertRaises((RuntimeError, ValueError, OSError)) as caught:
            workload.run_pipeline(self.root / "evidence", "normal", count=1)
        return caught.exception

    def output(self):
        return next((self.root / "evidence").glob("normal_*"))

    def manifest(self):
        return json.loads((self.output() / "run.json").read_text())

    def test_malformed_status_does_not_skip_other_child_cleanup(self):
        self.status_change = lambda stage, text: (
            "{broken" if stage == "consumer" else text
        )
        error = self.run_abnormal()
        self.assertTrue(
            all(p.returncode is not None for p in self.processes.values()),
            {s: p.returncode for s, p in self.processes.items()},
        )
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("primary pipeline failure", str(error))
        self.assertFalse(self.manifest()["execution_completed"])
        self.assertIsNone(self.manifest()["collectors"])
        unknown = self.manifest()["nodes"]["consumer"]
        self.assertFalse(unknown["status_available"])
        self.assertNotIn("stored", unknown)
        self.assertNotIn("flushed", unknown)
        self.assertTrue((self.output() / "consumer.status.json").exists())
        analysis = json.loads((self.output() / "analysis.json").read_text())
        self.assertNotEqual(analysis["collection"]["status"], "complete")
        self.assertTrue(self.manifest()["status_errors"])
        self.assertIs(error.__cause__, self.primary)

    def test_signal_failure_does_not_skip_other_child_cleanup(self):
        def change(stage, process):
            if stage == "consumer":
                process.signal_error = PermissionError("SIGINT denied")

        self.process_change = change
        error = self.run_abnormal()
        self.assertTrue(
            all(p.returncode is not None for p in self.processes.values()),
            {s: p.returncode for s, p in self.processes.items()},
        )
        self.assertIn("primary pipeline failure", str(error))
        self.assertFalse(self.manifest()["execution_completed"])
        self.assertEqual(self.manifest()["cleanup_errors"][0]["operation"], "sigint")

    def test_wait_failure_escalates_and_other_children_are_cleaned(self):
        def change(stage, process):
            if stage == "consumer":
                process.wait_errors[4] = OSError("wait failed")

        self.process_change = change
        error = self.run_abnormal()
        process = self.processes["consumer"]
        self.assertEqual(process.waits, [4, 2])
        self.assertEqual(process.kills, 1)
        self.assertTrue(all(p.returncode is not None for p in self.processes.values()))
        self.assertIs(error.__cause__, self.primary)
        self.assertEqual(
            self.manifest()["cleanup_errors"][0]["operation"], "wait_after_sigint"
        )

    def test_failed_kill_and_wait_are_bounded_and_do_not_skip_other_children(self):
        def change(stage, process):
            if stage == "consumer":
                process.wait_errors = {
                    4: subprocess.TimeoutExpired("child", 4),
                    2: subprocess.TimeoutExpired("child", 2),
                }
                process.kill_error = PermissionError("kill denied")

        self.process_change = change
        self.run_abnormal()
        self.assertEqual(self.processes["consumer"].waits, [4, 2])
        self.assertEqual(self.processes["consumer"].kills, 1)
        self.assertIsNone(self.processes["consumer"].returncode)
        self.assertTrue(
            all(self.processes[s].returncode == 0 for s in ("processor", "source"))
        )
        manifest = self.manifest()
        self.assertFalse(manifest["execution_completed"])
        self.assertIsNone(manifest["collectors"])
        self.assertEqual(
            [item["operation"] for item in manifest["cleanup_errors"]],
            ["wait_after_sigint", "kill", "wait_after_kill"],
        )

    def test_primary_failure_is_saved_without_replacing_it(self):
        error = self.run_abnormal()
        manifest = self.manifest()
        self.assertEqual(manifest["execution_error"], repr(self.primary))
        self.assertFalse(manifest["execution_completed"])
        self.assertIs(error.__cause__, self.primary)
        self.assertEqual(len(manifest["collectors"]), 3)
        self.assertTrue((self.output() / "consumer.status.json").exists())

    def test_normal_shutdown_behavior_and_complete_evidence_are_preserved(self):
        path, result = workload.run_pipeline(self.root / "evidence", "normal", count=1)
        manifest = self.manifest()
        self.assertTrue(manifest["execution_completed"])
        self.assertTrue(manifest["clock_verified"])
        self.assertEqual(result["collection"]["status"], "complete")
        self.assertEqual(result["observations"]["status"], "complete")
        for process in self.processes.values():
            self.assertEqual(process.signals, [signal.SIGINT])
            self.assertEqual(process.waits, [5])
            self.assertEqual(process.kills, 0)
        self.assertFalse(list(path.glob("*.status.json")))
        self.assertNotIn("cleanup_errors", manifest)
        self.assertNotIn("status_errors", manifest)
        self.assertNotIn("metadata_errors", manifest)

    def test_version_query_failure_still_saves_failure_manifest(self):
        self.versions.side_effect = subprocess.CalledProcessError(1, "dpkg-query")
        error = self.run_abnormal()
        manifest = self.manifest()
        self.assertFalse(manifest["execution_completed"])
        self.assertNotIn("versions", manifest)
        self.assertEqual(manifest["metadata_errors"][0]["operation"], "read_versions")
        self.assertIs(error.__cause__, self.primary)
        self.assertEqual(self.versions.call_args.kwargs["timeout"], 5)

    def test_missing_binary_does_not_mask_primary_failure_or_prevent_manifest(self):
        (self.root / "bin/pipeline").unlink()
        error = self.run_abnormal()
        manifest = self.manifest()
        self.assertNotIn("binary_sha256", manifest)
        self.assertFalse(manifest["execution_completed"])
        self.assertIs(error.__cause__, self.primary)

    def test_manifest_save_failure_surfaces_both_errors(self):
        self.patch(patch.object(workload, "write", side_effect=OSError("disk full")))
        error = self.run_abnormal()
        self.assertIn("failed to save run.json", str(error))
        self.assertIn("disk full", str(error))
        self.assertIn("primary pipeline failure", str(error))
        self.assertIs(error.__cause__, self.primary)
        self.assertTrue(all(p.returncode is not None for p in self.processes.values()))

    def test_manifest_save_failure_preserves_status_files_on_healthy_path(self):
        self.patch(patch.object(workload, "write", side_effect=OSError("disk full")))
        with self.assertRaisesRegex(RuntimeError, "failed to save run.json"):
            workload.run_pipeline(self.root / "evidence", "normal", count=1)
        self.assertEqual(len(list(self.output().glob("*.status.json"))), 3)

    def test_analysis_failure_does_not_mask_primary_error(self):
        self.patch(
            patch.object(
                workload, "load_and_analyze", side_effect=ValueError("analysis failed")
            )
        )
        error = self.run_abnormal()
        self.assertIn("primary pipeline failure", str(error))
        self.assertIn("analysis failed", str(error))
        self.assertIs(error.__cause__, self.primary)
        self.assertFalse(self.manifest()["execution_completed"])

    def test_missing_status_is_unknown_not_a_zero_count_collector(self):
        self.status_change = lambda stage, text: None if stage == "consumer" else text
        self.run_abnormal()
        manifest = self.manifest()
        self.assertIsNone(manifest["collectors"])
        self.assertNotIn("dropped", manifest["nodes"]["consumer"])
        self.assertFalse(manifest["execution_completed"])
        result = json.loads((self.output() / "analysis.json").read_text())
        self.assertNotEqual(result["collection"]["status"], "complete")

    def test_unflushed_status_cannot_mark_run_completed(self):
        def change(stage, text):
            node = json.loads(text)
            if stage == "consumer":
                node["flushed"] = False
            return json.dumps(node)

        self.status_change = change
        with self.assertRaises(RuntimeError):
            workload.run_pipeline(self.root / "evidence", "normal", count=1)
        self.assertFalse(self.manifest()["execution_completed"])
        result = json.loads((self.output() / "analysis.json").read_text())
        self.assertEqual(result["collection"]["status"], "incomplete")

    def test_marker_cleanup_failure_is_recorded_after_manifest_is_saved(self):
        original = Path.unlink

        def denied(path, *args, **kwargs):
            if path.name == "consumer.started":
                raise PermissionError("marker removal denied")
            return original(path, *args, **kwargs)

        with patch.object(Path, "unlink", denied):
            with self.assertRaisesRegex(RuntimeError, "Marker cleanup failed"):
                workload.run_pipeline(self.root / "evidence", "normal", count=1)
        manifest = self.manifest()
        self.assertFalse(manifest["execution_completed"])
        self.assertEqual(manifest["metadata_errors"][0]["operation"], "remove_marker")


if __name__ == "__main__":
    unittest.main()
