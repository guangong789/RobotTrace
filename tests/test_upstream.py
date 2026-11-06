"""独立接入的证据边界和运行器故障注入，不依赖 ROS2 调度。"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from robottrace import upstream
from robottrace.analysis import analyze
from robottrace.events import CollectorEvidence, RunEvidence
from test_analysis import message
from test_workload import FakeProcess


class UpstreamContractTests(unittest.TestCase):
    def test_pinned_pristine_sources_and_licenses(self):
        records = upstream.read(upstream.APP / "upstream.json")
        self.assertEqual(set(records), {"demo_nodes_cpp", "topic_tools"})
        for name, record in records.items():
            self.assertRegex(record["revision"], r"^[0-9a-f]{40}$")
            self.assertIn(
                "Apache License",
                (upstream.APP / "upstream" / name / "LICENSE").read_text(),
            )
            for path, identity in record["files"].items():
                self.assertIn(record["revision"], identity["url"])
                self.assertEqual(
                    upstream.digest(upstream.APP / path), identity["sha256"]
                )
        # 补丁保留实际 SerializedMessage 发布，没有把 Relay 换成自制转发器。
        patch_text = (upstream.APP / "instrumentation.patch").read_text()
        self.assertIn("pub_->publish(*msg);", patch_text)
        self.assertNotIn("serialize_message(&decoded", patch_text)

    def test_binary_and_library_identity_are_checked_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [
                root / name
                for name in ("talker", "relay", "listener", "component.so", "input.cpp")
            ]
            for index, path in enumerate(files):
                path.write_text(str(index))
            value = {
                "executables": {
                    role: {"path": str(path), "sha256": upstream.digest(path)}
                    for role, path in zip(upstream.ROLES, files)
                },
                "libraries": {str(files[3]): upstream.digest(files[3])},
                "build_inputs": {"input.cpp": upstream.digest(files[4])},
            }
            (root / "build.json").write_text(json.dumps(value))
            with patch.object(upstream, "BIN", root), patch.object(
                upstream, "ROOT", root
            ):
                self.assertEqual(upstream.software_identity(), value)
                for path in files:
                    previous = path.read_text()
                    path.write_text("changed")
                    with self.assertRaises(ValueError):
                        upstream.software_identity()
                    path.write_text(previous)

    def test_missing_prefix_is_observation_gap_not_collector_loss(self):
        events = []
        for seq in range(1, 6):
            events += [
                event
                for event in message(source="demo:talker", sequence=seq)
                if seq > 2 or event.stage == "source"
            ]
        collectors = tuple(
            CollectorEvidence(
                stage, sum(e.stage == stage for e in events), 0, True, pid
            )
            for stage, pid in (("source", 1), ("processor", 2), ("consumer", 3))
        )
        context = RunEvidence(
            "run", "host", True, {"demo:talker": (1, 2, 3, 4, 5)}, True, collectors
        )
        result = analyze(events, context)
        self.assertEqual(result["collection"]["status"], "complete")
        self.assertEqual(result["observations"]["status"], "incomplete")
        self.assertEqual(result["observations"]["missing_observation_count"], 12)
        for item in result["messages"][:2]:
            self.assertIsNone(item["metrics"]["consumer_age_ns"]["value_ns"])
            self.assertEqual(
                item["metrics"]["consumer_age_ns"]["reason"], "MISSING_ENDPOINT"
            )

    def test_loaded_plugin_provenance_uses_real_process_mappings(self):
        maps = (
            "100-200 r-xp 0 0:0 1 /opt/ros/humble/lib/plugin.so\n"
            + "200-300 rw-p 0 0:0 1 /opt/ros/humble/lib/plugin.so\n"
            + "300-400 r-xp 0 0:0 1 /usr/lib/unrelated.so\n"
        )
        with patch.object(Path, "read_text", return_value=maps), patch.object(
            upstream, "digest", return_value="measured-hash"
        ) as hashed:
            result = upstream.loaded_libraries({"processor": FakeProcess(123)})
        self.assertEqual(
            result, {"processor": {"/opt/ros/humble/lib/plugin.so": "measured-hash"}}
        )
        hashed.assert_called_once()

    def test_clock_overflow_unflushed_and_overlap_keep_core_contract(self):
        events = message(source="demo:talker")
        collectors = tuple(
            CollectorEvidence(
                stage, sum(e.stage == stage for e in events), 0, True, pid
            )
            for stage, pid in (("source", 1), ("processor", 2), ("consumer", 3))
        )
        context = RunEvidence(
            "run", "host", True, {"demo:talker": (1,)}, True, collectors
        )
        self.assertEqual(analyze(events, context)["observations"]["status"], "complete")
        for bad in (
            replace(collectors[1], dropped=1),
            replace(collectors[1], flushed=False),
        ):
            result = analyze(
                events, replace(context, collectors=(collectors[0], bad, collectors[2]))
            )
            self.assertEqual(result["collection"]["status"], "incomplete")
        result = analyze(events, replace(context, clock_verified=False))
        self.assertIsNone(
            result["messages"][0]["metrics"]["consumer_age_ns"]["value_ns"]
        )
        incompatible = [
            replace(e, clock_domain="other") if e.stage == "consumer" else e
            for e in events
        ]
        self.assertIsNone(
            analyze(incompatible, context)["messages"][0]["metrics"]["consumer_age_ns"][
                "value_ns"
            ]
        )
        # test_analysis 的合法重叠样本已让消费进入早于转发回调退出。
        self.assertEqual(analyze(events, context)["trace_integrity"]["status"], "valid")


class UpstreamLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.processes = {}
        self.boot, self.offsets, self.domain = (
            "boot",
            "monotonic 0 0\nboottime 0 0\n",
            "host",
        )
        self.changes = lambda role, process, node: None
        self.mock(
            patch.object(
                upstream,
                "software_identity",
                return_value={"executables": {role: role for role in upstream.ROLES}},
            )
        )
        self.mock(
            patch.object(
                upstream,
                "loaded_libraries",
                return_value={role: {"mock.so": "test"} for role in upstream.ROLES},
            )
        )
        self.mock(
            patch.object(
                upstream,
                "clock_evidence",
                return_value=(self.boot, self.offsets, self.domain),
            )
        )
        self.mock(
            patch.dict(
                upstream.os.environ,
                {
                    "ROBOTTRACE_HOST_CLOCK": json.dumps(
                        {"boot_id": self.boot, "offsets": self.offsets}
                    ),
                    "ROBOTTRACE_IMAGE_ID": "test-image",
                },
            )
        )
        self.mock(patch.object(upstream.subprocess, "Popen", side_effect=self.spawn))
        self.graph = self.mock(
            patch.object(
                upstream.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "actual graph", ""),
            )
        )

    def mock(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def spawn(self, args, env, **kwargs):
        role = {value: key for key, value in upstream.ROLES.items()}[Path(args[0]).name]
        out = Path(env["RT_OUTPUT"])
        process = FakeProcess({"source": 1, "processor": 2, "consumer": 3}[role])
        self.processes[role] = process
        events = [
            replace(e, run_id=env["RT_RUN"])
            for e in message(source="demo:talker")
            if e.stage == role
        ]
        (out / (role + ".jsonl")).write_text(
            "".join(json.dumps(asdict(e)) + "\n" for e in events)
        )
        node = {
            "stage": role,
            "process_id": process.pid,
            "stored": len(events),
            "dropped": 0,
            "flushed": True,
            "invalid_messages": 0,
            "clock_domain": self.domain,
            "boot_id": self.boot,
            "time_offsets": self.offsets,
        }
        upstream.write(out / (role + ".started"), node)
        self.changes(role, process, node)
        upstream.write(out / (role + ".status.json"), node)
        qos = {"history": 1, "reliability": 1, "durability": 2, "depth": 10}
        upstream.write(
            out / (role + ".ready.json"),
            {
                "matched_publishers": int(role != "source"),
                "matched_subscribers": int(role == "processor"),
                "input_qos": qos if role != "source" else None,
                "output_qos": qos if role != "consumer" else None,
            },
        )
        if role == "processor":
            info = upstream.read(out / "source.ready.json")
            info["matched_subscribers"] = 1
            upstream.write(out / "source.ready.json", info)
        if role == "source":
            upstream.write(
                out / "prefix.json", {"published": 3, "matched_subscribers": 0}
            )
            upstream.write(out / "published.json", {"published": 20})
            upstream.write(out / "delivered.json", {"last_sequence": 20})
        return process

    def manifest(self):
        path = next(self.root.glob("upstream_*/run.json"))
        return upstream.read(path)

    def test_normal_shutdown_and_multi_executable_provenance(self):
        path, result = upstream.run_experiment(self.root, "normal")
        value = self.manifest()
        self.assertTrue(value["execution_completed"])
        self.assertTrue(value["clock_verified"])
        self.assertEqual(result["collection"]["status"], "complete")
        self.assertEqual(set(value["software"]["executables"]), set(upstream.ROLES))
        self.assertNotIn("binary_sha256", value)
        self.assertFalse(value["identity_binding"]["run_uuid_in_payload"])
        for process in self.processes.values():
            self.assertEqual(process.signals, [signal.SIGINT])
            self.assertEqual(process.waits, [4])
            self.assertEqual(process.kills, 0)
        self.assertTrue((path / "source.status.json").exists())

    def test_late_relay_is_started_only_after_prefix_confirmation(self):
        upstream.run_experiment(self.root, "late-relay")
        value = self.manifest()
        phases = [item["phase"] for item in value["phases"]]
        prefix_index = phases.index("prefix_completed")
        relay_index = next(
            i
            for i, item in enumerate(value["phases"])
            if item.get("stage") == "processor"
        )
        self.assertLess(prefix_index, relay_index)
        self.assertLess(relay_index, phases.index("ready"))
        self.assertLess(phases.index("ready"), phases.index("resumed"))
        self.assertEqual(value["expected_sequences"]["demo:talker"], list(range(1, 21)))

    def test_observer_failure_cleans_all_children_and_preserves_primary(self):
        self.graph.side_effect = OSError("observer failed")
        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            upstream.run_experiment(self.root, "normal")
        self.assertTrue(all(p.returncode == 0 for p in self.processes.values()))
        self.assertFalse(self.manifest()["execution_completed"])
        self.assertIn("observer failed", self.manifest()["execution_error"])

    def test_child_failure_and_cleanup_escalation_remain_bounded(self):
        def change(role, process, node):
            if role == "processor":
                process.returncode = 2
            if role == "consumer":
                process.wait_errors[4] = subprocess.TimeoutExpired("child", 4)
                process.wait_errors[2] = subprocess.TimeoutExpired("child", 2)
                process.kill_error = PermissionError("kill denied")

        self.changes = change
        with self.assertRaisesRegex(RuntimeError, "Unexpected process exit"):
            upstream.run_experiment(self.root, "normal")
        self.assertEqual(self.processes["consumer"].waits, [4, 2])
        self.assertEqual(self.processes["source"].returncode, 0)
        self.assertFalse(self.manifest()["execution_completed"])
        self.assertIsNone(self.manifest()["collectors"])
        self.assertTrue(self.manifest()["cleanup_errors"])

    def test_malformed_status_does_not_skip_cleanup_or_manifest(self):
        original = upstream.read

        def broken(path):
            return [] if Path(path).name == "consumer.status.json" else original(path)

        with patch.object(upstream, "read", side_effect=broken):
            with self.assertRaisesRegex(RuntimeError, "finalization incomplete"):
                upstream.run_experiment(self.root, "normal")
        self.assertTrue(all(p.returncode == 0 for p in self.processes.values()))
        self.assertIsNone(self.manifest()["collectors"])
        self.assertFalse(self.manifest()["nodes"]["consumer"]["status_available"])

    def test_unflushed_status_is_not_success(self):
        def change(role, process, node):
            if role == "processor":
                node["flushed"] = False

        self.changes = change
        with self.assertRaisesRegex(RuntimeError, "finalization incomplete"):
            upstream.run_experiment(self.root, "normal")
        self.assertFalse(self.manifest()["execution_completed"])
        analysis = upstream.read(next(self.root.glob("upstream_*/analysis.json")))
        self.assertEqual(analysis["collection"]["status"], "incomplete")

    def test_invalid_identity_fails_execution_without_discarding_valid_collector_report(
        self,
    ):
        def change(role, process, node):
            if role == "processor":
                node["invalid_messages"] = 1

        self.changes = change
        with self.assertRaisesRegex(RuntimeError, "finalization incomplete"):
            upstream.run_experiment(self.root, "normal")
        manifest = self.manifest()
        self.assertFalse(manifest["execution_completed"])
        self.assertEqual(manifest["nodes"]["processor"]["invalid_messages"], 1)
        self.assertEqual(len(manifest["collectors"]), 3)
        self.assertTrue(manifest["status_errors"])

    def test_manifest_write_failure_surfaces_primary_and_keeps_status(self):
        original = upstream.write

        def denied(path, value):
            if Path(path).name == "run.json":
                raise OSError("disk full")
            return original(path, value)

        self.graph.side_effect = OSError("primary observer error")
        with patch.object(upstream, "write", side_effect=denied):
            with self.assertRaisesRegex(
                RuntimeError, "failed to save run.json.*primary observer error"
            ):
                upstream.run_experiment(self.root, "normal")
        self.assertEqual(len(list(self.root.glob("upstream_*/*.status.json"))), 3)

    def test_phase_timeout_checks_liveness_and_has_deadline(self):
        with patch.object(upstream.time, "monotonic", side_effect=(0, 1, 4)):
            with patch.object(upstream.time, "sleep"):
                with self.assertRaises(TimeoutError):
                    upstream.wait_until(lambda: None, {}, 2, "phase")

    def test_lock_failure_still_produces_failure_manifest(self):
        with patch.object(upstream.fcntl, "flock", side_effect=BlockingIOError("busy")):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                upstream.run_experiment(self.root, "normal")
        self.assertFalse(self.manifest()["execution_completed"])
        self.assertIsNone(self.manifest()["collectors"])
        self.assertFalse(self.processes)


if __name__ == "__main__":
    unittest.main()
