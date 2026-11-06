"""用可控文件状态和假时钟验证 readiness 诊断，不依赖 DDS 时序。"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robottrace import upstream
import test_upstream as fixtures


class ReadinessHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = upstream.ReadinessHistory()
        self.qos = {"history": 1, "depth": 10, "reliability": 1, "durability": 2}

    def reports(self, processor_counts=(1, 1)):
        values = {}
        for role, counts in upstream.READINESS_COUNTS.items():
            if role == "processor":
                counts = processor_counts
            values[role] = {
                "matched_publishers": counts[0],
                "matched_subscribers": counts[1],
                "input_qos": deepcopy(self.qos) if role != "source" else None,
                "output_qos": deepcopy(self.qos) if role != "consumer" else None,
            }
            upstream.write(self.root / (role + ".ready.json"), values[role])
        return values

    def observe(self):
        return upstream.ready_snapshot(self.root, self.history)

    def test_missing_startup_files_are_unknown_endpoints_not_zero_matches(self):
        self.assertIsNone(self.observe())
        last = self.history.data["last_observation"]
        for state in last["roles"].values():
            self.assertFalse(state["file_present"])
            self.assertEqual(state["state"], "missing")
            self.assertIsNone(state["input_exists"])
            self.assertIsNone(state["matched_publishers"])
        self.assertTrue(last["unsatisfied"])

    def test_absent_endpoint_differs_from_existing_endpoint_with_zero_matches(self):
        values = self.reports((0, 0))
        values["processor"]["input_qos"] = None
        values["processor"]["output_qos"] = None
        upstream.write(self.root / "processor.ready.json", values["processor"])
        self.assertIsNone(self.observe())
        absent = self.history.data["last_observation"]["roles"]["processor"]
        self.assertFalse(absent["input_exists"])
        self.assertFalse(absent["output_exists"])
        self.reports((0, 0))
        self.assertIsNone(self.observe())
        present = self.history.data["last_observation"]["roles"]["processor"]
        self.assertTrue(present["input_exists"])
        self.assertTrue(present["output_exists"])
        self.assertEqual(present["matched_publishers"], 0)
        self.assertEqual(len(self.history.data["entries"]), 2)

    def test_partial_then_full_matching_preserves_original_result(self):
        self.reports((1, 0))
        self.assertIsNone(self.observe())
        last = self.history.data["last_observation"]
        self.assertEqual(
            last["unsatisfied"],
            [
                {
                    "role": "processor",
                    "condition": "matched_subscribers",
                    "expected": 1,
                    "observed": 0,
                }
            ],
        )
        values = self.reports()
        self.assertEqual(self.observe(), values)
        self.history.boundary("success")
        final = self.history.data["entries"][-1]
        self.assertEqual(final["event"], "success")
        self.assertEqual(final["predicate_result"], "ready")
        self.assertFalse(final["unsatisfied"])
        self.assertFalse(final["state_reobserved"])

    def test_unchanged_polls_are_counted_but_only_changes_are_retained(self):
        self.reports((0, 0))
        with patch.object(upstream.time, "monotonic_ns", side_effect=range(10, 20)):
            for _ in range(5):
                self.observe()
            self.reports((1, 0))
            self.observe()
        data = self.history.data
        self.assertEqual(data["poll_count"], 6)
        self.assertEqual(len(data["entries"]), 2)
        self.assertEqual([x["poll_index"] for x in data["entries"]], [1, 6])
        self.assertEqual(data["last_observation"]["observed_ns"], 15)

    def test_timeout_retains_last_actual_observation_and_separate_boundary(self):
        self.reports((1, 0))
        with patch.object(
            upstream.time, "monotonic", side_effect=(0, 0, 0.2, 0.4, 1.1)
        ), patch.object(
            upstream.time, "monotonic_ns", side_effect=(100, 200, 300, 400)
        ), patch.object(
            upstream.time, "sleep"
        ):
            with self.assertRaisesRegex(TimeoutError, "both DDS links matched"):
                upstream.wait_until(
                    self.observe,
                    {},
                    1,
                    "both DDS links matched",
                    diagnostics=self.history,
                )
        data = self.history.data
        self.assertEqual(data["poll_count"], 3)
        self.assertEqual(len(data["entries"]), 2)
        self.assertEqual(data["entries"][-1]["event"], "timeout")
        self.assertEqual(data["entries"][-1]["observed_ns"], 300)
        self.assertEqual(data["entries"][-1]["boundary_ns"], 400)
        self.assertFalse(data["entries"][-1]["state_reobserved"])

    def test_malformed_json_preserves_original_exception_and_observed_scope(self):
        self.reports()
        (self.root / "processor.ready.json").write_text("{broken")
        with self.assertRaises(json.JSONDecodeError):
            self.observe()
        states = self.history.data["last_observation"]["roles"]
        self.assertTrue(states["processor"]["file_present"])
        self.assertEqual(states["processor"]["state"], "malformed")
        self.assertIsNone(states["processor"]["matched_publishers"])
        # 原轮询在异常处终止，不能把尚未读取的 Consumer 说成文件缺失。
        self.assertIsNone(states["consumer"]["file_present"])
        self.assertEqual(states["consumer"]["state"], "unknown")

    def test_unreadable_status_is_explicit_and_does_not_become_unmatched(self):
        self.reports()
        original = Path.read_text

        def denied(path, *args, **kwargs):
            if path.name == "processor.ready.json":
                raise PermissionError("status denied")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", denied):
            with self.assertRaises(PermissionError):
                self.observe()
        state = self.history.data["last_observation"]["roles"]["processor"]
        self.assertEqual(state["state"], "unreadable")
        self.assertIsNone(state["input_exists"])
        self.assertIsNone(state["matched_publishers"])

    def test_malformed_fields_and_qos_keep_existing_failure_behavior(self):
        for broken in (
            [],
            {},
            {"matched_publishers": 1},
            {
                "matched_publishers": 1,
                "matched_subscribers": 1,
                "input_qos": [],
                "output_qos": None,
            },
        ):
            with self.subTest(broken=broken):
                self.reports()
                upstream.write(self.root / "processor.ready.json", broken)
                with self.assertRaises((KeyError, TypeError)):
                    self.observe()
                self.assertEqual(
                    self.history.data["last_observation"]["predicate_result"], "error"
                )
                self.assertEqual(
                    self.history.data["last_observation"]["roles"]["processor"][
                        "state"
                    ],
                    "malformed",
                )

    def test_json_null_is_present_but_unknown_and_still_waits(self):
        self.reports()
        upstream.write(self.root / "processor.ready.json", None)
        self.assertIsNone(self.observe())
        state = self.history.data["last_observation"]["roles"]["processor"]
        self.assertTrue(state["file_present"])
        self.assertEqual(state["state"], "malformed")
        self.assertIsNone(state["output_exists"])

    def test_exact_match_and_qos_policy_are_not_relaxed(self):
        self.reports((1, 2))
        self.assertIsNone(self.observe())
        values = self.reports()
        values["processor"]["output_qos"]["reliability"] = 2
        upstream.write(self.root / "processor.ready.json", values["processor"])
        with self.assertRaisesRegex(ValueError, "Unsupported actual endpoint QoS"):
            self.observe()

    def test_bound_keeps_first_latest_and_terminal_entries_with_explicit_loss(self):
        count = upstream.READINESS_HISTORY_LIMIT + 10
        for n in range(count):
            self.reports((1, n))
            self.observe()
        self.history.boundary("timeout")
        data = self.history.data
        self.assertEqual(len(data["entries"]), upstream.READINESS_HISTORY_LIMIT)
        self.assertTrue(data["truncated"])
        self.assertEqual(
            data["dropped_entries"], count + 1 - upstream.READINESS_HISTORY_LIMIT
        )
        self.assertEqual(data["entries"][0]["poll_index"], 1)
        self.assertEqual(data["entries"][-2]["poll_index"], count)
        self.assertEqual(data["entries"][-1]["event"], "timeout")
        self.assertEqual(
            data["entries"][-1]["roles"]["processor"]["matched_subscribers"], count - 1
        )

    def test_freezing_detaches_state_and_prevents_later_overwrite(self):
        self.reports((1, 0))
        self.observe()
        states = deepcopy(self.history.data["last_observation"]["roles"])
        self.history.freeze(states)
        before = deepcopy(self.history.data)
        states["processor"]["matched_subscribers"] = 99
        self.reports()
        self.observe()
        self.history.boundary("success")
        self.history.freeze(
            {role: upstream._role_readiness(False) for role in upstream.ROLES}
        )
        self.assertEqual(self.history.data, before)

    def test_diagnostic_recording_error_does_not_change_success(self):
        values = self.reports()
        with patch.object(
            self.history, "_append", side_effect=OSError("diagnostic failed")
        ):
            for _ in range(8):
                self.assertEqual(self.observe(), values)
        self.assertEqual(self.history.data["diagnostic_error_count"], 8)
        self.assertEqual(len(self.history.data["diagnostic_errors"]), 4)
        self.assertEqual(
            self.history.data["last_observation"]["predicate_result"], "ready"
        )


class ReadinessFailureBoundaryTests(unittest.TestCase):
    def setUp(self):
        # 复用现有进程模拟器，不重新运行或继承其整组测试。
        self.harness = fixtures.UpstreamLifecycleTests()
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        self.root = self.harness.root
        self.primary = TimeoutError("Deadline waiting for both DDS links matched")
        original = upstream.wait_until

        def timeout(predicate, processes, seconds, description, **kwargs):
            if description == "both DDS links matched":
                self.assertEqual(seconds, 15)
                out = next(self.root.glob("upstream_*"))
                report = upstream.read(out / "processor.ready.json")
                report["matched_subscribers"] = 0
                upstream.write(out / "processor.ready.json", report)
                self.assertIsNone(predicate())
                kwargs["diagnostics"].boundary("timeout")
                raise self.primary
            return original(predicate, processes, seconds, description, **kwargs)

        self.harness.mock(patch.object(upstream, "wait_until", side_effect=timeout))

    def test_timeout_snapshot_survives_cleanup_overwriting_status_files(self):
        original = upstream._cleanup_children

        def overwrite(processes):
            out = next(self.root.glob("upstream_*"))
            report = upstream.read(out / "processor.ready.json")
            report.update(input_qos=None, output_qos=None, matched_publishers=0)
            upstream.write(out / "processor.ready.json", report)
            return original(processes)

        with patch.object(upstream, "_cleanup_children", side_effect=overwrite):
            with self.assertRaises(RuntimeError) as caught:
                upstream.run_experiment(self.root, "late-relay")
        self.assertIs(caught.exception.__cause__, self.primary)
        manifest = self.harness.manifest()
        data = manifest["readiness_diagnostics"]
        self.assertTrue(data["frozen_before_cleanup"])
        self.assertTrue(data["pre_cleanup"]["roles"]["processor"]["output_exists"])
        self.assertEqual(
            data["pre_cleanup"]["roles"]["processor"]["matched_subscribers"], 0
        )
        self.assertIsNotNone(manifest["failure_readiness"]["processor"]["output_qos"])
        terminal = upstream.read(
            next(self.root.glob("upstream_*/processor.ready.json"))
        )
        self.assertIsNone(terminal["output_qos"])
        self.assertIn("timeout", [entry["event"] for entry in data["entries"]])
        self.assertTrue(all(p.returncode == 0 for p in self.harness.processes.values()))

    def test_manifest_diagnostic_persistence_failure_occurs_after_all_cleanup(self):
        original = upstream.write

        def denied(path, value):
            if Path(path).name == "run.json":
                self.assertIn("pre_cleanup", value["readiness_diagnostics"])
                self.assertTrue(
                    all(p.returncode == 0 for p in self.harness.processes.values())
                )
                raise OSError("diagnostic manifest disk full")
            return original(path, value)

        with patch.object(upstream, "write", side_effect=denied):
            with self.assertRaisesRegex(
                RuntimeError, "failed to save run.json.*disk full"
            ) as caught:
                upstream.run_experiment(self.root, "late-relay")
        self.assertIs(caught.exception.__cause__, self.primary)
        self.assertEqual(len(list(self.root.glob("upstream_*/*.status.json"))), 3)

    def test_capture_failure_keeps_primary_and_still_cleans_every_child(self):
        with patch.object(
            upstream.ReadinessHistory,
            "_append",
            side_effect=OSError("diagnostic storage failed"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                upstream.run_experiment(self.root, "late-relay")
        self.assertIs(caught.exception.__cause__, self.primary)
        manifest = self.harness.manifest()
        self.assertFalse(manifest["execution_completed"])
        self.assertGreater(
            manifest["readiness_diagnostics"]["diagnostic_error_count"], 0
        )
        self.assertTrue(all(p.returncode == 0 for p in self.harness.processes.values()))


if __name__ == "__main__":
    unittest.main()
