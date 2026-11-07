"""推理接入的状态、身份、参考输出及失败证据测试，不启动 ROS2。"""

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from robottrace import inference
from test_workload import FakeProcess

BOOT = "11111111-2222-3333-4444-555555555555"
OFFSETS = "monotonic 0 0\nboottime 0 0\n"
DOMAIN = "linux:CLOCK_MONOTONIC:" + BOOT + ":offset=0"


def processor_configuration(threads=1):
    return dict(
        requested_intra_op_threads=threads, intra_op_threads=threads,
        inter_op_threads=1, execution_mode="sequential", graph_optimization="basic",
        cpu_arena=True, provider="CPUExecutionProvider", onnxruntime="1.20.1",
        applied_via="ORT setter and Session construction succeeded",
    )


class InferenceTests(unittest.TestCase):
    def test_processor_thread_selection_and_applied_configuration_evidence(self):
        for threads in (1, 4):
            self.assertEqual(inference.validate_intra_op_threads(threads), threads)
            value = processor_configuration(threads)
            self.assertEqual(inference.verify_processor_configuration(value, threads), value)
        for invalid in (0, -1, 2, True, 1.0, "4", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                inference.validate_intra_op_threads(invalid)
        for key, wrong in (("intra_op_threads", 1), ("requested_intra_op_threads", 1),
                           ("inter_op_threads", 4), ("cpu_arena", 1),
                           ("execution_mode", "parallel"), ("graph_optimization", "all"),
                           ("applied_via", "CLI argument only")):
            value = processor_configuration(4)
            value[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                inference.verify_processor_configuration(value, 4)
        with self.assertRaises(ValueError):
            inference.verify_processor_configuration(None, 4)

    def test_load_parameters_preserve_defaults_and_explicit_bounds(self):
        self.assertEqual(inference.validate_workload(20, 5), 200000000)
        self.assertEqual(inference.validate_workload(1000, 1000), 1000000)
        self.assertEqual(inference.validate_workload(1000, 100), 10000000)
        for count, rate in [
            (0, 5),
            (1001, 1000),
            (True, 5),
            (20, 0),
            (20, float("nan")),
            (20, float("inf")),
            (20, True),
            (20, 10001),
            (1000, 5),
        ]:
            with self.subTest(count=count, rate=rate), self.assertRaises(ValueError):
                inference.validate_workload(count, rate)

    def test_observed_cadence_uses_intervals_not_requested_rate(self):
        from experiments.cpu_inference.load_summary import cadence

        measured = cadence([3_000_000_000, 0, 1_000_000_000])
        self.assertAlmostEqual(measured["hz"], 2 / 3)
        self.assertEqual(measured["span_ns"], 3_000_000_000)
        self.assertEqual(measured["gap_ns"]["p95"], 2_000_000_000)
        for values in ([], [1], [1, 1]):
            self.assertIsNone(cadence(values)["hz"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixtures(self, count=5):
        expected, _ = inference.fixture_data()
        manifest = {
            "run_id": "r",
            "nodes": {
                "source": {"published": count},
                "processor": {"published": count, "inference_operations": count},
                "consumer": {"valid_unique": count},
            },
        }
        result = [
            {
                "run_id": "r",
                "source_id": "source:1",
                "sequence": seq,
                "sample_id": (seq - 1) % inference.SAMPLE_COUNT,
                "valid": True,
                "logits": expected[(seq - 1) % inference.SAMPLE_COUNT],
                "input_payload_verified": True,
                "api_returned": True,
            }
            for seq in range(1, count + 1)
        ]
        for role in ("processor", "consumer"):
            self.write_rows(role, result)
        return manifest, result

    def write_rows(self, role, values):
        (self.root / (role + ".results.jsonl")).write_text(
            "".join(json.dumps(v) + "\n" for v in values)
        )

    def test_distinct_photo_tensors_and_all_thousand_golden_outputs(self):
        outputs, hashes = inference.fixture_data()
        self.assertEqual(len(set(hashes)), 2)
        self.assertEqual([len(v) for v in outputs], [1000, 1000])
        manifest, _ = self.fixtures()
        self.assertEqual(
            inference.correctness(self.root, manifest, 5)["status"], "passed"
        )

    def test_incorrect_nonfinite_sample_and_duplicate_results_fail(self):
        for fault in (
            "logit",
            "last_logit",
            "shape",
            "nonfinite",
            "sample",
            "duplicate",
            "foreign",
        ):
            with self.subTest(fault=fault):
                manifest, result = self.fixtures()
                if fault == "logit":
                    result[0]["logits"][0] += 10
                elif fault == "last_logit":
                    result[0]["logits"][-1] += 10
                elif fault == "shape":
                    result[0]["logits"].pop()
                elif fault == "nonfinite":
                    result[0]["logits"][0] = None
                elif fault == "sample":
                    result[0]["sample_id"] = 1
                elif fault == "duplicate":
                    result[1] = deepcopy(result[0])
                else:
                    result[0]["run_id"] = "other"
                self.write_rows("consumer", result)
                self.assertEqual(
                    inference.correctness(self.root, manifest, 5)["status"], "failed"
                )

    def test_missing_result_evidence_is_not_proof_of_incorrect_numeric_output(self):
        manifest, result = self.fixtures()
        self.write_rows("consumer", result[:-1])
        self.assertEqual(
            inference.correctness(self.root, manifest, 5)["status"],
            "insufficient_evidence",
        )

    def test_numeric_correctness_is_separate_from_later_execution_failure(self):
        manifest, _ = self.fixtures()
        manifest["nodes"]["consumer"]["error"] = "failure after correct results"
        self.assertEqual(
            inference.correctness(self.root, manifest, 5)["status"], "passed"
        )

    def test_readiness_requires_warmup_exact_matches_and_actual_qos(self):
        qos = {
            "depth": 10,
            "reliability": "RELIABLE",
            "durability": "VOLATILE",
            "history": "KEEP_LAST",
        }
        states = {
            role: {
                "initialized": True,
                "session_initializations": 1 if role == "processor" else 0,
                "warmup_count": 5 if role == "processor" else 0,
                "matched_publishers": 0 if role == "source" else 1,
                "matched_subscribers": 0 if role == "consumer" else 1,
                "input_qos": qos if role != "source" else None,
                "output_qos": qos if role != "consumer" else None,
            }
            for role in inference.ROLES
        }
        last = {}
        self.assertIsNone(inference.endpoint_ready(self.root, inference.ROLES, last))
        self.assertTrue(all(v is None for v in last.values()))

        def save():
            for role, state in states.items():
                inference.write(self.root / (role + ".ready.json"), state)

        save()
        self.assertTrue(inference.endpoint_ready(self.root, inference.ROLES, last))
        for key, value in [
            ("warmup_count", 4),
            ("matched_subscribers", 0),
            ("matched_subscribers", 2),
        ]:
            old = states["processor"][key]
            states["processor"][key] = value
            save()
            self.assertIsNone(
                inference.endpoint_ready(self.root, inference.ROLES, last)
            )
            states["processor"][key] = old
        states["processor"]["output_qos"] = {**qos, "depth": 1}
        save()
        self.assertIsNone(inference.endpoint_ready(self.root, inference.ROLES, last))

    def test_wait_detects_child_failure_and_deadline_without_scheduler_timing(self):
        process = FakeProcess(1)
        process.returncode = 2
        with self.assertRaisesRegex(RuntimeError, "Unexpected process exit"):
            inference.wait_until(lambda: None, {"processor": process}, 1, "test")
        with patch.object(inference.time, "monotonic", side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                inference.wait_until(lambda: None, {}, 1, "test")

    def test_failed_launch_preserves_manifest_and_cleans_started_child(self):
        process = FakeProcess(12)
        host = {"boot_id": BOOT, "offsets": OFFSETS}
        with patch.object(inference, "verify_artifacts", return_value={}), patch.object(
            inference, "clock_evidence", return_value=(BOOT, OFFSETS, DOMAIN)
        ), patch.dict(
            os.environ,
            {"ROS_LOCALHOST_ONLY": "1", "ROBOTTRACE_HOST_CLOCK": json.dumps(host), "ROBOTTRACE_IMAGE_ID": "test"},
        ), patch.object(
            inference.subprocess,
            "Popen",
            side_effect=[process, RuntimeError("processor spawn failed")],
        ), patch.object(
            inference, "wait_until", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "processor spawn failed"):
                inference.run_experiment(self.root)
        out = next(self.root.glob("vision_cpu_*"))
        manifest = inference.read(out / "run.json")
        analysis = inference.read(out / "analysis.json")
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertFalse(manifest["execution_completed"])
        self.assertIsNone(manifest["collectors"])
        self.assertFalse(manifest["clock_verified"])
        self.assertEqual(analysis["execution"]["status"], "failed")
        self.assertEqual(analysis["observations"]["missing_observation_count"], 180)

    def test_persistence_failure_still_cleans_children_and_preserves_primary_error(
        self,
    ):
        process = FakeProcess(12)
        host = {"boot_id": BOOT, "offsets": OFFSETS}
        with patch.object(inference, "verify_artifacts", return_value={}), patch.object(
            inference, "clock_evidence", return_value=(BOOT, OFFSETS, DOMAIN)
        ), patch.dict(
            os.environ,
            {"ROS_LOCALHOST_ONLY": "1", "ROBOTTRACE_HOST_CLOCK": json.dumps(host), "ROBOTTRACE_IMAGE_ID": "test"},
        ), patch.object(
            inference.subprocess,
            "Popen",
            side_effect=[process, RuntimeError("primary spawn error")],
        ), patch.object(
            inference, "wait_until", return_value=True
        ), patch.object(
            inference, "write", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(
                RuntimeError, "primary spawn error.*manifest persistence failed"
            ):
                inference.run_experiment(self.root)
        self.assertEqual(process.signals, [signal.SIGINT])

    def test_untrusted_clock_or_trace_withholds_local_diagnostic_summary(self):
        result = inference.diagnostics(
            self.root,
            {"clock_verified": False},
            {"trace_integrity": {"status": "valid"}},
        )
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
