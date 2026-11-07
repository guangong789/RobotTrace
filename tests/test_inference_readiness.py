"""推理启动诊断的确定性文件状态测试，不依赖 DDS 调度。"""

from copy import deepcopy
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from robottrace import inference
from test_workload import FakeProcess
from test_inference import BOOT, OFFSETS, DOMAIN, processor_configuration


class InferenceReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history, self.last = inference.ReadinessHistory(), {}
        qos = dict(
            depth=10, reliability="RELIABLE", durability="VOLATILE", history="KEEP_LAST"
        )
        self.states = {
            role: dict(
                initialized=True,
                session_initializations=int(role == "processor"),
                warmup_count=5 if role == "processor" else 0,
                matched_publishers=int(role != "source"),
                matched_subscribers=int(role != "consumer"),
                input_qos=qos if role != "source" else None,
                output_qos=qos if role != "consumer" else None,
                input_exists=role != "source",
                output_exists=role != "consumer",
                sample_begin_ns=100,
                sample_end_ns=101,
            )
            for role in inference.ROLES
        }

    def save(self):
        for role, state in self.states.items():
            inference.write(self.root / (role + ".ready.json"), state)

    def observe(self):
        return inference.endpoint_ready(
            self.root, inference.ROLES, self.last, self.history
        )

    def test_missing_files_zero_matches_and_delayed_full_matching(self):
        self.assertIsNone(self.observe())
        self.assertTrue(all(v is None for v in self.last.values()))
        self.states["processor"]["matched_subscribers"] = 0
        self.save()
        self.assertIsNone(self.observe())
        snap = self.history.data["last_observation"]
        self.assertEqual(snap["unsatisfied"], ["processor.matched_subscribers"])
        self.assertTrue(snap["roles"]["processor"]["output_exists"])
        size = len(self.history.data["entries"])
        for i in range(5):
            self.states["processor"]["sample_end_ns"] += 1
            self.save()
            self.assertIsNone(self.observe())
        self.assertEqual(len(self.history.data["entries"]), size)
        self.assertEqual(
            self.history.data["last_observation"]["roles"]["processor"][
                "sample_end_ns"
            ],
            106,
        )
        self.states["processor"]["matched_subscribers"] = 1
        self.save()
        self.assertTrue(self.observe())
        self.assertEqual(len(self.history.data["entries"]), size + 1)

    def test_unknown_absent_and_existing_endpoint_states_remain_distinct(self):
        self.observe()
        self.states["processor"].update(output_exists=False, matched_subscribers=0)
        self.save()
        self.observe()
        self.assertFalse(
            self.history.data["last_observation"]["roles"]["processor"]["output_exists"]
        )
        self.states["processor"]["output_exists"] = True
        self.save()
        self.observe()
        self.assertTrue(
            self.history.data["last_observation"]["roles"]["processor"]["output_exists"]
        )

    def test_history_bound_truncation_and_frozen_pre_cleanup_snapshot(self):
        for i in range(100):
            self.states["processor"]["matched_subscribers"] = i
            self.save()
            self.observe()
        data = self.history.data
        self.assertEqual(len(data["entries"]), 64)
        self.assertTrue(data["truncated"])
        self.assertEqual(data["discarded_entries"], 36)
        self.assertEqual(
            data["entries"][0]["roles"]["processor"]["matched_subscribers"], 0
        )
        self.history.freeze("timeout")
        before = deepcopy(data)
        self.states["processor"]["matched_subscribers"] = 0
        self.save()
        self.observe()
        self.history.freeze("success")
        self.assertEqual(self.history.data, before)
        self.assertEqual(
            data["pre_cleanup"]["last_observation"]["roles"]["processor"][
                "matched_subscribers"
            ],
            99,
        )

    def test_malformed_and_unreadable_data_is_explicit_before_failure(self):
        self.save()
        (self.root / "processor.ready.json").write_text("{bad")
        with self.assertRaises(ValueError):
            self.observe()
        self.assertIn(
            "read_error", self.history.data["last_observation"]["roles"]["processor"]
        )
        with patch.object(inference, "read", side_effect=PermissionError("unreadable")):
            with self.assertRaises(PermissionError):
                self.observe()
        self.assertIn(
            "unreadable",
            self.history.data["last_observation"]["roles"]["processor"]["read_error"],
        )

    def test_diagnostic_error_does_not_change_success_predicate(self):
        self.save()
        with patch.object(
            self.history, "observe", side_effect=OSError("diagnostic failed")
        ):
            self.assertTrue(self.observe())
        self.assertEqual(self.history.data["diagnostic_error_count"], 1)

    def test_diagnostic_freeze_error_does_not_skip_cleanup_or_replace_primary_error(
        self,
    ):
        child = FakeProcess(12)
        with patch.object(inference, "verify_artifacts", return_value={}), patch.object(
            inference, "fixture_data", return_value=([], [])
        ), patch.object(
            inference, "clock_evidence", return_value=(BOOT, OFFSETS, DOMAIN)
        ), patch.dict(
            inference.os.environ,
            {
                "ROS_LOCALHOST_ONLY": "1", "ROBOTTRACE_HOST_CLOCK": inference.json.dumps(
                    {"boot_id": BOOT, "offsets": OFFSETS}
                ),
                "ROBOTTRACE_IMAGE_ID": "test",
            },
        ), patch.object(
            inference.subprocess,
            "Popen",
            side_effect=[child, RuntimeError("spawn failed")],
        ), patch.object(
            inference, "wait_until", return_value=True
        ), patch.object(
            inference.ReadinessHistory, "freeze", side_effect=OSError("freeze failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "spawn failed"):
                inference.run_experiment(self.root)
        self.assertEqual(child.signals, [signal.SIGINT])
        manifest = inference.read(next(self.root.glob("vision_cpu_*")) / "run.json")
        self.assertIn("spawn failed", manifest["execution_error"])
        self.assertEqual(manifest["readiness_diagnostics"]["diagnostic_error_count"], 1)

    def test_startup_only_never_releases_source_or_claims_inference_correctness(self):
        processes = {}
        original_read = inference.read

        def launch(command, **kwargs):
            role, out = command[1], Path(command[3])
            processes[role] = FakeProcess(len(processes) + 1)
            (out / (role + ".jsonl")).write_text("")
            config = processor_configuration() if role == "processor" else None
            inference.write(out / (role + ".started"), dict(
                initialized=True, run_id=command[2], process_id=processes[role].pid,
                inference_configuration=config,
            ))
            inference.write(out / (role + ".ready.json"), dict(
                self.states[role], inference_configuration=config,
            ))
            return processes[role]

        def read(path):
            if path.name.endswith(".status.json"):
                role = path.name.split(".")[0]
                return dict(
                    stage=role,
                    process_id=processes[role].pid,
                    stored=0,
                    dropped=0,
                    flushed=True,
                    boot_id=BOOT,
                    time_offsets=OFFSETS,
                    clock_domain=DOMAIN,
                    published=0,
                    received=0,
                    inference_configuration=processor_configuration() if role == "processor" else None,
                )
            return original_read(path)

        with patch.object(inference, "verify_artifacts", return_value={}), patch.object(
            inference, "clock_evidence", return_value=(BOOT, OFFSETS, DOMAIN)
        ), patch.dict(
            inference.os.environ,
            {
                "ROS_LOCALHOST_ONLY": "1", "ROBOTTRACE_HOST_CLOCK": inference.json.dumps(
                    {"boot_id": BOOT, "offsets": OFFSETS}
                ),
                "ROBOTTRACE_IMAGE_ID": "test",
            },
        ), patch.object(
            inference.subprocess, "Popen", side_effect=launch
        ), patch.object(
            inference, "wait_until", side_effect=lambda predicate, *args: predicate()
        ), patch.object(
            inference, "read", side_effect=read
        ):
            out, analysis = inference.run_experiment(self.root, startup_only=True)
        manifest = original_read(out / "run.json")
        self.assertTrue(manifest["execution_completed"])
        self.assertFalse((out / "release").exists())
        self.assertEqual(manifest["correctness"]["status"], "not_exercised")
        self.assertEqual(analysis["event_count"], 0)
        self.assertEqual(analysis["observations"]["status"], "incomplete")
        self.assertEqual(manifest["expected_sequences"], {"source:1": []})
