"""固定启动顺序、进程身份及失败清理测试，不依赖 DDS 调度。"""

from contextlib import ExitStack
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from robottrace import inference
from test_inference import BOOT, OFFSETS, DOMAIN, processor_configuration
from test_workload import FakeProcess


class InferenceStartupTests(unittest.TestCase):
    def run_case(self, timeout=False, spawn_error=False, bad_marker=None, threads=1,
                 bad_ready_configuration=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launched, commands, waits, cleaned = [], {}, [], []
            processes = {}
            qos = dict(depth=10, reliability="RELIABLE", durability="VOLATILE", history="KEEP_LAST")

            def save_ready(out, role, matching=False):
                inference.write(out / (role + ".ready.json"), {
                    "initialized": True,
                    "session_initializations": int(role == "processor"),
                    "warmup_count": 5 if role == "processor" else 0,
                    "matched_publishers": int(matching and role != "source"),
                    "matched_subscribers": int(matching and role != "consumer" and not (timeout and role == "processor")),
                    "input_qos": qos if role != "source" else None,
                    "output_qos": qos if role != "consumer" else None,
                    "inference_configuration": processor_configuration(
                        1 if bad_ready_configuration else threads
                    ) if role == "processor" else None,
                })

            def launch(command, **kwargs):
                role, out = command[1], Path(command[3])
                if spawn_error and role == "processor":
                    raise RuntimeError("processor spawn failed")
                launched.append(role)
                commands[role] = command
                process = processes[role] = FakeProcess({"consumer": 17, "processor": 29, "source": 41}[role])
                marker = {
                    "initialized": True,
                    "run_id": command[2],
                    "process_id": process.pid,
                    "inference_configuration": processor_configuration(threads) if role == "processor" else None,
                }
                if bad_marker:
                    marker.update(bad_marker)
                inference.write(out / (role + ".started"), marker)
                save_ready(out, role)
                (out / (role + ".jsonl")).write_text("")
                inference.write(out / (role + ".status.json"), {
                    "stage": role, "process_id": process.pid,
                    "stored": 0, "dropped": 0, "flushed": True,
                    "boot_id": BOOT, "time_offsets": OFFSETS, "clock_domain": DOMAIN,
                    "published": 0, "received": 0, "error": "",
                    "inference_configuration": processor_configuration(threads) if role == "processor" else None,
                })
                return process

            def wait(predicate, children, seconds, description):
                waits.append((description, seconds))
                out = Path(next(iter(commands.values()))[3])
                self.assertFalse((out / "release").exists())
                if description.endswith(" initialization"):
                    self.assertTrue(predicate())
                    return True
                self.assertEqual(description, "all four matched endpoints")
                for role in processes:
                    save_ready(out, role, matching=True)
                ready = predicate()
                if timeout:
                    self.assertIsNone(ready)
                    raise TimeoutError("Deadline waiting for all four matched endpoints")
                self.assertTrue(ready)
                return ready

            original_cleanup = inference._cleanup_children

            def cleanup(children):
                cleaned.extend(children)
                # 模拟 shutdown 更新状态；manifest 必须保留此前冻结的零匹配。
                if timeout:
                    out = Path(commands["processor"][3])
                    state = inference.read(out / "processor.ready.json")
                    state["matched_subscribers"] = 1
                    inference.write(out / "processor.ready.json", state)
                return original_cleanup(children)

            with ExitStack() as stack:
                stack.enter_context(patch.object(inference, "verify_artifacts", return_value={}))
                stack.enter_context(patch.object(inference, "fixture_data", return_value=([], [])))
                stack.enter_context(patch.object(inference, "clock_evidence", return_value=(BOOT, OFFSETS, DOMAIN)))
                stack.enter_context(patch.dict(inference.os.environ, {
                    "ROS_LOCALHOST_ONLY": "1",
                    "ROBOTTRACE_HOST_CLOCK": inference.json.dumps({"boot_id": BOOT, "offsets": OFFSETS}),
                    "ROBOTTRACE_IMAGE_ID": "test",
                }))
                stack.enter_context(patch.object(inference.subprocess, "Popen", side_effect=launch))
                stack.enter_context(patch.object(inference, "wait_until", side_effect=wait))
                stack.enter_context(patch.object(inference, "_cleanup_children", side_effect=cleanup))
                if timeout or spawn_error or bad_marker or bad_ready_configuration:
                    expected = (
                        "processor spawn failed" if spawn_error else
                        "Wrong initialized process/run identity" if bad_marker else
                        "Processor applied inference configuration mismatch" if bad_ready_configuration else
                        "Deadline waiting"
                    )
                    with self.assertRaisesRegex(RuntimeError, expected):
                        inference.run_experiment(root, 1, 5, startup_only=True, intra_op_threads=threads)
                else:
                    inference.run_experiment(root, 1, 5, startup_only=True, intra_op_threads=threads)

            out = next(root.glob("vision_startup_*"))
            manifest = inference.read(out / "run.json")
            expected_order = list(inference.ROLES)
            self.assertEqual(manifest["launch_order"], expected_order)
            self.assertEqual(manifest["cleanup_order"], list(inference.ROLES))
            self.assertEqual(launched, ["consumer"] if spawn_error or bad_marker else expected_order)
            self.assertEqual(cleaned, launched)
            self.assertFalse((out / "release").exists())
            self.assertEqual(manifest["execution_completed"], not (timeout or spawn_error or bad_marker or bad_ready_configuration))
            self.assertEqual(manifest["configuration"]["intra_op_threads"], threads)
            if not spawn_error and not bad_marker:
                self.assertEqual(manifest["processor_configuration"], processor_configuration(threads))
                self.assertTrue(manifest["processor_status_configuration_verified"])
                self.assertEqual(manifest["processor_readiness_configuration_verified"], not bad_ready_configuration)
            self.assertEqual(manifest["correctness"]["status"], "not_exercised")
            for role, command in commands.items():
                self.assertEqual(command, [
                    str(inference.PROGRAM), role, manifest["run_id"], str(out),
                    "1", str(inference.FIXTURES), str(inference.MODEL), "5",
                ] + (["--intra-op-threads", str(threads)] if role == "processor" else []))
                self.assertEqual(processes[role].signals, [signal.SIGINT])
                self.assertEqual(processes[role].waits, [4])
            if not spawn_error and not bad_marker:
                self.assertEqual(waits, [(role + " initialization", 25) for role in expected_order]
                                 + [("all four matched endpoints", 15)])
            if timeout:
                self.assertIn("Deadline waiting", manifest["execution_error"])
                self.assertEqual(
                    manifest["readiness_diagnostics"]["pre_cleanup"]["last_observation"]["unsatisfied"],
                    ["processor.matched_subscribers"],
                )
                self.assertEqual(manifest["pre_cleanup_readiness"]["processor"]["matched_subscribers"], 0)
                self.assertEqual(inference.read(out / "processor.ready.json")["matched_subscribers"], 1)
            if spawn_error:
                self.assertIn("processor spawn failed", manifest["execution_error"])

    def test_fixed_order_preserves_arguments_initialization_and_paused_source(self):
        self.run_case()

    def test_four_thread_configuration_is_verified_in_ready_status_and_manifest(self):
        self.run_case(threads=4)

    def test_wrong_ready_configuration_never_releases_source(self):
        self.run_case(threads=4, bad_ready_configuration=True)

    def test_timeout_freezes_failure_before_shutdown_overwrites_state(self):
        self.run_case(timeout=True)

    def test_partial_launch_failure_cleans_consumer(self):
        self.run_case(spawn_error=True)

    def test_initialized_marker_rejects_wrong_process_run_and_uninitialized_state(self):
        for marker in ({"process_id": 999}, {"run_id": "foreign"}, {"initialized": False}):
            with self.subTest(marker=marker):
                self.run_case(bad_marker=marker)

    def test_cli_forwards_startup_only_in_original_environment(self):
        with patch("sys.argv", ["inference", "--startup-only", "--intra-op-threads", "4"]), patch.object(
            inference, "clock_evidence", return_value=(BOOT, OFFSETS, DOMAIN)
        ), patch.object(inference.subprocess, "check_output", return_value="image\n"), patch.object(
            inference.subprocess, "call", return_value=0
        ) as call:
            self.assertEqual(inference.main(), 0)
        command = call.call_args.args[0]
        self.assertIn("--startup-only", command)
        self.assertIn("ROS_LOCALHOST_ONLY=1", command)
        self.assertEqual(command[command.index("--intra-op-threads") + 1], "4")


if __name__ == "__main__":
    unittest.main()
