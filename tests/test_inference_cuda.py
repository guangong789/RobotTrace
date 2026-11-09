"""CUDA 构建、逐次执行证据与失败边界；不启动 ROS2 或依赖 DDS 调度。"""

from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from robottrace import inference, inference_cuda as cuda
from test_inference import BOOT, OFFSETS, DOMAIN
from test_workload import FakeProcess


def profile():
    events = []
    nodes = [(f"{op}_{i}", op, "CUDAExecutionProvider")
             for op, count in cuda.CUDA_OPS.items() for i in range(count)]
    nodes += [(name, op, "CPUExecutionProvider") for name, op in cuda.CPU_NODES.items()]
    for call in range(5):
        start = 100 + call * 1000
        events.append(dict(cat="Session", ph="X", name="model_run", pid=23,
                           tid=23, ts=start, dur=900, args={}))
        for index, (name, op, provider) in enumerate(nodes):
            events.append(dict(cat="Node", ph="X", name=name + "_kernel_time", pid=23,
                               tid=23, ts=start + 10 + 2 * index, dur=1,
                               args=dict(provider=provider, op_name=op, node_index=str(index))))
    return events


class CudaEvidenceTests(unittest.TestCase):
    def test_five_calls_are_individually_proven(self):
        result = cuda.validate_profile(list(reversed(profile())), 23)
        self.assertEqual(result["inference_calls"], 5)
        self.assertEqual([row["cuda_nodes"] for row in result["calls"]], [101] * 5)
        self.assertTrue(all(row["cpu_nodes"] == sorted(cuda.CPU_NODES) for row in result["calls"]))

    def test_rejects_missing_malformed_cpu_only_extra_fallback_and_grouping(self):
        cases = [None, [], profile()[:-1], profile() + [deepcopy(profile()[1])]]
        for mutation in ("cpu_only", "extra_cpu", "provider", "pid", "tid", "outside",
                         "overlap", "missing_boundary", "aggregate_only", "missing_index", "malformed"):
            events = profile()
            if mutation == "cpu_only":
                for event in events:
                    if event["cat"] == "Node": event["args"]["provider"] = "CPUExecutionProvider"
            elif mutation == "extra_cpu": events[1]["args"]["provider"] = "CPUExecutionProvider"
            elif mutation == "provider": del events[1]["args"]["provider"]
            elif mutation == "pid": events[1]["pid"] = 99
            elif mutation == "tid": events[1]["tid"] = 99
            elif mutation == "outside": events[1]["ts"] = 99999
            elif mutation == "overlap": events[105]["ts"] = 100
            elif mutation == "missing_boundary": del events[105]
            elif mutation == "aggregate_only": events[1]["ts"] = events[106]["ts"]
            elif mutation == "missing_index": del events[1]["args"]["node_index"]
            elif mutation == "malformed": events[1]["dur"] = True
            cases.append(events)
        for index, events in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                cuda.validate_profile(events, 23)

    def test_json_rejects_duplicates_nonfinite_and_missing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            with self.assertRaises(FileNotFoundError): cuda.read_json(path)
            for text in ('{"a":1,"a":2}', '[NaN]', '{'):
                path.write_text(text)
                with self.assertRaises(ValueError): cuda.read_json(path)

    def test_registration_initialization_and_options_are_separate_checks(self):
        value = dict(requested_provider="cuda", build_mode="cuda", provider="CUDAExecutionProvider",
                     onnxruntime="1.20.1", requested_intra_op_threads=1, intra_op_threads=1,
                     inter_op_threads=1, execution_mode="sequential", graph_optimization="basic",
                     provider_registered=True, session_created=True, sdk_root="/opt/rt/ort",
                     cuda_options_readback="device_id=0;use_tf32=0;", warmup_execution={})
        for field, bad in (("provider_registered", False), ("session_created", False),
                           ("build_mode", "cpu"), ("provider", "CPUExecutionProvider"),
                           ("cuda_options_readback", "device_id=0;"),
                           ("cuda_options_readback", "device_id=0;use_tf32=1;")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                inference.verify_processor_configuration({**value, field: bad}, 1, provider="cuda", expected_pid=23)
        proof = cuda.validate_profile(profile(), 23)
        proof.update(profile_path="profile.json", profile_sha256="abc", loaded_libraries=[])
        value["warmup_execution"] = proof
        with patch.object(cuda, "read_json", return_value=profile()), patch.object(cuda, "digest", return_value="abc"), patch.object(cuda, "verify_loaded_libraries", return_value=[]):
            self.assertEqual(inference.verify_processor_configuration(value, 1, provider="cuda", expected_pid=23), value)
            # 可选 CUDA Graph readback 缺失保持未知，不影响必需选项检查。
            self.assertNotIn("enable_cuda_graph", value["cuda_options_readback"])
        with patch.object(cuda, "read_json", return_value=profile()), patch.object(cuda, "digest", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "profile proof mismatch"):
                inference.verify_processor_configuration(value, 1, provider="cuda", expected_pid=23)

    def test_loaded_sdk_requires_all_three_actual_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk = Path(temporary)
            (sdk / "lib").mkdir()
            expected = {}
            for name in cuda.GPU_HASHES:
                path = sdk / "lib" / name
                path.write_bytes(name.encode())
                expected[name] = cuda.digest(path)
            paths = [str(sdk / "lib" / name) for name in expected]
            with patch.object(cuda, "GPU_HASHES", expected):
                self.assertEqual(len(cuda.verify_loaded_libraries(paths, sdk)), 3)
                for bad in (paths[:-1], paths + paths[:1], ["/wrong/libonnxruntime.so"]):
                    with self.assertRaises(ValueError): cuda.verify_loaded_libraries(bad, sdk)
                Path(paths[0]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    cuda.verify_loaded_libraries(paths, sdk)

    def test_provider_rejects_before_creating_run_or_children(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(inference.subprocess, "Popen") as launch:
            parent = Path(temporary) / "absent"
            for provider, threads in (("auto", 1), ("cuda", 4), ("cuda", 1)):
                with self.assertRaises(ValueError):
                    inference.run_experiment(parent, provider=provider, intra_op_threads=threads)
            self.assertFalse(parent.exists())
            launch.assert_not_called()

    def test_frozen_sources_preserve_complete_artifact_validation(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            app, sdk, fixtures = root / "ros2/robottrace_inference", root / "sdk", root / "models/mobilenet"
            model = fixtures / "data/model.onnx"
            for name, value in (("ROOT", root), ("APP", app), ("SDK", sdk), ("FIXTURES", fixtures), ("MODEL", model)):
                stack.enter_context(patch.object(inference, name, value))
            stack.enter_context(patch.object(inference, "fixture_data", return_value=([], [])))
            binary = root / "bin/cpu/lib/robottrace_inference/inference_pipeline"
            source = app / "src/pipeline.cpp"
            for path in (binary, source, sdk / "lib/libonnxruntime.so.1.20.1", model):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"original")
            fixtures.joinpath("fixtures.json").write_text('{"files":{}}')
            for path in inference.build_sources():
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"source")
            stack.enter_context(patch.object(inference, "MODEL_SHA", inference.digest(model)))
            directory = root / "bin/cpu"
            paths = inference.build_sources() + inference.build_artifacts(directory)
            manifest = dict(provider="CPUExecutionProvider", onnxruntime="1.20.1",
                            files={str(path.relative_to(root)): inference.digest(path) for path in paths})
            inference.write(directory / "build.json", manifest)
            self.assertEqual(inference.verify_artifacts(directory), manifest)
            frozen = root / "evidence/frozen"
            for path in inference.build_sources():
                if path.relative_to(root).parts[0] in ("ros2", "experiments", "scripts", "tests"):
                    target = frozen / path.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
            source.write_bytes(b"new source")
            with self.assertRaisesRegex(ValueError, "changed"):
                inference.verify_artifacts(directory)
            self.assertEqual(inference.verify_artifacts(directory, source_root=frozen), manifest)
            with self.assertRaisesRegex(ValueError, "Missing/mismatched inference executable"):
                inference.verify_artifacts(directory, provider="cuda")
            binary.write_bytes(b"changed executable")
            with self.assertRaisesRegex(ValueError, "changed"):
                inference.verify_artifacts(directory, source_root=frozen)
            binary.unlink()
            with self.assertRaisesRegex(ValueError, "Missing/mismatched"):
                inference.verify_artifacts(directory, source_root=frozen)

    def test_failed_cuda_child_never_releases_source_and_is_cleaned(self):
        process = FakeProcess(17)
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            stack.enter_context(patch.object(inference, "verify_artifacts", return_value={}))
            stack.enter_context(patch.object(inference, "fixture_data", return_value=([], [])))
            stack.enter_context(patch.object(inference, "clock_evidence", return_value=(BOOT, OFFSETS, DOMAIN)))
            stack.enter_context(patch.dict(inference.os.environ, {
                "ROS_LOCALHOST_ONLY": "1", "ROBOTTRACE_HOST_CLOCK": json.dumps(dict(boot_id=BOOT, offsets=OFFSETS)),
                "ROBOTTRACE_IMAGE_ID": "test"}))
            launch = stack.enter_context(patch.object(inference.subprocess, "Popen", side_effect=[process, RuntimeError("CUDA initialization failed")]))
            stack.enter_context(patch.object(inference, "wait_until", return_value=True))
            with self.assertRaisesRegex(RuntimeError, "CUDA initialization failed"):
                inference.run_experiment(temporary, provider="cuda", cpu_artifact_dir="/cpu", cuda_artifact_dir="/gpu")
            out = next(Path(temporary).glob("vision_cuda_*"))
            self.assertFalse((out / "release").exists())
            self.assertIn("CUDA initialization failed", inference.read(out / "run.json")["execution_error"])
            self.assertEqual(process.signals, [signal.SIGINT])
            cpu_call, gpu_call = launch.call_args_list
            self.assertTrue(cpu_call.args[0][0].startswith("/cpu/"))
            self.assertTrue(gpu_call.args[0][0].startswith("/gpu/"))
            self.assertIn("--provider", gpu_call.args[0])
            self.assertNotIn("/opt/rt/ort", cpu_call.kwargs["env"]["LD_LIBRARY_PATH"])
            self.assertIn("/opt/rt/ort", gpu_call.kwargs["env"]["LD_LIBRARY_PATH"])

    def test_failed_container_preserved_success_removes_only_its_own_container(self):
        command = ["docker", "run", "--name", "new-task-container", "image"]
        with patch.object(cuda.subprocess, "run") as run:
            run.return_value.returncode = 2
            self.assertEqual(cuda.run_container(command), 2)
            self.assertEqual(run.call_count, 1)
        with patch.object(cuda.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(cuda.run_container(command), 0)
            self.assertEqual(run.call_args.args[0], ["docker", "rm", "new-task-container"])


if __name__ == "__main__":
    unittest.main()
