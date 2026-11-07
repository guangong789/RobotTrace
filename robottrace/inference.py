"""固定照片 MobileNetV2 CPU 推理实验；不改变 V1 核心或原有运行器。"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
import uuid

from .analysis import statistics
from .events import CollectorEvidence
from .io import load_and_analyze, save_report, unique_fields
from .workload import _cleanup_children, clock_evidence, write

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "ros2/robottrace_inference"
BIN = ROOT / "bin/inference"
PROGRAM = BIN / "lib/robottrace_inference/inference_pipeline"
FIXTURES = ROOT / "models/mobilenet"
SDK = ROOT / ".deps/onnxruntime-linux-x64-1.20.1"
MODEL = FIXTURES / "data/mobilenetv2-12.onnx"
MODEL_SHA = "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
SAMPLE_COUNT, INPUT_ELEMENTS, OUTPUT_ELEMENTS = 2, 150528, 1000
ATOL, RTOL = 1e-4, 1e-4
ROLES = ("consumer", "processor", "source")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_fields)


def rows(path):
    return [
        json.loads(line, object_pairs_hook=unique_fields)
        for line in Path(path).read_text().splitlines()
    ]


def fixture_data():
    metadata = read(FIXTURES / "fixtures.json")
    if (
        metadata["model_sha256"] != MODEL_SHA
        or metadata["sample_count"] != SAMPLE_COUNT
        or metadata["input_shape"] != [1, 3, 224, 224]
        or metadata["output_shape"] != [1, 1000]
    ):
        raise ValueError("Vision fixture contract mismatch")
    outputs, hashes = [], []
    for sample in range(SAMPLE_COUNT):
        for label, count in (("input", INPUT_ELEMENTS), ("expected", OUTPUT_ELEMENTS)):
            name = f"{label}-{sample}.f32"
            payload = (FIXTURES / "data" / name).read_bytes()
            fingerprint = hashlib.sha256(payload).hexdigest()
            if len(payload) != count * 4 or fingerprint != metadata["files"][name]:
                raise ValueError("Fixture bytes/hash mismatch: " + name)
            values = struct.unpack(f"<{count}f", payload)
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Non-finite fixture: " + name)
            if label == "input":
                hashes.append(fingerprint)
            else:
                outputs.append(list(values))
    if len(set(hashes)) != SAMPLE_COUNT:
        raise ValueError("Distinct photo tensors required")
    return outputs, hashes


def record_build():
    if (
        digest(MODEL) != MODEL_SHA
        or (SDK / "VERSION_NUMBER").read_text().strip() != "1.20.1"
    ):
        raise ValueError("Wrong model or CPU SDK")
    fixture_data()
    inputs = [p for p in APP.rglob("*") if p.is_file()]
    inputs += [
        ROOT / "scripts/build-inference.sh",
        ROOT / "tests/inference_contract.cpp",
        ROOT / "ros2/robottrace_examples/include/robottrace/events.hpp",
        ROOT / "experiments/cpu_inference/mobilenet.hpp",
        FIXTURES / "fixtures.json",
        FIXTURES / "prepare.py",
    ]
    artifacts = [p for p in (BIN / "lib").rglob("*") if p.is_file()]
    artifacts += [MODEL, SDK / "lib/libonnxruntime.so.1.20.1"]
    artifacts += [
        FIXTURES / "data" / name for name in read(FIXTURES / "fixtures.json")["files"]
    ]
    write(
        BIN / "build.json",
        {
            "files": {
                str(p.relative_to(ROOT)): digest(p)
                for p in sorted(set(inputs + artifacts))
            },
            "compiler": subprocess.check_output(
                ["g++", "--version"], text=True, timeout=5
            ).splitlines()[0],
            "cxx_standard": 17,
            "onnxruntime": "1.20.1",
            "provider": "CPUExecutionProvider",
            "versions": subprocess.check_output(
                [
                    "dpkg-query",
                    "-W",
                    "ros-humble-rclcpp",
                    "ros-humble-rmw-fastrtps-cpp",
                    "ros-humble-rmw-fastrtps-shared-cpp",
                    "ros-humble-fastrtps",
                    "ros-humble-rosidl-typesupport-cpp",
                ],
                text=True,
                timeout=5,
            ),
        },
    )


def verify_artifacts():
    value = read(BIN / "build.json")
    for name, expected in value["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Build input/artifact changed; rebuild inference: " + name)
    if digest(MODEL) != MODEL_SHA:
        raise ValueError("Wrong inference model")
    fixture_data()
    return value


def wait_until(predicate, processes, seconds, description):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        failed = {
            role: p.returncode for role, p in processes.items() if p.poll() is not None
        }
        if failed:
            raise RuntimeError("Unexpected process exit: " + repr(failed))
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise TimeoutError("Deadline waiting for " + description)


class ReadinessHistory:
    """只记录启动观测，保留首条、最近变化和冻结边界；不参与放行判定。"""

    def __init__(self):
        self.data = {
            "limit": 64,
            "entries": [],
            "observations": 0,
            "truncated": False,
            "discarded_entries": 0,
            "errors": [],
        }
        self.signature = None
        self.frozen = False

    def error(self, exc):
        self.data["diagnostic_error_count"] = (
            self.data.get("diagnostic_error_count", 0) + 1
        )
        if len(self.data["errors"]) < 4:
            self.data["errors"].append(repr(exc)[:240])

    def observe(self, states, unsatisfied, begin, phase):
        if self.frozen:
            return
        snapshot = {
            "read_begin_ns": begin,
            "read_end_ns": time.monotonic_ns(),
            "phase": phase,
            "roles": deepcopy(states),
            "unsatisfied": list(unsatisfied),
            "ready": not unsatisfied,
        }
        self.data["observations"] += 1
        self.data.setdefault("first_observation", snapshot)
        self.data["last_observation"] = snapshot
        stable = {
            role: (
                {
                    k: v
                    for k, v in state.items()
                    if k not in ("sample_begin_ns", "sample_end_ns")
                }
                if isinstance(state, dict)
                else state
            )
            for role, state in states.items()
        }
        signature = (phase, stable, list(unsatisfied))
        if signature != self.signature:
            self.signature = deepcopy(signature)
            entries = self.data["entries"]
            if len(entries) == self.data["limit"]:
                del entries[1]
                self.data["truncated"] = True
                self.data["discarded_entries"] += 1
            entries.append(snapshot)

    def freeze(self, outcome):
        if not self.frozen:
            self.data["pre_cleanup"] = {
                "boundary_ns": time.monotonic_ns(),
                "outcome": outcome,
                "last_observation": deepcopy(self.data.get("last_observation")),
            }
            self.frozen = True


def readiness_unsatisfied(states):
    unsatisfied = []
    for role, state in states.items():
        if not state or state.get("initialized") is not True:
            unsatisfied.append(role + ".initialized")
            continue
        if role == "processor" and (
            state.get("session_initializations") != 1 or state.get("warmup_count") != 5
        ):
            unsatisfied.append("processor.model_and_warmup")
        pubs, subs = {"source": (0, 1), "processor": (1, 1), "consumer": (1, 0)}[role]
        for key, expected in (
            ("matched_publishers", pubs),
            ("matched_subscribers", subs),
        ):
            if state.get(key) != expected:
                unsatisfied.append(role + "." + key)
        for name in ("input_qos", "output_qos"):
            required = (name == "input_qos" and role != "source") or (
                name == "output_qos" and role != "consumer"
            )
            if required and state.get(name) != {
                "depth": 10,
                "reliability": "RELIABLE",
                "durability": "VOLATILE",
                "history": "KEEP_LAST",
            }:
                unsatisfied.append(role + "." + name)
    return unsatisfied


def endpoint_ready(out, roles, last, history=None, phase="matching"):
    begin, states, errors = time.monotonic_ns(), {}, []
    for role in roles:
        path = out / (role + ".ready.json")
        try:
            state = read(path) if path.is_file() else None
            if state is not None and not isinstance(state, dict):
                raise ValueError("Non-object readiness state: " + role)
            states[role] = state
        except (OSError, ValueError, TypeError) as exc:
            states[role] = {"read_error": repr(exc)}
            errors.append(exc)
    last.clear()
    last.update(states)
    unsatisfied = readiness_unsatisfied(states)
    if history is not None:
        try:
            history.observe(states, unsatisfied, begin, phase)
        except (OSError, ValueError, TypeError) as exc:
            history.error(exc)
    if errors:
        raise errors[0]
    return states if not unsatisfied else None


def correctness(out, manifest, count):
    nodes = manifest["nodes"]
    if any(n.get("invalid") or n.get("duplicates") for n in nodes.values()):
        return {
            "status": "failed",
            "reason": "Node reported invalid or duplicate processing",
        }
    files = [out / (r + ".results.jsonl") for r in ("processor", "consumer")]
    if len(nodes) != 3 or not all(p.is_file() for p in files):
        return {
            "status": "insufficient_evidence",
            "reason": "Missing node/result evidence",
        }
    expected, _ = fixture_data()
    observed = [rows(p) for p in files]
    if any(len(records) != count for records in observed):
        return {
            "status": "insufficient_evidence",
            "reason": "Incomplete per-message result evidence",
        }
    for records in observed:
        identities = set()
        for record in records:
            seq = record["sequence"]
            key = (record["run_id"], record["source_id"], seq)
            if (
                key in identities
                or key[:2] != (manifest["run_id"], "source:1")
                or type(seq) is not int
                or not 1 <= seq <= count
            ):
                return {
                    "status": "failed",
                    "reason": "Result identity mismatch/duplicate",
                }
            identities.add(key)
            sample = (seq - 1) % SAMPLE_COUNT
            logits = record["logits"]
            if (
                record["sample_id"] != sample
                or record["valid"] is not True
                or not isinstance(logits, list)
                or len(logits) != OUTPUT_ELEMENTS
            ):
                return {
                    "status": "failed",
                    "reason": "Result sample/shape/validation mismatch",
                }
            if any(
                type(a) not in (int, float)
                or not math.isfinite(a)
                or abs(a - b) > ATOL + RTOL * abs(b)
                for a, b in zip(logits, expected[sample])
            ):
                return {"status": "failed", "reason": "Independent golden mismatch"}
    processor = observed[0]
    if any(
        r.get("input_payload_verified") is not True or r.get("api_returned") is not True
        for r in processor
    ):
        return {"status": "failed", "reason": "Input payload/API evidence not verified"}
    if (
        nodes["source"]["published"] != count
        or nodes["processor"]["inference_operations"] != count
        or nodes["processor"]["published"] != count
        or nodes["consumer"]["valid_unique"] != count
    ):
        return {
            "status": "insufficient_evidence",
            "reason": "Publication/inference/consumption counts differ",
        }
    return {
        "status": "passed",
        "valid_unique": count,
        "sample_count": SAMPLE_COUNT,
        "atol": ATOL,
        "rtol": RTOL,
    }


def diagnostics(out, manifest, analysis):
    if (
        analysis["trace_integrity"]["status"] != "valid"
        or not manifest["clock_verified"]
    ):
        return {"status": "unavailable", "reason": "Trace or clock not trusted"}
    if not all(
        (out / (r + ".results.jsonl")).is_file() for r in ("processor", "consumer")
    ):
        return {"status": "unavailable", "reason": "Result evidence missing"}
    raw = [
        row
        for name in manifest["event_files"]
        if (out / name).is_file()
        for row in rows(out / name)
    ]
    points = {
        (r["source_id"], r["sequence"], r["stage"], r["event_type"]): r["timestamp_ns"]
        for r in raw
    }
    durations, prepublish = [], []
    for role in ("processor", "consumer"):
        for row in rows(out / (role + ".results.jsonl")):
            seq = row["sequence"]
            if (
                row["run_id"] != manifest["run_id"]
                or row["source_id"] != "source:1"
                or row["clock_domain"] != manifest["clock_domain"]
                or row["generated_ns"]
                != points.get(("source:1", seq, "source", "generate"))
                or row["entry_ns"]
                != points.get(("source:1", seq, role, "callback_entry"))
            ):
                raise ValueError("Business timestamps differ from authoritative events")
            if role == "processor" and row["api_returned"]:
                a, b = row["api_begin_ns"], row["api_end_ns"]
                if not row["entry_ns"] <= a <= b:
                    raise ValueError("Invalid local API interval")
                publish = points.get(("source:1", seq, "processor", "publish_begin"))
                if publish is not None:
                    if b > publish:
                        raise ValueError("API completion follows publish begin")
                    prepublish.append(publish - row["entry_ns"])
                durations.append(b - a)
    return {
        "status": "available",
        "api_label": "Host-observed synchronous inference API duration",
        "api_ns": statistics(durations),
        "processor_pre_publish_ns": statistics(prepublish),
    }


def validate_workload(count, rate):
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError("count must be 1..1000 (unchanged 4096-event buffers)")
    if (
        type(rate) not in (int, float)
        or not math.isfinite(rate)
        or not 1 <= rate <= 10000
    ):
        raise ValueError("rate must be finite and 1..10000 Hz")
    period = int(1_000_000_000 / rate)
    if (count - 1) * period > 10_000_000_000:
        raise ValueError("Planned source window exceeds ten seconds")
    return period


def validate_intra_op_threads(value):
    if type(value) is not int or value not in (1, 4):
        raise ValueError("intra-op-threads must be 1 or 4")
    return value


def verify_processor_configuration(value, requested):
    """检查实际进程完成 ORT 配置后的证据，不把请求参数当作应用证明。"""
    expected = {
        "requested_intra_op_threads": validate_intra_op_threads(requested),
        "intra_op_threads": requested,
        "inter_op_threads": 1,
        "execution_mode": "sequential",
        "graph_optimization": "basic",
        "cpu_arena": True,
        "provider": "CPUExecutionProvider",
        "onnxruntime": "1.20.1",
        "applied_via": "ORT setter and Session construction succeeded",
    }
    if not isinstance(value, dict) or any(
        type(value.get(k)) is not type(v) or value.get(k) != v
        for k, v in expected.items()
    ):
        raise ValueError("Processor applied inference configuration mismatch")
    return deepcopy(value)


def run_experiment(
    parent, count=20, rate=5, startup_only=False, intra_op_threads=1,
):
    validate_workload(count, rate)
    validate_intra_op_threads(intra_op_threads)
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    out = parent / (("vision_startup_" if startup_only else "vision_cpu_") + run_id)
    out.mkdir()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "scenario": "vision-startup-only" if startup_only else "vision-cpu",
        "startup_only": startup_only,
        "startup_readiness_completed": False,
        "launch_order": list(ROLES),
        "cleanup_order": list(ROLES),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "orchestration_sha256": digest(__file__),
        "configuration": {
            "count": count,
            "nominal_rate_hz": rate,
            "schedule": "absolute targets; catch up without dropping sequence IDs",
            "completion_timeout_s": 15,
            "depth": 10,
            "collector_capacity_per_stage": 4096,
            "executor": "SingleThreadedExecutor",
            "domain_id": 42,
            "reliability": "RELIABLE",
            "durability": "VOLATILE",
            "history": "KEEP_LAST",
            "model_sha256": MODEL_SHA,
            "model": "MobileNetV2-12 FP32",
            "model_source": "https://huggingface.co/onnxmodelzoo/mobilenetv2-12",
            "model_revision": "23a2a25eaaa6fcd61865414c8a1acf13dda66b7a",
            "provider": "CPUExecutionProvider",
            "onnxruntime": "1.20.1",
            "intra_op_threads": intra_op_threads,
            "inter_op_threads": 1,
            "execution_mode": "sequential",
            "graph_optimization": "basic",
            "cpu_arena": True,
            "warmup_count": 5,
            "sample_count": SAMPLE_COUNT,
            "input_shape": [1, 3, 224, 224],
            "output_shape": [1, 1000],
            "input_tensor_bytes": INPUT_ELEMENTS * 4,
            "output_tensor_bytes": OUTPUT_ELEMENTS * 4,
        },
        "expected_sequences": {
            "source:1": [] if startup_only else list(range(1, count + 1))
        },
        "event_files": [r + ".jsonl" for r in ROLES],
        "execution_completed": False,
        "clock_verified": False,
        "collectors": None,
        "nodes": {},
        "commands": {},
        "phases": [],
        "correctness": {"status": "insufficient_evidence"},
        "processor_configuration": None,
        "processor_readiness_configuration_verified": False,
        "processor_status_configuration_verified": False,
    }
    processes, error, last = {}, None, {}
    history = ReadinessHistory()
    with (parent / ".workload.lock").open("w") as lock, (out / "console.log").open(
        "w"
    ) as log:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest["software"] = verify_artifacts()
            if os.environ.get("ROS_LOCALHOST_ONLY") != "1":
                raise ValueError("Inference requires ROS_LOCALHOST_ONLY=1")
            _, hashes = fixture_data()
            manifest["input_tensor_sha256"] = hashes
            manifest["fixture_reference"] = read(FIXTURES / "fixtures.json")
            boot, offsets, domain = clock_evidence()
            host = json.loads(os.environ["ROBOTTRACE_HOST_CLOCK"])
            manifest.update(
                host_clock=host,
                clock_domain=domain,
                runtime_image_id=os.environ["ROBOTTRACE_IMAGE_ID"],
            )
            if host != {"boot_id": boot, "offsets": offsets}:
                raise ValueError("Host/container clock mismatch")

            def launch(role):
                command = [
                    str(PROGRAM),
                    role,
                    run_id,
                    str(out),
                    str(count),
                    str(FIXTURES),
                    str(MODEL),
                    str(rate),
                ]
                if role == "processor":
                    command += ["--intra-op-threads", str(intra_op_threads)]
                manifest["commands"][role] = command
                processes[role] = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                manifest["phases"].append(
                    {
                        "phase": "started",
                        "role": role,
                        "timestamp_ns": time.monotonic_ns(),
                    }
                )

                def initialized():
                    endpoint_ready(out, ROLES, last, history, "initializing_" + role)
                    marker = out / (role + ".started")
                    if not marker.is_file():
                        return False
                    value = read(marker)
                    if (
                        value.get("initialized") is not True
                        or value.get("process_id") != processes[role].pid
                        or value.get("run_id") != run_id
                    ):
                        raise ValueError("Wrong initialized process/run identity: " + role)
                    if role == "processor":
                        manifest["processor_configuration"] = verify_processor_configuration(
                            value.get("inference_configuration"), intra_op_threads
                        )
                    return True

                wait_until(
                    initialized,
                    processes,
                    25,
                    role + " initialization",
                )

            for role in ROLES:
                launch(role)

            def matched_endpoints():
                states = endpoint_ready(out, ROLES, last, history)
                processor = last.get("processor")
                if isinstance(processor, dict) and processor.get("initialized") is True:
                    applied = verify_processor_configuration(
                        processor.get("inference_configuration"), intra_op_threads
                    )
                    if applied != manifest["processor_configuration"]:
                        raise ValueError("Processor configuration changed during startup")
                    manifest["processor_readiness_configuration_verified"] = True
                return states

            manifest["readiness"] = wait_until(
                matched_endpoints,
                processes,
                15,
                "all four matched endpoints",
            )
            manifest["phases"].append(
                {"phase": "endpoints_ready", "timestamp_ns": time.monotonic_ns()}
            )
            manifest["startup_readiness_completed"] = True
            if not startup_only:
                manifest["phases"].append(
                    {"phase": "release", "timestamp_ns": time.monotonic_ns()}
                )
                (out / "release").touch()
                wait_until(
                    lambda: (out / "published.json").is_file()
                    and (out / "consumed.json").is_file(),
                    processes,
                    15,
                    "all source publications and unique Consumer results",
                )
                if (
                    read(out / "published.json").get("published") != count
                    or read(out / "consumed.json").get("valid_unique") != count
                ):
                    raise ValueError("Completion acknowledgement mismatch")
                manifest["phases"].append(
                    {"phase": "consumer_complete", "timestamp_ns": time.monotonic_ns()}
                )
        except BaseException as exc:
            error = exc
            manifest["execution_error"] = repr(exc)
        finally:
            manifest["measurement_stop_ns"] = time.monotonic_ns()
            manifest["pre_cleanup_readiness"] = deepcopy(last)
            try:
                history.freeze(
                    "success"
                    if error is None
                    else "timeout" if isinstance(error, TimeoutError) else "failure"
                )
            except (OSError, ValueError, TypeError) as exc:
                history.error(exc)
            finally:
                cleanup = _cleanup_children({
                    role: processes[role] for role in ROLES if role in processes
                })
            manifest["readiness_diagnostics"] = history.data
            if cleanup:
                manifest["cleanup_errors"] = cleanup
            manifest["exit_codes"] = {r: p.returncode for r, p in processes.items()}
            collectors, status_errors = [], []
            for role in ROLES:
                try:
                    node = read(out / (role + ".status.json"))
                    if node["stage"] != role:
                        raise ValueError("Wrong stage status")
                    if (
                        role not in processes
                        or node["process_id"] != processes[role].pid
                    ):
                        raise ValueError("Wrong process status")
                    if role == "processor":
                        applied = verify_processor_configuration(
                            node.get("inference_configuration"), intra_op_threads
                        )
                        if applied != manifest["processor_configuration"]:
                            raise ValueError("Processor configuration changed before shutdown")
                        manifest["processor_status_configuration_verified"] = True
                    collector = {
                        k: node[k]
                        for k in ("stage", "stored", "dropped", "flushed", "process_id")
                    }
                    CollectorEvidence(**collector)
                    manifest["nodes"][role] = node
                    collectors.append(collector)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    status_errors.append({"role": role, "error": repr(exc)})
            if len(collectors) == 3:
                manifest["collectors"] = collectors
            if status_errors:
                manifest["status_errors"] = status_errors
            host = manifest.get("host_clock")
            manifest["clock_verified"] = (
                host is not None
                and len(manifest["nodes"]) == 3
                and all(
                    n.get("boot_id") == host["boot_id"]
                    and n.get("time_offsets") == host["offsets"]
                    and n.get("clock_domain") == manifest.get("clock_domain")
                    for n in manifest["nodes"].values()
                )
            )
            try:
                if startup_only:
                    manifest["correctness"] = {
                        "status": "not_exercised",
                        "reason": "startup-only; Source not released",
                    }
                    if (out / "release").exists() or any(
                        n.get("published") or n.get("received")
                        for n in manifest["nodes"].values()
                    ):
                        raise ValueError(
                            "Unexpected workload activity in startup-only attempt"
                        )
                else:
                    manifest["correctness"] = correctness(out, manifest, count)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                manifest["correctness"] = {
                    "status": "insufficient_evidence",
                    "error": repr(exc),
                }
            if error is None and (
                cleanup
                or status_errors
                or not manifest["clock_verified"]
                or manifest["correctness"]["status"]
                != ("not_exercised" if startup_only else "passed")
                or len(processes) != 3
                or any(p.returncode != 0 for p in processes.values())
                or any(
                    n.get("error") or n.get("diagnostic_dropped")
                    for n in manifest["nodes"].values()
                )
            ):
                error = RuntimeError("Execution/correctness/clock finalization failed")
                manifest["execution_error"] = repr(error)
            manifest["execution_completed"] = error is None
            manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                write(out / "run.json", manifest)
                analysis = load_and_analyze(out)
                manifest["inference_diagnostics"] = diagnostics(out, manifest, analysis)
                if error is None and any(
                    analysis[axis]["status"] != expected
                    for axis, expected in (
                        ("collection", "complete"),
                        ("trace_integrity", "valid"),
                        ("observations", "incomplete" if startup_only else "complete"),
                    )
                ):
                    error = RuntimeError(
                        "Trace/collection/observation acceptance failed"
                    )
                if error is not None:
                    manifest["execution_completed"] = False
                    manifest["execution_error"] = repr(error)
                write(out / "run.json", manifest)
                analysis = load_and_analyze(out)
                save_report(out, analysis)
            except (OSError, ValueError, TypeError, KeyError) as finalization:
                manifest["execution_completed"] = False
                manifest["execution_error"] = repr(
                    error if error is not None else finalization
                )
                manifest["finalization_error"] = repr(finalization)
                try:
                    write(out / "run.json", manifest)
                except OSError as persistence:
                    raise RuntimeError(
                        f"{out}: {error or finalization}; manifest persistence failed: {persistence}"
                    ) from finalization
                if error is not None:
                    raise RuntimeError(
                        f"{out}: {error}; evidence finalization failed: {finalization}"
                    ) from error
                raise
    if error is not None:
        raise RuntimeError(f"{out}: {error}") from error
    return out, analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--rate", type=float, default=5)
    parser.add_argument("--intra-op-threads", type=int, choices=(1, 4), default=1)
    parser.add_argument(
        "--startup-only",
        action="store_true",
        help="Check startup without releasing Source; inference correctness is not exercised",
    )
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--record-build", action="store_true")
    args = parser.parse_args()
    if args.record_build:
        record_build()
        return 0
    try:
        validate_workload(args.count, args.rate)
    except ValueError as error:
        parser.error(str(error))
    if not args.inside:
        boot, offsets, _ = clock_evidence()
        image = subprocess.check_output(
            [str(ROOT / "environment/image.sh")], text=True
        ).strip()
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            text=True,
            timeout=10,
        ).strip()
        return subprocess.call(
            [
                str(ROOT / "scripts/ros.sh"),
                "env",
                "ROS_LOCALHOST_ONLY=1",
                "LD_LIBRARY_PATH=/work/bin/inference/lib:/work/.deps/onnxruntime-linux-x64-1.20.1/lib:/opt/ros/humble/lib",
                "ROBOTTRACE_HOST_CLOCK="
                + json.dumps({"boot_id": boot, "offsets": offsets}),
                "ROBOTTRACE_IMAGE_ID=" + image_id,
                "timeout",
                "--signal=INT",
                "--kill-after=20s",
                "100s",
                "python3",
                "-B",
                "-m",
                "robottrace.inference",
                "--inside",
                "--count",
                str(args.count),
                "--rate",
                str(args.rate),
                "--intra-op-threads",
                str(args.intra_op_threads),
                *(["--startup-only"] if args.startup_only else []),
            ]
        )
    path, analysis = run_experiment(
        ROOT / "evidence", args.count, args.rate, args.startup_only, args.intra_op_threads,
    )
    from .cli import display

    display(analysis)
    manifest = read(path / "run.json")
    if args.startup_only:
        print("startup_readiness=passed; Source not released; workload not exercised")
    print("inference_correctness=" + manifest["correctness"]["status"])
    print("diagnostics:", json.dumps(manifest["inference_diagnostics"]))
    print("Evidence:", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
