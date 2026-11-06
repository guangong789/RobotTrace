"""仅用于固定 Talker→Relay→Listener 接入的有界实验，不扩展原示例运行器。"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from .events import CollectorEvidence
from .io import load_and_analyze, save_report, unique_fields
from .workload import _cleanup_children, clock_evidence, write

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "ros2/robottrace_upstream"
BIN = ROOT / "bin/upstream"
ROLES = {"source": "talker", "processor": "relay", "consumer": "listener"}
READINESS_COUNTS = {"source": (0, 1), "processor": (1, 1), "consumer": (1, 0)}
READINESS_HISTORY_LIMIT = 32
PACKAGES = (
    "ros-humble-rclcpp",
    "ros-humble-rclcpp-components",
    "ros-humble-std-msgs",
    "ros-humble-rmw-fastrtps-cpp",
    "ros-humble-rmw-fastrtps-shared-cpp",
    "ros-humble-fastrtps",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_fields)


def record_build():
    """仅记录本次构建输入和其真实运行库，不扫描任何历史证据。"""
    upstream = read(APP / "upstream.json")
    for repository in upstream.values():
        for name, identity in repository["files"].items():
            if digest(APP / name) != identity["sha256"]:
                raise ValueError("Pristine upstream source changed: " + name)
    executables, libraries, dependencies = {}, {}, {}
    targets = {
        **ROLES,
        "demo_component": "librobottrace_demo_topics.so",
        "relay_component": "librobottrace_relay.so",
    }
    for role, name in targets.items():
        path = BIN / name
        if role in ROLES:
            executables[role] = {"path": str(path), "sha256": digest(path)}
        else:
            libraries[str(path)] = digest(path)
        output = subprocess.check_output(["ldd", str(path)], text=True, timeout=5)
        if "not found" in output:
            raise RuntimeError(output)
        dependencies[role] = output
        for library in re.findall(r"=> (/\S+)", output):
            resolved = str(Path(library).resolve())
            libraries[resolved] = digest(resolved)
    # RMW/Fast DDS 由插件动态加载，不能只依赖可执行文件的 ldd 输出。
    for name in (
        "librmw_fastrtps_cpp.so",
        "librmw_fastrtps_shared_cpp.so",
        "libfastrtps.so",
    ):
        path = (Path("/opt/ros/humble/lib") / name).resolve(strict=True)
        libraries[str(path)] = digest(path)
    inputs = [p for p in APP.rglob("*") if p.is_file()]
    inputs += [
        ROOT / "ros2/robottrace_examples/include/robottrace/events.hpp",
        ROOT / "scripts/build-upstream.sh",
        ROOT / "tests/upstream_identity.cpp",
    ]
    value = {
        "upstream": upstream,
        "executables": executables,
        "libraries": libraries,
        "dependencies_ldd": dependencies,
        "build_inputs": {str(p.relative_to(ROOT)): digest(p) for p in sorted(inputs)},
        "versions": subprocess.check_output(
            ["dpkg-query", "-W", *PACKAGES], text=True, timeout=5
        ),
        "compiler": subprocess.check_output(["g++", "--version"], text=True, timeout=5),
        "build_type": "Release",
        "cxx_standard": 17,
    }
    write(BIN / "build.json", value)


def software_identity():
    value = read(BIN / "build.json")
    for item in value["executables"].values():
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Instrumented executable changed since build")
    for path, expected in value["libraries"].items():
        if digest(path) != expected:
            raise ValueError("Runtime library changed since build: " + path)
    for path, expected in value["build_inputs"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(
                "Build input changed; rebuild upstream application: " + path
            )
    return value


def wait_until(predicate, processes, seconds, description, *, diagnostics=None):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        failed = {
            role: p.returncode for role, p in processes.items() if p.poll() is not None
        }
        if failed:
            if diagnostics is not None:
                diagnostics.boundary("process_exit")
            raise RuntimeError("Unexpected process exit: " + repr(failed))
        value = predicate()
        if value:
            if diagnostics is not None:
                diagnostics.boundary("success")
            return value
        time.sleep(0.02)
    if diagnostics is not None:
        diagnostics.boundary("timeout")
    raise TimeoutError("Deadline waiting for " + description)


def optional_json(path):
    return read(path) if path.is_file() else None


def _ready_values(values):
    """保持原有匹配顺序、精确计数和 QoS 检查，不以诊断结果决定放行。"""
    if any(value is None for value in values.values()):
        return None
    for role, value in values.items():
        if (
            value["matched_publishers"],
            value["matched_subscribers"],
        ) != READINESS_COUNTS[role]:
            return None
        for side in ("input_qos", "output_qos"):
            qos = value[side]
            if qos is not None and (
                qos["history"],
                qos["reliability"],
                qos["durability"],
            ) != (1, 1, 2):
                raise ValueError("Unsupported actual endpoint QoS: " + repr(qos))
    return values


def _diagnostic_value(value):
    # 只保留状态契约字段；异常值和错误文本限长，不复制任意扩展数据。
    if value is None or type(value) in (bool, int):
        return (
            value
            if type(value) is not int or value.bit_length() <= 64
            else repr(value)[:160]
        )
    return repr(value)[:160]


def _role_readiness(present=None, value=None, error=None):
    state = {
        "file_present": present,
        "state": "unknown",
        "input_exists": None,
        "output_exists": None,
        "matched_publishers": None,
        "matched_subscribers": None,
    }
    if error is not None:
        state.update(
            state="unreadable" if isinstance(error, OSError) else "malformed",
            error=repr(error)[:256],
        )
    elif present is False:
        state["state"] = "missing"
    elif present is True:
        state["state"] = "malformed"
        if isinstance(value, dict):
            complete = all(
                key in value
                for key in (
                    "matched_publishers",
                    "matched_subscribers",
                    "input_qos",
                    "output_qos",
                )
            )
            for key in ("matched_publishers", "matched_subscribers"):
                state[key] = _diagnostic_value(value.get(key))
                complete &= type(value.get(key)) is int and value[key] >= 0
            for side in ("input", "output"):
                key = side + "_qos"
                qos = value.get(key)
                if key in value and (qos is None or isinstance(qos, dict)):
                    state[side + "_exists"] = qos is not None
                    state[key] = (
                        None
                        if qos is None
                        else {
                            k: _diagnostic_value(qos.get(k))
                            for k in ("history", "depth", "reliability", "durability")
                        }
                    )
                    complete &= qos is None or all(
                        type(qos.get(k)) is int
                        for k in ("history", "depth", "reliability", "durability")
                    )
                else:
                    complete = False
            if complete:
                state["state"] = "reported"
    return state


def _unsatisfied_readiness(states):
    conditions = []
    for role, state in states.items():
        if state["state"] != "reported":
            conditions.append(
                {"role": role, "condition": "status_report", "observed": state["state"]}
            )
        for key, expected in zip(
            ("matched_publishers", "matched_subscribers"), READINESS_COUNTS[role]
        ):
            actual = state[key]
            if actual is None or actual != expected:
                conditions.append(
                    {
                        "role": role,
                        "condition": key,
                        "expected": expected,
                        "observed": actual,
                    }
                )
        for side in ("input", "output"):
            qos = state.get(side + "_qos")
            if isinstance(qos, dict):
                for key, expected in (
                    ("history", 1),
                    ("reliability", 1),
                    ("durability", 2),
                ):
                    if qos[key] != expected:
                        conditions.append(
                            {
                                "role": role,
                                "condition": side + "_qos." + key,
                                "expected": expected,
                                "observed": qos[key],
                            }
                        )
    return conditions


class ReadinessHistory:
    """保留首次及最近状态变化；只观察现有文件轮询，不增加发现探针。"""

    def __init__(self):
        self.data = {
            "max_entries": READINESS_HISTORY_LIMIT,
            "poll_count": 0,
            "dropped_entries": 0,
            "truncated": False,
            "entries": [],
            "diagnostic_error_count": 0,
            "diagnostic_errors": [],
            "frozen_before_cleanup": False,
        }
        self.previous = None

    def _guard(self, operation, action):
        try:
            action()
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
        ) as exc:
            self.data["diagnostic_error_count"] += 1
            if len(self.data["diagnostic_errors"]) < 4:
                self.data["diagnostic_errors"].append(
                    {"operation": operation, "error": repr(exc)[:256]}
                )

    def _append(self, entry):
        entries = self.data["entries"]
        if len(entries) == READINESS_HISTORY_LIMIT:
            # 固定保留首条，淘汰最早的中间条目，末尾始终容纳终止边界。
            del entries[1]
            self.data["dropped_entries"] += 1
            self.data["truncated"] = True
        entries.append(entry)

    def observe(self, states, result, error=None):
        def record():
            if self.data["frozen_before_cleanup"]:
                return
            self.data["poll_count"] += 1
            state = {
                "roles": deepcopy(states),
                "predicate_result": result,
                "unsatisfied": _unsatisfied_readiness(states),
            }
            if error is not None:
                state["predicate_error"] = repr(error)[:256]
            observation = {
                "event": "poll",
                "observed_ns": time.monotonic_ns(),
                "poll_index": self.data["poll_count"],
                **state,
            }
            self.data["last_observation"] = observation
            if state != self.previous:
                self._append(observation)
                self.previous = state

        self._guard("observe", record)

    def boundary(self, outcome):
        def record():
            if not self.data["frozen_before_cleanup"]:
                # 边界不再读文件；观察时间保留原值，另记边界时间。
                self._append(
                    {
                        **deepcopy(self.data.get("last_observation", {})),
                        "event": outcome,
                        "boundary_ns": time.monotonic_ns(),
                        "state_reobserved": False,
                    }
                )

        self._guard("boundary", record)

    def freeze(self, states=None):
        def record():
            if self.data["frozen_before_cleanup"]:
                return
            if states is not None:
                self.data["pre_cleanup"] = {
                    "observed_ns": time.monotonic_ns(),
                    "roles": deepcopy(states),
                    "unsatisfied": _unsatisfied_readiness(states),
                }
            self.data["frozen_before_cleanup"] = True

        self._guard("freeze", record)
        self.data["frozen_before_cleanup"] = True


def ready_snapshot(out, diagnostics=None):
    values = {}
    states = {role: _role_readiness() for role in ROLES}
    try:
        for role in ROLES:
            present = None
            try:
                path = out / (role + ".ready.json")
                present = path.is_file()
                value = read(path) if present else None
            except (OSError, ValueError, TypeError) as exc:
                states[role] = _role_readiness(present, error=exc)
                raise
            values[role] = value
            states[role] = _role_readiness(present, value)
        result = _ready_values(values)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        if diagnostics is not None:
            diagnostics.observe(states, "error", exc)
        raise
    if diagnostics is not None:
        diagnostics.observe(states, "ready" if result else "waiting")
    return result


def loaded_libraries(processes):
    """记录三个真实进程已加载的 ROS/应用库，补足 dlopen 类型支持插件。"""
    result, hashes = {}, {}
    for role, process in processes.items():
        paths = set()
        for line in Path(f"/proc/{process.pid}/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6:
                path = fields[5]
                if ".so" in path and path.startswith(
                    ("/opt/ros/humble/lib/", str(BIN) + "/")
                ):
                    paths.add(path)
        if not paths:
            raise ValueError("No loaded application/ROS libraries for " + role)
        for path in sorted(paths):
            if path not in hashes:
                hashes[path] = digest(path)
        result[role] = {path: hashes[path] for path in sorted(paths)}
    return result


def collect_status(out, processes, domain, boot, offsets):
    nodes, collectors, errors = {}, [], []
    for role, process in processes.items():
        node = {"process_id": process.pid, "status_available": False}
        try:
            if process.returncode is None:
                raise ValueError("Process exit unconfirmed")
            candidate = read(out / (role + ".status.json"))
            if not isinstance(candidate, dict):
                raise ValueError("Process status must be an object")
            collector = {
                key: candidate[key]
                for key in ("stage", "stored", "dropped", "flushed", "process_id")
            }
            CollectorEvidence(**collector)
            if candidate["stage"] != role or candidate["process_id"] != process.pid:
                raise ValueError("Process identity mismatch")
            node = candidate
            collectors.append(collector)
            if (
                type(candidate["invalid_messages"]) is not int
                or candidate["invalid_messages"] != 0
            ):
                raise ValueError("Rejected invalid/duplicate application identity")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({"stage": role, "error": repr(exc)})
        node["exit_status"] = process.returncode
        nodes[role] = node
    verified = (
        domain is not None
        and set(nodes) == set(ROLES)
        and all(
            node.get("clock_domain") == domain
            and node.get("boot_id") == boot
            and node.get("time_offsets") == offsets
            for node in nodes.values()
        )
    )
    complete = (
        not errors
        and len(collectors) == 3
        and verified
        and all(c["flushed"] for c in collectors)
        and all(p.returncode == 0 for p in processes.values())
    )
    return (
        nodes,
        collectors if len(collectors) == 3 else None,
        errors,
        verified,
        complete,
    )


def run_experiment(parent, scenario):
    if scenario not in ("normal", "late-relay"):
        raise ValueError("Unsupported upstream scenario")
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    out = parent / ("upstream_" + scenario + "_" + run_id)
    out.mkdir()
    prefix = 3 if scenario == "late-relay" else 0
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "scenario": "upstream-" + scenario,
        "orchestration_sha256": digest(__file__),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "application": "demo_nodes_cpp→topic_tools relay→demo_nodes_cpp",
            "count": 20,
            "nominal_rate_hz": 1,
            "prefix_count": prefix,
            "collector_capacity_per_stage": 4096,
            "executor": "SingleThreadedExecutor",
            "domain_id": 42,
        },
        "identity_binding": {
            "source_id": "demo:talker",
            "payload": "Hello World: <sequence>",
            "run_uuid_in_payload": False,
            "run_binding": "fresh isolated topics; one declared source; shared run configuration",
        },
        "expected_sequences": {"demo:talker": list(range(1, 21))},
        "event_files": [role + ".jsonl" for role in ROLES],
        "execution_completed": False,
        "clock_verified": False,
        "collectors": None,
        "nodes": {},
        "commands": {},
        "phases": [],
        "startup_observers": [],
    }
    processes, error, log = {}, None, None
    diagnostics = ReadinessHistory()
    manifest["readiness_diagnostics"] = diagnostics.data
    boot = offsets = domain = None
    with (parent / ".workload.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            boot, offsets, domain = clock_evidence()
            host = json.loads(os.environ["ROBOTTRACE_HOST_CLOCK"])
            manifest.update(host_clock=host, clock_domain=domain)
            if host != {"boot_id": boot, "offsets": offsets}:
                raise ValueError("Host/container clock evidence mismatch")
            manifest["software"] = software_identity()
            manifest["runtime_image_id"] = os.environ["ROBOTTRACE_IMAGE_ID"]
            topic = "/robottrace/r_" + run_id
            manifest["topics"] = {
                "input": topic + "/input",
                "output": topic + "/output",
            }
            env = {
                **os.environ,
                "RT_RUN": run_id,
                "RT_OUTPUT": str(out),
                "RT_COUNT": "20",
                "RT_PREFIX": str(prefix),
                "RT_CAPACITY": "4096",
                "LD_LIBRARY_PATH": str(BIN)
                + ":"
                + os.environ.get("LD_LIBRARY_PATH", ""),
            }
            log = (out / "console.log").open("w")

            def phase(name, **details):
                manifest["phases"].append(
                    {"phase": name, "timestamp_ns": time.monotonic_ns(), **details}
                )

            def start(role):
                args = [str(BIN / ROLES[role])]
                if role == "processor":
                    args += [topic + "/input", topic + "/output"]
                args += ["--ros-args", "-r", "__ns:=" + topic]
                if role != "processor":
                    args += [
                        "-r",
                        "chatter:=" + ("input" if role == "source" else "output"),
                    ]
                manifest["commands"][role] = args
                processes[role] = subprocess.Popen(
                    args,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                info = wait_until(
                    lambda: optional_json(out / (role + ".started")),
                    processes,
                    8,
                    role + " startup",
                )
                if info["process_id"] != processes[role].pid:
                    raise ValueError("Startup process identity mismatch")
                phase("started", stage=role, process_id=info["process_id"])

            start("consumer")
            start("source")
            if prefix:
                source = wait_until(
                    lambda: optional_json(out / "source.ready.json"),
                    processes,
                    5,
                    "source unmatched endpoint",
                )
                if source["matched_subscribers"] != 0:
                    raise ValueError("Prefix requires no Relay subscription")
                phase("prefix_enabled", relay_started=False)
                (out / "prefix.enable").touch()
                progress = wait_until(
                    lambda: optional_json(out / "prefix.json"),
                    processes,
                    8,
                    "prefix completion",
                )
                if (
                    progress["published"] != prefix
                    or progress["matched_subscribers"] != 0
                ):
                    raise ValueError("Invalid prefix phase evidence")
                phase("prefix_completed", **progress)
            start("processor")
            manifest["readiness"] = wait_until(
                lambda: ready_snapshot(out, diagnostics),
                processes,
                15,
                "both DDS links matched",
                diagnostics=diagnostics,
            )
            manifest["loaded_libraries"] = loaded_libraries(processes)
            for name in ("input", "output"):
                args = [
                    "ros2",
                    "topic",
                    "info",
                    topic + "/" + name,
                    "--verbose",
                    "--no-daemon",
                    "--spin-time",
                    "1",
                ]
                observed = subprocess.run(
                    args, capture_output=True, text=True, timeout=6
                )
                manifest["startup_observers"].append(
                    {
                        "command": args,
                        "exit_status": observed.returncode,
                        "stdout": observed.stdout,
                        "stderr": observed.stderr,
                    }
                )
                if observed.returncode:
                    raise RuntimeError("Endpoint graph inspection failed")
            phase("ready")
            (out / "resume").touch()
            phase("resumed")
            manifest["publication"] = wait_until(
                lambda: optional_json(out / "published.json"),
                processes,
                26,
                "20 source publications",
            )
            if manifest["publication"]["published"] != 20:
                raise ValueError("Source publication count mismatch")
            manifest["delivery"] = wait_until(
                lambda: optional_json(out / "delivered.json"),
                processes,
                6,
                "last consumer callback",
            )
            phase("delivery_complete", **manifest["delivery"])
            manifest["execution_completed"] = True
        except BaseException as exc:
            error = exc
            manifest["execution_error"] = repr(exc)
            diagnostics.boundary("failure")
            # 关闭节点会改变匹配数；先保留失败发生时的端点快照。
            manifest["failure_readiness"] = {}
            failure_states = {role: _role_readiness() for role in ROLES}
            for role in processes:
                present = None
                try:
                    path = out / (role + ".ready.json")
                    present = path.is_file()
                    value = read(path) if present else None
                    manifest["failure_readiness"][role] = value
                    failure_states[role] = _role_readiness(present, value)
                except (OSError, ValueError, TypeError) as diagnostic:
                    manifest["failure_readiness"][role] = {"error": repr(diagnostic)}
                    failure_states[role] = _role_readiness(present, error=diagnostic)
            diagnostics.freeze(failure_states)
        finally:
            diagnostics.freeze()
            cleanup = _cleanup_children(processes)
            nodes, collectors, errors, verified, complete = collect_status(
                out, processes, domain, boot, offsets
            )
            manifest.update(
                nodes=nodes,
                collectors=collectors,
                clock_verified=verified,
                ended_utc=datetime.now(timezone.utc).isoformat(),
            )
            if cleanup:
                manifest["cleanup_errors"] = cleanup
            if errors:
                manifest["status_errors"] = errors
            if log is not None:
                try:
                    log.close()
                except OSError as exc:
                    errors.append({"operation": "close_console", "error": repr(exc)})
                    manifest["status_errors"] = errors
            manifest["execution_completed"] = bool(
                manifest["execution_completed"]
                and complete
                and not cleanup
                and not errors
                and error is None
            )
            if not manifest["execution_completed"] and error is None:
                error = RuntimeError("Upstream finalization incomplete")
                manifest["execution_error"] = repr(error)
            try:
                write(out / "run.json", manifest)
            except (OSError, ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"{out}: failed to save run.json: {exc!r}; primary: {error!r}"
                ) from (error or exc)
        try:
            result = load_and_analyze(out)
            save_report(out, result)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            if error is not None:
                raise RuntimeError(
                    f"{out}: {error}; analysis/report failure: {exc!r}"
                ) from error
            raise
        if error is not None:
            raise RuntimeError(f"{out}: {error}") from error
        return out, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("normal", "late-relay"), nargs="?")
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--record-build", action="store_true")
    args = parser.parse_args()
    if args.record_build:
        record_build()
        return 0
    if args.scenario is None:
        parser.error("scenario required")
    if not args.inside:
        boot, offsets, _ = clock_evidence()
        return subprocess.call(
            [
                str(ROOT / "scripts/ros.sh"),
                "env",
                "ROBOTTRACE_HOST_CLOCK="
                + json.dumps({"boot_id": boot, "offsets": offsets}),
                "ROBOTTRACE_IMAGE_ID="
                + (ROOT / "environment/image-id.txt").read_text().strip(),
                "timeout",
                "--signal=INT",
                "--kill-after=5s",
                "100s",
                sys.executable,
                "-B",
                "-m",
                "robottrace.upstream",
                args.scenario,
                "--inside",
            ]
        )
    path, result = run_experiment(ROOT / "evidence", args.scenario)
    from .cli import display

    display(result)
    print("Evidence:", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
