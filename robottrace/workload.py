"""独立于核心的示例运行器：复用一个 ROS2 镜像，启动三个进程。"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from .io import load_and_analyze, save_report
from .events import CollectorEvidence

ROOT = Path(__file__).resolve().parent.parent
ROLES = ("consumer", "processor", "source")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def wait_file(path, processes, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size:
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        if any(p.poll() is not None for p in processes.values()):
            raise RuntimeError("Node exited before " + path.name)
        time.sleep(0.02)
    raise TimeoutError("Deadline waiting for " + path.name)


def clock_evidence():
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    offsets = Path("/proc/self/timens_offsets").read_text()
    if any(int(x) != 0 for row in offsets.splitlines() for x in row.split()[1:]):
        raise RuntimeError("Unsupported time namespace offsets")
    return boot, offsets, "linux:CLOCK_MONOTONIC:" + boot + ":offset=0"


def _cleanup_children(processes):
    """分别尝试清理每个子进程；单次失败不阻断其他进程，所有等待均有时限。"""
    failures = []
    for stage, process in processes.items():
        try:
            if process.poll() is not None:
                continue
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append({"stage": stage, "operation": "poll", "error": repr(exc)})
        try:
            process.send_signal(signal.SIGINT)
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append({"stage": stage, "operation": "sigint", "error": repr(exc)})
        try:
            process.wait(timeout=4)
            continue
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(
                {"stage": stage, "operation": "wait_after_sigint", "error": repr(exc)}
            )
        try:
            process.kill()
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append({"stage": stage, "operation": "kill", "error": repr(exc)})
        try:
            process.wait(timeout=2)
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(
                {"stage": stage, "operation": "wait_after_kill", "error": repr(exc)}
            )
    return failures


def run_pipeline(
    parent, scenario, *, count=100, rate=100, depth=10, work_ms=None, capacity=4096
):
    if work_ms is None:
        work_ms = 1 if scenario == "normal" else 50
    if not (
        1 <= count <= 10000
        and 0 < rate <= 1000
        and 1 <= depth <= 10000
        and 0 <= work_ms <= 1000
        and 0 <= capacity <= 1000000
    ):
        raise ValueError("Invalid workload bounds")
    # 确保配置的消息生产时间不超出示例明确规定的 20 秒预算。
    if (count - 1) / rate > 15:
        raise ValueError("Configured publication duration exceeds the example budget")
    import fcntl

    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    out = parent / (scenario + "_" + run_id)
    out.mkdir()
    with (ROOT / "evidence/.workload.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        boot, offsets, domain = clock_evidence()
        host = json.loads(os.environ.get("ROBOTTRACE_HOST_CLOCK", "null"))
        if host is None or host["boot_id"] != boot or host["offsets"] != offsets:
            raise RuntimeError("Host/container clock-domain evidence does not match")
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "scenario": scenario,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "count": count,
                "nominal_rate_hz": rate,
                "depth": depth,
                "processor_work_ms": work_ms,
                "drain_s": 6,
                "collector_capacity_per_stage": capacity,
                "reliability": "RELIABLE",
                "durability": "VOLATILE",
                "history": "KEEP_LAST",
                "executor": "SingleThreadedExecutor",
                "domain_id": 42,
                "layout": "three independent C++ processes in one ephemeral ROS2 container",
            },
            "clock_domain": domain,
            "clock_verified": False,
            "host_clock": host,
            "expected_sequences": {"source:1": list(range(1, count + 1))},
            "execution_completed": False,
            "collectors": [],
            "nodes": {},
            "commands": {},
            "startup_observers": [],
            "event_files": [
                role + ".jsonl" for role in ("source", "processor", "consumer")
            ],
            "thresholds_ns": {
                "processor_callback_ns": 10_000_000,
                "consumer_age_ns": 50_000_000,
            },
        }
        processes = {}
        log = (out / "console.log").open("w")
        error = None
        try:
            for role in ROLES:
                args = [
                    str(ROOT / "bin/pipeline"),
                    role,
                    run_id,
                    str(out),
                    str(count),
                    str(rate),
                    str(depth),
                    str(work_ms),
                    str(capacity),
                    "1",
                ]
                manifest["commands"][role] = args
                processes[role] = subprocess.Popen(
                    args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
                )
                # 创建标记不代表 DDS 已就绪。
                started = out / (role + ".started")
                deadline = time.monotonic() + 8
                while not started.exists():
                    if (
                        time.monotonic() > deadline
                        or processes[role].poll() is not None
                    ):
                        raise RuntimeError(role + " startup failed")
                    time.sleep(0.02)
            for attempt in range(4):
                for topic in ("input", "output"):
                    args = [
                        "ros2",
                        "topic",
                        "info",
                        "/robottrace/r_" + run_id + "/" + topic,
                        "--verbose",
                        "--no-daemon",
                        "--spin-time",
                        "1",
                    ]
                    r = subprocess.run(args, capture_output=True, text=True, timeout=6)
                    manifest["startup_observers"].append(
                        {
                            "command": args,
                            "exit_status": r.returncode,
                            "stdout": r.stdout,
                            "stderr": r.stderr,
                        }
                    )
                if (out / "matched.json").exists() or processes[
                    "source"
                ].poll() is not None:
                    break
            published = wait_file(out / "published.json", processes, 45)
            matched = wait_file(out / "matched.json", processes, 1)
            if published["published"] != count:
                raise RuntimeError("Publisher did not complete configured count")
            manifest["readiness"] = matched
            manifest["publication_complete_ns"] = published["completed_ns"]
            deadline = published["completed_ns"] + 6_000_000_000
            manifest["drain_deadline_ns"] = deadline
            while time.monotonic_ns() < deadline:
                if any(p.poll() is not None for p in processes.values()):
                    raise RuntimeError("Node exited before drain deadline")
                time.sleep(min(0.02, max(0, (deadline - time.monotonic_ns()) / 1e9)))
            manifest["stop_signal_ns"] = time.monotonic_ns()
            for p in processes.values():
                p.send_signal(signal.SIGINT)
            for p in processes.values():
                p.wait(timeout=5)
            manifest["execution_completed"] = all(
                p.returncode == 0 for p in processes.values()
            )
        except BaseException as exc:
            error = exc
            manifest["execution_error"] = repr(exc)
        finally:
            cleanup_errors = _cleanup_children(processes)
            status_errors = []
            metadata_errors = []
            # 先完成全部子进程的清理尝试，再独立读取各自状态。
            for role, p in processes.items():
                status_file = out / (role + ".status.json")
                try:
                    if p.returncode is None:
                        raise ValueError("Process exit not confirmed")
                    node = json.loads(status_file.read_text())
                    if not isinstance(node, dict):
                        raise ValueError("Node status must be an object")
                    json.dumps(node, allow_nan=False)
                    fields = ("stage", "process_id", "stored", "dropped", "flushed")
                    collector = {key: node[key] for key in fields}
                    CollectorEvidence(**collector)
                    if node["stage"] != role or node["process_id"] != p.pid:
                        raise ValueError("Node status identity mismatch")
                    manifest["collectors"].append(collector)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    status_errors.append(
                        {"stage": role, "operation": "read_status", "error": repr(exc)}
                    )
                    # 无状态不等于计数为零；只保留确实已知的进程身份。
                    node = {
                        "stage": role,
                        "process_id": p.pid,
                        "status_available": False,
                    }
                node["exit_status"] = p.returncode
                manifest["nodes"][role] = node
            if len(manifest["collectors"]) != len(ROLES):
                manifest["collectors"] = None
            try:
                log.close()
            except OSError as exc:
                metadata_errors.append(
                    {"operation": "close_console", "error": repr(exc)}
                )
            manifest["clock_verified"] = len(manifest["nodes"]) == 3 and all(
                n.get("clock_domain") == domain
                and n.get("boot_id") == boot
                and n.get("time_offsets") == offsets
                for n in manifest["nodes"].values()
            )
            manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                manifest["versions"] = subprocess.check_output(
                    [
                        "dpkg-query",
                        "-W",
                        "-f=${binary:Package}\t${Version}\n",
                        "ros-humble-rclcpp",
                        "ros-humble-rmw-fastrtps-cpp",
                        "ros-humble-std-msgs",
                        "ros-humble-std-srvs",
                    ],
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                metadata_errors.append(
                    {"operation": "read_versions", "error": repr(exc)}
                )
            try:
                manifest["binary_sha256"] = hashlib.sha256(
                    (ROOT / "bin/pipeline").read_bytes()
                ).hexdigest()
            except OSError as exc:
                metadata_errors.append({"operation": "read_binary", "error": repr(exc)})
            failures = {
                key: values
                for key, values in (
                    ("cleanup_errors", cleanup_errors),
                    ("status_errors", status_errors),
                    ("metadata_errors", metadata_errors),
                )
                if values
            }
            manifest.update(failures)
            if failures and error is None:
                error = RuntimeError("Run finalization failed: " + repr(failures))
                manifest["execution_error"] = repr(error)
            manifest["execution_completed"] = (
                manifest["execution_completed"]
                and error is None
                and manifest["clock_verified"]
                and all(
                    not n.get("safety_deadline", True)
                    and n["exit_status"] == 0
                    and n.get("flushed") is True
                    for n in manifest["nodes"].values()
                )
            )

            def persist_manifest():
                try:
                    write(out / "run.json", manifest)
                except (OSError, ValueError, TypeError) as exc:
                    cause = error if error is not None else exc
                    raise RuntimeError(
                        f"{out}: failed to save run.json: {exc!r}; primary failure: {error!r}"
                    ) from cause

            # 先保存汇总，成功后才删除健康运行的临时标记；失败证据保持原样。
            persist_manifest()
            if error is None and manifest["execution_completed"]:
                for name in (
                    [r + ".started" for r in ROLES]
                    + [r + ".status.json" for r in ROLES]
                    + ["published.json", "matched.json"]
                ):
                    try:
                        (out / name).unlink(missing_ok=True)
                    except OSError as exc:
                        metadata_errors.append(
                            {
                                "operation": "remove_marker",
                                "file": name,
                                "error": repr(exc),
                            }
                        )
                if metadata_errors:
                    error = RuntimeError(
                        "Marker cleanup failed: " + repr(metadata_errors)
                    )
                    manifest["metadata_errors"] = metadata_errors
                    manifest["execution_error"] = repr(error)
                    manifest["execution_completed"] = False
                    persist_manifest()
        try:
            result = load_and_analyze(out)
            save_report(out, result)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            if error is not None:
                raise RuntimeError(
                    f"{out}: {error}; analysis/report failure: {exc!r}"
                ) from error
            raise
        if error:
            raise RuntimeError(f"{out}: {error}") from error
        if not manifest["execution_completed"] or not manifest["clock_verified"]:
            raise RuntimeError("Pipeline execution/clock verification failed")
        return out, result


def run_host(args):
    host = {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "offsets": Path("/proc/self/timens_offsets").read_text(),
    }
    inside = [
        "python3",
        "-m",
        "robottrace.workload",
        "--inside",
        args.scenario,
        str(args.count),
        str(args.rate),
        str(args.depth),
        str(
            args.work_ms
            if args.work_ms is not None
            else (1 if args.scenario == "normal" else 50)
        ),
        str(args.capacity),
    ]
    return subprocess.call(
        [
            str(ROOT / "scripts/ros.sh"),
            "env",
            "ROBOTTRACE_HOST_CLOCK=" + json.dumps(host),
            "timeout",
            "--signal=INT",
            "--kill-after=5s",
            "110s",
            *inside,
        ]
    )


if __name__ == "__main__":
    import sys
    from .cli import display

    if len(sys.argv) != 8 or sys.argv[1] != "--inside":
        raise SystemExit("Use python3 -m robottrace run")
    path, result = run_pipeline(
        ROOT / "evidence",
        sys.argv[2],
        count=int(sys.argv[3]),
        rate=float(sys.argv[4]),
        depth=int(sys.argv[5]),
        work_ms=float(sys.argv[6]),
        capacity=int(sys.argv[7]),
    )
    display(result)
    print("Evidence:", path.relative_to(ROOT))
