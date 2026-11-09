"""固定 MobileNetV2 CUDA 初始化证据与隔离容器入口。"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
GPU_SDK = ROOT / ".deps/gpu/onnxruntime-linux-x64-gpu-1.20.1"
CUDNN = ROOT / ".deps/gpu/cudnn-linux-x86_64-9.3.0.75_cuda12-archive/lib"
GPU_HASHES = {
    "libonnxruntime.so.1.20.1": "add8e0506fac2b323350ef403465c03bfa0e4fbc7e0eeed39645a3e33627af94",
    "libonnxruntime_providers_cuda.so": "94f603314574743b33d3ffb3558328d7db9d10639553e064a0001237180b47a4",
    "libonnxruntime_providers_shared.so": "f9400f9550e63513760f33a5c259b69fe93373865933fea5823b74c0036c4d2a",
}
CUDA_SONAMES = (
    "libcublasLt.so.12", "libcublas.so.12", "libcurand.so.10", "libcufft.so.11",
    "libcudart.so.12", "libnvrtc.so.12", "libnvrtc-builtins.so.12.4",
)
CUDA_HOST = Path("/usr/local/cuda-12.4/targets/x86_64-linux/lib")
CUDA_OPS = Counter(Conv=52, Clip=35, Add=10, Shape=1, GlobalAveragePool=1, Reshape=1, Gemm=1)
CPU_NODES = {"Gather_100": "Gather", "Unsqueeze_101": "Unsqueeze", "Concat_102": "Concat"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field: " + key)
        result[key] = value
    return result


def read_json(path):
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_fields,
                      parse_constant=invalid_constant)


def validate_profile(events, pid, calls=5):
    """按真实 model_run 区间唯一归组，不能用总节点数替代逐次验收。"""
    if type(pid) is not int or pid <= 0 or calls != 5 or not isinstance(events, list):
        raise ValueError("Invalid warm-up profile context")

    def interval(event):
        if not isinstance(event, dict) or event.get("ph") != "X":
            raise ValueError("Unsupported profile event")
        for key in ("ts", "dur", "pid", "tid"):
            if type(event.get(key)) is not int:
                raise ValueError("Missing or invalid profile " + key)
        if event["ts"] < 0 or event["dur"] < 0 or event["pid"] != pid or event["tid"] <= 0:
            raise ValueError("Invalid profile time/process identity")
        return event["ts"], event["ts"] + event["dur"]

    if any(not isinstance(event, dict) for event in events):
        raise ValueError("Non-object profile observation")
    runs = [event for event in events if event.get("name") == "model_run"]
    if len(runs) != calls:
        raise ValueError("Expected exactly five model_run observations")
    for event in runs:
        interval(event)
        if event.get("cat") != "Session" or event["dur"] <= 0:
            raise ValueError("Invalid model_run boundary")
    runs.sort(key=lambda event: event["ts"])
    if any(interval(left)[1] > interval(right)[0] for left, right in zip(runs, runs[1:])):
        raise ValueError("Overlapping/ambiguous model_run intervals")
    groups = [[] for _ in runs]
    for event in events:
        if event.get("cat") != "Node":
            continue
        name = event.get("name")
        if not isinstance(name, str):
            raise ValueError("Missing node name")
        if name.endswith(("_fence_before", "_fence_after")):
            if not isinstance(event.get("args"), dict) or event["args"].get("provider"):
                raise ValueError("Unexpected execution provider on fence marker")
            continue
        if not name.endswith("_kernel_time"):
            raise ValueError("Unsupported node observation")
        start, end = interval(event)
        args = event.get("args")
        if not isinstance(args, dict) or args.get("provider") not in (
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ) or not isinstance(args.get("op_name"), str):
            raise ValueError("Missing/unsupported node execution provider")
        index = args.get("node_index")
        if not isinstance(index, str) or not index.isdecimal():
            raise ValueError("Missing node index")
        owners = [i for i, run in enumerate(runs)
                  if run["ts"] <= start and end <= run["ts"] + run["dur"]
                  and event["tid"] == run["tid"]]
        if len(owners) != 1:
            raise ValueError("Missing/ambiguous per-call node attribution")
        groups[owners[0]].append(event)
    summaries, fingerprints = [], []
    for i, (run, nodes) in enumerate(zip(runs, groups)):
        gpu = [node for node in nodes if node["args"]["provider"] == "CUDAExecutionProvider"]
        cpu = [node for node in nodes if node["args"]["provider"] == "CPUExecutionProvider"]
        identifiers = [(node["args"]["node_index"], node["name"],
                        node["args"]["op_name"], node["args"]["provider"]) for node in nodes]
        if len(gpu) != 101 or len(cpu) != 3 or len(set(identifiers)) != 104:
            raise ValueError("Incomplete/duplicate warm-up execution: call " + str(i))
        if Counter(node["args"]["op_name"] for node in gpu) != CUDA_OPS:
            raise ValueError("CUDA graph fingerprint mismatch")
        actual_cpu = {node["name"].removesuffix("_kernel_time"): node["args"]["op_name"] for node in cpu}
        if actual_cpu != CPU_NODES or len(actual_cpu) != len(cpu):
            raise ValueError("Unexpected CPU fallback nodes")
        fingerprint = digest_bytes(json.dumps(sorted(identifiers), separators=(",", ":")).encode())
        fingerprints.append(fingerprint)
        summaries.append(dict(call_index=i, pid=pid, tid=run["tid"], start_us=run["ts"],
                              duration_us=run["dur"], cuda_nodes=101, cpu_nodes=sorted(actual_cpu),
                              node_fingerprint_sha256=fingerprint))
    if len(set(fingerprints)) != 1:
        raise ValueError("Graph placement changed between warm-ups")
    return dict(status="passed", calls=summaries, inference_calls=5,
                attribution="unique enclosing model_run interval and matching pid/tid")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def verify_loaded_libraries(paths, sdk_root):
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError("Malformed loaded ORT library evidence")
    expected = {str((Path(sdk_root) / "lib" / name).resolve(strict=True)): fingerprint
                for name, fingerprint in GPU_HASHES.items()}
    actual = set(paths)
    if actual != set(expected) or len(paths) != len(actual):
        raise ValueError("Loaded GPU SDK paths missing, duplicated or mismatched")
    result = []
    for path, fingerprint in expected.items():
        if digest(path) != fingerprint:
            raise ValueError("Loaded GPU SDK identity mismatch: " + path)
        result.append(dict(path=path, sha256=fingerprint))
    return result


def cuda_runtime_paths():
    if ROOT == Path("/work"):
        return {name: Path("/opt/rt/cuda") / name for name in CUDA_SONAMES}
    return {name: CUDA_HOST / name for name in CUDA_SONAMES}


def container_command(command, *, build_output=None, evidence_dir=None):
    """仅供这条固定推理链路使用；编译不申请 GPU，运行显式使用 CDI。"""
    if not command or bool(build_output) == bool(evidence_dir):
        raise ValueError("Select one build output or runtime evidence directory")
    if os.environ.get("LD_PRELOAD"):
        raise ValueError("Unexpected LD_PRELOAD")
    for name, expected in GPU_HASHES.items():
        if digest(GPU_SDK / "lib" / name) != expected:
            raise ValueError("Unexpected GPU SDK: " + name)
    if (GPU_SDK / "VERSION_NUMBER").read_text().strip() != "1.20.1":
        raise ValueError("Unexpected GPU SDK version")
    image = subprocess.check_output([str(ROOT / "environment/image.sh")], text=True).strip()
    image_id = subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
    fixed = "sha256:45472e053829134c3f508485e444eb5605a16a0c6f91a5e7a7e6ea86fe960582"
    if image_id != fixed:
        raise ValueError("Unexpected Humble image identity")
    mounts = [(ROOT, "/work", True), (GPU_SDK, "/opt/rt/ort", True), (CUDNN, "/opt/rt/cudnn", True)]
    mounts += [((CUDA_HOST / name).resolve(strict=True), "/opt/rt/cuda/" + name, True) for name in CUDA_SONAMES]
    if build_output:
        requested = Path(build_output)
        target = requested if requested.is_absolute() else ROOT / requested
        if ".." in requested.parts or target.parent != ROOT / "bin" or target.exists() or target.is_symlink() or (ROOT / "bin").is_symlink():
            raise ValueError("CUDA build output must be a new direct child of bin/")
        subprocess.run(["git", "check-ignore", "--quiet", "--", str(target)], cwd=ROOT, check=True)
        target.mkdir()
    else:
        target = Path(evidence_dir).resolve(strict=True)
        if target == ROOT / "evidence" or not target.is_dir() or not target.is_relative_to(ROOT / "evidence"):
            raise ValueError("CUDA runtime requires a dedicated evidence subdirectory")
    mounts.append((target, "/work/" + str(target.relative_to(ROOT)), False))
    args = ["docker", "run", "--name", "robottrace-inference-" + uuid.uuid4().hex,
            "--init", "--pull=never", "--user", f"{os.getuid()}:{os.getgid()}",
            "--workdir", "/work", "--label", "robottrace=inference-cuda"]
    args += ["--network", "none"] if build_output else ["--network", "host", "--ipc", "host", "--device=nvidia.com/gpu=all"]
    for key, value in dict(HOME="/tmp/robottrace-home", CUDA_CACHE_PATH="/tmp/robottrace-cuda-cache",
                           ROS_DOMAIN_ID="42", ROS_LOCALHOST_ONLY="1", RMW_IMPLEMENTATION="rmw_fastrtps_cpp",
                           RCUTILS_COLORIZED_OUTPUT="0", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1").items():
        args += ["--env", key + "=" + value]
    for source, destination, readonly in mounts:
        value = f"type=bind,source={source},target={destination}"
        args += ["--mount", value + (",readonly" if readonly else "")]
    args += ["--entrypoint", "/bin/bash", fixed, "-c",
             'mkdir -p "$HOME"; source /opt/ros/humble/setup.bash; exec "$@"', "robottrace", *command]
    return args


def run_container(command):
    """失败容器保留；仅在成功退出后删除本次新建的容器。"""
    result = subprocess.run(command)
    if result.returncode == 0:
        name = command[command.index("--name") + 1]
        subprocess.run(["docker", "rm", name], check=True, stdout=subprocess.DEVNULL)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("verify-profile")
    profile.add_argument("profile")
    profile.add_argument("--pid", type=int, required=True)
    profile.add_argument("--libraries", required=True)
    profile.add_argument("--sdk-root", required=True)
    profile.add_argument("--output", required=True)
    launch = commands.add_parser("container")
    launch.add_argument("--build-output")
    launch.add_argument("--evidence-dir")
    launch.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "container":
        argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        return run_container(container_command(argv, build_output=args.build_output, evidence_dir=args.evidence_dir))
    # 校验子进程不应在 Processor 超时退出后继续存活。
    import ctypes
    import signal
    if os.getppid() != args.pid or ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != args.pid:
        raise RuntimeError("CUDA evidence validator parent identity mismatch")
    try:
        result = validate_profile(read_json(args.profile), args.pid)
        result.update(profile_path=args.profile, profile_sha256=digest(args.profile),
                      loaded_libraries=verify_loaded_libraries(read_json(args.libraries), args.sdk_root))
    except (OSError, ValueError, TypeError, KeyError) as error:
        Path(args.output).write_text(json.dumps(dict(status="failed", error=repr(error))) + "\n")
        print("CUDA execution gate failed: " + str(error), file=sys.stderr)
        return 2
    Path(args.output).write_text(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
