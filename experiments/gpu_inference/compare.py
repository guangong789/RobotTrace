"""固定 A1→B1→C1→C2→B2→A2 的独立进程实验；失败即停，不重试。"""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment(evidence):
    plan = json.loads((evidence / "plan.json").read_text())
    assert plan["order"] == ["A1", "B1", "C1", "C2", "B2", "A2"]
    assert plan["warmup_count"] == 5 and plan["measured_count"] == 100
    build = Path(plan["build_root"])
    provenance = json.loads((evidence / "provenance.json").read_text())
    model = ROOT / "models/mobilenet/data/mobilenetv2-12.onnx"
    assert digest(model) == plan["model_sha256"]
    for path, expected in provenance.items():
        assert digest(ROOT / path) == expected, path
    assert not (evidence / "execution.json").exists(), "Refusing to overwrite an experiment"
    libraries = {
        "cpu": ROOT / ".deps/onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so.1.20.1",
        "cuda": ROOT / ".deps/gpu/onnxruntime-linux-x64-gpu-1.20.1/lib/libonnxruntime.so.1.20.1"}
    prior = json.loads((ROOT / "evidence/g002_20261108T125917Z/resume_20261108T131755Z/gpu-result.json").read_text())
    prior_options = dict(item.split("=", 1) for item in prior["cuda_options"].split(";") if item)
    records = []

    def prefix(provider):
        value = ["sh", str(ROOT / "scripts/gpu-inference.sh")]
        if provider == "cpu":
            value += ["env", "LD_LIBRARY_PATH=" + str(libraries["cpu"].parent) +
                      ":/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib"]
        return value

    def command(provider, threads, check=False, profile=False):
        kind = "cpu" if provider == "cpu" else "gpu"
        value = prefix(provider) + [str(build / kind / "benchmark_inference"),
            str(ROOT / "models/mobilenet"), "--provider", provider,
            "--intra-op-threads", str(threads), "--expect-ort-library", str(libraries[provider])]
        if check:
            value += ["--check"]
        if profile:
            value += ["--profile-prefix", str(evidence / "verification-profile")]
        return value

    def snapshot(label):
        process = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        (evidence / (label + "-gpu.log")).write_text(process.stdout + process.stderr)
        assert process.returncode == 0, "GPU inspection failed"
        assert "No running processes found" in process.stdout, "Competing or unclassified GPU process"

    def invoke(label, args, expected_error=None):
        row = {"identity": label, "command": args,
               "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "start_monotonic_ns": time.monotonic_ns(), "status": "running"}
        records.append(row)
        write(evidence / "execution.json", records)
        with (evidence / (label + ".json")).open("x") as out, (evidence / (label + ".stderr.log")).open("x") as err:
            process = subprocess.run(args, cwd=ROOT, stdout=out, stderr=err, timeout=120)
        row.update(returncode=process.returncode, end_monotonic_ns=time.monotonic_ns(),
                   end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(evidence / "execution.json", records)
        result = json.loads((evidence / (label + ".json")).read_text())
        if expected_error is not None:
            assert process.returncode == 1 and result["status"] == "failed", label
            assert expected_error in (evidence / (label + ".stderr.log")).read_text(), label
            assert result["observations"] == [] and result["session_initializations"] == 0
            row["status"] = "expected rejection"
        else:
            assert process.returncode == 0 and result["status"] == "passed", label
            row["status"] = "completed; awaiting validation"
        write(evidence / "execution.json", records)
        return result

    def validate(result, provider, threads, check=False, profile=False):
        assert result["runtime"] == "1.20.1" and result["provider"] == provider
        assert result["mode"] == ("check" if check else "benchmark")
        assert result["library_verified"] and result["provider_registered"]
        assert result["session_initializations"] == 1 and result["intra_op_num_threads"] == threads
        assert result["inter_op_num_threads"] == 1 and result["execution_mode"] == "sequential"
        assert result["graph_optimization"] == "basic" and result["cpu_arena"]
        assert not result["spinning_explicitly_set"] and result["host_input_output"]
        assert result["clock"] == "CLOCK_MONOTONIC" and result["profiling_enabled"] == profile
        assert str(libraries[provider]) in result["loaded_libraries"]
        opposite = libraries["cuda" if provider == "cpu" else "cpu"]
        assert str(opposite) not in result["loaded_libraries"]
        assert math.isclose(result["atol"], 1e-4, rel_tol=1e-7)
        assert math.isclose(result["rtol"], 1e-4, rel_tol=1e-7)
        if provider == "cuda":
            options = dict(item.split("=", 1) for item in result["cuda_options"].split(";") if item)
            assert options == prior_options, "CUDA provider configuration changed"
            assert options["device_id"] == "0" and options["use_tf32"] == "0"
            assert any("libonnxruntime_providers_cuda.so" in p for p in result["loaded_libraries"])
            assert any("/.deps/gpu/" in p and "libcudnn.so.9.3.0" in p for p in result["loaded_libraries"])
        else:
            assert result["cuda_options"] == ""
        warmup, measured = (0, 4) if check else (5, 100)
        assert result["warmup_count"] == warmup and result["valid_inferences"] == measured
        assert result["session_construction_ns"] > 0
        assert result["model_initialization_ns"] >= result["session_construction_ns"]
        observations = result["observations"]
        assert len(observations) == warmup + measured
        for index, row in enumerate(observations):
            assert row["sample_id"] == index % 2 and row["warmup"] == (index < warmup)
            assert row["index"] == (index if index < warmup else index - warmup)
            assert row["api_returned"] and row["api_end_ns"] > row["api_begin_ns"]
            assert row["duration_ns"] == row["api_end_ns"] - row["api_begin_ns"]
            assert row["numerically_correct"] and row["output_values_checked"] == 1000
            assert row["mismatches"] == row["nonfinite"] == 0
            assert math.isfinite(row["max_abs_error"])
        samples = sorted(row["duration_ns"] for row in observations if not row["warmup"])
        assert result["api_ns"] == {"count": measured, "p50": samples[(measured + 1) // 2 - 1],
            "p95": samples[(95 * measured + 99) // 100 - 1], "min": samples[0], "max": samples[-1]}
        assert result["measured_wall_ns"] >= sum(samples)
        assert result["measured_begin_ns"] <= observations[warmup]["api_begin_ns"]
        assert observations[-1]["api_end_ns"] <= result["measured_end_ns"]

    for provider in ["cpu", "cuda"]:
        kind = "cpu" if provider == "cpu" else "gpu"
        with (evidence / (kind + "-linkage.log")).open("x") as log:
            for tool in [["readelf", "-d"], ["ldd"]]:
                process = subprocess.run(prefix(provider) + tool + [str(build / kind / "benchmark_inference")],
                                         capture_output=True, text=True, timeout=20)
                log.write(process.stdout + process.stderr)
                log.flush()
                assert process.returncode == 0 and "not found" not in process.stdout
                if tool[0] == "ldd":
                    assert str(libraries[provider].parent) in process.stdout
    bad = command("cpu", 1)
    bad[bad.index("--intra-op-threads") + 1] = "2"
    invoke("check-invalid-threads", bad, "intra-op threads must be 1 or 4")
    invoke("check-no-timed-profiling", command("cuda", 1, profile=True), "Usage:")
    invoke("check-no-cuda-four-threads", command("cuda", 4), "Usage:")
    for provider, threads, label in [("cpu", 1, "check-A"), ("cpu", 4, "check-B"), ("cuda", 1, "check-C")]:
        snapshot(label + "-before")
        result = invoke(label, command(provider, threads, check=True, profile=provider == "cuda"))
        validate(result, provider, threads, check=True, profile=provider == "cuda")
        if provider == "cuda":
            events = json.loads(Path(result["profile_path"]).read_text())
            kernels = [event for event in events if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")]
            groups = collections.defaultdict(collections.Counter)
            for event in kernels:
                groups[event["args"]["provider"]][(event["name"], event["args"]["op_name"])] += 1
            expected = {"CUDAExecutionProvider": 101, "CPUExecutionProvider": 3}
            assert {provider: len(nodes) for provider, nodes in groups.items()} == expected
            assert all(count == 4 for nodes in groups.values() for count in nodes.values())
            proof = {provider: [{"name": name, "operator": op, "executions": count}
                               for (name, op), count in sorted(nodes.items())]
                     for provider, nodes in groups.items()}
            write(evidence / "cuda-execution-proof.json", {"binary_sha256": digest(build / "gpu/benchmark_inference"),
                "profile": result["profile_path"], "profile_sha256": digest(Path(result["profile_path"])),
                "provider_options_unchanged_from_G002": True, "nodes": proof})
        records[-1]["status"] = "validated focused check"
        write(evidence / "execution.json", records)
        snapshot(label + "-after")
        print("Focused check passed:", label, flush=True)
    write(evidence / "focused-checks.json", {"status": "passed", "CLI_rejections": 3,
          "CPU1_check_calls": 4, "CPU4_check_calls": 4, "GPU_profile_check_calls": 4,
          "timed_profiling_rejected": True})
    print("Focused checks and separate CUDA execution proof passed; starting fixed six-run experiment.", flush=True)
    runs = []
    for identity in plan["order"]:
        config = plan["configurations"][identity[0]]
        provider, threads = config["provider"], config["intra_op_num_threads"]
        snapshot(identity + "-before")
        result = invoke(identity, command(provider, threads))
        validate(result, provider, threads)
        assert all(result["pid"] != previous["pid"] for previous in runs)
        records[-1]["status"] = "validated benchmark"
        write(evidence / "execution.json", records)
        row = {"identity": identity, "provider": provider, "threads": threads, "pid": result["pid"],
               "api_p50_ms": result["api_ns"]["p50"] / 1e6, "api_p95_ms": result["api_ns"]["p95"] / 1e6,
               "api_min_ms": result["api_ns"]["min"] / 1e6, "api_max_ms": result["api_ns"]["max"] / 1e6,
               "session_construction_ms": result["session_construction_ns"] / 1e6,
               "model_initialization_ms": result["model_initialization_ns"] / 1e6,
               "warmup_ms": result["warmup_wall_ns"] / 1e6,
               "measured_wall_ms": result["measured_wall_ns"] / 1e6,
               "correct": result["valid_inferences"], "total": 100, "max_abs_error": result["max_abs_error"]}
        runs.append(row)
        write(evidence / "run-summary.json", runs)
        snapshot(identity + "-after")
        print(json.dumps(row), flush=True)
    summaries = {}
    for group in "ABC":
        selected = [row for row in runs if row["identity"].startswith(group)]
        summaries[group] = {}
        for metric in ["api_p50_ms", "api_p95_ms"]:
            values = [row[metric] for row in selected]
            summaries[group][metric] = {"values": values, "median": statistics.median(values),
                "min": min(values), "max": max(values), "range": max(values) - min(values)}
    comparisons = {}
    for before, after in [("A", "B"), ("B", "C"), ("A", "C")]:
        baseline, candidate = summaries[before]["api_p50_ms"]["median"], summaries[after]["api_p50_ms"]["median"]
        comparisons[before + "_to_" + after] = {"basis": "median of two run-level API p50 values",
            "latency_reduction_percent": (1 - candidate / baseline) * 100, "speedup": baseline / candidate}
    write(evidence / "summary.json", {"status": "passed", "runs": runs, "run_level_variation": summaries,
        "comparisons": comparisons, "measured_samples": 600, "measured_output_values_checked": 600000,
        "warmup_samples": 30, "limitations": ["Two processes per configuration; no statistical significance",
            "Host synchronous API times, including GPU transfers and synchronization; not kernel times",
            "WSL GPU visibility and discrete snapshots do not exclude all interference or throttling",
            "No inference on ROS2 Consumer Age or energy; CPU and GPU SDK native builds differ",
            "CUDA Graph key absent in readback; no enable call; pinned 1.20.1 source default false"]})


if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve(strict=True)
    try:
        experiment(target)
    except Exception as error:
        write(target / "STOP.json", {"phase": "focused validation or fixed experiment", "error": str(error),
              "traceback": traceback.format_exc(), "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
