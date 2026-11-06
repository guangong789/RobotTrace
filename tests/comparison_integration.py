#!/usr/bin/env python3
"""运行一个有界的真实处理时长实验，并通过公共 CLI 重现比较结果。"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robottrace.comparison import compare_runs
from robottrace.io import load_comparison_run
from robottrace.workload import run_pipeline, write


def main():
    if "--inside" not in sys.argv:
        clock = {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "offsets": Path("/proc/self/timens_offsets").read_text(),
        }
        return subprocess.call(
            [
                str(ROOT / "scripts/ros.sh"),
                "env",
                "ROBOTTRACE_HOST_CLOCK=" + json.dumps(clock),
                "timeout",
                "--signal=INT",
                "--kill-after=5s",
                "200s",
                "python3",
                "tests/comparison_integration.py",
                "--inside",
            ]
        )
    suite = (
        ROOT
        / "evidence"
        / (
            "r1-02_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
            + uuid.uuid4().hex[:6]
        )
    )
    suite.mkdir(parents=True)
    baseline, _ = run_pipeline(suite, "slow", count=20, rate=10, work_ms=50)
    candidate, _ = run_pipeline(suite, "slow", count=20, rate=10, work_ms=10)
    inputs = [p for run in (baseline, candidate) for p in run.iterdir() if p.is_file()]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    args = [
        sys.executable,
        "-m",
        "robottrace",
        "compare",
        str(baseline),
        str(candidate),
    ]
    completed = subprocess.run(
        [
            *args,
            "--allow-change",
            "processor_work_ms",
            "--json",
            "--output",
            str(suite / "comparison.json"),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    compared = json.loads(completed.stdout)
    expected = compare_runs(
        load_comparison_run(baseline),
        load_comparison_run(candidate),
        allow_changes=["processor_work_ms"],
    )
    assert compared == expected == json.loads((suite / "comparison.json").read_text())
    assert compared["conditions"]["status"] == "COMPARABLE"
    assert compared["declared_changes"] == [
        {"key": "processor_work_ms", "baseline": 50, "candidate": 10}
    ]
    for name, metric in compared["metrics"].items():
        if metric["status"] != "COMPARABLE":
            assert metric["verdict"] == "WITHHELD"
        # 不断言调度产生的精确数值、百分比或改进方向。
        print(
            name,
            metric["status"],
            metric["verdict"],
            json.dumps(metric["changes"]),
            flush=True,
        )
    rejected = subprocess.run(
        [*args, "--json"], capture_output=True, text=True, timeout=20
    )
    assert rejected.returncode == 0, rejected.stderr
    unexplained = json.loads(rejected.stdout)
    assert unexplained["status"] == "NOT_COMPARABLE"
    assert all(m["verdict"] == "WITHHELD" for m in unexplained["metrics"].values())
    write(suite / "comparison-undeclared.json", unexplained)
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest() == value
        for p, value in hashes.items()
    )
    print(
        "PASS real comparison and persisted-evidence reproduction:",
        suite.relative_to(ROOT),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
