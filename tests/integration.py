#!/usr/bin/env python3
"""有时限的真实 ROS2 验收，以及派生的缺失观测样本。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robottrace.workload import run_pipeline, write
from robottrace.io import load_and_analyze, save_report


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
                "tests/integration.py",
                "--inside",
            ]
        )
    suite = (
        ROOT
        / "evidence"
        / (
            "r1-01_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
            + uuid.uuid4().hex[:6]
        )
    )
    suite.mkdir(parents=True)
    completed = []
    for scenario in ("normal", "slow"):
        path, r = run_pipeline(suite, scenario)
        assert r["execution"]["status"] == "completed"
        assert r["collection"]["status"] == "complete"
        assert r["trace_integrity"]["status"] == "valid"
        assert r["clock"]["verified_by_producer_evidence"]
        assert r["event_count"] > 0 and r["correlated_message_count"] > 0
        assert r["metrics"]["processor_callback_ns"]["count"] > 0
        minimum = 1_000_000 if scenario == "normal" else 50_000_000
        assert r["metrics"]["processor_callback_ns"]["min"] >= minimum
        if scenario == "normal":
            assert r["observations"]["status"] == "complete"
            assert r["complete_message_count"] == 100
            normal_path = path
        # 不对慢速消费者的精确接收数量或调度时序作断言。
        completed.append(
            {
                "scenario": scenario,
                "directory": path.name,
                "run_id": r["run_id"],
                "event_count": r["event_count"],
                "correlated_messages": r["correlated_message_count"],
                "complete_messages": r["complete_message_count"],
                "observations": r["observations"]["status"],
                "processing": r["metrics"]["processor_callback_ns"],
                "age": r["metrics"]["consumer_age_ns"],
            }
        )
        print(
            f"PASS {scenario}: {path.name}, events={r['event_count']} complete_messages={r['complete_message_count']}",
            flush=True,
        )
    fixture = suite / "incomplete_fixture"
    fixture.mkdir()
    original = json.loads((normal_path / "run.json").read_text())
    all_events = []
    for filename in original["event_files"]:
        all_events.extend(
            json.loads(line)
            for line in (normal_path / filename).read_text().splitlines()
        )
    removed = next(
        e
        for e in all_events
        if e["sequence"] == 1
        and e["stage"] == "processor"
        and e["event_type"] == "callback_exit"
    )
    all_events.remove(removed)
    (fixture / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in all_events)
    )
    original.update(
        event_files=["events.jsonl"],
        collectors=None,
        execution_completed=None,
        fixture={
            "derived_from": normal_path.name,
            "removed_event": removed,
            "execution": "not a new ROS run",
        },
    )
    write(fixture / "run.json", original)
    r = load_and_analyze(fixture)
    save_report(fixture, r)
    target = next(m for m in r["messages"] if m["sequence"] == 1)
    assert r["observations"]["status"] == "incomplete"
    assert r["trace_integrity"]["status"] == "valid"
    assert target["metrics"]["processor_callback_ns"]["value_ns"] is None
    assert target["metrics"]["processor_callback_ns"]["reason"] == "MISSING_ENDPOINT"
    assert target["metrics"]["consumer_age_ns"]["status"] == "available"
    assert r["metrics"]["processor_callback_ns"]["count"] == 99
    print(
        "PASS incomplete_fixture: missing duration remains null; independent age remains available.",
        flush=True,
    )
    overhead = json.loads(
        subprocess.check_output([str(ROOT / "bin/event_cost")], text=True, timeout=10)
    )
    assert (
        overhead["overflow_probe"]["stored"] == 1
        and overhead["overflow_probe"]["dropped"] == 1
    )
    write(suite / "overhead.json", overhead)
    write(
        suite / "suite.json",
        {
            "assessment": "PASS",
            "integration_cases": 3,
            "workloads": completed,
            "fixture": "incomplete_fixture",
            "instrumentation_overflow_probe": "PASS",
        },
    )
    (ROOT / "evidence/latest-suite.txt").write_text(suite.name + "\n")
    print(
        "PASS integration (3 cases) and bounded event-collection/overhead probe:",
        suite.relative_to(ROOT),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
