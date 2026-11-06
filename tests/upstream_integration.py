#!/usr/bin/env python3
"""真实执行两个固定上游场景，重读原始证据验证原有核心的完整性语义。"""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robottrace import upstream
from robottrace.io import load_and_analyze


def validate(path, scenario):
    manifest = upstream.read(path / "run.json")
    saved = upstream.read(path / "analysis.json")
    assert load_and_analyze(path) == saved
    assert manifest["expected_sequences"] == {"demo:talker": list(range(1, 21))}
    assert saved["execution"]["status"] == "completed"
    assert saved["collection"]["status"] == "complete"
    assert saved["trace_integrity"]["status"] == "valid"
    assert saved["clock"]["verified_by_producer_evidence"]
    assert not saved["trace_integrity"]["issues"]
    assert saved["correlated_message_count"] == (20 if scenario == "normal" else 17)
    assert saved["complete_message_count"] == (20 if scenario == "normal" else 17)
    assert saved["event_count"] == (180 if scenario == "normal" else 162)
    assert set(manifest["nodes"]) == set(upstream.ROLES)
    assert len({n["process_id"] for n in manifest["nodes"].values()}) == 3
    assert all(c["dropped"] == 0 and c["flushed"] for c in manifest["collectors"])
    assert all(n["invalid_messages"] == 0 for n in manifest["nodes"].values())
    assert manifest["publication"]["published"] == 20
    assert manifest["delivery"]["last_sequence"] == 20
    software = manifest["software"]
    assert set(software["executables"]) == set(upstream.ROLES)
    assert len({value["sha256"] for value in software["executables"].values()}) == 3
    assert any("librobottrace_demo_topics.so" in path for path in software["libraries"])
    assert any("librobottrace_relay.so" in path for path in software["libraries"])
    assert all(manifest["loaded_libraries"][role] for role in upstream.ROLES)
    assert any(
        "libstd_msgs__rosidl_typesupport_fastrtps_cpp.so" in library
        for library in manifest["loaded_libraries"]["processor"]
    )
    assert manifest["identity_binding"]["run_uuid_in_payload"] is False
    assert "binary_sha256" not in manifest
    readiness = manifest["readiness"]
    diagnostics = manifest["readiness_diagnostics"]
    assert diagnostics["frozen_before_cleanup"]
    assert diagnostics["poll_count"] >= 1
    assert len(diagnostics["entries"]) <= upstream.READINESS_HISTORY_LIMIT
    assert diagnostics["truncated"] == (diagnostics["dropped_entries"] > 0)
    assert diagnostics["diagnostic_error_count"] == 0
    assert diagnostics["entries"][-1]["event"] == "success"
    assert diagnostics["last_observation"]["predicate_result"] == "ready"
    assert not diagnostics["last_observation"]["unsatisfied"]
    for role in upstream.ROLES:
        last = diagnostics["last_observation"]["roles"][role]
        assert last["state"] == "reported" and last["file_present"]
        for key in ("matched_publishers", "matched_subscribers"):
            assert last[key] == readiness[role][key]
    assert readiness["source"]["output_qos"]["depth"] == 7
    assert readiness["processor"]["input_qos"]["depth"] == 10
    assert readiness["processor"]["output_qos"]["depth"] == 10
    assert readiness["consumer"]["input_qos"]["depth"] == 10
    assert all(
        item["exit_status"] == 0 and "std_msgs/msg/String" in item["stdout"]
        for item in manifest["startup_observers"]
    )
    for name, metric in saved["metrics"].items():
        expected = (
            20 if scenario == "normal" or name == "source_publish_call_ns" else 17
        )
        assert metric["count"] == expected, (name, metric)
        assert metric["unavailable_count"] == 20 - expected
    phases = manifest["phases"]
    ready = next(p["timestamp_ns"] for p in phases if p["phase"] == "ready")
    first_source = next(
        json.loads(line)
        for line in (path / "source.jsonl").read_text().splitlines()
        if json.loads(line)["event_type"] == "generate"
    )
    if scenario == "normal":
        assert saved["observations"]["status"] == "complete"
        assert saved["observations"]["missing_observation_count"] == 0
        assert ready < first_source["timestamp_ns"]
    else:
        assert saved["observations"]["status"] == "incomplete"
        assert saved["observations"]["missing_observation_count"] == 18
        prefix = next(p for p in phases if p["phase"] == "prefix_completed")
        relay = next(p for p in phases if p.get("stage") == "processor")
        assert prefix["published"] == 3 and prefix["matched_subscribers"] == 0
        assert prefix["timestamp_ns"] < relay["timestamp_ns"] < ready
        for item in saved["messages"][:3]:
            assert item["stages_present"] == ["source"]
            for name, metric in item["metrics"].items():
                if name != "source_publish_call_ns":
                    assert (
                        metric["value_ns"] is None
                        and metric["reason"] == "MISSING_ENDPOINT"
                    )
        for stage in ("processor", "consumer"):
            sequence = next(s for s in saved["sequences"] if s["stage"] == stage)
            assert sequence["expected_not_observed"] == [1, 2, 3]
            assert sequence["missing_prefix_count"] == 3
    print(
        f"PASS {scenario}: {path.name}; events={saved['event_count']}; "
        f"complete={saved['complete_message_count']}/20; collection=complete; "
        f"observations={saved['observations']['status']}; persisted reanalysis identical",
        flush=True,
    )
    return saved


def main():
    if "--inside" not in sys.argv:
        boot, offsets, _ = upstream.clock_evidence()
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
                "170s",
                "python3",
                "-B",
                "tests/upstream_integration.py",
                "--inside",
            ]
        )
    for scenario in ("normal", "late-relay"):
        path, _ = upstream.run_experiment(ROOT / "evidence", scenario)
        validate(path, scenario)
    print("PASS independent upstream integration (2 scenarios)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
