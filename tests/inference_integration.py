#!/usr/bin/env python3
"""核对真实推理运行的原始证据；目录由独立的有界运行器创建。"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robottrace import inference
from robottrace.io import load_and_analyze, load_comparison_run
from robottrace.comparison import compare_runs


def validate(path):
    path = Path(path)
    manifest = inference.read(path / "run.json")
    analysis = load_and_analyze(path)
    count = manifest["configuration"]["count"]
    assert analysis == inference.read(path / "analysis.json")
    assert manifest["correctness"]["status"] == "passed"
    assert manifest["expected_sequences"] == {"source:1": list(range(1, count + 1))}
    assert analysis["event_count"] == 9 * count
    assert analysis["complete_message_count"] == count
    assert analysis["trace_integrity"]["status"] == "valid"
    assert analysis["collection"]["status"] == "complete"
    assert analysis["observations"]["status"] == "complete"
    assert analysis["execution"]["status"] == "completed"
    assert (
        manifest["clock_verified"]
        and analysis["clock"]["verified_by_producer_evidence"]
    )
    assert all(c["dropped"] == 0 and c["flushed"] for c in manifest["collectors"])
    assert len({n["process_id"] for n in manifest["nodes"].values()}) == 3
    assert all(code == 0 for code in manifest["exit_codes"].values())
    assert manifest["nodes"]["source"]["published"] == count
    assert manifest["nodes"]["processor"]["inference_operations"] == count
    assert manifest["nodes"]["processor"]["session_initializations"] == 1
    assert manifest["nodes"]["processor"]["warmup_count"] == 5
    if "processor_configuration" in manifest:
        threads = manifest["configuration"]["intra_op_threads"]
        applied = inference.verify_processor_configuration(
            manifest["processor_configuration"], threads
        )
        assert manifest["processor_readiness_configuration_verified"]
        assert manifest["processor_status_configuration_verified"]
        assert manifest["readiness"]["processor"]["inference_configuration"] == applied
        assert manifest["nodes"]["processor"]["inference_configuration"] == applied
    assert manifest["nodes"]["consumer"]["valid_unique"] == count
    assert all(
        n["invalid"] == n["duplicates"] == n["diagnostic_dropped"] == 0
        for n in manifest["nodes"].values()
    )
    assert all(
        value["count"] == count and value["unavailable_count"] == 0
        for value in analysis["metrics"].values()
    )
    assert len(set(manifest["input_tensor_sha256"])) == inference.SAMPLE_COUNT
    expected, _ = inference.fixture_data()
    raw = [
        row for name in manifest["event_files"] for row in inference.rows(path / name)
    ]
    points = {
        (r["sequence"], r["stage"], r["event_type"]): r["timestamp_ns"] for r in raw
    }
    release = next(
        p["timestamp_ns"] for p in manifest["phases"] if p["phase"] == "release"
    )
    first = min(points[(seq, "source", "generate")] for seq in range(1, count + 1))
    assert manifest["nodes"]["processor"]["warmup_end_ns"] < release < first
    for role in ("processor", "consumer"):
        data = inference.rows(path / (role + ".results.jsonl"))
        assert len(data) == count and len({row["sequence"] for row in data}) == count
        assert {row["sample_id"] for row in data} == set(
            range(min(count, inference.SAMPLE_COUNT))
        )
        for row in data:
            seq = row["sequence"]
            sample = (seq - 1) % inference.SAMPLE_COUNT
            assert (
                row["run_id"] == manifest["run_id"] and row["source_id"] == "source:1"
            )
            assert row["sample_id"] == sample and row["valid"]
            assert row["generated_ns"] == points[(seq, "source", "generate")]
            assert row["entry_ns"] == points[(seq, role, "callback_entry")]
            assert len(row["logits"]) == inference.OUTPUT_ELEMENTS
            assert all(
                abs(a - b) <= inference.ATOL + inference.RTOL * abs(b)
                for a, b in zip(row["logits"], expected[sample])
            )
            if role == "processor":
                assert row["input_payload_verified"] and row["api_returned"]
                assert (
                    row["entry_ns"]
                    <= row["api_begin_ns"]
                    <= row["api_end_ns"]
                    <= points[(seq, "processor", "publish_begin")]
                )
    assert manifest["inference_diagnostics"] == inference.diagnostics(
        path, manifest, analysis
    )
    assert manifest["inference_diagnostics"]["api_ns"]["count"] == count
    assert (
        manifest["inference_diagnostics"]["processor_pre_publish_ns"]["count"] == count
    )
    comparable = load_comparison_run(path)
    rejected = compare_runs(comparable, comparable)
    assert rejected["status"] == "NOT_COMPARABLE"
    assert all(v["verdict"] == "WITHHELD" for v in rejected["metrics"].values())
    print(
        f"PASS {path.name}: {count} correct unique results; {9*count} events; clock verified; comparison withheld"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: inference_integration.py RUN_DIR [RUN_DIR ...]")
    for directory in sys.argv[1:]:
        validate(directory)
