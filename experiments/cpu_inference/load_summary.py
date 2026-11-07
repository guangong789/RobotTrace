"""只读汇总指定的推理运行；不选择负载、不启动实验、不形成性能 verdict。"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from robottrace.analysis import statistics
from robottrace.io import load_and_analyze
from robottrace.inference import read, rows


def cadence(timestamps):
    ordered = sorted(timestamps)
    span = ordered[-1] - ordered[0] if len(ordered) >= 2 else None
    return {
        "samples": len(ordered),
        "span_ns": span,
        "hz": (
            (len(ordered) - 1) * 1e9 / span if span is not None and span > 0 else None
        ),
        "gap_ns": statistics([b - a for a, b in zip(ordered, ordered[1:])]),
    }


def summarize(directory):
    directory = Path(directory)
    manifest = read(directory / "run.json")
    analysis = load_and_analyze(directory)
    if (directory / "analysis.json").is_file() and analysis != read(
        directory / "analysis.json"
    ):
        raise ValueError("Saved analysis differs from raw evidence")
    nodes = manifest.get("nodes", {})
    result = {
        "directory": str(directory),
        "run_id": manifest["run_id"],
        "count": manifest["configuration"]["count"],
        "requested_hz": manifest["configuration"]["nominal_rate_hz"],
        "counts": {
            role: {
                name: nodes.get(role, {}).get(name)
                for name in (
                    "publish_attempts",
                    "published",
                    "received",
                    "observed_unique",
                    "inference_operations",
                    "valid_unique",
                    "invalid",
                    "duplicates",
                )
            }
            for role in ("source", "processor", "consumer")
        },
        "statuses": {
            axis: analysis[axis]["status"]
            for axis in ("execution", "collection", "trace_integrity", "observations")
        },
        "correctness": manifest["correctness"],
        "complete_identities": analysis["complete_message_count"],
        "incomplete_identities": analysis["message_identity_count"]
        - analysis["complete_message_count"],
        "missing_endpoints": analysis["observations"]["missing_observation_count"],
        "trace_issues": analysis["trace_integrity"]["issues"],
        "collector_reports": manifest.get("collectors"),
        "clock_verified": analysis["clock"]["verified_by_producer_evidence"],
        "failure": manifest.get("execution_error"),
        "timing": None,
    }
    if not result["clock_verified"] or analysis["trace_integrity"]["status"] != "valid":
        return result
    raw = [
        row
        for name in manifest["event_files"]
        if (directory / name).is_file()
        for row in rows(directory / name)
    ]

    def points(stage, kind):
        return {
            row["sequence"]: row["timestamp_ns"]
            for row in raw
            if row["stage"] == stage and row["event_type"] == kind
        }

    generated = points("source", "generate")
    published = points("source", "publish_end")
    processor = points("processor", "callback_exit")
    consumer = points("consumer", "callback_exit")
    # 完成吞吐只计入身份明确且输出校验通过的业务记录。
    for role, mapping in (("processor", processor), ("consumer", consumer)):
        path = directory / (role + ".results.jsonl")
        valid = (
            {
                r["sequence"]
                for r in rows(path)
                if r["valid"]
                and r["run_id"] == manifest["run_id"]
                and r["source_id"] == "source:1"
            }
            if path.is_file()
            else set()
        )
        for seq in list(mapping):
            if seq not in valid:
                del mapping[seq]
    first = min(generated.values()) if generated else None
    last = max((r["timestamp_ns"] for r in raw), default=None)
    stop = manifest.get("measurement_stop_ns")
    source = nodes.get("source", {})
    start, period = source.get("schedule_start_ns"), source.get("schedule_period_ns")
    lag = (
        [
            timestamp - (start + (seq - 1) * period)
            for seq, timestamp in sorted(generated.items())
        ]
        if start and period
        else []
    )
    ages = [
        m["metrics"]["consumer_age_ns"]["value_ns"]
        for m in analysis["messages"]
        if m["metrics"]["consumer_age_ns"]["status"] == "available"
    ]
    quarter = max(1, len(ages) // 4)
    result["timing"] = {
        "observation_window_ns": (
            stop - first if first is not None and stop is not None else None
        ),
        "active_event_window_ns": (
            last - first if first is not None and last is not None else None
        ),
        "source_generate": cadence(list(generated.values())),
        "source_publish_end": cadence(list(published.values())),
        "processor_callback_completion": cadence(list(processor.values())),
        "consumer_valid_completion": cadence(list(consumer.values())),
        "source_schedule_lateness_ns": statistics(lag),
        "source_final_lateness_ns": lag[-1] if lag else None,
        "source_publish_call_ns": analysis["metrics"]["source_publish_call_ns"],
        "inference_api_ns": manifest.get("inference_diagnostics", {}).get("api_ns"),
        "processor_callback_ns": analysis["metrics"]["processor_callback_ns"],
        "consumer_age_ns": analysis["metrics"]["consumer_age_ns"],
        "consumer_age_first_quarter_ns": statistics(ages[:quarter]),
        "consumer_age_last_quarter_ns": statistics(ages[-quarter:]),
    }
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: load_summary.py RUN_DIR [RUN_DIR ...]")
    for path in sys.argv[1:]:
        print(json.dumps(summarize(path), ensure_ascii=False, allow_nan=False))
