"""只读、按完整身份分解 Consumer 年龄；沿用 V1 的时钟与矛盾判定。"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from robottrace.analysis import statistics
from robottrace.io import _analyze_events, _read_events
from robottrace.inference import read, rows


def attribute(events, analysis, processor_records=()):
    run = analysis["run_id"]
    domain = analysis["clock"]["domain"]
    endpoints = defaultdict(dict)
    for event in events:
        if event.run_id == run:
            endpoints[(run, event.source_id, event.sequence)][
                (event.stage, event.event_type)
            ] = event.timestamp_ns
    records, api_issues = defaultdict(list), Counter()
    for record in processor_records:
        if not isinstance(record, dict) or not all(
            key in record for key in ("run_id", "source_id", "sequence")
        ):
            api_issues["MALFORMED_API_IDENTITY"] += 1
            continue
        key = (record["run_id"], record["source_id"], record["sequence"])
        if (
            not all(isinstance(v, str) and v for v in key[:2])
            or type(key[2]) is not int
        ):
            api_issues["MALFORMED_API_IDENTITY"] += 1
            continue
        records[key].append(record)
    paired, excluded, used = [], [], set()
    for message in analysis["messages"]:
        identity = dict(
            run_id=run, source_id=message["source_id"], sequence=message["sequence"]
        )
        key = (run, message["source_id"], message["sequence"])
        if (
            message["missing"]
            or message["issues"]
            or any(
                value["status"] != "available" for value in message["metrics"].values()
            )
        ):
            excluded.append(
                dict(
                    identity,
                    missing=message["missing"],
                    issues=message["issues"],
                    reasons=sorted(
                        {
                            v["reason"]
                            for v in message["metrics"].values()
                            if v["status"] != "available"
                        }
                    ),
                )
            )
            continue
        point = endpoints[key]
        generated = point[("source", "generate")]
        entry = point[("processor", "callback_entry")]
        publish = point[("processor", "publish_begin")]
        consumed = point[("consumer", "callback_entry")]
        a, b, c, total = (
            entry - generated,
            publish - entry,
            consumed - publish,
            consumed - generated,
        )
        if min(a, b, c) < 0 or a + b + c != total:
            raise ValueError("Invalid paired timestamp decomposition")
        # P95 只从各消息的原始区间求取，不在分位数之间做加减。
        item = dict(
            identity,
            A_ns=a,
            B_ns=b,
            C_ns=c,
            total_ns=total,
            processor_callback_ns=message["metrics"]["processor_callback_ns"][
                "value_ns"
            ],
            api_ns=None,
            api_unavailable_reason=None,
        )
        found = records.get(key, [])
        if len(found) != 1:
            item["api_unavailable_reason"] = "MISSING_OR_AMBIGUOUS_API_RECORD"
        else:
            record = found[0]
            start, end = record.get("api_begin_ns"), record.get("api_end_ns")
            if (
                record.get("valid") is not True
                or record.get("api_returned") is not True
                or record.get("clock_domain") != domain
                or record.get("entry_ns") != entry
                or record.get("generated_ns") != generated
                or type(start) is not int
                or type(end) is not int
                or not entry <= start <= end <= publish
            ):
                item["api_unavailable_reason"] = "INVALID_API_CORRELATION_OR_INTERVAL"
            else:
                item["api_ns"] = end - start
        used.add(key)
        if item["api_unavailable_reason"]:
            api_issues[item["api_unavailable_reason"]] += 1
        paired.append(item)
    api_issues["UNMATCHED_API_RECORD"] += sum(
        len(value) for key, value in records.items() if key not in used
    )
    names = ("A_ns", "B_ns", "C_ns", "total_ns", "api_ns", "processor_callback_ns")
    metrics = {}
    for name in names:
        metrics[name] = dict(
            statistics([item[name] for item in paired if item[name] is not None]),
            unavailable_count=analysis["message_identity_count"]
            - sum(item[name] is not None for item in paired),
        )
    return {
        "run_id": run,
        "identity_count": analysis["message_identity_count"],
        "complete_identity_count": analysis["complete_message_count"],
        "incomplete_identity_count": sum(
            bool(m["missing"]) for m in analysis["messages"]
        ),
        "eligible_identity_count": len(paired),
        "withheld_identity_count": sum(not item["missing"] for item in excluded),
        "clock": analysis["clock"],
        "trace_integrity": analysis["trace_integrity"],
        "collection": analysis["collection"],
        "metrics": metrics,
        "api_record_issues": {k: v for k, v in api_issues.items() if v},
        "messages": paired,
        "excluded": excluded,
    }


def summarize(directory):
    directory = Path(directory)
    manifest = read(directory / "run.json")
    events, issues = _read_events(directory, manifest)
    analysis = _analyze_events(manifest, events, issues)
    path = directory / "processor.results.jsonl"
    result = attribute(events, analysis, rows(path) if path.is_file() else [])
    result["directory"] = str(directory)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: latency_attribution.py RUN_DIR [RUN_DIR ...]")
    for directory in sys.argv[1:]:
        print(json.dumps(summarize(directory), allow_nan=False))
