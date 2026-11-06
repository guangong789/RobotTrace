"""仅接收并返回数据的关联分析；不依赖 ROS、I/O、子进程或展示逻辑。"""

from collections import defaultdict
import math
from .events import Event, REQUIRED, RunEvidence

METRICS = {
    "source_publish_call_ns": (("source", "publish_begin"), ("source", "publish_end")),
    "processor_callback_ns": (
        ("processor", "callback_entry"),
        ("processor", "callback_exit"),
    ),
    "processor_publish_call_ns": (
        ("processor", "publish_begin"),
        ("processor", "publish_end"),
    ),
    "source_to_processor_entry_ns": (
        ("source", "generate"),
        ("processor", "callback_entry"),
    ),
    "processor_to_consumer_entry_ns": (
        ("processor", "publish_begin"),
        ("consumer", "callback_entry"),
    ),
    "consumer_age_ns": (("source", "generate"), ("consumer", "callback_entry")),
    "consumer_callback_ns": (
        ("consumer", "callback_entry"),
        ("consumer", "callback_exit"),
    ),
    "source_to_consumer_exit_ns": (
        ("source", "generate"),
        ("consumer", "callback_exit"),
    ),
}
# 不要求处理回调退出先于消费回调进入：不同进程的执行可以重叠。
ORDER = (
    (("source", "generate"), ("source", "publish_begin")),
    (("source", "publish_begin"), ("source", "publish_end")),
    (("source", "publish_begin"), ("processor", "callback_entry")),
    (("processor", "callback_entry"), ("processor", "publish_begin")),
    (("processor", "publish_begin"), ("processor", "publish_end")),
    (("processor", "publish_end"), ("processor", "callback_exit")),
    (("processor", "publish_begin"), ("consumer", "callback_entry")),
    (("consumer", "callback_entry"), ("consumer", "callback_exit")),
)


def statistics(values):
    """仅基于可用观测，以最近秩法计算整数分位数。"""
    v = sorted(values)
    if not v:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(v),
        "min": v[0],
        "p50": v[math.ceil(0.5 * len(v)) - 1],
        "p95": v[math.ceil(0.95 * len(v)) - 1],
        "max": v[-1],
    }


def sequence_summary(values, expected=None):
    v = sorted(set(values))
    gaps = [
        {"first": a + 1, "last": b - 1, "count": b - a - 1}
        for a, b in zip(v, v[1:])
        if b > a + 1
    ]
    missing = sorted(set(expected) - set(v)) if expected is not None else None
    return {
        "observed_count": len(v),
        "first": v[0] if v else None,
        "last": v[-1] if v else None,
        "internal_missing_ranges": gaps,
        "internal_missing_count": sum(g["count"] for g in gaps),
        "expected_not_observed": missing,
        "missing_prefix_count": (
            sum(x < v[0] for x in missing) if v and missing is not None else None
        ),
        "missing_suffix_count": (
            sum(x > v[-1] for x in missing) if v and missing is not None else None
        ),
    }


def analyze(events: list[Event], evidence: RunEvidence, input_issues=()):
    groups = defaultdict(lambda: defaultdict(list))
    malformed_inputs = [i for i in input_issues if i["reason"] != "EVENT_FILE_MISSING"]
    issues = list(malformed_inputs)
    collector_issues = [i for i in input_issues if i["reason"] == "EVENT_FILE_MISSING"]
    observed_stage_counts = defaultdict(int)
    for name, threshold in evidence.thresholds_ns.items():
        if name not in METRICS or type(threshold) is not int or threshold < 0:
            raise ValueError("INVALID_THRESHOLD")
    for event in events:
        observed_stage_counts[(event.stage, event.process_id)] += 1
        if event.run_id != evidence.run_id:
            issues.append({"reason": "FOREIGN_RUN", "run_id": event.run_id})
            continue
        if evidence.expected_sequences is not None:
            expected = evidence.expected_sequences.get(event.source_id)
            if expected is None or event.sequence not in expected:
                issues.append(
                    {
                        "reason": "UNEXPECTED_MESSAGE_IDENTITY",
                        "source_id": event.source_id,
                        "sequence": event.sequence,
                    }
                )
        groups[(event.source_id, event.sequence)][
            (event.stage, event.event_type)
        ].append(event)
    if evidence.expected_sequences is not None:
        for source, sequences in evidence.expected_sequences.items():
            if len(set(sequences)) != len(sequences) or any(
                type(s) is not int or s < 0 for s in sequences
            ):
                raise ValueError("INVALID_EXPECTED_SEQUENCES")
            for sequence in sequences:
                groups[(source, sequence)]
    collection_status = "unknown"
    if evidence.collectors is not None:
        stages = [c.stage for c in evidence.collectors]
        if sorted(stages) != ["consumer", "processor", "source"]:
            collector_issues.append({"reason": "MISSING_OR_DUPLICATE_COLLECTOR"})
        for c in evidence.collectors:
            if (
                type(c.stored) is not int
                or type(c.dropped) is not int
                or min(c.stored, c.dropped) < 0
            ):
                collector_issues.append(
                    {"reason": "INVALID_COLLECTOR_COUNTER", "stage": c.stage}
                )
            if c.dropped:
                collector_issues.append(
                    {
                        "reason": "EVENT_CAPACITY_EXCEEDED",
                        "stage": c.stage,
                        "dropped": c.dropped,
                    }
                )
            if not c.flushed:
                collector_issues.append(
                    {"reason": "COLLECTOR_NOT_FLUSHED", "stage": c.stage}
                )
            if observed_stage_counts[(c.stage, c.process_id)] != c.stored:
                collector_issues.append(
                    {"reason": "EVENT_FILE_COUNT_MISMATCH", "stage": c.stage}
                )
        known = {(c.stage, c.process_id) for c in evidence.collectors}
        if set(observed_stage_counts) - known:
            issues.append({"reason": "UNEXPECTED_PROCESS_IDENTITY"})
        collection_status = "incomplete" if collector_issues else "complete"
    else:
        collector_issues.append({"reason": "COLLECTION_REPORTS_UNAVAILABLE"})
        if input_issues and any(
            i["reason"] == "EVENT_FILE_MISSING" for i in input_issues
        ):
            collection_status = "incomplete"

    messages, sequences_by_stage = [], defaultdict(list)
    for (source, sequence), observations in sorted(groups.items()):
        local_issues = []
        missing = [f"{s}.{t}" for s, t in REQUIRED if not observations.get((s, t))]
        ambiguous = set()
        for endpoint, found in observations.items():
            if len(found) > 1:
                reason = (
                    "DUPLICATE_EVENT" if len(set(found)) == 1 else "CONFLICTING_EVENT"
                )
                local_issues.append({"reason": reason, "endpoint": ".".join(endpoint)})
                ambiguous.add(endpoint)
        # 该契约中每个阶段对应一个进程；不会静默合并进程身份的变化。
        for stage in ("source", "processor", "consumer"):
            pids = {
                e.process_id
                for key, found in observations.items()
                if key[0] == stage
                for e in found
            }
            if len(pids) > 1:
                local_issues.append(
                    {"reason": "CONFLICTING_PROCESS_IDENTITY", "stage": stage}
                )
                ambiguous.update(key for key in observations if key[0] == stage)

        def endpoints(a, b):
            if not observations.get(a) or not observations.get(b):
                return None, "MISSING_ENDPOINT"
            if a in ambiguous or b in ambiguous:
                return None, "AMBIGUOUS_ENDPOINT"
            x, y = observations[a][0], observations[b][0]
            if not evidence.clock_verified or evidence.clock_domain is None:
                return None, "CLOCK_ASSUMPTION_UNVERIFIED"
            if (
                x.clock_domain != y.clock_domain
                or x.clock_domain != evidence.clock_domain
            ):
                return None, "CLOCK_DOMAIN_MISMATCH"
            return (x, y), None

        contradictory = False
        # 每个已定义指标都有因果方向；直接检查两端，不依赖中间观测。
        # 合并并去重，避免同一端点对重复报告矛盾。
        for a, b in dict.fromkeys((*ORDER, *METRICS.values())):
            pair, reason = endpoints(a, b)
            if pair and pair[1].timestamp_ns < pair[0].timestamp_ns:
                local_issues.append(
                    {
                        "reason": "CONTRADICTORY_EVENT_ORDER",
                        "from": ".".join(a),
                        "to": ".".join(b),
                    }
                )
                contradictory = True
        metrics = {}
        for name, (a, b) in METRICS.items():
            pair, reason = endpoints(a, b)
            if pair and contradictory:
                reason = "CONTRADICTORY_MESSAGE_OBSERVATIONS"
            if reason:
                metrics[name] = {
                    "status": "insufficient_evidence",
                    "value_ns": None,
                    "reason": reason,
                }
            else:
                metrics[name] = {
                    "status": "available",
                    "value_ns": pair[1].timestamp_ns - pair[0].timestamp_ns,
                }
        for stage, kind in (
            ("source", "generate"),
            ("processor", "callback_entry"),
            ("consumer", "callback_entry"),
        ):
            if observations.get((stage, kind)):
                sequences_by_stage[(source, stage)].append(sequence)
        messages.append(
            {
                "source_id": source,
                "sequence": sequence,
                "stages_present": sorted({s for s, _ in observations}),
                "missing": missing,
                "issues": local_issues,
                "metrics": metrics,
            }
        )
        issues.extend(
            dict(i, source_id=source, sequence=sequence) for i in local_issues
        )

    metric_summaries, performance = {}, {}
    for name in METRICS:
        values = [
            m["metrics"][name]["value_ns"]
            for m in messages
            if m["metrics"][name]["status"] == "available"
        ]
        metric_summaries[name] = dict(
            statistics(values), unavailable_count=len(messages) - len(values)
        )
        if name in evidence.thresholds_ns:
            threshold = evidence.thresholds_ns[name]
            exceeded = [
                dict(source_id=m["source_id"], sequence=m["sequence"])
                for m in messages
                if m["metrics"][name]["value_ns"] is not None
                and m["metrics"][name]["value_ns"] > threshold
            ]
            performance[name] = {
                "threshold_ns": threshold,
                "exceeded": exceeded,
                "evidence_complete": evidence.expected_sequences is not None
                and len(values) == len(messages)
                and bool(messages),
                "status": (
                    "threshold_exceeded"
                    if exceeded
                    else (
                        "within_threshold"
                        if evidence.expected_sequences is not None
                        and values
                        and len(values) == len(messages)
                        else "insufficient_evidence"
                    )
                ),
            }
    reasons = {i["reason"] for i in issues}
    trace_status = (
        "malformed"
        if malformed_inputs
        or reasons
        & {"FOREIGN_RUN", "UNEXPECTED_MESSAGE_IDENTITY", "UNEXPECTED_PROCESS_IDENTITY"}
        else (
            "contradictory"
            if reasons
            & {
                "CONFLICTING_EVENT",
                "CONTRADICTORY_EVENT_ORDER",
                "CONFLICTING_PROCESS_IDENTITY",
            }
            else "ambiguous" if "DUPLICATE_EVENT" in reasons else "valid"
        )
    )
    observation_status = (
        "ambiguous"
        if any(m["issues"] for m in messages)
        else (
            "incomplete"
            if not messages or any(m["missing"] for m in messages)
            else (
                "complete"
                if evidence.expected_sequences is not None
                else "insufficient_evidence"
            )
        )
    )
    sequence_results = []
    sources = {s for s, _ in groups}
    for source in sorted(sources):
        expected = (
            evidence.expected_sequences.get(source)
            if evidence.expected_sequences is not None
            else None
        )
        for stage in ("source", "processor", "consumer"):
            sequence_results.append(
                dict(
                    source_id=source,
                    stage=stage,
                    **sequence_summary(sequences_by_stage[(source, stage)], expected),
                )
            )
    return {
        "result_schema_version": 1,
        "run_id": evidence.run_id,
        "event_count": len(events),
        "message_identity_count": len(messages),
        "correlated_message_count": sum(
            len(m["stages_present"]) >= 2 for m in messages
        ),
        "complete_message_count": sum(
            not m["missing"] and not m["issues"] for m in messages
        ),
        "execution": {
            "status": (
                "unknown"
                if evidence.execution_completed is None
                else "completed" if evidence.execution_completed else "failed"
            )
        },
        "collection": {"status": collection_status, "issues": collector_issues},
        "trace_integrity": {"status": trace_status, "issues": issues},
        "observations": {
            "status": observation_status,
            "missing_observation_count": sum(len(m["missing"]) for m in messages),
            "expected_set_available": evidence.expected_sequences is not None,
        },
        "clock": {
            "domain": evidence.clock_domain,
            "verified_by_producer_evidence": evidence.clock_verified,
        },
        "metrics": metric_summaries,
        "performance": performance,
        "sequences": sequence_results,
        "messages": messages,
        "limitations": [
            "Application observation intervals are not pure DDS transport latency.",
            "Missing observations do not identify packet loss or a root cause.",
            "A source manifest defines expected observations, not downstream completion.",
            "Clock verification is trusted run evidence; no multi-host synchronization is implemented.",
        ],
    }
