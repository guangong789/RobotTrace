"""基于结构化结果比较运行；不导入 ROS、文件 I/O、运行器或展示代码。"""

import json
import math
import re

from .analysis import METRICS

DEFAULT_METRICS = ("processor_callback_ns", "consumer_age_ns")
EXPERIMENT_KEYS = ("processor_work_ms",)
CONFIG_KEYS = (
    "count",
    "nominal_rate_hz",
    "depth",
    "processor_work_ms",
    "drain_s",
    "collector_capacity_per_stage",
    "reliability",
    "durability",
    "history",
    "executor",
    "domain_id",
    "layout",
)
LAYOUT = "three independent C++ processes in one ephemeral ROS2 container"
PACKAGES = {
    "ros-humble-rclcpp",
    "ros-humble-rmw-fastrtps-cpp",
    "ros-humble-std-msgs",
    "ros-humble-std-srvs",
}
CLOCK = re.compile(
    r"linux:CLOCK_MONOTONIC:([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}):offset=0"
)


class ComparisonError(ValueError):
    pass


def _issue(code, status="INSUFFICIENT_EVIDENCE", **details):
    return {"code": code, "status": status, **details}


def _status(issues):
    if any(i["status"] == "NOT_COMPARABLE" for i in issues):
        return "NOT_COMPARABLE"
    return "INSUFFICIENT_EVIDENCE" if issues else "COMPARABLE"


def _integer(value):
    return type(value) is int and value >= 0


def _zero_offsets(value):
    if not isinstance(value, str):
        return False
    rows = [line.split() for line in value.splitlines() if line.strip()]
    try:
        return (
            len(rows) == 2
            and {row[0] for row in rows} == {"monotonic", "boottime"}
            and all(len(row) == 3 and int(row[1]) == int(row[2]) == 0 for row in rows)
        )
    except (ValueError, IndexError):
        return False


def _clock_issues(manifest, result, side):
    issues = []
    clock = result.get("clock", {})
    if not isinstance(clock, dict):
        raise ComparisonError("MALFORMED_CLOCK_RESULT")
    if (
        manifest.get("clock_verified") is not True
        or clock.get("verified_by_producer_evidence") is not True
    ):
        issues.append(_issue("CLOCK_UNVERIFIED", run=side))
    domain = manifest.get("clock_domain")
    if domain is None or clock.get("domain") is None:
        issues.append(_issue("MISSING_CLOCK_DOMAIN", run=side))
        return issues
    match = CLOCK.fullmatch(domain) if isinstance(domain, str) else None
    if not match or clock["domain"] != domain:
        return issues + [
            _issue("UNSUPPORTED_OR_CONFLICTING_CLOCK", "NOT_COMPARABLE", run=side)
        ]
    boot = match[1]
    host = manifest.get("host_clock")
    nodes = manifest.get("nodes")
    if not isinstance(host, dict) or not isinstance(nodes, dict):
        return issues + [_issue("MISSING_CLOCK_PROVENANCE", run=side)]
    if host.get("boot_id") != boot or not _zero_offsets(host.get("offsets")):
        issues.append(_issue("CLOCK_EVIDENCE_CONFLICT", "NOT_COMPARABLE", run=side))
    for stage in ("source", "processor", "consumer"):
        node = nodes.get(stage)
        if not isinstance(node, dict):
            issues.append(_issue("MISSING_NODE_CLOCK", run=side, stage=stage))
        elif (
            node.get("boot_id") != boot
            or node.get("clock_domain") != domain
            or not _zero_offsets(node.get("time_offsets"))
        ):
            issues.append(
                _issue(
                    "CLOCK_EVIDENCE_CONFLICT", "NOT_COMPARABLE", run=side, stage=stage
                )
            )
    return issues


def _run(value, side):
    if not isinstance(value, dict):
        raise ComparisonError("RUN_INPUT_MUST_BE_A_MAPPING")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ComparisonError("RUN_INPUT_MUST_BE_FINITE_JSON_DATA") from exc
    manifest, result = value.get("manifest"), value.get("analysis")
    if not isinstance(manifest, dict) or not isinstance(result, dict):
        raise ComparisonError("MISSING_MANIFEST_OR_ANALYSIS")
    for data, key in ((manifest, "schema_version"), (result, "result_schema_version")):
        if type(data.get(key)) is not int or data[key] != 1:
            raise ComparisonError("UNSUPPORTED_" + key.upper())
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id or result.get("run_id") != run_id:
        raise ComparisonError("RUN_ID_CONFLICT")
    issues = []
    config = manifest.get("configuration")
    if config is None:
        config = {}
    if not isinstance(config, dict) or any(not isinstance(k, str) for k in config):
        raise ComparisonError("MALFORMED_CONFIGURATION")
    if "workload_configuration" in result and result[
        "workload_configuration"
    ] != manifest.get("configuration"):
        raise ComparisonError("ANALYSIS_CONFIGURATION_CONFLICT")
    numeric = {
        "count",
        "nominal_rate_hz",
        "depth",
        "processor_work_ms",
        "drain_s",
        "collector_capacity_per_stage",
        "domain_id",
    }
    for key in CONFIG_KEYS:
        if key not in config:
            issues.append(_issue("MISSING_CONFIGURATION", run=side, key=key))
        elif key in numeric:
            v = config[key]
            if (
                type(v) not in (int, float)
                or (type(v) is float and not math.isfinite(v))
                or v < 0
            ):
                raise ComparisonError("INVALID_CONFIGURATION_" + key.upper())
            if (
                key in {"count", "depth", "collector_capacity_per_stage", "domain_id"}
                and type(v) is not int
            ):
                raise ComparisonError("INVALID_CONFIGURATION_" + key.upper())
            if key in {"count", "depth", "nominal_rate_hz"} and v == 0:
                raise ComparisonError("INVALID_CONFIGURATION_" + key.upper())
        elif not isinstance(config[key], str) or not config[key]:
            raise ComparisonError("INVALID_CONFIGURATION_" + key.upper())
    scenario = manifest.get("scenario")
    if scenario is None:
        issues.append(_issue("MISSING_WORKLOAD_TYPE", run=side))
    elif scenario not in ("normal", "slow"):
        issues.append(_issue("UNSUPPORTED_WORKLOAD_TYPE", "NOT_COMPARABLE", run=side))
    for key, supported in (
        ("layout", LAYOUT),
        ("executor", "SingleThreadedExecutor"),
        ("reliability", "RELIABLE"),
        ("durability", "VOLATILE"),
        ("history", "KEEP_LAST"),
    ):
        if key in config and config[key] != supported:
            issues.append(
                _issue(
                    "UNSUPPORTED_PIPELINE_SEMANTICS",
                    "NOT_COMPARABLE",
                    run=side,
                    key=key,
                )
            )
    expected = manifest.get("expected_sequences")
    identities = None
    if expected is None:
        issues.append(_issue("MISSING_EXPECTED_IDENTITIES", run=side))
    else:
        if not isinstance(expected, dict) or not expected:
            raise ComparisonError("INVALID_EXPECTED_IDENTITIES")
        identities = set()
        for source, seqs in expected.items():
            if (
                not isinstance(source, str)
                or not source
                or not isinstance(seqs, (list, tuple))
            ):
                raise ComparisonError("INVALID_EXPECTED_IDENTITIES")
            if (
                not seqs
                or any(not _integer(s) for s in seqs)
                or len(set(seqs)) != len(seqs)
            ):
                raise ComparisonError("INVALID_EXPECTED_IDENTITIES")
            identities.update((source, s) for s in seqs)
        if "count" in config and len(identities) != config["count"]:
            issues.append(_issue("EXPECTED_COUNT_CONFLICT", "NOT_COMPARABLE", run=side))
    messages = result.get("messages")
    if not isinstance(messages, list):
        raise ComparisonError("MISSING_MESSAGE_EVIDENCE")
    seen = set()
    for message in messages:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("source_id"), str)
            or not message["source_id"]
            or not _integer(message.get("sequence"))
        ):
            raise ComparisonError("MALFORMED_MESSAGE_EVIDENCE")
        identity = (message["source_id"], message["sequence"])
        if identity in seen:
            raise ComparisonError("DUPLICATE_MESSAGE_IDENTITY")
        seen.add(identity)
    if result.get("message_identity_count") != len(messages):
        raise ComparisonError("MESSAGE_COUNT_CONFLICT")
    if identities is not None:
        if seen - identities:
            issues.append(
                _issue("UNEXPECTED_MESSAGE_IDENTITIES", "NOT_COMPARABLE", run=side)
            )
        if identities - seen:
            issues.append(_issue("MISSING_EXPECTED_MESSAGE_ENTRIES", run=side))
    binary = manifest.get("binary_sha256")
    if binary is not None and (
        not isinstance(binary, str) or not re.fullmatch(r"[0-9a-f]{64}", binary)
    ):
        raise ComparisonError("INVALID_BINARY_SHA256")
    versions = manifest.get("versions")
    packages = {}
    if versions is not None:
        if not isinstance(versions, str):
            raise ComparisonError("MALFORMED_SOFTWARE_VERSIONS")
        for line in versions.splitlines():
            parts = line.split()
            if len(parts) != 2 or parts[0] in packages:
                raise ComparisonError("MALFORMED_SOFTWARE_VERSIONS")
            packages[parts[0]] = parts[1]
    if binary is None or not PACKAGES.issubset(packages):
        issues.append(_issue("MISSING_SOFTWARE_IDENTITY", run=side))
    issues.extend(_clock_issues(manifest, result, side))
    return (
        manifest,
        result,
        config,
        identities,
        {"binary_sha256": binary, "packages": packages},
        issues,
    )


def _quality(manifest, result, side):
    issues = []
    axes = {
        "execution": {"completed", "failed", "unknown"},
        "collection": {"complete", "incomplete", "unknown"},
        "trace_integrity": {
            "valid",
            "ambiguous",
            "contradictory",
            "malformed",
            "unknown",
        },
        "observations": {
            "complete",
            "incomplete",
            "ambiguous",
            "insufficient_evidence",
            "unknown",
        },
    }
    for axis, supported in axes.items():
        data = result.get(axis, {"status": "unknown"})
        if not isinstance(data, dict) or data.get("status") not in supported:
            raise ComparisonError("INVALID_" + axis.upper() + "_STATUS")
        status = data["status"]
        if axis == "execution" and status != "completed":
            issues.append(_issue("EXECUTION_NOT_COMPLETED", run=side))
        if axis == "collection" and (status != "complete" or data.get("issues")):
            issues.append(_issue("COLLECTION_NOT_COMPLETE", run=side))
        if axis == "trace_integrity" and (status != "valid" or data.get("issues")):
            restriction = (
                "INSUFFICIENT_EVIDENCE" if status == "unknown" else "NOT_COMPARABLE"
            )
            issues.append(_issue("TRACE_NOT_VALID", restriction, run=side))
        if axis == "observations" and status == "ambiguous":
            issues.append(_issue("AMBIGUOUS_OBSERVATIONS", "NOT_COMPARABLE", run=side))
        if axis == "observations" and status in ("unknown", "insufficient_evidence"):
            issues.append(_issue("OBSERVATION_EVIDENCE_UNKNOWN", run=side))
    if manifest.get("fixture") is not None:
        issues.append(_issue("DERIVED_EVIDENCE", run=side))
    if result.get("observations", {}).get("expected_set_available") is not True:
        issues.append(_issue("EXPECTED_SET_NOT_VERIFIED", run=side))
    return issues


def _metric(result, expected, name, side):
    issues = []
    summaries = result.get("metrics", {})
    if not isinstance(summaries, dict):
        raise ComparisonError("MALFORMED_METRIC_SUMMARIES")
    summary = summaries.get(name)
    if summary is None:
        summary = {}
    if not isinstance(summary, dict):
        raise ComparisonError("MALFORMED_METRIC_SUMMARY")
    available = set()
    for message in result["messages"]:
        metrics = message.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ComparisonError("MALFORMED_MESSAGE_METRICS")
        metric = metrics.get(name)
        if metric is None:
            issues.append(_issue("MISSING_MESSAGE_METRIC", run=side))
            continue
        if not isinstance(metric, dict) or metric.get("status") not in (
            "available",
            "insufficient_evidence",
        ):
            raise ComparisonError("INVALID_MESSAGE_METRIC")
        if metric["status"] == "available":
            if not _integer(metric.get("value_ns")):
                raise ComparisonError("INVALID_AVAILABLE_DURATION")
            available.add((message["source_id"], message["sequence"]))
        elif metric.get("value_ns") is not None:
            raise ComparisonError("UNAVAILABLE_DURATION_MUST_BE_NULL")
    count, missing = summary.get("count"), summary.get("unavailable_count")
    for value in (count, missing):
        if value is not None and not _integer(value):
            raise ComparisonError("INVALID_METRIC_COUNT")
    if count is not None and count != len(available):
        raise ComparisonError("METRIC_COUNT_CONFLICT")
    if (
        count is not None
        and missing is not None
        and count + missing != len(result["messages"])
    ):
        raise ComparisonError("METRIC_POPULATION_CONFLICT")
    values = [summary.get(k) for k in ("min", "p50", "p95", "max")]
    if any(v is not None and not _integer(v) for v in values):
        raise ComparisonError("INVALID_METRIC_STATISTIC")
    if all(v is not None for v in values) and values != sorted(values):
        raise ComparisonError("INCONSISTENT_METRIC_STATISTICS")
    if count == 0 and any(v is not None for v in values):
        raise ComparisonError("EMPTY_METRIC_HAS_STATISTICS")
    if count is None or missing is None or any(v is None for v in values):
        issues.append(_issue("MISSING_METRIC_STATISTICS", run=side))
    if expected is None or not expected or available != expected:
        issues.append(_issue("INCOMPLETE_METRIC_COVERAGE", run=side))
    if expected is not None and available - expected:
        issues.append(
            _issue("UNEXPECTED_METRIC_IDENTITIES", "NOT_COMPARABLE", run=side)
        )
    covered = len(available & expected) if expected is not None else None
    return {
        "available_count": count,
        "unavailable_count": missing,
        "expected_count": len(expected) if expected is not None else None,
        "covered_expected_count": covered,
        "coverage": covered / len(expected) if expected else None,
        "p50_ns": summary.get("p50"),
        "p95_ns": summary.get("p95"),
    }, issues


def _change(baseline, candidate):
    delta = (
        candidate - baseline if baseline is not None and candidate is not None else None
    )
    percent = None
    if delta is not None and baseline != 0:
        try:
            value = delta / baseline * 100
            percent = value if math.isfinite(value) else None
        except OverflowError:
            pass
    return {"delta_ns": delta, "percent_change": percent}


def compare_runs(baseline, candidate, *, allow_changes=(), metrics=DEFAULT_METRICS):
    """比较显式的 manifest/analysis 数据；判定仅描述已记录条件下的 P95。"""
    if (
        not isinstance(allow_changes, (tuple, list, set, frozenset))
        or any(not isinstance(key, str) for key in allow_changes)
        or set(allow_changes) - set(EXPERIMENT_KEYS)
    ):
        raise ComparisonError("UNSUPPORTED_EXPERIMENT_KEY")
    if (
        not isinstance(metrics, (tuple, list))
        or not metrics
        or any(not isinstance(m, str) or m not in METRICS for m in metrics)
        or len(set(metrics)) != len(metrics)
    ):
        raise ComparisonError("UNSUPPORTED_METRIC_SELECTION")
    runs = [_run(baseline, "baseline"), _run(candidate, "candidate")]
    bm, br, bc, be, bs, bi = runs[0]
    cm, cr, cc, ce, cs, ci = runs[1]
    conditions = bi + ci
    declared = []
    for key in sorted(set(bc) | set(cc)):
        if key in CONFIG_KEYS and (key not in bc or key not in cc):
            continue
        different = bc.get(key) != cc.get(key)
        if key not in CONFIG_KEYS:
            different = json.dumps(bc.get(key), sort_keys=True) != json.dumps(
                cc.get(key), sort_keys=True
            )
        if key not in bc or key not in cc or different:
            change = {"key": key, "baseline": bc.get(key), "candidate": cc.get(key)}
            if key in allow_changes and key in bc and key in cc:
                declared.append(change)
            else:
                conditions.append(
                    _issue("CONFIGURATION_DRIFT", "NOT_COMPARABLE", **change)
                )
    if bm.get("scenario") != cm.get("scenario"):
        change = {
            "key": "scenario",
            "baseline": bm.get("scenario"),
            "candidate": cm.get("scenario"),
        }
        if any(c["key"] == "processor_work_ms" for c in declared):
            declared.append({**change, "explained_by": "processor_work_ms"})
        else:
            conditions.append(_issue("WORKLOAD_TYPE_DRIFT", "NOT_COMPARABLE", **change))
    if be is not None and ce is not None and be != ce:
        conditions.append(_issue("EXPECTED_IDENTITIES_DIFFER", "NOT_COMPARABLE"))
    for key in ("binary_sha256", "packages"):
        changed = bool(bs[key] and cs[key] and bs[key] != cs[key])
        if key == "packages":
            changed = any(
                bs[key][p] != cs[key][p] for p in bs[key].keys() & cs[key].keys()
            )
            if bs[key].keys() != cs[key].keys():
                conditions.append(_issue("SOFTWARE_PACKAGE_SET_UNKNOWN"))
        if changed:
            conditions.append(
                _issue(
                    "SOFTWARE_IDENTITY_DRIFT",
                    "NOT_COMPARABLE",
                    key=key,
                    baseline=bs[key],
                    candidate=cs[key],
                )
            )
    quality = _quality(bm, br, "baseline") + _quality(cm, cr, "candidate")
    compared = {}
    all_issues = list(conditions + quality)
    for name in metrics:
        b, b_issues = _metric(br, be, name, "baseline")
        c, c_issues = _metric(cr, ce, name, "candidate")
        reasons = conditions + quality + b_issues + c_issues
        status = _status(reasons)
        changes = {p: _change(b[p + "_ns"], c[p + "_ns"]) for p in ("p50", "p95")}
        verdict = "WITHHELD"
        if status == "COMPARABLE":
            delta = changes["p95"]["delta_ns"]
            verdict = (
                "OBSERVED_IMPROVEMENT"
                if delta < 0
                else "OBSERVED_REGRESSION" if delta > 0 else "NO_OBSERVED_DIFFERENCE"
            )
        compared[name] = {
            "baseline": b,
            "candidate": c,
            "changes": changes,
            "status": status,
            "verdict": verdict,
            "reasons": reasons,
        }
        all_issues.extend(b_issues + c_issues)
    summaries = {}
    for side, (manifest, result, config, expected, software, _) in zip(
        ("baseline", "candidate"), runs
    ):
        summaries[side] = {
            "run_id": manifest["run_id"],
            "scenario": manifest.get("scenario"),
            "configuration": config,
            "expected_count": len(expected) if expected is not None else None,
            "clock": result.get("clock"),
            "software": software,
            "thresholds_ns": manifest.get("thresholds_ns"),
            **{
                axis: result.get(axis)
                for axis in (
                    "execution",
                    "collection",
                    "trace_integrity",
                    "observations",
                )
            },
        }
    return {
        "comparison_schema_version": 1,
        "status": _status(all_issues),
        "conditions": {"status": _status(conditions), "issues": conditions},
        "evidence_issues": quality,
        "runs": summaries,
        "allowed_changes": sorted(set(allow_changes)),
        "declared_changes": declared,
        "metrics": compared,
        "policy": {
            "verdict_basis": "p95_ns",
            "coverage": "all_expected_identities_per_metric",
            "scope": "recorded_conditions_only",
            "statistical_significance": "not_assessed",
            "threshold_findings": "not_used_for_relative_verdict",
        },
        "limitations": [
            "Observed P95 differences are descriptive, not statistically confirmed improvements.",
            "Docker image identity, hardware, kernel and system load are not recorded by this runner.",
            "Provenance is supplied evidence, not independently authenticated.",
        ],
    }
