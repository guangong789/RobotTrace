"""JSONL 与运行清单适配层；分析引擎本身不读取文件。"""

from pathlib import Path
import json
from .events import Event, EventError, RunEvidence, CollectorEvidence
from .analysis import analyze


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EventError("DUPLICATE_JSON_FIELD")
        result[key] = value
    return result


def load_and_analyze(directory):
    directory = Path(directory)
    # 保留普通入口对旧清单的兼容解析；比较入口单独维持严格校验。
    manifest = json.loads((directory / "run.json").read_text())
    return _analyze_manifest(directory, manifest)


def _analyze_manifest(directory, manifest):
    return _analyze_events(manifest, *_read_events(directory, manifest))


def _read_events(directory, manifest):
    """使用入口已解析的清单，保证上下文来自同一份读取结果。"""
    events, issues = [], []
    for name in manifest["event_files"]:
        path = directory / name
        if not path.is_file():
            issues.append({"reason": "EVENT_FILE_MISSING", "location": name})
            continue
        for line, text in enumerate(path.read_text().splitlines(), 1):
            try:
                events.append(
                    Event.from_dict(json.loads(text, object_pairs_hook=unique_fields))
                )
            except (ValueError, TypeError) as exc:
                reason = (
                    str(exc) if isinstance(exc, EventError) else "MALFORMED_JSON_EVENT"
                )
                issues.append({"reason": reason, "location": f"{name}:{line}"})
    return events, issues


def _analyze_events(manifest, events, issues):
    context = RunEvidence(
        run_id=manifest["run_id"],
        clock_domain=manifest.get("clock_domain"),
        clock_verified=manifest.get("clock_verified") is True,
        expected_sequences=(
            {s: tuple(v) for s, v in manifest["expected_sequences"].items()}
            if manifest.get("expected_sequences") is not None
            else None
        ),
        execution_completed=manifest.get("execution_completed"),
        collectors=(
            tuple(CollectorEvidence(**c) for c in manifest["collectors"])
            if manifest.get("collectors") is not None
            else None
        ),
        thresholds_ns=manifest.get("thresholds_ns", {}),
    )
    result = analyze(events, context, issues)
    result["workload_configuration"] = manifest.get("configuration")
    result["fixture_provenance"] = manifest.get("fixture")
    return result


def save_report(directory, result):
    target = Path(directory) / "analysis.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    return target


def load_comparison_run(directory):
    """严格读取比较所需清单，并用现有核心重新分析原始事件，不信任缓存结果。"""
    directory = Path(directory)
    manifest = json.loads(
        (directory / "run.json").read_text(), object_pairs_hook=unique_fields
    )
    if not isinstance(manifest, dict):
        raise EventError("INVALID_COMPARISON_MANIFEST")
    names = manifest.get("event_files")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise EventError("INVALID_EVENT_FILE_LIST")
    for name in names:
        path = (directory / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(directory.resolve()):
            raise EventError("EVENT_FILE_OUTSIDE_RUN_DIRECTORY")
    events, issues = _read_events(directory, manifest)
    return {
        "manifest": manifest,
        "analysis": _analyze_events(manifest, events, issues),
    }


def save_comparison(path, result):
    """仅在用户显式指定输出位置时保存比较 JSON。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with target.open("x") as output:
        output.write(text)
    return target
