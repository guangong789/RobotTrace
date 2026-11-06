"""展示与命令行适配层；分析模块不会导入本模块。"""

import argparse
import json
import sys
from pathlib import Path
from .io import load_and_analyze, load_comparison_run, save_comparison, save_report
from .comparison import DEFAULT_METRICS, EXPERIMENT_KEYS, compare_runs
from .analysis import METRICS


def display(result):
    print(
        f"run={result['run_id']} events={result['event_count']} correlated={result['correlated_message_count']} complete_messages={result['complete_message_count']}"
    )
    print(
        " ".join(
            f"{axis}={result[axis]['status']}"
            for axis in ("execution", "collection", "trace_integrity", "observations")
        )
    )
    print(
        f"missing_observations={result['observations']['missing_observation_count']} trace_issues={len(result['trace_integrity']['issues'])}"
    )
    for issue in result["collection"]["issues"]:
        print("collection_reason:", issue)
    affected = [m for m in result["messages"] if m["missing"] or m["issues"]]
    for message in affected[:3]:
        print(
            f"evidence {message['source_id']} seq={message['sequence']}: missing={message['missing']} issues={message['issues']}"
        )
    if len(affected) > 3:
        print(f"... {len(affected)-3} more affected identities in JSON")
    for metric in ("processor_callback_ns", "consumer_age_ns"):
        value = result["metrics"][metric]
        print(
            f"{metric}: count={value['count']} p50={value['p50']} p95={value['p95']} max={value['max']} unavailable={value['unavailable_count']}"
        )
    for metric, assessment in result["performance"].items():
        print(
            f"threshold {metric}: {assessment['status']} ({len(assessment['exceeded'])} exceedances; evidence_complete={assessment['evidence_complete']})"
        )
    for row in result["sequences"]:
        print(
            f"sequence {row['source_id']} {row['stage']}: observed={row['observed_count']} internal_missing={row['internal_missing_count']}"
        )


def display_comparison(result):
    """展示描述性差异和证据限制，不将命令成功解释为性能验收通过。"""
    print(f"comparison={result['status']} conditions={result['conditions']['status']}")
    print("Verdicts describe observed P95; statistical significance is not assessed.")
    print("Scope: recorded conditions only; image, hardware and load are unrecorded.")
    for change in result["declared_changes"]:
        print(
            f"declared {change['key']}: {change['baseline']} -> {change['candidate']}"
        )
    for issue in result["conditions"]["issues"]:
        print("condition:", json.dumps(issue, ensure_ascii=False))
    for name, metric in result["metrics"].items():
        baseline, candidate = metric["baseline"], metric["candidate"]
        print(f"{name}: {metric['status']} verdict={metric['verdict']}")
        for side, values in (("baseline", baseline), ("candidate", candidate)):
            print(
                f"  {side}: available={values['available_count']} "
                f"unavailable={values['unavailable_count']} "
                f"expected={values['expected_count']} coverage={values['coverage']} "
                f"p50_ns={values['p50_ns']} p95_ns={values['p95_ns']}"
            )
        for percentile, change in metric["changes"].items():
            print(
                f"  {percentile}: delta_ns={change['delta_ns']} "
                f"percent_change={change['percent_change']}"
            )
        for reason in metric["reasons"]:
            print("  restricted:", json.dumps(reason, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="RobotTrace application-event analysis and evidence-aware comparison."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser(
        "analyze", help="Analyze persisted observations; no ROS required."
    )
    analysis.add_argument("directory", type=Path)
    run = commands.add_parser(
        "run", help="Run a bounded real ROS2 pipeline using the existing image."
    )
    run.add_argument("--scenario", choices=["normal", "slow"], default="normal")
    run.add_argument("--count", type=int, default=100)
    run.add_argument("--rate", type=float, default=100)
    run.add_argument("--depth", type=int, default=10)
    run.add_argument("--work-ms", type=float)
    run.add_argument("--capacity", type=int, default=4096)
    compare = commands.add_parser(
        "compare", help="Compare two persisted runs with explicit evidence checks."
    )
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument(
        "--allow-change", action="append", choices=EXPERIMENT_KEYS, default=[]
    )
    compare.add_argument("--metric", action="append", choices=tuple(METRICS))
    compare.add_argument(
        "--json", action="store_true", help="Print only comparison JSON."
    )
    compare.add_argument("--output", type=Path, help="Explicitly save comparison JSON.")
    args = parser.parse_args()
    if args.command == "compare":
        try:
            result = compare_runs(
                load_comparison_run(args.baseline),
                load_comparison_run(args.candidate),
                allow_changes=args.allow_change,
                metrics=args.metric or DEFAULT_METRICS,
            )
            if args.output:
                save_comparison(args.output, result)
            if args.json:
                print(json.dumps(result, indent=2, allow_nan=False))
            else:
                display_comparison(result)
                if args.output:
                    print("JSON:", args.output)
            # 成功形成比较结果不等于性能验收通过；受限或拒绝判定也返回 0。
            return 0
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            print(f"Comparison input/output error: {exc}", file=sys.stderr)
            return 2
    if args.command == "run":
        from .workload import run_host

        return run_host(args)
    result = load_and_analyze(args.directory)
    display(result)
    print("JSON:", save_report(args.directory, result))
    # 观测不完整和性能判定属于分析结果，不表示分析进程执行失败。
    return 0 if result["trace_integrity"]["status"] == "valid" else 2
