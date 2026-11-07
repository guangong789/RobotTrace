"""只读汇总 E3 ROS2 尝试；启动失败、可用延迟与结果覆盖分开报告。"""

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.cpu_inference.load_summary import summarize as load_summary
from experiments.cpu_inference.latency_attribution import summarize as latency_summary
from robottrace.inference import (
    ATOL, RTOL, fixture_data, read, rows, verify_processor_configuration,
)


def check_results(records, manifest, role, expected):
    """校验所有实际存在的输出；缺失身份保持缺失，不要求序号连续。"""
    seen, valid, invalid = set(), 0, 0
    for record in records:
        seq = record.get('sequence')
        identity_ok = (
            type(seq) is int and 1 <= seq <= manifest['configuration']['count']
            and record.get('run_id') == manifest['run_id']
            and record.get('source_id') == 'source:1' and seq not in seen
        )
        if type(seq) is int:
            seen.add(seq)
        sample = (seq - 1) % 2 if identity_ok else None
        logits = record.get('logits')
        correct = (identity_ok and record.get('sample_id') == sample
                   and record.get('valid') is True and isinstance(logits, list)
                   and len(logits) == 1000)
        if correct:
            correct = all(type(a) in (int, float) and math.isfinite(a)
                          and abs(a - b) <= ATOL + RTOL * abs(b)
                          for a, b in zip(logits, expected[sample]))
        if role == 'processor':
            correct = (correct and record.get('input_payload_verified') is True
                       and record.get('api_returned') is True)
        valid += bool(correct)
        invalid += not correct
    return dict(observed=len(records), valid=valid, invalid=invalid)


def summarize(directory):
    directory = Path(directory)
    manifest = read(directory / 'run.json')
    load = load_summary(directory)
    latency = latency_summary(directory)
    requested = manifest['configuration']['intra_op_threads']
    applied, configuration_error = None, None
    try:
        applied = verify_processor_configuration(manifest.get('processor_configuration'), requested)
        if not manifest.get('processor_readiness_configuration_verified'):
            raise ValueError('No verified Processor readiness configuration')
        if not manifest.get('processor_status_configuration_verified'):
            raise ValueError('No verified Processor final configuration')
    except ValueError as error:
        configuration_error = str(error)
    expected, _ = fixture_data()
    observed, missing_files = {}, []
    for role in ('processor', 'consumer'):
        path = directory / (role + '.results.jsonl')
        if not path.is_file():
            missing_files.append(role)
        observed[role] = check_results(rows(path) if path.is_file() else [], manifest, role, expected)
    numeric = ('failed' if any(v['invalid'] for v in observed.values()) else
               'insufficient_evidence' if missing_files else
               'passed_for_observed' if all(v['observed'] for v in observed.values()) else
               'not_exercised')
    source = load['counts']['source']['published']
    consumer = observed['consumer']['valid']
    processor = observed['processor']['valid']
    frozen = manifest.get('readiness_diagnostics', {}).get('pre_cleanup', {})
    snapshot = frozen.get('last_observation') or {}
    states = snapshot.get('roles', {})
    matched = {}
    for label, role, key in (
        ('source_subscribers', 'source', 'matched_subscribers'),
        ('processor_publishers', 'processor', 'matched_publishers'),
        ('processor_subscribers', 'processor', 'matched_subscribers'),
        ('consumer_publishers', 'consumer', 'matched_publishers'),
    ):
        state = states.get(role)
        matched[label] = state.get(key) if isinstance(state, dict) else None
    qos = dict(depth=10, reliability='RELIABLE', durability='VOLATILE', history='KEEP_LAST')
    qos_states = [states.get(role, {}).get(key) if isinstance(states.get(role), dict) else None
                  for role, key in (('source', 'output_qos'), ('processor', 'input_qos'),
                                    ('processor', 'output_qos'), ('consumer', 'input_qos'))]
    timing = load['timing'] or {}
    window = timing.get('observation_window_ns')
    metrics = {name: timing.get(name) for name in
               ('inference_api_ns', 'processor_callback_ns', 'consumer_age_ns')}
    metrics.update({name: latency['metrics'][name] for name in ('A_ns', 'B_ns', 'C_ns')})
    valid_trace = (load['clock_verified'] and load['statuses']['collection'] == 'complete'
                   and load['statuses']['trace_integrity'] == 'valid'
                   and all(c['dropped'] == 0 and c['flushed'] for c in (load['collector_reports'] or []))
                   and len(load['collector_reports'] or []) == 3)
    usable = (manifest['execution_completed'] and manifest['startup_readiness_completed']
              and configuration_error is None and valid_trace
              and numeric == 'passed_for_observed' and source == consumer == processor == load['count']
              and load['complete_identities'] == load['count']
              and all(metrics[name] and metrics[name]['count'] == load['count'] for name in metrics))
    return dict(
        directory=str(directory), run_id=manifest['run_id'], requested_threads=requested,
        applied_configuration=applied, configuration_error=configuration_error,
        startup_ready=manifest['startup_readiness_completed'], source_released=(directory / 'release').exists(),
        execution_completed=manifest['execution_completed'], error=manifest.get('execution_error'),
        frozen_matches=matched, frozen_unsatisfied=snapshot.get('unsatisfied'),
        actual_qos_valid=all(v == qos for v in qos_states) if all(v is not None for v in qos_states) else None,
        startup_phases=manifest['phases'], exit_codes=manifest.get('exit_codes'),
        expected=load['count'], requested_hz=load['requested_hz'], source_published=source,
        processor_operations=manifest.get('nodes', {}).get('processor', {}).get('inference_operations'),
        valid_consumer_results=consumer, observed_results=observed,
        numerical_status=numeric, missing_result_files=missing_files,
        result_coverage=consumer / source if source else None,
        complete_identities=load['complete_identities'], incomplete_identities=load['incomplete_identities'],
        trace_coverage=load['complete_identities'] / load['count'] if source else None, missing_endpoints=load['missing_endpoints'],
        statuses=load['statuses'], clock_verified=load['clock_verified'], collectors=load['collector_reports'],
        valid_trace=valid_trace, usable_complete_run=usable, metrics=metrics,
        source_rate=timing.get('source_publish_end'), processor_throughput=timing.get('processor_callback_completion') if source else None,
        consumer_throughput=timing.get('consumer_valid_completion') if source else None,
        observation_window_ns=timing.get('observation_window_ns'),
        processor_window_throughput_hz=processor * 1e9 / window if source and window and window > 0 else None,
        consumer_window_throughput_hz=consumer * 1e9 / window if source and window and window > 0 else None,
        processor_resources=manifest.get('nodes', {}).get('processor', {}).get('process_resources'),
        paired_latency_samples=latency['eligible_identity_count'],
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: e3_summary.py RUN_DIR [RUN_DIR ...]')
    for path in sys.argv[1:]:
        print(json.dumps(summarize(path), allow_nan=False))
