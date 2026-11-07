"""固定 E4 CPU 归因实验：四次默认配置，再按预声明门槛决定 spinning 对照。"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

from characterize import digest, summarize as api_summary, write

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('session_construction', 'warmup', 'active_inference', 'live_session_idle')


def validate(value, threads, spinning):
    api = api_summary(value, threads)
    resource = value['resource_measurement']
    if (resource['wall_clock'] != 'CLOCK_MONOTONIC'
            or resource['cpu_clock'] != 'CLOCK_PROCESS_CPUTIME_ID'
            or resource['wall_resolution_ns'] <= 0 or resource['cpu_resolution_ns'] <= 0
            or resource['session_creation_success'] is not True
            or resource['total_run_calls'] != 105):
        raise ValueError('Resource clock or Session contract mismatch')
    config = resource['spinning']
    disabled = spinning == 'disabled'
    if (config['key'] != 'session.intra_op.allow_spinning'
            or config['sdk_documented_default'] != '1'
            or config['explicitly_set'] is not disabled
            or config['option_entry_present'] is not disabled
            or config['option_entry_value'] != ('0' if disabled else None)):
        raise ValueError('Spinning configuration/readback mismatch')
    phases = resource['phases']
    if [p['name'] for p in phases] != list(NAMES):
        raise ValueError('Missing or reordered phase')
    previous_wall = resource['fixture_load']['wall_end_ns']
    previous_cpu = resource['fixture_load']['cpu_end_ns']
    for phase, before, after, valid, alive in zip(phases, (0, 0, 5, 105), (0, 5, 105, 105), (0, 5, 100, 0), (False, True, True, True)):
        if (not previous_wall <= phase['wall_begin_ns'] < phase['wall_end_ns']
                or not previous_cpu <= phase['cpu_begin_ns'] <= phase['cpu_end_ns']
                or phase['wall_ns'] != phase['wall_end_ns'] - phase['wall_begin_ns']
                or phase['cpu_ns'] != phase['cpu_end_ns'] - phase['cpu_begin_ns']
                or phase['run_calls_before'] != before or phase['run_calls_after'] != after
                or phase['valid_outputs'] != valid
                or phase['session_alive_before'] is not alive
                or phase['session_alive_after'] is not True
                or not math.isclose(phase['cpu_wall_ratio'], phase['cpu_ns'] / phase['wall_ns'], rel_tol=1e-8, abs_tol=1e-12)):
            raise ValueError('Invalid phase delta, sequencing, calls or Session lifetime')
        previous_wall, previous_cpu = phase['wall_end_ns'], phase['cpu_end_ns']
    warmup, active, idle = phases[1:]
    if (not warmup['wall_begin_ns'] <= value['warmup_begin_ns'] <= value['warmup_end_ns'] <= warmup['wall_end_ns']
            or idle['wall_ns'] < 2_000_000_000
            or any(not active['wall_begin_ns'] <= row['api_begin_ns'] < row['api_end_ns'] <= active['wall_end_ns'] for row in value['observations'])):
        raise ValueError('Warm-up, active API or idle boundary mismatch')
    return dict(api_ns=api, phases=phases, idle_cpu_wall=idle['cpu_wall_ratio'])


def idle_gate(runs, gate):
    a = [r['phases'][3]['cpu_ns'] for r in runs if r['threads'] == 1]
    b = [r['phases'][3]['cpu_ns'] for r in runs if r['threads'] == 4]
    return (len(a) == len(b) == 2 and all(
        cpu >= gate['minimum_b_idle_cpu_ns'] and cpu >= max(a) * gate['minimum_ratio_to_max_a']
        and cpu - max(a) >= gate['minimum_excess_ns'] for cpu in b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    binary, out = args.binary.resolve(strict=True), args.evidence.resolve(strict=True)
    if (out / 'provenance.json').exists():
        parser.error('Existing experiment; refusing overwrite or retry')
    plan = json.loads((out / 'plan.json').read_text())
    assert [(r['threads'], r['spinning']) for r in plan['stage_c']] == [(1, 'default'), (4, 'default'), (4, 'default'), (1, 'default')]
    assert [r['spinning'] for r in plan['stage_d']] == ['default', 'disabled', 'disabled', 'default']
    assert not plan['results']
    fixtures = ROOT / 'models/mobilenet'
    metadata = json.loads((fixtures / 'fixtures.json').read_text())
    for name, expected in metadata['files'].items():
        if digest(fixtures / 'data' / name) != expected:
            raise ValueError('Fixture hash mismatch: ' + name)
    model = fixtures / 'data/mobilenetv2-12.onnx'
    if digest(model) != metadata['model_sha256']:
        raise ValueError('Model hash mismatch')
    sdk = ROOT / '.deps/onnxruntime-linux-x64-1.20.1'
    inputs = [binary, model, fixtures / 'fixtures.json', sdk / 'lib/libonnxruntime.so.1.20.1',
              sdk / 'include/onnxruntime_session_options_config_keys.h']
    inputs += [fixtures / 'data' / n for n in metadata['files']]
    inputs += [Path(__file__).parent / n for n in ('vision.cpp', 'mobilenet.hpp', 'resource_cost.py', 'test_resource_phases.py', 'RESOURCE_COST.md', 'CMakeLists.txt')]
    frozen = {str(p): digest(p) for p in inputs}
    provenance = dict(
        files_sha256=frozen, runtime_image_id=os.environ['ROBOTTRACE_IMAGE_ID'],
        compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0],
        build_cache=[s for s in (binary.parent / 'CMakeCache.txt').read_text().splitlines() if s.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_COMPILER:', 'CMAKE_CXX_FLAGS:', 'CMAKE_CXX_FLAGS_RELEASE:'))],
        sdk_version=(sdk / 'VERSION_NUMBER').read_text().strip(),
        uname=list(platform.uname()), os_release=Path('/etc/os-release').read_text(),
        cpu_model=next(s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
        reported_logical_cpus=os.cpu_count(), allowed_logical_cpus=sorted(os.sched_getaffinity(0)),
        cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
        relevant_environment={k:v for k,v in os.environ.items() if k.startswith(('ORT_', 'OMP_', 'MKL_', 'OPENBLAS_'))},
        git_branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], text=True),
        fixtures=metadata, phase_order=list(NAMES),
        spinning_reference='https://github.com/microsoft/onnxruntime/blob/v1.20.1/onnxruntime/core/session/inference_session.cc#L450-L463',
        attribution_limits=['Session phase brackets make_unique<Ort::Session>, includes model-file loading inside ORT; Env/options and model-contract checks outside.', 'Fixture loading separately measured before Session.', 'Warm-up/active CPU and wall include tensor wrapping, output copying and full numerical checks, outside individual API timing.', 'Idle is a single immediate post-active 2s wait with Session alive; no inference, I/O or validation inside.', 'Process CPU includes all process threads; no per-worker, per-core or energy attribution.', 'CPU/wall clocks read separately with CPU interval inside wall interval; resolution is not accuracy.', 'Phase order fixed; idle may include a finite post-inference spin burst rather than sustained idle activity.'])
    write(out / 'provenance.json', provenance)

    def run(stage, item):
        if any(digest(p) != expected for p, expected in frozen.items()):
            raise ValueError('Frozen software or assets changed; stopping')
        name = stage + '_' + item['label']
        command = [str(binary), str(fixtures), '--intra-op-num-threads', str(item['threads']), '--resource-phases', '--intra-op-spinning', item['spinning']]
        record = dict(stage=stage, **item, command=command, raw_file=name + '.json', stderr_file=name + '.stderr.log')
        plan['results'].append(record)
        write(out / 'plan.json', plan)
        print('START',name,flush=True)
        try:
            process = subprocess.run(command, text=True, capture_output=True, timeout=60)
            stdout, stderr = process.stdout, process.stderr
            record['exit_status'] = process.returncode
        except subprocess.TimeoutExpired as error:
            stdout, stderr = (error.stdout or b'').decode(), (error.stderr or b'').decode()
            record.update(exit_status=None, error='Timeout; process terminated, no retry')
        (out / record['raw_file']).write_text(stdout)
        (out / record['stderr_file']).write_text(stderr)
        try:
            if record['exit_status'] != 0:
                raise ValueError('Nonzero or missing exit status')
            record.update(validate(json.loads(stdout), item['threads'], item['spinning']), correctness='passed')
        except (ValueError, TypeError, KeyError) as error:
            record['validation_error'] = str(error)
            write(out / 'plan.json', plan)
            raise
        write(out / 'plan.json', plan)
        print('DONE',name,'idle CPU ns',record['phases'][3]['cpu_ns'],'ratio',record['idle_cpu_wall'],flush=True)

    for item in plan['stage_c']:
        run('C', item)
    plan['stage_d_gate_passed'] = idle_gate(plan['results'], plan['stage_d_gate'])
    plan['stage_d_decision'] = 'Reproducible material post-active idle CPU overhead; verified key supports no-spin comparison' if plan['stage_d_gate_passed'] else 'Skip: default four-thread idle CPU did not pass the predeclared materiality/repetition gate'
    write(out / 'plan.json', plan)
    print('STAGE D GATE',plan['stage_d_gate_passed'],flush=True)
    if plan['stage_d_gate_passed']:
        for item in plan['stage_d']:
            run('D', item)
    plan['completed'] = True
    write(out / 'plan.json', plan)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
