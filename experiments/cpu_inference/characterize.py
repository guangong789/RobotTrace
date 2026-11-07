"""固定六进程 CPU 线程对照；只运行 standalone，不调用 ROS2 或比较引擎。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'models/mobilenet'
SEQUENCE = (1, 2, 4, 4, 2, 1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(values):
    ordered = sorted(values)
    return dict(count=len(ordered), min=ordered[0], max=ordered[-1],
                p50=ordered[math.ceil(len(ordered) * .5) - 1],
                p95=ordered[math.ceil(len(ordered) * .95) - 1],
                mean=sum(ordered) / len(ordered))


def summarize(value, threads):
    """只接受配置、warm-up、数值和原始观测均完整的进程结果。"""
    if not isinstance(value, dict):
        raise ValueError('Non-object process result')
    settings = dict(status='passed', runtime='1.20.1', provider='CPUExecutionProvider',
                    intra_op_num_threads=threads, inter_op_num_threads=1,
                    execution_mode='sequential', graph_optimization='basic',
                    cpu_arena=True, clock='CLOCK_MONOTONIC', session_initializations=1,
                    warmup_count=5, expected_warmup_count=5,
                    expected_measured_iterations=100, sample_count=2,
                    input_elements=150528, output_elements=1000, atol=1e-4, rtol=1e-4)
    if any(value.get(k) != v for k, v in settings.items()):
        raise ValueError('Failed run or configuration/numerical contract mismatch')
    if not 0 < value['warmup_begin_ns'] < value['warmup_end_ns']:
        raise ValueError('Invalid warm-up interval')
    rows = value['observations']
    if len(rows) != 100:
        raise ValueError('Incomplete measured observations')
    previous = value['warmup_end_ns']
    for i, row in enumerate(rows, 1):
        if (row['iteration'] != i or row['sample_id'] != (5 + i - 1) % 2
                or row['api_returned'] is not True
                or row['output_available'] is not True
                or row['numerically_correct'] is not True
                or any(type(row[k]) is not int for k in
                       ('api_begin_ns', 'api_end_ns', 'duration_ns'))
                or not previous <= row['api_begin_ns'] < row['api_end_ns']
                or row['duration_ns'] != row['api_end_ns'] - row['api_begin_ns']):
            raise ValueError('Invalid identity, numerical result or measured interval')
        previous = row['api_end_ns']
    result = statistics([r['duration_ns'] for r in rows])
    if any(value['api_ns'].get(k) != result[k] for k in ('count', 'min', 'max', 'p50', 'p95')):
        raise ValueError('Aggregate differs from raw observations')
    return result


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    binary, out = args.binary.resolve(strict=True), args.evidence.resolve(strict=True)
    if (out / 'provenance.json').exists() or (out / 'results.json').exists():
        parser.error('Evidence already contains an experiment; refusing overwrite')
    metadata = json.loads((FIXTURES / 'fixtures.json').read_text())
    for name, expected in metadata['files'].items():
        if digest(FIXTURES / 'data' / name) != expected:
            raise ValueError('Fixture hash mismatch: ' + name)
    model = FIXTURES / 'data/mobilenetv2-12.onnx'
    if digest(model) != metadata['model_sha256']:
        raise ValueError('Model hash mismatch')
    inputs = [binary, model, FIXTURES / 'fixtures.json',
              ROOT / '.deps/onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so.1.20.1']
    inputs += [FIXTURES / 'data' / name for name in metadata['files']]
    inputs += [ROOT / 'experiments/cpu_inference' / name for name in
               ('vision.cpp', 'mobilenet.hpp', 'CMakeLists.txt', 'characterize.py',
                'test_characterization.py', 'README.md')]
    frozen = {str(p): digest(p) for p in inputs}
    cpuinfo = Path('/proc/cpuinfo').read_text()
    cpu_model = next((line.split(':', 1)[1].strip() for line in cpuinfo.splitlines()
                      if line.startswith('model name')), 'unreported')
    compiler = subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]
    build_cache = [line for line in (binary.parent / 'CMakeCache.txt').read_text().splitlines()
                   if line.startswith(('CMAKE_CXX_COMPILER:', 'CMAKE_CXX_FLAGS:',
                                       'CMAKE_CXX_FLAGS_RELEASE:', 'CMAKE_BUILD_TYPE:'))]
    provenance = dict(environment='Humble container on WSL2 host',
                      runtime_image_id=os.environ['ROBOTTRACE_IMAGE_ID'],
                      uname=list(platform.uname()), os_release=Path('/etc/os-release').read_text(),
                      compiler=compiler, build_cache=build_cache, cpu_model=cpu_model,
                      onnxruntime_sdk_version=(ROOT / '.deps/onnxruntime-linux-x64-1.20.1/VERSION_NUMBER').read_text().strip(),
                      relevant_environment={key: os.environ.get(key) for key in
                                            ('OMP_NUM_THREADS', 'OMP_WAIT_POLICY',
                                             'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                             'LD_LIBRARY_PATH')},
                      reported_logical_cpus=os.cpu_count(),
                      allowed_logical_cpus=sorted(os.sched_getaffinity(0)),
                      cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text()
                      if Path('/sys/fs/cgroup/cpu.max').exists() else None,
                      fixture_metadata=metadata, files_sha256=frozen,
                      git_branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
                      git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      git_status=subprocess.check_output(['git', 'status', '--short'], text=True),
                      sequence=list(SEQUENCE), warmup_count=5, measured_iterations=100,
                      percentile_method='nearest rank, ceil(q*N)',
                      unmeasured_or_uncontrolled=['CPU frequency', 'other host activity',
                                                 'thermal state', 'CPU placement'])
    write(out / 'provenance.json', provenance)
    results = dict(sequence=list(SEQUENCE), runs=[], completed=False,
                   decision='pending analysis; no automatic performance verdict')
    write(out / 'results.json', results)
    for index, threads in enumerate(SEQUENCE, 1):
        if any(digest(Path(p)) != expected for p, expected in frozen.items()):
            raise ValueError('Frozen software/model/fixture changed; stopping')
        command = [str(binary), str(FIXTURES), '--intra-op-num-threads', str(threads)]
        record = dict(index=index, intra_op_num_threads=threads, command=command,
                      started_monotonic_ns=time.monotonic_ns(),
                      loadavg_before=Path('/proc/loadavg').read_text().strip())
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=60)
            stdout, stderr = process.stdout, process.stderr
            record['exit_status'] = process.returncode
        except subprocess.TimeoutExpired as error:
            stdout = (error.stdout or b'').decode()
            stderr = (error.stderr or b'').decode()
            record.update(exit_status=None, error='Process exceeded 60s; terminated, no retry')
        name = f'run_{index:02d}_threads_{threads}'
        (out / (name + '.json')).write_text(stdout)
        (out / (name + '.stderr.log')).write_text(stderr)
        record.update(raw_file=name + '.json', stderr_file=name + '.stderr.log',
                      ended_monotonic_ns=time.monotonic_ns(),
                      loadavg_after=Path('/proc/loadavg').read_text().strip())
        try:
            if record['exit_status'] != 0:
                raise ValueError('Nonzero or missing process exit status')
            value = json.loads(stdout)
            record.update(api_ns=summarize(value, threads), correctness='passed',
                          process_peak_rss_kib=value['process_peak_rss_kib'],
                          process_user_us=value['process_user_us'],
                          process_system_us=value['process_system_us'])
        except (ValueError, KeyError, TypeError) as error:
            record.update(correctness='unverified', validation_error=str(error))
        results['runs'].append(record)
        results['completed'] = len(results['runs']) == 6 and 'validation_error' not in record
        write(out / 'results.json', results)
        print(json.dumps(record, allow_nan=False), flush=True)
        if 'validation_error' in record:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
