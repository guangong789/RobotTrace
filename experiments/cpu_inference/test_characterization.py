"""Standalone 配置、完整输出、原始观测及失败处理契约；不启动 ROS2。"""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

from characterize import summarize

BINARY, FIXTURES = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
del sys.argv[1:3]


def invoke(root, *options):
    return subprocess.run([str(BINARY), str(root), *options],
                          capture_output=True, text=True, timeout=60)


class CharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for threads in (1, 2, 4):
            process = invoke(FIXTURES, '--intra-op-num-threads', str(threads))
            if process.returncode != 0:
                raise AssertionError(process.stderr)
            cls.results[threads] = json.loads(process.stdout)

    def test_all_thread_candidates_preserve_complete_numeric_and_timing_contract(self):
        for threads, value in self.results.items():
            with self.subTest(threads=threads):
                measured = summarize(value, threads)
                self.assertEqual(measured['count'], 100)
                self.assertEqual([r['sample_id'] for r in value['observations']].count(0), 50)
                self.assertEqual([r['sample_id'] for r in value['observations']].count(1), 50)
                self.assertEqual(value['warmup_count'], 5)
                self.assertGreaterEqual(value['observations'][0]['api_begin_ns'], value['warmup_end_ns'])
                self.assertEqual(value['session_initializations'], 1)

    def test_invalid_parameters_are_rejected_without_measurement(self):
        for invalid in ('0', '-1', '3', 'x', '1.5', '2junk', '2147483647', '2147483648', ''):
            with self.subTest(invalid=invalid):
                process = invoke(FIXTURES, '--intra-op-num-threads', invalid)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn('positive integer', process.stderr)
                value = json.loads(process.stdout)
                self.assertEqual(value['status'], 'failed')
                self.assertEqual(value['observations'], [])
                self.assertEqual(value['api_ns']['count'], 0)

    def test_unknown_or_missing_option_arguments_are_rejected(self):
        for options in (('--unknown', '2'), ('--intra-op-num-threads',)):
            process = invoke(FIXTURES, *options)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('Usage:', process.stderr)

    def test_incomplete_failed_or_contradictory_observations_are_not_complete_runs(self):
        for fault in ('missing', 'warmup', 'identity', 'numerical', 'interval', 'config', 'failed'):
            with self.subTest(fault=fault):
                value = deepcopy(self.results[1])
                if fault == 'missing':
                    value['observations'].pop()
                elif fault == 'warmup':
                    value['observations'][0]['api_begin_ns'] = value['warmup_begin_ns']
                elif fault == 'identity':
                    value['observations'][0]['sample_id'] = 0
                elif fault == 'numerical':
                    value['observations'][0]['numerically_correct'] = False
                elif fault == 'interval':
                    value['observations'][0]['duration_ns'] += 1
                elif fault == 'config':
                    value['inter_op_num_threads'] = 2
                else:
                    value['status'] = 'failed'
                with self.assertRaises(ValueError):
                    summarize(value, 1)

    def test_bad_tensor_golden_nonfinite_and_missing_model_fail_honestly(self):
        for fault in ('input_size', 'input_nonfinite', 'golden', 'golden_nonfinite', 'model'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data = root / 'data'
                data.mkdir()
                for name in ('input-0.f32', 'input-1.f32', 'expected-0.f32', 'expected-1.f32'):
                    shutil.copyfile(FIXTURES / 'data' / name, data / name)
                if fault != 'model':
                    (data / 'mobilenetv2-12.onnx').symlink_to(FIXTURES / 'data/mobilenetv2-12.onnx')
                if fault == 'input_size':
                    (data / 'input-0.f32').write_bytes(b'bad')
                elif fault in ('input_nonfinite', 'golden_nonfinite', 'golden'):
                    name = 'input-0.f32' if fault == 'input_nonfinite' else 'expected-0.f32'
                    payload = (data / name).read_bytes()
                    changed = float('nan') if 'nonfinite' in fault else struct.unpack('<f', payload[:4])[0] + 10
                    (data / name).write_bytes(struct.pack('<f', changed) + payload[4:])
                process = invoke(root)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn('FAIL:', process.stderr)
                value = json.loads(process.stdout)
                self.assertEqual(value['status'], 'failed')
                self.assertEqual(value['observations'], [])
                self.assertEqual(value['api_ns']['count'], 0)
                self.assertIsNone(value['api_ns']['p95'])


if __name__ == '__main__':
    unittest.main()
