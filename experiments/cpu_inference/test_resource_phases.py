"""资源模式的时钟、边界、Session 生命周期与失败契约；不启动 ROS2。"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from resource_cost import validate, idle_gate

BINARY, FIXTURES = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
del sys.argv[1:3]


class ResourcePhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        process = subprocess.run([str(BINARY), str(FIXTURES), '--resource-phases'],
                                 text=True, capture_output=True, timeout=60)
        if process.returncode != 0:
            raise AssertionError(process.stderr)
        cls.value = json.loads(process.stdout)

    def test_real_default_session_has_four_complete_phases_and_full_correct_outputs(self):
        result = validate(self.value, 1, 'default')
        self.assertEqual(result['api_ns']['count'], 100)
        self.assertEqual([p['valid_outputs'] for p in result['phases']], [0, 5, 100, 0])
        idle = result['phases'][3]
        self.assertTrue(idle['session_alive_before'] and idle['session_alive_after'])
        self.assertEqual(idle['run_calls_before'], idle['run_calls_after'])
        self.assertGreaterEqual(idle['wall_ns'], 2_000_000_000)

    def test_wrong_clock_delta_order_idle_calls_or_lifetime_cannot_pass(self):
        for fault in ('clock', 'resolution', 'negative', 'order', 'idle_run', 'lifetime', 'warmup', 'missing_sample', 'wrong_spinning'):
            value = deepcopy(self.value)
            resource = value['resource_measurement']
            if fault == 'clock':
                resource['cpu_clock'] = 'CLOCK_THREAD_CPUTIME_ID'
            elif fault == 'resolution':
                resource['cpu_resolution_ns'] = 0
            elif fault == 'negative':
                resource['phases'][1]['cpu_end_ns'] = resource['phases'][1]['cpu_begin_ns'] - 1
            elif fault == 'order':
                resource['phases'].reverse()
            elif fault == 'idle_run':
                resource['phases'][3]['run_calls_after'] += 1
            elif fault == 'lifetime':
                resource['phases'][3]['session_alive_after'] = False
            elif fault == 'warmup':
                value['warmup_count'] = 6
            elif fault == 'missing_sample':
                value['observations'].pop()
            else:
                resource['spinning']['explicitly_set'] = True
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                validate(value, 1, 'default')

    def test_default_spinning_is_unset_not_an_explicit_one(self):
        spin = self.value['resource_measurement']['spinning']
        self.assertFalse(spin['explicitly_set'])
        self.assertFalse(spin['option_entry_present'])
        self.assertIsNone(spin['option_entry_value'])
        with self.assertRaises(ValueError):
            validate(self.value, 1, 'disabled')

    def test_invalid_or_out_of_design_options_do_not_create_a_session(self):
        cases = [('--resource-phases', '--intra-op-num-threads', '2'),
                 ('--intra-op-spinning', 'disabled'),
                 ('--resource-phases', '--intra-op-spinning', 'invalid'),
                 ('--resource-phases', '--intra-op-spinning', 'disabled'),
                 ('--resource-phases', '--intra-op-num-threads', '0')]
        for options in cases:
            process = subprocess.run([str(BINARY), str(FIXTURES), *options],
                                     text=True, capture_output=True, timeout=10)
            with self.subTest(options=options):
                self.assertNotEqual(process.returncode, 0)
                value = json.loads(process.stdout)
                self.assertEqual(value['status'], 'failed')
                self.assertEqual(value['session_initializations'], 0)
                self.assertFalse(value['observations'])

    def test_conditional_gate_requires_two_material_repeated_idle_observations(self):
        gate = dict(minimum_b_idle_cpu_ns=20_000_000, minimum_ratio_to_max_a=10,
                    minimum_excess_ns=10_000_000)
        def record(threads, cpu):
            return dict(threads=threads, phases=[{}, {}, {}, dict(cpu_ns=cpu)])
        runs = [record(1, 100_000), record(4, 50_000_000),
                record(4, 60_000_000), record(1, 200_000)]
        self.assertTrue(idle_gate(runs, gate))
        runs[2]['phases'][3]['cpu_ns'] = 5_000_000
        self.assertFalse(idle_gate(runs, gate))
        self.assertFalse(idle_gate(runs[:3], gate))


if __name__ == '__main__':
    unittest.main()
