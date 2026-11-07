"""部分输出仍须数值正确；缺失结果不能变成成功或补造样本。"""

from copy import deepcopy
import unittest

from experiments.cpu_inference.e3_summary import check_results, summarize
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from test_inference import processor_configuration


class E3SummaryTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(run_id='r', configuration=dict(count=5))
        self.expected = [[0.] * 1000, [1.] * 1000]
        self.records = [dict(run_id='r', source_id='source:1', sequence=seq,
                             sample_id=(seq-1) % 2, valid=True,
                             logits=self.expected[(seq-1) % 2][:]) for seq in (1, 4)]

    def test_noncontiguous_partial_results_are_checked_without_inventing_missing_rows(self):
        result = check_results(self.records, self.manifest, 'consumer', self.expected)
        self.assertEqual(result, dict(observed=2, valid=2, invalid=0))

    def test_last_logit_nonfinite_duplicate_identity_and_payload_fail(self):
        for fault in ('last', 'nonfinite', 'duplicate', 'identity', 'payload'):
            records = deepcopy(self.records)
            role = 'consumer'
            if fault == 'last':
                records[0]['logits'][-1] = 10
            elif fault == 'nonfinite':
                records[0]['logits'][0] = float('nan')
            elif fault == 'duplicate':
                records[1] = deepcopy(records[0])
            elif fault == 'identity':
                records[0]['run_id'] = 'foreign'
            else:
                role = 'processor'
            with self.subTest(fault=fault):
                self.assertGreater(check_results(records, self.manifest, role, self.expected)['invalid'], 0)

    def test_empty_results_do_not_create_valid_observations(self):
        self.assertEqual(check_results([], self.manifest, 'consumer', self.expected),
                         dict(observed=0, valid=0, invalid=0))

    def test_startup_failure_and_partial_coverage_remain_distinct_without_comparison_verdict(self):
        for published, complete in ((0, 0), (5, 2)):
            manifest = dict(
                self.manifest, configuration=dict(count=5, intra_op_threads=1),
                processor_configuration=processor_configuration(),
                processor_readiness_configuration_verified=True,
                processor_status_configuration_verified=True,
                startup_readiness_completed=bool(published), execution_completed=False,
                phases=[], nodes={'processor': {'inference_operations': complete}},
            )
            load = dict(
                count=5, requested_hz=30, complete_identities=complete,
                incomplete_identities=5-complete, missing_endpoints=(5-complete)*6,
                clock_verified=True, collector_reports=[dict(dropped=0, flushed=True)]*3,
                counts={'source': {'published': published}},
                timing={'observation_window_ns': 1_000_000_000} if published else {},
                statuses=dict(collection='complete', trace_integrity='valid', observations='incomplete'),
            )
            values = deepcopy(self.records) if complete else []
            for row in values:
                row.update(input_payload_verified=True, api_returned=True)
            with self.subTest(published=published), TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for role in ('processor', 'consumer'):
                    (directory / (role + '.results.jsonl')).touch()
                with patch('robottrace.comparison.compare_runs', side_effect=AssertionError('Unexpected comparison')), patch(
                    'experiments.cpu_inference.e3_summary.read', return_value=manifest
                ), patch('experiments.cpu_inference.e3_summary.rows', return_value=values), patch(
                    'experiments.cpu_inference.e3_summary.fixture_data', return_value=(self.expected, [])
                ), patch('experiments.cpu_inference.e3_summary.load_summary', return_value=load), patch(
                    'experiments.cpu_inference.e3_summary.latency_summary',
                    return_value=dict(metrics={k: dict(count=complete) for k in ('A_ns', 'B_ns', 'C_ns')}, eligible_identity_count=complete),
                ):
                    result = summarize(directory)
                self.assertFalse(result['usable_complete_run'])
                self.assertEqual(result['result_coverage'], .4 if published else None)
                self.assertEqual(result['trace_coverage'], .4 if published else None)
                self.assertEqual(result['numerical_status'], 'passed_for_observed' if published else 'not_exercised')
                self.assertEqual(result['consumer_window_throughput_hz'], 2 if published else None)
                if not published:
                    self.assertIsNone(result['consumer_throughput'])
                    self.assertIsNone(result['processor_throughput'])


if __name__ == '__main__':
    unittest.main()
