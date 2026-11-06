import json
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from robottrace import Event, RunEvidence, CollectorEvidence, analyze
from robottrace.events import REQUIRED, EventError
from robottrace.analysis import sequence_summary
from robottrace.io import load_and_analyze


def message(sequence=1, source="s", offset=0):
    # 消费回调可以先于处理回调退出或发布调用返回开始执行。
    stamps = (100, 110, 130, 150, 200, 240, 260, 220, 250)
    return [
        Event(
            "run",
            source,
            sequence,
            stage,
            kind,
            stamp + offset,
            "host",
            {"source": 1, "processor": 2, "consumer": 3}[stage],
        )
        for (stage, kind), stamp in zip(REQUIRED, stamps)
    ]


def context(expected=(1,), **kwargs):
    options = dict(
        run_id="run",
        clock_domain="host",
        clock_verified=True,
        expected_sequences={"s": expected},
        execution_completed=True,
    )
    options.update(kwargs)
    return RunEvidence(**options)


class AnalysisTests(unittest.TestCase):
    def test_normal_correlation_and_overlap(self):
        r = analyze(message(), context())
        self.assertEqual(r["trace_integrity"]["status"], "valid")
        self.assertEqual(r["complete_message_count"], 1)
        self.assertEqual(r["observations"]["status"], "complete")
        self.assertEqual(
            r["messages"][0]["metrics"]["consumer_age_ns"]["value_ns"], 120
        )

    def test_processing_duration(self):
        self.assertEqual(
            analyze(message(), context())["metrics"]["processor_callback_ns"]["p50"],
            110,
        )

    def test_publish_duration(self):
        self.assertEqual(
            analyze(message(), context())["metrics"]["source_publish_call_ns"]["p50"],
            20,
        )

    def test_consumer_completion_is_separate(self):
        r = analyze(message(), context())["messages"][0]["metrics"]
        self.assertEqual(r["consumer_age_ns"]["value_ns"], 120)
        self.assertEqual(r["source_to_consumer_exit_ns"]["value_ns"], 150)

    def test_missing_endpoint_does_not_invent_duration(self):
        r = analyze(
            [
                e
                for e in message()
                if (e.stage, e.event_type) != ("processor", "callback_exit")
            ],
            context(),
        )
        self.assertEqual(r["observations"]["status"], "incomplete")
        self.assertIsNone(
            r["messages"][0]["metrics"]["processor_callback_ns"]["value_ns"]
        )
        self.assertEqual(r["metrics"]["consumer_age_ns"]["count"], 1)

    def test_missing_entire_expected_message(self):
        r = analyze(message(), context((1, 2)))
        self.assertEqual(r["message_identity_count"], 2)
        self.assertEqual(r["observations"]["missing_observation_count"], 9)

    def test_identical_duplicate_is_ambiguous(self):
        r = analyze(message() + [message()[0]], context())
        self.assertEqual(r["trace_integrity"]["status"], "ambiguous")
        self.assertIsNone(r["messages"][0]["metrics"]["consumer_age_ns"]["value_ns"])

    def test_conflicting_timestamp(self):
        r = analyze(message() + [replace(message()[0], timestamp_ns=101)], context())
        self.assertEqual(r["trace_integrity"]["status"], "contradictory")

    def test_out_of_order_input(self):
        self.assertEqual(
            analyze(list(reversed(message())), context()), analyze(message(), context())
        )

    def test_multiple_sequences(self):
        r = analyze(message(2, offset=1000) + message(1), context((1, 2)))
        self.assertEqual(r["complete_message_count"], 2)
        self.assertEqual(r["metrics"]["consumer_age_ns"]["count"], 2)

    def test_multiple_sources_do_not_collide(self):
        r = analyze(
            message() + message(source="other"),
            context(expected_sequences={"s": (1,), "other": (1,)}),
        )
        self.assertEqual(r["message_identity_count"], 2)

    def test_clock_mismatch(self):
        events = message()
        events[0] = replace(events[0], clock_domain="remote")
        r = analyze(events, context())
        self.assertEqual(r["metrics"]["consumer_age_ns"]["count"], 0)
        self.assertEqual(
            r["messages"][0]["metrics"]["consumer_age_ns"]["reason"],
            "CLOCK_DOMAIN_MISMATCH",
        )
        self.assertEqual(r["metrics"]["processor_callback_ns"]["count"], 1)

    def test_clock_assumption_not_verified(self):
        r = analyze(message(), context(clock_verified=False))
        self.assertEqual(r["metrics"]["consumer_age_ns"]["count"], 0)

    def test_negative_semantic_order(self):
        events = message()
        events[4] = replace(events[4], timestamp_ns=140)
        r = analyze(events, context())
        self.assertEqual(r["trace_integrity"]["status"], "contradictory")
        self.assertTrue(
            all(v["value_ns"] is None for v in r["messages"][0]["metrics"].values())
        )

    def test_internal_sequence_gaps(self):
        r = analyze(message(1) + message(3, offset=1000), context((1, 2, 3)))
        row = next(x for x in r["sequences"] if x["stage"] == "consumer")
        self.assertEqual(row["internal_missing_count"], 1)
        self.assertEqual(row["expected_not_observed"], [2])

    def test_huge_sequence_gap_is_compressed(self):
        r = sequence_summary([1, 2**63])
        self.assertEqual(len(r["internal_missing_ranges"]), 1)
        self.assertEqual(r["internal_missing_count"], 2**63 - 2)

    def test_absent_manifest_is_not_completeness(self):
        r = analyze(message(), context(expected_sequences=None))
        self.assertEqual(r["observations"]["status"], "insufficient_evidence")

    def test_overflow_is_not_silent_success(self):
        collectors = tuple(
            CollectorEvidence(
                s,
                sum(e.stage == s for e in message()),
                1 if s == "source" else 0,
                True,
                p,
            )
            for s, p in [("source", 1), ("processor", 2), ("consumer", 3)]
        )
        r = analyze(message(), context(collectors=collectors))
        self.assertEqual(r["collection"]["status"], "incomplete")
        self.assertIn(
            "EVENT_CAPACITY_EXCEEDED", [i["reason"] for i in r["collection"]["issues"]]
        )

    def test_missing_flush(self):
        r = analyze(message(), context(collectors=()))
        self.assertEqual(r["collection"]["status"], "incomplete")

    def test_threshold_exceeded_separate_from_execution(self):
        r = analyze(message(), context(thresholds_ns={"consumer_age_ns": 100}))
        self.assertEqual(r["execution"]["status"], "completed")
        self.assertEqual(
            r["performance"]["consumer_age_ns"]["status"], "threshold_exceeded"
        )

    def test_missing_metric_is_not_performance_failure(self):
        r = analyze(message()[1:], context(thresholds_ns={"consumer_age_ns": 100}))
        self.assertEqual(
            r["performance"]["consumer_age_ns"]["status"], "insufficient_evidence"
        )

    def test_foreign_run(self):
        r = analyze([replace(message()[0], run_id="other")], context())
        self.assertEqual(r["trace_integrity"]["status"], "malformed")

    def test_schema_validation(self):
        with self.assertRaisesRegex(EventError, "UNSUPPORTED_SCHEMA"):
            replace(message()[0], schema_version=2)
        with self.assertRaises(EventError):
            replace(message()[0], timestamp_ns=-1)
        with self.assertRaises(EventError):
            replace(message()[0], sequence=True)

    def test_parse_issue_is_malformed(self):
        r = analyze(message(), context(), [{"reason": "MALFORMED_JSON_EVENT"}])
        self.assertEqual(r["trace_integrity"]["status"], "malformed")

    def test_collector_process_identity_conflict(self):
        c = tuple(
            CollectorEvidence(s, sum(e.stage == s for e in message()), 0, True, 999)
            for s in ("source", "processor", "consumer")
        )
        r = analyze(message(), context(collectors=c))
        self.assertEqual(r["trace_integrity"]["status"], "malformed")

    def test_missing_file_is_an_evidence_gap(self):
        r = analyze(
            message(),
            context(),
            [{"reason": "EVENT_FILE_MISSING", "location": "processor.jsonl"}],
        )
        self.assertEqual(r["trace_integrity"]["status"], "valid")
        self.assertEqual(r["collection"]["status"], "incomplete")

    def test_execution_flag_type_is_strict(self):
        with self.assertRaises(EventError):
            context(execution_completed="false")

    def test_missing_persisted_version_is_malformed(self):
        d = __import__("dataclasses").asdict(message()[0])
        d.pop("schema_version")
        with self.assertRaises(EventError):
            Event.from_dict(d)

    def test_unknown_expected_set_is_not_full_threshold_coverage(self):
        r = analyze(
            message(),
            context(expected_sequences=None, thresholds_ns={"consumer_age_ns": 1000}),
        )
        self.assertEqual(
            r["performance"]["consumer_age_ns"]["status"], "insufficient_evidence"
        )
        self.assertFalse(r["performance"]["consumer_age_ns"]["evidence_complete"])

    def test_json_result_has_no_nan(self):
        json.dumps(analyze(message(), context()), allow_nan=False)


class CausalIntervalTests(unittest.TestCase):
    # 按应用契约独立列出指标端点，不从待测代码读取因果关系。
    metric_pairs = (
        (
            "source_publish_call_ns",
            ("source", "publish_begin"),
            ("source", "publish_end"),
        ),
        (
            "processor_callback_ns",
            ("processor", "callback_entry"),
            ("processor", "callback_exit"),
        ),
        (
            "processor_publish_call_ns",
            ("processor", "publish_begin"),
            ("processor", "publish_end"),
        ),
        (
            "source_to_processor_entry_ns",
            ("source", "generate"),
            ("processor", "callback_entry"),
        ),
        (
            "processor_to_consumer_entry_ns",
            ("processor", "publish_begin"),
            ("consumer", "callback_entry"),
        ),
        (
            "consumer_age_ns",
            ("source", "generate"),
            ("consumer", "callback_entry"),
        ),
        (
            "consumer_callback_ns",
            ("consumer", "callback_entry"),
            ("consumer", "callback_exit"),
        ),
        (
            "source_to_consumer_exit_ns",
            ("source", "generate"),
            ("consumer", "callback_exit"),
        ),
    )

    def pair(self, start, end, start_ns=200, end_ns=150):
        events = {(e.stage, e.event_type): e for e in message()}
        return [
            replace(events[start], timestamp_ns=start_ns),
            replace(events[end], timestamp_ns=end_ns),
        ]

    def test_negative_source_to_processor_missing_intermediates(self):
        events = [
            replace(message()[0], timestamp_ns=200),
            replace(message()[3], timestamp_ns=150),
        ]
        result = analyze(events, context())
        metric = result["messages"][0]["metrics"]["source_to_processor_entry_ns"]
        self.assertEqual(
            metric,
            {
                "status": "insufficient_evidence",
                "value_ns": None,
                "reason": "CONTRADICTORY_MESSAGE_OBSERVATIONS",
            },
        )
        self.assertEqual(result["trace_integrity"]["status"], "contradictory")

    def test_negative_metric_pairs_with_missing_intermediates(self):
        for name, start, end in self.metric_pairs:
            with self.subTest(metric=name):
                result = analyze(self.pair(start, end), context())
                metric = result["messages"][0]["metrics"][name]
                self.assertEqual(result["trace_integrity"]["status"], "contradictory")
                self.assertEqual(metric["status"], "insufficient_evidence")
                self.assertIsNone(metric["value_ns"])
                self.assertEqual(metric["reason"], "CONTRADICTORY_MESSAGE_OBSERVATIONS")
                self.assertEqual(result["metrics"][name]["count"], 0)
                self.assertEqual(
                    result["messages"][0]["issues"],
                    [
                        {
                            "reason": "CONTRADICTORY_EVENT_ORDER",
                            "from": ".".join(start),
                            "to": ".".join(end),
                        }
                    ],
                )

    def test_valid_consumer_overlap_before_processor_completion(self):
        for entry, exit_ in ((220, 250), (250, 280)):
            with self.subTest(consumer_entry=entry, consumer_exit=exit_):
                events = message()
                events[7] = replace(events[7], timestamp_ns=entry)
                events[8] = replace(events[8], timestamp_ns=exit_)
                result = analyze(events, context())
                self.assertEqual(result["trace_integrity"]["status"], "valid")
                self.assertEqual(result["observations"]["status"], "complete")
                metrics = result["messages"][0]["metrics"]
                self.assertTrue(
                    all(metric["status"] == "available" for metric in metrics.values())
                )
                self.assertEqual(metrics["consumer_age_ns"]["value_ns"], entry - 100)
                self.assertEqual(
                    metrics["processor_to_consumer_entry_ns"]["value_ns"], entry - 200
                )

    def test_missing_endpoint_preserves_independent_metrics(self):
        events = [
            e
            for e in message()
            if (e.stage, e.event_type) != ("processor", "callback_exit")
        ]
        result = analyze(events, context())
        metrics = result["messages"][0]["metrics"]
        self.assertEqual(result["trace_integrity"]["status"], "valid")
        self.assertEqual(result["observations"]["status"], "incomplete")
        self.assertEqual(metrics["processor_callback_ns"]["reason"], "MISSING_ENDPOINT")
        self.assertIsNone(metrics["processor_callback_ns"]["value_ns"])
        self.assertTrue(
            all(
                metric["status"] == "available"
                for name, metric in metrics.items()
                if name != "processor_callback_ns"
            )
        )

    def test_negative_pairs_require_comparable_verified_clocks(self):
        for name, start, end in self.metric_pairs:
            for clock_case in ("unverified", "missing_domain", "mismatched"):
                with self.subTest(metric=name, clock=clock_case):
                    events = self.pair(start, end)
                    evidence = context()
                    reason = "CLOCK_ASSUMPTION_UNVERIFIED"
                    if clock_case == "unverified":
                        evidence = context(clock_verified=False)
                    elif clock_case == "missing_domain":
                        evidence = context(clock_domain=None)
                    else:
                        events[1] = replace(events[1], clock_domain="remote")
                        reason = "CLOCK_DOMAIN_MISMATCH"
                    result = analyze(events, evidence)
                    metric = result["messages"][0]["metrics"][name]
                    self.assertEqual(result["trace_integrity"]["status"], "valid")
                    self.assertEqual(result["messages"][0]["issues"], [])
                    self.assertEqual(metric["status"], "insufficient_evidence")
                    self.assertIsNone(metric["value_ns"])
                    self.assertEqual(metric["reason"], reason)

    def test_equal_causal_endpoints_allow_zero(self):
        for name, start, end in self.metric_pairs:
            with self.subTest(metric=name):
                result = analyze(self.pair(start, end, end_ns=200), context())
                metric = result["messages"][0]["metrics"][name]
                self.assertEqual(result["trace_integrity"]["status"], "valid")
                self.assertEqual(metric, {"status": "available", "value_ns": 0})
                self.assertEqual(result["metrics"][name]["count"], 1)

    def test_jsonl_input_order_preserves_causal_analysis(self):
        cases = (
            (message(), "valid"),
            (
                self.pair(("source", "generate"), ("processor", "callback_entry")),
                "contradictory",
            ),
        )
        for events, status in cases:
            with self.subTest(trace_status=status):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    (root / "run.json").write_text(
                        json.dumps(
                            {
                                "run_id": "run",
                                "event_files": ["events.jsonl"],
                                "clock_domain": "host",
                                "clock_verified": True,
                                "expected_sequences": {"s": [1]},
                                "execution_completed": True,
                            }
                        )
                    )
                    results = []
                    for ordered in (events, list(reversed(events))):
                        (root / "events.jsonl").write_text(
                            "".join(json.dumps(asdict(e)) + "\n" for e in ordered)
                        )
                        results.append(load_and_analyze(root))
                    self.assertEqual(results[0], results[1])
                    self.assertEqual(results[0]["trace_integrity"]["status"], status)

    def test_ambiguous_endpoints_do_not_create_causal_contradictions(self):
        for conflict in (False, True):
            with self.subTest(conflicting_endpoint=conflict):
                events = self.pair(
                    ("source", "generate"), ("processor", "callback_entry")
                )
                events.append(
                    replace(events[0], timestamp_ns=201) if conflict else events[0]
                )
                result = analyze(events, context())
                metric = result["messages"][0]["metrics"][
                    "source_to_processor_entry_ns"
                ]
                self.assertIsNone(metric["value_ns"])
                self.assertEqual(metric["reason"], "AMBIGUOUS_ENDPOINT")
                self.assertEqual(
                    result["trace_integrity"]["status"],
                    "contradictory" if conflict else "ambiguous",
                )
                self.assertEqual(
                    [issue["reason"] for issue in result["messages"][0]["issues"]],
                    ["CONFLICTING_EVENT" if conflict else "DUPLICATE_EVENT"],
                )

    def test_overlapping_rules_report_each_endpoint_pair_once(self):
        events = self.pair(
            ("processor", "publish_begin"), ("consumer", "callback_entry")
        )
        result = analyze(events, context())
        self.assertEqual(len(result["messages"][0]["issues"]), 1)
        self.assertEqual(len(result["trace_integrity"]["issues"]), 1)
        self.assertEqual(
            result["messages"][0]["issues"][0],
            {
                "reason": "CONTRADICTORY_EVENT_ORDER",
                "from": "processor.publish_begin",
                "to": "consumer.callback_entry",
            },
        )

    def test_contradiction_invalidates_only_its_message(self):
        events = self.pair(("source", "generate"), ("processor", "callback_entry"))
        events.extend(message()[7:])
        events.extend(message(sequence=2, offset=1000))
        result = analyze(events, context((1, 2)))
        self.assertEqual(result["trace_integrity"]["status"], "contradictory")
        bad, good = result["messages"]
        self.assertTrue(all(m["value_ns"] is None for m in bad["metrics"].values()))
        self.assertEqual(
            bad["metrics"]["consumer_callback_ns"]["reason"],
            "CONTRADICTORY_MESSAGE_OBSERVATIONS",
        )
        self.assertTrue(
            all(m["status"] == "available" for m in good["metrics"].values())
        )
        self.assertEqual(result["metrics"]["consumer_age_ns"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
