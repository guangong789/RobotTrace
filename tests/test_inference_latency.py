"""整数纳秒、完整身份和 V1 证据约束下的离线年龄分解。"""

from dataclasses import replace
import unittest

from experiments.cpu_inference.latency_attribution import attribute
from robottrace import analyze
from test_analysis import message, context


def api(source="s", run="run", **changes):
    value = dict(
        run_id=run,
        source_id=source,
        sequence=1,
        clock_domain="host",
        valid=True,
        api_returned=True,
        generated_ns=100,
        entry_ns=150,
        api_begin_ns=160,
        api_end_ns=190,
    )
    value.update(changes)
    return value


class LatencyAttributionTests(unittest.TestCase):
    def result(self, events=None, evidence=None, records=()):
        events = message() if events is None else events
        return attribute(events, analyze(events, evidence or context()), records)

    def test_exact_decomposition_and_legal_cross_process_overlap(self):
        r = self.result(records=[api()])
        sample = r["messages"][0]
        self.assertEqual(
            [sample[k] for k in ("A_ns", "B_ns", "C_ns", "total_ns", "api_ns")],
            [50, 50, 20, 120, 30],
        )
        self.assertEqual(
            sample["A_ns"] + sample["B_ns"] + sample["C_ns"], sample["total_ns"]
        )
        offset = 2**60
        r = self.result(
            [replace(e, timestamp_ns=e.timestamp_ns + offset) for e in message()]
        )
        self.assertEqual(r["metrics"]["total_ns"]["p95"], 120)

    def test_missing_identity_and_missing_endpoint_are_excluded_and_counted(self):
        partial = [e for e in message(2) if e.stage == "source"]
        r = self.result(message() + partial, context((1, 2)))
        self.assertEqual(
            (r["complete_identity_count"], r["incomplete_identity_count"]), (1, 1)
        )
        self.assertEqual(r["metrics"]["A_ns"]["count"], 1)
        self.assertEqual(r["metrics"]["total_ns"]["unavailable_count"], 1)
        r = self.result([e for e in message() if e.event_type != "callback_exit"])
        self.assertEqual(r["eligible_identity_count"], 0)

    def test_invalid_clock_contradiction_and_ambiguity_withhold_intervals(self):
        scenarios = [
            (message(), context(clock_verified=False)),
            (
                [
                    replace(e, clock_domain="other") if e.stage == "consumer" else e
                    for e in message()
                ],
                context(),
            ),
            (
                [
                    (
                        replace(e, timestamp_ns=149)
                        if (e.stage, e.event_type) == ("processor", "publish_begin")
                        else e
                    )
                    for e in message()
                ],
                context(),
            ),
            (message() + [message()[0]], context()),
        ]
        for events, evidence in scenarios:
            with self.subTest(evidence=evidence, events=events):
                r = self.result(events, evidence, [api()])
                self.assertEqual(r["eligible_identity_count"], 0)
                self.assertTrue(
                    all(
                        v["count"] == 0 and v["p95"] is None
                        for v in r["metrics"].values()
                    )
                )

    def test_percentiles_are_computed_from_paired_messages(self):
        events = []
        for seq, (a, b, c) in enumerate(((100, 3, 3), (3, 100, 3), (3, 3, 100)), 1):
            stamps = [
                100,
                101,
                102,
                100 + a,
                100 + a + b,
                102 + a + b,
                103 + a + b,
                100 + a + b + c,
                101 + a + b + c,
            ]
            events.extend(
                replace(e, timestamp_ns=t) for e, t in zip(message(seq), stamps)
            )
        r = self.result(events, context((1, 2, 3)))
        self.assertEqual(r["metrics"]["total_ns"]["p95"], 106)
        self.assertEqual(
            sum(r["metrics"][k]["p95"] for k in ("A_ns", "B_ns", "C_ns")), 300
        )

    def test_api_join_uses_run_source_and_sequence_and_rejects_duplicates(self):
        events = message(source="a") + message(source="b")
        evidence = context(expected_sequences={"a": (1,), "b": (1,)})
        r = self.result(events, evidence, [api("a", run="foreign"), api("b")])
        self.assertIsNone(r["messages"][0]["api_ns"])
        self.assertEqual(r["messages"][1]["api_ns"], 30)
        self.assertEqual(r["metrics"]["api_ns"]["count"], 1)
        r = self.result(records=[api(), api()])
        self.assertEqual(r["metrics"]["api_ns"]["count"], 0)
        self.assertEqual(r["metrics"]["total_ns"]["count"], 1)

    def test_bad_api_metadata_does_not_become_zero_or_corrupt_event_intervals(self):
        for change in (
            {"clock_domain": "other"},
            {"entry_ns": 151},
            {"generated_ns": 99},
            {"api_begin_ns": 149},
            {"api_end_ns": 201},
            {"api_begin_ns": 160.0},
        ):
            with self.subTest(change=change):
                r = self.result(records=[api(**change)])
                self.assertIsNone(r["messages"][0]["api_ns"])
                self.assertEqual(r["metrics"]["total_ns"]["p95"], 120)

    def test_one_contradictory_identity_does_not_invalidate_another(self):
        other = [
            (
                replace(e, timestamp_ns=149)
                if (e.stage, e.event_type) == ("processor", "publish_begin")
                else e
            )
            for e in message(2)
        ]
        r = self.result(message() + other, context((1, 2)))
        self.assertEqual(r["eligible_identity_count"], 1)
        self.assertEqual(r["withheld_identity_count"], 1)
        self.assertEqual(r["messages"][0]["sequence"], 1)
