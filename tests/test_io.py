import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from robottrace.io import load_and_analyze, load_comparison_run


class InputTests(unittest.TestCase):
    def read_fixture(self, text):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": "r",
                        "event_files": ["events.jsonl"],
                        "expected_sequences": {"s": [1]},
                        "clock_domain": "host",
                        "clock_verified": True,
                    }
                )
            )
            if text is not None:
                (root / "events.jsonl").write_text(text)
            return load_and_analyze(root)

    def test_missing_file_is_not_a_malformed_event(self):
        r = self.read_fixture(None)
        self.assertEqual(r["collection"]["status"], "incomplete")
        self.assertEqual(r["trace_integrity"]["status"], "valid")
        self.assertEqual(r["observations"]["missing_observation_count"], 9)

    def test_malformed_json_keeps_missing_metrics_null(self):
        r = self.read_fixture("{bad json}\n")
        self.assertEqual(r["trace_integrity"]["status"], "malformed")
        self.assertIsNone(r["messages"][0]["metrics"]["consumer_age_ns"]["value_ns"])

    def test_duplicate_json_fields_are_not_silently_overwritten(self):
        r = self.read_fixture('{"timestamp_ns":1,"timestamp_ns":2}\n')
        self.assertEqual(
            r["trace_integrity"]["issues"][0]["reason"], "DUPLICATE_JSON_FIELD"
        )


class ManifestLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {
            "run_id": "r",
            "event_files": ["events.jsonl"],
            "expected_sequences": {"s": [1]},
            "clock_domain": "host",
            "clock_verified": True,
        }
        (self.root / "run.json").write_text(json.dumps(self.manifest))
        (self.root / "events.jsonl").write_text("")

    def test_comparison_manifest_is_read_once(self):
        original = Path.read_text
        reads = []

        def tracked(path, *args, **kwargs):
            if path == self.root / "run.json":
                reads.append(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", tracked):
            load_comparison_run(self.root)
        self.assertEqual(len(reads), 1)

    def test_analysis_uses_the_same_snapshot_as_comparison_metadata(self):
        original = Path.read_text
        reads = []

        def changing(path, *args, **kwargs):
            if path == self.root / "run.json":
                reads.append(path)
                data = dict(self.manifest)
                if len(reads) > 1:
                    data["run_id"] = "changed-between-reads"
                return json.dumps(data)
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", changing):
            loaded = load_comparison_run(self.root)
        self.assertEqual(loaded["manifest"]["run_id"], loaded["analysis"]["run_id"])
        self.assertEqual(loaded["analysis"]["run_id"], "r")
        self.assertEqual(len(reads), 1)

    def test_existing_valid_analysis_result_is_identical(self):
        loaded = load_comparison_run(self.root)
        self.assertEqual(loaded["manifest"], self.manifest)
        self.assertEqual(loaded["analysis"], load_and_analyze(self.root))

    def test_malformed_and_duplicate_manifest_json_remain_rejected(self):
        for text in (
            "{bad",
            '{"run_id":"a","run_id":"r","event_files":["events.jsonl"]}',
            "[]",
        ):
            with self.subTest(text=text):
                (self.root / "run.json").write_text(text)
                with self.assertRaises(ValueError):
                    load_comparison_run(self.root)

    def test_invalid_file_lists_and_paths_remain_rejected(self):
        for names in (
            None,
            [],
            ["events.jsonl", "events.jsonl"],
            [""],
            [1],
            ["../outside.jsonl"],
            ["/tmp/outside.jsonl"],
        ):
            with self.subTest(names=names):
                data = {**self.manifest, "event_files": names}
                (self.root / "run.json").write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_comparison_run(self.root)

    def test_event_duplicate_field_validation_is_unchanged(self):
        (self.root / "events.jsonl").write_text('{"timestamp_ns":1,"timestamp_ns":2}\n')
        loaded = load_comparison_run(self.root)
        self.assertEqual(loaded["analysis"]["trace_integrity"]["status"], "malformed")
        self.assertEqual(
            loaded["analysis"]["trace_integrity"]["issues"][0]["reason"],
            "DUPLICATE_JSON_FIELD",
        )

    def test_missing_event_file_remains_an_evidence_gap(self):
        (self.root / "events.jsonl").unlink()
        result = load_comparison_run(self.root)["analysis"]
        self.assertEqual(result["collection"]["status"], "incomplete")
        self.assertEqual(result["trace_integrity"]["status"], "valid")

    def test_legacy_analysis_manifest_policy_is_preserved(self):
        # 普通入口仍接受旧的最小清单；严格比较入口仍拒绝重复键。
        (self.root / "run.json").write_text(
            '{"run_id":"old","run_id":"r","event_files":["events.jsonl"]}'
        )
        self.assertEqual(load_and_analyze(self.root)["run_id"], "r")
        with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_FIELD"):
            load_comparison_run(self.root)
