import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from robottrace import CollectorEvidence, RunEvidence, analyze
from robottrace.comparison import ComparisonError, compare_runs
from robottrace.io import load_comparison_run
from test_analysis import message

ROOT = Path(__file__).resolve().parents[1]
BOOT = "11111111-2222-3333-4444-555555555555"
OFFSETS = "monotonic 0 0\nboottime 0 0\n"
PACKAGES = (
    "ros-humble-rclcpp",
    "ros-humble-rmw-fastrtps-cpp",
    "ros-humble-std-msgs",
    "ros-humble-std-srvs",
)


def fixture(run_id="a" * 32, *, shift=0, work_ms=1, omit=(), boot=BOOT, zero=False):
    """构造确定性的测试证据；时间戳不是实际 ROS2 测量。"""
    domain = f"linux:CLOCK_MONOTONIC:{boot}:offset=0"
    events = []
    pid_offset = 0 if run_id == "a" * 32 else 1000
    for sequence in (1, 2, 3):
        for event in message(sequence, source="source:1"):
            if (sequence, event.stage, event.event_type) in omit:
                continue
            stamp = 200 if zero else event.timestamp_ns * 1_000_000
            if not zero and (
                event.stage == "consumer"
                or (event.stage == "processor" and event.event_type != "callback_entry")
            ):
                stamp += shift
            events.append(
                replace(
                    event,
                    run_id=run_id,
                    timestamp_ns=stamp + sequence * 1_000_000_000,
                    clock_domain=domain,
                    process_id=event.process_id + pid_offset,
                )
            )
    collectors = [
        CollectorEvidence(
            stage, sum(e.stage == stage for e in events), 0, True, pid + pid_offset
        )
        for stage, pid in (("source", 1), ("processor", 2), ("consumer", 3))
    ]
    config = {
        "count": 3,
        "nominal_rate_hz": 100,
        "depth": 10,
        "processor_work_ms": work_ms,
        "drain_s": 6,
        "collector_capacity_per_stage": 4096,
        "reliability": "RELIABLE",
        "durability": "VOLATILE",
        "history": "KEEP_LAST",
        "executor": "SingleThreadedExecutor",
        "domain_id": 42,
        "layout": "three independent C++ processes in one ephemeral ROS2 container",
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "scenario": "normal",
        "configuration": config,
        "expected_sequences": {"source:1": [1, 2, 3]},
        "clock_domain": domain,
        "clock_verified": True,
        "execution_completed": True,
        "host_clock": {"boot_id": boot, "offsets": OFFSETS},
        "nodes": {
            stage: {"boot_id": boot, "clock_domain": domain, "time_offsets": OFFSETS}
            for stage in ("source", "processor", "consumer")
        },
        "collectors": [asdict(c) for c in collectors],
        "event_files": ["events.jsonl"],
        "versions": "".join(f"{p}\t1\n" for p in PACKAGES),
        "binary_sha256": "0" * 64,
        "commands": {"source": ["pipeline", run_id]},
    }
    result = analyze(
        events,
        RunEvidence(
            run_id, domain, True, {"source:1": (1, 2, 3)}, True, tuple(collectors)
        ),
    )
    result.update(workload_configuration=copy.deepcopy(config), fixture_provenance=None)
    return {"manifest": manifest, "analysis": result}, events


def persist(path, run, events):
    path.mkdir()
    (path / "run.json").write_text(json.dumps(run["manifest"]))
    (path / "events.jsonl").write_text(
        "".join(json.dumps(asdict(e)) + "\n" for e in events)
    )


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.baseline, _ = fixture()
        self.candidate, _ = fixture("b" * 32)

    def compare(self, **kwargs):
        return compare_runs(self.baseline, self.candidate, **kwargs)

    def test_identical_compatible_results_and_serialization(self):
        result = self.compare()
        self.assertEqual(result["status"], "COMPARABLE")
        self.assertEqual(result["comparison_schema_version"], 1)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        for metric in result["metrics"].values():
            self.assertEqual(metric["verdict"], "NO_OBSERVED_DIFFERENCE")
            self.assertEqual(metric["baseline"]["coverage"], 1)
            self.assertEqual(metric["candidate"]["available_count"], 3)
            self.assertEqual(metric["changes"]["p95"]["delta_ns"], 0)

    def test_known_numerical_regression(self):
        self.candidate, _ = fixture("b" * 32, shift=100_000_000)
        result = self.compare()
        metric = result["metrics"]["consumer_age_ns"]
        self.assertEqual(metric["verdict"], "OBSERVED_REGRESSION")
        self.assertEqual(metric["changes"]["p50"]["delta_ns"], 100_000_000)
        self.assertAlmostEqual(
            metric["changes"]["p95"]["percent_change"], 100 / 120 * 100
        )

    def test_lower_p95_with_incomplete_population_withholds_verdict(self):
        self.baseline, _ = fixture(shift=20_000_000)
        self.candidate, _ = fixture("b" * 32, omit={(1, "consumer", "callback_entry")})
        result = self.compare()
        age = result["metrics"]["consumer_age_ns"]
        self.assertLess(age["changes"]["p95"]["delta_ns"], 0)
        self.assertEqual(age["candidate"]["available_count"], 2)
        self.assertEqual(age["candidate"]["unavailable_count"], 1)
        self.assertEqual(age["candidate"]["coverage"], 2 / 3)
        self.assertEqual(age["verdict"], "WITHHELD")
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(
            result["metrics"]["processor_callback_ns"]["status"], "COMPARABLE"
        )

    def test_equal_partial_counts_are_not_equal_sample_populations(self):
        self.baseline, _ = fixture(omit={(1, "consumer", "callback_entry")})
        self.candidate, _ = fixture("b" * 32, omit={(2, "consumer", "callback_entry")})
        self.assertEqual(
            self.compare()["metrics"]["consumer_age_ns"]["verdict"], "WITHHELD"
        )

    def test_explicit_processor_work_experiment(self):
        self.baseline, _ = fixture(work_ms=50, shift=40_000_000)
        self.candidate, _ = fixture("b" * 32, work_ms=10)
        result = self.compare(allow_changes=["processor_work_ms"])
        self.assertEqual(result["conditions"]["status"], "COMPARABLE")
        self.assertEqual(
            result["declared_changes"],
            [{"key": "processor_work_ms", "baseline": 50, "candidate": 10}],
        )
        self.assertEqual(
            result["metrics"]["processor_callback_ns"]["verdict"],
            "OBSERVED_IMPROVEMENT",
        )

    def test_undeclared_processor_work_difference_is_drift(self):
        self.candidate, _ = fixture("b" * 32, work_ms=10)
        result = self.compare()
        self.assertEqual(result["status"], "NOT_COMPARABLE")
        self.assertEqual(
            result["metrics"]["processor_callback_ns"]["verdict"], "WITHHELD"
        )

    def test_preset_change_must_be_explained_by_declared_work_change(self):
        self.candidate, _ = fixture("b" * 32, work_ms=50)
        self.candidate["manifest"]["scenario"] = "slow"
        self.assertEqual(self.compare()["status"], "NOT_COMPARABLE")
        allowed = self.compare(allow_changes=["processor_work_ms"])
        self.assertEqual(allowed["status"], "COMPARABLE")
        self.assertEqual(
            allowed["declared_changes"][1]["explained_by"], "processor_work_ms"
        )

    def test_unsupported_declarations_are_rejected(self):
        for keys in (
            ["depth"],
            ["binary_sha256"],
            ["clock_verified"],
            ["anything"],
            [["processor_work_ms"]],
        ):
            with self.subTest(keys=keys), self.assertRaises(ComparisonError):
                self.compare(allow_changes=keys)

    def test_clock_verification_is_not_overridable(self):
        self.candidate["analysis"]["clock"]["verified_by_producer_evidence"] = False
        result = self.compare(allow_changes=["processor_work_ms"])
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(
            all(m["verdict"] == "WITHHELD" for m in result["metrics"].values())
        )

    def test_clock_conflicts_and_unsupported_kinds(self):
        for domain in (
            "linux:CLOCK_REALTIME:host",
            "host",
            "linux:CLOCK_MONOTONIC:" + BOOT + ":offset=1",
        ):
            with self.subTest(domain=domain):
                self.candidate["manifest"]["clock_domain"] = domain
                self.assertEqual(self.compare()["status"], "NOT_COMPARABLE")

    def test_missing_clock_provenance_and_nonzero_offsets(self):
        del self.candidate["manifest"]["host_clock"]
        self.assertEqual(self.compare()["status"], "INSUFFICIENT_EVIDENCE")
        self.candidate, _ = fixture("b" * 32)
        self.candidate["manifest"]["host_clock"][
            "offsets"
        ] = "monotonic 1 0\nboottime 0 0\n"
        self.assertEqual(self.compare()["status"], "NOT_COMPARABLE")

    def test_different_verified_boots_need_no_timestamp_alignment(self):
        self.candidate, _ = fixture(
            "b" * 32, boot="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )
        self.assertEqual(self.compare()["status"], "COMPARABLE")

    def test_collection_incomplete_blocks_all_verdicts(self):
        self.candidate["analysis"]["collection"] = {
            "status": "incomplete",
            "issues": [{"reason": "EVENT_CAPACITY_EXCEEDED"}],
        }
        result = self.compare()
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(
            all(m["verdict"] == "WITHHELD" for m in result["metrics"].values())
        )

    def test_invalid_traces_are_not_comparable(self):
        for status in ("ambiguous", "contradictory", "malformed"):
            with self.subTest(status=status):
                self.candidate["analysis"]["trace_integrity"]["status"] = status
                self.assertEqual(self.compare()["status"], "NOT_COMPARABLE")

    def test_failed_execution_withholds_performance_conclusions(self):
        self.candidate["analysis"]["execution"]["status"] = "failed"
        self.assertEqual(self.compare()["status"], "INSUFFICIENT_EVIDENCE")

    def test_missing_statistics_and_missing_metric_are_not_zero(self):
        for mode in ("p95", "metric"):
            with self.subTest(mode=mode):
                self.candidate, _ = fixture("b" * 32)
                if mode == "p95":
                    self.candidate["analysis"]["metrics"]["consumer_age_ns"][
                        "p95"
                    ] = None
                else:
                    del self.candidate["analysis"]["metrics"]["consumer_age_ns"]
                metric = self.compare()["metrics"]["consumer_age_ns"]
                self.assertIsNone(metric["candidate"]["p95_ns"])
                self.assertIsNone(metric["changes"]["p95"]["delta_ns"])
                self.assertEqual(metric["verdict"], "WITHHELD")

    def test_zero_baseline_percentage_is_undefined(self):
        self.baseline, _ = fixture(zero=True, work_ms=0)
        self.candidate, _ = fixture("b" * 32, work_ms=0)
        metric = self.compare()["metrics"]["consumer_age_ns"]
        self.assertEqual(metric["baseline"]["p95_ns"], 0)
        self.assertIsNone(metric["changes"]["p95"]["percent_change"])
        self.assertEqual(metric["verdict"], "OBSERVED_REGRESSION")

    def test_uuid_pid_and_commands_are_not_configuration_drift(self):
        self.assertNotEqual(
            self.baseline["manifest"]["run_id"], self.candidate["manifest"]["run_id"]
        )
        self.assertNotEqual(
            self.baseline["manifest"]["collectors"][0]["process_id"],
            self.candidate["manifest"]["collectors"][0]["process_id"],
        )
        self.assertEqual(self.compare()["status"], "COMPARABLE")

    def test_expected_identity_ranges_must_match(self):
        self.candidate["manifest"]["expected_sequences"] = {"source:1": [1, 2, 4]}
        self.assertEqual(self.compare()["status"], "NOT_COMPARABLE")

    def test_rate_qos_and_unknown_configuration_drift(self):
        for key, value in (
            ("nominal_rate_hz", 50),
            ("depth", 1),
            ("reliability", "BEST_EFFORT"),
            ("new_option", True),
        ):
            with self.subTest(key=key):
                self.candidate, _ = fixture("b" * 32)
                self.candidate["manifest"]["configuration"][key] = value
                self.candidate["analysis"]["workload_configuration"][key] = value
                self.assertEqual(
                    self.compare(allow_changes=["processor_work_ms"])["status"],
                    "NOT_COMPARABLE",
                )

    def test_software_changes_are_not_silently_accepted(self):
        for field, value in (
            ("binary_sha256", "f" * 64),
            ("versions", "".join(f"{p}\t2\n" for p in PACKAGES)),
        ):
            with self.subTest(field=field):
                self.candidate, _ = fixture("b" * 32)
                self.candidate["manifest"][field] = value
                self.assertEqual(
                    self.compare(allow_changes=["processor_work_ms"])["status"],
                    "NOT_COMPARABLE",
                )

    def test_missing_software_expected_set_or_configuration(self):
        for key in ("versions", "binary_sha256", "expected_sequences", "configuration"):
            with self.subTest(key=key):
                self.candidate, _ = fixture("b" * 32)
                del self.candidate["manifest"][key]
                if key == "configuration":
                    self.candidate["analysis"]["workload_configuration"] = None
                self.assertEqual(self.compare()["status"], "INSUFFICIENT_EVIDENCE")

    def test_partially_recorded_software_is_missing_evidence_not_version_drift(self):
        self.candidate["manifest"]["versions"] = f"{PACKAGES[0]}\t1\n"
        result = self.compare()
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")
        self.assertNotIn(
            "SOFTWARE_IDENTITY_DRIFT",
            [i["code"] for i in result["conditions"]["issues"]],
        )

    def test_malformed_inputs_and_incompatible_schemas(self):
        for mutate in (
            lambda r: r["analysis"].update(result_schema_version=2),
            lambda r: r["manifest"].update(schema_version=True),
            lambda r: r["analysis"].update(run_id="foreign"),
            lambda r: r["analysis"]["metrics"]["consumer_age_ns"].update(
                p95=float("nan")
            ),
            lambda r: r["analysis"]["metrics"]["consumer_age_ns"].update(count=2),
            lambda r: r["analysis"]["metrics"]["consumer_age_ns"].update(p95=-1),
        ):
            with self.subTest(mutation=mutate):
                self.candidate, _ = fixture("b" * 32)
                mutate(self.candidate)
                with self.assertRaises(ComparisonError):
                    self.compare()

    def test_inputs_are_unchanged_and_generic_metrics_work(self):
        before = copy.deepcopy((self.baseline, self.candidate))
        result = self.compare(metrics=["source_publish_call_ns"])
        self.assertEqual(
            result["metrics"]["source_publish_call_ns"]["status"], "COMPARABLE"
        )
        self.assertEqual(before, (self.baseline, self.candidate))

    def test_matching_but_unsupported_pipeline_semantics_are_not_accepted(self):
        for key, value in (
            ("layout", "unknown pipeline"),
            ("executor", "MultiThreadedExecutor"),
            ("reliability", "unknown qos"),
            ("durability", "unknown durability"),
            ("history", "unknown history"),
        ):
            with self.subTest(key=key):
                self.baseline, _ = fixture()
                self.candidate, _ = fixture("b" * 32)
                for run in (self.baseline, self.candidate):
                    run["manifest"]["configuration"][key] = value
                    run["analysis"]["workload_configuration"][key] = value
                self.assertEqual(self.compare()["status"], "NOT_COMPARABLE")


class ComparisonCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        self.br, be = fixture()
        self.cr, ce = fixture("b" * 32)
        persist(self.baseline, self.br, be)
        persist(self.candidate, self.cr, ce)

    def cli(self, *args):
        return subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "robottrace",
                "compare",
                str(self.baseline),
                str(self.candidate),
                *args,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_json_output_and_explicit_persistence(self):
        target = self.root / "comparison.json"
        r = self.cli("--json", "--output", str(target))
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertEqual(result["status"], "COMPARABLE")
        self.assertEqual(json.loads(target.read_text()), result)
        self.assertFalse((self.candidate / "analysis.json").exists())

    def test_default_summary_has_no_output_file(self):
        r = self.cli()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("comparison=COMPARABLE", r.stdout)
        self.assertIn("coverage=1.0", r.stdout)
        self.assertFalse((self.candidate / "comparison.json").exists())

    def test_explicit_output_does_not_overwrite_existing_evidence(self):
        target = self.candidate / "run.json"
        before = target.read_bytes()
        r = self.cli("--output", str(target))
        self.assertEqual(r.returncode, 2)
        self.assertEqual(target.read_bytes(), before)

    def test_rejected_conditions_are_computed_results_not_acceptance(self):
        self.cr["manifest"]["configuration"]["processor_work_ms"] = 10
        (self.candidate / "run.json").write_text(json.dumps(self.cr["manifest"]))
        r = self.cli("--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["status"], "NOT_COMPARABLE")
        allowed = self.cli("--json", "--allow-change", "processor_work_ms")
        self.assertEqual(json.loads(allowed.stdout)["status"], "COMPARABLE")

    def test_bad_manifest_schema_and_unsupported_override_return_two(self):
        for text in (
            "{bad",
            '{"run_id":"a","run_id":"b"}',
            json.dumps({**self.cr["manifest"], "schema_version": 2}),
        ):
            with self.subTest(text=text):
                (self.candidate / "run.json").write_text(text)
                r = self.cli("--json")
                self.assertEqual(r.returncode, 2)
                self.assertFalse(r.stdout)
        r = self.cli("--allow-change", "depth")
        self.assertEqual(r.returncode, 2)

    def test_cached_analysis_is_not_trusted_or_modified(self):
        cached = self.candidate / "analysis.json"
        cached.write_text('{"result_schema_version":999}')
        result = load_comparison_run(self.candidate)
        self.assertEqual(result["analysis"]["result_schema_version"], 1)
        self.assertEqual(self.cli("--json").returncode, 0)
        self.assertEqual(cached.read_text(), '{"result_schema_version":999}')

    def test_manifest_cannot_reference_files_outside_run(self):
        self.cr["manifest"]["event_files"] = ["../outside.jsonl"]
        (self.candidate / "run.json").write_text(json.dumps(self.cr["manifest"]))
        self.assertEqual(self.cli("--json").returncode, 2)


if __name__ == "__main__":
    unittest.main()
