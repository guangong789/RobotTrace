# Measurement semantics — version 1

## Identity and payload

Events are JSON objects, one per line. All fields are mandatory: `schema_version` (integer 1), `run_id`, `source_id`, `sequence`, `stage`, `event_type`, `timestamp_ns`, `clock_domain`, `process_id`. JSON duplicate fields, unknown/missing fields, unsupported versions and invalid integer/identity values are not silently accepted. Process IDs are meaningful within this recorded run's PID namespace, not global machine identities.

The demonstration uses `std_msgs/msg/UInt64MultiArray`, empty layout, zero data offset, exactly six uint64 values:

| Index | Meaning |
| --- | --- |
| 0 | Payload schema version 1 |
| 1, 2 | High/low 64 bits of the 128-bit run UUID |
| 3 | Source token 1, represented as `source:1` in events (single-source demo) |
| 4 | Source sequence, starting at 1 |
| 5 | Source-side CLOCK_MONOTONIC sample, before storing this field and publishing |

The processor republishes the same message, preserving all identity fields and the source stamp. Processor and consumer reject foreign/malformed identities and future source timestamps. Core correlation does not infer identity from topic names, MCAP counters, arrival order or process exit status.

## Required observations and boundaries

| Stage | Required events in local order |
| --- | --- |
| source | generate → publish_begin → publish_end |
| processor | callback_entry → publish_begin → publish_end → callback_exit |
| consumer | callback_entry → callback_exit |

Generate is a local publish-side observation, not sensor acquisition. Entry is sampled before validating the message and recording the entry observation. Processor exit is sampled after simulated work and output publication; consumer exit follows its identity checks. An exit observation is just before callback return, so it excludes that observation's own append and function epilogue.

Publish begin is sampled just before the bounded buffer append preceding `publish()`; end is sampled immediately after successful return. Thus publish-call intervals include the small begin-record append and timestamp boundary costs. An exception has no publish_end observation; node metadata records publish errors and the process fails, while available buffered evidence is flushed. Returning from publish does not prove subscriber receipt or completion.

The consumer can enter before processor publish returns or before the processor callback exits. Those cross-process overlaps are legal. No processor-exit-before-consumer-entry invariant is imposed.

## Intervals

| Result field | End minus start |
| --- | --- |
| source_publish_call_ns | source.publish_end − source.publish_begin |
| processor_callback_ns | processor.callback_exit − processor.callback_entry |
| processor_publish_call_ns | processor.publish_end − processor.publish_begin |
| source_to_processor_entry_ns | processor.callback_entry − source.generate |
| processor_to_consumer_entry_ns | consumer.callback_entry − processor.publish_begin |
| consumer_age_ns | consumer.callback_entry − source.generate |
| consumer_callback_ns | consumer.callback_exit − consumer.callback_entry |
| source_to_consumer_exit_ns | consumer.callback_exit − source.generate |

Processor callback duration includes identity checks, simulated delay, instrumentation and output publication. Consumer age covers the whole application pipeline up to entry; source-to-consumer-exit additionally includes the consumer callback. None is isolated DDS transport latency or physical robot-task completion.

Every metric explicitly reports `available` or `insufficient_evidence`, with integer nanoseconds or `null` and a reason. Missing endpoints never become zero. Duplicate/conflicting endpoints cannot be arbitrarily selected. Contradictory ordering invalidates that message's interval conclusions. Unaffected metrics can remain available when an unrelated endpoint is absent. Statistics use only available values, report sample/unavailable counts, and use integer nearest-rank p50/p95.

上述各指标都具有因果方向：当两端观测唯一且时钟证据允许比较时，终点早于起点即构成矛盾，即使中间观测缺失也必须检查。沿用 `CONTRADICTORY_EVENT_ORDER`，并使该消息的区间全部不可用；相同端点对只报告一次。时间戳相等允许得到零区间。缺端点或时钟不可比较仅表示证据不足，不据此推断矛盾；无矛盾时，其他具备证据的指标仍可计算。这不要求 processor 发布返回或回调退出先于 consumer 进入或退出，合法的跨进程重叠仍然允许。

## Clock domain

All application observations use Linux CLOCK_MONOTONIC. The supported producer checks Linux boot UUID and zero monotonic/boottime offsets in `/proc/self/timens_offsets`. The host supplies its independently read boot/offset evidence; all three processes report matching values. The domain label includes the boot UUID and monotonic clock identity. The runner sets clock_verified only after agreement.

The core requires explicit verified run evidence and matching domain labels for both endpoints. A matching string alone is insufficient. Mismatch/unverified assumptions produce unavailable intervals, not a guessed correction. It does not synchronize machines or validate dishonest metadata. CLOCK_MONOTONIC excludes suspended time and may have rate adjustments; reported nanosecond units do not imply nanosecond accuracy. WSL scheduling and suspend/resume behavior are not benchmarked.

Administrative UTC start/end labels are never subtracted from monotonic timestamps. A later independent replay or another boot needs new evidence; labels from different hosts cannot be assumed comparable.

## Collection, execution and status axes

Each single-thread-owned buffer reserves capacity before processing. A hot record samples the clock and appends one numeric value object. Capacity exhaustion increments a dropped counter and continues the workload; it cannot silently retain a complete-trace classification. Flushing formats JSONL and writes files only on normal process shutdown. Flushed means normal output completion, not fsync/crash durability. SIGKILL, crash or power loss can lose the entire unflushed buffer; no recovery journal is implemented. Missing files/count mismatches are measurement incompleteness. Malformed JSON/schema is a separate malformed-trace finding.

- `execution`: completed / failed / unknown, supplied from explicit process/run evidence.
- `collection`: complete / incomplete / unknown, based on stage files, counters, process identities and flush evidence. Complete does not mean every ROS message was observed.
- `trace_integrity`: valid / ambiguous / contradictory / malformed.
- `observations`: complete / incomplete / ambiguous / insufficient_evidence. Complete requires the declared expected identity set, not just a nonempty trace.
- `performance[metric]`: within_threshold / threshold_exceeded / insufficient_evidence. An observed exceedance remains an observed fact even when other identities lack metrics; `evidence_complete` stays false. Missing evidence alone is not a performance failure.

An identical repeated endpoint is a duplicate/ambiguity; different observations for the same endpoint are conflicts. The core does not merge stages from incompatible process identities. The example's one-source expected set is a configured observation contract; source publish counts and execution status separately show whether it was actually emitted successfully.

Sequence summaries are sets anchored by source.generate, processor.callback_entry and consumer.callback_entry. Internal gaps are compressed ranges strictly between the observed minimum and maximum. Missing prefix/suffix are relative to an explicit expected set; without it those completeness claims remain unknown. No observation at all has no inferred start/end. Missing observations are not automatically network loss, queue overwrite or any other root cause.

## Workload bounds and overhead

Source production uses absolute 10 ms targets by default and does not intentionally skip sequence IDs to catch up. Readiness is a matched-count plus ROS Trigger service gate, not a startup sleep. Expected source production is bounded by count and safety budget. After publication, the controller targets a 6-second drain and then signals its own child processes. Analysis includes observations until their actual normal shutdown; the stop target is not a hard real-time observation cutoff. Manifests retain the target, stop-signal and node-exit times.

Standard startup graph observers can overlap early samples and consume resources. Their commands/outputs are recorded; the benchmark below does not isolate this whole-workload perturbation.

The overhead probe measures seven batches of 20,000 Buffer.record calls, including clock sampling, loop cost, stores and possible first-touch memory effects. Reservation, JSON formatting, file I/O, ROS serialization and delivery are outside the timing. The buffer is consumed through `/dev/null` output after the timed region to prevent dead-store elimination. Values are batch-average nanoseconds/event, not per-call percentiles or total pipeline overhead. There is no uninstrumented pipeline comparison and no zero-overhead claim.

References: [Linux time namespaces](https://man7.org/linux/man-pages/man7/time_namespaces.7.html), [Humble executor behavior](https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Executors.rst). Workload-dependent measurements are not correctness invariants.
