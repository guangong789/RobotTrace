"""版本 1 的应用观测模型，以及显式的证据前提。"""

from dataclasses import dataclass, field
from typing import Mapping

SCHEMA_VERSION = 1
REQUIRED = (
    ("source", "generate"),
    ("source", "publish_begin"),
    ("source", "publish_end"),
    ("processor", "callback_entry"),
    ("processor", "publish_begin"),
    ("processor", "publish_end"),
    ("processor", "callback_exit"),
    ("consumer", "callback_entry"),
    ("consumer", "callback_exit"),
)


class EventError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    run_id: str
    source_id: str
    sequence: int
    stage: str
    event_type: str
    timestamp_ns: int
    clock_domain: str
    process_id: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise EventError("UNSUPPORTED_SCHEMA_VERSION")
        for value in (self.run_id, self.source_id, self.clock_domain):
            if not isinstance(value, str) or not value:
                raise EventError("INVALID_IDENTITY_OR_CLOCK_LABEL")
        if any(
            type(v) is not int or v < 0
            for v in (self.sequence, self.timestamp_ns, self.process_id)
        ):
            raise EventError("INVALID_INTEGER_FIELD")
        if self.process_id == 0 or (self.stage, self.event_type) not in REQUIRED:
            raise EventError("INVALID_STAGE_OR_EVENT")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise EventError("MALFORMED_EVENT_FIELDS")
        try:
            return cls(**value)
        except (TypeError, KeyError) as exc:
            raise EventError("MALFORMED_EVENT_FIELDS") from exc


@dataclass(frozen=True)
class CollectorEvidence:
    stage: str
    stored: int
    dropped: int
    flushed: bool
    process_id: int

    def __post_init__(self):
        if (
            self.stage not in ("source", "processor", "consumer")
            or type(self.flushed) is not bool
        ):
            raise EventError("INVALID_COLLECTOR_METADATA")
        if (
            any(
                type(v) is not int or v < 0
                for v in (self.stored, self.dropped, self.process_id)
            )
            or self.process_id == 0
        ):
            raise EventError("INVALID_COLLECTOR_METADATA")


@dataclass(frozen=True)
class RunEvidence:
    run_id: str
    clock_domain: str | None = None
    clock_verified: bool = False
    expected_sequences: Mapping[str, tuple[int, ...]] | None = None
    execution_completed: bool | None = None
    collectors: tuple[CollectorEvidence, ...] | None = None
    thresholds_ns: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if (
            not isinstance(self.run_id, str)
            or not self.run_id
            or type(self.clock_verified) is not bool
        ):
            raise EventError("INVALID_RUN_EVIDENCE")
        if (
            self.execution_completed is not None
            and type(self.execution_completed) is not bool
        ):
            raise EventError("INVALID_EXECUTION_EVIDENCE")
