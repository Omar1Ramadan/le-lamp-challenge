from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True)
class TraceManifest:
    schema_version: str
    application_version: str
    session_id: str
    configuration_hash: str

    @classmethod
    def example(cls) -> Self:
        return cls("1.0", "test", "session-test", "config-test")


@dataclass(frozen=True)
class TraceRecord:
    sequence: int
    record_type: str
    recorded_at_mono_ns: int
    body: dict[str, object]


@dataclass(frozen=True)
class ReplayTrace:
    manifest: TraceManifest
    records: tuple[TraceRecord, ...]


class TraceWriter:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._records: list[str] = []
        self._opened = False

    async def open(self, manifest: TraceManifest) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        _validate_manifest(manifest)
        (self._directory / "manifest.json").write_text(
            json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        self._opened = True

    async def append(self, record_type: str, mono_ns: int, body: dict[str, object]) -> None:
        if not self._opened:
            raise RuntimeError("trace writer is not open")
        record = TraceRecord(len(self._records) + 1, record_type, mono_ns, body)
        self._records.append(
            json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))
        )

    async def close(self) -> None:
        content = "\n".join(self._records) + "\n"
        events_path = self._directory / "events.jsonl"
        events_path.write_bytes(content.encode())
        digest = hashlib.sha256(events_path.read_bytes()).hexdigest()
        (self._directory / "events.sha256").write_text(digest, encoding="ascii")


class TraceReader:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def manifest(self) -> TraceManifest:
        return self.read_manifest_text(
            (self._directory / "manifest.json").read_text(encoding="utf-8")
        )

    def read_manifest_text(self, text: str) -> TraceManifest:
        data: dict[str, Any] = json.loads(text)
        schema_version = str(data.get("schema_version", ""))
        if schema_version.split(".", 1)[0] != "1":
            raise ValueError(f"unsupported trace schema version: {schema_version}")
        manifest = TraceManifest(**data)
        _validate_manifest(manifest)
        return manifest

    def records(self) -> Iterator[TraceRecord]:
        self.manifest()
        for line in (self._directory / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            yield TraceRecord(**json.loads(line))

    def read(self) -> ReplayTrace:
        return ReplayTrace(self.manifest(), tuple(self.records()))

    def verify_checksum(self) -> bool:
        content = (self._directory / "events.jsonl").read_bytes()
        expected = (self._directory / "events.sha256").read_text(encoding="ascii").strip()
        return hashlib.sha256(content).hexdigest() == expected


def _validate_manifest(manifest: TraceManifest) -> None:
    major = manifest.schema_version.split(".", 1)[0]
    if major != "1":
        raise ValueError(f"unsupported trace schema version: {manifest.schema_version}")
