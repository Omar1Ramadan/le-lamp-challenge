from pathlib import Path

import pytest
from social_lamp.replay.trace import TraceManifest, TraceReader, TraceWriter


@pytest.mark.asyncio
async def test_trace_round_trip_preserves_order_and_checksum(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path)
    await writer.open(TraceManifest.example())
    await writer.append("observation", 20, {"kind": "face_presence"})
    await writer.append("snapshot", 30, {"revision": 1})
    await writer.close()
    records = list(TraceReader(tmp_path).records())
    assert [record.sequence for record in records] == [1, 2]
    assert [record.record_type for record in records] == ["observation", "snapshot"]
    assert TraceReader(tmp_path).verify_checksum()


def test_reader_rejects_incompatible_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        TraceReader(tmp_path).read_manifest_text('{"schema_version":"2.0"}')


def test_core_engagement_fixture_replays_in_order() -> None:
    reader = TraceReader(Path("evaluation/fixtures/core-engagement"))
    records = list(reader.records())
    assert [record.sequence for record in records] == [1, 2, 3, 4, 5]
    assert [record.record_type for record in records] == [
        "observation",
        "snapshot",
        "snapshot",
        "intent",
        "timeline",
    ]
    assert reader.verify_checksum()
