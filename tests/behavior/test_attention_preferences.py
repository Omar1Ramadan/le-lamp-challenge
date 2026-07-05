from pathlib import Path

import pytest
from social_lamp.behavior.policy import AttentionSchedule
from social_lamp.behavior.preferences import PreferenceModel
from social_lamp.memory.repository import MemoryRepository


def test_attention_schedule_escalates_then_exhausts() -> None:
    schedule = AttentionSchedule(disengaged_at_ns=0)
    assert schedule.intent_at(4_999_000_000) is None
    assert schedule.intent_at(5_000_000_000).parameters["level"] == 1
    assert schedule.intent_at(15_000_000_000).parameters["level"] == 2
    assert schedule.intent_at(35_000_000_000).parameters["level"] == 3
    assert schedule.intent_at(36_000_000_000) is None
    assert schedule.exhausted


def test_attention_schedule_suppresses_and_respects_cooldown() -> None:
    schedule = AttentionSchedule(disengaged_at_ns=0)
    schedule.suppress("directed_speech")
    assert schedule.intent_at(5_000_000_000) is None
    assert schedule.suppression_reason == "directed_speech"
    assert not schedule.can_restart_at(59_999_999_999)
    assert schedule.can_restart_at(60_000_000_000)


def test_preferences_are_bounded_auditable_and_decay() -> None:
    model = PreferenceModel()
    for _ in range(10):
        model.record("seek_attention:quiet", "light-pulse", "reengaged")
    assert model.score("seek_attention:quiet", "light-pulse") == 1.5
    audit = model.audit[-1]
    assert audit.previous_score <= audit.new_score
    model.start_session()
    assert model.score("seek_attention:quiet", "light-pulse") == 1.475


@pytest.mark.asyncio
async def test_preference_audit_persists_transactionally(tmp_path: Path) -> None:
    repository = await MemoryRepository.open(tmp_path / "memory.db")
    await repository.record_preference_update(
        context="seek_attention:quiet",
        behavior="light-pulse",
        outcome="reengaged",
        previous_score=1.0,
        new_score=1.1,
        correlation_id="correlation-1",
    )
    updates = await repository.preference_audit()
    assert updates[-1].context == "seek_attention:quiet"
    assert updates[-1].new_score == 1.1
    score = await repository.preference_score("seek_attention:quiet", "light-pulse")
    assert score == 1.1
    await repository.close()
