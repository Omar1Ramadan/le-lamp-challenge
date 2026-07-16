import pytest
from social_lamp.conversation.grounding import validate_grounding
from social_lamp.conversation.template import TemplateConversationProvider
from social_lamp.domain.contracts import MemoryResult, ObservationSummary


class _FakeMemory:
    def __init__(self, result: MemoryResult) -> None:
        self._result = result
        self._observations: tuple[ObservationSummary, ...] = ()

    def with_observations(self, observations: tuple[ObservationSummary, ...]) -> "_FakeMemory":
        self._observations = observations
        return self

    async def find_last_seen(
        self,
        object_label: str,
        *,
        session_scope: str | None = None,
        before_utc: str | None = None,
    ) -> MemoryResult:
        del object_label, session_scope, before_utc
        return self._result

    async def find_location(
        self,
        entity_label: str,
        *,
        session_scope: str | None = None,
    ) -> MemoryResult:
        del entity_label, session_scope
        return self._result

    async def list_recent_observations(
        self,
        *,
        limit: int = 10,
        before_utc: str | None = None,
    ) -> tuple[ObservationSummary, ...]:
        del limit, before_utc
        return self._observations


@pytest.mark.asyncio
async def test_template_provider_answers_from_evidence() -> None:
    evidence = MemoryResult(
        status="found",
        canonical_label="keys",
        horizontal_region="right",
        depth_band="foreground",
        anchor_name="desk",
        observed_at_utc="2026-07-04T12:00:00Z",
        evidence_ids=("observation-1",),
        alternatives=(),
    )
    provider = TemplateConversationProvider(_FakeMemory(evidence))
    response = await provider.handle_text("turn-1", "Where are my keys?")
    assert response.text == "I last saw the keys on the right side of the desk."
    assert response.evidence_ids == ("observation-1",)
    assert response.grounded is True
    assert response.source == "deterministic_template"


def test_grounding_rejects_location_not_present_in_evidence() -> None:
    evidence = MemoryResult.not_found()
    assert not validate_grounding("Your wallet is on the shelf.", evidence)


@pytest.mark.asyncio
async def test_template_provider_handles_not_found_ambiguous_and_unsupported() -> None:
    provider = TemplateConversationProvider(_FakeMemory(MemoryResult.not_found()))
    missing = await provider.handle_text("turn-2", "Where is the wallet?")
    assert missing.status == "not_found"
    assert not missing.grounded
    assert validate_grounding(missing.text, MemoryResult.not_found())

    ambiguous_result = MemoryResult(
        status="ambiguous", alternatives=("keys", "keyboard"), evidence_ids=()
    )
    ambiguous = await TemplateConversationProvider(
        _FakeMemory(ambiguous_result)
    ).handle_text("turn-3", "Where are the key?")
    assert ambiguous.status == "ambiguous"
    assert "keys, keyboard" in ambiguous.text

    unsupported = await provider.handle_text("turn-4", "Hello lamp")
    assert unsupported.status == "unsupported"
    assert not unsupported.grounded


@pytest.mark.asyncio
async def test_template_provider_recent_observations() -> None:
    observations = (
        ObservationSummary(
            id="obs-1", type="object_seen", summary="cup on the right side of the desk",
            observed_at_utc="2026-07-04T12:00:00Z", evidence_ids=("obs-1",),
        ),
        ObservationSummary(
            id="obs-2", type="object_seen", summary="phone near the keyboard",
            observed_at_utc="2026-07-04T12:01:00Z", evidence_ids=("obs-2",),
        ),
    )
    memory = _FakeMemory(MemoryResult.not_found()).with_observations(observations)
    provider = TemplateConversationProvider(memory)
    response = await provider.handle_text("turn-5", "What have you seen?")
    assert response.status == "found"
    assert response.grounded
    assert "cup" in response.text and "phone" in response.text


@pytest.mark.asyncio
async def test_template_provider_emits_behavior_events() -> None:
    events: list[tuple[str, dict]] = []

    async def emitter(event: str, data: dict) -> None:
        events.append((event, data))

    evidence = MemoryResult(
        status="found",
        canonical_label="keys",
        horizontal_region="right",
        depth_band="foreground",
        anchor_name="desk",
        observed_at_utc="2026-07-04T12:00:00Z",
        evidence_ids=("observation-1",),
    )
    provider = TemplateConversationProvider(_FakeMemory(evidence), event_emitter=emitter)
    await provider.handle_text("turn-6", "Where are my keys?")
    assert any(e == "think" for e, _ in events)
    assert any(e == "recall_success" for e, _ in events)

    events.clear()
    provider2 = TemplateConversationProvider(
        _FakeMemory(MemoryResult.not_found()), event_emitter=emitter
    )
    await provider2.handle_text("turn-7", "Where is the wallet?")
    assert any(e == "think" for e, _ in events)
    assert any(e == "recall_unknown" for e, _ in events)
