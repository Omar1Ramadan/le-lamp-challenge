import pytest
from social_lamp.conversation.grounding import validate_grounding
from social_lamp.conversation.template import TemplateConversationProvider
from social_lamp.domain.contracts import MemoryResult


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
    provider = TemplateConversationProvider(lambda _: evidence)
    response = await provider.handle_text("turn-1", "Where are my keys?")
    assert response.text == "I last saw the keys on the right side of the desk."
    assert response.evidence_ids == ("observation-1",)


def test_grounding_rejects_location_not_present_in_evidence() -> None:
    evidence = MemoryResult.not_found()
    assert not validate_grounding("Your wallet is on the shelf.", evidence)


@pytest.mark.asyncio
async def test_template_provider_handles_not_found_ambiguous_and_unsupported() -> None:
    provider = TemplateConversationProvider(lambda _: MemoryResult.not_found())
    missing = await provider.handle_text("turn-2", "Where is the wallet?")
    assert missing.status == "not_found"
    assert validate_grounding(missing.text, MemoryResult.not_found())

    ambiguous_result = MemoryResult(
        status="ambiguous", alternatives=("keys", "keyboard"), evidence_ids=()
    )
    ambiguous = await TemplateConversationProvider(lambda _: ambiguous_result).handle_text(
        "turn-3", "Where are the key?"
    )
    assert ambiguous.status == "ambiguous"
    assert "keys, keyboard" in ambiguous.text

    unsupported = await provider.handle_text("turn-4", "Hello lamp")
    assert unsupported.status == "unsupported"
