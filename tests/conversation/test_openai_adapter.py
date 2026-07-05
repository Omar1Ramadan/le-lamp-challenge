import pytest
from social_lamp.config import Settings
from social_lamp.conversation.openai_realtime import OpenAIRealtimeProvider
from social_lamp.conversation.template import TemplateConversationProvider
from social_lamp.domain.contracts import MemoryResult
from social_lamp.runtime.providers import build_conversation_provider


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.answer = "I do not have reliable evidence for that."
        self.not_found_result = MemoryResult.not_found()
        self.cancelled_turns: list[str] = []
        self.closed_reason: str | None = None
        self.calls = 0

    async def answer_text(
        self,
        *,
        model: str,
        turn_id: str,
        text: str,
        evidence_result: MemoryResult,
        tools: tuple[dict[str, object], ...],
    ) -> str:
        del model, turn_id, text, evidence_result, tools
        self.calls += 1
        return self.answer

    async def cancel(self, turn_id: str, reason: str) -> None:
        del reason
        self.cancelled_turns.append(turn_id)

    async def close(self, reason: str) -> None:
        self.closed_reason = reason


@pytest.fixture
def fake_openai_client() -> FakeOpenAIClient:
    return FakeOpenAIClient()


def test_missing_api_key_selects_template_provider() -> None:
    provider = build_conversation_provider(Settings(openai_api_key=None))
    assert isinstance(provider, TemplateConversationProvider)


def test_provider_name_and_key_select_cloud_provider(fake_openai_client: FakeOpenAIClient) -> None:
    settings = Settings(conversation_provider="openai", openai_api_key="secret")
    provider = build_conversation_provider(settings, openai_client=fake_openai_client)
    assert isinstance(provider, OpenAIRealtimeProvider)


@pytest.mark.asyncio
async def test_ungrounded_cloud_answer_is_replaced_by_template(
    fake_openai_client: FakeOpenAIClient,
) -> None:
    fake_openai_client.answer = "The keys are on the shelf."
    provider = OpenAIRealtimeProvider(client=fake_openai_client, model="test-model")
    response = await provider.handle_grounded_turn(
        turn_id="turn-1",
        text="Where are my keys?",
        evidence_result=fake_openai_client.not_found_result,
    )
    assert response.status == "not_found"
    assert "do not have reliable evidence" in response.text


@pytest.mark.asyncio
async def test_grounded_cloud_answer_is_kept(fake_openai_client: FakeOpenAIClient) -> None:
    evidence = MemoryResult(
        status="found",
        canonical_label="keys",
        horizontal_region="right",
        depth_band="foreground",
        anchor_name="desk",
        observed_at_utc="2026-07-04T12:00:00Z",
        evidence_ids=tuple(str(index) for index in range(12)),
    )
    fake_openai_client.answer = "I last saw the keys on the right side of the desk."
    provider = OpenAIRealtimeProvider(client=fake_openai_client, model="test-model")
    response = await provider.handle_grounded_turn(
        turn_id="turn-2",
        text="Where are my keys?",
        evidence_result=evidence,
    )
    assert response.status == "found"
    assert response.evidence_ids == tuple(str(index) for index in range(10))


@pytest.mark.asyncio
async def test_interruption_cancels_provider(fake_openai_client: FakeOpenAIClient) -> None:
    provider = OpenAIRealtimeProvider(client=fake_openai_client, model="test-model")
    await provider.interrupt("turn-3", "barge-in")
    assert fake_openai_client.cancelled_turns == ["turn-3"]


@pytest.mark.asyncio
async def test_close_forwards_reason(fake_openai_client: FakeOpenAIClient) -> None:
    provider = OpenAIRealtimeProvider(client=fake_openai_client, model="test-model")
    await provider.close("shutdown")
    assert fake_openai_client.closed_reason == "shutdown"
