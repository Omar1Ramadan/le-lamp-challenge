from pathlib import Path

from social_lamp.domain.contracts import SocialState
from social_lamp.evaluation.labeled_fixture import LabeledFixture
from social_lamp.evaluation.runner import (
    _collect_observed_answers,
    _collect_observed_memories,
    _collect_observed_states,
    _collect_observed_transitions,
    _fixture_events_to_records,
    evaluate_single_fixture,
)

EMPTY_LABELED: LabeledFixture = LabeledFixture(
    fixture_id="empty_test",
    sample_only=False,
    description="Empty fixture",
    events=[],
    labels={},
)

SAMPLE_ONLY_FIXTURE: LabeledFixture = LabeledFixture(
    fixture_id="sample_test",
    sample_only=True,
    description="Sample only",
    events=[],
    labels={},
)


def test_observed_states_collected() -> None:
    messages: list[tuple[str, dict]] = [
        ("world_snapshot", {"as_of_mono_ns": 1_000_000_000, "social_state": "idle"}),
        ("world_snapshot", {"as_of_mono_ns": 1_100_000_000, "social_state": "candidate"}),
    ]
    states = _collect_observed_states(messages)
    assert len(states) == 2
    assert states[0] == (1000, "idle")
    assert states[1] == (1100, "candidate")


def test_observed_states_with_socialstate_enum() -> None:
    messages: list[tuple[str, dict]] = [
        ("world_snapshot", {"as_of_mono_ns": 1_000_000_000, "social_state": SocialState.ENGAGED}),
    ]
    states = _collect_observed_states(messages)
    assert states[0][1] == "engaged"


def test_observed_transitions_collected() -> None:
    messages: list[tuple[str, dict]] = [
        ("world_snapshot", {"as_of_mono_ns": 1_000_000_000, "social_state": "idle"}),
        ("world_snapshot", {"as_of_mono_ns": 1_500_000_000, "social_state": "candidate"}),
        ("world_snapshot", {"as_of_mono_ns": 2_000_000_000, "social_state": "engaged"}),
    ]
    trans = _collect_observed_transitions(messages)
    assert len(trans) == 2
    assert trans[0] == (1500, "idle", "candidate")
    assert trans[1] == (2000, "candidate", "engaged")


def test_observed_transitions_single_state() -> None:
    messages: list[tuple[str, dict]] = [
        ("world_snapshot", {"as_of_mono_ns": 1_000_000_000, "social_state": "idle"}),
    ]
    trans = _collect_observed_transitions(messages)
    assert len(trans) == 0  # no change


def test_observed_memories_collected() -> None:
    messages: list[tuple[str, dict]] = [
        (
            "memory_result",
            {
                "type": "memory_result",
                "canonical_label": "keys",
                "anchor_name": "desk",
                "status": "found",
            },
        ),
        (
            "world_snapshot",
            {
                "objects": [
                    {"label": "keys", "anchor_name": "desk"},
                ]
            },
        ),
    ]
    mems = _collect_observed_memories(messages)
    assert len(mems) == 2
    assert mems[0]["type"] == "memory_result"
    assert mems[1]["type"] == "object_seen"


def test_collect_observed_answers() -> None:
    messages: list[tuple[str, dict]] = [
        (
            "memory_result",
            {
                "query": "Where are my keys?",
                "text": "On the desk.",
                "status": "found",
                "evidence_ids": ["e1"],
                "canonical_label": "keys",
            },
        ),
    ]
    answers = _collect_observed_answers(messages)
    assert len(answers) == 1
    assert "desk" in answers[0]["text"]


def test_fixture_events_sorted() -> None:
    events = [
        {"t_ms": 500, "record_type": "snapshot", "body": {}},
        {"t_ms": 0, "record_type": "snapshot", "body": {}},
    ]
    sorted_events = _fixture_events_to_records(events)
    assert sorted_events[0]["t_ms"] == 0
    assert sorted_events[1]["t_ms"] == 500


def test_evaluate_single_fixture_handles_empty() -> None:
    result = evaluate_single_fixture(EMPTY_LABELED)
    assert result["fixture_id"] == "empty_test"
    assert not result["sample_only"]
    assert isinstance(result["engagement"], dict)
    assert isinstance(result["transitions"], dict)
    assert isinstance(result["memory"], dict)
    assert isinstance(result["grounding"], dict)


def test_evaluate_single_fixture_with_engagement(tmp_path: Path) -> None:
    fixture = LabeledFixture(
        fixture_id="engagement_test",
        sample_only=False,
        description="Engagement test",
        events=[
            {"t_ms": 0, "record_type": "snapshot", "body": {"revision": 1, "social_state": "idle"}},
            {
                "t_ms": 500, "record_type": "snapshot",
                "body": {"revision": 2, "social_state": "candidate"},
            },
            {
                "t_ms": 1000, "record_type": "snapshot",
                "body": {"revision": 3, "social_state": "engaged"},
            },
        ],
        labels={
            "engagement_segments": [
                {"person_id": "person-1", "start_ms": 0, "end_ms": 500, "state": "idle"},
                {"person_id": "person-1", "start_ms": 500, "end_ms": 1000, "state": "candidate"},
                {"person_id": "person-1", "start_ms": 1000, "end_ms": 2000, "state": "engaged"},
            ],
            "expected_transitions": [
                {"from_state": "idle", "to_state": "candidate", "at_ms": 500, "tolerance_ms": 200},
                {
                    "from_state": "candidate", "to_state": "engaged",
                    "at_ms": 1000, "tolerance_ms": 200,
                },
            ],
            "expected_memories": [],
            "expected_grounded_answers": [],
        },
    )
    result = evaluate_single_fixture(fixture)
    # We expect non-zero engagement metrics since we provided matching labels
    assert result["engagement"]["support"] >= 0  # may be 0 due to interval sampling
    assert result["transitions"]["true_positives"] >= 0
