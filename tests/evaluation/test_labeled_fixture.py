from pathlib import Path

from social_lamp.evaluation.labeled_fixture import (
    LabeledFixture,
    load_fixtures,
    split_fixtures,
)

SAMPLE_FIXTURE = {
    "fixture_id": "test_001",
    "sample_only": True,
    "description": "A sample fixture",
    "events": [
        {"t_ms": 0, "record_type": "snapshot", "body": {"revision": 1, "social_state": "idle"}}
    ],
    "labels": {
        "engagement_segments": [
            {"person_id": "person-1", "start_ms": 0, "end_ms": 1000, "state": "idle"}
        ]
    },
}

LABELED_FIXTURE = {
    "fixture_id": "test_002",
    "sample_only": False,
    "description": "A labeled fixture",
    "events": [
        {"t_ms": 0, "record_type": "snapshot", "body": {"revision": 1, "social_state": "idle"}},
        {
            "t_ms": 500, "record_type": "snapshot",
            "body": {"revision": 2, "social_state": "engaged"},
        },
    ],
    "labels": {
        "engagement_segments": [
            {"person_id": "person-1", "start_ms": 500, "end_ms": 2000, "state": "engaged"}
        ],
        "expected_transitions": [
            {"from_state": "idle", "to_state": "engaged", "at_ms": 500, "tolerance_ms": 200}
        ],
        "expected_memories": [
            {"type": "memory_result", "label": "keys", "within_ms": [0, 5000], "location": "desk"}
        ],
        "expected_grounded_answers": [
            {
                "query": "Where are my keys?",
                "expected_answer_contains": ["desk"],
                "required_evidence_types": ["object_seen"],
                "required_memory_labels": ["keys"],
            }
        ],
    },
}


def test_load_labeled_fixture(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    import json

    path.write_text(json.dumps(LABELED_FIXTURE), encoding="utf-8")
    fixture = LabeledFixture.load(path)
    assert fixture.fixture_id == "test_002"
    assert not fixture.sample_only
    assert len(fixture.events) == 2
    assert len(fixture.labels["engagement_segments"]) == 1


def test_engagement_segments_parsed() -> None:
    fixture = LabeledFixture(
        fixture_id="test",
        sample_only=False,
        description="test",
        events=[],
        labels={
            "engagement_segments": [
                {"person_id": "p1", "start_ms": 0, "end_ms": 100, "state": "engaged"}
            ]
        },
    )
    segments = fixture.engagement_segments()
    assert len(segments) == 1
    assert segments[0].person_id == "p1"
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 100
    assert segments[0].state == "engaged"


def test_expected_transitions_parsed() -> None:
    fixture = LabeledFixture(
        fixture_id="test",
        sample_only=False,
        description="test",
        events=[],
        labels={
            "expected_transitions": [
                {"from_state": "idle", "to_state": "engaged", "at_ms": 500, "tolerance_ms": 200}
            ]
        },
    )
    transitions = fixture.expected_transitions()
    assert len(transitions) == 1
    assert transitions[0].from_state == "idle"
    assert transitions[0].to_state == "engaged"
    assert transitions[0].at_ms == 500
    assert transitions[0].tolerance_ms == 200


def test_expected_memories_parsed() -> None:
    fixture = LabeledFixture(
        fixture_id="test",
        sample_only=False,
        description="test",
        events=[],
        labels={
            "expected_memories": [
                {
                    "type": "object_seen", "label": "keys",
                    "within_ms": [0, 10000], "location": "desk",
                }
            ]
        },
    )
    memories = fixture.expected_memories()
    assert len(memories) == 1
    assert memories[0].type == "object_seen"
    assert memories[0].label == "keys"
    assert memories[0].location == "desk"


def test_expected_grounded_answers_parsed() -> None:
    fixture = LabeledFixture(
        fixture_id="test",
        sample_only=False,
        description="test",
        events=[],
        labels={
            "expected_grounded_answers": [
                {
                    "query": "Where are my keys?",
                    "expected_answer_contains": ["desk"],
                    "required_evidence_types": ["object_seen"],
                    "required_memory_labels": ["keys"],
                }
            ]
        },
    )
    answers = fixture.expected_grounded_answers()
    assert len(answers) == 1
    assert answers[0].query == "Where are my keys?"
    assert "desk" in answers[0].expected_answer_contains


def test_load_fixtures_from_directory(tmp_path: Path) -> None:
    import json

    (tmp_path / "a.json").write_text(json.dumps(LABELED_FIXTURE), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(SAMPLE_FIXTURE), encoding="utf-8")
    fixtures = load_fixtures(tmp_path)
    assert len(fixtures) == 2


def test_split_fixtures_separates_sample_only() -> None:
    labeled_f = LabeledFixture(
        fixture_id="l1", sample_only=False, description="", events=[], labels={}
    )
    sample_f = LabeledFixture(
        fixture_id="s1", sample_only=True, description="", events=[], labels={}
    )
    labeled, sample = split_fixtures([labeled_f, sample_f])
    assert len(labeled) == 1
    assert len(sample) == 1
    assert labeled[0].fixture_id == "l1"
    assert sample[0].fixture_id == "s1"


def test_default_tolerance_when_missing() -> None:
    fixture = LabeledFixture(
        fixture_id="test",
        sample_only=False,
        description="test",
        events=[],
        labels={
            "expected_transitions": [
                {"from_state": "idle", "to_state": "engaged", "at_ms": 500}
            ]
        },
    )
    transitions = fixture.expected_transitions()
    assert transitions[0].tolerance_ms == 750


def test_empty_fixture_directory(tmp_path: Path) -> None:
    fixtures = load_fixtures(tmp_path)
    assert fixtures == []
