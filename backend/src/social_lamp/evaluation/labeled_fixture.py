from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EngagementSegment:
    person_id: str
    start_ms: int
    end_ms: int
    state: str


@dataclass
class ExpectedTransition:
    from_state: str
    to_state: str
    at_ms: int
    tolerance_ms: int = 750


@dataclass
class ExpectedMemory:
    type: str
    label: str
    within_ms: list[int]
    location: str | None = None


@dataclass
class ExpectedGroundedAnswer:
    query: str
    expected_answer_contains: list[str] = field(default_factory=list)
    required_evidence_types: list[str] = field(default_factory=list)
    required_memory_labels: list[str] = field(default_factory=list)


@dataclass
class LabeledFixture:
    fixture_id: str
    sample_only: bool
    description: str
    events: list[dict[str, Any]]
    labels: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> LabeledFixture:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            fixture_id=str(raw["fixture_id"]),
            sample_only=bool(raw.get("sample_only", False)),
            description=str(raw.get("description", "")),
            events=list(raw.get("events", [])),
            labels=dict(raw.get("labels", {})),
        )

    def engagement_segments(self) -> list[EngagementSegment]:
        raw = self.labels.get("engagement_segments", [])
        return [
            EngagementSegment(
                person_id=str(s.get("person_id", "person-1")),
                start_ms=int(s["start_ms"]),
                end_ms=int(s["end_ms"]),
                state=str(s["state"]),
            )
            for s in raw
        ]

    def expected_transitions(self) -> list[ExpectedTransition]:
        raw = self.labels.get("expected_transitions", [])
        return [
            ExpectedTransition(
                from_state=str(t["from_state"]),
                to_state=str(t["to_state"]),
                at_ms=int(t["at_ms"]),
                tolerance_ms=int(t.get("tolerance_ms", 750)),
            )
            for t in raw
        ]

    def expected_memories(self) -> list[ExpectedMemory]:
        raw = self.labels.get("expected_memories", [])
        return [
            ExpectedMemory(
                type=str(m["type"]),
                label=str(m.get("label", "")),
                within_ms=list(m.get("within_ms", [])),
                location=m.get("location"),
            )
            for m in raw
        ]

    def expected_grounded_answers(self) -> list[ExpectedGroundedAnswer]:
        raw = self.labels.get("expected_grounded_answers", [])
        return [
            ExpectedGroundedAnswer(
                query=str(g.get("query", "")),
                expected_answer_contains=list(g.get("expected_answer_contains", [])),
                required_evidence_types=list(g.get("required_evidence_types", [])),
                required_memory_labels=list(g.get("required_memory_labels", [])),
            )
            for g in raw
        ]


def load_fixtures(directory: Path) -> list[LabeledFixture]:
    if not directory.is_dir():
        return []
    fixtures: list[LabeledFixture] = []
    for child in sorted(directory.iterdir()):
        if child.suffix == ".json":
            fixtures.append(LabeledFixture.load(child))
    return fixtures


def split_fixtures(
    fixtures: list[LabeledFixture],
) -> tuple[list[LabeledFixture], list[LabeledFixture]]:
    labeled = [f for f in fixtures if not f.sample_only]
    sample = [f for f in fixtures if f.sample_only]
    return labeled, sample
