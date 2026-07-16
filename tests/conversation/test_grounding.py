from social_lamp.conversation.grounding import (
    validate_grounding,
    validate_grounding_detailed,
)
from social_lamp.domain.contracts import MemoryResult, ObservationSummary


def test_grounding_found_with_valid_location() -> None:
    evidence = MemoryResult(
        status="found",
        canonical_label="keys",
        horizontal_region="right",
        anchor_name="desk",
        evidence_ids=("obs-1",),
    )
    assert validate_grounding("I last saw the keys on the right side of the desk.", evidence)


def test_grounding_found_with_unsupported_location() -> None:
    evidence = MemoryResult(
        status="found",
        canonical_label="keys",
        horizontal_region="left",
        evidence_ids=("obs-1",),
    )
    assert not validate_grounding("Your keys are on the shelf.", evidence)


def test_grounding_not_found_returns_unknown() -> None:
    evidence = MemoryResult.not_found()
    assert validate_grounding("I do not have reliable evidence for that.", evidence)


def test_grounding_not_found_but_claims_location() -> None:
    evidence = MemoryResult.not_found()
    result = validate_grounding_detailed("Your wallet is on the shelf.", evidence)
    assert not result.valid
    assert result.reason is not None


def test_grounding_ambiguous_passes() -> None:
    evidence = MemoryResult(status="ambiguous", alternatives=("keys", "keyboard"))
    assert validate_grounding("I found more than one possible match: keys, keyboard.", evidence)


def test_grounding_detailed_returns_evidence_ids() -> None:
    evidence = MemoryResult(
        status="found",
        canonical_label="cup",
        horizontal_region="right",
        anchor_name="desk",
        evidence_ids=("obs-1",),
    )
    result = validate_grounding_detailed(
        "I last saw the cup on the right side of the desk.", evidence
    )
    assert result.valid
    assert "obs-1" in result.evidence_ids


def test_grounding_detailed_rejects_wrong_label() -> None:
    evidence = MemoryResult(
        status="found",
        canonical_label="keys",
        horizontal_region="right",
        evidence_ids=("obs-1",),
    )
    result = validate_grounding_detailed("I last saw the wallet on the right side.", evidence)
    assert not result.valid


def test_grounding_uses_recent_observations() -> None:
    evidence = MemoryResult(status="not_found")
    observations = (
        ObservationSummary(
            id="obs-1",
            type="object_seen",
            summary="cup on the right side of the desk",
            observed_at_utc="2026-07-04T12:00:00Z",
            evidence_ids=("obs-1",),
        ),
    )
    result = validate_grounding_detailed(
        "I saw a cup on the right side of the desk recently.", evidence, observations
    )
    assert result.valid
    assert "obs-1" in result.evidence_ids
