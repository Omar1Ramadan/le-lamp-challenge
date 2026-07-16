import re

from social_lamp.domain.contracts import GroundingValidation, MemoryResult, ObservationSummary

_LOCATION_TOKENS = {
    "left", "center", "right", "foreground", "midground", "background",
    "shelf", "desk", "table", "cabinet", "floor", "wall",
}


def validate_grounding(text: str, evidence: MemoryResult) -> bool:
    result = _validate(text, evidence, ())
    return result.valid


def validate_grounding_detailed(
    text: str,
    evidence: MemoryResult,
    recent_observations: tuple[ObservationSummary, ...] = (),
) -> GroundingValidation:
    return _validate(text, evidence, recent_observations)


def _validate(
    text: str,
    evidence: MemoryResult,
    recent_observations: tuple[ObservationSummary, ...],
) -> GroundingValidation:
    text_lower = text.lower()
    tokens = set(re.findall(r"[a-z]+", text_lower))

    obs_evidence: tuple[str, ...] = ()
    for obs in recent_observations:
        if obs.summary.lower() in text_lower:
            obs_evidence = obs_evidence + obs.evidence_ids

    if obs_evidence:
        return GroundingValidation(
            valid=True,
            evidence_ids=evidence.evidence_ids + obs_evidence,
            reason="supported by recent observations",
        )

    if evidence.status == "not_found":
        unknown_phrases = (
            "do not", "don't", "no evidence", "not know",
            "do not have", "cannot answer", "can't answer",
        )
        has_unknown = any(phrase in text_lower for phrase in unknown_phrases)
        has_location_claim = bool(tokens & _LOCATION_TOKENS)
        if has_location_claim and not has_unknown:
            return GroundingValidation(
                valid=False,
                reason="answer claims location but evidence is not_found",
            )
        return GroundingValidation(
            valid=True,
            evidence_ids=(),
            reason="not_found acknowledged correctly" if has_unknown else None,
        )

    if evidence.status == "ambiguous":
        return GroundingValidation(
            valid=True,
            evidence_ids=evidence.evidence_ids + obs_evidence,
            reason="ambiguous results handled",
        )

    allowed_locations: set[str] = set()
    for value in (evidence.horizontal_region, evidence.depth_band, evidence.anchor_name):
        if value is not None:
            allowed_locations.add(value.lower())

    mentioned_locations = tokens & _LOCATION_TOKENS
    if mentioned_locations and not mentioned_locations.issubset(allowed_locations):
        return GroundingValidation(
            valid=False,
            evidence_ids=evidence.evidence_ids + obs_evidence,
            reason=f"location {mentioned_locations - allowed_locations} not in evidence",
        )

    if evidence.horizontal_region is not None and mentioned_locations:
        region_tokens = set(evidence.horizontal_region.lower().split())
        if not region_tokens & mentioned_locations:
            return GroundingValidation(
                valid=False,
                evidence_ids=evidence.evidence_ids + obs_evidence,
                reason=f"horizontal_region '{evidence.horizontal_region}' not mentioned",
            )

    if evidence.canonical_label and evidence.canonical_label.lower() not in text_lower:
        label_words = set(evidence.canonical_label.lower().split())
        if not label_words & tokens:
            return GroundingValidation(
                valid=False,
                evidence_ids=evidence.evidence_ids + obs_evidence,
                reason=f"canonical label '{evidence.canonical_label}' not in answer",
            )

    return GroundingValidation(
        valid=True,
        evidence_ids=evidence.evidence_ids + obs_evidence,
    )
