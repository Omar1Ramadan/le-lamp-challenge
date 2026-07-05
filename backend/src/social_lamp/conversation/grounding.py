import re

from social_lamp.domain.contracts import MemoryResult

LOCATION_TOKENS = {
    "left",
    "center",
    "right",
    "foreground",
    "midground",
    "background",
    "shelf",
    "desk",
}


def validate_grounding(text: str, evidence: MemoryResult) -> bool:
    text_lower = text.lower()
    normalized = set(re.findall(r"[a-z]+", text_lower))
    if evidence.status == "not_found":
        phrases = ("do not", "don't", "no evidence", "not know", "do not have")
        return any(phrase in text_lower for phrase in phrases) and not (
            normalized & LOCATION_TOKENS
        )

    allowed = {
        value.lower()
        for value in (
            evidence.horizontal_region,
            evidence.depth_band,
            evidence.anchor_name,
        )
        if value is not None
    }
    mentioned_locations = normalized & LOCATION_TOKENS
    if not mentioned_locations.issubset(allowed):
        return False
    if evidence.anchor_name is not None:
        return evidence.anchor_name.lower() in text_lower or not mentioned_locations
    return bool(evidence.evidence_ids)
