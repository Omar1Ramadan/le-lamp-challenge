import json
from pathlib import Path
from typing import Any

from social_lamp.domain.contracts import (
    BehaviorTimeline,
    MemoryResult,
    ObservationEvent,
    WorldSnapshot,
)

OUTPUT = Path("frontend/src/contracts/domain.schema.json")
MODELS = (WorldSnapshot, BehaviorTimeline, MemoryResult, ObservationEvent)


def _extract_defs(model_schema: dict[str, Any]) -> dict[str, Any]:
    nested = model_schema.pop("$defs", {})
    return dict(nested) | {str(model_schema["title"]): model_schema}


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    defs: dict[str, Any] = {}
    for model in MODELS:
        defs.update(_extract_defs(model.model_json_schema()))
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SocialLampContracts",
        "type": "object",
        "properties": {name: {"$ref": f"#/$defs/{name}"} for name in defs},
        "$defs": defs,
    }
    OUTPUT.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
