# Evaluation

The evaluation framework measures system behavior against explicit labeled fixtures,
replacing the previous approach of comparing replay output against itself.

## Fixture Schema

Fixtures live in `tests/fixtures/evaluation/`. There are two directories:

- `labeled/` — ground-truth fixtures that produce headline metrics
- `sample/` — demo fixtures excluded from aggregate metrics

### JSON Structure

```json
{
  "fixture_id": "engagement_basic_001",
  "sample_only": false,
  "description": "Person becomes engaged after facing lamp.",
  "events": [
    {"t_ms": 0, "record_type": "snapshot", "body": {"revision": 1, "social_state": "idle"}},
    {"t_ms": 800, "record_type": "snapshot", "body": {"revision": 2, "social_state": "engaged"}}
  ],
  "labels": {
    "engagement_segments": [
      {"person_id": "person-1", "start_ms": 0, "end_ms": 800, "state": "idle"},
      {"person_id": "person-1", "start_ms": 800, "end_ms": 2000, "state": "engaged"}
    ],
    "expected_transitions": [
      {"from_state": "idle", "to_state": "engaged", "at_ms": 800, "tolerance_ms": 200}
    ],
    "expected_memories": [
      {"type": "memory_result", "label": "keys", "within_ms": [0, 5000], "location": "desk"}
    ],
    "expected_grounded_answers": [
      {
        "query": "Where are my keys?",
        "expected_answer_contains": ["desk"],
        "required_evidence_types": ["object_seen"],
        "required_memory_labels": ["keys"]
      }
    ]
  }
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `fixture_id` | string | yes | Unique identifier |
| `sample_only` | boolean | yes | `true` excludes from headline metrics |
| `description` | string | no | Human-readable summary |
| `events` | array | yes | Replay event timeline |
| `labels` | object | yes | Expected outcomes |

#### Event format

Each event in `events[]`:
- `t_ms` — relative timestamp from fixture start
- `record_type` — `snapshot`, `observation`, `memory`, `memory_result`, `intent`, `timeline`, `bonus`
- `body` — matching the trace record body format

#### Label types

**engagement_segments**: Time intervals mapping to expected social state.
- `person_id`, `start_ms`, `end_ms`, `state`

**expected_transitions**: Expected state changes with tolerance window.
- `from_state`, `to_state`, `at_ms`, `tolerance_ms` (default 750ms)

**expected_memories**: Expected memory records.
- `type` (e.g. `memory_result`, `object_seen`), `label`, `within_ms`, `location`

**expected_grounded_answers**: Expected conversation evidence.
- `query`, `expected_answer_contains`, `required_evidence_types`, `required_memory_labels`

## Metric Definitions

### Engagement (F1)
- Precision, recall, F1 over binary engaged/not-engaged classification
- Evaluated at 250ms intervals across the fixture timeline
- Confusion matrix reported per fixture
- Reports should preserve engagement calibration state when present, including calibration state, quality, sample count, and whether scoring used fallback, partial, or calibrated mode.

### False Transitions
- True positives: expected transitions observed within tolerance window
- False positives: observed transitions with no matching expected transition
- False negatives: expected transitions not observed
- Reported as false transitions per minute

### Latency
- P50, P95, max latency from input event to corresponding output
- Missing output count

### Memory Accuracy
- Precision, recall, F1 over expected vs observed memories
- Extra memory count reported

### Grounding Rate
- Fraction of expected grounded answers with all required evidence
- Requires supporting evidence types and memory labels, not just text match

## How to Add a Labeled Fixture

1. Create a JSON file in `tests/fixtures/evaluation/labeled/`
2. Set `"sample_only": false`
3. Define replay events and expected labels
4. Run evaluation:

```bash
uv run python -m social_lamp.evaluation.cli evaluate \
  --fixtures-dir tests/fixtures/evaluation/labeled \
  --output reports/evaluation/$(date +%s)
```

5. Check the generated report in `reports/evaluation/`

## How to Run Evaluation Locally

```bash
# Run evaluation tests
uv run pytest tests/evaluation -v

# Run full evaluation on labeled fixtures
uv run python -m social_lamp.evaluation.cli evaluate \
  --fixtures-dir tests/fixtures/evaluation/labeled \
  --output reports/evaluation/current

# Legacy single-fixture evaluation
uv run python -m social_lamp.evaluation.cli fixture \
  --fixture evaluation/fixtures/core-engagement \
  --output reports/evaluation/legacy
```

## Sample vs Labeled Fixtures

- **Sample fixtures** (`sample_only: true`) are for demonstration and report generation testing only
- **Labeled fixtures** (`sample_only: false`) produce headline aggregate metrics
- Reports explicitly state: "Sample-only fixtures excluded from aggregate metrics"
- Evaluation will fail or warn if no labeled fixtures are present

## Known Limitations

- Engagement evaluation uses binary (engaged/not_engaged) classification; multi-class support is not yet implemented
- Grounding metrics depend on evidence types being populated in output messages
- Latency metrics use simple event matching (type-based, not content-based)
- Transition tolerance is uniform per fixture; per-transition tolerance can be customized
