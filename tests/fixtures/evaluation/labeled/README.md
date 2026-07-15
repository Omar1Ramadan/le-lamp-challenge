# Labeled Evaluation Fixtures

Each JSON file defines a self-contained evaluation scenario with:

- `fixture_id` — unique string identifier
- `sample_only` — must be `false` for labeled fixtures
- `description` — human-readable what this fixture tests
- `events` — timeline of replay events (t_ms relative, record_type + body)
- `labels` — expected outcomes:
  - `engagement_segments`: time intervals with expected social state
  - `expected_transitions`: state changes with tolerance windows
  - `expected_memories`: expected memory records
  - `expected_grounded_answers`: expected conversation evidence

Add new fixtures here to expand the evaluation coverage.
