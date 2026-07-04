# Object Perception and Memory

## Responsibility

Detect and track visible objects, enrich stable unknown objects without blocking live perception, convert image evidence into scene-relative locations, persist world-model-approved append-only memories, and answer deterministic evidence-bearing queries. Perception publishes object observations; the world model determines stability and emits the memory-formation event consumed by memory.

## Cascaded Perception

### Fast path

A small object detector processes the newest eligible frame at a target rate of 8-12 Hz. It covers common household categories and returns class, confidence, and bounding box. A tracker associates detections using motion, overlap, and class compatibility.

A track is stable after at least five supporting detections within one second, mean confidence of at least `0.55`, and no class conflict above `0.25`. Unstable tracks appear in diagnostic overlays but do not create memories.

### Slow enrichment

Stable tracks with unknown, generic, or conflicting labels enter a capacity-two enrichment queue. Requests are coalesced by track. An open-vocabulary detector or vision-capable provider proposes labels from the cropped object and scene context. Results time out after two seconds, require confidence at least `0.60`, and can enrich but never erase the original evidence.

When enrichment is unavailable, the fast label or `unknown object` remains valid. Engagement processing never waits for enrichment.

## Scene-Relative Location

Every stable observation contains:

- Horizontal region: `left`, `center`, or `right`, based on calibrated image thirds.
- Apparent depth: `foreground`, `midground`, or `background`, based on object image scale and optional class-size prior. It is explicitly marked heuristic.
- Optional named anchor: calibrated image polygon such as `desk`, `shelf`, or `floor`.
- Bounding box in normalized image coordinates.
- Location confidence reflecting box stability, anchor overlap, and depth evidence.

If signals conflict, the answer uses only defensible fields. The product never reports metric coordinates without a future calibrated depth-capable adapter.

## Memory Formation Rules

The world model emits a memory-formation event and an append-only observation is written when:

- A track first becomes stable.
- Its horizontal region, depth band, or named anchor changes and remains changed for one second.
- Its accepted label is enriched or materially corrected.
- A stable track is observed again after 30 seconds, providing freshness without per-frame flooding.

The derived last-known table is updated in the same SQLite transaction. A persistence failure leaves the prior derived state intact and emits a fault.

## SQLite Model

```text
schema_meta(version, migrated_at_utc)
sessions(session_id, started_at_utc, ended_at_utc, source_mode, config_hash)
object_tracks(track_id, session_id, first_seen_utc, last_seen_utc, current_label,
              current_label_confidence, active)
observations(observation_id, track_id, session_id, observed_at_utc,
             observed_at_mono_ns, label, label_source, detection_confidence,
             bbox_json, horizontal_region, depth_band, anchor_name,
             location_confidence, frame_ref, snapshot_path, correlation_id)
last_known_objects(canonical_label, observation_id, updated_at_utc)
behavior_preferences(context_key, behavior_key, score, evidence_count, updated_at_utc)
```

Indexes cover normalized label and observation time, session and time, track and time, and anchor. Canonical labels are lowercase singular forms with an alias table held in application configuration. SQL parameters are always bound.

## Snapshot Policy

The saved snapshot is a JPEG with the evidence box and timestamp metadata stored separately. Raw full-session video is not retained by default. Snapshots remain local, default to seven-day retention, and can be deleted per session or globally. Deleting a snapshot leaves the structured observation but marks visual evidence unavailable.

## MemoryQuery

```text
MemoryQuery
  query_id, correlation_id, session_scope
  kind: last_seen | location | recent_history
  object_label: normalized user phrase
  before_utc: optional timestamp
  limit: integer [1, 20]

MemoryResult
  query_id
  status: found | ambiguous | not_found
  canonical_label: optional
  answer_fields: label, time, horizontal_region, depth_band, anchor_name
  evidence: list[observation_id, confidence, snapshot_path]
  alternatives: list[canonical_label]
```

Retrieval is deterministic: exact alias, canonical label, then SQLite full-text prefix matching. If top candidates are too close, return `ambiguous` with choices. The LLM does not select arbitrary database rows.

## Grounded Recall

A factual response may mention only populated `answer_fields`. It must attach at least one observation ID internally. If evidence is missing, stale beyond the configured horizon, or low-confidence, the response states that limitation. “Last seen” always means the newest qualifying observation within the requested scope, not the currently visible frame unless that frame has formed a memory.

## Failure and Degradation

- Detector unavailable: existing memory remains queryable; the UI reports no new observations.
- Enrichment unavailable: preserve fast labels and mark enrichment health degraded.
- Database locked: retry critical writes with short bounded backoff, then fault without blocking perception.
- Snapshot write failure: persist structured evidence with `snapshot_path=null` and a fault flag.
- Conflicting labels: retain all observations and avoid updating canonical state until stability rules resolve the conflict.
- Missing evidence: answer `not_found`; never synthesize location.

## Tests

- Track stability, movement-change, label-enrichment, and refresh formation rules.
- Region and anchor calculations at exact boundaries.
- Transaction rollback preserving prior last-known state.
- Alias, ambiguity, time-scope, and not-found query behavior.
- Evidence deletion and retention cleanup.
- Recall tests proving every factual field originates from selected observations.
- Replay scenarios with stationary, moved, disappeared, similar, unknown, and reappearing objects.

## References

- [YOLO-World: Real-Time Open-Vocabulary Object Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html)
- [Python SQLite interface](https://docs.python.org/3/library/sqlite3.html)
