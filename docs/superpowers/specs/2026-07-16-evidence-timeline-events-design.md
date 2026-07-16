# Evidence Timeline Events for Live Operation

## Purpose

Make live operation inspectable by emitting evidence timeline events for engagement
transitions, behavior decisions, object memory writes, user queries, and faults.
Events preserve correlation IDs across subsystems, render as compact UI cards,
and link answers back to supporting evidence.

## Scope (Core Subset)

Implement 7 event types across 5 domains:

| Domain | Event Types |
|---|---|
| Engagement | `engagement_transition` |
| Behavior | `behavior_selected`, `behavior_suppressed`, `behavior_cancelled` |
| Memory | `object_memory_created` |
| Query | `query_received`, `answer_grounded` |
| Fault | `fault` |

## WebSocket Contract

### New Message Type

```json
{"seq": 42, "type": "evidence_event", "body": { ... EvidenceEvent ... }}
```

### EvidenceEvent Model

```python
class EvidenceEvent(FrozenModel):
    event_id: str
    event_type: str                  # enumeration of core types
    correlation_id: str | None       # parent correlation ID
    occurred_at_mono_ns: int
    source: str                      # "runtime" | "policy" | "conversation" | "audio"
    summary: str                     # human-readable one-liner
    severity: str                    # "info" | "warning" | "error"
    entity_refs: tuple[dict, ...] = ()    # [{kind, id, label}]
    evidence_refs: tuple[str, ...] = ()   # observation IDs
    metadata: dict[str, Any] = {}
```

## Architecture

### Coordinator-Centric Publisher Pattern

Add `evidence_publisher: Callable | None` parameter to `RuntimeCoordinator`,
parallel to the existing `snapshot_publisher`. The coordinator owns a private
`_emit_evidence()` method that builds `EvidenceEvent` instances and calls the
publisher.

```
live.py: hub.broadcast({"type": "evidence_event", "body": ...})
                 ↕
coordinator._emit_evidence()
    ↙        ↘          ↘            ↘
policy    vision      conversation   health
(decisions) (transitions + (queries +    (status
            memory writes)  answers)     transitions)
```

### Fix conversation_event Bug

`_emit_conversation_event` currently calls `_snapshot_publisher`, which wraps
the payload as `world_snapshot` type, corrupting the frontend world state.
Fix: route conversation events through `evidence_publisher` instead.

### Policy Integration

The policy itself does not know about evidence events. After calling
`policy.evaluate()`, the coordinator reads the returned `BehaviorDecision`
and emits `behavior_selected` or `behavior_suppressed` accordingly.
If `decision.replacement` is set, a `behavior_cancelled` event is emitted
for the replaced timeline kind.

## Emission Points

| Event Type | Emitter | Condition |
|---|---|---|
| `engagement_transition` | `process_vision_frame` | Social state changes vs previous |
| `behavior_selected` | After `simulator.execute()` | decision.intent and not suppressed |
| `behavior_suppressed` | After `policy.evaluate()` | decision.suppressed == True |
| `behavior_cancelled` | When replacement occurs | decision.replacement is set |
| `object_memory_created` | After `memory.record()` | should_record returned True |
| `query_received` | `submit_text` entry | On every submit |
| `answer_grounded` | After conversation returns | response.grounded == True |
| `fault` | `_set_health` | On component status transitions |

## Frontend Changes

### Store (`store.ts`)

- Add `EvidenceEvent` type to `ServerMessage` union
- Add `evidence_events: EvidenceEvent[]` to `DashboardState`
- Reducer: append on `evidence_event`, deduped by `event_id`
- Cap array at 500 events (FIFO eviction)

### UI (`EvidenceTimeline.tsx`)

- Accept and render `EvidenceEvent[]` alongside existing `MemoryResult[]`
- Card groups with type badge, severity indicator, summary, entity refs
- Layout: chronological ordered by arrival, compact single-line cards
- Graceful handling of unknown event types

## Deduplication Rules

- `engagement_transition`: only emit when social_state actually changes
- `object_memory_created`: same dedupe rules as memory writes (on stable tracks)
- `fault`: only emit on status *transitions* (ok→degraded, degraded→ok, etc.)
- `behavior_cancelled`: only when a replacement actually happens

## Testing

### Backend (`tests/runtime/test_evidence_events.py`)

- engagement transition emits one event per state change
- repeated stable state does not flood
- behavior selected emits event with kind and correlation_id
- suppressed behavior emits suppression_reason in metadata
- object memory write emits object_memory_created
- query + answer emits query_received + answer_grounded
- health transition emits fault event
- duplicate health status does NOT emit duplicate fault event
- correlation_id persists through perception→behavior

### Frontend (update existing test files)

- evidence_event reducer appends to evidence_events
- Timeline renders engagement card
- Timeline renders behavior card
- Timeline renders fault card
- Timeline handles unknown event_type gracefully
