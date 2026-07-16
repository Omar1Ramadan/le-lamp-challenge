import type { MemoryResult } from "../contracts/generated";
import type { EvidenceEvent } from "../state/store";

interface EvidenceTimelineProps {
  evidence: MemoryResult[];
  evidenceEvents: EvidenceEvent[];
}

const EVENT_LABELS: Record<string, string> = {
  engagement_transition: "ENGAGEMENT",
  behavior_selected: "BEHAVIOR",
  behavior_suppressed: "SUPPRESSED",
  behavior_cancelled: "CANCELLED",
  object_memory_created: "MEMORY",
  query_received: "QUERY",
  answer_grounded: "ANSWER",
  fault: "FAULT",
};

function eventCard(event: EvidenceEvent) {
  const label = EVENT_LABELS[event.event_type] ?? event.event_type;
  const severityClass =
    event.severity === "error"
      ? "event-error"
      : event.severity === "warning"
        ? "event-warning"
        : "event-info";
  return (
    <li key={event.event_id} className={`evidence-card ${severityClass}`} role="article" aria-label={`event: ${event.event_type}`}>
      <span className="event-badge">{label}</span>
      <span className="event-summary">{event.summary}</span>
      {event.entity_refs.length > 0 ? (
        <span className="event-entities">
          {event.entity_refs.map((ref) => {
            const r = ref as { kind?: string; id?: string; label?: string };
            return <small key={r.id ?? ""}>{r.label ?? r.id}</small>;
          })}
        </span>
      ) : null}
    </li>
  );
}

function memoryCard(item: MemoryResult, index: number) {
  const location = [item.horizontal_region, item.depth_band, item.anchor_name]
    .filter(Boolean)
    .join(" / ");
  return (
    <li key={`${item.canonical_label ?? "unknown"}-${index}`} className="evidence-card event-info" role="article" aria-label={`memory: ${item.canonical_label ?? "unknown"}`}>
      <span className="event-badge">MEMORY</span>
      <span className="event-summary">
        {item.canonical_label ?? "unknown"}: {item.status}
        {item.status === "found" && location ? ` — ${location}` : ""}
      </span>
    </li>
  );
}

export function EvidenceTimeline({ evidence, evidenceEvents = [] }: EvidenceTimelineProps) {
  const allItems = [
    ...evidenceEvents.map((e) => ({ kind: "event" as const, event: e })),
    ...evidence.map((m, i) => ({ kind: "memory" as const, memory: m as MemoryResult, index: i })),
  ];

  return (
    <section className="panel" aria-label="Evidence timeline">
      <h2>Evidence timeline</h2>
      {allItems.length === 0 ? (
        <p>No evidence recorded.</p>
      ) : (
        <ol className="evidence-list">
          {allItems.map((item) =>
            item.kind === "event" ? eventCard(item.event) : memoryCard(item.memory, item.index),
          )}
        </ol>
      )}
    </section>
  );
}
