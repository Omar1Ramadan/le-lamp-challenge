import type { MemoryResult } from "../contracts/generated";

interface EvidenceTimelineProps {
  evidence: MemoryResult[];
}

export function EvidenceTimeline({ evidence }: EvidenceTimelineProps) {
  const location = (item: MemoryResult) =>
    [
      item.horizontal_region ? `${item.horizontal_region} side` : "",
      item.anchor_name ? `of the ${item.anchor_name}` : "",
    ]
      .filter(Boolean)
      .join(" ");

  return (
    <section className="panel" aria-label="Evidence timeline">
      <h2>Evidence timeline</h2>
      {evidence.length === 0 ? (
        <p>No memory results recorded.</p>
      ) : (
        <ol>
          {evidence.map((item, index) => (
            <li key={`${item.canonical_label ?? "unknown"}-${index}`}>
              <article aria-label={`memory: ${item.canonical_label ?? "unknown"}`}>
                <strong>{item.canonical_label ?? "unknown"}</strong>: {item.status}
                {item.status === "found" ? <span> — {location(item)}</span> : null}
                {(item.evidence_ids ?? []).map((id) => (
                  <small key={id}> {id}</small>
                ))}
              </article>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
