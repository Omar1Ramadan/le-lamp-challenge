import type { MemoryResult } from "../contracts/generated";

interface EvidenceTimelineProps {
  evidence: MemoryResult[];
}

export function EvidenceTimeline({ evidence }: EvidenceTimelineProps) {
  return (
    <section className="panel" aria-label="Evidence timeline">
      <h2>Evidence timeline</h2>
      {evidence.length === 0 ? (
        <p>No memory results recorded.</p>
      ) : (
        <ol>
          {evidence.map((item, index) => (
            <li key={`${item.canonical_label ?? "unknown"}-${index}`}>
              {item.canonical_label ?? "unknown"}: {item.status}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
