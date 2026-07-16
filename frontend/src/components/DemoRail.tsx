interface DemoRailProps {
  metrics: Record<string, number>;
  needsResync: boolean;
  connection: string;
}

export function DemoRail({ metrics, needsResync, connection }: DemoRailProps) {
  const labels: Record<string, string> = {
    active_speaker_person_b: "Active speaker: Person B",
    affect_confidence_gated: "Affect confidence gated below 0.60",
    preference_score_changed_reset: "Preference score changed then reset",
    speech_interruption_cancelled: "Speech interruption cancellation under 120 ms",
    television_suppression_active: "Television suppression active",
  };

  return (
    <section className="panel" aria-label="Demo controls and metrics">
      <h2>Demo rail</h2>
      <p>
        {needsResync ? "⚠ Sequence gap — awaiting resync" : "✓ Sequence healthy"}
      </p>
      {connection === "resyncing" && <p>⟳ Resyncing from backend...</p>}
      {connection === "frozen" && <p>⏸ State frozen — stale data</p>}
      <dl>
        {Object.entries(metrics).map(([name, value]) => (
          <div key={name}>
            <dt>{labels[name] ?? name}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
