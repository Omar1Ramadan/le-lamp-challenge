interface DemoRailProps {
  metrics: Record<string, number>;
  needsResync: boolean;
}

export function DemoRail({ metrics, needsResync }: DemoRailProps) {
  return (
    <section className="panel" aria-label="Demo controls and metrics">
      <h2>Demo rail</h2>
      <p>{needsResync ? "Sequence gap detected; resync required." : "Sequence healthy."}</p>
      <dl>
        {Object.entries(metrics).map(([name, value]) => (
          <div key={name}>
            <dt>{name}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
