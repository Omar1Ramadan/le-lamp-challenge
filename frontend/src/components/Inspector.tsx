export interface InspectorEvidence {
  id: string;
  label: string;
  location: string;
}

export interface InspectorHealth {
  component: string;
  status: string;
  detail?: string | null;
}

interface InspectorProps {
  state: string;
  audioMode: string;
  evidence: InspectorEvidence[];
  health: InspectorHealth[];
}

export function Inspector({ state, audioMode, evidence, health }: InspectorProps) {
  return (
    <aside className="panel inspector" aria-label="Evidence inspector">
      <h2>Inspector</h2>
      <p>
        Social state: <strong>{state}</strong>
      </p>
      <p>
        Audio mode: <strong>{audioMode}</strong>
      </p>
      <section aria-label="Evidence">
        <h3>Evidence</h3>
        {evidence.length === 0 ? (
          <p>No grounded evidence yet.</p>
        ) : (
          <ul>
            {evidence.map((item) => (
              <li key={item.id}>
                <strong>{item.label}</strong>
                <span>{item.location}</span>
                <small>{item.id}</small>
              </li>
            ))}
          </ul>
        )}
      </section>
      <section aria-label="Health">
        <h3>Health</h3>
        {health.length === 0 ? (
          <p>All monitored components nominal.</p>
        ) : (
          <ul>
            {health.map((item) => (
              <li key={item.component}>
                <strong>{item.component}</strong>
                <span>Status: {item.status}</span>
                {item.detail ? <small>{item.detail}</small> : null}
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
