import type { ObjectState, PersonState } from "../contracts/generated";

interface PerceptionPanelProps {
  people: PersonState[];
  objects: ObjectState[];
}

export function PerceptionPanel({ people, objects }: PerceptionPanelProps) {
  return (
    <section className="panel" aria-label="Perception state">
      <h2>Perception</h2>
      <p>{people.length} people tracked</p>
      <p>{objects.length} objects tracked</p>
      <ul>
        {people.map((person) => (
          <li key={person.person_id}>
            {person.person_id} · engagement {Math.round(person.engagement_score * 100)}% ·
            confidence {Math.round(person.engagement_confidence * 100)}%
          </li>
        ))}
      </ul>
      <ul>
        {objects.map((object) => (
          <li key={object.track_id}>
            {object.label} · {object.horizontal_region ?? "unknown"}
          </li>
        ))}
      </ul>
    </section>
  );
}
