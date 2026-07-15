import type { ComponentHealth, ObjectState, PersonState } from "../contracts/generated";

interface PerceptionPanelProps {
  people: PersonState[];
  objects: ObjectState[];
  health: ComponentHealth[];
}

export function PerceptionPanel({ people, objects, health }: PerceptionPanelProps) {
  const objectDetectorHealth = health.find((h) => h.component === "object_detector");
  const isObjectDetectionDisabled = objectDetectorHealth?.status === "disabled";
  const isObjectDetectionDegraded = objectDetectorHealth?.status === "degraded";
  const isObjectDetectionActive = objectDetectorHealth?.status === "active";

  const objectSummary = isObjectDetectionDisabled
    ? "Object detection disabled"
    : isObjectDetectionDegraded
      ? "Object detector degraded"
      : `${objects.length} object${objects.length === 1 ? "" : "s"} tracked`;

  return (
    <section className="panel" aria-label="Perception state">
      <h2>Perception</h2>
      <p>{people.length} person{people.length === 1 ? "" : "s"} tracked</p>
      <p className={
        isObjectDetectionDisabled ? "status-disabled" :
        isObjectDetectionDegraded ? "status-degraded" : ""
      }>{objectSummary}</p>
      {objectDetectorHealth?.detail && (
        <p className="status-detail">{objectDetectorHealth.detail}</p>
      )}
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
