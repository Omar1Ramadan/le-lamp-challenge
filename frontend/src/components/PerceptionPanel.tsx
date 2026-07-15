import type { ComponentHealth, ObjectState, PersonState } from "../contracts/generated";
import {
  shortDetectorName,
  badgeForStatus,
  isHeuristic,
  isDisabled,
  isDegraded,
  type VisionStatus,
} from "../lib/vision";

interface PerceptionPanelProps {
  people: PersonState[];
  objects: ObjectState[];
  health: ComponentHealth[];
  visionStatus?: VisionStatus | null;
}

function StatusBadge({ status, detail }: { status: string; detail?: string | null }) {
  const badge = badgeForStatus(status, detail);
  return (
    <span className={`status-badge ${badge.className}`} aria-label={`Status: ${badge.label}`}>
      {badge.label}
    </span>
  );
}

function DetectorLine({
  label,
  detector,
}: {
  label: string;
  detector: { name: string; status: string; detail?: string | null } | null | undefined;
}) {
  if (!detector) return null;
  const short = shortDetectorName(detector.name);
  return (
    <p
      className={
        isDisabled(detector.status)
          ? "status-disabled"
          : isDegraded(detector.status)
            ? "status-degraded"
            : ""
      }
    >
      {label}: <strong>{short}</strong> <StatusBadge status={detector.status} detail={detector.detail} />
      {detector.detail ? <span className="status-detail"> — {detector.detail}</span> : null}
    </p>
  );
}

export function PerceptionPanel({ people, objects, health, visionStatus }: PerceptionPanelProps) {
  const faceHealth = health.find((h) => h.component === "face_detector");
  const objectHealth = health.find((h) => h.component === "object_detector");

  const faceDetectorInfo =
    visionStatus?.face_detector ??
    (faceHealth
      ? { name: faceHealth.detail ?? "unknown", status: faceHealth.status, detail: faceHealth.detail }
      : null);

  const objectDetectorInfo =
    visionStatus?.object_detector ??
    (objectHealth
      ? {
          name: objectHealth.detail ?? "object_detector",
          status: objectHealth.status,
          detail: objectHealth.detail,
        }
      : null);

  const isObjectDisabled = objectDetectorInfo?.status === "disabled";
  const isObjectDegraded = objectDetectorInfo?.status === "degraded";
  const isFaceDisabled = faceDetectorInfo?.status === "disabled";
  const isFaceHeuristic = faceDetectorInfo != null && isHeuristic(faceDetectorInfo.name);

  const objectsSummary = isObjectDisabled
    ? "Object detection disabled"
    : isObjectDegraded
      ? "Object detector degraded"
      : objects.length === 0
        ? "0 objects detected"
        : `${objects.length} object${objects.length === 1 ? "" : "s"} tracked`;

  const peopleSummary = isFaceDisabled
    ? "Face detection disabled"
    : isFaceHeuristic
      ? "Possible person detection only — low reliability"
      : people.length === 0
        ? "0 people detected"
        : `${people.length} person${people.length === 1 ? "" : "s"} tracked`;

  return (
    <section className="panel" aria-label="Perception state">
      <h2>Perception</h2>

      <DetectorLine label="Face" detector={faceDetectorInfo} />
      <DetectorLine label="Objects" detector={objectDetectorInfo} />

      <p
        className={
          isFaceDisabled || isFaceHeuristic
            ? "status-degraded"
            : ""
        }
      >
        {peopleSummary}
      </p>
      <p
        className={
          isObjectDisabled ? "status-disabled" : isObjectDegraded ? "status-degraded" : ""
        }
      >
        {objectsSummary}
      </p>

      {faceDetectorInfo?.detail && isDegraded(faceDetectorInfo.status) && (
        <p className="status-detail">{faceDetectorInfo.detail}</p>
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
