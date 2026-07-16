import type {
  ComponentHealth,
  EngagementCalibrationSnapshot,
  ObjectState,
  PersonState,
} from "../contracts/generated";
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
  primaryPersonId?: string | null;
  engagementCalibration?: EngagementCalibrationSnapshot | null;
  onStartEngagementCalibration?: () => void;
  onCancelEngagementCalibration?: () => void;
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

export function PerceptionPanel({
  people,
  objects,
  health,
  visionStatus,
  primaryPersonId,
  engagementCalibration,
  onStartEngagementCalibration,
  onCancelEngagementCalibration,
}: PerceptionPanelProps) {
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
        : `${people.length} ${people.length === 1 ? "person" : "people"} tracked`;

  const calibration = engagementCalibration ?? {
    state: "uncalibrated",
    person_id: null,
    sample_count: 0,
    quality: "unavailable",
    failure_reason: null,
    mode: "fallback",
    progress: 0,
  };
  const calibrationProgress = Math.round((calibration.progress ?? 0) * 100);
  const calibrationPersonLabel =
    calibration.state === "calibrated"
      ? "Calibrated"
      : calibration.state === "calibrating"
        ? "Calibrating"
        : "Person";

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

      <section className="perception-calibration" aria-label="Engagement calibration">
        <h3>Engagement calibration</h3>
        <p>Calibration: {calibration.state}</p>
        <p>Mode: {calibration.mode}</p>
        {calibration.state === "calibrating" ? <p>Progress: {calibrationProgress}%</p> : null}
        {calibration.person_id ? <p>{calibrationPersonLabel}: {calibration.person_id}</p> : null}
        {calibration.failure_reason ? <p className="status-detail">{calibration.failure_reason}</p> : null}
        {calibration.state === "calibrating" ? (
          <button
            type="button"
            onClick={onCancelEngagementCalibration}
          >
            Cancel calibration
          </button>
        ) : (
          <button
            type="button"
            onClick={onStartEngagementCalibration}
          >
            {calibration.state === "calibrated" ? "Recalibrate" : "Start calibration"}
          </button>
        )}
      </section>

      {faceDetectorInfo?.detail && isDegraded(faceDetectorInfo.status) && (
        <p className="status-detail">{faceDetectorInfo.detail}</p>
      )}

      <ul>
        {people.map((person) => {
          const isPrimary = person.person_id === primaryPersonId;
          const lowConfidence = person.engagement_confidence < 0.5;
          return (
            <li key={person.person_id} className={lowConfidence ? "person-low-confidence" : isPrimary ? "person-primary" : ""}>
              {person.person_id}
              {isPrimary ? <strong> (primary)</strong> : null} · engagement {Math.round(person.engagement_score * 100)}% ·
              confidence {Math.round(person.engagement_confidence * 100)}%
              {person.is_active_speaker ? " · active speaker" : ""}
            </li>
          );
        })}
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
