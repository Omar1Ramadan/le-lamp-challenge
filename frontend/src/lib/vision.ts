export interface VisionDetectorStatus {
  name: string;
  status: "active" | "degraded" | "disabled";
  detail: string | null;
}

export interface VisionStatus {
  face_detector?: VisionDetectorStatus | null;
  object_detector?: VisionDetectorStatus | null;
}

const DETECTOR_NAME_MAP: Record<string, string> = {
  mediapipe_face_landmarker: "MediaPipe",
  opencv_haar: "OpenCV",
  heuristic_skin_region: "heuristic fallback",
  object_detector: "Objects",
  face_detector: "Face",
  none: "None",
};

export function shortDetectorName(name: string): string {
  return DETECTOR_NAME_MAP[name] ?? name;
}

export interface BadgeClass {
  className: string;
  label: string;
}

export function badgeForStatus(status: string, detail?: string | null): BadgeClass {
  switch (status) {
    case "active":
      return { className: "badge-active", label: "Active" };
    case "degraded":
      return { className: "badge-degraded", label: "Degraded" };
    case "disabled":
      return { className: "badge-disabled", label: "Disabled" };
    default:
      return { className: "badge-unknown", label: status };
  }
}

export function badgeForDetector(detector: VisionDetectorStatus | null | undefined): BadgeClass | null {
  if (!detector) return null;
  return badgeForStatus(detector.status, detector.detail);
}

export function isHeuristic(name: string): boolean {
  return name === "heuristic_skin_region";
}

export function isDisabled(status: string): boolean {
  return status === "disabled";
}

export function isDegraded(status: string): boolean {
  return status === "degraded";
}
