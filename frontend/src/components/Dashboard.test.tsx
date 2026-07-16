import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EvidenceTimeline } from "./EvidenceTimeline";
import { Inspector } from "./Inspector";
import { PerceptionPanel } from "./PerceptionPanel";
import type { EvidenceEvent } from "../state/store";

afterEach(cleanup);

describe("Inspector", () => {
  it("shows evidence and degraded health without color-only meaning", () => {
    render(
      <Inspector
        state="engaged"
        audioMode="silent"
        evidence={[{ id: "observation-1", label: "keys", location: "right side of desk" }]}
        health={[{ component: "cloud", status: "degraded", detail: "offline fallback" }]}
      />,
    );
    expect(screen.getByText("keys")).toBeVisible();
    expect(screen.getAllByText(/degraded/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/offline fallback/i)).toBeVisible();
  });

  it("shows explicit microphone vad cloud and lamp audio status", () => {
    render(
      <Inspector
        state="engaged"
        audioMode="listening"
        evidence={[]}
        health={[
          { component: "microphone", status: "ok" },
          { component: "vad", status: "background_media", detail: "background_media" },
          { component: "cloud", status: "disabled", detail: "template provider" },
        ]}
      />,
    );

    expect(screen.getByText(/Microphone: ok/i)).toBeVisible();
    expect(screen.getByText(/VAD: background_media/i)).toBeVisible();
    expect(screen.getByText(/Cloud: disabled/i)).toBeVisible();
    expect(screen.getByText(/Lamp audio: listening/i)).toBeVisible();
  });
});

describe("PerceptionPanel", () => {
  it("shows face detector status from health", () => {
    render(
      <PerceptionPanel
        people={[]}
        objects={[]}
        health={[
          { component: "face_detector", status: "active", detail: "mediapipe_face_landmarker" },
        ]}
      />,
    );
    const mpElements = screen.getAllByText(/MediaPipe/i);
    expect(mpElements.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Active/i)).toBeVisible();
    expect(screen.getByText(/mediapipe_face_landmarker/i)).toBeVisible();
  });

  it("shows degraded face detector status", () => {
    render(
      <PerceptionPanel
        people={[]}
        objects={[]}
        health={[
          { component: "face_detector", status: "degraded", detail: "opencv_haar fallback" },
        ]}
      />,
    );
    expect(screen.getByText(/Degraded/i)).toBeVisible();
  });

  it("shows disabled object detector copy", () => {
    render(
      <PerceptionPanel
        people={[]}
        objects={[]}
        health={[
          { component: "object_detector", status: "disabled", detail: "not configured" },
        ]}
      />,
    );
    expect(screen.getByText(/Object detection disabled/i)).toBeVisible();
    const disabledElements = screen.getAllByText(/Disabled/i);
    expect(disabledElements.length).toBeGreaterThanOrEqual(1);
  });

  it("shows active no objects copy", () => {
    render(
      <PerceptionPanel
        people={[]}
        objects={[]}
        health={[
          { component: "object_detector", status: "active" },
        ]}
      />,
    );
    expect(screen.getByText(/0 objects detected/i)).toBeVisible();
  });

  it("shows heuristic face detector degraded warning", () => {
    render(
      <PerceptionPanel
        people={[]}
        objects={[]}
        health={[
          { component: "face_detector", status: "degraded", detail: "heuristic_skin_region" },
        ]}
      />,
    );
    expect(screen.getByText(/low reliability/i)).toBeVisible();
  });

  it("shows disabled face detector copy", () => {
    render(
      <PerceptionPanel
        people={[]}
        objects={[]}
        health={[
          { component: "face_detector", status: "disabled", detail: "face detection disabled" },
        ]}
      />,
    );
    const fdElements = screen.getAllByText(/Face detection disabled/i);
    expect(fdElements.length).toBeGreaterThanOrEqual(1);
  });

  it("shows active tracking count", () => {
    render(
      <PerceptionPanel
        people={[{ person_id: "p1", engagement_score: 0.8, engagement_confidence: 0.9, is_active_speaker: false }]}
        objects={[{ track_id: "t1", label: "cup", confidence: 0.9 }]}
        health={[
          { component: "face_detector", status: "active", detail: "mediapipe_face_landmarker" },
          { component: "object_detector", status: "active" },
        ]}
      />,
    );
    expect(screen.getByText(/1 person tracked/i)).toBeVisible();
    expect(screen.getByText(/1 object tracked/i)).toBeVisible();
  });

  it("renders multiple people and marks the primary person", () => {
    render(
      <PerceptionPanel
        people={[
          { person_id: "person-1", engagement_score: 0.8, engagement_confidence: 0.9, is_active_speaker: false },
          { person_id: "person-2", engagement_score: 0.6, engagement_confidence: 0.7, is_active_speaker: false },
        ]}
        primaryPersonId="person-1"
        objects={[]}
        health={[
          { component: "face_detector", status: "active", detail: "mediapipe_face_landmarker" },
        ]}
      />,
    );

    expect(screen.getByText(/2 people tracked/i)).toBeVisible();
    expect(screen.getByText(/person-1/i)).toHaveTextContent(/primary/i);
    expect(screen.getByText(/person-2/i)).not.toHaveTextContent(/primary/i);
  });

  it("labels the active speaker", () => {
    render(
      <PerceptionPanel
        people={[
          { person_id: "person-1", engagement_score: 0.8, engagement_confidence: 0.9, is_active_speaker: true },
        ]}
        primaryPersonId="person-1"
        objects={[]}
        health={[]}
      />,
    );

    expect(screen.getByText(/person-1/i)).toHaveTextContent(/active speaker/i);
  });

  it("shows engagement calibration state and controls", () => {
    const onStartEngagementCalibration = vi.fn();
    render(
      <PerceptionPanel
        people={[]}
        objects={[]}
        health={[]}
        onStartEngagementCalibration={onStartEngagementCalibration}
        engagementCalibration={{
          state: "uncalibrated",
          person_id: null,
          sample_count: 0,
          quality: "unavailable",
          failure_reason: null,
          mode: "fallback",
          progress: 0,
        }}
      />,
    );

    expect(screen.getByText(/calibration: uncalibrated/i)).toBeVisible();
    expect(screen.getByText(/mode: fallback/i)).toBeVisible();
    screen.getByRole("button", { name: /start calibration/i }).click();
    expect(onStartEngagementCalibration).toHaveBeenCalledTimes(1);
  });

  it("sends cancellation through the calibration callback", () => {
    const onCancelEngagementCalibration = vi.fn();
    render(
      <PerceptionPanel
        people={[]}
        objects={[]}
        health={[]}
        onCancelEngagementCalibration={onCancelEngagementCalibration}
        engagementCalibration={{
          state: "calibrating",
          person_id: "person-1",
          sample_count: 12,
          quality: "limited",
          failure_reason: null,
          mode: "fallback",
          progress: 0.5,
        }}
      />,
    );

    expect(screen.getByText(/progress: 50%/i)).toBeVisible();
    screen.getByRole("button", { name: /cancel calibration/i }).click();
    expect(onCancelEngagementCalibration).toHaveBeenCalledTimes(1);
  });

  it("does not label a failed calibration as calibrated", () => {
    render(
      <PerceptionPanel
        people={[]}
        objects={[]}
        health={[]}
        engagementCalibration={{
          state: "failed",
          person_id: "person-1",
          sample_count: 0,
          quality: "failed",
          failure_reason: "not enough valid samples",
          mode: "fallback",
          progress: 0,
        }}
      />,
    );

    expect(screen.getByText(/person: person-1/i)).toBeVisible();
    expect(screen.queryByText(/calibrated: person-1/i)).toBeNull();
  });
});

describe("EvidenceTimeline with evidence events", () => {
  it("renders engagement transition card", () => {
    const events: EvidenceEvent[] = [
      {
        event_id: "evt-1",
        event_type: "engagement_transition",
        correlation_id: null,
        occurred_at_mono_ns: 100,
        source: "runtime",
        summary: "Social state: idle -> engaged",
        severity: "info",
        entity_refs: [],
        evidence_refs: [],
        metadata: { previous_state: "idle", next_state: "engaged" },
      },
    ];
    const { container } = render(
      <EvidenceTimeline evidence={[]} evidenceEvents={events} />,
    );
    expect(container.textContent).toContain("ENGAGEMENT");
    expect(container.textContent).toContain("idle -> engaged");
  });

  it("renders behavior selected card", () => {
    const events: EvidenceEvent[] = [
      {
        event_id: "evt-2",
        event_type: "behavior_selected",
        correlation_id: null,
        occurred_at_mono_ns: 200,
        source: "policy",
        summary: "Behavior: acknowledge_engagement",
        severity: "info",
        entity_refs: [{ kind: "behavior", id: "acknowledge_engagement", label: "acknowledge_engagement" }],
        evidence_refs: [],
        metadata: {},
      },
    ];
    const { container } = render(
      <EvidenceTimeline evidence={[]} evidenceEvents={events} />,
    );
    expect(container.textContent).toContain("acknowledge_engagement");
  });

  it("renders fault card with severity", () => {
    const events: EvidenceEvent[] = [
      {
        event_id: "evt-3",
        event_type: "fault",
        correlation_id: null,
        occurred_at_mono_ns: 300,
        source: "runtime",
        summary: "Health: camera -> degraded (no signal)",
        severity: "error",
        entity_refs: [{ kind: "component", id: "camera", label: "camera" }],
        evidence_refs: [],
        metadata: {},
      },
    ];
    const { container } = render(
      <EvidenceTimeline evidence={[]} evidenceEvents={events} />,
    );
    expect(container.textContent).toContain("camera");
  });

  it("handles unknown event type gracefully", () => {
    const events: EvidenceEvent[] = [
      {
        event_id: "evt-4",
        event_type: "unknown_event_type",
        correlation_id: null,
        occurred_at_mono_ns: 400,
        source: "runtime",
        summary: "Something happened",
        severity: "info",
        entity_refs: [],
        evidence_refs: [],
        metadata: {},
      },
    ];
    const { container } = render(
      <EvidenceTimeline evidence={[]} evidenceEvents={events} />,
    );
    expect(container.textContent).toContain("unknown_event_type");
  });
});
