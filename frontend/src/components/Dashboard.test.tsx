import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Inspector } from "./Inspector";
import { PerceptionPanel } from "./PerceptionPanel";

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
    expect(screen.getByText(/degraded/i)).toBeVisible();
    expect(screen.getByText(/offline fallback/i)).toBeVisible();
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
});
