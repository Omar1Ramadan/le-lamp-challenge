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
    expect(screen.getByText(/face detector/i)).toBeVisible();
    expect(screen.getByText(/active/i)).toBeVisible();
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
    expect(screen.getByText(/degraded/i)).toBeVisible();
  });
});
