import { describe, expect, it } from "vitest";
import { poseFromTimeline } from "../state/store";
import { poseToRotations } from "./pose";

describe("poseToRotations", () => {
  it("maps all six normalized channels to bounded radians", () => {
    const rotations = poseToRotations({
      base_yaw: 1,
      shoulder_pitch: -1,
      elbow_pitch: 0.5,
      wrist_pitch: 0.25,
      head_yaw: -0.5,
      head_pitch: 0,
    });
    expect(Object.keys(rotations)).toHaveLength(6);
    expect(rotations.base_yaw).toBeCloseTo(Math.PI / 2);
    expect(rotations.shoulder_pitch).toBeCloseTo(-Math.PI / 3);
  });
});

describe("poseFromTimeline", () => {
  it("interpolates active timeline keyframes before returning neutral", () => {
    const pose = poseFromTimeline(
      {
        cancellable: true,
        correlation_id: "correlation-1",
        duration_ms: 1000,
        intent_id: "intent-1",
        motion_tracks: [
          {
            channel: "head_yaw",
            keyframes: [
              { easing: "ease_in_out", offset_ms: 0, value: 0 },
              { easing: "ease_in_out", offset_ms: 500, value: 1 },
              { easing: "ease_in_out", offset_ms: 1000, value: 0 },
            ],
          },
        ],
        priority: 60,
        timeline_id: "timeline-1",
      },
      250,
    );

    expect(pose.head_yaw).toBeCloseTo(0.5);
  });
});
