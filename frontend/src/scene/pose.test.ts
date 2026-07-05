import { describe, expect, it } from "vitest";
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
