import type { LampPose } from "../contracts/domain";

const LIMITS: LampPose = {
  base_yaw: Math.PI / 2,
  shoulder_pitch: Math.PI / 3,
  elbow_pitch: Math.PI / 2,
  wrist_pitch: Math.PI / 2,
  head_yaw: Math.PI / 2,
  head_pitch: Math.PI / 3,
};

export function neutralPose(): LampPose {
  return {
    base_yaw: 0,
    shoulder_pitch: 0,
    elbow_pitch: 0,
    wrist_pitch: 0,
    head_yaw: 0,
    head_pitch: 0,
  };
}

export function poseToRotations(pose: LampPose): LampPose {
  return Object.fromEntries(
    Object.entries(pose).map(([channel, value]) => [
      channel,
      Math.max(-1, Math.min(1, value)) * LIMITS[channel as keyof LampPose],
    ]),
  ) as LampPose;
}
