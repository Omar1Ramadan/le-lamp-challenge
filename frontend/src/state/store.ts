import type { BehaviorTimeline, LampPose, MotionChannel } from "../contracts/domain";
import { neutralPose } from "../scene/pose";

export type ConnectionState = "connecting" | "connected" | "offline";

export interface LampStoreSnapshot {
  connection: ConnectionState;
  pose: LampPose;
  timeline: BehaviorTimeline | null;
}

export const initialLampStore: LampStoreSnapshot = {
  connection: "offline",
  pose: neutralPose(),
  timeline: null,
};

export function poseFromTimeline(timeline: BehaviorTimeline | null): LampPose {
  const pose = neutralPose();
  if (timeline === null) {
    return pose;
  }
  for (const track of timeline.motion_tracks) {
    const finalKeyframe = track.keyframes.at(-1);
    if (finalKeyframe) {
      pose[track.channel as MotionChannel] = finalKeyframe.value;
    }
  }
  return pose;
}
