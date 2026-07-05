export type MotionChannel =
  | "base_yaw"
  | "shoulder_pitch"
  | "elbow_pitch"
  | "wrist_pitch"
  | "head_yaw"
  | "head_pitch";

export type LampPose = Record<MotionChannel, number>;

export interface MotionKeyframe {
  offset_ms: number;
  value: number;
  easing: string;
}

export interface MotionTrack {
  channel: MotionChannel;
  keyframes: MotionKeyframe[];
}

export interface BehaviorTimeline {
  timeline_id: string;
  intent_id: string;
  duration_ms: number;
  motion_tracks: MotionTrack[];
}
