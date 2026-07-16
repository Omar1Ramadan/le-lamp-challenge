import type { LampPose, MotionChannel } from "../contracts/domain";
import type {
  BehaviorTimeline as DashboardTimeline,
  MemoryResult,
  ObservationEvent,
  WorldSnapshot,
} from "../contracts/generated";
import { neutralPose } from "../scene/pose";

export type ConnectionState =
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "frozen"
  | "resyncing";

export interface LampStoreSnapshot {
  connection: ConnectionState;
  pose: LampPose;
  timeline: DashboardTimeline | null;
}

export const initialLampStore: LampStoreSnapshot = {
  connection: "disconnected",
  pose: neutralPose(),
  timeline: null,
};

export function poseFromTimeline(timeline: DashboardTimeline | null, elapsedMs?: number): LampPose {
  const pose = neutralPose();
  if (timeline === null) {
    return pose;
  }
  if (elapsedMs !== undefined) {
    const boundedElapsed = Math.max(0, Math.min(elapsedMs, timeline.duration_ms));
    for (const track of timeline.motion_tracks) {
      const keyframes = [...track.keyframes].sort((a, b) => a.offset_ms - b.offset_ms);
      if (keyframes.length === 0) {
        continue;
      }
      const nextIndex = keyframes.findIndex((keyframe) => keyframe.offset_ms >= boundedElapsed);
      if (nextIndex <= 0) {
        pose[track.channel as MotionChannel] = keyframes[0].value;
        continue;
      }
      if (nextIndex === -1) {
        pose[track.channel as MotionChannel] = keyframes.at(-1)?.value ?? 0;
        continue;
      }
      const previous = keyframes[nextIndex - 1];
      const next = keyframes[nextIndex];
      const span = Math.max(next.offset_ms - previous.offset_ms, 1);
      const progress = (boundedElapsed - previous.offset_ms) / span;
      pose[track.channel as MotionChannel] =
        previous.value + (next.value - previous.value) * progress;
    }
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

interface MetricBody {
  name: string;
  value?: number;
  labels?: Record<string, string>;
}

interface FaultBody {
  component: string;
  detail: string;
}

export type DashboardWorldSnapshot = Partial<WorldSnapshot> & {
  revision: number;
  social_state: string;
  people: WorldSnapshot["people"];
  objects: WorldSnapshot["objects"];
  health: WorldSnapshot["health"];
};

export interface DashboardState {
  world: DashboardWorldSnapshot | null;
  timeline: DashboardTimeline | null;
  evidence: MemoryResult[];
  metrics: Record<string, number>;
  faults: FaultBody[];
  lastSequence: number;
  needsResync: boolean;
}

export type ServerMessage =
  | { seq: number; type: "world_snapshot"; body: DashboardWorldSnapshot }
  | { seq: number; type: "behavior_timeline"; body: DashboardTimeline }
  | { seq: number; type: "observation"; body: ObservationEvent }
  | { seq: number; type: "memory_result"; body: MemoryResult }
  | { seq: number; type: "metric"; body: MetricBody }
  | { seq: number; type: "fault"; body: FaultBody };

export const initialState: DashboardState = {
  world: null,
  timeline: null,
  evidence: [],
  metrics: {},
  faults: [],
  lastSequence: 0,
  needsResync: false,
};

export function reduceServerMessage(
  state: DashboardState,
  message: ServerMessage,
): DashboardState {
  const gap = state.lastSequence !== 0 && message.seq !== state.lastSequence + 1;
  const next = {
    ...state,
    lastSequence: message.seq,
    needsResync: state.needsResync || gap,
  };
  switch (message.type) {
    case "world_snapshot":
      return { ...next, world: message.body };
    case "behavior_timeline":
      return { ...next, timeline: message.body };
    case "memory_result": {
      const evidenceKey = (message.body.evidence_ids ?? []).join("|") || message.body.status;
      const seen = state.evidence.some(
        (item) => ((item.evidence_ids ?? []).join("|") || item.status) === evidenceKey,
      );
      return { ...next, evidence: seen ? state.evidence : [...state.evidence, message.body] };
    }
    case "metric":
      return {
        ...next,
        metrics: {
          ...state.metrics,
          [message.body.name]: message.body.value ?? (state.metrics[message.body.name] ?? 0) + 1,
        },
      };
    case "fault":
      return { ...next, faults: [...state.faults, message.body] };
    case "observation":
      return next;
  }
}
