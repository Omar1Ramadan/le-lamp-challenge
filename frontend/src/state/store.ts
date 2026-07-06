import type { LampPose, MotionChannel } from "../contracts/domain";
import type {
  BehaviorTimeline as DashboardTimeline,
  MemoryResult,
  ObservationEvent,
  WorldSnapshot,
} from "../contracts/generated";
import { neutralPose } from "../scene/pose";

export type ConnectionState = "connecting" | "connected" | "offline";

export interface LampStoreSnapshot {
  connection: ConnectionState;
  pose: LampPose;
  timeline: DashboardTimeline | null;
}

export const initialLampStore: LampStoreSnapshot = {
  connection: "offline",
  pose: neutralPose(),
  timeline: null,
};

export function poseFromTimeline(timeline: DashboardTimeline | null): LampPose {
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
    case "memory_result":
      return { ...next, evidence: [...state.evidence, message.body] };
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
