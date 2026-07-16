import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";

import "./App.css";
import { DemoRail } from "./components/DemoRail";
import { DevicePanel } from "./components/DevicePanel";
import { EvidenceTimeline } from "./components/EvidenceTimeline";
import { Inspector, type InspectorEvidence } from "./components/Inspector";
import { PerceptionPanel } from "./components/PerceptionPanel";
import {
  cancelEngagementCalibration,
  getHealth,
  getReplays,
  getWorld,
  runReplay,
  startEngagementCalibration,
  startSession,
  stopSession,
  submitText,
  type ReplaySummary,
} from "./lib/api";
import type { VisionStatus } from "./lib/vision";
import { connectSocket, sendAck } from "./lib/socket";
import { LampScene } from "./scene/LampScene";
import {
  initialLampStore,
  initialState,
  poseFromTimeline,
  reduceServerMessage,
  type DashboardState,
  type DashboardWorldSnapshot,
  type ServerMessage,
} from "./state/store";

function dashboardReducer(state: DashboardState, message: ServerMessage) {
  return reduceServerMessage(state, message);
}

function App() {
  const [state, dispatch] = useReducer(dashboardReducer, initialState);
  const [connection, setConnection] = useState(initialLampStore.connection);
  const [replays, setReplays] = useState<ReplaySummary[]>([]);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [lampAction, setLampAction] = useState("Lamp is neutral.");
  const [timelineElapsedMs, setTimelineElapsedMs] = useState(0);
  const [showEvidence, setShowEvidence] = useState(false);
  const [sessionRunning, setSessionRunning] = useState(true);
  const [visionStatus, setVisionStatus] = useState<VisionStatus | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const world = state.world;
  const pose = useMemo(
    () => poseFromTimeline(state.timeline, timelineElapsedMs),
    [state.timeline, timelineElapsedMs],
  );
  const evidence = useMemo(() => inspectorEvidence(state.evidence), [state.evidence]);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([getHealth(), getWorld<DashboardWorldSnapshot>(), getReplays()])
      .then(([, snapshot, replayList]) => {
        if (cancelled) {
          return;
        }
        setReplays(replayList);
        dispatch({ seq: 1, type: "world_snapshot", body: snapshot });
      })
      .catch(() => setConnection("offline"));

    const socket = connectSocket((message) => dispatch(message));
    socketRef.current = socket;
    socket.onopen = () => setConnection("connected");
    socket.onclose = () => setConnection("offline");
    setConnection("connecting");
    return () => {
      cancelled = true;
      socket.close();
    };
  }, []);

  useEffect(() => {
    const currentTimeline = state.timeline;
    if (!currentTimeline) {
      setTimelineElapsedMs(0);
      return;
    }

    const socket = socketRef.current;
    const timelineId = currentTimeline.timeline_id;

    if (socket) sendAck(socket, timelineId, "timeline_received");

    let firstFrameSent = false;
    let completed = false;
    let animationFrame = 0;
    const startedAt = performance.now();
    const tick = () => {
      const elapsed = performance.now() - startedAt;
      setTimelineElapsedMs(elapsed);
      if (!firstFrameSent) {
        firstFrameSent = true;
        if (socket) sendAck(socket, timelineId, "first_visible_frame");
      }
      if (elapsed >= currentTimeline.duration_ms) {
        completed = true;
        if (socket) sendAck(socket, timelineId, "timeline_complete");
        return;
      }
      animationFrame = window.requestAnimationFrame(tick);
    };
    animationFrame = window.requestAnimationFrame(tick);
    return () => {
      window.cancelAnimationFrame(animationFrame);
      if (socket && !completed) {
        sendAck(socket, timelineId, "timeline_cancelled", { reason: "replaced" });
      }
    };
  }, [state.timeline]);

  async function loadReplay(replay: ReplaySummary) {
    setAnswer("");
    const messages = await runReplay(replay.directory);
    for (const message of messages) {
      dispatch(message as ServerMessage);
    }
  }

  async function askLamp() {
    setAnswer(await submitText(question));
  }

  async function handleSessionToggle() {
    const response = sessionRunning ? await stopSession() : await startSession();
    setSessionRunning(response.running);
  }

  async function handleStartEngagementCalibration() {
    const calibration = await startEngagementCalibration<
      NonNullable<DashboardWorldSnapshot["engagement_calibration"]>
    >();
    if (world) {
      dispatch({
        seq: state.lastSequence + 1,
        type: "world_snapshot",
        body: { ...world, engagement_calibration: calibration },
      });
    }
  }

  async function handleCancelEngagementCalibration() {
    const calibration = await cancelEngagementCalibration<
      NonNullable<DashboardWorldSnapshot["engagement_calibration"]>
    >();
    if (world) {
      dispatch({
        seq: state.lastSequence + 1,
        type: "world_snapshot",
        body: { ...world, engagement_calibration: calibration },
      });
    }
  }

  return (
    <main className="app-shell">
      <section className="status-banner" aria-live="polite">
        <span className="status-dot" />
        Simulator connection: {connection}
      </section>
      <section className="dashboard-grid">
        <section className="viewport" aria-label="Simulated social lamp viewport">
          <Canvas camera={{ position: [0, 1.6, 5], fov: 45 }}>
            <color attach="background" args={["#101522"]} />
            <ambientLight intensity={0.7} />
            <directionalLight position={[3, 5, 4]} intensity={1.8} />
            <LampScene pose={pose} />
          </Canvas>
          <div className="lamp-action" aria-live="polite">
            {lampAction}
          </div>
        </section>
        <PerceptionPanel
          people={world?.people ?? []}
          primaryPersonId={world?.primary_person_id ?? null}
          objects={world?.objects ?? []}
          health={world?.health ?? []}
          visionStatus={visionStatus}
          engagementCalibration={world?.engagement_calibration ?? null}
          onStartEngagementCalibration={() => void handleStartEngagementCalibration()}
          onCancelEngagementCalibration={() => void handleCancelEngagementCalibration()}
        />
        <EvidenceTimeline evidence={state.evidence} />
        <DevicePanel
          calibrationState={world?.engagement_calibration?.state ?? null}
          onBehaviorTimeline={(timeline) => {
            dispatch({
              seq: state.lastSequence + 2,
              type: "behavior_timeline",
              body: timeline,
            });
            setLampAction(`Lamp action: visible ${timeline.duration_ms} ms response.`);
          }}
          onWorldSnapshot={(snapshot) =>
            dispatch({ seq: state.lastSequence + 1, type: "world_snapshot", body: snapshot })
          }
          onVisionStatus={setVisionStatus}
        />
        <Inspector
          state={world?.social_state ?? "idle"}
          audioMode={world?.audio_mode ?? "silent"}
          evidence={evidence}
          health={world?.health ?? []}
        />
        <DemoRail metrics={state.metrics} needsResync={state.needsResync} />
        <section className="panel demo-proof" aria-label="Replay proof controls">
          <h2>Replay proof</h2>
          <p>
            Live camera and microphone run on the backend host. Enable
            <code> ENABLE_LIVE_CAPTURE=true </code>
            in <code>.env</code> before starting the API if you want hardware input.
          </p>
          <button type="button" onClick={() => void handleSessionToggle()}>
            {sessionRunning ? "Stop live session" : "Start live session"}
          </button>
          {replays.map((replay) => (
            <button type="button" key={replay.id} onClick={() => void loadReplay(replay)}>
              Load {replay.label.toLowerCase()} replay
            </button>
          ))}
          <div
            data-testid="demo-step-engagement"
            data-complete={
              world?.social_state === "engaged" || state.metrics.engagement_seen
                ? "true"
                : "false"
            }
          >
            Engagement
          </div>
          <div
            data-testid="demo-step-attention"
            data-complete={state.timeline ? "true" : "false"}
          >
            {state.metrics.attention_level
              ? `Seeking attention: level ${state.metrics.attention_level}`
              : state.timeline
                ? "Attention timeline received from backend"
                : "Attention pending"}
          </div>
          <label>
            Ask the lamp
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              aria-label="Ask the lamp"
            />
          </label>
          <button type="button" onClick={() => void askLamp()}>
            Ask
          </button>
          <p data-testid="lamp-answer">{answer}</p>
          <button type="button" onClick={() => setShowEvidence(true)}>
            Show evidence
          </button>
          {showEvidence ? (
            <ul aria-label="Visible evidence identifiers">
              {state.evidence.flatMap((item) => item.evidence_ids ?? []).map((id) => (
                <li key={id}>{id}</li>
              ))}
            </ul>
          ) : null}
        </section>
      </section>
    </main>
  );
}

function inspectorEvidence(evidence: DashboardState["evidence"]): InspectorEvidence[] {
  return evidence.flatMap((item) => {
    if (item.status !== "found" || !item.canonical_label) {
      return [];
    }
    const location = [item.horizontal_region, item.depth_band, item.anchor_name]
      .filter(Boolean)
      .join(" / ");
    return (item.evidence_ids ?? [item.canonical_label]).map((id) => ({
      id,
      label: item.canonical_label ?? "unknown",
      location,
    }));
  });
}

export default App;
