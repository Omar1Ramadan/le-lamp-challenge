import { useEffect, useMemo, useReducer, useState } from "react";
import { Canvas } from "@react-three/fiber";

import "./App.css";
import { DemoRail } from "./components/DemoRail";
import { EvidenceTimeline } from "./components/EvidenceTimeline";
import { Inspector, type InspectorEvidence } from "./components/Inspector";
import { PerceptionPanel } from "./components/PerceptionPanel";
import { getHealth, getReplays, getWorld, runReplay, submitText, type ReplaySummary } from "./lib/api";
import { connectSocket } from "./lib/socket";
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
  const [showEvidence, setShowEvidence] = useState(false);
  const world = state.world;
  const pose = useMemo(() => poseFromTimeline(state.timeline), [state.timeline]);
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
    socket.onopen = () => setConnection("connected");
    socket.onclose = () => setConnection("offline");
    setConnection("connecting");
    return () => {
      cancelled = true;
      socket.close();
    };
  }, []);

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
        </section>
        <PerceptionPanel people={world?.people ?? []} objects={world?.objects ?? []} />
        <EvidenceTimeline evidence={state.evidence} />
        <Inspector
          state={world?.social_state ?? "idle"}
          evidence={evidence}
          health={world?.health ?? []}
        />
        <DemoRail metrics={state.metrics} needsResync={state.needsResync} />
        <section className="panel demo-proof" aria-label="Replay proof controls">
          <h2>Replay proof</h2>
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
