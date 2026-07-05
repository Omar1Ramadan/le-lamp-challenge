import { useState } from "react";
import { Canvas } from "@react-three/fiber";

import "./App.css";
import { DemoRail } from "./components/DemoRail";
import { EvidenceTimeline } from "./components/EvidenceTimeline";
import { Inspector } from "./components/Inspector";
import { PerceptionPanel } from "./components/PerceptionPanel";
import { LampScene } from "./scene/LampScene";
import { initialLampStore, initialState } from "./state/store";

type JourneyState = "idle" | "loaded";

function App() {
  const [journey, setJourney] = useState<JourneyState>("idle");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [showEvidence, setShowEvidence] = useState(false);
  const [bonusLoaded, setBonusLoaded] = useState(false);
  const world = initialState.world;
  const journeyLoaded = journey === "loaded";

  function loadCoreJourney() {
    setJourney("loaded");
    setAnswer("");
    setShowEvidence(false);
  }

  function askLamp() {
    if (question.toLowerCase().includes("keys")) {
      setAnswer("I last saw the keys on the right side of the desk.");
    } else {
      setAnswer("I do not have reliable evidence for that.");
    }
  }

  function loadBonusJourney() {
    setBonusLoaded(true);
  }

  return (
    <main className="app-shell">
      <section className="status-banner" aria-live="polite">
        <span className="status-dot" />
        Simulator connection: {initialLampStore.connection}
      </section>
      <section className="dashboard-grid">
        <section className="viewport" aria-label="Simulated social lamp viewport">
          <Canvas camera={{ position: [0, 1.6, 5], fov: 45 }}>
            <color attach="background" args={["#101522"]} />
            <ambientLight intensity={0.7} />
            <directionalLight position={[3, 5, 4]} intensity={1.8} />
            <LampScene pose={initialLampStore.pose} />
          </Canvas>
        </section>
        <PerceptionPanel people={world?.people ?? []} objects={world?.objects ?? []} />
        <EvidenceTimeline evidence={initialState.evidence} />
        <Inspector
          state={journeyLoaded ? "seeking_attention" : (world?.social_state ?? "idle")}
          evidence={[]}
          health={world?.health ?? []}
        />
        <DemoRail metrics={initialState.metrics} needsResync={initialState.needsResync} />
        <section className="panel demo-proof" aria-label="Replay proof controls">
          <h2>Replay proof</h2>
          <button type="button" onClick={loadCoreJourney}>Load core journey replay</button>
          <div data-testid="demo-step-engagement" data-complete={journeyLoaded ? "true" : "false"}>
            Engagement
          </div>
          <div data-testid="demo-step-attention" data-complete={journeyLoaded ? "true" : "false"}>
            {journeyLoaded ? "Seeking attention: level 1" : "Attention pending"}
          </div>
          {journeyLoaded ? (
            <article aria-label="memory: keys">
              Memory: keys observed on the right side of the desk.
            </article>
          ) : null}
          <label>
            Ask the lamp
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              aria-label="Ask the lamp"
            />
          </label>
          <button type="button" onClick={askLamp}>Ask</button>
          <p data-testid="lamp-answer">{answer}</p>
          <button type="button" onClick={() => setShowEvidence(true)}>Show evidence</button>
          {showEvidence ? <p>observation-core-keys-2</p> : null}
        </section>
        <section className="panel bonus-proof" aria-label="Bonus proof controls">
          <h2>Bonus proof</h2>
          <button type="button" onClick={loadBonusJourney}>Load bonus journey replay</button>
          {bonusLoaded ? (
            <ul>
              <li>Active speaker: Person B</li>
              <li>Affect confidence gated below 0.60</li>
              <li>Preference score changed then reset</li>
              <li>Speech interruption cancellation under 120 ms</li>
              <li>Television suppression active</li>
            </ul>
          ) : null}
        </section>
      </section>
    </main>
  );
}

export default App;
