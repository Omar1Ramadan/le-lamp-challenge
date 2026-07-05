import { Canvas } from "@react-three/fiber";

import "./App.css";
import { DemoRail } from "./components/DemoRail";
import { EvidenceTimeline } from "./components/EvidenceTimeline";
import { Inspector } from "./components/Inspector";
import { PerceptionPanel } from "./components/PerceptionPanel";
import { LampScene } from "./scene/LampScene";
import { initialLampStore, initialState } from "./state/store";

function App() {
  const world = initialState.world;
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
          state={world?.social_state ?? "idle"}
          evidence={[]}
          health={world?.health ?? []}
        />
        <DemoRail metrics={initialState.metrics} needsResync={initialState.needsResync} />
      </section>
    </main>
  );
}

export default App;
