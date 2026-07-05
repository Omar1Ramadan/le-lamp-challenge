import { Canvas } from "@react-three/fiber";

import "./App.css";
import { initialLampStore } from "./state/store";
import { LampScene } from "./scene/LampScene";

function App() {
  return (
    <main className="app-shell">
      <section className="status-banner" aria-live="polite">
        <span className="status-dot" />
        Simulator connection: {initialLampStore.connection}
      </section>
      <section className="viewport" aria-label="Simulated social lamp viewport">
        <Canvas camera={{ position: [0, 1.6, 5], fov: 45 }}>
          <color attach="background" args={["#101522"]} />
          <ambientLight intensity={0.7} />
          <directionalLight position={[3, 5, 4]} intensity={1.8} />
          <LampScene pose={initialLampStore.pose} />
        </Canvas>
      </section>
    </main>
  );
}

export default App;
