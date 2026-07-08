import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@react-three/fiber", () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => <div data-testid="canvas">{children}</div>,
}));

vi.mock("./scene/LampScene", () => ({
  LampScene: ({ pose }: { pose: Record<string, number> }) => (
    <div data-testid="lamp-pose">{pose.head_yaw}</div>
  ),
}));

import App from "./App";

class MockSocket {
  static instances: MockSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: (() => void) | null = null;
  sent: string[] = [];
  url: string;

  constructor(url: string) {
    this.url = url;
    MockSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.onclose?.();
  }

  emit(message: unknown) {
    this.onmessage?.({ data: JSON.stringify(message) } as MessageEvent);
  }
}

const world = {
  revision: 1,
  social_state: "idle",
  audio_mode: "silent",
  people: [],
  objects: [],
  health: [],
};

beforeEach(() => {
  MockSocket.instances = [];
  vi.stubGlobal("WebSocket", MockSocket);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/health")) {
        return Response.json({ status: "healthy" });
      }
      if (url.endsWith("/api/world")) {
        return Response.json(world);
      }
      if (url.endsWith("/api/replays")) {
        return Response.json({ replays: [{ id: "core", label: "Core Journey", directory: "fixtures/core" }] });
      }
      if (url.endsWith("/api/replay")) {
        return Response.json({ ok: true, revision: 2 });
      }
      if (url.endsWith("/api/session/start")) {
        return Response.json({ ok: true, running: true });
      }
      if (url.endsWith("/api/session/stop")) {
        return Response.json({ ok: true, running: false });
      }
      if (url.endsWith("/api/text")) {
        return Response.json({ ok: true, response: { text: "I found keys from stored evidence." } });
      }
      return Response.json({}, { status: 404 });
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("App backend integration", () => {
  it("loads replay buttons from the backend and updates proof from socket evidence", async () => {
    render(<App />);

    const replayButton = await screen.findByRole("button", { name: "Load core journey replay" });
    fireEvent.click(replayButton);

    expect(fetch).toHaveBeenCalledWith("/api/replay", {
      body: JSON.stringify({ directory: "fixtures/core" }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    });

    MockSocket.instances[0].emit({
      seq: 1,
      type: "world_snapshot",
      body: { ...world, revision: 2, social_state: "engaged" },
    });
    MockSocket.instances[0].emit({
      seq: 2,
      type: "behavior_timeline",
      body: {
        timeline_id: "timeline-1",
        intent_id: "intent-1",
        correlation_id: "correlation-1",
        priority: 80,
        duration_ms: 700,
        cancellable: true,
        motion_tracks: [{ channel: "head_yaw", keyframes: [{ offset_ms: 700, value: 0.5 }] }],
      },
    });
    MockSocket.instances[0].emit({
      seq: 3,
      type: "memory_result",
      body: {
        status: "found",
        canonical_label: "keys",
        horizontal_region: "right",
        depth_band: "foreground",
        anchor_name: "desk",
        evidence_ids: ["observation-core-keys-2"],
      },
    });

    await waitFor(() => expect(screen.getByTestId("demo-step-engagement")).toHaveAttribute("data-complete", "true"));
    expect(screen.getByTestId("lamp-pose")).toHaveTextContent("0.5");
    expect(screen.getByRole("article", { name: /memory: keys/i })).toHaveTextContent("keys");
    expect(screen.getAllByText("observation-core-keys-2")[0]).toBeVisible();
    expect(screen.getByText(/Audio mode:/i)).toBeVisible();
  });

  it("submits text questions to the backend", async () => {
    render(<App />);

    fireEvent.change(await screen.findByLabelText("Ask the lamp"), {
      target: { value: "Where are my keys?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    await screen.findByText("I found keys from stored evidence.");
  });
});
