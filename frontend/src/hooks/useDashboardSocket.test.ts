import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

let fakeSocket: {
  onopen: ((...args: unknown[]) => void) | null;
  onclose: ((...args: unknown[]) => void) | null;
  onmessage: ((event: MessageEvent) => void) | null;
  close: ReturnType<typeof vi.fn>;
  send: ReturnType<typeof vi.fn>;
  readyState: number;
};

let socketConstructor: (...args: unknown[]) => object;

function MockWebSocket() {
  fakeSocket = {
    onopen: null,
    onclose: null,
    onmessage: null,
    close: vi.fn(),
    send: vi.fn(),
    readyState: 1,
  };
  socketConstructor?.();
  return fakeSocket;
}

beforeEach(() => {
  vi.useFakeTimers();
  socketConstructor = vi.fn();
  vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          revision: 1,
          social_state: "idle",
          people: [],
          objects: [],
          health: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    ),
  );
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
  cleanup();
});

describe("useDashboardSocket", () => {
  it("starts in connecting state", async () => {
    const mod = await import("./useDashboardSocket");
    const { result } = renderHook(() => mod.useDashboardSocket());
    expect(result.current.connection).toBe("connecting");
  });

  it("transitions to connected after socket opens", async () => {
    const mod = await import("./useDashboardSocket");
    const { result } = renderHook(() => mod.useDashboardSocket());
    act(() => {
      fakeSocket.onopen?.();
    });
    await vi.advanceTimersByTimeAsync(0);
    expect(result.current.connection).toBe("connected");
  });

  it("reconnects after socket close with backoff", async () => {
    const mod = await import("./useDashboardSocket");
    const { result } = renderHook(() => mod.useDashboardSocket());
    act(() => { fakeSocket.onopen?.(); });
    await vi.advanceTimersByTimeAsync(0);

    act(() => { fakeSocket.onclose?.(); });
    expect(result.current.connection).toBe("reconnecting");

    // Advance past the initial backoff (250ms + up to 100ms jitter)
    await vi.advanceTimersByTimeAsync(500);
    expect(socketConstructor).toHaveBeenCalledTimes(2);
  });

  it("uses increasing backoff delays", async () => {
    const mod = await import("./useDashboardSocket");
    renderHook(() => mod.useDashboardSocket());

    act(() => { fakeSocket.onopen?.(); });
    await vi.advanceTimersByTimeAsync(0);

    // First close → reconnect at ~250ms
    act(() => { fakeSocket.onclose?.(); });
    await vi.advanceTimersByTimeAsync(500);
    expect(socketConstructor).toHaveBeenCalledTimes(2);

    // Second close → reconnect at ~500ms
    act(() => { fakeSocket.onclose?.(); });
    await vi.advanceTimersByTimeAsync(800);
    expect(socketConstructor).toHaveBeenCalledTimes(3);
  });

  it("does not create multiple simultaneous sockets", async () => {
    const mod = await import("./useDashboardSocket");
    renderHook(() => mod.useDashboardSocket());
    act(() => { fakeSocket.onopen?.(); });
    await vi.advanceTimersByTimeAsync(0);
    expect(socketConstructor).toHaveBeenCalledTimes(1);

    act(() => { fakeSocket.onclose?.(); });
    await vi.advanceTimersByTimeAsync(500);
    expect(socketConstructor).toHaveBeenCalledTimes(2);
  });

  it("fetches /api/world after reconnect", async () => {
    const mod = await import("./useDashboardSocket");
    renderHook(() => mod.useDashboardSocket());
    act(() => { fakeSocket.onopen?.(); });
    await vi.advanceTimersByTimeAsync(0);

    // Bypass initialConnectRef: send a message so lastSequence > 0
    act(() => {
      fakeSocket.onmessage?.(
        { data: JSON.stringify({ seq: 1, type: "world_snapshot", body: { revision: 2, social_state: "engaged", people: [], objects: [], health: [] } }) } as MessageEvent,
      );
    });
    await vi.advanceTimersByTimeAsync(0);

    const fetchSpy = vi.mocked(window.fetch);
    fetchSpy.mockClear();

    // Close, reconnect, open the new socket
    act(() => { fakeSocket.onclose?.(); });
    await vi.advanceTimersByTimeAsync(500);
    act(() => { fakeSocket.onopen?.(); });
    await vi.advanceTimersByTimeAsync(0);

    const worldCalls = fetchSpy.mock.calls.filter(([url]) =>
      (url as string).includes("/api/world"),
    );
    expect(worldCalls.length).toBeGreaterThanOrEqual(1);
  });

  it("cleans up on unmount", async () => {
    const mod = await import("./useDashboardSocket");
    const { unmount } = renderHook(() => mod.useDashboardSocket());
    const closeSpy = fakeSocket.close;
    unmount();
    expect(closeSpy).toHaveBeenCalled();
  });
});
