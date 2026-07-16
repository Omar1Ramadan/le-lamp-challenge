import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api helpers", () => {
  it("maps the Vite dev server port to the backend API origin", async () => {
    const { apiBaseForPort } = await import("./api");

    expect(apiBaseForPort("5173")).toBe("http://127.0.0.1:8000");
    expect(apiBaseForPort("8000")).toBe("");
  });

  it("posts calibration requests through the shared API helper", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json({
        state: "calibrating",
        person_id: null,
        sample_count: 0,
        quality: "unavailable",
        failure_reason: null,
        mode: "fallback",
        progress: 0,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const { startEngagementCalibration } = await import("./api");

    await startEngagementCalibration();

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/calibration/engagement/start",
      { method: "POST" },
    );
  });
});
