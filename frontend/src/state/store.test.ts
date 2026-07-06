import { describe, expect, it } from "vitest";
import { initialState, reduceServerMessage } from "./store";

describe("reduceServerMessage", () => {
  it("replaces state from a full snapshot and detects sequence gaps", () => {
    const first = reduceServerMessage(initialState, {
      seq: 1,
      type: "world_snapshot",
      body: { revision: 4, social_state: "engaged", people: [], objects: [], health: [] },
    });
    expect(first.world?.revision).toBe(4);
    const gap = reduceServerMessage(first, {
      seq: 3,
      type: "metric",
      body: { name: "frame_age_ms", value: 20 },
    });
    expect(gap.needsResync).toBe(true);
  });
});
