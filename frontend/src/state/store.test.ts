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

  it("does not set needsResync on sequential messages", () => {
    const first = reduceServerMessage(initialState, {
      seq: 1,
      type: "world_snapshot",
      body: { revision: 1, social_state: "idle", people: [], objects: [], health: [] },
    });
    const second = reduceServerMessage(first, {
      seq: 2,
      type: "metric",
      body: { name: "test", value: 1 },
    });
    expect(second.needsResync).toBe(false);
  });

  it("keeps needsResync true once set, even on valid sequences", () => {
    const first = reduceServerMessage(initialState, {
      seq: 1,
      type: "world_snapshot",
      body: { revision: 1, social_state: "idle", people: [], objects: [], health: [] },
    });
    const gap = reduceServerMessage(first, {
      seq: 3,
      type: "metric",
      body: { name: "frame_age_ms", value: 20 },
    });
    expect(gap.needsResync).toBe(true);

    const afterGap = reduceServerMessage(gap, {
      seq: 4,
      type: "world_snapshot",
      body: { revision: 2, social_state: "engaged", people: [], objects: [], health: [] },
    });
    expect(afterGap.needsResync).toBe(true);
  });
});
