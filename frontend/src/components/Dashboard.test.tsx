import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Inspector } from "./Inspector";

describe("Inspector", () => {
  it("shows evidence and degraded health without color-only meaning", () => {
    render(
      <Inspector
        state="engaged"
        evidence={[{ id: "observation-1", label: "keys", location: "right side of desk" }]}
        health={[{ component: "cloud", status: "degraded", detail: "offline fallback" }]}
      />,
    );
    expect(screen.getByText("keys")).toBeVisible();
    expect(screen.getByText(/degraded/i)).toBeVisible();
    expect(screen.getByText(/offline fallback/i)).toBeVisible();
  });
});
