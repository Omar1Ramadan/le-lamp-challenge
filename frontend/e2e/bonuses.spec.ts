import { expect, test } from "@playwright/test";

test("bonus replay demonstrates speaker, affect, preference, interruption, and media suppression", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Load bonus journey replay" }).click();
  await expect(page.getByText("Active speaker: Person B")).toBeVisible();
  await expect(page.getByText("Affect confidence gated below 0.60")).toBeVisible();
  await expect(page.getByText("Preference score changed then reset")).toBeVisible();
  await expect(page.getByText("Speech interruption cancellation under 120 ms")).toBeVisible();
  await expect(page.getByText("Television suppression active")).toBeVisible();
});
