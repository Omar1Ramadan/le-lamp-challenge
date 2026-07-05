import { expect, test } from "@playwright/test";

test("replay demonstrates engagement, attention seeking, memory, and recall", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Load core journey replay" }).click();
  await expect(page.getByTestId("demo-step-engagement")).toHaveAttribute("data-complete", "true");
  await expect(page.getByText("Seeking attention: level 1")).toBeVisible();
  await expect(page.getByRole("article", { name: /memory: keys/i })).toContainText("right side");
  await page.getByLabel("Ask the lamp").fill("Where are my keys?");
  await page.getByRole("button", { name: "Ask" }).click();
  await expect(page.getByTestId("lamp-answer")).toContainText("right side of the desk");
  await page.getByRole("button", { name: "Show evidence" }).click();
  await expect(page.getByText("observation-core-keys-2")).toBeVisible();
});
