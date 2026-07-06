export interface ReplaySummary {
  id: string;
  label: string;
  directory: string;
}

export async function getHealth(): Promise<{ status: string }> {
  return fetchJson("/api/health");
}

export async function getWorld<T>(): Promise<T> {
  return fetchJson("/api/world");
}

export async function getReplays(): Promise<ReplaySummary[]> {
  const response = await fetchJson<{ replays: ReplaySummary[] }>("/api/replays");
  return response.replays;
}

export async function runReplay(directory: string): Promise<void> {
  await fetchJson("/api/replay", {
    body: JSON.stringify({ directory }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });
}

export async function submitText(text: string): Promise<string> {
  const response = await fetchJson<{ response?: { text?: string; status?: string } }>("/api/text", {
    body: JSON.stringify({ text }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });
  return response.response?.text ?? response.response?.status ?? "No response.";
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`${url} failed with ${response.status}`);
  }
  return (await response.json()) as T;
}
