export interface ReplaySummary {
  id: string;
  label: string;
  directory: string;
}

interface ReplayResponse {
  messages?: unknown[];
}

const API_BASE = window.location.port === "5173" ? "http://127.0.0.1:8000" : "";

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

export async function runReplay(directory: string): Promise<unknown[]> {
  const response = await fetchJson<ReplayResponse>("/api/replay", {
    body: JSON.stringify({ directory }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });
  return response.messages ?? [];
}

export async function submitText(text: string): Promise<string> {
  const response = await fetchJson<{ response?: { text?: string; status?: string } }>("/api/text", {
    body: JSON.stringify({ text }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });
  return response.response?.text ?? response.response?.status ?? "No response.";
}

export async function startSession(): Promise<{ ok: boolean; running: boolean }> {
  return fetchJson("/api/session/start", { method: "POST" });
}

export async function stopSession(): Promise<{ ok: boolean; running: boolean }> {
  return fetchJson("/api/session/stop", { method: "POST" });
}

export async function uploadVisionFrame<T>(
  imageBase64: string,
): Promise<{
  behavior_timeline?: unknown;
  ok: boolean;
  vision_debug?: unknown;
  vision_status?: {
    face_detector: {
      name: string;
      status: string;
      detail: string | null;
    };
  } | null;
  world_snapshot?: T;
}> {
  return fetchJson("/api/vision/frame", {
    body: JSON.stringify({ image_base64: imageBase64 }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, init);
  if (!response.ok) {
    throw new Error(`${url} failed with ${response.status}`);
  }
  return (await response.json()) as T;
}
