import type { ServerMessage } from "../state/store";

export function connectSocket(
  onMessage: (message: ServerMessage) => void,
  onOpen?: (socket: WebSocket) => void,
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.port === "5173" ? "127.0.0.1:8000" : window.location.host;
  const socket = new WebSocket(`${protocol}://${host}/ws`);
  socket.onmessage = (event) => {
    const raw = JSON.parse(event.data as string) as ServerMessage;
    onMessage(raw);
  };
  socket.onopen = () => onOpen?.(socket);
  return socket;
}

export function sendAck(
  socket: WebSocket,
  timelineId: string,
  ackType: string,
  extra?: Record<string, unknown>,
): void {
  if (socket.readyState !== WebSocket.OPEN) return;
  socket.send(
    JSON.stringify({
      type: "simulator_ack",
      body: {
        timeline_id: timelineId,
        ack_type: ackType,
        frontend_time_mono_ms: Math.floor(performance.now()),
        ...extra,
      },
    }),
  );
}
