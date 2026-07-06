import type { ServerMessage } from "../state/store";

export function connectSocket(onMessage: (message: ServerMessage) => void): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
  socket.onmessage = (event) => {
    const raw = JSON.parse(event.data as string) as ServerMessage;
    onMessage(raw);
  };
  return socket;
}
