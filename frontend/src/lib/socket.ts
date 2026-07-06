import type { ServerMessage } from "../state/store";

export function connectSocket(onMessage: (message: ServerMessage) => void): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.port === "5173" ? "127.0.0.1:8000" : window.location.host;
  const socket = new WebSocket(`${protocol}://${host}/ws`);
  socket.onmessage = (event) => {
    const raw = JSON.parse(event.data as string) as ServerMessage;
    onMessage(raw);
  };
  return socket;
}
