import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { getHealth, getReplays, getWorld, type ReplaySummary } from "../lib/api";
import { connectSocket } from "../lib/socket";
import {
  initialState,
  reduceServerMessage,
  type ConnectionState,
  type DashboardState,
  type DashboardWorldSnapshot,
  type ServerMessage,
} from "../state/store";

export function useDashboardSocket() {
  const [state, dispatch] = useReducer(reduceServerMessage, initialState);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [replays, setReplays] = useState<ReplaySummary[]>([]);

  const socketRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(250);
  const retryRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastMessageRef = useRef(0);
  const intentionalCloseRef = useRef(false);
  const resyncInFlightRef = useRef(false);
  const initialConnectRef = useRef(true);
  const stateRef = useRef(state);
  stateRef.current = state;

  const resync = useCallback(async () => {
    if (resyncInFlightRef.current) return;
    resyncInFlightRef.current = true;
    setConnection("resyncing");
    try {
      const snapshot = await getWorld<DashboardWorldSnapshot>();
      dispatch({
        seq: stateRef.current.lastSequence + 1,
        type: "world_snapshot",
        body: snapshot,
      });
      setConnection("connected");
    } catch {
      setConnection("frozen");
    } finally {
      resyncInFlightRef.current = false;
    }
  }, [dispatch]);

  useEffect(() => {
    if (state.needsResync) {
      resync();
    }
  }, [state.needsResync, resync]);

  const openSocket = useCallback(() => {
    if (intentionalCloseRef.current) return;
    socketRef.current?.close();

    const socket = connectSocket((msg) => {
      dispatch(msg);
      lastMessageRef.current = Date.now();
    });
    socketRef.current = socket;

    socket.onopen = () => {
      backoffRef.current = 250;
      retryRef.current = 0;
      lastMessageRef.current = Date.now();

      if (!heartbeatTimerRef.current) {
        heartbeatTimerRef.current = setInterval(() => {
          const elapsed = Date.now() - lastMessageRef.current;
          if (elapsed > 60_000) {
            setConnection("reconnecting");
            socket.close();
          }
        }, 10_000);
      }

      if (initialConnectRef.current) {
        initialConnectRef.current = false;
        setConnection("connected");
      } else {
        resync();
      }
    };

    socket.onclose = () => {
      if (intentionalCloseRef.current) return;
      retryRef.current += 1;
      if (retryRef.current > 10) {
        setConnection("disconnected");
        return;
      }
      setConnection("reconnecting");
      const delay = Math.min(backoffRef.current, 5000);
      backoffRef.current *= 2;
      reconnectTimerRef.current = setTimeout(openSocket, delay + Math.random() * 100);
    };
  }, [dispatch, resync]);

  useEffect(() => {
    void Promise.all([getHealth(), getWorld<DashboardWorldSnapshot>(), getReplays()])
      .then(([, snapshot, replayList]) => {
        setReplays(replayList);
        dispatch({ seq: 1, type: "world_snapshot", body: snapshot });
      })
      .catch(() => setConnection("disconnected"));
    openSocket();
    return () => {
      intentionalCloseRef.current = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (heartbeatTimerRef.current) {
        clearInterval(heartbeatTimerRef.current);
      }
      socketRef.current?.close();
    };
  }, [openSocket, dispatch]);

  const sendAck = useCallback(
    (timelineId: string, ackType: string, extra?: Record<string, unknown>) => {
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send(
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
    },
    [],
  );

  return { connection, state, replays, dispatch, sendAck };
}
