import { useEffect, useRef, useState } from "react";

import { uploadVisionFrame } from "../lib/api";
import type { BehaviorTimeline } from "../contracts/generated";
import type { DashboardWorldSnapshot } from "../state/store";

type DeviceStatus = "idle" | "starting" | "live" | "error";

interface VisionDebug {
  attention_proxy: number;
  box: { height: number; width: number; x: number; y: number };
  eyes?: { height: number; width: number; x: number; y: number }[];
  gaze_source?: string;
  target: { x: number; y: number };
}

interface DevicePanelProps {
  onBehaviorTimeline?: (timeline: BehaviorTimeline) => void;
  onWorldSnapshot?: (snapshot: DashboardWorldSnapshot) => void;
}

export function DevicePanel({ onBehaviorTimeline, onWorldSnapshot }: DevicePanelProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const uploadInFlightRef = useRef(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [status, setStatus] = useState<DeviceStatus>("idle");
  const [message, setMessage] = useState("Camera and microphone are off.");
  const [monitorAudio, setMonitorAudio] = useState(false);
  const [visionDebug, setVisionDebug] = useState<VisionDebug | null>(null);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
    }
    if (audioRef.current) {
      audioRef.current.srcObject = monitorAudio ? stream : null;
    }
  }, [monitorAudio, stream]);

  useEffect(() => {
    return () => stopStream(stream);
  }, [stream]);

  useEffect(() => {
    if (!stream) {
      return;
    }
    const timer = window.setInterval(() => {
      void sendVideoFrame();
    }, 1000);
    return () => window.clearInterval(timer);
  }, [stream]);

  async function startDevices() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("error");
      setMessage("This browser does not expose camera or microphone access.");
      return;
    }

    setStatus("starting");
    setMessage("Requesting browser permission...");
    try {
      const nextStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: true,
      });
      stopStream(stream);
      setStream(nextStream);
      setStatus("live");
      setMessage("Browser camera and microphone are live.");
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Device permission was denied.");
    }
  }

  function stopDevices() {
    stopStream(stream);
    setStream(null);
    setMonitorAudio(false);
    setStatus("idle");
    setMessage("Camera and microphone are off.");
  }

  async function sendVideoFrame() {
    if (uploadInFlightRef.current) {
      return;
    }
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return;
    }
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width === 0 || height === 0) {
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    context.drawImage(video, 0, 0, width, height);
    const imageBase64 = await canvasToJpegBase64(canvas);
    if (!imageBase64) {
      return;
    }
    uploadInFlightRef.current = true;
    try {
      const response = await uploadVisionFrame<DashboardWorldSnapshot>(imageBase64);
      if (response.world_snapshot) {
        onWorldSnapshot?.(response.world_snapshot);
        const peopleCount = response.world_snapshot.people.length;
        setMessage(
          peopleCount > 0
            ? `Backend vision updated: ${peopleCount} person${peopleCount === 1 ? "" : "s"} tracked.`
            : "Backend vision updated: no person tracked.",
        );
      }
      setVisionDebug(isVisionDebug(response.vision_debug) ? response.vision_debug : null);
      if (response.behavior_timeline) {
        onBehaviorTimeline?.(response.behavior_timeline as BehaviorTimeline);
      }
    } catch {
      setMessage("Browser devices are live, but backend vision frame upload failed.");
    } finally {
      uploadInFlightRef.current = false;
    }
  }

  return (
    <section className="panel device-panel" aria-label="Camera and microphone tools">
      <h2>Input devices</h2>
      <div className="device-preview">
        {stream ? (
          <>
            <video ref={videoRef} autoPlay muted playsInline aria-label="Camera preview" />
            <VisionOverlay debug={visionDebug} />
          </>
        ) : (
          <div className="device-placeholder">No camera preview</div>
        )}
      </div>
      <audio ref={audioRef} autoPlay />
      <p>
        Status: <strong>{status}</strong>
      </p>
      <p>{message}</p>
      <div className="device-actions">
        <button type="button" onClick={() => void startDevices()} disabled={status === "starting"}>
          {stream ? "Restart devices" : "Start camera and mic"}
        </button>
        <button type="button" onClick={stopDevices} disabled={!stream}>
          Stop
        </button>
      </div>
      <label className="device-monitor">
        <input
          type="checkbox"
          checked={monitorAudio}
          disabled={!stream}
          onChange={(event) => setMonitorAudio(event.target.checked)}
        />
        Monitor microphone audio
      </label>
    </section>
  );
}

function VisionOverlay({ debug }: { debug: VisionDebug | null }) {
  if (!debug) {
    return <div className="vision-overlay vision-overlay-empty">No backend pose</div>;
  }

  const boxStyle = {
    height: `${debug.box.height * 100}%`,
    left: `${debug.box.x * 100}%`,
    top: `${debug.box.y * 100}%`,
    width: `${debug.box.width * 100}%`,
  };
  const targetStyle = {
    left: `${debug.target.x * 100}%`,
    top: `${debug.target.y * 100}%`,
  };
  const eyes = Array.isArray(debug.eyes) ? debug.eyes : [];

  return (
    <div className="vision-overlay" aria-label="Backend vision pose overlay">
      <div className="vision-box" style={boxStyle} />
      {eyes.map((eye, index) => (
        <div
          className="vision-eye"
          key={`${eye.x}-${eye.y}-${index}`}
          style={{
            height: `${eye.height * debug.box.height * 100}%`,
            left: `${(debug.box.x + eye.x * debug.box.width) * 100}%`,
            top: `${(debug.box.y + eye.y * debug.box.height) * 100}%`,
            width: `${eye.width * debug.box.width * 100}%`,
          }}
        />
      ))}
      <div className="vision-target" style={targetStyle} />
      <span className="vision-label">
        {debug.gaze_source ?? "backend pose"} · attention{" "}
        {Math.round(debug.attention_proxy * 100)}%
      </span>
    </div>
  );
}

function isVisionDebug(value: unknown): value is VisionDebug {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candidate = value as Partial<VisionDebug>;
  return (
    typeof candidate.attention_proxy === "number" &&
    typeof candidate.box?.height === "number" &&
    typeof candidate.box.width === "number" &&
    typeof candidate.box.x === "number" &&
    typeof candidate.box.y === "number" &&
    typeof candidate.target?.x === "number" &&
    typeof candidate.target.y === "number"
  );
}

function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

function canvasToJpegBase64(canvas: HTMLCanvasElement): Promise<string | null> {
  return new Promise((resolve) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          resolve(null);
          return;
        }
        const reader = new FileReader();
        reader.onload = () => {
          const result = typeof reader.result === "string" ? reader.result : "";
          resolve(result.split(",", 2)[1] || null);
        };
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(blob);
      },
      "image/jpeg",
      0.7,
    );
  });
}
