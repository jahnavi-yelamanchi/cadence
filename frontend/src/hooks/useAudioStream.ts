/**
 * useAudioStream
 *
 * Captures microphone audio, resamples from browser native rate (44.1/48kHz)
 * to 16kHz using an AudioWorklet, and streams 20ms PCM Float32 chunks over
 * a WebSocket to the Cadence server.
 *
 * Returns the latest server decision event and controls to start/stop.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const SERVER_SAMPLE_RATE = 16000;
const CHUNK_MS = 20;
const CHUNK_SAMPLES = (SERVER_SAMPLE_RATE * CHUNK_MS) / 1000; // 320

export interface TurnEvent {
  cadence: { label: string; confidence: number; latency_ms: number } | null;
  vad: { label: string; confidence?: number } | null;
  ts: number;
}

export function useAudioStream(sessionId: string) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [lastEvent, setLastEvent] = useState<TurnEvent | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const resampleBufferRef = useRef<Float32Array>(new Float32Array(0));

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;

      const ctx = new AudioContext();
      audioCtxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      // ScriptProcessor is deprecated but universally supported; adequate for demo
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      const nativeRate = ctx.sampleRate;
      const ratio = nativeRate / SERVER_SAMPLE_RATE;

      const wsUrl =
        window.location.hostname === "localhost"
          ? `ws://localhost:8000/stream/${sessionId}`
          : `wss://${window.location.host}/stream/${sessionId}`;
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => setIsStreaming(true);
      ws.onclose = () => setIsStreaming(false);
      ws.onerror = () => setError("WebSocket error — is the server running?");
      ws.onmessage = (e) => {
        try {
          setLastEvent(JSON.parse(e.data as string) as TurnEvent);
        } catch {}
      };

      processor.onaudioprocess = (e) => {
        if (ws.readyState !== WebSocket.OPEN) return;

        const input = e.inputBuffer.getChannelData(0);
        // Downsample by averaging blocks of `ratio` samples
        const downsampled = downsample(input, ratio);

        // Accumulate into buffer and flush complete 320-sample chunks
        const combined = new Float32Array(
          resampleBufferRef.current.length + downsampled.length
        );
        combined.set(resampleBufferRef.current);
        combined.set(downsampled, resampleBufferRef.current.length);

        let offset = 0;
        while (offset + CHUNK_SAMPLES <= combined.length) {
          ws.send(combined.slice(offset, offset + CHUNK_SAMPLES).buffer);
          offset += CHUNK_SAMPLES;
        }
        resampleBufferRef.current = combined.slice(offset);
      };

      source.connect(processor);
      processor.connect(ctx.destination);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Microphone access denied");
    }
  }, [sessionId]);

  const stop = useCallback(() => {
    processorRef.current?.disconnect();
    audioCtxRef.current?.close();
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setIsStreaming(false);
  }, []);

  useEffect(() => () => stop(), [stop]);

  return { isStreaming, lastEvent, error, start, stop };
}

function downsample(input: Float32Array, ratio: number): Float32Array {
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.floor((i + 1) * ratio);
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    out[i] = sum / (end - start);
  }
  return out;
}
