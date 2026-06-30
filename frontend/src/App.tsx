import { useEffect, useId, useRef, useState } from "react";
import { MetricPanel } from "./components/MetricPanel";
import { TurnIndicator } from "./components/TurnIndicator";
import { TurnEvent, useAudioStream } from "./hooks/useAudioStream";

function useSessionId() {
  const id = useRef(crypto.randomUUID());
  return id.current;
}

export default function App() {
  const sessionId = useSessionId();
  const { isStreaming, lastEvent, error, start, stop } = useAudioStream(sessionId);

  // Running metric state
  const [cadenceFIR, setCadenceFIR] = useState(0);
  const [vadFIR, setVadFIR] = useState(0);
  const [cadenceFI, setCadenceFI] = useState(0);
  const [vadFI, setVadFI] = useState(0);
  const [totalDecisions, setTotalDecisions] = useState(0);
  const [meanDeadAir, setMeanDeadAir] = useState(0);
  const [showHow, setShowHow] = useState(false);

  // Fetch session metrics every 2s while streaming
  useEffect(() => {
    if (!isStreaming) return;
    const iv = setInterval(async () => {
      try {
        const r = await fetch(`/metrics/${sessionId}`);
        if (!r.ok) return;
        const d = await r.json();
        setCadenceFIR(d.cadence_FIR ?? 0);
        setVadFIR(d.vad_FIR ?? 0);
        setCadenceFI(d.cadence_false_interruptions ?? 0);
        setVadFI(d.vad_false_interruptions ?? 0);
        setTotalDecisions(d.total_decisions ?? 0);
        setMeanDeadAir(d.mean_dead_air_ms ?? 0);
      } catch {}
    }, 2000);
    return () => clearInterval(iv);
  }, [isStreaming, sessionId]);

  const cadenceLabel = lastEvent?.cadence?.label ?? null;
  const cadenceConf = lastEvent?.cadence?.confidence ?? null;
  const cadenceLatency = lastEvent?.cadence?.latency_ms ?? null;
  const vadLabel = lastEvent?.vad?.label ?? null;

  const falseTrigger =
    cadenceLabel === "mid_thought" && vadLabel === "turn_end";

  return (
    <div className="min-h-screen flex flex-col items-center px-4 py-12 gap-8 max-w-2xl mx-auto">
      {/* Header */}
      <div className="w-full">
        <h1 className="text-3xl font-semibold tracking-tight">Cadence</h1>
        <p className="text-zinc-400 text-sm mt-1">
          Smart turn-taking endpointer — fine-tuned wav2vec2 vs silero-VAD baseline
        </p>
      </div>

      {/* Controls */}
      <div className="w-full flex items-center gap-4">
        <button
          onClick={isStreaming ? stop : start}
          className={`px-6 py-3 rounded-lg font-semibold text-sm transition-all ${
            isStreaming
              ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-100"
              : "bg-emerald-600 hover:bg-emerald-500 text-white"
          }`}
        >
          {isStreaming ? "⏹ Stop" : "🎙 Start talking"}
        </button>
        {isStreaming && (
          <span className="flex items-center gap-2 text-sm text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            Listening…
          </span>
        )}
        {error && <span className="text-red-400 text-sm">{error}</span>}
      </div>

      {/* Live indicators */}
      <div className="w-full grid grid-cols-2 gap-4">
        <TurnIndicator
          name="Cadence (ours)"
          label={cadenceLabel}
          confidence={cadenceConf}
          latency={cadenceLatency}
        />
        <TurnIndicator
          name="silero-VAD baseline"
          label={vadLabel}
          confidence={null}
        />
      </div>

      {/* False trigger callout */}
      {falseTrigger && (
        <div className="w-full border border-red-800 bg-red-950/40 rounded-lg px-4 py-3 text-red-400 text-sm">
          ← VAD would have interrupted here (false trigger). Cadence correctly detected mid-thought.
        </div>
      )}

      {/* Metrics */}
      <div className="w-full">
        <MetricPanel
          cadenceFIR={cadenceFIR}
          vadFIR={vadFIR}
          cadenceFalseInterrupts={cadenceFI}
          vadFalseInterrupts={vadFI}
          totalDecisions={totalDecisions}
          meanDeadAir={meanDeadAir}
        />
      </div>

      {/* How it works */}
      <div className="w-full border border-zinc-800 rounded-lg overflow-hidden">
        <button
          className="w-full px-5 py-3 text-left text-sm text-zinc-400 hover:text-zinc-200 flex justify-between"
          onClick={() => setShowHow((p) => !p)}
        >
          <span>How it works</span>
          <span>{showHow ? "▲" : "▼"}</span>
        </button>
        {showHow && (
          <div className="px-5 pb-5 text-sm text-zinc-400 space-y-3 border-t border-zinc-800 pt-4">
            <p>
              Your browser streams 20ms PCM audio chunks over a WebSocket. On silence
              onset, the server runs two models in parallel:
            </p>
            <ol className="list-decimal list-inside space-y-1">
              <li>
                <strong className="text-zinc-200">Cadence</strong> — fine-tuned{" "}
                <code className="text-xs bg-zinc-800 px-1 rounded">wav2vec2-base</code> on
                2s audio windows, classifying pauses as <em>turn_end</em> vs <em>mid_thought</em>
              </li>
              <li>
                <strong className="text-zinc-200">silero-VAD</strong> — standard voice
                activity detector that only knows if audio is present
              </li>
            </ol>
            <p>
              The <strong className="text-amber-400">False Interruption Rate</strong> counts
              how often a model says "turn ended" when you were still thinking — the key metric
              for voice agent quality.
            </p>
            <div className="flex gap-3 mt-2 text-xs">
              <a
                href="https://github.com/jahnaviyelamanchi/cadence"
                className="text-zinc-400 hover:text-zinc-100 underline"
                target="_blank"
                rel="noreferrer"
              >
                GitHub
              </a>
              <a
                href="https://huggingface.co/jahnaviyelamanchi/cadence"
                className="text-zinc-400 hover:text-zinc-100 underline"
                target="_blank"
                rel="noreferrer"
              >
                Model card
              </a>
            </div>
          </div>
        )}
      </div>

      <footer className="text-zinc-700 text-xs mt-auto">
        Cadence · jy4857@nyu.edu
      </footer>
    </div>
  );
}
