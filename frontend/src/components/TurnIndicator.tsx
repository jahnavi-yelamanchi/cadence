interface Props {
  label: string | null;
  confidence: number | null;
  name: string;
  latency?: number | null;
}

const LABEL_STYLE: Record<string, string> = {
  turn_end: "text-amber-400 border-amber-400",
  mid_thought: "text-emerald-400 border-emerald-400",
};

const LABEL_TEXT: Record<string, string> = {
  turn_end: "TURN END",
  mid_thought: "MID-THOUGHT",
};

export function TurnIndicator({ label, confidence, name, latency }: Props) {
  const style = label ? LABEL_STYLE[label] ?? "text-zinc-400 border-zinc-700" : "text-zinc-600 border-zinc-800";
  const display = label ? LABEL_TEXT[label] ?? label.toUpperCase() : "WAITING";

  return (
    <div className={`border rounded-lg p-4 flex flex-col gap-1 ${style}`}>
      <span className="text-xs text-zinc-500 uppercase tracking-widest">{name}</span>
      <span className={`text-2xl font-semibold tracking-tight ${style}`}>{display}</span>
      <div className="flex gap-4 text-xs text-zinc-500 mt-1">
        {confidence != null && (
          <span>conf <span className="text-zinc-300">{(confidence * 100).toFixed(0)}%</span></span>
        )}
        {latency != null && (
          <span>latency <span className="text-zinc-300">{latency.toFixed(0)}ms</span></span>
        )}
      </div>
    </div>
  );
}
