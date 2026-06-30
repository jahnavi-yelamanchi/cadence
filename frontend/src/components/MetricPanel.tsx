interface MetricRowProps {
  label: string;
  cadence: number | string;
  vad: number | string;
  lowerIsBetter?: boolean;
}

function MetricRow({ label, cadence, vad, lowerIsBetter = true }: MetricRowProps) {
  const cadenceNum = typeof cadence === "number" ? cadence : parseFloat(String(cadence));
  const vadNum = typeof vad === "number" ? vad : parseFloat(String(vad));
  const cadenceWins = lowerIsBetter ? cadenceNum <= vadNum : cadenceNum >= vadNum;

  return (
    <div className="grid grid-cols-3 gap-4 py-2 border-b border-zinc-800 last:border-0">
      <span className="text-zinc-400 text-sm">{label}</span>
      <span className={`text-sm text-right ${cadenceWins ? "text-emerald-400 font-semibold" : "text-zinc-300"}`}>
        {typeof cadence === "number" ? cadence.toFixed(3) : cadence}
      </span>
      <span className={`text-sm text-right ${!cadenceWins ? "text-emerald-400 font-semibold" : "text-zinc-300"}`}>
        {typeof vad === "number" ? vad.toFixed(3) : vad}
      </span>
    </div>
  );
}

interface Props {
  cadenceFIR: number;
  vadFIR: number;
  cadenceFalseInterrupts: number;
  vadFalseInterrupts: number;
  totalDecisions: number;
  meanDeadAir: number;
}

export function MetricPanel({
  cadenceFIR,
  vadFIR,
  cadenceFalseInterrupts,
  vadFalseInterrupts,
  totalDecisions,
  meanDeadAir,
}: Props) {
  return (
    <div className="border border-zinc-800 rounded-lg p-5">
      <h2 className="text-xs text-zinc-500 uppercase tracking-widest mb-3">
        Session Metrics <span className="text-zinc-700">({totalDecisions} decisions)</span>
      </h2>
      <div className="grid grid-cols-3 gap-4 mb-2">
        <span />
        <span className="text-xs text-zinc-500 text-right">Cadence</span>
        <span className="text-xs text-zinc-500 text-right">silero-VAD</span>
      </div>
      <MetricRow
        label="False Interruption Rate"
        cadence={cadenceFIR}
        vad={vadFIR}
        lowerIsBetter
      />
      <MetricRow
        label="False Interruptions"
        cadence={cadenceFalseInterrupts}
        vad={vadFalseInterrupts}
        lowerIsBetter
      />
      <MetricRow
        label="Mean Dead Air (ms)"
        cadence={meanDeadAir}
        vad="—"
        lowerIsBetter
      />
    </div>
  );
}
