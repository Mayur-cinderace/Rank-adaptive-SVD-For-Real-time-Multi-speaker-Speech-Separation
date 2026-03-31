"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Tooltip,
  Filler,
  Legend,
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Tooltip, Filler, Legend
);

// ─── Types ───────────────────────────────────────────────────────────────────

type Mode = "upload" | "dataset" | "live";
type PipelineStage = "input" | "beamforming" | "diarization" | "separation" | "enhancement";

type InputResponse = {
  input_id: string;
  source: string;
  sample_rate: number;
  channels: number;
  samples: number;
  duration_sec: number;
  snr_db: number | null;
  rms_db: number | null;
  peak_db: number | null;
  waveform: number[];
  spectrogram_png_base64: string;
  channel_files?: string[];
  all_zip_audio_files?: string[];
  selected_zip_audio_files?: string[];
};

type MetricsEntry = {
  stoi: number | null;
  pesq: number | null;
  sdr_db: number | null;
  sir_db: number | null;
  sar_db: number | null;
  rtf: number | null;
};

type ChannelEntry = {
  index: number;
  waveform: number[];
  n_samples: number;
  wav_b64: string | null;
};

type MethodResult = {
  method: string;
  sr: number;
  n_sources: number;
  channels: ChannelEntry[];
  metrics: MetricsEntry;
  metadata: Record<string, unknown>;
  error?: string;
};

type CompareResponse = {
  input_id: string;
  n_channels: number;
  duration_sec: number;
  sr: number;
  preprocessing: {
    normalized: boolean;
    n_fft: number;
    hop_length: number;
    win_length: number;
    target_sr: number;
  };
  comparison_table: Array<{
    method: string;
    stoi: number | null;
    pesq: number | null;
    sdr_db: number | null;
    sir_db: number | null;
    sar_db: number | null;
    rtf: number | null;
    n_sources: number;
    metadata: Record<string, unknown>;
  }>;
  methods: Record<string, MethodResult>;
};

const STAGES: { id: PipelineStage; label: string; short: string }[] = [
  { id: "input",       label: "Input",       short: "01" },
  { id: "beamforming", label: "Beamforming", short: "02" },
  { id: "diarization", label: "Diarization", short: "03" },
  { id: "separation",  label: "Separation",  short: "04" },
  { id: "enhancement", label: "Enhancement", short: "05" },
];

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const METHOD_COLORS: Record<string, string> = {
  beamforming: "#2563EB",
  svd:         "#7C3AED",
  neural:      "#059669",
};

const METHOD_LABELS: Record<string, string> = {
  beamforming: "MVDR Beamforming",
  svd:         "Rank-Adaptive SVD",
  neural:      "Conv-TasNet (Neural)",
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmt(n: number | null | undefined, decimals = 2, unit = ""): string {
  if (n == null) return "—";
  return n.toFixed(decimals) + unit;
}

async function post<T>(path: string, fd: FormData): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
  return data as T;
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-pill">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

function WaveChart({ data, color, height = 80 }: { data: number[]; color: string; height?: number }) {
  const chartData = useMemo(() => ({
    labels: data.map((_, i) => i),
    datasets: [{
      data,
      borderColor: color,
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0,
      fill: "origin",
      backgroundColor: color + "18",
    }],
  }), [data, color]);

  return (
    <Line
      data={chartData}
      options={{
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: { x: { display: false }, y: { display: false, min: -1, max: 1 } },
      }}
      height={height}
    />
  );
}

function MetricBar({ label, value, max, color }: { label: string; value: number | null; max: number; color: string }) {
  const pct = value != null ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}>
        <span style={{ color: "var(--color-text-secondary, #6b7280)", fontWeight: 500 }}>{label}</span>
        <span style={{ fontWeight: 600, color: "var(--color-text-primary, #111)" }}>
          {value != null ? value.toFixed(3) : "—"}
        </span>
      </div>
      <div style={{ height: 6, background: "var(--color-background-secondary, #f3f4f6)", borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 3, transition: "width 0.4s" }} />
      </div>
    </div>
  );
}

function PipelineNav({
  active, completed, onChange,
}: {
  active: PipelineStage;
  completed: Set<PipelineStage>;
  onChange: (s: PipelineStage) => void;
}) {
  return (
    <nav className="pipeline-nav" aria-label="Pipeline stages">
      {STAGES.map((s, i) => {
        const isDone = completed.has(s.id);
        const isActive = s.id === active;
        const isLocked = !isDone && s.id !== "input" && !isActive;
        return (
          <button
            key={s.id}
            className={`stage-btn ${isActive ? "stage-active" : ""} ${isDone ? "stage-done" : ""} ${isLocked ? "stage-locked" : ""}`}
            onClick={() => !isLocked && onChange(s.id)}
            disabled={isLocked}
            aria-current={isActive ? "step" : undefined}
          >
            <span className="stage-num">{isDone ? "✓" : s.short}</span>
            <span className="stage-label">{s.label}</span>
            {i < STAGES.length - 1 && <span className="stage-line" aria-hidden />}
          </button>
        );
      })}
    </nav>
  );
}

// ─── Icons ────────────────────────────────────────────────────────────────────

const BeamIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M12 2v20M4 6l8 6-8 6M20 6l-8 6 8 6" />
  </svg>
);
const DiarIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <circle cx="8" cy="8" r="3" /><circle cx="16" cy="8" r="3" />
    <path d="M2 20c0-3.3 2.7-6 6-6M16 14c3.3 0 6 2.7 6 6" />
    <line x1="12" y1="4" x2="12" y2="20" strokeDasharray="2 2" />
  </svg>
);
const SepIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 12h4l3-7 4 14 3-7h4" />
    <line x1="12" y1="2" x2="12" y2="22" strokeDasharray="2 2" opacity="0.4" />
  </svg>
);
const EnhIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 18l5-10 4 6 3-4 6 8" />
    <circle cx="19" cy="5" r="2" fill="currentColor" opacity="0.5" />
  </svg>
);
const MicIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <rect x="9" y="2" width="6" height="12" rx="3" />
    <path d="M5 10a7 7 0 0014 0M12 19v3M8 22h8" />
  </svg>
);
const StopIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <rect x="5" y="5" width="14" height="14" rx="2" />
  </svg>
);

// ─── Comparison Panel ────────────────────────────────────────────────────────

function ComparisonPanel({ data, inputId }: { data: CompareResponse | null; inputId: string | null }) {
  const [activeMethod, setActiveMethod] = useState<string | null>(null);

  if (!data) {
    return (
      <div className="preview-empty">
        <div className="preview-empty-icon"><SepIcon /></div>
        <p>{inputId ? "Run comparison to see results." : "Complete the Input stage first."}</p>
      </div>
    );
  }

  const table = data.comparison_table;
  const methods = Object.keys(data.methods);
  const selectedResult = activeMethod ? data.methods[activeMethod] : null;

  // Bar chart data for STOI comparison
  const stoiChartData = {
    labels: table.map(r => METHOD_LABELS[r.method] ?? r.method),
    datasets: [{
      label: "STOI",
      data: table.map(r => r.stoi ?? 0),
      backgroundColor: table.map(r => (METHOD_COLORS[r.method] ?? "#888") + "cc"),
      borderColor: table.map(r => METHOD_COLORS[r.method] ?? "#888"),
      borderWidth: 1.5,
      borderRadius: 4,
    }],
  };

  return (
    <div>
      {/* Preprocessing badge */}
      <div style={{
        display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20,
        padding: "10px 14px",
        background: "var(--color-background-secondary, #f9f9f8)",
        borderRadius: 8,
        border: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.08))",
      }}>
        <span style={{ fontSize: 11, color: "var(--color-text-secondary, #6b7280)", fontWeight: 600, marginRight: 4 }}>
          Shared preprocessing:
        </span>
        {[
          `SR ${data.preprocessing.target_sr / 1000} kHz`,
          `FFT ${data.preprocessing.n_fft}`,
          `Hop ${data.preprocessing.hop_length}`,
          `Normalized`,
        ].map(t => (
          <span key={t} style={{
            fontSize: 11, padding: "2px 8px", borderRadius: 12,
            background: "rgba(15,118,110,0.08)", color: "var(--accent-input)",
            fontWeight: 500,
          }}>{t}</span>
        ))}
      </div>

      {/* STOI bar chart */}
      <p className="section-title" style={{ marginBottom: 10 }}>STOI Comparison</p>
      <div style={{ height: 120, marginBottom: 20 }}>
        <Bar
          data={stoiChartData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: (ctx) => ` STOI: ${(ctx.raw as number).toFixed(3)}`,
                },
              },
            },
            scales: {
              x: { grid: { display: false }, ticks: { font: { size: 11 } } },
              y: { min: 0, max: 1, grid: { color: "rgba(0,0,0,0.05)" }, ticks: { font: { size: 10 } } },
            },
          }}
        />
      </div>

      {/* Metric table */}
      <p className="section-title" style={{ marginBottom: 10 }}>Full Metrics</p>
      <div style={{ overflowX: "auto", marginBottom: 20 }}>
        <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1.5px solid var(--color-border-tertiary, rgba(0,0,0,0.1))" }}>
              {["Method", "STOI", "PESQ", "SDR (dB)", "SIR (dB)", "SAR (dB)", "RTF"].map(h => (
                <th key={h} style={{ textAlign: "left", padding: "6px 10px", fontWeight: 600, fontSize: 11, color: "var(--color-text-secondary, #6b7280)", letterSpacing: "0.04em" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.map((row) => (
              <tr
                key={row.method}
                style={{
                  borderBottom: "0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.06))",
                  cursor: "pointer",
                  background: activeMethod === row.method ? (METHOD_COLORS[row.method] + "08") : "transparent",
                }}
                onClick={() => setActiveMethod(activeMethod === row.method ? null : row.method)}
              >
                <td style={{ padding: "8px 10px" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{
                      width: 8, height: 8, borderRadius: "50%",
                      background: METHOD_COLORS[row.method] ?? "#888",
                      flexShrink: 0,
                    }} />
                    <span style={{ fontWeight: 500 }}>{METHOD_LABELS[row.method] ?? row.method}</span>
                  </span>
                </td>
                <td style={{ padding: "8px 10px", fontVariantNumeric: "tabular-nums", fontWeight: row.method === "svd" ? 600 : 400 }}>{fmt(row.stoi, 3)}</td>
                <td style={{ padding: "8px 10px", fontVariantNumeric: "tabular-nums" }}>{fmt(row.pesq, 2)}</td>
                <td style={{ padding: "8px 10px", fontVariantNumeric: "tabular-nums" }}>{fmt(row.sdr_db, 1, " dB")}</td>
                <td style={{ padding: "8px 10px", fontVariantNumeric: "tabular-nums" }}>{fmt(row.sir_db, 1, " dB")}</td>
                <td style={{ padding: "8px 10px", fontVariantNumeric: "tabular-nums" }}>{fmt(row.sar_db, 1, " dB")}</td>
                <td style={{ padding: "8px 10px", fontVariantNumeric: "tabular-nums", color: "var(--color-text-secondary, #6b7280)" }}>{fmt(row.rtf, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Method selector tabs */}
      <p className="section-title" style={{ marginBottom: 10 }}>Separated Audio</p>
      <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
        {methods.map(m => (
          <button
            key={m}
            onClick={() => setActiveMethod(m === activeMethod ? null : m)}
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              border: `1.5px solid ${METHOD_COLORS[m] ?? "#888"}`,
              background: activeMethod === m ? METHOD_COLORS[m] : "transparent",
              color: activeMethod === m ? "#fff" : (METHOD_COLORS[m] ?? "#888"),
              fontSize: 12, fontWeight: 500, cursor: "pointer",
            }}
          >
            {METHOD_LABELS[m] ?? m}
          </button>
        ))}
      </div>

      {selectedResult && !selectedResult.error && selectedResult.channels?.length > 0 && (
        <div>
          {selectedResult.channels.map((ch) => (
            <div key={ch.index} style={{ marginBottom: 14 }}>
              <p className="signal-label">Source {ch.index + 1} — {ch.n_samples} samples</p>
              <div style={{ height: 60, marginBottom: 8 }}>
                <WaveChart data={ch.waveform} color={METHOD_COLORS[selectedResult.method] ?? "#888"} height={60} />
              </div>
              {ch.wav_b64 && (
                <audio
                  controls
                  src={`data:audio/wav;base64,${ch.wav_b64}`}
                  style={{ width: "100%", borderRadius: 6 }}
                />
              )}
            </div>
          ))}

          {/* Per-method metric bars */}
          <div style={{ marginTop: 14 }}>
            <MetricBar label="STOI" value={selectedResult.metrics.stoi} max={1} color={METHOD_COLORS[selectedResult.method] ?? "#888"} />
            <MetricBar label="PESQ (normalised)" value={selectedResult.metrics.pesq != null ? selectedResult.metrics.pesq / 4.5 : null} max={1} color={METHOD_COLORS[selectedResult.method] ?? "#888"} />
          </div>

          {selectedResult.metadata && (
            <details style={{ marginTop: 10 }}>
              <summary style={{ fontSize: 11, cursor: "pointer", color: "var(--color-text-secondary, #6b7280)" }}>
                Method metadata
              </summary>
              <pre style={{
                fontSize: 10, marginTop: 6, padding: 10,
                background: "var(--color-background-secondary, #f3f4f6)",
                borderRadius: 6, overflow: "auto", maxHeight: 200,
              }}>
                {JSON.stringify(selectedResult.metadata, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}

      {selectedResult?.error && (
        <div className="error-msg">{selectedResult.error}</div>
      )}
    </div>
  );
}

// ─── Beamforming Stage Panel ─────────────────────────────────────────────────

function BeamformingPanel({
  inputId,
  onDone,
}: {
  inputId: string | null;
  onDone: (result: MethodResult) => void;
}) {
  const [variant, setVariant] = useState<"mvdr" | "das">("mvdr");
  const [micSpacing, setMicSpacing] = useState(0.05);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<MethodResult | null>(null);

  async function run() {
    if (!inputId) return;
    setLoading(true); setError("");
    try {
      const fd = new FormData();
      fd.append("input_id", inputId);
      fd.append("variant", variant);
      fd.append("mic_spacing_m", String(micSpacing));
      const data = await post<MethodResult>("/api/separation/beamforming", fd);
      setResult(data);
      onDone(data);
    } catch (e) { setError(e instanceof Error ? e.message : "Failed"); }
    finally { setLoading(false); }
  }

  return (
    <div>
      <div className="form-row">
        <label className="form-label">Variant</label>
        <div style={{ display: "flex", gap: 8 }}>
          {(["mvdr", "das"] as const).map(v => (
            <button
              key={v}
              className={`mode-tab ${variant === v ? "active" : ""}`}
              onClick={() => setVariant(v)}
              style={{ flex: 1 }}
            >
              {v === "mvdr" ? "MVDR (Capon)" : "Delay-and-Sum"}
            </button>
          ))}
        </div>
      </div>

      <div className="form-row">
        <label className="form-label">Mic spacing — {micSpacing.toFixed(3)} m</label>
        <div className="slider-row">
          <input type="range" min={0.01} max={0.3} step={0.01} value={micSpacing}
            onChange={e => setMicSpacing(Number(e.target.value))} />
          <span className="slider-val">{micSpacing.toFixed(2)} m</span>
        </div>
      </div>

      <p style={{ fontSize: 11, color: "var(--color-text-secondary, #6b7280)", marginBottom: 12, lineHeight: 1.5 }}>
        Delays are estimated automatically from the audio using GCC-PHAT cross-correlation — no oracle DOA required.
      </p>

      <button className="btn-primary" onClick={run} disabled={loading || !inputId}
        style={{ "--btn-color": "var(--accent-b)" } as React.CSSProperties}>
        {loading ? "Running beamforming…" : "Run Beamforming →"}
      </button>

      {loading && <div className="loading-bar" />}
      {error && <div className="error-msg">{error}</div>}

      {result && (
        <div style={{ marginTop: 20 }}>
          <div className="stats-grid" style={{ gridTemplateColumns: "repeat(3,1fr)" }}>
            <StatPill label="STOI" value={fmt(result.metrics.stoi, 3)} />
            <StatPill label="PESQ" value={fmt(result.metrics.pesq, 2)} />
            <StatPill label="RTF" value={fmt(result.metrics.rtf, 3)} />
          </div>
          {result.channels[0] && (
            <div style={{ height: 70, marginTop: 12 }}>
              <WaveChart data={result.channels[0].waveform} color={METHOD_COLORS.beamforming} height={70} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Separation Stage Panel ───────────────────────────────────────────────────

function SeparationPanel({
  inputId,
}: {
  inputId: string | null;
}) {
  const [tau, setTau] = useState(0.90);
  const [freqDomain, setFreqDomain] = useState(true);
  const [useWiener, setUseWiener] = useState(true);
  const [runNeural, setRunNeural] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);

  async function runCompare() {
    if (!inputId) return;
    setLoading(true); setError("");
    try {
      const fd = new FormData();
      fd.append("input_id", inputId);
      fd.append("run_beamforming_flag", "true");
      fd.append("run_svd_flag", "true");
      fd.append("run_neural_flag", String(runNeural));
      fd.append("svd_tau", String(tau));
      fd.append("svd_frequency_domain", String(freqDomain));
      fd.append("svd_use_wiener", String(useWiener));
      const data = await post<CompareResponse>("/api/separation/compare", fd);
      setCompareResult(data);
    } catch (e) { setError(e instanceof Error ? e.message : "Comparison failed"); }
    finally { setLoading(false); }
  }

  return (
    <div>
      <p className="section-title">SVD hyperparameters</p>

      <div className="form-row">
        <label className="form-label">Energy threshold τ — {tau.toFixed(2)}</label>
        <div className="slider-row">
          <input type="range" min={0.80} max={0.99} step={0.01} value={tau}
            onChange={e => setTau(Number(e.target.value))} />
          <span className="slider-val">{tau.toFixed(2)}</span>
        </div>
      </div>

      <div style={{ marginBottom: 16 }}>
        <div className="toggle-row">
          <span className="toggle-label">Frequency-domain SVD (STFT-based)</span>
          <label className="toggle">
            <input type="checkbox" checked={freqDomain} onChange={e => setFreqDomain(e.target.checked)} />
            <span className="toggle-track" /><span className="toggle-thumb" />
          </label>
        </div>
        <div className="toggle-row">
          <span className="toggle-label">Wiener post-filter</span>
          <label className="toggle">
            <input type="checkbox" checked={useWiener} onChange={e => setUseWiener(e.target.checked)} />
            <span className="toggle-track" /><span className="toggle-thumb" />
          </label>
        </div>
        <div className="toggle-row">
          <span className="toggle-label">Include Conv-TasNet (requires asteroid)</span>
          <label className="toggle">
            <input type="checkbox" checked={runNeural} onChange={e => setRunNeural(e.target.checked)} />
            <span className="toggle-track" /><span className="toggle-thumb" />
          </label>
        </div>
      </div>

      <button className="btn-primary" onClick={runCompare} disabled={loading || !inputId}>
        {loading ? "Running comparison…" : "Compare All Methods →"}
      </button>

      {loading && <div className="loading-bar" />}
      {error && <div className="error-msg">{error}</div>}
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [activeStage, setActiveStage] = useState<PipelineStage>("input");
  const [completedStages, setCompletedStages] = useState<Set<PipelineStage>>(new Set());
  const [inputId, setInputId] = useState<string | null>(null);

  // Input mode
  const [mode, setMode] = useState<Mode>("upload");
  const [fastMode, setFastMode] = useState(true);
  const [normalize, setNormalize] = useState(true);
  const [noiseLevel, setNoiseLevel] = useState(0);

  const [audioFiles, setAudioFiles] = useState<FileList | null>(null);
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [autoSelectN, setAutoSelectN] = useState(2);

  // Live recording
  const [numMics, setNumMics] = useState(2);
  const [perMicDelay, setPerMicDelay] = useState(3);
  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const [liveWaveform, setLiveWaveform] = useState<number[]>(new Array(256).fill(0));

  // Results
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<InputResponse | null>(null);

  // Stage results
  const [beamformingResult, setBeamformingResult] = useState<MethodResult | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    };
  }, []);

  function markComplete(stage: PipelineStage) {
    setCompletedStages(prev => new Set([...prev, stage]));
  }

  // ── Input API ──────────────────────────────────────────────────────────────

  function onInputSuccess(data: InputResponse) {
    setResult(data);
    setInputId(data.input_id);
    markComplete("input");
  }

  async function handleUploadAudio() {
    if (!audioFiles?.length) return setError("Select one or more audio files first.");
    setLoading(true); setError("");
    try {
      const fd = new FormData();
      Array.from(audioFiles).forEach(f => fd.append("files", f));
      fd.append("fast_mode", String(fastMode));
      fd.append("normalize", String(normalize));
      fd.append("noise_level", String(noiseLevel));
      onInputSuccess(await post<InputResponse>("/api/input/upload-audio", fd));
    } catch (e) { setError(e instanceof Error ? e.message : "Upload failed."); }
    finally { setLoading(false); }
  }

  async function handleUploadZip() {
    if (!zipFile) return setError("Select a ZIP file first.");
    setLoading(true); setError("");
    try {
      const fd = new FormData();
      fd.append("zip_file", zipFile);
      fd.append("auto_select_n", String(autoSelectN));
      fd.append("fast_mode", String(fastMode));
      fd.append("normalize", String(normalize));
      fd.append("noise_level", String(noiseLevel));
      onInputSuccess(await post<InputResponse>("/api/input/upload-zip", fd));
    } catch (e) { setError(e instanceof Error ? e.message : "ZIP processing failed."); }
    finally { setLoading(false); }
  }

  async function handleTestSignal() {
    setLoading(true); setError("");
    try {
      const fd = new FormData();
      fd.append("duration_sec", "1.0");
      fd.append("sr", "16000");
      fd.append("delay_samples", "200");
      onInputSuccess(await post<InputResponse>("/api/input/test-signal", fd));
    } catch (e) { setError(e instanceof Error ? e.message : "Test signal failed."); }
    finally { setLoading(false); }
  }

  // ── Live recording ─────────────────────────────────────────────────────────

  function drawLiveWave() {
    if (!analyserRef.current) return;
    const buf = new Uint8Array(analyserRef.current.fftSize);
    analyserRef.current.getByteTimeDomainData(buf);
    setLiveWaveform(Array.from(buf).slice(0, 300).map(v => (v - 128) / 128));
    rafRef.current = requestAnimationFrame(drawLiveWave);
  }

  async function startRecording() {
    setError(""); setRecordedBlob(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const ctx = new AudioContext();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      src.connect(analyser);
      analyserRef.current = analyser;
      drawLiveWave();
      const mime = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "audio/ogg";
      const rec = new MediaRecorder(stream, { mimeType: mime });
      mediaRecorderRef.current = rec;
      chunksRef.current = [];
      rec.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      rec.onstop = () => {
        setRecordedBlob(new Blob(chunksRef.current, { type: mime }));
        setRecording(false);
        streamRef.current?.getTracks().forEach(t => t.stop());
        if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
      };
      rec.start();
      setRecording(true);
    } catch (e) { setError(e instanceof Error ? e.message : "Mic permission denied."); }
  }

  function stopRecording() {
    if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
  }

  async function processRecording() {
    if (!recordedBlob) return setError("No recording found.");
    setLoading(true); setError("");
    try {
      const ext = recordedBlob.type.includes("ogg") ? "ogg" : "webm";
      const fd = new FormData();
      fd.append("audio_file", new File([recordedBlob], `live.${ext}`, { type: recordedBlob.type }));
      fd.append("num_mics", String(numMics));
      fd.append("per_mic_delay_ms", String(perMicDelay));
      fd.append("fast_mode", String(fastMode));
      fd.append("normalize", String(normalize));
      fd.append("noise_level", String(noiseLevel));
      onInputSuccess(await post<InputResponse>("/api/input/live", fd));
    } catch (e) { setError(e instanceof Error ? e.message : "Processing failed."); }
    finally { setLoading(false); }
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  const rightPanelTitle = activeStage === "input" ? "Signal Preview"
    : activeStage === "beamforming" ? "Beamforming Output"
    : activeStage === "separation" ? "Method Comparison"
    : `${STAGES.find(s => s.id === activeStage)?.label} Output`;

  return (
    <>
      <style>{`
        :root {
          --accent-b: #2563EB;
          --accent-g: #059669;
          --accent-p: #7C3AED;
          --accent-o: #D97706;
          --accent-input: #0F766E;
          --radius: 10px;
          --radius-sm: 6px;
          --gap: 20px;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        *, *::before, *::after { box-sizing: border-box; margin: 0; }
        button { font: inherit; cursor: pointer; }
        input, select { font: inherit; }

        .lab-root { min-height: 100vh; background: var(--color-background-tertiary, #f5f5f4); padding: 0 0 64px; }

        .lab-header { background: var(--color-background-primary, #fff); border-bottom: 0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.12)); padding: 20px 32px; display: flex; align-items: center; gap: 16px; }
        .lab-logo { width: 36px; height: 36px; background: linear-gradient(135deg, #0F766E, #2563EB); border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
        .lab-title { font-size: 16px; font-weight: 600; letter-spacing: -0.01em; }
        .lab-subtitle { font-size: 12px; color: var(--color-text-secondary, #6b7280); margin-top: 1px; }
        .header-badge { margin-left: auto; font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 20px; background: var(--color-background-secondary, #f3f4f6); color: var(--color-text-secondary, #6b7280); border: 0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.1)); }

        .pipeline-nav { display: flex; align-items: center; padding: 20px 32px; gap: 0; overflow-x: auto; background: var(--color-background-primary, #fff); border-bottom: 0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.1)); }
        .stage-btn { display: flex; align-items: center; gap: 8px; background: none; border: none; padding: 6px 0; color: var(--color-text-secondary, #6b7280); font-size: 13px; font-weight: 400; white-space: nowrap; transition: color 0.15s; }
        .stage-btn:disabled { cursor: not-allowed; opacity: 0.4; }
        .stage-btn:not(:disabled):hover { color: var(--color-text-primary, #111); }
        .stage-num { width: 24px; height: 24px; border-radius: 50%; border: 1.5px solid currentColor; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 600; flex-shrink: 0; transition: background 0.15s, border-color 0.15s; }
        .stage-active { color: var(--accent-input) !important; }
        .stage-active .stage-num { background: var(--accent-input); border-color: var(--accent-input); color: #fff; }
        .stage-done { color: var(--color-text-primary, #111); }
        .stage-done .stage-num { background: #10B981; border-color: #10B981; color: #fff; font-size: 12px; }
        .stage-line { display: block; width: 32px; height: 1px; background: var(--color-border-tertiary, rgba(0,0,0,0.12)); flex-shrink: 0; margin: 0 8px; }

        .lab-body { max-width: 1100px; margin: 0 auto; padding: 28px 32px; display: grid; grid-template-columns: 340px 1fr; gap: var(--gap); align-items: start; }
        @media (max-width: 820px) { .lab-body { grid-template-columns: 1fr; padding: 20px 16px; } }

        .card { background: var(--color-background-primary, #fff); border: 0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.1)); border-radius: var(--radius); overflow: hidden; }
        .card-header { padding: 16px 20px 14px; border-bottom: 0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.08)); display: flex; align-items: center; gap: 8px; }
        .card-title { font-size: 13px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: var(--color-text-secondary, #6b7280); }
        .card-body { padding: 20px; }

        .mode-tabs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 20px; }
        .mode-tab { padding: 8px 4px; font-size: 12px; font-weight: 500; border-radius: var(--radius-sm); border: 0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.12)); background: transparent; color: var(--color-text-secondary, #6b7280); transition: all 0.15s; }
        .mode-tab:hover { background: var(--color-background-secondary, #f9f9f8); }
        .mode-tab.active { background: var(--accent-input); color: #fff; border-color: var(--accent-input); }

        .form-row { margin-bottom: 16px; }
        .form-label { display: block; font-size: 12px; font-weight: 500; color: var(--color-text-secondary, #6b7280); margin-bottom: 6px; }
        .form-input { width: 100%; padding: 8px 10px; font-size: 13px; border: 0.5px solid var(--color-border-secondary, rgba(0,0,0,0.2)); border-radius: var(--radius-sm); background: var(--color-background-primary, #fff); color: var(--color-text-primary, #111); transition: border-color 0.15s; }
        .form-input:focus { outline: none; border-color: var(--accent-input); }

        .toggle-row { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.06)); }
        .toggle-row:last-of-type { border-bottom: none; }
        .toggle-label { font-size: 13px; }
        .toggle { position: relative; width: 36px; height: 20px; flex-shrink: 0; }
        .toggle input { opacity: 0; width: 100%; height: 100%; position: absolute; margin: 0; cursor: pointer; z-index: 1; }
        .toggle-track { position: absolute; inset: 0; border-radius: 10px; background: var(--color-border-secondary, rgba(0,0,0,0.2)); transition: background 0.2s; }
        .toggle input:checked ~ .toggle-track { background: var(--accent-input); }
        .toggle-thumb { position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.2); transition: transform 0.2s; pointer-events: none; }
        .toggle input:checked ~ .toggle-thumb { transform: translateX(16px); }

        .slider-row { display: flex; align-items: center; gap: 10px; }
        .slider-row input[type=range] { flex: 1; }
        .slider-val { font-size: 12px; font-weight: 500; min-width: 36px; text-align: right; color: var(--color-text-primary, #111); }

        .file-zone { border: 1.5px dashed var(--color-border-secondary, rgba(0,0,0,0.2)); border-radius: var(--radius-sm); padding: 20px; text-align: center; font-size: 12px; color: var(--color-text-secondary, #6b7280); cursor: pointer; transition: border-color 0.15s, background 0.15s; position: relative; }
        .file-zone:hover { border-color: var(--accent-input); background: rgba(15,118,110,0.03); }
        .file-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
        .file-zone-icon { font-size: 20px; margin-bottom: 6px; display: block; }
        .file-names { font-size: 11px; color: var(--accent-input); margin-top: 6px; font-weight: 500; }

        .btn-primary { width: 100%; padding: 10px; border-radius: var(--radius-sm); border: none; background: var(--accent-input); color: #fff; font-size: 13px; font-weight: 600; margin-top: 16px; transition: opacity 0.15s, transform 0.1s; }
        .btn-primary:hover:not(:disabled) { opacity: 0.9; }
        .btn-primary:active:not(:disabled) { transform: scale(0.99); }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-ghost { width: 100%; padding: 9px; border-radius: var(--radius-sm); border: 0.5px solid var(--color-border-secondary, rgba(0,0,0,0.2)); background: transparent; color: var(--color-text-secondary, #6b7280); font-size: 12px; font-weight: 500; margin-top: 8px; transition: background 0.15s, color 0.15s; }
        .btn-ghost:hover:not(:disabled) { background: var(--color-background-secondary, #f9f9f8); color: var(--color-text-primary, #111); }
        .btn-ghost:disabled { opacity: 0.4; cursor: not-allowed; }

        .mic-area { text-align: center; padding: 8px 0; }
        .mic-btn { width: 60px; height: 60px; border-radius: 50%; border: 2px solid var(--color-border-secondary, rgba(0,0,0,0.15)); background: var(--color-background-secondary, #f9f9f8); color: var(--color-text-primary, #111); display: inline-flex; align-items: center; justify-content: center; transition: all 0.2s; margin: 8px auto; }
        .mic-btn:hover { border-color: var(--accent-input); color: var(--accent-input); }
        .mic-btn.recording { background: #FEF2F2; border-color: #EF4444; color: #EF4444; animation: pulse 1.5s ease-in-out infinite; }
        @keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.2); } 50% { box-shadow: 0 0 0 10px rgba(239,68,68,0); } }
        .mic-status { font-size: 12px; color: var(--color-text-secondary, #6b7280); margin-top: 4px; }
        .mic-status.live { color: #EF4444; font-weight: 500; }

        .error-msg { font-size: 12px; color: var(--color-text-danger, #dc2626); background: var(--color-background-danger, #fef2f2); border: 0.5px solid var(--color-border-danger, rgba(220,38,38,0.2)); border-radius: var(--radius-sm); padding: 8px 12px; margin-top: 12px; }
        .loading-bar { height: 2px; background: linear-gradient(90deg, transparent, var(--accent-input), transparent); background-size: 200% 100%; animation: shimmer 1.2s linear infinite; border-radius: 1px; margin-top: 12px; }
        @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

        .preview-empty { padding: 60px 20px; text-align: center; }
        .preview-empty-icon { width: 48px; height: 48px; border-radius: 50%; background: var(--color-background-secondary, #f3f4f6); display: flex; align-items: center; justify-content: center; margin: 0 auto 12px; color: var(--color-text-secondary, #9ca3af); }
        .preview-empty p { font-size: 13px; color: var(--color-text-secondary, #9ca3af); }

        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 20px; }
        .stat-pill { background: var(--color-background-secondary, #f9f9f8); border: 0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.08)); border-radius: var(--radius-sm); padding: 10px 12px; }
        .stat-label { font-size: 10px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--color-text-secondary, #9ca3af); display: block; margin-bottom: 3px; }
        .stat-value { font-size: 15px; font-weight: 600; color: var(--color-text-primary, #111); display: block; }

        .signal-section { margin-bottom: 20px; }
        .signal-label { font-size: 11px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--color-text-secondary, #9ca3af); margin-bottom: 8px; }
        .wave-wrap { height: 80px; }
        .spectrogram { width: 100%; border-radius: var(--radius-sm); margin-top: 16px; display: block; }

        .input-id-row { display: flex; align-items: center; gap: 8px; margin-top: 16px; padding-top: 16px; border-top: 0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.08)); }
        .input-id-badge { flex: 1; font-size: 11px; font-family: 'SF Mono', 'Fira Code', monospace; background: var(--color-background-secondary, #f3f4f6); border: 0.5px solid var(--color-border-tertiary, rgba(0,0,0,0.08)); border-radius: 4px; padding: 6px 10px; color: var(--color-text-secondary, #6b7280); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .ready-badge { font-size: 11px; font-weight: 600; padding: 5px 10px; border-radius: 20px; background: #ECFDF5; color: #065F46; border: 0.5px solid #A7F3D0; white-space: nowrap; }

        .channel-files { margin-top: 12px; font-size: 11px; color: var(--color-text-secondary, #9ca3af); }
        .channel-file { display: inline-block; background: var(--color-background-secondary, #f3f4f6); border-radius: 4px; padding: 2px 8px; margin: 2px 3px 2px 0; font-family: 'SF Mono', 'Fira Code', monospace; }

        .next-cta { display: flex; align-items: center; gap: 8px; padding: 12px 16px; background: linear-gradient(90deg, rgba(15,118,110,0.06), rgba(37,99,235,0.04)); border: 0.5px solid rgba(15,118,110,0.2); border-radius: var(--radius-sm); margin-top: 16px; cursor: pointer; transition: background 0.15s; }
        .next-cta:hover { background: linear-gradient(90deg, rgba(15,118,110,0.1), rgba(37,99,235,0.07)); }
        .next-cta-text { flex: 1; font-size: 12px; color: var(--color-text-secondary, #6b7280); }
        .next-cta-text strong { display: block; font-size: 13px; color: var(--accent-input); font-weight: 600; margin-bottom: 1px; }
        .next-cta-arrow { color: var(--accent-input); font-size: 16px; }

        .divider { height: 0.5px; background: var(--color-border-tertiary, rgba(0,0,0,0.08)); margin: 20px 0; }
        .section-title { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--color-text-secondary, #9ca3af); margin-bottom: 12px; }
      `}</style>

      <div className="lab-root">
        <header className="lab-header">
          <div className="lab-logo">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="1.5" strokeLinecap="round">
              <path d="M3 12h4l3-7 4 14 3-7h4" />
            </svg>
          </div>
          <div>
            <div className="lab-title">Speech Separation Lab</div>
            <div className="lab-subtitle">Rank-Adaptive SVD vs Beamforming vs Neural</div>
          </div>
          <span className="header-badge">Research Preview</span>
        </header>

        <PipelineNav active={activeStage} completed={completedStages} onChange={setActiveStage} />

        <div className="lab-body">

          {/* ── Left: Controls ── */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">
                {activeStage === "input" ? "Input Configuration"
                  : activeStage === "beamforming" ? "Beamforming Config"
                  : activeStage === "separation" ? "Separation & Comparison"
                  : STAGES.find(s => s.id === activeStage)?.label}
              </span>
            </div>

            <div className="card-body">
              {/* ── Input stage ── */}
              {activeStage === "input" && (
                <>
                  <div className="mode-tabs">
                    {(["upload", "dataset", "live"] as Mode[]).map(m => (
                      <button key={m} className={`mode-tab ${mode === m ? "active" : ""}`} onClick={() => setMode(m)}>
                        {m === "upload" ? "Audio Files" : m === "dataset" ? "ZIP Dataset" : "Live Mic"}
                      </button>
                    ))}
                  </div>

                  <p className="section-title">Processing options</p>
                  <div style={{ marginBottom: 16 }}>
                    <div className="toggle-row">
                      <span className="toggle-label">Fast resampling</span>
                      <label className="toggle">
                        <input type="checkbox" checked={fastMode} onChange={e => setFastMode(e.target.checked)} />
                        <span className="toggle-track" /><span className="toggle-thumb" />
                      </label>
                    </div>
                    <div className="toggle-row">
                      <span className="toggle-label">Normalize amplitude</span>
                      <label className="toggle">
                        <input type="checkbox" checked={normalize} onChange={e => setNormalize(e.target.checked)} />
                        <span className="toggle-track" /><span className="toggle-thumb" />
                      </label>
                    </div>
                  </div>

                  <div className="form-row">
                    <label className="form-label">Noise augmentation — {noiseLevel.toFixed(3)}</label>
                    <div className="slider-row">
                      <input type="range" min={0} max={0.1} step={0.005} value={noiseLevel}
                        onChange={e => setNoiseLevel(Number(e.target.value))} />
                      <span className="slider-val">{noiseLevel.toFixed(3)}</span>
                    </div>
                  </div>

                  <div className="divider" />

                  {mode === "upload" && (
                    <>
                      <p className="section-title">Audio files</p>
                      <label className="file-zone">
                        <span className="file-zone-icon">🎵</span>
                        <span>Drag or click to select .wav, .mp3, .ogg, .webm</span>
                        <input type="file" accept=".wav,.mp3,.ogg,.webm" multiple
                          onChange={e => setAudioFiles(e.target.files)} />
                        {audioFiles && <div className="file-names">{Array.from(audioFiles).map(f => f.name).join(", ")}</div>}
                      </label>
                      <button className="btn-primary" onClick={handleUploadAudio} disabled={loading || !audioFiles?.length}>
                        {loading ? "Processing…" : "Process Audio →"}
                      </button>
                    </>
                  )}

                  {mode === "dataset" && (
                    <>
                      <p className="section-title">ZIP dataset</p>
                      <label className="file-zone">
                        <span className="file-zone-icon">🗜</span>
                        <span>Drop a ZIP archive containing audio files</span>
                        <input type="file" accept=".zip" onChange={e => setZipFile(e.target.files?.[0] ?? null)} />
                        {zipFile && <div className="file-names">{zipFile.name}</div>}
                      </label>
                      <div className="form-row" style={{ marginTop: 14 }}>
                        <label className="form-label">Auto-select first {autoSelectN} files</label>
                        <div className="slider-row">
                          <input type="range" min={1} max={8} step={1} value={autoSelectN}
                            onChange={e => setAutoSelectN(Number(e.target.value))} />
                          <span className="slider-val">{autoSelectN}</span>
                        </div>
                      </div>
                      <button className="btn-primary" onClick={handleUploadZip} disabled={loading || !zipFile}>
                        {loading ? "Processing…" : "Process Dataset →"}
                      </button>
                    </>
                  )}

                  {mode === "live" && (
                    <>
                      <p className="section-title">Microphone capture</p>
                      <div className="mic-area">
                        <button className={`mic-btn ${recording ? "recording" : ""}`}
                          onClick={recording ? stopRecording : startRecording} disabled={loading}>
                          {recording ? <StopIcon /> : <MicIcon />}
                        </button>
                        <p className={`mic-status ${recording ? "live" : ""}`}>
                          {recording ? "Recording — tap to stop" : recordedBlob ? "Recording captured" : "Tap to start recording"}
                        </p>
                      </div>
                      <div style={{ height: 60, marginBottom: 12 }}>
                        <WaveChart data={liveWaveform} color="#EF4444" height={60} />
                      </div>
                      <div className="form-row">
                        <label className="form-label">Simulated microphones — {numMics}</label>
                        <div className="slider-row">
                          <input type="range" min={1} max={6} step={1} value={numMics}
                            onChange={e => setNumMics(Number(e.target.value))} />
                          <span className="slider-val">{numMics}</span>
                        </div>
                      </div>
                      <div className="form-row">
                        <label className="form-label">Inter-mic delay — {perMicDelay.toFixed(1)} ms</label>
                        <div className="slider-row">
                          <input type="range" min={0} max={20} step={0.5} value={perMicDelay}
                            onChange={e => setPerMicDelay(Number(e.target.value))} />
                          <span className="slider-val">{perMicDelay.toFixed(1)}</span>
                        </div>
                      </div>
                      {recordedBlob && (
                        <>
                          <audio controls src={URL.createObjectURL(recordedBlob)}
                            style={{ width: "100%", marginBottom: 8, borderRadius: 6 }} />
                          <button className="btn-primary" onClick={processRecording} disabled={loading}>
                            {loading ? "Processing…" : "Process Recording →"}
                          </button>
                        </>
                      )}
                    </>
                  )}

                  <button className="btn-ghost" onClick={handleTestSignal} disabled={loading}>
                    Generate synthetic test signal
                  </button>

                  {loading && <div className="loading-bar" />}
                  {error && <div className="error-msg">{error}</div>}
                </>
              )}

              {/* ── Beamforming stage ── */}
              {activeStage === "beamforming" && (
                <BeamformingPanel
                  inputId={inputId}
                  onDone={r => {
                    setBeamformingResult(r);
                    markComplete("beamforming");
                  }}
                />
              )}

              {/* ── Separation stage ── */}
              {activeStage === "separation" && (
                <SeparationPanel inputId={inputId} />
              )}

              {/* ── Other placeholder stages ── */}
              {(activeStage === "diarization" || activeStage === "enhancement") && (
                <div style={{ padding: "32px 0", textAlign: "center", color: "var(--color-text-secondary, #6b7280)", fontSize: 13 }}>
                  <p>This stage is not yet implemented.</p>
                  <p style={{ marginTop: 8, fontSize: 11 }}>
                    {inputId ? `Input ready: ${inputId.slice(0, 8)}…` : "Complete the Input stage first."}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* ── Right: Output ── */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">{rightPanelTitle}</span>
              {result && activeStage === "input" && (
                <span className="ready-badge" style={{ marginLeft: "auto" }}>✓ Ready</span>
              )}
              {beamformingResult && activeStage === "beamforming" && (
                <span className="ready-badge" style={{ marginLeft: "auto" }}>✓ Done</span>
              )}
            </div>

            <div className="card-body">
              {/* Input preview */}
              {activeStage === "input" && (
                !result ? (
                  <div className="preview-empty">
                    <div className="preview-empty-icon">
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                        <path d="M3 12h4l3-7 4 14 3-7h4" />
                      </svg>
                    </div>
                    <p>No audio processed yet.<br />Configure the input and run the pipeline.</p>
                  </div>
                ) : (
                  <>
                    <div className="stats-grid">
                      <StatPill label="Sample rate" value={`${result.sample_rate / 1000} kHz`} />
                      <StatPill label="Channels" value={String(result.channels)} />
                      <StatPill label="Duration" value={`${result.duration_sec.toFixed(2)} s`} />
                      <StatPill label="RMS" value={fmt(result.rms_db, 1, " dB")} />
                      <StatPill label="Peak" value={fmt(result.peak_db, 1, " dB")} />
                      <StatPill label="SNR" value={fmt(result.snr_db, 1, " dB")} />
                    </div>

                    <div className="signal-section">
                      <p className="signal-label">Waveform — channel 0</p>
                      <div className="wave-wrap">
                        <WaveChart data={result.waveform} color="#0F766E" />
                      </div>
                    </div>

                    <div className="signal-section">
                      <p className="signal-label">Log-frequency spectrogram</p>
                      <img className="spectrogram"
                        src={`data:image/png;base64,${result.spectrogram_png_base64}`}
                        alt="Spectrogram" />
                    </div>

                    {result.channel_files && result.channel_files.length > 0 && (
                      <div className="channel-files">
                        {result.channel_files.map(f => <span key={f} className="channel-file">{f}</span>)}
                      </div>
                    )}

                    <div className="input-id-row">
                      <span className="input-id-badge">{result.input_id}</span>
                      <span style={{ fontSize: 11, color: "var(--color-text-secondary)", flexShrink: 0 }}>{result.source}</span>
                    </div>

                    <button className="next-cta" style={{ marginTop: 16, width: "100%", border: "none", textAlign: "left" }}
                      onClick={() => setActiveStage("beamforming")}>
                      <div className="next-cta-text">
                        <strong>Continue to Beamforming</strong>
                        Spatially filter the {result.channels}-channel signal
                      </div>
                      <span className="next-cta-arrow">→</span>
                    </button>
                  </>
                )
              )}

              {/* Beamforming output */}
              {activeStage === "beamforming" && (
                !beamformingResult ? (
                  <div className="preview-empty">
                    <div className="preview-empty-icon"><BeamIcon /></div>
                    <p>{inputId ? "Configure and run beamforming." : "Complete the Input stage first."}</p>
                  </div>
                ) : (
                  <>
                    <div className="stats-grid">
                      <StatPill label="STOI" value={fmt(beamformingResult.metrics.stoi, 3)} />
                      <StatPill label="PESQ" value={fmt(beamformingResult.metrics.pesq, 2)} />
                      <StatPill label="RTF" value={fmt(beamformingResult.metrics.rtf, 3)} />
                    </div>

                    {beamformingResult.channels[0] && (
                      <div className="signal-section">
                        <p className="signal-label">Beamformed output</p>
                        <div className="wave-wrap">
                          <WaveChart data={beamformingResult.channels[0].waveform} color={METHOD_COLORS.beamforming} />
                        </div>
                        {beamformingResult.channels[0].wav_b64 && (
                          <audio controls
                            src={`data:audio/wav;base64,${beamformingResult.channels[0].wav_b64}`}
                            style={{ width: "100%", marginTop: 10, borderRadius: 6 }} />
                        )}
                      </div>
                    )}

                    <details style={{ marginTop: 12 }}>
                      <summary style={{ fontSize: 11, cursor: "pointer", color: "var(--color-text-secondary, #6b7280)" }}>
                        Delay estimates (GCC-PHAT)
                      </summary>
                      <pre style={{ fontSize: 10, marginTop: 6, padding: 10, background: "var(--color-background-secondary, #f3f4f6)", borderRadius: 6, overflow: "auto" }}>
                        {JSON.stringify(beamformingResult.metadata, null, 2)}
                      </pre>
                    </details>

                    <button className="next-cta" style={{ marginTop: 16, width: "100%", border: "none", textAlign: "left" }}
                      onClick={() => { markComplete("beamforming"); setActiveStage("separation"); }}>
                      <div className="next-cta-text">
                        <strong>Continue to Method Comparison</strong>
                        Compare SVD, Beamforming, and Neural side-by-side
                      </div>
                      <span className="next-cta-arrow">→</span>
                    </button>
                  </>
                )
              )}

              {/* Separation comparison output */}
              {activeStage === "separation" && (
                <SeparationOutputPanel inputId={inputId} />
              )}

              {/* Placeholder for other stages */}
              {(activeStage === "diarization" || activeStage === "enhancement") && (
                <div className="preview-empty">
                  <div className="preview-empty-icon">
                    {activeStage === "diarization" ? <DiarIcon /> : <EnhIcon />}
                  </div>
                  <p>{inputId ? `Run ${STAGES.find(s => s.id === activeStage)?.label} to see output here.` : "Complete the Input stage first."}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// ─── Separation Output Panel (stateful, wired to /compare endpoint) ──────────

function SeparationOutputPanel({ inputId }: { inputId: string | null }) {
  const [tau, setTau] = useState(0.90);
  const [freqDomain, setFreqDomain] = useState(true);
  const [useWiener, setUseWiener] = useState(true);
  const [runNeural, setRunNeural] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);

  async function runCompare() {
    if (!inputId) return;
    setLoading(true); setError("");
    try {
      const fd = new FormData();
      fd.append("input_id", inputId);
      fd.append("run_beamforming_flag", "true");
      fd.append("run_svd_flag", "true");
      fd.append("run_neural_flag", String(runNeural));
      fd.append("svd_tau", String(tau));
      fd.append("svd_frequency_domain", String(freqDomain));
      fd.append("svd_use_wiener", String(useWiener));
      const data = await post<CompareResponse>("/api/separation/compare", fd);
      setCompareResult(data);
    } catch (e) { setError(e instanceof Error ? e.message : "Comparison failed"); }
    finally { setLoading(false); }
  }

  if (!inputId) {
    return (
      <div className="preview-empty">
        <div className="preview-empty-icon"><SepIcon /></div>
        <p>Complete the Input stage first.</p>
      </div>
    );
  }

  return (
    <div>
      {/* Config controls at top of right panel */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16, alignItems: "center" }}>
        <div style={{ flex: "1 1 140px" }}>
          <label style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary, #6b7280)", display: "block", marginBottom: 4 }}>
            τ — {tau.toFixed(2)}
          </label>
          <input type="range" min={0.80} max={0.99} step={0.01} value={tau}
            onChange={e => setTau(Number(e.target.value))} style={{ width: "100%" }} />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {[
            { label: "Freq-domain", val: freqDomain, set: setFreqDomain },
            { label: "Wiener", val: useWiener, set: setUseWiener },
            { label: "Neural", val: runNeural, set: setRunNeural },
          ].map(({ label, val, set }) => (
            <label key={label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, cursor: "pointer" }}>
              <input type="checkbox" checked={val} onChange={e => set(e.target.checked)} />
              {label}
            </label>
          ))}
        </div>

        <button
          onClick={runCompare}
          disabled={loading}
          style={{
            padding: "8px 16px", borderRadius: 6, border: "none",
            background: "var(--accent-p)", color: "#fff",
            fontSize: 12, fontWeight: 600, cursor: "pointer",
            opacity: loading ? 0.6 : 1, whiteSpace: "nowrap",
          }}
        >
          {loading ? "Running…" : "▶ Compare"}
        </button>
      </div>

      {loading && <div className="loading-bar" />}
      {error && <div className="error-msg">{error}</div>}

      <ComparisonPanel data={compareResult} inputId={inputId} />
    </div>
  );
}