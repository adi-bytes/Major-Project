import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import { useValidation } from "@/api/client";
import { LoadingSpinner, ErrorBox } from "@/components/ui";

function safeGet(obj: unknown, ...keys: string[]): unknown {
  let cur: unknown = obj;
  for (const k of keys) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[k];
  }
  return cur;
}

function fmt(v: unknown, digits = 4): string {
  if (v == null || v === undefined) return "—";
  if (typeof v === "number") return isFinite(v) ? v.toFixed(digits) : "—";
  return String(v);
}

interface AblationEntry {
  model: string;
  "test_mae"?: number;
  "test_rmse"?: number;
  test_r2?: number;
  [key: string]: unknown;
}

function AblationTable({ ablation }: { ablation: unknown }) {
  if (!ablation || typeof ablation !== "object") return null;
  const rows = Object.entries(ablation as Record<string, unknown>).map(
    ([model, m]) => ({ model, ...(m as Record<string, unknown>) } as AblationEntry)
  );
  if (!rows.length) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="table-header text-left py-2">Model variant</th>
            <th className="table-header text-right py-2">Test MAE (log₁₀)</th>
            <th className="table-header text-right py-2">Test RMSE</th>
            <th className="table-header text-right py-2">R²</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {rows.map((row) => (
            <tr key={row.model}>
              <td className="py-2 font-mono text-gray-200">{row.model}</td>
              <td className="py-2 text-right font-mono text-gray-300">
                {fmt(row["test_mae"] ?? row["mae"])}
              </td>
              <td className="py-2 text-right font-mono text-gray-300">
                {fmt(row["test_rmse"] ?? row["rmse"])}
              </td>
              <td className="py-2 text-right font-mono text-gray-300">
                {fmt(row["test_r2"] ?? row["r2"])}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReverseTable({ rm }: { rm: unknown }) {
  if (!rm || typeof rm !== "object") return null;
  const entries = Object.entries(rm as Record<string, unknown>);
  if (!entries.length) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="table-header text-left py-2">Noise level</th>
            <th className="table-header text-right py-2">WF MAE (eV)</th>
            <th className="table-header text-right py-2">T MAE (K)</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {entries.map(([noise, m]) => (
            <tr key={noise}>
              <td className="py-2 font-mono text-gray-200">{noise}</td>
              <td className="py-2 text-right font-mono text-gray-300">
                {fmt(safeGet(m, "wf_mae"))}
              </td>
              <td className="py-2 text-right font-mono text-gray-300">
                {fmt(safeGet(m, "t_mae"), 2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BarChart({
  labels,
  values,
  title,
  color = "#38bdf8",
}: {
  labels: string[];
  values: number[];
  title: string;
  color?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    Plotly.react(
      ref.current,
      [
        {
          type: "bar",
          x: labels,
          y: values,
          marker: { color },
          text: values.map((v) => v.toFixed(4)),
          textposition: "outside",
        } as Plotly.Data,
      ],
      {
        title: { text: title, font: { color: "#e2e8f0", size: 13 } },
        paper_bgcolor: "rgba(17,24,39,0)",
        plot_bgcolor: "rgba(17,24,39,0.6)",
        height: 260,
        margin: { t: 40, b: 60, l: 60, r: 20 },
        xaxis: { tickfont: { color: "#94a3b8" }, linecolor: "#334155" },
        yaxis: {
          title: { text: "MAE (log₁₀ I)", font: { color: "#94a3b8" } },
          tickfont: { color: "#94a3b8" },
          gridcolor: "#1e293b",
          linecolor: "#334155",
        },
      },
      { responsive: true, displayModeBar: false }
    );
  }, [labels, values, title, color]);
  return <div ref={ref} className="w-full" />;
}

export function ValidationDashboardView() {
  const { data, isLoading, error, refetch } = useValidation();

  if (isLoading)
    return <LoadingSpinner text="Loading validation metrics from API…" />;
  if (error) return <ErrorBox message={error.message} retry={() => refetch()} />;

  const metrics = data?.metrics ?? {};
  const test = safeGet(metrics, "test") as Record<string, unknown> | undefined;
  const extrap = safeGet(metrics, "extrapolation") as Record<string, unknown> | undefined;
  const ablation = safeGet(metrics, "ablation");
  const rm = safeGet(metrics, "reverse_mode");

  // Build bar chart for test MAE across ablation configs
  const ablationEntries = ablation && typeof ablation === "object"
    ? Object.entries(ablation as Record<string, unknown>)
    : [];
  const barLabels = ablationEntries.map(([k]) => k);
  const barValues = ablationEntries.map(([, v]) => {
    const mae = safeGet(v, "test_mae") ?? safeGet(v, "mae");
    return typeof mae === "number" ? mae : 0;
  });

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Validation Dashboard</h2>
        <p className="text-gray-400 text-sm">
          All metrics loaded exclusively from{" "}
          <code className="text-brand-300">GET /api/validation</code> — no
          hard-coded numbers.
        </p>
      </div>

      <div className="badge badge-green py-1 px-3 text-sm">
        Evidence: {data?.evidence_level ?? "—"}
      </div>

      {/* Test split */}
      {test ? (
        <div className="card">
          <h3 className="font-semibold text-brand-300 mb-3">Test Split Metrics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {(["mae", "rmse", "r2", "max_error"] as const).map((k) => {
              const val = safeGet(test, k);
              return (
                <div key={k} className="bg-gray-800/50 rounded-xl p-3 text-center">
                  <p className="text-xs text-gray-400 mb-1 uppercase">{k}</p>
                  <p className="font-mono text-lg font-bold text-white">
                    {fmt(val)}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}

      {/* Extrapolation */}
      {extrap ? (
        <div className="card">
          <h3 className="font-semibold text-brand-300 mb-3">Extrapolation Block</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {(["mae", "rmse", "r2", "max_error"] as const).map((k) => {
              const val = safeGet(extrap, k);
              return (
                <div key={k} className="bg-gray-800/50 rounded-xl p-3 text-center">
                  <p className="text-xs text-gray-400 mb-1 uppercase">{k}</p>
                  <p className="font-mono text-lg font-bold text-white">
                    {fmt(val)}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}

      {/* Ablation bar chart */}
      {ablationEntries.length > 0 && (
        <div className="card">
          <h3 className="font-semibold text-brand-300 mb-3">Ablation Study</h3>
          {barValues.some((v) => v > 0) && (
            <BarChart
              labels={barLabels}
              values={barValues}
              title="Test MAE by Model Variant"
            />
          )}
          <div className="mt-3">
            <AblationTable ablation={ablation} />
          </div>
        </div>
      )}

      {/* Reverse mode noise benchmark */}
      {rm ? (
        <div className="card">
          <h3 className="font-semibold text-brand-300 mb-3">
            Reverse Mode Noise Benchmark
          </h3>
          <ReverseTable rm={rm} />
          <p className="text-xs text-gray-500 mt-2">
            Multiplicative Gaussian noise applied to I–V before recovery. MAE
            computed on held-out TCAD curves.
          </p>
        </div>
      ) : null}

      {/* Raw JSON toggle for transparency */}
      <details className="card cursor-pointer">
        <summary className="text-sm text-gray-400 font-medium select-none">
          Show raw metrics JSON (for transparency)
        </summary>
        <pre className="mt-3 text-xs font-mono text-gray-400 overflow-x-auto max-h-96">
          {JSON.stringify(metrics, null, 2)}
        </pre>
      </details>
    </div>
  );
}
