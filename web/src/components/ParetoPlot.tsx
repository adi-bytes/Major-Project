import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import type { DesignCandidate, ParetoPoint } from "@/api/types";

interface ParetoPlotProps {
  paretoPoints: ParetoPoint[];
  candidates: DesignCandidate[];
  selectedCandidateIndex: number;
  onSelectCandidate: (index: number) => void;
  height?: number;
}

export function ParetoPlot({
  paretoPoints,
  candidates,
  selectedCandidateIndex,
  onSelectCandidate,
  height = 360,
}: ParetoPlotProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    // 1. Pareto Front Population Trace
    const paretoX = paretoPoints.map((p) => p.v_br);
    const paretoY = paretoPoints.map((p) => p.r_on_sp);
    const paretoText = paretoPoints.map(
      (p) =>
        `V_br: ${p.v_br.toFixed(1)} V<br>Ron·A: ${p.r_on_sp.toFixed(2)} mΩ·cm²<br>t_D: ${p.t_D_um.toFixed(1)} μm`
    );

    const traces: Plotly.Data[] = [
      {
        x: paretoX,
        y: paretoY,
        mode: "markers",
        type: "scatter",
        name: "Pareto Front",
        marker: {
          color: "rgba(56, 189, 248, 0.4)",
          size: 7,
          symbol: "circle",
        },
        text: paretoText,
        hoverinfo: "text",
      },
    ];

    // 2. Clustered Representative Candidates Trace
    const candX = candidates.map((c) => c.metrics.v_br);
    const candY = candidates.map((c) => c.metrics.r_on_sp);
    const candColors = candidates.map((_c, i) =>
      i === selectedCandidateIndex ? "#f59e0b" : "#38bdf8"
    );
    const candSizes = candidates.map((_c, i) =>
      i === selectedCandidateIndex ? 14 : 10
    );
    const candSymbols = candidates.map((_c, i) =>
      i === selectedCandidateIndex ? "diamond" : "circle"
    );
    const candText = candidates.map(
      (c) =>
        `<b>${c.archetype}</b> [${c.verdict}]<br>V_br: ${c.metrics.v_br.toFixed(1)} V<br>Ron·A: ${c.metrics.r_on_sp.toFixed(2)} mΩ·cm²`
    );

    traces.push({
      x: candX,
      y: candY,
      mode: "markers+text",
      type: "scatter",
      name: "Synthesized Archetypes",
      marker: {
        color: candColors,
        size: candSizes,
        symbol: candSymbols,
        line: { color: "#ffffff", width: 1.5 },
      },
      text: candidates.map((c) => c.archetype),
      textposition: "top center",
      textfont: { color: "#cbd5e1", size: 10 },
      hoverinfo: "text",
      hovertext: candText,
    });

    const layout: Partial<Plotly.Layout> = {
      title: {
        text: "Trade-off Space: Breakdown Voltage vs. Specific On-Resistance",
        font: { color: "#e2e8f0", size: 13 },
      },
      paper_bgcolor: "rgba(17,24,39,0)",
      plot_bgcolor: "rgba(17,24,39,0.6)",
      height,
      margin: { t: 40, b: 50, l: 60, r: 20 },
      xaxis: {
        title: { text: "Breakdown Voltage V_br (V)", font: { color: "#94a3b8" } },
        tickfont: { color: "#94a3b8" },
        gridcolor: "#1e293b",
        linecolor: "#334155",
      },
      yaxis: {
        title: { text: "Ron·A (mΩ·cm²)", font: { color: "#94a3b8" } },
        tickfont: { color: "#94a3b8" },
        type: "log" as const,
        gridcolor: "#1e293b",
        linecolor: "#334155",
        exponentformat: "power",
      },
      legend: {
        font: { color: "#cbd5e1", size: 10 },
        bgcolor: "rgba(15,23,42,0.6)",
        bordercolor: "#334155",
        borderwidth: 1,
      },
    };

    Plotly.react(ref.current, traces, layout, {
      responsive: true,
      displayModeBar: false,
    });

    // Wire click event to select candidate
    const el = ref.current as unknown as {
      on?: (event: string, callback: (eventData: { points?: { pointIndex: number; curveNumber: number }[] }) => void) => void;
    };
    if (el && typeof el.on === "function") {
      el.on("plotly_click", (data) => {
        if (data.points && data.points.length > 0) {
          const pt = data.points[0];
          // If clicked on curve 1 (candidates)
          if (pt.curveNumber === 1 && pt.pointIndex < candidates.length) {
            onSelectCandidate(pt.pointIndex);
          }
        }
      });
    }
  }, [paretoPoints, candidates, selectedCandidateIndex, onSelectCandidate, height]);

  return <div ref={ref} className="w-full rounded-xl overflow-hidden" />;
}
