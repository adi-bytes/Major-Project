import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

interface IVPlotProps {
  V: number[];
  I_mean: number[];
  I_lower?: number[];
  I_upper?: number[];
  I_phys?: number[];
  inputV?: number[];
  inputI?: number[];
  title?: string;
  height?: number;
}

export function IVPlot({
  V,
  I_mean,
  I_lower,
  I_upper,
  I_phys,
  inputV,
  inputI,
  title = "I–V Characteristic",
  height = 400,
}: IVPlotProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    const traces: Plotly.Data[] = [];

    // CI band (upper)
    if (I_upper && I_lower) {
      traces.push({
        x: [...V, ...V.slice().reverse()],
        y: [...I_upper, ...I_lower.slice().reverse()],
        fill: "toself",
        fillcolor: "rgba(14,165,233,0.12)",
        line: { color: "transparent" },
        name: "95% CI",
        showlegend: true,
        type: "scatter",
        hoverinfo: "skip",
      });
    }

    // Physics baseline
    if (I_phys) {
      traces.push({
        x: V,
        y: I_phys,
        mode: "lines",
        line: { color: "#f97316", width: 1.5, dash: "dash" },
        name: "Physics-only",
        type: "scatter",
      });
    }

    // Ensemble mean
    traces.push({
      x: V,
      y: I_mean,
      mode: "lines",
      line: { color: "#38bdf8", width: 2.5 },
      name: "Hybrid surrogate (mean)",
      type: "scatter",
    });

    // Input curve (reverse mode)
    if (inputV && inputI) {
      traces.push({
        x: inputV,
        y: inputI,
        mode: "markers+lines",
        marker: { color: "#a3e635", size: 5 },
        line: { color: "#a3e635", width: 1.5, dash: "dot" },
        name: "Input curve",
        type: "scatter",
      });
    }

    const layout: Partial<Plotly.Layout> = {
      title: { text: title, font: { color: "#e2e8f0", size: 14 } },
      paper_bgcolor: "rgba(17,24,39,0)",
      plot_bgcolor: "rgba(17,24,39,0.6)",
      height,
      margin: { t: 40, b: 60, l: 70, r: 20 },
      xaxis: {
        title: { text: "Voltage (V)", font: { color: "#94a3b8" } },
        tickfont: { color: "#94a3b8" },
        gridcolor: "#1e293b",
        linecolor: "#334155",
        range: [0, 4],
      },
      yaxis: {
        title: { text: "Current (A)", font: { color: "#94a3b8" } },
        tickfont: { color: "#94a3b8" },
        type: "log" as const,
        gridcolor: "#1e293b",
        linecolor: "#334155",
        exponentformat: "power",
      },
      legend: {
        font: { color: "#cbd5e1", size: 11 },
        bgcolor: "rgba(15,23,42,0.6)",
        bordercolor: "#334155",
        borderwidth: 1,
      },
    };

    Plotly.react(ref.current, traces, layout, {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
    });
  }, [V, I_mean, I_lower, I_upper, I_phys, inputV, inputI, title, height]);

  return <div ref={ref} className="w-full rounded-xl overflow-hidden" />;
}
