import { useState } from "react";
import type { DesignCandidate } from "@/api/types";

interface CrossSectionProps {
  candidateA: DesignCandidate;
  candidateB?: DesignCandidate | null;
  className?: string;
}

export function ParametricCrossSection({
  candidateA,
  candidateB,
  className = "",
}: CrossSectionProps) {
  const [toScale, setToScale] = useState<boolean>(false);
  const [compareMode, setCompareMode] = useState<boolean>(false);

  // Parameter extraction for Candidate A
  const pA = candidateA.parameters;
  const wfA = pA.WF;
  const tdA = pA.t_D_um; // 3 to 15 um
  const nnA = pA.N_net_cm3; // 1e15 to 5e16
  const dA = pA.d_um; // 50 to 500 um

  // Parameter extraction for Candidate B (if comparing)
  const pB = candidateB?.parameters;
  const wfB = pB ? pB.WF : wfA;
  const tdB = pB ? pB.t_D_um : tdA;
  const nnB = pB ? pB.N_net_cm3 : nnA;
  const dB = pB ? pB.d_um : dA;

  // Geometry computation for Candidate A
  // Schematic dimensions (clamped for visual aesthetics)
  const driftHeightA = toScale
    ? Math.max(30, Math.min(260, tdA * 16))
    : 70 + ((tdA - 3.0) / 12.0) * 90; // 70px to 160px
  const anodeWidthA = toScale
    ? Math.max(60, Math.min(280, (dA / 500.0) * 280))
    : 100 + ((dA - 50.0) / 450.0) * 160; // 100px to 260px

  const substrateHeight = toScale ? 80 : 50;
  const chipWidth = 320;
  const anodeLeftA = (chipWidth - anodeWidthA) / 2;

  // Geometry computation for Candidate B
  const driftHeightB = toScale
    ? Math.max(30, Math.min(260, tdB * 16))
    : 70 + ((tdB - 3.0) / 12.0) * 90;
  const anodeWidthB = toScale
    ? Math.max(60, Math.min(280, (dB / 500.0) * 280))
    : 100 + ((dB - 50.0) / 450.0) * 160;
  const anodeLeftB = (chipWidth - anodeWidthB) / 2;

  // Color shading based on doping N_net (log-scale hue from light cyan to deep blue)
  const dopingHueA = Math.min(1.0, Math.max(0.0, (Math.log10(nnA) - 15.0) / 1.7));
  const driftColorA = `rgb(${Math.round(20 + 30 * (1 - dopingHueA))}, ${Math.round(
    60 + 50 * dopingHueA
  )}, ${Math.round(110 + 80 * dopingHueA)})`;

  const dopingHueB = Math.min(1.0, Math.max(0.0, (Math.log10(nnB) - 15.0) / 1.7));
  const driftColorB = `rgb(${Math.round(20 + 30 * (1 - dopingHueB))}, ${Math.round(
    60 + 50 * dopingHueB
  )}, ${Math.round(110 + 80 * dopingHueB)})`;

  return (
    <div className={`card ${className}`}>
      {/* Controls Bar */}
      <div className="flex justify-between items-center flex-wrap gap-2 mb-4 border-b border-gray-800 pb-3">
        <div>
          <h3 className="font-semibold text-brand-300 flex items-center gap-2">
            Parametric Device Cross-Section
            <span className="badge badge-yellow text-[10px]">
              physics_extrapolated
            </span>
          </h3>
          <p className="text-xs text-gray-500">
            Interactive vertical SBD geometry with 300 ms transitions
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          {/* Schematic vs To-Scale Toggle */}
          <button
            onClick={() => setToScale(!toScale)}
            className={`px-3 py-1.5 rounded-lg border transition-all duration-300 font-medium ${
              toScale
                ? "bg-brand-600 border-brand-400 text-white"
                : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
            }`}
          >
            {toScale ? "📏 Mode: To-Scale (μm)" : "📐 Mode: Schematic View"}
          </button>

          {/* Compare Mode Toggle */}
          {candidateB && (
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={`px-3 py-1.5 rounded-lg border transition-all duration-300 font-medium ${
                compareMode
                  ? "bg-purple-600 border-purple-400 text-white"
                  : "bg-gray-800 border-gray-700 text-purple-300 hover:bg-gray-700"
              }`}
            >
              {compareMode ? "⚖️ Comparing 2 Candidates" : "⚖️ Enable Compare Mode"}
            </button>
          )}
        </div>
      </div>

      {/* SVG Canvas Container */}
      <div className="flex justify-center items-center py-2 overflow-x-auto">
        <div className="flex gap-6 items-center flex-wrap justify-center">
          {/* Device Diagram A */}
          <div className="flex flex-col items-center">
            <span className="text-xs font-mono text-brand-300 mb-1 font-semibold">
              {candidateA.archetype} (Candidate A)
            </span>

            <svg
              width={chipWidth}
              height={anodeWidthA ? driftHeightA + substrateHeight + 70 : 250}
              viewBox={`0 0 ${chipWidth} ${driftHeightA + substrateHeight + 70}`}
              className="rounded-lg border border-gray-800 bg-gray-950/80 shadow-inner"
            >
              <defs>
                <pattern
                  id="metalHatch"
                  width="8"
                  height="8"
                  patternTransform="rotate(45 0 0)"
                  patternUnits="userSpaceOnUse"
                >
                  <line
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="8"
                    stroke="#94a3b8"
                    strokeWidth="1.5"
                  />
                </pattern>
              </defs>

              {/* 1. Anode Top Metal (Schottky) */}
              <rect
                x={anodeLeftA}
                y={15}
                width={anodeWidthA}
                height={20}
                fill="url(#metalHatch)"
                stroke="#cbd5e1"
                strokeWidth="1.5"
                rx="2"
                style={{ transition: "all 300ms ease-in-out" }}
              />
              <text
                x={chipWidth / 2}
                y={29}
                textAnchor="middle"
                fill="#f8fafc"
                fontSize="10"
                fontFamily="monospace"
                fontWeight="bold"
                style={{ transition: "all 300ms ease-in-out" }}
              >
                Anode (WF={wfA.toFixed(3)} eV)
              </text>

              {/* 2. Drift Layer (beta-Ga2O3) */}
              <rect
                x={10}
                y={35}
                width={chipWidth - 20}
                height={driftHeightA}
                fill={driftColorA}
                stroke="#0284c7"
                strokeWidth="1.5"
                style={{ transition: "all 300ms ease-in-out" }}
              />
              <text
                x={chipWidth / 2}
                y={35 + driftHeightA / 2 - 6}
                textAnchor="middle"
                fill="#e0f2fe"
                fontSize="11"
                fontWeight="bold"
                fontFamily="sans-serif"
                style={{ transition: "all 300ms ease-in-out" }}
              >
                n⁻ Drift β-Ga₂O₃ Layer
              </text>
              <text
                x={chipWidth / 2}
                y={35 + driftHeightA / 2 + 10}
                textAnchor="middle"
                fill="#bae6fd"
                fontSize="10"
                fontFamily="monospace"
                style={{ transition: "all 300ms ease-in-out" }}
              >
                t_D = {tdA.toFixed(2)} μm | N_net = {nnA.toExponential(2)} cm⁻³
              </text>

              {/* 3. Substrate Layer (n+ beta-Ga2O3) */}
              <rect
                x={10}
                y={35 + driftHeightA}
                width={chipWidth - 20}
                height={substrateHeight}
                fill="#1e293b"
                stroke="#334155"
                strokeWidth="1.5"
                style={{ transition: "all 300ms ease-in-out" }}
              />
              <text
                x={chipWidth / 2}
                y={35 + driftHeightA + substrateHeight / 2}
                textAnchor="middle"
                fill="#94a3b8"
                fontSize="10"
                fontFamily="sans-serif"
                style={{ transition: "all 300ms ease-in-out" }}
              >
                n⁺ Substrate (10 μm, 4×10¹⁸ cm⁻³)
              </text>

              {/* 4. Cathode Bottom Metal (Ohmic) */}
              <rect
                x={10}
                y={35 + driftHeightA + substrateHeight}
                width={chipWidth - 20}
                height={15}
                fill="url(#metalHatch)"
                stroke="#64748b"
                strokeWidth="1.5"
                rx="1"
                style={{ transition: "all 300ms ease-in-out" }}
              />
              <text
                x={chipWidth / 2}
                y={46 + driftHeightA + substrateHeight}
                textAnchor="middle"
                fill="#cbd5e1"
                fontSize="9"
                fontFamily="sans-serif"
                style={{ transition: "all 300ms ease-in-out" }}
              >
                Cathode Ohmic Contact
              </text>
            </svg>
          </div>

          {/* Compare Device Diagram B (if enabled) */}
          {compareMode && candidateB && (
            <div className="flex flex-col items-center">
              <span className="text-xs font-mono text-purple-300 mb-1 font-semibold">
                {candidateB.archetype} (Candidate B)
              </span>

              <svg
                width={chipWidth}
                height={driftHeightB + substrateHeight + 70}
                viewBox={`0 0 ${chipWidth} ${driftHeightB + substrateHeight + 70}`}
                className="rounded-lg border border-purple-900 bg-gray-950/80 shadow-inner"
              >
                <rect
                  x={anodeLeftB}
                  y={15}
                  width={anodeWidthB}
                  height={20}
                  fill="url(#metalHatch)"
                  stroke="#c084fc"
                  strokeWidth="1.5"
                  rx="2"
                  style={{ transition: "all 300ms ease-in-out" }}
                />
                <text
                  x={chipWidth / 2}
                  y={29}
                  textAnchor="middle"
                  fill="#f8fafc"
                  fontSize="10"
                  fontFamily="monospace"
                  fontWeight="bold"
                  style={{ transition: "all 300ms ease-in-out" }}
                >
                  Anode (WF={wfB.toFixed(3)} eV)
                </text>

                <rect
                  x={10}
                  y={35}
                  width={chipWidth - 20}
                  height={driftHeightB}
                  fill={driftColorB}
                  stroke="#a855f7"
                  strokeWidth="1.5"
                  style={{ transition: "all 300ms ease-in-out" }}
                />
                <text
                  x={chipWidth / 2}
                  y={35 + driftHeightB / 2 - 6}
                  textAnchor="middle"
                  fill="#f3e8ff"
                  fontSize="11"
                  fontWeight="bold"
                  fontFamily="sans-serif"
                  style={{ transition: "all 300ms ease-in-out" }}
                >
                  n⁻ Drift β-Ga₂O₃ Layer
                </text>
                <text
                  x={chipWidth / 2}
                  y={35 + driftHeightB / 2 + 10}
                  textAnchor="middle"
                  fill="#e9d5ff"
                  fontSize="10"
                  fontFamily="monospace"
                  style={{ transition: "all 300ms ease-in-out" }}
                >
                  t_D = {tdB.toFixed(2)} μm | N_net = {nnB.toExponential(2)} cm⁻³
                </text>

                <rect
                  x={10}
                  y={35 + driftHeightB}
                  width={chipWidth - 20}
                  height={substrateHeight}
                  fill="#1e293b"
                  stroke="#334155"
                  strokeWidth="1.5"
                  style={{ transition: "all 300ms ease-in-out" }}
                />
                <text
                  x={chipWidth / 2}
                  y={35 + driftHeightB + substrateHeight / 2}
                  textAnchor="middle"
                  fill="#94a3b8"
                  fontSize="10"
                  fontFamily="sans-serif"
                  style={{ transition: "all 300ms ease-in-out" }}
                >
                  n⁺ Substrate (10 μm, 4×10¹⁸ cm⁻³)
                </text>

                <rect
                  x={10}
                  y={35 + driftHeightB + substrateHeight}
                  width={chipWidth - 20}
                  height={15}
                  fill="url(#metalHatch)"
                  stroke="#64748b"
                  strokeWidth="1.5"
                  rx="1"
                  style={{ transition: "all 300ms ease-in-out" }}
                />
                <text
                  x={chipWidth / 2}
                  y={46 + driftHeightB + substrateHeight}
                  textAnchor="middle"
                  fill="#cbd5e1"
                  fontSize="9"
                  fontFamily="sans-serif"
                  style={{ transition: "all 300ms ease-in-out" }}
                >
                  Cathode Ohmic Contact
                </text>
              </svg>
            </div>
          )}
        </div>
      </div>

      {/* Extrapolated Dimension Callouts with Permanent Badges */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-2 border-t border-gray-800 pt-3 text-xs">
        <div className="bg-gray-800/40 p-2.5 rounded-lg flex flex-col justify-between">
          <div className="flex justify-between items-center mb-1">
            <span className="text-gray-400">Drift Thickness (t_D)</span>
            <span className="badge badge-yellow text-[9px]">physics_extrapolated</span>
          </div>
          <span className="font-mono text-white font-bold">
            {tdA.toFixed(2)} μm
            {compareMode && candidateB && (
              <span className="text-purple-400 ml-2 font-normal">
                vs {tdB.toFixed(2)} μm (Δ={(tdA - tdB).toFixed(2)})
              </span>
            )}
          </span>
        </div>

        <div className="bg-gray-800/40 p-2.5 rounded-lg flex flex-col justify-between">
          <div className="flex justify-between items-center mb-1">
            <span className="text-gray-400">Doping Concentration (N_net)</span>
            <span className="badge badge-yellow text-[9px]">physics_extrapolated</span>
          </div>
          <span className="font-mono text-white font-bold">
            {nnA.toExponential(2)} cm⁻³
            {compareMode && candidateB && (
              <span className="text-purple-400 ml-2 font-normal">
                vs {nnB.toExponential(2)}
              </span>
            )}
          </span>
        </div>

        <div className="bg-gray-800/40 p-2.5 rounded-lg flex flex-col justify-between">
          <div className="flex justify-between items-center mb-1">
            <span className="text-gray-400">Anode Diameter (d)</span>
            <span className="badge badge-yellow text-[9px]">physics_extrapolated</span>
          </div>
          <span className="font-mono text-white font-bold">
            {dA.toFixed(1)} μm
            {compareMode && candidateB && (
              <span className="text-purple-400 ml-2 font-normal">
                vs {dB.toFixed(1)} μm (Δ={(dA - dB).toFixed(1)})
              </span>
            )}
          </span>
        </div>
      </div>
    </div>
  );
}
