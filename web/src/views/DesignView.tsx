import React, { useState } from "react";
import {
  useDesignOptimization,
  useDesignPresets,
  useExportReport,
  useExportTCAD,
} from "@/api/client";
import type { DesignCandidate, DesignRequest, DesignPreset } from "@/api/types";
import { IVPlot } from "@/components/IVPlot";
import { ParametricCrossSection } from "@/components/ParametricCrossSection";
import { ParetoPlot } from "@/components/ParetoPlot";
import { ErrorBox, LoadingSpinner } from "@/components/ui";

export function DesignView() {
  const presetsQuery = useDesignPresets();
  const designMutation = useDesignOptimization();
  const exportTCADMutation = useExportTCAD();
  const exportReportMutation = useExportReport();

  // Form State
  const [targetVbr, setTargetVbr] = useState<number>(1200);
  const [targetRon, setTargetRon] = useState<number>(10.0);
  const [targetVo, setTargetVo] = useState<number>(0.85);
  const [targetSS, setTargetSS] = useState<number>(90.0);
  const [targetIon, setTargetIon] = useState<number>(0.1);
  const [tolerance, setTolerance] = useState<number>(0.10);
  const [method, setMethod] = useState<"nsga2" | "gradient">("nsga2");

  // Selection State
  const [selectedIdxA, setSelectedIdxA] = useState<number>(0);
  const [selectedIdxB, setSelectedIdxB] = useState<number | null>(null);
  const [exportedTCAD, setExportedTCAD] = useState<{
    sdevice: string;
    sde: string;
    readme: string;
  } | null>(null);
  const [exportedReport, setExportedReport] = useState<string | null>(null);

  // Apply a preset
  function applyPreset(preset: DesignPreset) {
    const r = preset.requirements;
    if (r.target_v_br) setTargetVbr(r.target_v_br);
    if (r.target_r_on_sp) setTargetRon(r.target_r_on_sp);
    if (r.target_v_o) setTargetVo(r.target_v_o);
    if (r.target_ss) setTargetSS(r.target_ss);
    if (r.target_i_on_4v) setTargetIon(r.target_i_on_4v);
    if (r.tolerance) setTolerance(r.tolerance);
  }

  // Run Optimization
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const payload: DesignRequest = {
      target_v_br: targetVbr,
      target_r_on_sp: targetRon,
      target_v_o: targetVo,
      target_ss: targetSS,
      target_i_on_4v: targetIon,
      tolerance: tolerance,
      method: method,
      fixed_temperature: 300.0,
    };
    designMutation.mutate(payload, {
      onSuccess: () => {
        setSelectedIdxA(0);
        setSelectedIdxB(null);
        setExportedTCAD(null);
        setExportedReport(null);
      },
    });
  }

  const result = designMutation.data;
  const candidates = result?.candidates || [];
  const paretoPoints = result?.pareto_front || [];

  const candidateA = candidates[selectedIdxA] || null;
  const candidateB = selectedIdxB !== null ? candidates[selectedIdxB] : null;

  // Handle Export TCAD
  function handleExportTCAD(candidate: DesignCandidate) {
    exportTCADMutation.mutate(
      { parameters: candidate.parameters },
      {
        onSuccess: (data) => {
          setExportedTCAD({
            sdevice: data.sdevice_cmd,
            sde: data.sde_cmd,
            readme: data.readme_md,
          });
        },
      }
    );
  }

  // Handle Export Report
  function handleExportReport() {
    if (!result) return;
    exportReportMutation.mutate(
      {
        candidates: candidates,
        requirements: result.requirements,
      },
      {
        onSuccess: (data) => {
          setExportedReport(data.markdown);
        },
      }
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 flex-wrap mb-1">
          <h2 className="text-2xl font-bold text-white">Design Workbench</h2>
          <span className="badge badge-yellow text-xs py-0.5 px-2">
            Tier B — physics_extrapolated
          </span>
          <span className="badge badge-blue text-xs py-0.5 px-2">
            NSGA-II + 256-Sobol
          </span>
        </div>
        <p className="text-gray-400 text-sm">
          Multi-objective device synthesis for vertical β-Ga₂O₃ Schottky Barrier
          Diodes. Synthesizes drift thickness ($t_D$), doping ($N_{"{net}"}$), anode
          diameter ($d$), and work function ($WF$).
        </p>
      </div>

      {/* Pre-verified Presets */}
      <div className="card">
        <h3 className="font-semibold text-brand-300 text-sm mb-2">
          ⚡ Pre-Verified Feasible Device Presets
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {presetsQuery.data?.presets.map((p) => (
            <button
              key={p.id}
              onClick={() => applyPreset(p)}
              className="p-3 text-left bg-gray-800/60 hover:bg-gray-800 rounded-xl border border-gray-700/60 transition-colors flex flex-col justify-between"
            >
              <div>
                <p className="font-semibold text-gray-200 text-sm">{p.name}</p>
                <p className="text-xs text-gray-400 mt-1 line-clamp-2">
                  {p.description}
                </p>
              </div>
              <span className="text-[11px] text-brand-400 mt-2 font-mono">
                V_br ≥ {p.requirements.target_v_br}V | Ron·A ≤{" "}
                {p.requirements.target_r_on_sp} mΩ·cm²
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Requirements Input Form */}
      <form onSubmit={handleSubmit} className="card space-y-4">
        <div className="flex justify-between items-center border-b border-gray-800 pb-2">
          <h3 className="font-semibold text-brand-300 text-sm">
            Target Electrical Requirements &amp; Tolerances
          </h3>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-gray-400">Solver:</span>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value as "nsga2" | "gradient")}
              className="bg-gray-800 border border-gray-700 rounded-lg px-2.5 py-1 text-gray-200 outline-none"
            >
              <option value="nsga2">NSGA-II (Multi-Objective Evolutionary)</option>
              <option value="gradient">256-Start Sobol + Adam Gradient</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Min Breakdown V_br (V)
            </label>
            <input
              type="number"
              min="100"
              max="5000"
              step="50"
              value={targetVbr}
              onChange={(e) => setTargetVbr(parseFloat(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-100 font-mono outline-none focus:border-brand-500"
            />
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Max Ron·A (mΩ·cm²)
            </label>
            <input
              type="number"
              min="0.5"
              max="100"
              step="0.5"
              value={targetRon}
              onChange={(e) => setTargetRon(parseFloat(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-100 font-mono outline-none focus:border-brand-500"
            />
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Max Turn-on V_o (V)
            </label>
            <input
              type="number"
              min="0.3"
              max="2.0"
              step="0.05"
              value={targetVo}
              onChange={(e) => setTargetVo(parseFloat(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-100 font-mono outline-none focus:border-brand-500"
            />
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Max SS (mV/dec)
            </label>
            <input
              type="number"
              min="60"
              max="250"
              step="5"
              value={targetSS}
              onChange={(e) => setTargetSS(parseFloat(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-100 font-mono outline-none focus:border-brand-500"
            />
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Min I_on @ 4V (A)
            </label>
            <input
              type="number"
              min="0.01"
              max="5.0"
              step="0.01"
              value={targetIon}
              onChange={(e) => setTargetIon(parseFloat(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-100 font-mono outline-none focus:border-brand-500"
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Acceptance Tolerance Band</span>
            <span className="font-mono text-brand-300">
              ±{(tolerance * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="0.05"
            max="0.25"
            step="0.01"
            value={tolerance}
            onChange={(e) => setTolerance(parseFloat(e.target.value))}
            className="input-range"
          />
        </div>

        <button
          type="submit"
          disabled={designMutation.isPending}
          className="btn-primary w-full"
        >
          {designMutation.isPending
            ? "Synthesizing Device Designs..."
            : "Run Inverse Synthesis"}
        </button>
      </form>

      {/* Loading & Error States */}
      {designMutation.isPending && (
        <LoadingSpinner text="Executing multi-objective synthesis & Pareto clustering (NSGA-II)..." />
      )}
      {designMutation.error && (
        <ErrorBox message={designMutation.error.message} />
      )}

      {/* Results Section */}
      {result && candidates.length > 0 && !designMutation.isPending && (
        <div className="space-y-6">
          {/* Candidate Cards Grid */}
          <div>
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-bold text-lg text-white">
                Synthesized Design Candidates ({candidates.length})
              </h3>
              <button
                onClick={handleExportReport}
                className="btn-secondary text-xs flex items-center gap-1.5"
              >
                📄 Export Synthesis Report
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {candidates.map((cand, idx) => {
                const isSelectedA = idx === selectedIdxA;
                const isSelectedB = idx === selectedIdxB;
                const v = cand.verdict;

                let verdictBadge = "badge-green";
                if (v === "met_within_tolerance") verdictBadge = "badge-blue";
                else if (v === "compromise") verdictBadge = "badge-yellow";
                else if (v === "infeasible") verdictBadge = "badge-red";

                return (
                  <div
                    key={idx}
                    className={`card flex flex-col justify-between border transition-all duration-200 ${
                      isSelectedA
                        ? "border-brand-500 ring-1 ring-brand-500 bg-gray-900"
                        : isSelectedB
                        ? "border-purple-500 ring-1 ring-purple-500 bg-gray-900"
                        : "border-gray-800 hover:border-gray-700"
                    }`}
                  >
                    <div>
                      <div className="flex justify-between items-start gap-1 mb-2">
                        <span className="font-bold text-white text-sm">
                          {cand.archetype}
                        </span>
                        <span className={`badge ${verdictBadge} text-[10px]`}>
                          {v}
                        </span>
                      </div>
                      <p className="text-[11px] text-gray-400 mb-3 leading-tight line-clamp-2">
                        {cand.verdict_reason}
                      </p>

                      {/* Figures of Merit */}
                      <div className="space-y-1.5 text-xs border-t border-b border-gray-800/80 py-2.5 mb-2.5">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Breakdown (V_br):</span>
                          <span className="font-mono text-emerald-400 font-semibold">
                            {cand.metrics.v_br.toFixed(1)} V
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Ron·A:</span>
                          <span className="font-mono text-sky-400 font-semibold">
                            {cand.metrics.r_on_sp.toFixed(2)} mΩ·cm²
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Turn-on (V_o):</span>
                          <span className="font-mono text-white">
                            {cand.metrics.v_o.toFixed(2)} V
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">I_on @ 4V:</span>
                          <span className="font-mono text-white">
                            {cand.metrics.i_on_4V.toFixed(3)} A
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">BFOM:</span>
                          <span className="font-mono text-purple-300">
                            {cand.metrics.bfom_mw_cm2?.toFixed(0)} MW/cm²
                          </span>
                        </div>
                      </div>

                      {/* Synthesized Parameters */}
                      <div className="text-[11px] text-gray-400 space-y-1">
                        <div className="flex justify-between">
                          <span>Drift t_D:</span>
                          <span className="font-mono text-gray-200">
                            {cand.parameters.t_D_um.toFixed(2)} μm
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Doping N_net:</span>
                          <span className="font-mono text-gray-200">
                            {cand.parameters.N_net_cm3.toExponential(2)} cm⁻³
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Anode d:</span>
                          <span className="font-mono text-gray-200">
                            {cand.parameters.d_um.toFixed(1)} μm
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Work Function:</span>
                          <span className="font-mono text-gray-200">
                            {cand.parameters.WF.toFixed(3)} eV
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Card Actions */}
                    <div className="mt-4 pt-2 border-t border-gray-800 flex gap-2">
                      <button
                        onClick={() => setSelectedIdxA(idx)}
                        className={`flex-1 py-1 px-2 rounded text-[11px] font-medium transition-colors ${
                          isSelectedA
                            ? "bg-brand-600 text-white"
                            : "bg-gray-800 text-gray-300 hover:bg-gray-700"
                        }`}
                      >
                        {isSelectedA ? "Active (A)" : "Inspect (A)"}
                      </button>
                      <button
                        onClick={() =>
                          setSelectedIdxB(selectedIdxB === idx ? null : idx)
                        }
                        className={`py-1 px-2 rounded text-[11px] font-medium transition-colors ${
                          isSelectedB
                            ? "bg-purple-600 text-white"
                            : "bg-gray-800 text-purple-300 hover:bg-gray-700"
                        }`}
                      >
                        {isSelectedB ? "Comparing" : "Compare (B)"}
                      </button>
                      <button
                        onClick={() => handleExportTCAD(cand)}
                        title="Export Sentaurus TCAD Deck"
                        className="py-1 px-2 rounded text-[11px] bg-gray-800 text-gray-300 hover:bg-gray-700"
                      >
                        ⚡
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Pareto Trade-Off Plot */}
          {paretoPoints.length > 0 && (
            <div className="card">
              <ParetoPlot
                paretoPoints={paretoPoints}
                candidates={candidates}
                selectedCandidateIndex={selectedIdxA}
                onSelectCandidate={setSelectedIdxA}
              />
            </div>
          )}

          {/* Device Cross-Section & Forward I-V Curves */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Parametric Cross Section */}
            {candidateA && (
              <ParametricCrossSection
                candidateA={candidateA}
                candidateB={candidateB}
              />
            )}

            {/* Forward I-V Curve Overlay */}
            {candidateA && (
              <div className="card flex flex-col justify-between">
                <div>
                  <h3 className="font-semibold text-brand-300 text-sm mb-1">
                    Forward I–V Curve: {candidateA.archetype}
                  </h3>
                  <p className="text-xs text-gray-400 mb-3">
                    Forward simulation with active anode area (
                    {candidateA.parameters.anode_area_cm2?.toExponential(2)} cm²)
                  </p>
                  <IVPlot
                    V={candidateA.iv_curve.V}
                    I_mean={candidateA.iv_curve.I}
                    height={300}
                    title={`${candidateA.archetype} I–V Characteristics`}
                  />
                </div>
                <div className="flex justify-between items-center text-xs text-gray-500 border-t border-gray-800 pt-2 mt-2">
                  <span>
                    Turn-on V_o: {candidateA.metrics.v_o.toFixed(2)} V
                  </span>
                  <span>
                    Forward I @ 4V: {candidateA.metrics.i_on_4V.toFixed(3)} A
                  </span>
                  <span className="badge badge-yellow text-[9px]">
                    physics_extrapolated
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Exported TCAD Deck Modal / Drawer */}
          {exportedTCAD && (
            <div className="card border-brand-800/80 bg-gray-900/90 space-y-3">
              <div className="flex justify-between items-center border-b border-gray-800 pb-2">
                <h3 className="font-bold text-brand-300 text-sm flex items-center gap-2">
                  ⚡ Sentaurus TCAD Simulation Deck (Generated)
                </h3>
                <button
                  onClick={() => setExportedTCAD(null)}
                  className="text-gray-400 hover:text-white text-xs"
                >
                  ✕ Close
                </button>
              </div>

              <div className="text-xs text-yellow-300 bg-yellow-950/40 p-2.5 rounded-lg border border-yellow-800">
                <strong>[UNTESTED DECK NOTICE]</strong> Research prototype generated
                deck. Convergence, meshing, and edge termination must be verified in
                Sentaurus before running.
              </div>

              <div className="space-y-2">
                <p className="text-xs font-semibold text-gray-300">
                  sdevice.cmd (Physics &amp; Solve Ramps):
                </p>
                <pre className="bg-gray-950 p-3 rounded-lg text-xs font-mono text-gray-300 overflow-x-auto max-h-48 border border-gray-800">
                  {exportedTCAD.sdevice}
                </pre>
              </div>

              <div className="space-y-2">
                <p className="text-xs font-semibold text-gray-300">
                  sde.cmd (Geometry &amp; Doping):
                </p>
                <pre className="bg-gray-950 p-3 rounded-lg text-xs font-mono text-gray-300 overflow-x-auto max-h-48 border border-gray-800">
                  {exportedTCAD.sde}
                </pre>
              </div>
            </div>
          )}

          {/* Exported Report Drawer */}
          {exportedReport && (
            <div className="card border-brand-800/80 bg-gray-900/90 space-y-3">
              <div className="flex justify-between items-center border-b border-gray-800 pb-2">
                <h3 className="font-bold text-brand-300 text-sm">
                  📄 Synthesis Report (Markdown)
                </h3>
                <button
                  onClick={() => setExportedReport(null)}
                  className="text-gray-400 hover:text-white text-xs"
                >
                  ✕ Close
                </button>
              </div>
              <pre className="bg-gray-950 p-3 rounded-lg text-xs font-mono text-gray-300 overflow-x-auto max-h-64 border border-gray-800">
                {exportedReport}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
