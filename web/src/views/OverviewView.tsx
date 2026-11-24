import { useMeta, useHealth } from "@/api/client";
import { LoadingSpinner, ErrorBox } from "@/components/ui";

export function OverviewView() {
  const { data: meta, isLoading, error } = useMeta();
  const { data: health } = useHealth();

  if (isLoading) return <LoadingSpinner text="Loading device metadata…" />;
  if (error) return <ErrorBox message={error.message} />;

  return (
    <div className="space-y-6">
      {/* Title */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">
          Overview &amp; Method
        </h2>
        <p className="text-gray-400 text-sm">
          Physics-guided surrogate modeling of vertical β-Ga₂O₃ Schottky barrier diodes
        </p>
      </div>

      {/* Evidence tier */}
      <div className="card flex flex-wrap gap-3 items-center">
        <span className="badge badge-green text-sm py-1 px-3">
          ✓ TCAD Validated
        </span>
        <span className="badge badge-blue text-sm py-1 px-3">
          Tier A — Reverse Mode Active
        </span>
        <span className="badge badge-purple text-sm py-1 px-3">
          Tier B — Design Workbench Active
        </span>
        {health && (
          <span className="text-xs text-gray-500 ml-auto">
            Evidence level: {health.evidence_level}
          </span>
        )}
      </div>

      {/* Major Project Context & Multi-Material Vision Callout */}
      <div className="card border-brand-500/30 bg-brand-950/20">
        <div className="flex items-start gap-3">
          <div className="text-brand-400 text-xl font-bold">💡</div>
          <div>
            <h3 className="font-semibold text-brand-200 mb-1">
              Major Project Scope: Adaptive Multi-Material Framework
            </h3>
            <p className="text-gray-300 text-sm leading-relaxed mb-2">
              This interactive suite demonstrates the core foundational engine of our Major Project:{" "}
              <strong>
                &ldquo;PHYSICS-GUIDED ML SURROGATE AND INVERSE DESIGN FRAMEWORK FOR POWER SEMICONDUCTOR DEVICES&rdquo;
              </strong>.
            </p>
            <p className="text-gray-400 text-xs leading-relaxed">
              While currently demonstrated on vertical β-Ga₂O₃ Schottky Barrier Diodes (SBD) as the inaugural
              ultra-wide bandgap test vehicle, this framework is engineered as an <strong>adaptive, plug-and-play
              architecture</strong>. Ongoing development extends this modular pipeline to additional wide-bandgap (WBG)
              and ultra-wide bandgap (UWBG) materials (including 4H-SiC, GaN, and Diamond) and complex topologies
              (HEMTs, MPS/JBS diodes, and MOSFETs) to serve application-specific power electronic requirements.
            </p>
          </div>
        </div>
      </div>

      {/* Device summary */}
      <div className="card">
        <h3 className="font-semibold text-brand-300 mb-2">Initial Reference Test Vehicle</h3>
        <p className="text-gray-200 text-sm mb-3">{meta?.device_name}</p>
        <p className="text-gray-400 text-sm leading-relaxed">{meta?.description}</p>
      </div>

      {/* Method */}
      <div className="card">
        <h3 className="font-semibold text-brand-300 mb-3">Methodology & System Architecture</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <p className="font-medium text-gray-100 mb-1">Differentiable Physics Core</p>
            <p className="text-gray-400">
              Lambert-W thermionic emission + drift transport. Calibrated on TCAD training
              split with multi-start bounded least squares (Huber loss on log₁₀ I).
            </p>
          </div>
          <div>
            <p className="font-medium text-gray-100 mb-1">Hybrid Physics-Residual Surrogate</p>
            <p className="text-gray-400">
              log₁₀ I<sub>phys</sub> + 0.7·tanh(g/0.7) residual neural network (2→128→128→52 SiLU).
              M=5 bootstrap ensemble providing Bayesian-like uncertainty quantification.
            </p>
          </div>
          <div>
            <p className="font-medium text-gray-100 mb-1">Multi-Start Inverse Parameter Recovery</p>
            <p className="text-gray-400">
              Multi-start L-BFGS-B optimization recovering physical parameters (WF, T)
              from arbitrary experimental or simulated I–V curves on any voltage grid.
            </p>
          </div>
          <div>
            <p className="font-medium text-gray-100 mb-1">Thermodynamic Constraints & Derated Breakdown</p>
            <p className="text-gray-400">
              Physical monotonicity loss (∂I/∂V &gt; 0, ∂Vo/∂WF &gt; 0, ∂Vo/∂T &lt; 0), smoothness penalty,
              and 1-D Poisson breakdown with realistic edge-termination derating (η<sub>term</sub> = 0.5).
            </p>
          </div>
        </div>
      </div>

      {/* Design variables */}
      <div className="card">
        <h3 className="font-semibold text-brand-300 mb-3">Design Variable Bounds</h3>
        {meta && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {meta.variables.map((v) => (
              <div key={v.name} className="bg-gray-800/60 rounded-lg p-3">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-mono text-brand-200 font-semibold">{v.name}</span>
                  <span className="text-xs text-gray-400">[{v.unit}]</span>
                </div>
                <div className="text-sm text-gray-300">
                  {v.min_val} → {v.max_val}
                </div>
                {v.is_sweepable && (
                  <span className="text-xs text-emerald-400">Sweepable</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Fixed parameters */}
      <div className="card">
        <h3 className="font-semibold text-brand-300 mb-3">Fixed Device Parameters</h3>
        {meta && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="table-header text-left py-2">Parameter</th>
                  <th className="table-header text-right py-2">Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {Object.entries(meta.fixed_parameters).map(([k, val]) => (
                  <tr key={k}>
                    <td className="py-2 font-mono text-gray-300">{k}</td>
                    <td className="py-2 text-right text-gray-400">
                      {typeof val === "number" ? val.toExponential(3) : String(val)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Citation */}
      <div className="card border-brand-800/50 bg-brand-950/20">
        <h3 className="font-semibold text-brand-300 mb-2">Data Source</h3>
        <p className="text-sm text-gray-400">
          S. Dhillon et al., "Impact of Work Function and Temperature on Vertical
          β-Ga₂O₃ Schottky Barrier Diodes: A TCAD Study," <em>IEEE Transactions on
          Electron Devices</em>, vol. 68, no. 11, 2021.
        </p>
      </div>
    </div>
  );
}
