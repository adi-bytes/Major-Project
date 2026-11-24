// Calibrated physics parameters from artifacts/calib.json
// Values are cited directly — not polled at runtime.
const CALIB_PARAMS = {
  chi_eff: { value: 3.9841, unit: "eV", label: "Effective barrier height χ_eff" },
  ideality_n: { value: 1.0463, unit: "—", label: "Ideality factor n" },
  A_eff_cm2: { value: 1.711e-4, unit: "cm²", label: "Effective junction area A_eff" },
  mu_300: { value: 250.0, unit: "cm²/V·s", label: "Mobility at 300 K μ₃₀₀" },
  alpha: { value: 1.4047, unit: "—", label: "Mobility exponent α" },
  R_c_ohm: { value: 6.5e-11, unit: "Ω", label: "Contact resistance R_c" },
};

const OMITTED = [
  "Thermodynamic lattice self-heating (Joule heating at high forward currents)",
  "Incomplete dopant ionization at low-to-moderate temperatures",
  "PHuMob doping and high-field saturation velocity",
  "Cylindrical 2D edge field enhancement at contact perimeter",
];

export function ModelDetailsView() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Model &amp; Data Details</h2>
        <p className="text-gray-400 text-sm">
          Architecture, calibrated physics, evidence taxonomy, and data provenance.
        </p>
      </div>

      {/* Architecture */}
      <div className="card">
        <h3 className="font-semibold text-brand-300 mb-3">Model Architecture</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2">
            <div className="flex justify-between border-b border-gray-800 py-1.5">
              <span className="text-gray-400">Ensemble size (M)</span>
              <span className="font-mono text-white">5</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 py-1.5">
              <span className="text-gray-400">Architecture</span>
              <span className="font-mono text-white">2→128→128→52 SiLU MLP</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 py-1.5">
              <span className="text-gray-400">Residual cap (c)</span>
              <span className="font-mono text-white">0.7 decades</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 py-1.5">
              <span className="text-gray-400">Output activation</span>
              <span className="font-mono text-white">0.7 · tanh(g/0.7)</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 py-1.5">
              <span className="text-gray-400">Inputs</span>
              <span className="font-mono text-white">WF [eV], T [K]</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-gray-400">Output</span>
              <span className="font-mono text-white">52-point log₁₀ I residual</span>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between border-b border-gray-800 py-1.5">
              <span className="text-gray-400">Training strategy</span>
              <span className="font-mono text-white">Bootstrap ensemble</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 py-1.5">
              <span className="text-gray-400">Loss: data term</span>
              <span className="font-mono text-white">MSE on log₁₀ I</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 py-1.5">
              <span className="text-gray-400">Loss: monotonicity</span>
              <span className="font-mono text-white">L_mono (sign-verified)</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 py-1.5">
              <span className="text-gray-400">Loss: smoothness</span>
              <span className="font-mono text-white">L_smooth (2nd diff.)</span>
            </div>
            <div className="flex justify-between border-b border-gray-800 py-1.5">
              <span className="text-gray-400">Off-grid interpolation</span>
              <span className="font-mono text-white">Cubic spline (PCHIP)</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-gray-400">Uncertainty</span>
              <span className="font-mono text-white">Ensemble std (log space)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Calibrated physics */}
      <div className="card">
        <h3 className="font-semibold text-brand-300 mb-1">
          Calibrated Physics Parameters
        </h3>
        <p className="text-xs text-gray-500 mb-3">
          From <code>artifacts/calib.json</code> · Calibrated on training split only
          using multi-start bounded least squares with Huber loss on log₁₀ I.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="table-header text-left py-2">Parameter</th>
                <th className="table-header text-right py-2">Value</th>
                <th className="table-header text-right py-2">Unit</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {Object.entries(CALIB_PARAMS).map(([key, p]) => (
                <tr key={key}>
                  <td className="py-2 text-gray-300">{p.label}</td>
                  <td className="py-2 text-right font-mono text-white">
                    {typeof p.value === "number" && p.value < 1e-3
                      ? p.value.toExponential(3)
                      : typeof p.value === "number"
                      ? p.value.toFixed(4)
                      : p.value}
                  </td>
                  <td className="py-2 text-right text-gray-500">{p.unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Omitted physics */}
      <div className="card border-yellow-800/40 bg-yellow-950/10">
        <h3 className="font-semibold text-yellow-400 mb-2">
          Physics Absorbed by Residual (not explicitly modeled)
        </h3>
        <ul className="list-disc list-inside space-y-1 text-sm text-yellow-200/70">
          {OMITTED.map((o) => (
            <li key={o}>{o}</li>
          ))}
        </ul>
      </div>

      {/* Evidence taxonomy */}
      <div className="card">
        <h3 className="font-semibold text-brand-300 mb-3">Evidence Taxonomy</h3>
        <div className="space-y-3 text-sm">
          <div className="flex gap-3 items-start">
            <span className="badge badge-green mt-0.5 shrink-0">Tier A</span>
            <div>
              <p className="text-gray-200 font-medium">TCAD Validated</p>
              <p className="text-gray-400">
                Forward prediction and reverse parameter recovery, validated on
                held-out TCAD curves with 0%, 1%, and 5% multiplicative noise.
              </p>
            </div>
          </div>
          <div className="flex gap-3 items-start">
            <span className="badge badge-purple mt-0.5 shrink-0">Tier B</span>
            <div>
              <p className="text-gray-200 font-medium">Physics-Extrapolated Inverse Design</p>
              <p className="text-gray-400">
                Multi-objective optimization (NSGA-II and 256-start Sobol-gradient) across structural
                variables (drift thickness, doping, diameter) with 1-D analytic breakdown estimation.
              </p>
            </div>
          </div>
          <div className="flex gap-3 items-start">
            <span className="badge badge-red mt-0.5 shrink-0">Fabrication</span>
            <div>
              <p className="text-gray-200 font-medium">Not validated</p>
              <p className="text-gray-400">
                No experimental validation against fabricated devices. All results
                are simulation-based. Do not use for device manufacturing decisions.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Data citation */}
      <div className="card">
        <h3 className="font-semibold text-brand-300 mb-2">Data Provenance</h3>
        <p className="text-sm text-gray-400 leading-relaxed">
          <strong className="text-gray-200">Source:</strong> S. Dhillon, D. Bhatt, A. Kumar,
          et al., "Impact of Work Function and Temperature on Vertical β-Ga₂O₃ Schottky
          Barrier Diodes: A Sentaurus TCAD Study," <em>IEEE Transactions on Electron
          Devices</em>, vol. 68, no. 11, pp. 5771–5778, Nov. 2021.
        </p>
        <p className="text-sm text-gray-500 mt-2">
          Simulator: Sentaurus TCAD with cylindrical coordinates, self-heating (lattice
          heat equation), and incomplete ionization (Fermi–Dirac statistics). All I–V
          curves are from simulation — no experimental measurements.
        </p>
        <div className="mt-3 text-xs text-yellow-600 bg-yellow-950/30 border border-yellow-800/40 rounded-lg px-3 py-2">
          ⚠️ This model was trained exclusively on TCAD simulation data. Generalization
          to real fabricated devices has not been tested.
        </div>
      </div>
    </div>
  );
}
