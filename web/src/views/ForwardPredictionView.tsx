import { useState } from "react";
import { usePrediction } from "@/api/client";
import { useAppStore } from "@/store/appStore";
import { IVPlot } from "@/components/IVPlot";
import { LoadingSpinner, ErrorBox, WarnBox } from "@/components/ui";

const WF_MIN = 4.99;
const WF_MAX = 5.19;
const T_MIN = 300;
const T_MAX = 500;

function fmtSci(v: number, digits = 3) {
  if (!isFinite(v) || v === 0) return "—";
  return v.toExponential(digits);
}

function fmtFixed(v: number, digits = 2) {
  if (!isFinite(v)) return "—";
  return v.toFixed(digits);
}

export function ForwardPredictionView() {
  const { wf, temperature, setWf, setTemperature } = useAppStore();
  const [committed, setCommitted] = useState({ wf, temperature });

  const { data, isLoading, error, refetch } = usePrediction(committed);

  const isOutOfBounds =
    wf < WF_MIN || wf > WF_MAX || temperature < T_MIN || temperature > T_MAX;

  function handleRun() {
    if (!isOutOfBounds) setCommitted({ wf, temperature });
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Forward Prediction</h2>
        <p className="text-gray-400 text-sm">
          Predict the I–V characteristic for a given (WF, T) using the M=5 physics-guided
          ensemble. All values from live API — no hard-coded metrics.
        </p>
      </div>

      {/* Controls */}
      <div className="card space-y-5">
        <div>
          <div className="flex justify-between mb-1">
            <label className="text-sm font-medium text-gray-300">
              Work Function (WF)
            </label>
            <span className="text-sm font-mono text-brand-300">
              {wf.toFixed(3)} eV
            </span>
          </div>
          <input
            type="range"
            min={WF_MIN}
            max={WF_MAX}
            step={0.001}
            value={wf}
            onChange={(e) => setWf(parseFloat(e.target.value))}
            className="input-range"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-0.5">
            <span>{WF_MIN} eV</span>
            <span>{WF_MAX} eV</span>
          </div>
        </div>

        <div>
          <div className="flex justify-between mb-1">
            <label className="text-sm font-medium text-gray-300">
              Temperature (T)
            </label>
            <span className="text-sm font-mono text-brand-300">{temperature} K</span>
          </div>
          <input
            type="range"
            min={T_MIN}
            max={T_MAX}
            step={1}
            value={temperature}
            onChange={(e) => setTemperature(parseInt(e.target.value))}
            className="input-range"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-0.5">
            <span>{T_MIN} K</span>
            <span>{T_MAX} K</span>
          </div>
        </div>

        {isOutOfBounds && (
          <div className="text-xs text-red-400 bg-red-950/40 border border-red-800 rounded-lg px-3 py-2">
            ⛔ Selected values are outside the validated design space. Prediction
            disabled until bounds are satisfied.
          </div>
        )}

        <button
          className="btn-primary w-full"
          disabled={isOutOfBounds}
          onClick={handleRun}
        >
          Run Prediction
        </button>
      </div>

      {/* Results */}
      {isLoading && <LoadingSpinner text="Running surrogate inference…" />}
      {error && (
        <ErrorBox message={error.message} retry={() => refetch()} />
      )}

      {data && !isLoading && (
        <>
          <WarnBox items={data.flags} />

          <div className="card">
            <IVPlot
              V={data.V}
              I_mean={data.I_mean}
              I_lower={data.I_lower}
              I_upper={data.I_upper}
              I_phys={data.I_phys}
              title={`I–V: WF=${committed.wf.toFixed(3)} eV, T=${committed.temperature} K`}
            />
          </div>

          {/* FOMs */}
          <div className="card">
            <h3 className="font-semibold text-brand-300 mb-3">
              Figures of Merit
              <span className="ml-2 text-xs font-normal text-gray-500">
                from live API · evidence: {data.evidence_level}
              </span>
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {[
                { label: "V₀ (turn-on)", value: fmtFixed(data.metrics.v_o), unit: "V" },
                {
                  label: "Ron·A (on-resistance)",
                  value: fmtFixed(data.metrics.r_on_sp),
                  unit: "mΩ·cm²",
                },
                {
                  label: "SS",
                  value: fmtFixed(data.metrics.ss_mV_dec),
                  unit: "mV/dec",
                },
                {
                  label: "I_on @ 4V",
                  value: fmtSci(data.metrics.i_on_4V),
                  unit: "A",
                },
                {
                  label: "I_leak @ 0V",
                  value: fmtSci(data.metrics.i_leak_0V),
                  unit: "A",
                },
              ].map((f) => (
                <div
                  key={f.label}
                  className="bg-gray-800/50 rounded-xl p-3 text-center"
                >
                  <p className="text-xs text-gray-400 mb-1">{f.label}</p>
                  <p className="font-mono text-lg font-bold text-white">{f.value}</p>
                  <p className="text-xs text-gray-500">{f.unit}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
