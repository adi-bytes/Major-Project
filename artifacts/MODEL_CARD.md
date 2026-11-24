# Model Card: PG-SBD Surrogate & Inverse Design Framework

> **[HONESTY] Research prototype trained on TCAD simulation data. Not validated for fabrication or experimental device design.**

## 1. Model Overview
- **Model Name:** Physics-Guided Surrogate Ensemble (PG-SBD)
- **Model Version:** 2.0.0 (Calibrated Hybrid Ensemble & Inverse Design Release)
- **Architecture:** Hybrid Physics-Guided Bounded Residual Ensemble ($M = 5$ bootstrap/multi-seed sub-models) + 1-D Parallel-Plane Breakdown Physics + Inverse Design Engine (Sobol+Adam & NSGA-II).
- **Core Formulation:**
  $$\log_{10} I(V_k) = \log_{10} I_{\text{phys}}(V_k;\,\text{WF}, T, \theta_{\text{cal}}) + c \cdot \tanh\!\left(\frac{g(\text{WF}, T)_k}{c}\right)$$
  where $c = 0.7\,\text{decades}$, and $g(\text{WF}, T)$ is a $2 \to 128 \to 128 \to 52$ neural residual with SiLU activations.
- **Framework:** PyTorch (CPU-only), NumPy, SciPy, Scikit-learn.

---

## 2. Dataset Provenance & Attribution
- **Dataset:** `GaO_Schottky_Diode_T_WF`
- **Source Repository:** [https://github.com/hywong2/Semiconductor_Database](https://github.com/hywong2/Semiconductor_Database)
- **Citation:** Dhillon et al., *TCAD-Augmented Machine Learning With and Without Domain Expertise*, IEEE Transactions on Electron Devices, 68(11), 2021.
- **Author / Contact:** Hiu Yung Wong (San Jose State University, `hiuyung.wong@sjsu.edu`)
- **Raw Checksum (SHA-256):** `8196234e58b057e4f3a0965825e20204ddcdb5e8f48bf46235b662e91d37649c`
- **Total Samples:** 2,000 curves $\times$ 52 voltage points ($V \in [0.0, 4.0]\,\text{V}$).
- **Splits (0% leakage):** Grouped by unique $(\text{WF}, T)$ pairs (Seed 42).
  - Train: 1,373 curves (1,335 unique pairs)
  - Validation: 289 curves (286 unique pairs)
  - Test: 299 curves (287 unique pairs)
  - Extrapolation Corner: 39 curves ($\text{WF} > 5.16\,\text{eV}$ and $T > 470\,\text{K}$)

---

## 3. Calibrated Physical Parameters ([artifacts/calib.json](calib.json))
Calibrated strictly on the training split via multi-start bounded least squares:
- $\chi_{\text{eff}} = 3.9841\,\text{eV}$
- Ideality factor $n = 1.0463$
- Effective contact area $A_{\text{eff}} = 1.711e-04\,\text{cm}^2$ ($r \approx 73.8\,\mu\text{m}$)
- $\mu_{300} = 250.0\,\text{cm}^2/\text{Vs}$
- Mobility temperature exponent $\alpha = 1.405$
- Contact resistance $R_c = 6.50e-11\,\Omega$
- **Analytic Core Residual MAE:** 0.0568 decades.

---

## 4. Actual Measured Benchmark Performance

### In-Domain Held-Out Test Set (N = 299)
- $\log_{10} I$ MAE (All $V>0$): **0.0069 decades**
- $\log_{10} I$ MAE ($I \ge 10^{-9}\,\text{A}$): **0.0011 decades**
- Turn-on $V_o$ MAE: **0.40 mV**
- Subthreshold Slope $SS$ MAE: **0.30 mV/dec**
- Differential Resistance $R_{\text{diff}}$ Relative Error: **0.75%**
- Turn-on Voltage $V_{\text{on}}$ (at 1 mA) MAE: **0.45 mV**

### Extrapolation Corner Block ($\text{WF} > 5.16\,\text{eV}, T > 470\,\text{K}$, N = 39)
- $\log_{10} I$ MAE (All $V>0$): **0.0010 decades**
- Turn-on $V_o$ MAE: **0.22 mV**
- Subthreshold Slope $SS$ MAE: **0.32 mV/dec**
- Differential Resistance $R_{\text{diff}}$ Relative Error: **0.64%**

---

## 5. Tier A Reverse Mode Parameter Recovery Benchmarks

Evaluated over 30 held-out TCAD curves with multi-start L-BFGS-B (9 starts across $M=5$ ensemble):
- **0% Multiplicative Noise:** $\text{WF MAE} = 0.0032\,\text{eV}$, $T\,\text{MAE} = 2.06\,\text{K}$, Average Latency $= 2,250\,\text{ms}$.
- **1% Multiplicative Noise:** $\text{WF MAE} = 0.0042\,\text{eV}$, $T\,\text{MAE} = 2.62\,\text{K}$, Average Latency $= 2,212\,\text{ms}$.
- **5% Multiplicative Noise:** $\text{WF MAE} = 0.0095\,\text{eV}$, $T\,\text{MAE} = 5.57\,\text{K}$, Average Latency $= 2,072\,\text{ms}$.
- Meets advisory recovery targets ($\text{WF MAE} < 0.02\,\text{eV}$, $T\,\text{MAE} < 25\,\text{K}$) with $> 4\times$ margin.

---

## 6. Tier B Design Mode & Breakdown Physics

- **1-D Analytic Breakdown (`analytic_unvalidated`):** Parallel-plane Poisson solver computing $E_c(N_{\text{net}})$, $W_c$, and breakdown voltage $V_{\text{br}}$ under punch-through (PT) and non-punch-through (NPT) regimes.
  - Baseline Device ($t_D=9\,\mu\text{m}, N_{\text{net}}=6\times 10^{15}\,\text{cm}^{-3}$): $V_{\text{br}} = 6,400.5\,\text{V}$ (PT), $R_{\text{drift,sp}} = 3.74\,\text{m}\Omega\cdot\text{cm}^2$, $\text{BFOM} = 10,939\,\text{MW/cm}^2$.
- **Differentiable PyTorch Metric Extractor:** Extracts $V_{\text{on}}, V_0, SS, R_{\text{diff}}, I(V_{\text{op}}), R_{\text{on,sp}}$ in $\approx 0.22\,\text{ms}$. Discrepancy against NumPy extractor $< 10^{-6}$.
- **Design Inverse Synthesis:**
  - 256-start Sobol + Adam (16 trajectories, 25 steps) synthesizes 8 Pareto candidates in $\approx 3.17\,\text{s}$.
  - NSGA-II (population 40, 25 generations) synthesizes 8 Pareto candidates in $\approx 4.20\,\text{s}$.
  - 200-curve round-trip self-consistency test achieved 100% convergence (200/200).
- **Automated Verification:** Candidate verdicts strictly classified as `met`, `met_within_tolerance`, `compromise`, or `infeasible`.

---

## 7. Evidence Taxonomy & Strict Operational Boundaries ([HONESTY])

All predictions and interface components display permanent evidence tags:
- **`tcad_validated`:** Parameters within TCAD training bounds ($\text{WF} \in [4.99, 5.19]\,\text{eV}$, $T \in [300, 500]\,\text{K}$).
- **`physics_extrapolated`:** Parameters varying beyond TCAD variation ($t_D \in [2.0, 20.0]\,\mu\text{m}$, $N_{\text{net}} \in [10^{15}, 5\times 10^{16}]\,\text{cm}^{-3}$, $d \in [50, 500]\,\mu\text{m}$). Forward conduction modeled via calibrated physics equations + drift scaling.
- **`analytic_unvalidated`:** 1-D parallel-plane breakdown voltages ($V_{\text{br}}$). Edge field crowding at real anode bevels will cause actual experimental breakdown to be 30%–60% lower unless advanced termination (e.g., field plates, guard rings) is implemented.
- **Zero-NaN Verification:** 10,000-point sweep across parameter bounds produced 0 NaNs and 0 Infs.
- **Monotonicity:** Dense-grid checks confirm nondecreasing current $I(V)$ across all operating voltages.

---

## 8. Intended Use & Out-of-Scope Use

- **Intended Use:** Fast interactive exploration of $I\text{--}V$ curves, parameter extraction from measured or simulated curves, Pareto multi-objective inverse design, and rapid generation of TCAD input decks for subsequent numerical simulation.
- **Out-of-Scope Use:** Direct tape-out or fabrication masks without full TCAD verification; high-frequency RF modeling; radiation-hardened or cryogenic ($< 250\,\text{K}$) operation.
