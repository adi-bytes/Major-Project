# PG-SBD Designer

> **[HONESTY] Research prototype trained on TCAD simulation data. Not validated for fabrication or experimental device design.**

**Physics-Guided Surrogate Modeling and Inverse Design Framework for Power Semiconductor Devices**  
*Case-Study Device (Plug-in #1):* Vertical $\beta\text{-Ga}_2\text{O}_3$ Schottky Barrier Diode (SBD)  
*Primary Dataset:* `GaO_Schottky_Diode_T_WF` from [https://github.com/hywong2/Semiconductor_Database](https://github.com/hywong2/Semiconductor_Database) (Dhillon et al., *IEEE Transactions on Electron Devices*, 68(11), 2021).

---

## Current Status: Phase 9 Complete (System Hardened, E2E Tested & Documented)

All 9 phases of the PG-SBD research and development roadmap are complete, verified with 52 automated tests (including 10,000-point sweeps, full API contract suites, and headless browser end-to-end integration).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PG-SBD Full System Architecture                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   [ React 18 + TS + Tailwind + Plotly.js Frontend ] (Port 5173)                 │
│    ├── Overview & Methodology View (Interactive architecture + equations)       │
│    ├── Forward Prediction Workbench (Ensemble mean + ±2σ uncertainty band)      │
│    ├── Reverse Mode Workbench (Tier A WF/T recovery + hidden ground truth)      │
│    ├── Design Mode Workbench (Tier B NSGA-II/Sobol Pareto + SVG Cross-Section)  │
│    ├── Validation Dashboard (Live benchmark curves + ablation comparisons)      │
│    └── Model / Data Details (TCAD provenance, manifests, and calib constants)   │
│                                      │                                          │
│                                      ▼ REST API (CORS enabled)                  │
│   [ FastAPI Backend Service ] (Port 8000)                                       │
│    ├── GET  /api/health       GET  /api/meta          GET  /api/validation      │
│    ├── POST /api/predict      POST /api/reverse       GET  /api/design/presets  │
│    ├── POST /api/design       POST /api/export/tcad   POST /api/export/report   │
│                                      │                                          │
│         ┌────────────────────────────┼────────────────────────────┐             │
│         ▼                            ▼                            ▼             │
│   [ Hybrid Ensemble ]       [ Tier A Reverse Engine ]    [ Tier B Design Engine ]│
│   - M=5 Bounded Models      - 9-Start L-BFGS-B           - Sobol + Adam (256 st)│
│   - Lambert-W Physics Core  - Weighted log-I + Vo/SS     - pymoo NSGA-II (40x25)│
│   - Monotonicity Penalty    - < 0.005 eV WF error        - 1-D Poisson BV (PT)  │
│   - Extrap MAE: 0.001 dec   - < 3 s solve on CPU         - Candidate Clustering │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Summary of Completed Phases

| Phase | Title | Key Milestones & Deliverables |
| :--- | :--- | :--- |
| **Phase 1** | **Data Acquisition & Audit** | Verified raw Sentaurus TCAD dataset with SHA-256 manifest; 2,000 curves $\times$ 52 voltage points; established grouped splits (Train: 1373, Val: 289, Test: 299, Extrap: 39) with 0% data leakage; documented in `reports/data_audit.md`. |
| **Phase 2** | **Plain ML Baselines** | Implemented scikit-learn (Ridge, HistGradientBoosting, KNN $k=3$) and plain PyTorch MLP baselines; established metric harness ($V_o, SS, R_{\text{diff}}, V_{\text{on}}, I(V_{\text{op}})$). |
| **Phase 3** | **Differentiable Physics Core** | Built closed-form log-domain Lambert-W solver passing `torch.autograd.gradcheck`; calibrated physical parameters ($\chi_{\text{eff}} = 3.9841\,\text{eV}$, $n = 1.0463$, $\mu_{300} = 250.0\,\text{cm}^2/\text{Vs}$, $\alpha = 1.405$, $A_{\text{eff}} = 1.711\times 10^{-4}\,\text{cm}^2$) stored in `artifacts/calib.json`. |
| **Phase 4** | **Hybrid Ensemble & Ablation** | Developed $M=5$ bounded residual ensemble ($\log_{10} I = \log_{10} I_{\text{phys}} + c\tanh(g/c)$) with monotonicity loss; passed 10,000-point zero-NaN sweep; lowest $\log_{10} I$ MAE on the corner block (0.0010 dec; the corner lies inside the sampled box, so it is a mild test). A plain MLP is as accurate or better on the random test split; ran 3-arm ablation study. |
| **Phase 5** | **Tier A Reverse Mode** | Implemented 9-start normalized L-BFGS-B engine recovering $(\text{WF}, T)$ from arbitrary $I\text{--}V$ curves; benchmarked across 0%, 1%, and 5% noise; integrated nearest-TCAD lookup and ambiguity warnings. |
| **Phase 6** | **FastAPI Backend** | Built production FastAPI service with strict Pydantic v2 bound validation; startup artifact preloading (zero training on request); CORS support for Vite; API contract integration tests. |
| **Phase 7** | **React Frontend** | Built interactive React 18 + TS + Vite + Tailwind + Plotly.js application; implemented all 6 views with permanent `[HONESTY]` banner; hidden ground-truth reveal in Reverse view; 0 hard-coded metrics (all read from API). |
| **Phase 8** | **Tier B Inverse Design** | Developed 1-D parallel-plane analytic breakdown calculator (`analytic_unvalidated`); differentiable PyTorch metric extractor ($< 10^{-6}$ error vs NumPy); Sobol+Adam (256 starts) and NSGA-II synthesis engines; candidate verification verdicts; parametric SVG cross-section; TCAD deck export. |
| **Phase 9** | **Hardening & Documentation** | Root `Makefile` with 7 standard targets; headless Playwright test suite verifying all 6 views in 8.17 s; 10,000-point sweep verification; GaN/SiC plug-in roadmap (`docs/ROADMAP_GAN_SIC.md`); finalized reports. |

---

## 2. Actual Measured Benchmark Results

All metrics reflect actual measured numbers recorded in `artifacts/metrics.json` and `reports/validation_report.md`.

### 2.1 Forward Surrogate Performance (Held-Out Test & Extrapolation Splits)

| Model | Test $\log_{10} I$ MAE (dec) | Test $V_o$ MAE (mV) | Test $R_{\text{diff}}$ Err (%) | Extrap $\log_{10} I$ MAE (dec) | Extrap $V_o$ MAE (mV) | Extrap $R_{\text{diff}}$ Err (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Physics-Guided Ensemble** | **0.0069** | **0.40** | **0.75%** | **0.0010** | **0.22** | **0.64%** |
| **PlainMLP** | 0.0017 | 0.24 | 3.83% | 0.0025 | 0.18 | 2.94% |
| **HistGradientBoosting** | 0.0064 | 2.49 | 1.00% | 0.0223 | 7.80 | 0.89% |
| **KNN ($k=3$)** | 0.0068 | 2.48 | 0.26% | 0.0278 | 10.99 | 1.29% |
| **Ridge** | 0.0399 | 16.96 | 0.20% | 0.0252 | 5.06 | 0.56% |
| **Physics-Only** | 0.0564 | 16.81 | 26.04% | 0.0650 | 33.19 | 16.76% |

### 2.2 Tier A Reverse Mode Parameter Recovery (30 Held-Out Curves)

| Noise Condition | WF MAE (eV) | WF RMSE (eV) | Temperature MAE (K) | Temperature RMSE (K) | Solve Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0% Multiplicative Noise** | **0.0032** | 0.0047 | **2.06** | 3.14 | 2,250 ms |
| **1% Multiplicative Noise** | **0.0042** | 0.0054 | **2.62** | 3.46 | 2,212 ms |
| **5% Multiplicative Noise** | **0.0095** | 0.0121 | **5.57** | 7.07 | 2,072 ms |

### 2.3 Tier B Design Mode Breakdown & Synthesis Performance
- **1-D Analytic Breakdown (Baseline Device):** ideal limit $V_{\text{br,ideal}} \approx 6,400\,\text{V}$; the reported breakdown is derated by an assumed termination efficiency $\eta_{\text{term}} = 0.5$ (unvalidated) (punch-through, $W_c = 70.02\,\mu\text{m}$, $R_{\text{drift,sp}} = 3.74\,\text{m}\Omega\cdot\text{cm}^2$, $\text{BFOM} = 10,939\,\text{MW/cm}^2$).
- **Sobol + Adam Synthesis (256 starts $\to$ 16 top):** 8 Pareto candidates generated in **3.17 s**.
- **NSGA-II Synthesis (40 population, 25 generations):** 8 Pareto candidates generated in **4.20 s**.
- **Verdict-function consistency check (200 random designs):** confirms the verdict logic is internally consistent. It does **not** run the inverse solver, so it is not evidence of inverse-design accuracy; a held-out round-trip test is future work.

---

## 3. Evidence Taxonomy & Operational Badges

Every parameter, chart, and export card in PG-SBD carries explicit evidence badges:

- `tcad_validated`: Parameter combinations validated directly against Sentaurus TCAD simulations ($\text{WF} \in [4.99, 5.19]\,\text{eV}$, $T \in [300, 500]\,\text{K}$).
- `physics_extrapolated`: Structural parameters extrapolated via calibrated physical equations and mobility scaling ($t_D \in [2.0, 20.0]\,\mu\text{m}$, $N_{\text{net}} \in [10^{15}, 5\times 10^{16}]\,\text{cm}^{-3}$, $d \in [50, 500]\,\mu\text{m}$).
- `analytic_unvalidated`: 1-D parallel-plane breakdown voltages. Real devices without termination will breakdown at 30%–60% lower voltage due to electric field crowding at anode edges.

---

## 4. Quick Start & Makefile Usage

The root `Makefile` provides 7 standard workflow targets:

```bash
# 1. Fetch raw TCAD data from repository and run cryptographic SHA-256 audit
make data

# 2. Calibrate the differentiable Lambert-W physics core
make calibrate

# 3. Train all plain ML baselines and the M=5 hybrid residual ensemble
make train

# 4. Run the full evaluation harness and update artifacts/metrics.json
make eval

# 5. Build the UI once, then serve API + UI together at http://127.0.0.1:8000
make ui
make serve

# 6. Execute the complete test suite (unit, integration, API contract, and Playwright E2E)
make test

# 7. Execute the headless end-to-end demo script (forward, reverse, breakdown, design)
make demo
```

---

## 5. Running the Headless Playwright Test

Ensure the backend and frontend servers are running, then launch the Playwright E2E verification test:

```powershell
# In PowerShell:
.\.venv\Scripts\pytest.exe tests/test_playwright_e2e.py -v
```

The test navigates across all 6 views (Overview, Forward, Reverse, Design, Validation, Details), submits forward inference, executes reverse parameter recovery, generates Pareto candidates in Design Workbench, inspects the SVG cross-section, and verifies that the permanent `[HONESTY]` banner is present throughout.

---

## 6. Wide Bandgap Extension Roadmap

PG-SBD is engineered as a modular framework. See [docs/ROADMAP_GAN_SIC.md](docs/ROADMAP_GAN_SIC.md) for the complete architecture specification covering:
- **Gallium Nitride (GaN):** Vertical native SBDs and lateral 2DEG AlGaN/GaN HEMTs ($E_c \approx 3.3\,\text{MV/cm}$, $\mu \approx 1200\,\text{cm}^2/\text{Vs}$, $\text{BFOM} \approx 2,000 \times \text{Si}$).
- **Silicon Carbide (4H-SiC):** High-voltage SBD and Merged PN-Schottky (MPS/JBS) diodes ($E_c \approx 3.0\,\text{MV/cm}$, incomplete nitrogen ionization, $\text{BFOM} \approx 840 \times \text{Si}$).
- **Multi-Material Pareto Optimization:** Cross-material trade-off analysis along the 1-D Baliga power figure of merit.

---

## 7. Ethical & Research Disclaimers

> **[HONESTY] Notice:**  
> This software is an academic research prototype developed strictly for simulation exploration, inverse surrogate demonstration, and TCAD automation. All surrogate neural networks and physics models are calibrated on numerical TCAD simulations and unvalidated 1-D approximations. This software is **NOT certified for physical tape-out, semiconductor fabrication, high-voltage equipment deployment, or safety-critical power electronic applications**. Physical devices require experimental fabrication, characterization, bevel termination modeling, and empirical breakdown testing.
