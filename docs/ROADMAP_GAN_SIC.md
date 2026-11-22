# Roadmap: GaN and 4H-SiC Plug-in Architecture

> **[HONESTY] Research prototype trained on TCAD simulation data. Not validated for fabrication or experimental device design.**

This document provides the technical roadmap and architectural specification for extending the **PG-SBD** framework from vertical $\beta\text{-Ga}_2\text{O}_3$ (Plug-in #1) to other wide-bandgap (WBG) and ultra-wide-bandgap (UWBG) power semiconductor technologies, specifically **Gallium Nitride (GaN)** and **Silicon Carbide (4H-SiC)**.

---

## 1. Modular Plug-in Architecture

The core philosophy of PG-SBD is the strict separation of **device-agnostic surrogate engines** from **material-specific physics specifications**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PG-SBD Inverse & Surrogate                      │
│      (FastAPI, React Frontend, NSGA-II, Sobol+Adam, Multi-start)       │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           │        Abstract Device Specification API         │
           │      (DeviceSpec: Geometry, Physics, Bounds)      │
           └──────┬────────────────────┬────────────────────┬──┘
                  │                    │                    │
                  ▼                    ▼                    ▼
     ┌────────────────────────┐  ┌─────────────┐  ┌──────────────────┐
     │  β-Ga2O3 Vertical SBD  │  │   GaN SBD   │  │    4H-SiC SBD    │
     │     (Plug-in #1)       │  │ (Plug-in #2)│  │   (Plug-in #3)   │
     │      [CURRENT]         │  │  [ROADMAP]  │  │    [ROADMAP]     │
     └────────────────────────┘  └─────────────┘  └──────────────────┘
```

Each plug-in implements the standard `DeviceSpec` interface:
1. **Material Properties:** Bandgap $E_g$, permittivity $\epsilon_r$, electron affinity $\chi$, effective mass $m^*$, Richardson constant $A^*$, impact ionization coefficients $(\alpha_n, \beta_n)$ or empirical critical field exponent $E_c(N_{\text{net}})$.
2. **Conduction Physics Core:** Differentiable forward current model combining thermionic emission with temperature- and doping-dependent drift series resistance $R_{\text{drift}}(\mu(N, T), t_D)$.
3. **Analytic Breakdown Core:** Parallel-plane 1-D Poisson integration under punch-through (PT) and non-punch-through (NPT) conditions.
4. **Parameter Bounds & Evidence Registry:** Defining explicit $[P_{\text{min}}, P_{\text{max}}]$ and evidence tiers (`tcad_validated`, `physics_extrapolated`, `analytic_unvalidated`).

---

## 2. Gallium Nitride (GaN) Plug-in Specification

### 2.1 Physical Properties & Parameters

| Parameter | Symbol | Vertical GaN | Lateral AlGaN/GaN | Unit |
| :--- | :--- | :--- | :--- | :--- |
| Bandgap ($300\text{ K}$) | $E_g$ | 3.40 | 3.40 | $\text{eV}$ |
| Relative Permittivity | $\epsilon_r$ | 8.9 | 8.9 | - |
| Electron Affinity | $\chi$ | 4.10 | 4.10 | $\text{eV}$ |
| Effective Mass | $m^* / m_0$ | 0.20 | 0.20 | - |
| Effective Richardson Const | $A^*$ | 24.0 | 24.0 | $\text{A/cm}^2\text{K}^2$ |
| Low-Field Mobility ($300\text{ K}$) | $\mu_{300}$ | 1000–1200 | 1500–2000 (2DEG) | $\text{cm}^2/\text{V}\cdot\text{s}$ |
| Mobility Temp Exponent | $\alpha_\mu$ | 1.5–2.0 | 1.5 | - |
| Critical Electric Field | $E_c$ | $\approx 3.3$ | $\approx 3.3$ | $\text{MV/cm}$ |
| Critical Field Scaling | $E_c(N)$ | $2.8 \times 10^5 \cdot N_{\text{net}}^{0.06}$ | N/A (lateral) | $\text{V/cm}$ |

### 2.2 Key Device Architectures
1. **Vertical GaN SBD on Native GaN Substrates:**
   - Low dislocation densities ($< 10^5\,\text{cm}^{-2}$) enable high breakdown voltages ($1.2\text{ kV} - 3.3\text{ kV}$).
   - Forward drop governed by barrier height $\phi_B \approx \text{WF} - \chi$ and native GaN substrate resistance.
   - Anode metals: Ni ($\phi_B \approx 1.1\,\text{eV}$), Pt ($\phi_B \approx 1.3\,\text{eV}$), Mo ($\phi_B \approx 0.9\,\text{eV}$).
2. **Lateral 2DEG Schottky Diodes (AlGaN/GaN on Si/Sapphire/SiC):**
   - High electron mobility 2DEG channel at the heterointerface.
   - Requires updating the drift resistance model from 3D bulk volume conduction to 2D sheet charge resistance $R_{\text{sheet}} = \frac{1}{q \mu_{\text{2DEG}} n_s} \frac{L_D}{W}$.

### 2.3 Modeling Challenges for GaN
- **High-Temperature Trapping & Current Collapse:** Virtual gate effect and surface state traps requiring dynamic resistance factors.
- **Polarization Charges:** Spontaneous and piezoelectric polarization charges must be included in Poisson boundary conditions for lateral heterostructures.

---

## 3. Silicon Carbide (4H-SiC) Plug-in Specification

### 3.1 Physical Properties & Parameters

| Parameter | Symbol | 4H-SiC Value | Unit | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Bandgap ($300\text{ K}$) | $E_g$ | 3.26 | $\text{eV}$ | Indirect bandgap |
| Relative Permittivity | $\epsilon_r$ | 9.66 | - | Anisotropic tensor $\epsilon_\parallel, \epsilon_\perp$ |
| Electron Affinity | $\chi$ | 3.70 | $\text{eV}$ | |
| Effective Mass | $m^* / m_0$ | 0.35 | - | Density-of-states effective mass |
| Effective Richardson Const | $A^*$ | 42.0 | $\text{A/cm}^2\text{K}^2$ | |
| Bulk Drift Mobility ($300\text{ K}$) | $\mu_{300}$ | 900–950 | $\text{cm}^2/\text{V}\cdot\text{s}$ | Along c-axis (0001) |
| Mobility Temp Exponent | $\alpha_\mu$ | 2.1–2.4 | - | Acoustic & optical phonon scattering |
| Critical Electric Field | $E_c$ | $\approx 3.0$ | $\text{MV/cm}$ | At $N_D = 10^{16}\,\text{cm}^{-3}$ |
| Critical Field Scaling | $E_c(N)$ | $2.49 \times 10^5 \cdot N_{\text{net}}^{0.065}$ | $\text{V/cm}$ | Hatakeyama et al. ionization model |

### 3.2 Incomplete Ionization Model
Unlike silicon, nitrogen donors in 4H-SiC have non-negligible ionization energy ($E_{d,h} \approx 50\text{ meV}$ on hexagonal sites, $E_{d,k} \approx 90\text{ meV}$ on cubic lattice sites). The ionized donor density $N_D^+$ as a function of temperature must be solved iteratively:
$$N_D^+(T) = \frac{N_D}{1 + g_D \exp\!\left(\frac{E_F - E_d}{k_B T}\right)}$$
This incomplete ionization causes drift resistance to drop with increasing temperature initially before lattice phonon scattering causes mobility degradation at $T > 350\text{ K}$.

### 3.3 Advanced Architectures: JBS and MPS Diodes
For commercial 4H-SiC power applications ($650\text{ V} - 3.3\text{ kV}$), pure SBDs suffer from barrier lowering under high reverse electric field ($\Delta \phi_B = \sqrt{q E_m / 4\pi\epsilon_s}$).
- **Junction Barrier Schottky (JBS) / Merged PN-Schottky (MPS):**
  - Boron/aluminum $p^+$ implants pinch off the Schottky contact under high reverse bias, shielding the metal-semiconductor interface from high field.
  - Forward mode: Unipolar Schottky conduction at low voltage; bipolar injection from $p^+$ regions during surge current events ($I_{\text{surge}} > 5 \times I_{\text{nom}}$).
  - Implementation requires a 2-stage piecewise conduction model combining Schottky Lambert-W with $p\text{-}n$ junction injection.

---

## 4. Multi-Material Pareto Benchmarking

When all three plug-ins are instantiated, the PG-SBD Design Workbench will support **cross-material Pareto trade-off comparisons** across the 1-D theoretical Baliga limits:

```
  Specific On-Resistance Ron,sp (mΩ·cm²)
    ▲
100 ┼                                   ● Si Limit: Ron,sp = 5.9e-9 · Vbr^2.5
    │                                ╱
 10 ┼                             ╱     ● 4H-SiC Limit (BFOM ~ 840)
    │                          ╱     ╱
  1 ┼                       ╱     ╱     ● GaN Limit (BFOM ~ 2,000)
    │                    ╱     ╱     ╱
0.1 ┼                 ╱     ╱     ╱     ● β-Ga2O3 Limit (BFOM ~ 10,000)
    │              ╱     ╱     ╱     ╱
    └──────┬──────┴─────┬─────┴─────┬────────►
          100          1000        10000
                    Breakdown Voltage Vbr (V)
```

### Theoretical Baliga Figure of Merit (BFOM) Comparison:
- **Silicon (Si):** $\text{BFOM} \equiv 1$ (Baseline)
- **4H-SiC:** $\epsilon_r \mu E_c^3 \approx 840 \times \text{Si}$
- **GaN:** $\epsilon_r \mu E_c^3 \approx 2,000 \times \text{Si}$
- **$\beta\text{-Ga}_2\text{O}_3$:** $\epsilon_r \mu E_c^3 \approx 10,939 \times \text{Si}$ (Ultra-wide bandgap advantage)

---

## 5. Software Implementation Roadmap & Milestones

| Milestone | Deliverable | Description |
| :--- | :--- | :--- |
| **M1: Spec Refactoring** | `src/pgsbd/device/base.py` | Extract `DeviceSpec` abstract base class from current $\beta\text{-Ga}_2\text{O}_3$ implementation. |
| **M2: GaN Spec Implementation** | `src/pgsbd/device/gan.py` | Implement vertical GaN parameters, field-dependent mobility, and analytic breakdown. |
| **M3: SiC Spec & Incomplete Ionization** | `src/pgsbd/device/sic.py` | Implement 4H-SiC parameters with incomplete nitrogen donor ionization solver. |
| **M4: API & Frontend Selector** | `POST /api/design` + UI | Add `material: "ga2o3" \| "gan" \| "sic"` selector in the Design Workbench requirement form. |
| **M5: TCAD Deck Generator Extension** | Sentaurus & Silvaco | Extend `POST /api/export/tcad` to generate device-specific `.cmd` and `.msh` decks for GaN and 4H-SiC. |
| **M6: Verification & Validation** | `tests/test_device_specs.py` | Unit test suite verifying physical bounds, numerical convergence, and 1-D Poisson breakdown consistency across all materials. |
