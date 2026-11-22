# Scientific Data Audit Report: GaO_Schottky_Diode_T_WF

> **[HONESTY] Research prototype trained on TCAD simulation data. Not validated for fabrication or experimental device design.**

**Audit Timestamp:** 2026-11-22  
**Dataset Source:** [https://github.com/hywong2/Semiconductor_Database](https://github.com/hywong2/Semiconductor_Database) (Folder: `GaO_Schottky_Diode_T_WF`)  
**Paper Reference:** Dhillon et al., *TCAD-Augmented Machine Learning With and Without Domain Expertise*, IEEE Transactions on Electron Devices, 68(11), 2021.  
**Contact / Attribution:** Hiu Yung Wong (San Jose State University, `hiuyung.wong@sjsu.edu`).

---

## 1. Provenance & File Manifest

| File | Size (Bytes) | SHA-256 Checksum | Source URL |
| :--- | :--- | :--- | :--- |
| `new_dataset_GaO_WF+T.csv` | 1,567,245 | `8196234e58b057e4f3a0965825e20204ddcdb5e8f48bf46235b662e91d37649c` | https://raw.githubusercontent.com/hywong2/Semiconductor_Database/master/GaO_Schottky_Diode_T_WF/new_dataset_GaO_WF%2BT.csv |
| `n9000_4e+16_3X4e+16_377_5014.csv` | 1,270 | `a57820bb5f1b8205521363c3934a36dba402bb318a92d3e58691cb1b36b67f36` | https://raw.githubusercontent.com/hywong2/Semiconductor_Database/master/GaO_Schottky_Diode_T_WF/n9000_4e%2B16_3X4e%2B16_377_5014.csv |
| `ds_file_9000_4e+16_3X4e+16_377_5014.cmd` | 2,368 | `5f7480ee1c62eb6f5d8759bb491604d8fe94e917b90ff85ec056cd384f40a0e1` | https://raw.githubusercontent.com/hywong2/Semiconductor_Database/master/GaO_Schottky_Diode_T_WF/ds_file_9000_4e%2B16_3X4e%2B16_377_5014.cmd |
| `n9000_4e+16_3X4e+16_377_5014_msh.cmd` | 1,733 | `7ff060a8cca1cf9ae84f3a4caf562cb89fee60c308733c24a2f746928839d8fd` | https://raw.githubusercontent.com/hywong2/Semiconductor_Database/master/GaO_Schottky_Diode_T_WF/n9000_4e%2B16_3X4e%2B16_377_5014_msh.cmd |
| `Readme` | 912 | `a130953a26bf7b4c546765f4dbb43f2430e162c5f78548b15db372c1db3fc862` | https://raw.githubusercontent.com/hywong2/Semiconductor_Database/master/GaO_Schottky_Diode_T_WF/Readme |
| `TCAD-Augmented_Machine_Learning_With_and_Without_Domain_Expertise.pdf` | 1,800,537 | `772578b4febf1d72bcdbbcbd48d1a89eb3d8f2a84b1727143b7251ae0eb96dac` | https://raw.githubusercontent.com/hywong2/Semiconductor_Database/master/GaO_Schottky_Diode_T_WF/TCAD-Augmented_Machine_Learning_With_and_Without_Domain_Expertise.pdf |

---

## 2. Dataset Schema & Structure Verification

- **Total Rows:** 2000
- **Total Columns:** 57
  - **I-V Columns (integer-named):** 52 columns (`"0"` through `"51"`)
  - **Voltage Grid:** $V = \text{linspace}(0.0, 4.0, 52)\text{ V}$, step = 0.0784 V
  - **Parameter Columns:** `t_Drift, temperature, WF, D_N, D_C`
- **Missing / NaN Values:** 0
- **Infinite Values:** 0
- **Duplicate Rows:** 54 (54 duplicate entries exist in the raw dataset; grouped splitting ensures identical curves never leak across splits)

---

## 3. Parameter Coverage & Bounds Audit

| Parameter | Unit | Min | Max | Unique Values | Description / Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `temperature` | K | 300.0 | 500.0 | 201 | Operating temperature (swept quasi-randomly) |
| `WF` | eV | 4.99 | 5.19 | 201 | Effective anode work function (swept quasi-randomly) |
| `t_Drift` | $\mu\text{m}$ | 9.0 | 9.0 | 1 | Drift layer thickness (constant baseline) |
| `D_N` | $\text{cm}^{-3}$ | 4.00e+16 | 4.00e+16 | 1 | Donor concentration ($4\times 10^{16}\,\text{cm}^{-3}$) |
| `D_C` | $\text{cm}^{-3}$ | 3.40e+16 | 3.40e+16 | 1 | Compensation acceptor concentration ($3.4\times 10^{16}\,\text{cm}^{-3}$) |

- **Net Donor Concentration ($N_{\text{net}}$):** $D_N - D_C = 4.0\times 10^{16} - 3.4\times 10^{16} = 6.0\times 10^{15}\,\text{cm}^{-3}$.
- **Unique $(\text{WF}, T)$ pairs:** 1946 out of 2000 rows.
- **Repeated $(\text{WF}, T)$ curves:** 54 (grouped splitting enforces zero data leakage).

---

## 4. Current Sign, Noise, and Monotonicity Audit

- **Current Range:** $[-9.518e-18, 1.234e-01]\text{ A}$
- **Nonpositive Values ($I \le 0\text{ A}$):**
  - Count: 959 entries out of 104000 total data points (0.922%).
  - Minimum negative current: -9.518e-18 A (strictly numerical noise near $0\text{ V}$, order of $10^{-17}\,\text{A}$ to $10^{-14}\,\text{A}$).
  - Cleaning rule: Tiny nonpositive values are floored at $10^{-14}\,\text{A}$ for log operations.
- **Monotonicity in $V$:**
  - Strict $(I_{k+1} - I_k < 0)$: 0 rows exhibit non-monotonicity.
  - With numerical tolerance $(-10^{-12}\,\text{A})$: 0 non-monotonic rows. Currents are monotonic across the active forward range.

---

## 5. Physical Trend Verification

- **$\partial V_o / \partial \text{WF}$:**
  - Correlation $r(V_o, \text{WF}) = 0.5620$ (strongly positive).
  - Empirical multivariable slope: $\frac{\partial V_o}{\partial \text{WF}} \approx 0.9999\,\text{V/eV}$ (matches ideal physical theory $d\phi_B/d\text{WF} = 1.0\,\text{V/eV}$).
  - **Conclusion:** As anode work function increases, Schottky barrier increases linearly, shifting turn-on $V_o$ to higher voltages. Penalizing $\partial V_o / \partial \text{WF} < 0$ in surrogate training is **empirically and physically verified**.

- **$\partial V_o / \partial T$:**
  - Correlation $r(V_o, T) = -0.8458$ (strongly negative).
  - Empirical multivariable slope: $\frac{\partial V_o}{\partial T} \approx -1.540\,\text{mV/K}$ (-0.001540 V/K).
  - **Conclusion:** As temperature increases, thermionic emission increases exponentially, shifting turn-on $V_o$ to lower voltages. Penalizing $\partial V_o / \partial T > 0$ in surrogate training is **empirically and physically verified**.

---

## 6. Discrepancy & TCAD Inspection Notes

1. **README Text vs. Folder / Paper:**
   - In `Readme`, the text states: *"The device is vertical GaN Schottky diode..."*
   - However, the repository directory name is `GaO_Schottky_Diode_T_WF`, the command files specify `material = "GaO"` / `beta-Ga2O3`, and the accompanying paper (Dhillon et al., 2021) explicitly analyzes $\beta\text{-Ga}_2\text{O}_3$ vertical SBDs.
   - **Resolution:** The README text contains a typographical copy-paste artifact referring to GaN. The physical device simulated is definitively vertical $\beta\text{-Ga}_2\text{O}_3$.

2. **TCAD Sdevice Physical Models Verified:**
   - Coordinate System: Cylindrical (`Math {{ Cylindrical }}`).
   - Self-Heating: Thermodynamic lattice temperature simulation (`Physics {{ Thermodynamic }}`).
   - Incomplete Ionization: Explicitly enabled (`IncompleteIonization`).
   - Doping & Mobility: `Phumob` (doping dependent mobility with high field saturation).
   - Recombination: Shockley-Read-Hall with doping dependence (`SRH(DopingDep)`).
   - Anode definition: `Schottky workfunction=...`
   - Cathode definition: `Resist=2.2116e-09`, `SurfaceResistance=1e-6`.
   - *Physical takeaway:* The analytic core provides the primary thermionic + drift resistance foundation, while the bounded neural residual naturally absorbs the higher-order self-heating, incomplete ionization, and PhuMob mobility effects along $(\text{WF}, T)$.

3. **Anode Geometry / Active Area:**
   - SDE file `n9000_4e+16_3X4e+16_377_5014_msh.cmd` defines cylindrical coordinates with refinement windows around 120–140 $\mu\text{m}$.
   - Absolute current levels reach $\sim 0.123\,\text{A}$ at $4\,\text{V}$. The effective active area $A_{\text{eff}}$ will be calibrated as part of Phase 3 rather than assumed fixed.

4. **Extrapolation Corner Feasibility:**
   - Samples with $\text{WF} > 5.16\,\text{eV}$ and $T > 470\,\text{K}$: **39 curves**.
   - This provides a solid held-out corner block for testing physical vs. ML extrapolation.
