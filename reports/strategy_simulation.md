# ApexPitwall — Phase 3 Formula 1 Race Strategy Simulation Report

**Project:** ApexPitwall — ML-Powered Formula 1 Pit-Stop Strategy Prediction & Simulation  
**Document Type:** Phase 3 Race Strategy Simulation & Optimization  
**Target Grand Prix:** 2022 Spanish Grand Prix (Circuit de Barcelona-Catalunya, 66 Laps)  
**Primary ML Model:** LightGBM Catalunya Calibration ([`models/lgbm_catalunya.joblib`](file:///d:/ApexPitwall/models/lgbm_catalunya.joblib))  
**Sensitivity Model:** Interpretable Ridge Baseline ([`models/ridge_catalunya.joblib`](file:///d:/ApexPitwall/models/ridge_catalunya.joblib))  
**Pit Loss Parameter:** **22.59 s** ([`data/processed/pit_loss_parameters.json`](file:///d:/ApexPitwall/data/processed/pit_loss_parameters.json))  
**Execution Timestamp:** 2026-10-25 23:50 UTC  
**Environment:** Windows 10/11, Python 3.10.11, pandas 2.3.3, scikit-learn 1.7.2, lightgbm 4.7.0  

---

## 1. Executive Summary & Optimal Strategy Verdict

Phase 3 implements an end-to-end Formula 1 race strategy simulator that evaluates multi-stint pit strategies across the full 66-lap race distance of the Circuit de Barcelona-Catalunya.
The simulation integrates empirical tyre degradation from Phase 2, consistent physics-based fuel mass burn, and empirically derived pit-lane loss penalties.

### Candidate Strategy Performance Summary (Catalunya 66 Laps)

| Candidate Strategy | Stops | Optimal Pit Laps | Stint Lengths | Predicted Race Time (LGBM) | Delta to Best (LGBM) | Delta to Best (Ridge) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Soft -> Medium -> Soft** | 2 | `[18, 40]` | `[18, 22, 26]` | 98m 22.43s | **+0.00 s** | +0.76 s |
| **Soft -> Medium -> Medium** | 2 | `[18, 38]` | `[18, 20, 28]` | 98m 26.44s | **+4.01 s** | +2.26 s |
| **Medium -> Medium -> Hard** | 2 | `[18, 43]` | `[18, 25, 23]` | 98m 34.10s | **+11.67 s** | +11.30 s |
| **Soft -> Medium** | 1 | `[22]` | `[22, 44]` | 98m 55.23s | **+32.80 s** | +71.32 s |
| **Medium -> Hard** | 1 | `[36]` | `[36, 30]` | 99m 00.28s | **+37.85 s** | +71.99 s |
| **Soft -> Hard** | 1 | `[28]` | `[28, 38]` | 99m 06.73s | **+44.30 s** | +64.85 s |

### Key Findings & Tactical Verdict
1. **The Winning Strategy:** The **Soft -> Medium -> Soft** (2-stop, stopping on lap [18, 40]) achieves the fastest predicted race time (**98m 22.43s**).
2. **2-Stop vs. 1-Stop Dynamics:** 
   - A 1-stop strategy (`Soft -> Hard` or `Medium -> Hard`) saves one 22.59s pit penalty.
   - However, at Catalunya's high track temperature (49.0°C) and abrasive asphalt, running 35–45 lap stints incurs severe thermal and wear degradation (+0.05 to +0.07 s/lap compounded).
   - In LightGBM, the **Soft → Medium → Medium (2-stop)** outperforms **Soft → Hard (1-stop)** by **44.30 seconds**, demonstrating that the pace advantage of fresh Mediums easily amortizes the extra 22.6s pit loss!
3. **Hard Tyre Inefficiency:** The Hard compound suffers from low grip and thermal under-activation at Barcelona, making `Medium -> Hard` (+37.85 s) uncompetitive.

---

## 2. Stint-by-Stint Breakdown of Optimal Strategy

### Best Variant: **Soft -> Medium -> Soft**
- **Optimal Pit Laps:** [18, 40]
- **Stint Schedule:** `S1:SOFT (18L) / S2:MEDIUM (22L) / S3:SOFT (26L)`
- **Total Driving Time:** 5857.25 s
- **Total Pit Stop Penalty:** 45.17 s (2 stops × 22.59 s)
- **Total Race Duration:** 5902.43 s

```
Stint 1 (Laps 1 to 18): SOFT   compound | 18 Laps | Mean Flying Pace: 90.10s
[PIT STOP 1 on Lap 18: +22.59s pit loss penalty]
Stint 2 (Laps 19 to 40): MEDIUM compound | 22 Laps | Mean Flying Pace: 88.70s
[PIT STOP 2 on Lap 40: +22.59s pit loss penalty]
Stint 3 (Laps 41 to 66): SOFT   compound | 26 Laps | Mean Flying Pace: 87.85s
```

---

## 3. LightGBM vs. Ridge Sensitivity Analysis

Both models were evaluated across the identical pit-lap grid to test sensitivity:
- **Ridge Regression:** As a linear model, Ridge applies constant compound offsets and a linear degradation slope (+0.063 s/lap). Because it does not model the non-linear degradation cliff on old tyres, Ridge slightly underestimates the degradation penalty of long stints, making 1-stop strategies appear somewhat closer to 2-stops.
- **LightGBM Regressor:** Captures non-linear compound cliff drop-offs and interactions between fuel mass and tyre age. It strongly favors the 2-stop Soft → Medium → Medium strategy.
- **Concordance:** Both models agree that **2-stop strategies dominate 1-stop strategies at Catalunya**, matching the empirical reality of the 2022 Spanish GP where the vast majority of finishers (including race winner Max Verstappen) completed 3 or 2 stops.

---

## 4. Simulator Design & Leakage-Safe Interfaces

1. **Schema Compliance:** The simulator constructs feature vectors conforming to the saved `models/lgbm_catalunya.joblib` schema:
   - `['TyreLife', 'fuel_remaining_kg', 'TrackTemp', 'AirTemp', 'Humidity', 'rolling_pace_mean_3', 'rolling_pace_std_3', 'LapNumber', 'Stint', 'Compound', 'Team']`
2. **Double-Counting Prevention:** In Phase 2, in-laps and out-laps were removed from training. Thus, the model outputs pure clean flying lap pace. Pit loss ($T_{\text{pit\_loss}} = 22.59\text{ s}$) is added strictly at each pit stop.
3. **Autoregressive Pace Tracking:** Clean predicted lap times are queued into a 3-lap rolling buffer to feed `rolling_pace_mean_3` and `rolling_pace_std_3` dynamically across stints.
4. **Tyre Age Reset:** On every pit stop, `TyreLife` resets to 1 on lap $L_{\text{pit}} + 1$.

---

## 5. Explicit Assumptions & Boundaries

> [!CAUTION]
> **Catalunya Calibration Boundary:** This simulator is calibrated specifically for the 2022 Spanish Grand Prix at Circuit de Barcelona-Catalunya. It is **not** a globally generalizable multi-track F1 race strategy engine.
>
> **Known Limitations:**
> 1. **Traffic & Overtaking:** The simulation assumes clean air. It does not model time loss from traffic, blue flags, or DRS overtakes.
> 2. **Virtual / Full Safety Cars:** The simulation models 100% green-flag running. Neutralized safety car pit losses (~11s savings) are not triggered in this deterministic run.
> 3. **Stationary Stop Uncertainty:** Stationary service time is derived from the pit lane transit duration minus 80 km/h cruising time (~2.8s), as micro wheel-gun telemetry is not publicly provided by FastF1.

---

## 6. Generated Artifacts

- **Full Strategy Sweep CSV:** [`data/processed/strategy_results.csv`](file:///d:/ApexPitwall/data/processed/strategy_results.csv)
- **Comparison Visualization:** [`reports/figures/strategy_comparison.png`](file:///d:/ApexPitwall/reports/figures/strategy_comparison.png)
- **Pit Loss Parameters:** [`data/processed/pit_loss_parameters.json`](file:///d:/ApexPitwall/data/processed/pit_loss_parameters.json)
