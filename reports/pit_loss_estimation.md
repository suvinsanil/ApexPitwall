# ApexPitwall — Pit Loss Parameter Estimation Report

**Project:** ApexPitwall — ML-Powered Formula 1 Pit-Stop Strategy Prediction & Simulation  
**Document Type:** Empirical Pit Timing & Loss Audit  
**Target Circuit:** Circuit de Barcelona-Catalunya (2022 Spanish Grand Prix)  
**Artifact Path:** [`data/processed/pit_loss_parameters.json`](file:///d:/ApexPitwall/data/processed/pit_loss_parameters.json)  
**Execution Timestamp:** 2026-10-25 23:45 UTC  
**Environment:** Windows 10/11, Python 3.10.11, fastf1 3.9.0, pandas 2.3.3  

---

## 1. Executive Summary & Recommended Parameters

Pit loss represents the total race time cost incurred by entering the pit lane, driving under the 80 km/h pit speed limiter, remaining stationary for tyre replacement, and rejoining the circuit at racing speed, relative to staying out on a clean flying lap.

### Recommended Simulation Parameters (Catalunya)

| Parameter | Recommended Value | Estimation Method | Status |
| :--- | :---: | :--- | :--- |
| **Total Pit Loss ($T_{\text{pit\_loss}}$)** | **22.59 s** | Empirical median delta over local flying reference pace | **EMPIRICALLY MEASURED** |
| **Pit Lane Transit Duration** | **22.39 s** | Timing loop interval: `PitOutTime - PitInTime` | **EMPIRICALLY MEASURED** |
| **In-Lap Loss (Sector 3 Entry)** | **4.80 s** | In-lap time minus driver local reference pace | **EMPIRICALLY MEASURED** |
| **Out-Lap Loss (Sector 1 Exit)** | **17.78 s** | Out-lap time minus driver local reference pace | **EMPIRICALLY MEASURED** |
| **Stationary Service Time** | **2.80 s** (2.2–3.5 s range) | Physical transit deduction (pit distance / 80 km/h) | **ASSUMPTION / INFERRED** |

> [!IMPORTANT]
> `PitOutTime - PitInTime` measures **Total Pit Lane Transit Time** (~22.4 seconds), NOT stationary tyre-change time. 
> Presenting 22.4 seconds as "stationary stop time" is a common error in amateur F1 data analysis. Stationary stop time is physically ~2.5–3.2 seconds.

---

## 2. Methodology & Mathematical Definition

### 2.1 The Two-Lap Stint Interface
At Catalunya, pit lane entry begins in the final sector (Sector 3), and pit lane exit merges into the circuit at Turn 1 (Sector 1). Sector 2 is entirely on-track racing on both the in-lap ($k$) and out-lap ($k+1$).

Total pit loss is measured directly by comparing the combined time of laps $k$ and $k+1$ against a driver's local flying reference pace $T_{\text{ref}}$ (average of lap $k-1$ and lap $k+2$):
$$\Delta T_{\text{pit\_loss}} = [T_{\text{lap}}(k) + T_{\text{lap}}(k+1)] - 2 \times T_{\text{ref}}$$

### 2.2 Sector Timing Breakdown (Catalunya)

| Timing Split | Clean Flying Lap (Median) | In-Lap $k$ (Median) | Out-Lap $k+1$ (Median) | Pit Induced Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Sector 1** | 24.55 s | 24.71 s | 43.57 s | **+19.02 s** (Stationary box + exit limiter) |
| **Sector 2** | 33.66 s | 34.04 s | 33.15 s | **-0.51 s** (Racing speed on fresh tyres) |
| **Sector 3** | 30.64 s | 34.68 s | 30.12 s | **+4.04 s** (Deceleration & entry limiter) |
| **Total Lap Time** | **88.92 s** | **93.45 s** | **107.12 s** | **+22.73 s combined delta** |

---

## 3. Empirical Distribution Statistics

### Sample Size & Filter Criteria
- Total Matched Pit Stops in 2022 Spanish GP: **53**
- Normal Green-Flag Pit Stops (excluding lap 1 incident stops): **51**

### Summary Statistics Table

| Metric | Sample ($n$) | Mean | Median | Std Dev | IQR (25% – 75%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pit Lane Transit Time** | 51 | 22.68 s | **22.39 s** | 1.03 s | 22.04 s – 23.09 s |
| **Total Pit Loss** | 51 | 23.13 s | **22.59 s** | 2.50 s | 21.86 s – 23.39 s |
| **In-Lap Delta** | 51 | 4.95 s | **4.80 s** | 1.70 s | 4.41 s – 5.17 s |
| **Out-Lap Delta** | 51 | 18.18 s | **17.78 s** | 1.38 s | 17.32 s – 18.90 s |

---

## 4. Limitations & Data Boundaries

1. **Stationary Stop Duration:** Standard FastF1 public feeds do not log wheel-gun trigger or jack release telemetry. Thus, stationary stop duration (~2.8 s) is a derived physical inference, not a direct micro-sensor measurement.
2. **Traffic Confounding:** Out-lap times can be prolonged by blue flags or traffic immediately after exiting the pit lane. The median metric (22.59 s) filters these right-tail delays effectively.
3. **Double Counting Prevention:** Clean laps in `data/processed/race_laps_processed.csv` explicitly filter out both in-laps and out-laps. Therefore, the ML models predict clean flying lap times, and the simulator must add exactly $T_{\text{pit\_loss}}$ once per pit stop.
