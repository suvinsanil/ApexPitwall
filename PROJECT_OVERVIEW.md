# ApexPitwall: Technical Blueprint & Architecture

## ML-Powered Formula 1 Pit-Stop Strategy Prediction & Simulation

ApexPitwall is a modular, local-first machine learning and simulation platform designed to analyse historical Formula 1 race data, model lap-time pace and tyre degradation, and simulate alternative pit-stop strategies under counterfactual and stochastic conditions.

---

## 1. Core Architecture: The Five Stages

The platform is organized into five sequential, decoupled stages:

```
┌────────────────────────────────────────────────────────┐
│ Stage A: Lap-Time Prediction (GBDT vs Linear Baseline) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage B: Tyre Degradation Analysis & Decomposition     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage C: Deterministic Strategy Simulator (1 vs 2 Stop)│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage D: Counterfactual Scenario Evaluator (Deltas)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage E: Monte Carlo Uncertainty Quantification        │
└────────────────────────────────────────────────────────┘
```

### Stage A: Lap-Time Prediction Engine
* **Objective:** Predict clean racing lap time $y_t \in \mathbb{R}^+$ using only information observed prior to lap $t$.
* **Models:** 
  - *Baseline:* Driver-specific 3-lap rolling median with linear offsets.
  - *Interpretable:* Standardized Ridge Regression.
  - *Primary:* LightGBM Regressor (handling non-linear interactions and tabular features).
* **Leakage Prevention:** All rolling pace and stint statistics must apply `.shift(1)` before computing `.rolling()` over clean laps only.

### Stage B: Tyre Degradation Decomposition
* **Objective:** Isolate the marginal pace penalty per lap of tyre age ($\partial \text{Pace} / \partial \text{TyreAge}$) separated by compound (`SOFT`, `MEDIUM`, `HARD`).
* **Confounder Disentanglement:** Cars shed up to 110 kg of fuel over a race (~1.6 kg/lap), gaining ~0.035 s/lap. Without subtracting this mechanistic fuel mass effect, empirical tyre degradation curves exhibit false negative slopes.

### Stage C: Deterministic Strategy Simulation
* **Objective:** Evaluate total elapsed race time for candidate pit strategies across a full race distance (e.g., 66 laps at Catalunya):
  $$T_{\text{race}}(S) = \sum_{t=1}^{N} \hat{y}_t(S) + N_{\text{stops}} \times L_{\text{pit}}$$
* **Rules & Constraints:** Enforce mandatory two-compound regulation, compound failure stint limits, and empirical pit loss penalty ($L_{\text{pit}} = 22.59\text{ s}$ at Catalunya).

### Stage D: Counterfactual Analysis Engine
* **Objective:** Quantify the time delta of strategic perturbations:
  $$\Delta T_{\text{cf}} = T_{\text{sim}}(\text{Observed Strategy}) - T_{\text{sim}}(\text{Counterfactual Strategy})$$
* **Scenarios:** Pit window shifts (pitting $k$ laps earlier/later), compound substitutions (Hard vs. Medium), stint count switches (1-stop vs. 2-stop), and Virtual Safety Car (VSC) exploitation.

### Stage E: Monte Carlo Uncertainty Quantification
* **Objective:** Generate empirical outcome distributions across stochastic racing conditions:
  - Base driver pace noise via Autoregressive $\text{AR}(1)$ process ($\sigma_{\text{pace}} \approx 0.25\text{s}, \rho = 0.4$).
  - Stochastic pit lane transit and Gamma-distributed stationary stop times.
  - Win probability estimation: $P(T_{\text{race}}(S_1) < T_{\text{race}}(S_2))$ and 90% confidence intervals.

---

## 2. Data Model & Grain

Data is collected from [FastF1](https://docs.fastf1.dev/) and structured into five logical tables:

| Table | Grain | Primary Key | Description |
| :--- | :--- | :--- | :--- |
| **`race_laps`** | One row per driver per lap | `(session_key, driver_id, lap_number)` | Timing, sector splits, track status, and clean lap eligibility flags. |
| **`tyre_stints`** | One row per driver stint | `(session_key, driver_id, stint_number)` | Compound type, stint length, and tyre age at stint start. |
| **`pit_stops`** | One row per pit event | `(session_key, driver_id, pit_stop_number)` | In/out-lap times, pit lane duration, and time loss vs. reference pace. |
| **`weather`** | One row per sensor reading | `(session_key, timestamp_utc)` | Track temp, air temp, humidity, and rainfall flags. Merged onto laps via backward as-of join. |
| **`race_context`** | One row per driver per lap | `(session_key, driver_id, lap_number)` | State at start of lap: gap to ahead, dirty air flag ($\le 1.5\text{s}$), and remaining fuel mass. |

---

## 3. Data Cleaning & Lap Eligibility Rules

To prevent noise from corrupting pace and degradation models, laps must satisfy strict filtering criteria:
1. **No In-Laps / Out-Laps:** `PitInTime` and `PitOutTime` must be null.
2. **Green Flag Only:** FastF1 `TrackStatus` must not contain non-green codes (`2` = Yellow, `4` = Safety Car, `5` = Red Flag, `6`/`7` = Virtual Safety Car).
3. **Valid Slicks:** Compound must be `SOFT`, `MEDIUM`, or `HARD` in dry conditions (`Rainfall == False`).
4. **Pace Outliers:** Exclude laps exceeding 115% of the race median clean lap time.
5. **Start Lap Excluded:** Lap 1 (standing start) is excluded from flying pace regression.

---

## 4. Empirical Evaluation Benchmark

Trained across four diverse 2022 Grands Prix (Bahrain, Zandvoort, Mexico City, Abu Dhabi) and evaluated strictly out-of-sample on the held-out **2022 Spanish Grand Prix (Catalunya, 1,060 clean laps)**:

| Model Architecture | Split & Calibration Strategy | Held-Out MAE (s) | Held-Out RMSE (s) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Naive Baseline** | 3-Lap Shifted Rolling Mean ($t-1$) | **0.5906 s** | **0.8509 s** | 0.6475 |
| **Ridge Regression** | Cross-Circuit Held-Out | **1.0189 s** | **1.2070 s** | 0.2907 |
| **LightGBM Regressor** | Cross-Circuit Held-Out | **3.0252 s** | **3.1822 s** | -3.9305 |
| **Ridge Regression** | Within-Circuit 5-Fold Cross Validation | **0.5520 s** | **0.7512 s** | 0.7265 |
| **LightGBM Regressor** | Within-Circuit 5-Fold Cross Validation | **0.4633 s** | **0.6710 s** | **0.7779** |

---

## 5. Scope Boundaries & Assumptions

* **Out of Scope:** Deep neural networks, continuous micro-physics/CFD, multi-agent collision/overtaking physics, and live telemetry streaming during active race weekends.
* **Fuel Assumption:** Standard 110 kg regulation capacity burned linearly to zero at the chequered flag.
* **Pit Loss Measurement:** Catalunya median green-flag pit loss is empirically measured at **22.59 seconds** (22.39s pit lane transit duration; ~2.8s inferred stationary service time).
* **Strategy Finding:** At Catalunya, high lateral energy and 49°C track temperature make a **2-stop Soft $\to$ Medium $\to$ Soft** (pits on laps 18, 40) **32.8s to 44.3s faster** than 1-stop options.
