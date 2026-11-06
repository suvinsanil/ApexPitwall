# ApexPitwall: Technical Blueprint & Project Overview
## ML-Powered Formula 1 Pit-Stop Strategy Prediction & Simulation

---

## 1. Project Summary and Scope (Including Explicit Exclusions)

### 1.1 Project Summary and Vision
ApexPitwall is a reproducible, modular, local-first machine learning and simulation platform designed to analyse historical Formula 1 race data, model lap-time pace and tyre degradation, and simulate alternative pit-stop strategies. The system allows an engineer or analyst to systematically investigate counterfactual race strategies under varying degradation rates, pit windows, and race contexts.

The platform directly answers eight foundational race engineering questions:
1. **Tyre Age Impact:** How does cumulative tyre mileage (measured in lap age and stint wear) affect lap time across different compounds?
2. **Compound Delta:** What are the baseline pace offsets and degradation slopes between Soft, Medium, and Hard tyre compounds at a given circuit?
3. **Pace Prediction:** How accurately can a machine learning model forecast a driver's subsequent lap time using only pre-lap observable information?
4. **Pit Window Optimization:** At what lap threshold does tyre degradation make a fresh set of tyres faster over remaining race distance than continuing on worn tyres?
5. **Stint Strategy Comparison:** Under specified pace, degradation, and track evolution assumptions, is a 1-stop strategy faster than a 2-stop strategy?
6. **Sensitivity Analysis:** How sensitive is the optimal pit lap recommendation to variations in pit-lane time loss (e.g., green flag vs. Virtual Safety Car) or anomalous tyre wear?
7. **Uncertainty Quantification:** What is the confidence interval or empirical distribution of total race time across simulated stochastic pace variations?
8. **Empirical Evidence & Limits:** What observable data supports a strategic recommendation, and what unobserved variables (e.g., dirty air, traffic, track temperature swings) remain unmodelled?

### 1.2 Core Technical Distinction: Prediction vs. Counterfactual Simulation
A foundational architectural principle of ApexPitwall is the strict operational and theoretical separation between two fundamentally distinct tasks:
* **Task 1: Predicting Observed Outcomes (Supervised ML Regression)**
  * *Goal:* Estimate the actual recorded lap time $y_t$ on lap $t$ for a historical car/driver given state $X_{<t}$.
  * *Constraint:* Trained exclusively on observational data where driver behavior, engine modes, fuel loads, and traffic existed as recorded.
  * *Failure Mode:* Conflating observational correlations (e.g., driving slowly while managing tyres) with physical capability.
* **Task 2: Simulating Counterfactual Outcomes (Generative / Mechanistic Scenario Modelling)**
  * *Goal:* Estimate what *would have happened* if a driver pitted on lap $k$ instead of lap $m$, or switched to Hard instead of Medium.
  * *Constraint:* Requires explicit causal and mechanistic assumptions (e.g., fuel burn correction, baseline pit-lane stationary and travel penalty, out-lap warm-up penalty, and clean-air pace projections).
  * *Failure Mode:* Naively feeding hypothetical inputs into a purely empirical regression model without accounting for unobserved confounding, competitor interactions, or domain constraints.

### 1.3 Explicit Scope Exclusions
To maintain portfolio-grade execution within a 8–10 hour/week solo workload on Windows CPU hardware, the following areas are **explicitly out of scope**:
* **Deep Learning & Reinforcement Learning:** No neural networks, LSTM/Transformers, or multi-agent RL (e.g., PPO self-play). Classical regression and gradient boosted decision trees (GBDT) are CPU-efficient, interpretable, and superior on tabular F1 timing data.
* **Live In-Race Telemetry Streaming:** No live socket feeds or sub-second real-time inference during live race weekends. The pipeline is an offline analytical engine evaluating completed sessions.
* **Continuous Vehicle Dynamics & Micro-Physics Engine:** No computational fluid dynamics (CFD), tyre slip-angle ODEs, or millisecond suspension kinematics. Pace and degradation are modelled at the lap/sector grain.
* **Multi-Agent Traffic & Collision Simulation:** No physical collision modelling or spatial overtaking simulators. Traffic is represented as an empirical lap-time penalty or probability distribution rather than a deterministic agent-based track simulation.
* **Cloud Infrastructure & Paid Services:** Zero paid cloud databases, GPU instances, or commercial data APIs.

---

## 2. Data Sources and Data-Inspection Checklist

### 2.1 Primary Source: FastF1
* **Documentation:** [FastF1 Documentation](https://docs.fastf1.dev/)
* **License & Access:** Open source (MIT License). Pulls timing and telemetry data directly from the official F1 live timing service (ergast / F1 core feeds). Free of charge.
* **Caching & Rate Limits:** FastF1 features built-in disk caching (`fastf1.Cache.enable_cache('path/to/cache')`). Caching is mandatory to avoid repeat HTTP requests and IP-level throttling from F1 data servers.
* **Primary Capabilities:**
  * Lap times, sector times (Sector 1, 2, 3), speed traps.
  * Driver information, team identifiers, tyre compound labels (`SOFT`, `MEDIUM`, `HARD`, `INTERMEDIATE`, `WET`).
  * Stint numbers and cumulative tyre age (in laps).
  * Pit in/out flags and lap-level pit indicators.
  * High-frequency telemetry (car speed, RPM, throttle, braking, DRS) via `lap.get_telemetry()`.
  * Weather packets recorded at trackside stations (air temp, track temp, humidity, pressure, wind speed/direction, rainfall flag).
  * Race control messages (Safety Car deployments, VSC, Yellow flags, Red flags).

### 2.2 Supplementary Source: OpenF1
* **Documentation:** [OpenF1 Documentation](https://openf1.org/docs/)
* **License & Access:** Free, open-source REST/JSON API.
* **Role in Project:** Strictly secondary fallback. FastF1 handles 95%+ of core requirements. OpenF1 is consulted only if FastF1 fails to parse specific pit-stop duration sub-components (such as stationary stop duration vs. pit lane transit duration) or session intervals in modern (2023+) races.
* **Alignment Rules:** OpenF1 uses UTC timestamps (`date`) and distinct `meeting_key` / `session_key` integers. If combined, records must be merged on standardized UTC lap start/end intervals and verified driver racing numbers, never on raw string names.

### 2.3 Data Availability and Historical Coverage Verification
*Note: All data coverage and field availability across historical seasons must be treated as **TO VERIFY** until Phase 1 data inspection explicitly confirms presence.*

| Data Field Category | 2018–2021 | 2022–2024 | Status / Verification Note |
| :--- | :--- | :--- | :--- |
| **Lap & Sector Timings** | TO VERIFY | TO VERIFY | TO VERIFY: check timing consistency, format, and null rates across sessions |
| **Tyre Compound Labels** | TO VERIFY | TO VERIFY | TO VERIFY: check presence of dry/wet compound labels and sub-grades (C1–C5) |
| **Tyre Age & Stint Boundaries** | TO VERIFY | TO VERIFY | TO VERIFY: check reliability for race sessions and practice stint carryover |
| **Pit Lane Total Duration** | TO VERIFY | TO VERIFY | TO VERIFY: check calculation via in-lap to out-lap delta vs. dedicated pit timing channels |
| **Stationary Pit Stop Duration** | TO VERIFY | TO VERIFY | TO VERIFY: check availability via FastF1 `pit_stops` or OpenF1 API |
| **Weather (Track & Air Temp)** | TO VERIFY | TO VERIFY | TO VERIFY: check recording frequency (~1 min intervals) and lap alignment |
| **Race Control Messages** | TO VERIFY | TO VERIFY | TO VERIFY: check SC / VSC flags and exact lap-fraction alignment |
| **Car Telemetry (Throttle/Brake)**| TO VERIFY | TO VERIFY | TO VERIFY: check volume, sync, and completeness; CPU intensive, deferred from MVP |

### 2.4 Race-Selection Rule for Initial Pipeline Validation
To ensure high data quality and avoid unrepresentative edge cases, the initial modeling dataset must adhere to the following selection criteria:
1. **Dry-Weather Only:** Exclude races with rain (`Rainfall == True`). Rain forces non-linear crossover windows (Intermediates/Wets) that distort dry-slick degradation modeling.
2. **Standard 2022–2023 Regulations:** Focus on the ground-effect regulatory era (2022 onwards) to ensure consistent car weight, 18-inch tyre characteristics, and aerodynamic baselines.
3. **Low Incident Disruption:** Choose races with minimal Safety Car / Red Flag interruptions so the majority of laps reflect green-flag racing pace.
4. **Representative Circuit Layouts:**
   * **Circuit de Barcelona-Catalunya (Spanish GP):** High-degradation, high-energy lateral tyre stress circuit. Candidate for distinguishing thermal degradation between C1, C2, and C3 compounds.
   * **Bahrain International Circuit (Sakhir):** High-abrasion rear tyre degradation track with multiple viable 1-stop vs. 2-stop strategic battles.
5. **Initial Pilot Candidate:** **2023 Spanish Grand Prix (Barcelona)** (TO VERIFY: weather conditions, incident history, and data completeness must be audited during Phase 1 inspection before confirming; it is a pilot candidate, not a confirmed choice).

---

## 3. Data Tables, Grain, and Join Keys

```
                    ┌───────────────────────────┐
                    │       weather             │
                    │ (grain: timestamp_utc)    │
                    └─────────────┬─────────────┘
                                  │ (asof join on lap start timestamp)
                                  ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│       tyre_stints         │   │        race_laps          │
│ (grain: session, drv, st) │   │ (grain: session, drv, lap)│
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │
              │ (join on session, drv, stint) │
              └───────────────┬───────────────┘
                              │
                              ▼
                    ┌───────────────────────────┐   ┌───────────────────────────┐
                    │       race_context        │◄──┤        pit_stops          │
                    │(grain: session, drv, lap) │   │ (grain: session, drv, pit)│
                    └───────────────────────────┘   └───────────────────────────┘
```

### 3.1 Logical Table Specifications

#### A. Table: `race_laps`
* **Grain:** One row per driver per lap in an official session.
* **Natural / Primary Key:** `(session_key, driver_id, lap_number)`
* **Foreign / Join Keys:** `(session_key, driver_id)`, `(session_key, event_key)`
* **Schema:**
  * `session_key` (string / int): Unique identifier for the Grand Prix session (e.g., `"2023_ESP_R"`).
  * `driver_id` (string): Stable 3-letter driver code (e.g., `"VER"`, `"HAM"`, `"ALO"`). *Never join on driver full name.*
  * `constructor_id` (string): Standardized team identifier (e.g., `"Red Bull Racing"`, `"Mercedes"`).
  * `lap_number` (int16): Lap sequence number (1 to total laps).
  * `lap_time_seconds` (float32): Total lap duration in seconds.
  * `sector_1_seconds`, `sector_2_seconds`, `sector_3_seconds` (float32): Individual sector elapsed times.
  * `lap_start_time_utc` (datetime64[ns, UTC]): Absolute timestamp at the start of the lap.
  * `is_pit_out_lap` (bool): True if the lap started from the pit lane.
  * `is_pit_in_lap` (bool): True if the lap concluded by entering the pit lane.
  * `is_valid_track_limits` (bool): True if lap was deleted or compromised by track limits.
  * `track_status` (string): FastF1 track status flag. Must be evaluated for combined status codes (e.g., "1", "12", "24"); green-flag status requires the absence of SC, VSC, Red, or Yellow codes (TO VERIFY exact code meanings in FastF1 documentation).
* **Missing-Value Strategy:**
  * Incomplete laps (crash, mechanical DNF) have `lap_time_seconds = NaN`. Drop these records from regression training; retain as termination flags in driver race histories.
  * Missing individual sectors with valid total lap time: Impute via median sector ratio for that driver-session, or flag `is_clean_lap = False`.
* **Data-Quality Caveats:** The first lap of the race has no standing start-to-finish line benchmark lap time; out-laps and in-laps carry massive artificial time deltas (initial assumption: 15–25 seconds, to be estimated from data) and must be tagged and isolated.

#### B. Table: `tyre_stints`
* **Grain:** One row per continuous tyre stint for a driver in a session.
* **Natural / Primary Key:** `(session_key, driver_id, stint_number)`
* **Foreign / Join Keys:** `(session_key, driver_id)`
* **Schema:**
  * `session_key` (string / int): Session identifier.
  * `driver_id` (string): Driver code.
  * `stint_number` (int8): 1-indexed sequential stint order for the race.
  * `compound` (category): Normalised compound label (`"SOFT"`, `"MEDIUM"`, `"HARD"`).
  * `stint_start_lap` (int16): First lap number on this tyre set.
  * `stint_end_lap` (int16): Final lap number before pit entry or race conclusion.
  * `stint_length` (int16): Total laps completed during this stint (`end_lap - start_lap + 1`).
  * `tyre_age_at_start` (int16): Cumulative laps previously run on this specific tyre set during qualifying/practice (0 if brand new sticker tyre).
  * `is_new_tyre` (bool): Observed indicator whether the tyre was scrubbed or new at stint start (**TO VERIFY** consistency across seasons).
* **Boundary Reconstruction Logic:** Stints are reconstructed whenever `is_pit_out_lap` is flagged or when FastF1 records a change in `Stint` index. Stint lap age on lap $t$ is defined as:
  $$\text{tyre\_age}_t = \text{tyre\_age\_at\_start} + (t - \text{stint\_start\_lap})$$
* **Missing-Value Strategy:** If `tyre_age_at_start` is unrecorded in historical sessions, assume 0 for race start unless flagged as used in qualifying Q2/Q3.

#### C. Table: `pit_stops`
* **Grain:** One row per pit-stop event conducted by a car.
* **Natural / Primary Key:** `(session_key, driver_id, pit_stop_number)`
* **Foreign / Join Keys:** `(session_key, driver_id)`, `(session_key, driver_id, pit_lap)`
* **Schema:**
  * `session_key` (string / int): Session identifier.
  * `driver_id` (string): Driver code.
  * `pit_stop_number` (int8): Sequential stop count for that car.
  * `pit_lap` (int16): Lap number on which the car entered the pit lane.
  * `pit_entry_time_utc` (datetime64[ns, UTC]): Recorded timestamp crossing pit-entry beam.
  * `pit_exit_time_utc` (datetime64[ns, UTC]): Recorded timestamp crossing pit-exit beam.
  * `pit_lane_duration_seconds` (float32): Elapsed duration traversing the entire pit lane speed-limit zone.
  * `stationary_duration_seconds` (float32): Duration the car was motionless on the wheel jacks (**TO VERIFY** across FastF1 sessions).
  * `pit_lane_time_loss_estimated` (float32): Estimated net time penalty compared to running a normal racing lap at green flag pace.
  * `safety_car_context` (category): `"GREEN"`, `"VSC"`, `"FULL_SC"`.
* **Important Operational Distinction:**
  * *Stationary Duration* ($\approx 2.0 - 3.5 \text{ seconds}$; initial assumption, to be estimated from data): Wheel-gun, jack, and tyre swap duration only.
  * *Pit Lane Total Duration* ($\approx 18.0 - 24.0 \text{ seconds}$; initial assumption, to be estimated from data): Pit limiter deceleration, traverse, stop, acceleration, release.
  * *Pit Lane Time Loss (Net Delta)* ($\approx 16.0 - 22.0 \text{ seconds}$ under green flag; $\approx 9.0 - 14.0 \text{ seconds}$ under VSC/SC; initial assumptions, to be estimated from data): The actual loss relative to a racing competitor remaining on track at racing speed.

#### D. Table: `weather`
* **Grain:** One row per physical sensor observation at trackside.
* **Natural / Primary Key:** `(session_key, timestamp_utc)`
* **Foreign / Join Keys:** `session_key`
* **Schema:**
  * `session_key` (string / int): Session identifier.
  * `timestamp_utc` (datetime64[ns, UTC]): Observation recording time.
  * `air_temperature` (float32): Ambient temperature in Celsius.
  * `track_temperature` (float32): Track tarmac temperature in Celsius.
  * `humidity` (float32): Relative humidity percentage.
  * `wind_speed` (float32): Wind velocity in m/s.
  * `wind_direction` (int16): Wind angle in degrees (0–360).
  * `rainfall` (bool): Binary precipitation flag.
* **Lap Alignment Procedure:** Weather observations arrive asynchronously at approximately 1-minute intervals, whereas laps occur every 70–100 seconds. Laps must be aligned using an **exact as-of merge (backward/nearest preceding timestamp)**:
  $$\text{LapWeather}_t = \max_{t_{\text{obs}} \le t_{\text{lap\_start}}} \text{WeatherObservation}(t_{\text{obs}})$$
  Never look ahead into future timestamps. If weather records are sparser than 5 minutes, linear interpolation between preceding timestamps is permitted, provided the interpolation boundary strictly terminates before $t_{\text{lap\_start}}$.

#### E. Table: `race_context`
* **Grain:** One row per driver per lap evaluated at lap start.
* **Natural / Primary Key:** `(session_key, driver_id, lap_number)`
* **Foreign / Join Keys:** `(session_key, driver_id, lap_number)`
* **Schema:**
  * `session_key`, `driver_id`, `lap_number`: Identifying keys.
  * `position` (int8): Track position at the completion of lap $t-1$.
  * `gap_to_ahead_seconds` (float32): Time gap to the immediate car ahead.
  * `gap_to_leader_seconds` (float32): Elapsed time behind the race leader.
  * `laps_remaining` (int16): Total scheduled race laps minus $(t - 1)$.
  * `active_race_control_status` (category): `"GREEN"`, `"YELLOW"`, `"VSC"`, `"SAFETY_CAR"`, `"RED_FLAG"`.
  * `is_in_dirty_air` (bool): Derived boolean flag: `True` if `gap_to_ahead_seconds <= 1.5` seconds (1.5 s threshold is an initial assumption, to be estimated from data).
  * `estimated_fuel_mass_kg` (float32): Mechanistic proxy calculated as:
    $$\text{Fuel}_t = \text{InitialFuelMass} \times \left(1.0 - \frac{t - 1}{\text{TotalRaceLaps}}\right)$$
    where $\text{InitialFuelMass}$ is the maximum race fuel allowance for the relevant regulations (TO VERIFY; do not hardcode until checked).

---

## 4. Stage-by-Stage Machine Learning Plan

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

---

### Stage A: Lap-Time Prediction Engine

#### 1. Objective and Target Variable
* **Objective:** Predict a driver's lap time $y_t$ (in seconds) on lap $t$ using exclusively historical information generated before lap $t$ begins.
* **Target Variable:** `lap_time_seconds` $\in \mathbb{R}^+$ for eligible clean laps (filtering out pit in/out laps, safety car periods, and severe red flag anomalies).

#### 2. Unit of Observation
One observation = one completed clean racing lap for a specific driver in a specific Grand Prix session.

#### 3. Candidate Features
* `compound` (one-hot/categorical: Soft, Medium, Hard).
* `tyre_age` (cumulative laps run on current set).
* `stint_length_so_far` (laps run in this race stint).
* `fuel_proxy_lap_fraction` ($(t-1)/\text{TotalLaps}$).
* `driver_pace_rolling_3` (exponential or rolling mean of driver's last 3 clean laps).
* `driver_pace_last_lap_delta` ($y_{t-1} - y_{t-2}$).
* `circuit_id` (target-encoded or categorical).
* `track_temperature`, `air_temperature`.
* `is_in_dirty_air` (binary indicator if within 1.5s of car ahead on lap $t-1$; 1.5s threshold is an initial assumption, to be estimated from data).

#### 4. Recommended Method and Simpler Alternative
* **Simpler Baseline:** Driver-specific rolling median of the last 3 clean laps, adjusted by a global linear compound offset:
  $$\hat{y}_t = \text{Median}(y_{t-3:t-1}) + \delta_{\text{fuel}} + \delta_{\text{wear}}$$
* **Interpretable Model:** Ridge / ElasticNet Regularized Linear Regression.
* **Recommended Method:** LightGBM / XGBoost Regressor (Histogram-based GBDT). CPU-friendly, handles tabular non-linear interactions natively, and handles categorical variables directly.

#### 5. Evaluation Metrics
* **Primary Metric:** Mean Absolute Error (MAE) in seconds.
* **Secondary Metric:** Root Mean Squared Error (RMSE) in seconds (penalizes large outlier mistakes).
* **Evaluation Segmentation:** MAE segmented by compound (`SOFT`, `MEDIUM`, `HARD`), by stint progression quartiles, and by track temperature bins.

#### 6. Main Risks and Limitations
* *Confounding:* A driver lifting and coasting to conserve fuel or managing tyre thermal limits will look like an inherently "degraded tyre" to the regression model.
* *Outlier Distortion:* Laps with driver mistakes (lockups, running wide) inflate error metrics if not flagged as anomalous.

#### 7. Leakage-Prevention Rules
* Never include lap $t$ sector times, lap $t$ speed traps, lap $t$ pit decisions, or any post-lap telemetry.
* Rolling pace aggregations must be computed with `.shift(1)` before computing `.rolling()` over eligible clean laps only.

#### 8. Measurable Definition of Done
The model is evaluated on held-out races using both within-circuit and cross-circuit splits. Rather than assuming an arbitrary metric threshold, the rule is: **Measure baseline MAE and RMSE first on held-out races, then set a target relative to that baseline and record it before tuning.** The model is considered done when it systematically outperforms the baseline according to this recorded target, with residual errors analyzed and documented.

#### 9. What I Should Be Able to Explain in an Interview
1. Why rolling features must be explicitly shifted to avoid lookahead leakage.
2. How fuel-load burn-off (car gets lighter with fuel consumption per lap, gaining pace per lap; values such as $\approx 1.6\text{ kg/lap}$ and $\approx 0.05-0.07\text{ s/lap}$ are initial assumptions, to be estimated from data or verified from technical regulations) masks tyre degradation, and how the model isolates these two opposing forces.
3. Why tree-based GBDTs cannot extrapolate beyond observed tyre age limits without domain-guided clipping or linear fallback.

---

### Stage B: Tyre Degradation Analysis and Decomposition

#### 1. Objective and Target Variable
* **Objective:** Quantify the isolated marginal lap time penalty per additional lap of tyre age:
  $$\frac{\partial \text{Pace}}{\partial \text{TyreAge}}$$
  separated by compound and circuit.
* **Target Variable:** Pace degradation component $\Delta y_{\text{deg}}(a, c)$, where $a$ is tyre age and $c$ is compound.

#### 2. Unit of Observation
Aggregated stint-level observations and lap-level residuals after adjusting for baseline driver pace and fuel burn.

#### 3. Candidate Features
* `compound` ($c \in \{\text{Soft}, \text{Medium}, \text{Hard}\}$).
* `tyre_age` ($a$).
* `track_temperature` ($T_{\text{track}}$).
* `accumulated_lateral_energy_proxy` (cumulative high-speed turn exposure).

#### 4. Recommended Method vs. Alternative
* **Approach 1 (Direct Feature Attribution via Stage A GBDT):** Use Partial Dependence Plots (PDP) or SHAP values from the Stage A regression model on `tyre_age` across compounds.
* **Approach 2 (Recommended for MVP: Mechanistic Two-Stage Decomposition):**
  1. Remove mechanistic fuel effect from observed lap time:
     $$y^*_t = y_t - (\text{fuel\_effect\_per\_kg} \times \text{fuel\_mass\_remaining\_kg}_t)$$
     where $\text{fuel\_effect\_per\_kg}$ is an initial assumption, to be estimated from data or verified from engineering literature. Sign convention: Because cars are slower on early laps due to carrying heavy fuel, subtracting $(\text{fuel\_effect\_per\_kg} \times \text{fuel\_mass\_remaining\_kg})$ removes the fuel penalty, normalizing early laps to an equivalent light-fuel pace benchmark.
  2. Fit a robust compound-specific polynomial or segmented linear regression:
     $$y^*_t = \alpha_{\text{driver}} + \beta_{1, c} \cdot a_t + \beta_{2, c} \cdot a_t^2 + \epsilon_t$$
* **Trade-off Decision:** Approach 2 is recommended for Phase 1/2 because it prevents the ML tree model from overfitting to erratic driver management styles and enforces interpretable wear profiles.

#### 5. Evaluation Metrics
* $R^2$ of the degradation curve on fuel-corrected clean stints.
* Monotonicity and residual stability checks across operational tyre life.

#### 6. Main Risks and Limitations
* *Tyre Management vs. Degradation:* Drivers often drive below the tyre's limit early in a stint to protect against overheating (thermal degradation vs. mechanical wear).
* *Track Evolution Confounding:* As more rubber is laid down on the track surface throughout a Grand Prix, baseline grip improves (initial assumption: $0.5 - 1.5\text{ seconds}$, to be estimated from data), offsetting tyre wear.

#### 7. Leakage-Prevention Rules
* Calculate driver fixed effects ($\alpha_{\text{driver}}$) on training folds only; do not demean stints using full-race driver means that include future stints.

#### 8. Measurable Definition of Done
Report fitted degradation slopes per compound with uncertainty intervals and residual error, whatever their ordering, and discuss any result that contradicts expectations.

#### 9. What I Should Be Able to Explain in an Interview
1. The mathematical collinearity between race progress, fuel consumption, and tyre age, and why unconstrained models attribute fuel benefits to tyre resilience.
2. The difference between thermal degradation (reversible via cooling) and physical wear (irreversible rubber loss).
3. Why linear degradation is an approximation that breaks down when tyres "fall off the cliff" (exponential non-linear phase).

---

### Stage C: Deterministic Strategy Simulation

#### 1. Objective and Target Variable
* **Objective:** Compare candidate pit-stop strategies (combinations of pit laps and tyre compounds) to determine the strategy that minimizes the total estimated race completion time $T_{\text{race}}$ from lap $k$ to the finish.
* **Target Variable:** Total Elapsed Race Time $T_{\text{race}}(S)$ for strategy candidate $S$:
  $$T_{\text{race}}(S) = \sum_{t=k}^{N} \hat{y}_t(S) + \sum_{p \in P_S} L_{\text{pit}}(p)$$
  where $\hat{y}_t(S)$ is the predicted pace on lap $t$ under strategy $S$, $P_S$ is the set of pit stop laps in $S$, and $L_{\text{pit}}(p)$ is the net pit lane time penalty.

#### 2. Unit of Observation
One candidate strategy profile evaluated over the remaining race distance for a specific car.

#### 3. Candidate Strategy Rules and Constraints
* **Mandatory Compound Rule:** FIA regulation dictates that in a dry race, a driver must use at least two distinct dry tyre compounds.
* **Maximum Stint Length:** Set hard upper bounds per compound (e.g., Soft $\le 25$ laps, Medium $\le 35$ laps, Hard $\le 45$ laps; initial assumptions, to be estimated from data) based on failure limits.
* **Pit Loss Parameter:** Nominal green-flag pit lane loss (e.g., $L_{\text{pit}} \approx 20.5\text{ seconds}$ for Barcelona; initial assumption, to be estimated from data).

#### 4. Recommended Method and Simpler Alternative
* **Simpler Alternative:** Static spreadsheet model using constant compound offsets and linear lap wear rates.
* **Recommended Method:** Discrete-event deterministic simulator evaluating a bounded grid of feasible strategies:
  * Strategy Space: All permutations of 1-stop and 2-stop configurations:
    * 1-stop: `M -> H`, `H -> M`, `S -> H`, `H -> S` with pit lap $p_1 \in [15, 40]$.
    * 2-stop: `S -> M -> H`, `M -> H -> M`, `S -> H -> H` with $(p_1, p_2) \in [12, 28] \times [32, 50]$ where $p_2 - p_1 \ge 10$.
  * For each strategy, pace on each lap $t$ is calculated sequentially using the Stage B degradation parameters + Stage A baseline pace.

#### 5. Evaluation Metrics
* **Simulated Strategy Regret:** Time difference between the executed historical strategy and the simulator's optimal candidate.
* **Observed Time Prediction Error:** Absolute difference between simulated total race time and actual total race time for the historical winning strategy.

#### 6. Main Risks and Limitations
* *Free Air Assumption:* The simulator initially assumes a car drives in free, undisturbed air. In reality, exiting into a traffic pack behind slower cars destroys the theoretical undercut advantage.
* *Track Evolution Disregard:* Failing to model track grip evolution will bias the simulator toward stopping unrealistically late.

#### 7. Leakage-Prevention Rules
* The simulation starting at lap $k$ must only use model weights, fuel baselines, and degradation slopes estimated prior to lap $k$ (or from previous race weekends).

#### 8. Measurable Definition of Done
The simulator evaluates a bounded grid of feasible 1-stop and 2-stop strategy candidates in under a few seconds of CPU execution time, outputs a ranked leaderboard of strategies by total elapsed time, and accurately flags FIA compound violations.

#### 9. What I Should Be Able to Explain in an Interview
1. The mathematical concept of the "undercut" (pitting early onto fresh rubber to gain lap delta over an opponent on worn tyres) versus the "overcut" (staying out when fresh tyres warm up slowly or grain).
2. Why strategy optimization in F1 cannot be solved purely by greedy lap-by-lap optimization, but requires global trajectory evaluation over the full horizon.
3. How discrete-event grid evaluation compares computationally to dynamic programming or mixed-integer linear programming (MILP).

---

### Stage D: Counterfactual Analysis Engine

#### 1. Objective and Target Variable
* **Objective:** Answer specific "what-if" operational questions by perturbing historical decisions while holding other variables constant.
* **Target Metric:** Counterfactual Time Delta ($\Delta T_{\text{cf}}$):
  $$\Delta T_{\text{cf}} = T_{\text{sim}}(\text{Observed Strategy}) - T_{\text{sim}}(\text{Counterfactual Strategy})$$
  A positive $\Delta T_{\text{cf}}$ indicates the counterfactual strategy was simulated to be faster than the observed real-world choice.

#### 2. Unit of Observation
A pair of comparative strategy simulations (Baseline Historical vs. Perturbed Alternative) for a single driver across identical remaining race laps.

#### 3. Core Counterfactual Scenarios
1. **Pit Window Shift:** What if Driver X pitted 5 laps earlier (lap 18 instead of lap 23)?
2. **Compound Substitution:** What if Driver X fitted Hard tyres on Lap 20 instead of Medium tyres?
3. **Stint Count Switch:** What if a 2-stop runner had converted to a 1-stop strategy (`S-H` instead of `S-M-M`)?
4. **VSC Exploitation:** What if Driver X pitted during an unexpected lap 25 Virtual Safety Car (pit loss = 12s; initial assumption, to be estimated from data) instead of staying out until lap 32 under green flag conditions (pit loss = 21s; initial assumption, to be estimated from data)?

#### 4. Recommended Method
* Mechanistic counterfactual projection: Re-generate the lap-time vector from the perturbation lap $p_{\text{cf}}$ onward, substituting compound degradation curves and adding/subtracting the estimated pit loss delta.
* *Contrast against naive ML:* Explicitly refuse to feed the raw counterfactual feature vector into an unconstrained regression model, because the observational distribution does not include that driver's pace on that compound at that race phase.

#### 5. Evaluation Metrics
* **Sanity Consistency Check:** If the simulated counterfactual is set identically to the historically observed strategy, $\Delta T_{\text{cf}}$ must evaluate to $0.00\text{ seconds}$.
* **Directional Plausibility:** Verifying that pitting onto fresh tyres during a VSC yields an advantage consistent with engineering expectations (~9–11 seconds saved; initial assumption, to be estimated from data).

#### 6. Main Risks and Limitations
* *Competitor Reaction Absence:* In a real race, if Driver A pits early to attempt an undercut, Driver B reacts on the next lap to cover. A single-car counterfactual ignores game-theoretic competitor reactions.
* *Dirty Air Neglect:* A driver pitting 5 laps earlier may rejoin behind a DRS train of midfield cars, losing pace in dirty air (initial assumption: ~1.5s/lap, to be estimated from data)—an effect invisible in free-air simulation.

#### 7. Leakage-Prevention Rules
* Counterfactuals must strictly evaluate forward from the branch point. Historical laps preceding the pit window shift remain immutable.

#### 8. Measurable Definition of Done
The system accepts any completed driver stint history, allows user-defined adjustments to pit lap and compound choices, and generates a structured counterfactual report with an explicit tabular breakdown of assumptions.

#### 9. What I Should Be Able to Explain in an Interview
1. The difference between an observational conditional expectation $\mathbb{E}[Y | X=x]$ and a causal counterfactual expectation $\mathbb{E}[Y | \text{do}(X=x)]$.
2. Why machine learning models trained on observational data fail when asked counterfactual questions without mechanistic constraints (the causal confounding trap).
3. How to communicate simulation results transparently to non-technical stakeholders without overclaiming certainty.

---

### Stage E: Monte Carlo Uncertainty Quantification

#### 1. Objective and Target Variable
* **Objective:** Quantify the probability distribution of race outcomes by introducing stochastic variance into pace, degradation, and pit-lane execution.
* **Target Output:**
  * Empirical distribution of total race time $P(T_{\text{race}} | S)$.
  * Win probability / Beat probability: $P(T_{\text{race}}(S_1) < T_{\text{race}}(S_2))$.
  * Strategy value-at-risk (95th percentile worst-case race duration).

#### 2. Unit of Observation
A Monte Carlo run consisting of $B = 1,000$ to $5,000$ stochastic race simulations per strategy candidate.

#### 3. Stochastic Variables and Parameter Distributions
1. **Base Driver Pace Noise:** $\epsilon_{\text{pace}} \sim \mathcal{N}(0, \sigma^2_{\text{pace}})$, where $\sigma_{\text{pace}} \approx 0.25\text{ seconds}$ (initial assumption, to be estimated from data).
2. **Tyre Degradation Slope Variance:** $\beta_{\text{deg}} \sim \text{TruncNormal}(\mu_{\text{deg}}, \sigma^2_{\text{deg}}, a=0)$, capturing unexpected tyre graining or thermal degradation.
3. **Pit Stop Stationary Duration:** Modelled as a skewed log-normal or Gamma distribution:
   $$t_{\text{stationary}} \sim \text{Gamma}(k=25, \theta=0.1) + 2.0$$
   (parameters $k=25, \theta=0.1$, shift $2.0\text{s}$: initial assumptions, to be estimated from data), capturing the rare long tail of stuck wheel nuts (5–10 second pit stop disasters; initial assumption, to be estimated from data).
4. **Safety Car Occurrence:** Poisson / Bernoulli process with historical probability $p_{\text{SC}}$ per lap for that specific circuit.

#### 4. Recommended Method and Simpler Alternative
* **Simpler Alternative:** Worst-case / Best-case 3-point deterministic sensitivity analysis ($\pm 20\%$ degradation, $\pm 2\text{ s}$ pit loss; initial assumptions, to be estimated from data).
* **Recommended Method:** Vectorized NumPy Monte Carlo engine running trials efficiently on a standard CPU.

#### 5. Evaluation Metrics
* **Coverage of Historical Variance:** Verify that historical race pace falls within the $90\%$ empirical prediction interval generated by the simulator.
* **Sensitivity Index (Sobol or Morris):** Percentage of total race time variance attributable to degradation uncertainty vs. pit stop execution variance.

#### 6. Main Risks and Limitations
* *False Precision:* Stating that "Strategy A has an $83.4\%$ win probability" when the underlying distribution parameters are subjective estimates.
* *Correlation Neglect:* Lap-to-lap pace noise is not independent white noise ($\text{AR}(1)$ autocorrelation exists: if a tyre grains on lap 20, it remains compromised on lap 21).

#### 7. Leakage-Prevention Rules
* Stochastic parameter distributions must be fitted strictly to past historical races, never calibrated against the variance of the race being simulated.

#### 8. Measurable Definition of Done
Generate side-by-side histogram distributions and cumulative density functions (CDFs) comparing a 1-stop vs. 2-stop strategy for a given driver, computing empirical probability of superiority and 5th/95th percentile confidence intervals.

#### 9. What I Should Be Able to Explain in an Interview
1. Why deterministic strategy optimization often leads to fragile "knife-edge" strategies that collapse if a pit stop is delayed by 1.5 seconds.
2. How to model autocorrelated noise ($\text{AR}(1)$ process) in lap times rather than assuming i.i.d. Gaussian error.
3. How Monte Carlo simulations help a race strategist quantify risk vs. reward (e.g., choosing a safe 2-stop over a high-variance 1-stop).

---

## 5. Feature Engineering Plan

### 5.1 Six Feature Categories

| Category | Specific Features | Computation & Unit | Source Table | Missingness / Fallback |
| :--- | :--- | :--- | :--- | :--- |
| **1. Tyre** | `compound`<br>`tyre_age`<br>`stint_number`<br>`is_new_compound` | One-hot (`SOFT`/`MED`/`HARD`)<br>Laps on current set ($a \ge 1$)<br>Integer stint index ($1, 2, \dots$)<br>Boolean: compound change | `tyre_stints`<br>`race_laps` | Impute initial age = 0 if unrecorded |
| **2. Pace** | `rolling_pace_mean_3`<br>`rolling_pace_std_3`<br>`last_lap_time`<br>`pace_delta_last_2` | Mean seconds over preceding 3 clean laps<br>Standard dev over preceding 3 clean laps<br>$y_{t-1}$ (seconds)<br>$y_{t-1} - y_{t-2}$ (seconds) | `race_laps` | First laps of stint: fallback to driver median clean pace |
| **3. Circuit** | `circuit_id`<br>`track_length_km`<br>`nominal_pit_loss` | Categorical circuit identifier<br>Track physical distance<br>Estimated green pit loss (s; initial assumption) | Config / Metadata | Pre-configured circuit lookup table |
| **4. Race-State** | `lap_number`<br>`race_progress_pct`<br>`fuel_proxy_kg`<br>`position`<br>`is_in_dirty_air` | Integer lap sequence ($1 \dots N$)<br>$(t-1) / N_{\text{total}}$<br>$\text{MaxFuel} \times (1 - \text{progress})$ (TO VERIFY allowance)<br>Position at $t-1$<br>$\text{gap\_ahead}_{t-1} \le 1.5\text{ s}$ (1.5s initial assumption) | `race_laps`<br>`race_context` | Leader gap = 0 for P1;<br>Missing gaps: forward-fill |
| **5. Environmental** | `track_temperature`<br>`air_temperature`<br>`track_temp_trend`<br>`rainfall` | Degrees Celsius<br>Degrees Celsius<br>$T_{\text{track}}(t-1) - T_{\text{track}}(t-5)$<br>Binary 0/1 | `weather` | As-of merge from preceding observation; forward-fill |
| **6. Telemetry (Deferred)** | `speed_trap_max`<br>`throttle_full_pct` | Max km/h on main straight<br>Fraction lap at $100\%$ throttle | FastF1 telemetry | **Deferred from MVP**; added only if Stage A model fails to beat the recorded baseline target |

### 5.2 Causal-Caution Statement
> [!CAUTION]
> **Predictive Association $\ne$ Causal Mechanism:**
> Feature importance scores (e.g., GBDT split gain or SHAP values) reflect empirical statistical associations within the training dataset, not physical or causal laws. For instance, a strong statistical association between high tyre age and slower lap times does not imply that tyre age alone caused the entirety of the observed delta—confounding factors such as traffic management, blue flags, fuel saving, engine detuning, and driver fatigue are correlated with race progress. Causal counterfactual interventions must rely on explicit mechanistic models and validated assumptions, not raw regression coefficients.

### 5.3 Leakage-Safe Rolling Features Implementation Pattern
To guarantee zero lookahead leakage, all historical rolling calculations must adhere to strict temporal shift rules. Below is the reference pandas design pattern:

```python
# Reference Leakage-Safe Rolling Pace Feature Computation
import pandas as pd

def compute_leakage_safe_pace_features(df_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling pace metrics for each driver in a session.
    Guarantees zero data leakage by enforcing an explicit shift(1)
    before calculating rolling windows over eligible clean laps only.
    """
    df = df_laps.copy().sort_values(["session_key", "driver_id", "lap_number"])
    
    # 1. Identify clean green-flag racing laps
    # Green flag rule: must contain no SC ('4'), VSC ('6'), Red ('5'), or Yellow ('2') codes (TO VERIFY exact FastF1 codes)
    is_green = ~df["track_status"].astype(str).str.contains(r"[2456]", regex=True, na=False)
    clean_mask = (
        (~df["is_pit_in_lap"]) & 
        (~df["is_pit_out_lap"]) & 
        is_green & 
        df["lap_time_seconds"].notna()
    )
    
    # 2. Extract eligible clean laps only so rolling windows span the last 3 actual clean laps, not NaN-padded calendar laps
    clean_laps = df[clean_mask].copy()
    
    # 3. Compute rolling statistics on shifted clean laps (shift(1) ensures lap t cannot see lap t)
    clean_grouped = clean_laps.groupby(["session_key", "driver_id"])["lap_time_seconds"]
    clean_laps["rolling_pace_mean_3"] = (
        clean_grouped.shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )
    clean_laps["rolling_pace_std_3"] = (
        clean_grouped.shift(1)
        .rolling(window=3, min_periods=2)
        .std()
        .fillna(0.0)
    )
    clean_laps["prev_lap_time"] = clean_grouped.shift(1)
    
    # Filtering to eligible clean laps before rolling prevents NaN-gap distortions; forward-filling maps the last known clean pace across subsequent laps.
    feature_cols = ["rolling_pace_mean_3", "rolling_pace_std_3", "prev_lap_time"]
    for col in feature_cols:
        df[col] = clean_laps[col]
        df[col] = df.groupby(["session_key", "driver_id"])[col].ffill()
        
    return df
```

---

## 6. Evaluation and Validation Plan

### 6.1 Lap-Time Regression Evaluation Protocol
* **Primary Metric:** Mean Absolute Error (MAE):
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
  *Interpretation:* Average error per lap in seconds. Directly relatable to race engineer terminology (e.g., "model is off by two tenths per lap").
* **Secondary Metric:** Root Mean Squared Error (RMSE):
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
  *Interpretation:* Exposes whether the model produces dangerous large errors on specific laps.
* **Error Segmentation:** All error metrics must be decomposed across:
  * Compound sub-cohorts (`SOFT`, `MEDIUM`, `HARD`).
  * Stint quartile segments (Laps 1–5, 6–15, 16–25, 26+).
  * High track temperature vs. moderate track temperature bins.
* **Worst-Error Inspection:** Top prediction residuals must be automatically extracted and categorized into: (a) unflagged driver lockup/mistake, (b) traffic/blue flag impediment, (c) unannounced car damage, or (d) model deficiency.

### 6.2 Strategy Simulation Evaluation Protocol
* **Delta to Observed Reality:** For a driver who executed a clean, uncompromised race, compare simulated total race time against actual race time:
  $$E_{\text{race}} = |T_{\text{sim}}(\text{Observed Strategy}) - T_{\text{actual}}|$$
* **Strategy Regret Metric:** In cases where a team made a widely acknowledged strategic blunder, verify whether the simulator ranks the blunder significantly below the alternative strategy that won on track.
* **Sensitivity Stability:** Assess whether small perturbations in inputs cause chaotic switching in the recommended pit lap (knife-edge instability). A robust strategy must display smooth preference degradation.

### 6.3 Strict Validation and Split Rules
1. **Within-Circuit vs. Cross-Circuit Split Designs:**
   * **Within-Circuit Evaluation (Secondary / Upper-Bound Sanity Check):** Training on one year's race at a circuit and testing on the same circuit in a later year (e.g., Train: 2022 Spanish GP; Test: 2023 Spanish GP). This evaluates temporal transfer under identical track geometry, but it can overstate generalization because circuit-specific factors (asphalt abrasiveness, corner energy profiles, layout) remain constant.
   * **Cross-Circuit Evaluation (Primary Benchmark):** Training on a set of circuits and evaluating on an entirely unseen circuit (e.g., Train: 2023 Bahrain GP + 2023 Austrian GP; Test: 2023 Spanish GP). This serves as the primary evaluation because it provides the stricter test of true predictive generalization across diverse circuit topologies, surfaces, and degradation regimes.
2. **Race Candidate Selection:**
   * Specific validation races must not be declared final until Phase 1 data coverage and weather checks are completed; present all proposed races strictly as candidates.
   * *Training / Tuning Candidates (TO VERIFY):* Candidate dry races such as 2023 Bahrain GP or 2023 Austrian GP. (Note: 2023 Monaco GP is excluded because rain occurred during the race, which directly conflicts with the dry-only selection rule; weather across all candidate sessions remains TO VERIFY).
   * *Test Candidate (TO VERIFY):* 2023 Spanish GP (dry status, low incident rate, and data completeness remain TO VERIFY in Phase 1).
3. **Preprocessing Isolation:** All scalers, imputers, target encodings, and fuel correction parameters must be fitted strictly on training races and applied out-of-sample to validation/test races.
4. **Small Sample Reality Limitation:** Given that the initial MVP evaluates 2–4 races, leave-one-race-out cross-validation across the small set is mandatory. Results must explicitly acknowledge the wide confidence interval around test-set performance metrics.

### 6.4 Counterfactual Validity Rule
> [!IMPORTANT]
> **Counterfactual Validity Constraint:**
> A simulation comparing counterfactual lap times does not prove an alternative strategy would have succeeded in reality. Real race outcomes are conditioned by interactive multi-agent dynamics: overtaking difficulty, dirty air wake turbulence, tyre graining when following closely, reactive pit strategies by competitors, and unpredictable Safety Car neutralizations. Any counterfactual recommendation generated by ApexPitwall must be explicitly documented as **conditional on stated assumptions (clean air, constant degradation, zero competitor reaction)**.

---

## 7. Suggested Repository Structure

```
apexpitwall/
├── README.md
├── PROJECT_OVERVIEW.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── base_config.yaml
│   └── circuits/
│       ├── catalunya.yaml
│       └── sakhir.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_eda_tyre_degradation.ipynb
│   └── 03_model_prototyping.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fastf1_loader.py
│   │   └── cleaner.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── fuel_model.py
│   │   ├── pace_features.py
│   │   └── weather_aligner.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── regressor.py
│   │   └── degradation.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── error_analysis.py
│   └── simulation/
│       ├── __init__.py
│       ├── simulator.py
│       ├── counterfactual.py
│       └── monte_carlo.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_feature_leakage.py
│   └── test_simulator.py
├── reports/
│   ├── figures/
│   └── evaluation/
└── app/
    ├── streamlit_app.py
    └── components/
        ├── telemetry_view.py
        └── strategy_view.py
```

---

## 8. Phased Roadmap with Time Estimates

*Workload Baseline: 8–10 hours per week on a standard personal Windows laptop (CPU-only).*

```
Week 1-2: Phase 1 (Inspection & Setup)  [16-20 hrs]
  └── Week 3-4: Phase 2 (EDA & Degradation) [16-20 hrs]
        └── Week 5-7: Phase 3 (Baseline & ML Models) [24-30 hrs]
              └── Week 8-10: Phase 4 (Strategy Simulator) [24-30 hrs]
                    └── Week 11-12: Phase 5 (Dashboard & Release) [16-20 hrs]
```

### Phase 1: Environment Setup, FastF1 Caching & Data Inspection
* **Estimated Time:** 16–20 hours (Weeks 1–2).
* **Learning Outcomes:** Understanding FastF1 data models, session caching mechanisms, F1 timing feeds, and data hygiene on Windows.
* **Deliverables:**
  * Configured local Python virtual environment (`.venv`) on Windows.
  * `fastf1` disk caching verified and operational.
  * Notebook `01_data_inspection.ipynb` inspecting the pilot candidate 2023 Spanish Grand Prix (TO VERIFY).
  * Initial Data Coverage & Missingness Report (`reports/data_inspection_report.md`).
* **Dependencies:** Python 3.10+, FastF1, pandas, Jupyter.
* **Acceptance Criteria:** FastF1 successfully downloads and caches pilot candidate 2023 Spanish GP race session in under a minute from cache (to be measured); missing-value rates for lap times, tyre compounds, and pit stops are quantitatively calculated and documented.
* **Potential Blockers:** Windows file path length limitations with caching; intermittent F1 live timing connection drops.

### Phase 2: Exploratory Data Analysis & Tyre Degradation Modeling
* **Estimated Time:** 16–20 hours (Weeks 3–4).
* **Learning Outcomes:** Separating fuel burn from tyre degradation; detecting thermal degradation; stint visualization techniques.
* **Deliverables:**
  * Cleaned tabular datasets saved to `data/processed/`.
  * Visualizations of compound degradation curves (`reports/figures/degradation_curves.png`).
  * Mechanistic fuel-correction module (`src/features/fuel_model.py`).
* **Dependencies:** Clean lap records from Phase 1.
* **Acceptance Criteria:** Fuel-corrected tyre degradation curves produced for Soft, Medium, and Hard compounds; report fitted degradation slopes per compound with uncertainty intervals and residual error, whatever their ordering, and discuss any result that contradicts expectations.
* **Potential Blockers:** Unflagged traffic laps distorting tyre wear curves; difficulty parsing scrubbed vs. new tyres.

### Phase 3: Leakage-Safe Baseline & ML Pace Modeling (Stage A)
* **Estimated Time:** 24–30 hours (Weeks 5–7).
* **Learning Outcomes:** Feature engineering on time-series tabular data; preventing lookahead bias; LightGBM/XGBoost CPU optimization; leave-race-out validation.
* **Deliverables:**
  * Rolling pace feature generator with strict unit test verification (`tests/test_feature_leakage.py`).
  * Rolling median baseline model (`src/models/baseline.py`).
  * LightGBM pace regressor (`src/models/regressor.py`).
  * Model evaluation report with compound/stint error segmentation.
* **Dependencies:** Processed features from Phase 2.
* **Acceptance Criteria:** Unit tests prove zero lookahead leakage. Measure baseline MAE and RMSE first on held-out races, then set a target relative to that baseline and record it before tuning.
* **Potential Blockers:** High residual errors caused by undetected dirty air or yellow flag micro-sectors.

### Phase 4: Strategy Simulation & Counterfactual Analysis (Stages C, D, E)
* **Estimated Time:** 24–30 hours (Weeks 8–10).
* **Learning Outcomes:** Discrete event simulation; counterfactual reasoning in sports analytics; vectorized Monte Carlo simulations.
* **Deliverables:**
  * Deterministic strategy simulator (`src/simulation/simulator.py`).
  * Counterfactual what-if perturbation engine (`src/simulation/counterfactual.py`).
  * Vectorized Monte Carlo uncertainty simulator (`src/simulation/monte_carlo.py`).
  * Comparative strategy evaluation notebook (`notebooks/03_strategy_evaluation.ipynb`).
* **Dependencies:** Stage A/B pace and degradation parameters.
* **Acceptance Criteria:** Simulator evaluates candidate strategies in under a few seconds; Monte Carlo engine runs trials efficiently on CPU; counterfactual engine models $\Delta T$ of shifting pit laps.
* **Potential Blockers:** Edge-case strategies producing negative stint lengths; numerical instability in Monte Carlo distributions.

### Phase 5: Streamlit Application, Documentation & Portfolio Packaging
* **Estimated Time:** 16–20 hours (Weeks 11–12).
* **Learning Outcomes:** Interactive dashboard engineering; model explainability presentation (SHAP); portfolio-grade open-source documentation.
* **Deliverables:**
  * Interactive Streamlit dashboard (`app/streamlit_app.py`).
  * SHAP explainability visualizations integrated into dashboard.
  * Comprehensive `README.md` with architecture diagrams and reproducible execution instructions.
  * Final test suite passing with `pytest`.
* **Dependencies:** Completed simulation engine and models.
* **Acceptance Criteria:** Changing a slider recomputes the displayed comparison in under a few seconds on the target laptop (to be measured); user can select a driver, adjust pit stop lap sliders, and view immediate counterfactual comparisons with uncertainty bands; complete repository is reproducible via `pip install -r requirements.txt`.
* **Potential Blockers:** Streamlit memory leaks when re-running large Monte Carlo simulations; SHAP computation bottlenecks on CPU (mitigated via TreeExplainer on small sample subsets).

---

## 9. Risks, Limitations, and Open Questions

### 9.1 Twelve Specific Project Risks and Mitigations

| Risk | Description & Impact | Practical Mitigation Strategy |
| :--- | :--- | :--- |
| **1. Inconsistent Data Coverage** | Certain historical seasons lack exact pit stationary durations or tyre scrub status. | Restrict initial scope to 2022–2023 where FastF1 schemas are mature. Tag all missing attributes with explicit fallbacks. |
| **2. Missing/Unreliable Stints** | Incorrectly assigned tyre compound or missing stint boundaries in practice/qualifying data. | Derive stint boundaries strictly from race session `is_pit_out_lap` flags; cross-check total stint laps against official FIA classification sheets. |
| **3. Inaccurate Pit Loss Estimates** | Pit lane transit times vary wildly depending on whether a pit stop occurred under Green, VSC, or Full Safety Car. | Condition pit loss calculations on `track_status`. Maintain two distinct pit loss parameters: $L_{\text{green}} \approx 20.5\text{s}$ and $L_{\text{VSC}} \approx 11.0\text{s}$ (initial assumptions, to be estimated from data; remove any claim of calibration until measured). |
| **4. Fuel-Tyre Confounding** | Car sheds fuel mass, gaining pace, which directly masks tyre degradation pace loss. | Use a two-stage modeling approach: mechanically subtract estimated fuel mass lap time benefit (maximum race fuel allowance and fuel effect per lap: initial assumptions to be estimated or verified from regulations) prior to fitting empirical tyre wear curves. |
| **5. Traffic & Race Control Noise** | Laps spent behind a slower car or under yellow flags produce artificial pace spikes that corrupt degradation curves. | Filter out laps where `track_status` contains SC, VSC, Red, or Yellow codes (TO VERIFY exact FastF1 codes) or where gap to ahead is below threshold (e.g., $\le 1.2\text{s}$; initial assumption, to be estimated from data) from baseline degradation curve fitting. |
| **6. Small Compound Sample Size** | Hard compounds might only be run for a small number of laps by few drivers in a single race, yielding insufficient data for regression. | Pool degradation slope priors across identical tyre compound allocations (e.g., C1, C2, C3) across multiple comparable race weekends. |
| **7. Regulatory & Temporal Shifts** | Car weights, aerodynamic regulations, and tyre constructions changed radically between 2021 and 2022. | Strictly forbid training on pre-2022 data when predicting 2022+ races. Treat regulatory eras as separate statistical regimes. |
| **8. Future Lookahead Leakage** | Accidentally using full-race driver pace averages or end-of-race pit strategies inside lap $t$ feature vectors. | Enforce automated unit tests (`tests/test_feature_leakage.py`) asserting that feature values at lap $t$ are identical when computed on partial vs. full dataframes. |
| **9. Driver / Team Overfitting** | Model learns that "Red Bull = fast" and fails when evaluating a midfield car under an aggressive strategic variant. | Standardize target lap times relative to driver/team baseline pace, or include driver random effects / fixed pace offsets. |
| **10. Counterfactual Realism Gap** | Simulator assumes a counterfactual pit stop re-enters into clean air, ignoring the reality of rejoining into a 4-car DRS train. | Explicitly display an "Estimated Re-entry Traffic Risk Index" alongside any counterfactual recommendation. |
| **11. Monte Carlo False Precision** | Providing overly narrow uncertainty bounds because lap noise is assumed to be independent Gaussian white noise. | Implement an Autoregressive $\text{AR}(1)$ noise model so that a simulated tyre graining event persists across multiple successive laps. |
| **12. Scope Creep & Complexity** | Attempting to build an interactive multi-car 3D simulator, exhausting the 8–10 hour weekly budget before achieving a working baseline. | Enforce strict stage gates: no simulator work until Stage A regression satisfies the target set relative to the baseline; no Streamlit dashboard until simulator passes unit tests. |

### 9.2 Open Questions to Be Resolved in Phase 1 Data Inspection
The following questions cannot be answered by assumption and must be verified directly against the FastF1 cache in Session 1:
1. *Does FastF1 provide reliable stationary pit stop duration (wheel-stop time) for the pilot candidate race, or only total pit-in to pit-out lap deltas?* (**TO VERIFY**)
2. *Are tyre sets flagged with exact previous mileage from qualifying/practice, or is `tyre_age` reset to 0 at the start of the race?* (**TO VERIFY**)
3. *What is the exact latency and sampling frequency of track temperature updates in the FastF1 weather stream?* (**TO VERIFY**)
4. *Can Safety Car and Virtual Safety Car deployment periods be accurately resolved to specific lap fractions without deep telemetry inspection?* (**TO VERIFY**)

---

## 10. Explainability and Application Vision

### 10.1 Explainability Architecture: Tree SHAP Plan
To transition the machine learning model from a black-box predictor into a trusted race engineering tool, ApexPitwall will integrate **SHapley Additive exPlanations (SHAP)**:
* **Method:** `shap.TreeExplainer` applied to the trained LightGBM pace regressor. TreeExplainer computes exact Shapley values in polynomial time, running efficiently on a standard CPU.
* **Global Explainability (Summary Plot):**
  * Visualize feature importance rankings across the entire validation dataset.
  * Verify domain consistency: `fuel_proxy` should exhibit a negative SHAP value (faster lap times), while `tyre_age` should exhibit an increasing positive SHAP value (slower lap times).
* **Local Explainability (Force Plot / Waterfall Plot per Lap):**
  * Break down any individual predicted lap time $\hat{y}_t$ into its constituent additive factors:
    $$\hat{y}_t = \text{BaseValue} + \phi_{\text{compound}} + \phi_{\text{tyre\_age}} + \phi_{\text{rolling\_pace}} + \phi_{\text{track\_temp}} + \phi_{\text{traffic}}$$
* **Attribution vs. Causation Warning:** The UI will explicitly display a disclaimer stating that SHAP values reflect model attribution, not physical intervention effects.

### 10.2 Application Vision: Streamlit Strategy Dashboard
The user-facing deliverable is a local Streamlit application (`app/streamlit_app.py`) structured into three analytical tabs:

```
┌────────────────────────────────────────────────────────────────────────┐
│  ApexPitwall Strategy Console | 2023 Spanish Grand Prix (Pilot Candidate)│
├────────────────────────────────────────────────────────────────────────┤
│ Selected Driver: [ Fernando Alonso (AST) ▼ ]    Target Lap: [ 25 / 66 ]│
├────────────────────────────────────────────────────────────────────────┤
│ [Tab 1: Stint & Telemetry]  [Tab 2: Strategy Simulator]  [Tab 3: SHAP] │
│                                                                        │
│  Candidate Strategy Leaderboard (Hypothetical, for illustration):       │
│  Rank  Strategy      Stints            Simulated Race Time    Delta    │
│  1     2-Stop (Opt)  S (18) -> M (24) -> H (24)   1h 28m 14.2s   -     │
│  2     1-Stop        M (30) -> H (36)             1h 28m 22.8s  +8.6s  │
│  3     2-Stop        S (14) -> H (26) -> S (26)   1h 28m 26.1s  +11.9s │
│                                                                        │
│  Interactive Counterfactual Playground (Hypothetical, for illustration):│
│  Pit Lap Slider: [---●-------] Lap 22 (Observed: Lap 19)               │
│  Next Compound:  ( ) Soft  (●) Medium  ( ) Hard                        │
│  Simulated Counterfactual Delta: +2.4s slower than observed            │
│                                                                        │
│  Monte Carlo Outcome Distribution (Hypothetical, for illustration):     │
│  [|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||] │
│  P(2-Stop beats 1-Stop): 78.4% | 90% Confidence Interval: [±4.2s]      │
│  *(All times, ranks, values, deltas, and probabilities in mock-up are  │
│    hypothetical, for illustration)*                                    │
└────────────────────────────────────────────────────────────────────────┘
```

* **Tab 1: Race Overview & Stint Telemetry:**
  * Interactive lap time scatter plot with compound color-coding (Red = Soft, Yellow = Medium, White = Hard).
  * Fuel-corrected tyre degradation curves with historical data points overlaid against fitted slopes.
* **Tab 2: Interactive Strategy Simulator & Counterfactual Playground:**
  * Sliders allowing the user to perturb pit stop laps and alter compound choices.
  * Real-time re-calculation of remaining race time using the Stage C simulator.
  * Side-by-side Monte Carlo distribution charts showing outcome variance and win probability.
  * Strict visual separation between **Historical Observed Data** (solid lines), **Model Pace Forecasts** (dashed lines), and **Simulated Counterfactuals** (dotted lines).
* **Tab 3: Model Diagnostics & SHAP Explanations:**
  * Live waterfall plot explaining the feature attribution for the selected driver's current lap prediction.
  * Table of top prediction residuals for model transparency.

---

## 11. Concrete Phase 1 Checklist for First Working Session

*Objective: Establish project foundation, configure local FastF1 disk caching, load the pilot candidate session (2023 Spanish Grand Prix, TO VERIFY), inspect data fields, calculate missingness rates, and generate a reproducible data coverage report. No machine learning models will be trained during this session.*

### Step-by-Step Execution Plan

- [ ] **Step 1: Project Directory Initialization**
  * Open PowerShell on Windows in the project root: `d:\ApexPitwall`.
  * Verify Python installation: `python --version` (Ensure Python $\ge 3.10$).

- [ ] **Step 2: Python Virtual Environment Setup**
  * Create isolated virtual environment: `python -m venv .venv`.
  * Activate environment: `.venv\Scripts\Activate.ps1`.
  * Upgrade package manager: `python -m pip install --upgrade pip setuptools wheel`.

- [ ] **Step 3: Minimal Dependency Installation**
  * Install only the packages required for data inspection:
    ```bash
    pip install fastf1 pandas numpy matplotlib jupyter
    ```
  * Freeze initial environment state: `pip freeze > requirements.txt`.

- [ ] **Step 4: Create Directory Skeleton**
  * Create raw storage, cache, and notebook directories:
    ```powershell
    New-Item -ItemType Directory -Force -Path data/cache, data/raw, data/processed, notebooks, reports/figures
    ```

- [ ] **Step 5: Verify FastF1 Caching Configuration**
  * Create a minimal test script or notebook cell verifying that FastF1 correctly writes cache objects to `data/cache`.

- [ ] **Step 6: Load Pilot Session (2023 Spanish Grand Prix - Race, Pilot Candidate TO VERIFY)**
  * Execute session load using FastF1:
    ```python
    import fastf1
    fastf1.Cache.enable_cache('data/cache')
    session = fastf1.get_session(2023, 'Spanish Grand Prix', 'R')
    session.load(telemetry=False, weather=True)
    ```

- [ ] **Step 7: Inspect Available Data Tables & Schema**
  * Inspect `session.laps` columns: check `LapTime`, `Sector1Time`, `Sector2Time`, `Sector3Time`, `Compound`, `Stint`, `TyreLife`, `PitInTime`, `PitOutTime`, `TrackStatus`.
  * Inspect `session.weather_data`: check recording timestamps, `TrackTemp`, `AirTemp`, `Rainfall`.
  * Inspect driver availability: confirm all classified and unclassified drivers have records.

- [ ] **Step 8: Missing-Value Rate Audit**
  * Compute missingness percentages across essential modeling fields:
    * `LapTime`: Missingness on completed laps vs. DNF laps.
    * `Compound`: Verify presence and completeness on dry racing laps.
    * `TyreLife`: Verify consistency of lap increments per stint.
    * `SectorXTime`: Check for missing individual sector timestamps.
    * `PitInTime` / `PitOutTime`: **Note on pit timestamps:** `PitInTime` and `PitOutTime` are expected to be NaN on most laps by design (cars only pit 1–3 times in a ~60 lap race). Report their missingness only as "laps with pit timestamps vs. pit stops counted from stint changes," not as a data-quality failure.

- [ ] **Step 9: Pit Stop & Stint Boundary Inspection**
  * Extract driver stint summary: check if `Stint` index changes reliably align with `PitInTime` / `PitOutTime`.
  * Calculate total pit lane duration for all completed pit stops.
  * Determine if stationary duration is directly parseable or requires calculation. (**TO VERIFY**)

- [ ] **Step 10: Generate Data Coverage Report**
  * Compile findings into `reports/data_inspection_report.md` detailing:
    * Session inspected: `2023 Spanish Grand Prix (Race) (Pilot Candidate)`.
    * Total laps recorded: total lap entries.
    * Missingness table for essential fields.
    * Verified schema quirks (e.g., timedelta format of FastF1 timing fields).

- [ ] **Step 11: Phase 1 Gate Review & Acceptance Decision**
  * Evaluate against Phase 1 Acceptance Criteria:
    1. Did FastF1 successfully cache the session locally?
    2. Are clean lap times, compounds, and stint numbers present without severe corruption?
    3. Are weather timestamps alignable via backward as-of merge?
  * *Go / No-Go Decision:* If Spanish GP data is corrupted or fails dry/completeness checks, substitute with candidate **2023 Bahrain Grand Prix (Race)** and re-run checklist.

### Minimal Phase 1 Inspection Snippet

```python
"""
Phase 1: Minimal FastF1 Data Inspection Script
Path: notebooks/inspect_session.py
"""
import os
import fastf1
import pandas as pd

# 1. Enable disk cache to avoid redundant network requests
cache_dir = os.path.join("data", "cache")
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# 2. Load 2023 Spanish Grand Prix Race (Pilot Candidate, TO VERIFY)
print("Fetching 2023 Spanish GP Race Session (Pilot Candidate)...")
session = fastf1.get_session(2023, "Spanish Grand Prix", "R")
session.load(telemetry=False, weather=True)

# 3. Extract laps dataframe
laps = session.laps
print(f"Total lap records loaded: {len(laps)}")

# 4. Audit essential modeling fields
fields_to_check = [
    "Driver", "LapNumber", "LapTime", "Sector1Time", 
    "Sector2Time", "Sector3Time", "Compound", "Stint", 
    "TyreLife", "TrackStatus"
]

print("\n--- Missing Value Audit ---")
missing_report = []
for field in fields_to_check:
    if field in laps.columns:
        n_missing = laps[field].isna().sum()
        pct_missing = (n_missing / len(laps)) * 100
        missing_report.append({"Field": field, "Missing Count": n_missing, "Missing %": pct_missing})
    else:
        missing_report.append({"Field": field, "Missing Count": "MISSING COLUMN", "Missing %": 100.0})

df_missing = pd.DataFrame(missing_report)
print(df_missing.to_string(index=False))

# 5. Audit Pit Timestamps (Expected NaN on non-pit laps by design)
pit_in_count = laps["PitInTime"].notna().sum() if "PitInTime" in laps.columns else 0
pit_out_count = laps["PitOutTime"].notna().sum() if "PitOutTime" in laps.columns else 0
print(f"\nPit In Timestamps: {pit_in_count} | Pit Out Timestamps: {pit_out_count}")
print("(Note: PitInTime/PitOutTime are expected to be NaN on non-pit laps by design; evaluated vs stint count).")

# 6. Check distinct tyre compounds
print("\nObserved Compounds in Race:")
print(laps["Compound"].value_counts(dropna=False))

# 7. Check weather stream
weather = session.weather_data
print(f"\nWeather records loaded: {len(weather)}")
print(f"Track Temp range: {weather['TrackTemp'].min():.1f}C to {weather['TrackTemp'].max():.1f}C")
print("\nPhase 1 Inspection complete. Data ready for coverage report.")
```

---

## Single Next Action

After reviewing this blueprint, your single next action is:
**Open PowerShell in `d:\ApexPitwall` and execute Step 1 through Step 3 of the Phase 1 Checklist to initialize your virtual environment and install `fastf1`, `pandas`, `matplotlib`, and `jupyter`.**
