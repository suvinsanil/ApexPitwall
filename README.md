# ApexPitwall 🏎️

**ML-Powered Formula 1 Pit-Stop Strategy Prediction & Simulation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

ApexPitwall is an educational, portfolio-grade, local-first Formula 1 race strategy engine. It combines empirical telemetry data extraction (via [FastF1](https://docs.fastf1.dev/)), leakage-safe machine learning pace regression, physics-based fuel mass disentanglement, deterministic pit strategy optimization, counterfactual 'what-if' analysis, and Monte Carlo uncertainty quantification.

---

## 🏗️ System Architecture

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

## 📊 Evaluation & Empirical Benchmarks

All models were trained across 4 diverse 2022 Grands Prix (Bahrain, Zandvoort, Mexico City, Abu Dhabi) and evaluated strictly out-of-sample on the held-out **2022 Spanish Grand Prix (Catalunya)** under high ambient track temperatures (47.6–49.7°C):

| Model Architecture | Calibration Strategy | Held-Out MAE (s) | Held-Out RMSE (s) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Naive Baseline** | 3-Lap Shifted Rolling Mean ($t-1$) | **0.5906 s** | **0.8509 s** | 0.6475 |
| **Ridge Regression** | Cross-Circuit Held-Out | **1.0189 s** | **1.2070 s** | 0.2907 |
| **LightGBM Regressor** | Cross-Circuit Held-Out | **3.0252 s** | **3.1822 s** | -3.9305 |
| **Ridge Regression** | Within-Circuit 5-Fold Cross Validation | **0.5520 s** | **0.7512 s** | 0.7265 |
| **LightGBM Regressor** | Within-Circuit 5-Fold Cross Validation | **0.4633 s** | **0.6710 s** | **0.7779** |

### Key Engineering Insights
1. **Zero Lookahead Leakage:** Feature engineering strictly calculates rolling pace stats with `.shift(1)` on prior clean laps.
2. **Fuel-Tyre Confounder Disentanglement:** Without controlling for the 110 kg fuel burn-off (~0.035 s/lap pace gain), tyres appear to gain speed with age. Controlling for fuel load reveals true positive wear rates (+0.04 to +0.06 s/lap).
3. **Empirical Pit Loss:** Spanish GP timing loops measure a median green-flag pit loss of **22.59 seconds** (transit: 22.39s; inferred stationary service: ~2.8s).
4. **Strategy Optimization Verdict:** On high-abrasion circuits like Catalunya, a **2-stop Soft -> Medium -> Soft strategy (stopping on Laps 18, 40)** beats 1-stop alternatives by **32.8 to 44.3 seconds**.

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/suvinsanil/ApexPitwall.git
cd ApexPitwall

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipelines
```bash
# 1. Run Data Inspection and Season Audit
python notebooks/inspect_session.py

# 2. Extract and Preprocess Leakage-Safe Race Data
python src/data_pipeline.py

# 3. Train Machine Learning Models
python src/train_model.py

# 4. Measure Empirical Pit Loss Parameters
python src/estimate_pit_loss.py

# 5. Run Race Strategy Optimization Sweep
python src/strategy_simulator.py

# 6. Run Full Pytest Suite
pytest tests
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 📂 Repository Layout

```
ApexPitwall/
├── README.md                           # Project documentation & benchmarks
├── PROJECT_OVERVIEW.md                 # Complete 11-section architectural blueprint
├── requirements.txt                    # Pinned dependencies
├── .gitignore
├── configs/
│   ├── base_config.yaml                # Global parameters & assumed constants
│   └── circuits/                       # Circuit specifications (Catalunya, Sakhir, etc.)
├── data/
│   ├── processed/
│   │   ├── race_laps_processed.csv    # 5,333 clean laps across 5 circuits
│   │   ├── pit_loss_parameters.json   # Empirically measured pit loss parameters
│   │   └── strategy_results.csv       # Complete 118-variant strategy sweep
│   └── cache/                          # FastF1 disk cache (git-ignored)
├── models/
│   ├── lgbm_catalunya.joblib           # Trained LightGBM Catalunya model
│   └── ridge_catalunya.joblib          # Trained Ridge regression baseline
├── src/
│   ├── config.py                       # Config helper
│   ├── data_pipeline.py                # Lap cleaning & weather alignment
│   ├── train_model.py                  # Model training pipeline
│   ├── evaluate_model.py               # Out-of-sample evaluation engine
│   ├── estimate_pit_loss.py            # Pit lane transit & time loss estimation
│   ├── strategy_simulator.py           # Multi-stint race simulator
│   ├── data/                           # Modular loaders & cleaners
│   ├── features/                       # Leakage-safe feature extractors
│   └── simulation/
│       ├── counterfactual.py           # Stage D: Perturbation engine
│       └── monte_carlo.py              # Stage E: Stochastic AR(1) uncertainty engine
├── tests/
│   ├── test_feature_leakage.py         # Zero-leakage unit tests
│   ├── test_data_loader.py             # Data hygiene tests
│   ├── test_simulator.py               # Deterministic simulation tests
│   └── test_simulation_engines.py     # Counterfactual & Monte Carlo tests
├── reports/
│   ├── data_inspection_report.md       # Phase 1 audit report
│   ├── model_evaluation.md             # Phase 2 ML performance report
│   ├── pit_loss_estimation.md          # Empirical pit timing audit
│   ├── strategy_simulation.md          # Phase 3 strategy optimization report
│   └── figures/                        # Generated charts & diagnostic plots
└── app/
    ├── streamlit_app.py                # Main multi-tab Streamlit dashboard
    └── components/
        ├── telemetry_view.py           # Driver pace & degradation view
        └── strategy_view.py            # Interactive simulator & Monte Carlo view
```

---

## ⚖️ License
MIT License. Built for educational and research purposes using Formula 1 timing feeds via FastF1.
